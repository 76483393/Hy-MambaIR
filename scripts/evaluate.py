#!/usr/bin/env python3
"""Clean single-model evaluation entry for the staged Hy-MambaIR repository."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

from register_models import load_compatible_network, register_all  # noqa: E402

register_all()

import torch  # noqa: E402
from basicsr.data import build_dataloader, build_dataset  # noqa: E402
from basicsr.models import build_model  # noqa: E402


def _deep_update(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_with_inherits(opt_path: Path) -> dict:
    with opt_path.open("r", encoding="utf-8") as f:
        current = yaml.safe_load(f) or {}

    inherit_key = current.pop("inherits", None)
    if not inherit_key:
        return current

    parent_path = (opt_path.parent / inherit_key).resolve()
    if not parent_path.exists():
        raise FileNotFoundError(f"Inherited config not found: {parent_path}")
    parent = _load_yaml_with_inherits(parent_path)
    return _deep_update(parent, current)


def load_opt(opt_path: Path) -> dict:
    opt = _load_yaml_with_inherits(opt_path)
    opt["is_train"] = False
    opt["dist"] = False
    opt["rank"] = 0
    opt["world_size"] = 1
    opt["num_gpu"] = 1
    opt.setdefault("manual_seed", 42)
    return opt


def build_val_loader_from_opt(opt: dict):
    val_opt = opt["datasets"]["val"]
    val_opt.setdefault("phase", "val")
    if "scale" not in val_opt and "scale" in opt:
        val_opt["scale"] = opt["scale"]
    val_set = build_dataset(val_opt)
    return build_dataloader(
        val_set,
        val_opt,
        num_gpu=opt.get("num_gpu", 1),
        dist=False,
        sampler=None,
        seed=opt.get("manual_seed", 42),
    )


def load_generator_weights(model, weight_path: Path, strict: bool = False) -> None:
    load_compatible_network(
        model,
        model.net_g,
        str(weight_path),
        strict=strict,
        param_key="params_ema",
    )


def evaluate(config_path: Path, weight_path: Path, output_json: Path) -> dict:
    opt = load_opt(config_path)
    model = build_model(opt)
    if hasattr(model, "net_g"):
        model.net_g.eval()

    strict_flag = bool(opt.get("path", {}).get("strict_load_g", False))
    load_generator_weights(model, weight_path, strict=strict_flag)
    val_loader = build_val_loader_from_opt(opt)

    with torch.no_grad():
        model.validation(val_loader, current_iter=0, tb_logger=None, save_img=False)

    metrics = getattr(model, "metric_results", {})
    result = {
        "config_path": str(config_path),
        "weight_path": str(weight_path),
        "dataset_name": val_loader.dataset.opt.get("name", "val"),
        "num_images": len(val_loader.dataset),
        "metrics": {
            "psnr": float(metrics.get("psnr", 0.0)),
            "ssim": float(metrics.get("ssim", 0.0)),
            "lpips": float(metrics.get("lpips", 0.0)) if "lpips" in metrics else None,
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a single Hy-MambaIR checkpoint with a paper config."
    )
    parser.add_argument(
        "--config", required=True, help="Path to evaluation config YAML."
    )
    parser.add_argument("--weights", required=True, help="Path to generator weights.")
    parser.add_argument(
        "--output",
        default="results/eval_result.json",
        help="Where to save metrics JSON.",
    )
    return parser.parse_args()


def main() -> None:
    os.chdir(ROOT)
    args = parse_args()
    config_path = (
        (ROOT / args.config).resolve()
        if not os.path.isabs(args.config)
        else Path(args.config)
    )
    weight_path = (
        (ROOT / args.weights).resolve()
        if not os.path.isabs(args.weights)
        else Path(args.weights)
    )
    output_json = (
        (ROOT / args.output).resolve()
        if not os.path.isabs(args.output)
        else Path(args.output)
    )

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not weight_path.exists():
        raise FileNotFoundError(f"Weight file not found: {weight_path}")

    result = evaluate(config_path, weight_path, output_json)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
