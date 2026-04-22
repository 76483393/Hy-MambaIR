#!/usr/bin/env python3
"""Register Hy-MambaIR extensions against an installed BasicSR package."""

from __future__ import annotations

import importlib.util
import os
from collections import OrderedDict
from collections.abc import Mapping

import torch
from basicsr.utils import get_root_logger
from basicsr.utils.registry import ARCH_REGISTRY, METRIC_REGISTRY, MODEL_REGISTRY


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CORE_DIR = os.path.join(ROOT, "core")


def _load_module(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rename_legacy_key(key: str, target_keys: set[str]) -> str:
    candidates: list[str] = []
    if ".assm." in key:
        candidates.append(key.replace(".assm.", ".hssm."))
    if key.startswith("assm."):
        candidates.append(f"hssm.{key[len('assm.') :]}")

    for candidate in candidates:
        if candidate in target_keys:
            return candidate
    return key


def _remap_legacy_checkpoint_keys(
    load_net: Mapping[str, torch.Tensor], target_keys: set[str]
) -> tuple[OrderedDict[str, torch.Tensor], int]:
    remapped: OrderedDict[str, torch.Tensor] = OrderedDict()
    source_keys: dict[str, str] = {}
    renamed_count = 0

    for raw_key, value in load_net.items():
        key = raw_key[7:] if raw_key.startswith("module.") else raw_key
        mapped_key = key if key in target_keys else _rename_legacy_key(key, target_keys)
        if mapped_key != key:
            renamed_count += 1
        if mapped_key in remapped and source_keys[mapped_key] != raw_key:
            raise KeyError(
                "Checkpoint key collision after remapping: "
                f"{source_keys[mapped_key]} and {raw_key} -> {mapped_key}"
            )
        remapped[mapped_key] = value
        source_keys[mapped_key] = raw_key

    return remapped, renamed_count


def load_compatible_network(
    model,
    net,
    load_path: str,
    strict: bool = True,
    param_key: str | None = "params",
) -> dict[str, int | str | None]:
    logger = get_root_logger()
    bare_net = model.get_bare_model(net)
    load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
    resolved_param_key = param_key
    if resolved_param_key is not None:
        if resolved_param_key not in load_net and "params" in load_net:
            resolved_param_key = "params"
            logger.info("Loading: params_ema does not exist, use params.")
        load_net = load_net[resolved_param_key]

    logger.info(
        "Loading %s model from %s, with param key: [%s].",
        bare_net.__class__.__name__,
        load_path,
        resolved_param_key,
    )

    remapped_net, renamed_count = _remap_legacy_checkpoint_keys(
        load_net, set(bare_net.state_dict().keys())
    )
    if renamed_count:
        logger.info(
            "Applied legacy checkpoint key remapping for %d entries (assm -> hssm).",
            renamed_count,
        )

    model._print_different_keys_loading(bare_net, remapped_net, strict)
    bare_net.load_state_dict(remapped_net, strict=strict)
    return {
        "resolved_param_key": resolved_param_key,
        "renamed_key_count": renamed_count,
        "loaded_key_count": len(remapped_net),
    }


def register_all() -> None:
    targets = [
        (
            ARCH_REGISTRY,
            "HyMambaIR",
            "hymambair_arch",
            os.path.join(CORE_DIR, "basicsr", "archs", "hymambair_arch.py"),
        ),
        (
            MODEL_REGISTRY,
            "HyMambaIRModel",
            "hymambair_model",
            os.path.join(CORE_DIR, "basicsr", "models", "hymambair_model.py"),
        ),
        (
            METRIC_REGISTRY,
            "calculate_lpips_hymambair",
            "lpips_metric",
            os.path.join(CORE_DIR, "basicsr", "metrics", "lpips_metric.py"),
        ),
    ]

    for registry, target_name, module_name, file_path in targets:
        if target_name not in registry:
            _load_module(module_name, file_path)


if __name__ == "__main__":
    register_all()
