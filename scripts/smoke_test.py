#!/usr/bin/env python3
"""Minimal smoke test for the staged Hy-MambaIR repository."""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from register_models import register_all  # noqa: E402


def main() -> None:
    import importlib

    modules = ["torch", "basicsr", "mamba_ssm", "causal_conv1d", "lpips", "timm"]
    result = {name: importlib.util.find_spec(name) is not None for name in modules}
    if all(result.values()):
        try:
            register_all()
            result["register_all"] = True
        except Exception as exc:
            result["register_all"] = False
            result["register_error"] = str(exc)
    else:
        result["register_all"] = False
        result["register_error"] = (
            "Skipped registration because one or more dependencies are missing."
        )
    result["cwd"] = os.getcwd()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
