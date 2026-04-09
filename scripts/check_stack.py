#!/usr/bin/env python
"""检查当前 pixi 环境中的关键依赖是否可用。"""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class PackageSpec:
    import_name: str
    display_name: str


COMMON_PACKAGES = [
    PackageSpec("torch", "PyTorch"),
    PackageSpec("transformers", "Transformers"),
    PackageSpec("datasets", "Datasets"),
    PackageSpec("accelerate", "Accelerate"),
    PackageSpec("trl", "TRL"),
    PackageSpec("sympy", "SymPy"),
]

DPO_ONLY_PACKAGES = [
    PackageSpec("llamafactory", "LLaMA-Factory"),
    PackageSpec("deepspeed", "DeepSpeed"),
    PackageSpec("bitsandbytes", "bitsandbytes"),
]

GRPO_ONLY_PACKAGES = [
    PackageSpec("vllm", "vLLM"),
    PackageSpec("ray", "Ray"),
    PackageSpec("tensordict", "TensorDict"),
]


def import_and_report(package: PackageSpec) -> bool:
    """导入模块并打印其版本信息。"""
    try:
        module = importlib.import_module(package.import_name)
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {package.display_name}: {exc}")
        return False

    version = getattr(module, "__version__", "unknown")
    print(f"[ OK ] {package.display_name}: {version}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="检查项目环境是否安装成功。")
    parser.add_argument(
        "--env",
        required=True,
        choices=["dpo", "grpo"],
        help="当前检查的是哪个 pixi 环境。",
    )
    args = parser.parse_args()

    package_specs = list(COMMON_PACKAGES)
    if args.env == "dpo":
        package_specs.extend(DPO_ONLY_PACKAGES)
    else:
        package_specs.extend(GRPO_ONLY_PACKAGES)

    print(f"Python: {sys.version.split()[0]}")
    print(f"Environment: {args.env}")

    all_ok = True
    for spec in package_specs:
        all_ok &= import_and_report(spec)

    try:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
            bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            print(f"BF16 supported: {bf16_supported}")
    except Exception as exc:  # noqa: BLE001
        all_ok = False
        print(f"[FAIL] CUDA check: {exc}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
