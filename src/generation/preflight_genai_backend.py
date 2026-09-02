from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch
import yaml
from huggingface_hub import HfApi


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = {
    "sdxl": REPO_ROOT / "configs/generation/sdxl_controlnet_v1.yaml",
    "qwen": REPO_ROOT / "configs/generation/qwen_controlnet_v1.yaml",
}
REQUIRED_PACKAGES = (
    "diffusers",
    "transformers",
    "accelerate",
    "huggingface-hub",
    "safetensors",
    "sentencepiece",
    "protobuf",
    "torch",
)


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def package_versions() -> tuple[dict[str, str], list[str]]:
    versions: dict[str, str] = {}
    missing: list[str] = []
    for package in REQUIRED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            missing.append(package)
    return versions, missing


def model_record(api: HfApi, entry: dict[str, str]) -> dict[str, Any]:
    requested_revision = entry["revision"]
    revision = None if requested_revision.startswith("resolve_and_freeze") else requested_revision
    info = api.model_info(entry["id"], revision=revision, files_metadata=True)
    known_size = sum(sibling.size or 0 for sibling in info.siblings)
    return {
        "id": entry["id"],
        "requested_revision": requested_revision,
        "resolved_revision": info.sha,
        "license_declared_in_config": entry["license"],
        "known_repository_bytes": known_size,
        "known_repository_gib": round(known_size / 1024**3, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve model revisions and verify a GenAI smoke-test environment"
    )
    parser.add_argument("--backend", choices=tuple(CONFIGS), required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    config_path = CONFIGS[args.backend]
    config = load_yaml(config_path)
    versions, missing = package_versions()
    cuda_available = torch.cuda.is_available()
    visible_devices = torch.cuda.device_count() if cuda_available else 0
    gpu_valid = cuda_available and 0 <= args.gpu < visible_devices
    api = HfApi()
    models = {
        "base_model": model_record(api, config["base_model"]),
        "controlnet": model_record(api, config["controlnet"]),
    }
    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    cache_root.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(cache_root)
    report = {
        "format_version": 1,
        "status": "ready_for_smoke_test" if not missing and gpu_valid else "not_ready",
        "backend": args.backend,
        "config": config_path.relative_to(REPO_ROOT).as_posix(),
        "packages": versions,
        "missing_packages": missing,
        "cuda_available": cuda_available,
        "visible_cuda_devices": visible_devices,
        "selected_logical_gpu": args.gpu,
        "gpu_name": torch.cuda.get_device_name(args.gpu) if gpu_valid else None,
        "cache_root": str(cache_root),
        "cache_free_gib": round(disk.free / 1024**3, 3),
        "models": models,
        "model_download_started": False,
        "inference_started": False,
        "notes": (
            "Qwen-Image is substantially larger than one RTX 3090; the smoke test must "
            "measure CPU offload or multi-GPU behavior before production."
            if args.backend == "qwen"
            else "SDXL smoke test should verify peak VRAM, runtime, mask locality, and output quality."
        ),
    }
    output_root = REPO_ROOT / "data/processed/genai_v1"
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{args.backend}_model_preflight.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "ready_for_smoke_test":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
