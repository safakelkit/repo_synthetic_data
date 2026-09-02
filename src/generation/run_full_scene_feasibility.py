"""Run one traceable full-scene ControlNet feasibility image per backend.

This is deliberately not a dataset generator. It validates the frozen model
pair, common synthetic Canny condition, RTX 3090 memory strategy, and output
provenance before an all-class pilot is implemented.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from huggingface_hub import model_info
from huggingface_hub.constants import HF_HUB_CACHE
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/generation/genai_feasibility_v3.yaml"
EXPECTED_PACKAGES = {
    "diffusers": "0.40.0",
    "transformers": "5.5.4",
    "accelerate": "1.13.0",
    "huggingface-hub": "1.29.0",
    "safetensors": "0.8.0",
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_state() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    porcelain = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    return {"revision": revision, "dirty": bool(porcelain)}


def installed_versions() -> dict[str, str]:
    names = [
        "torch", "diffusers", "transformers", "accelerate",
        "huggingface-hub", "safetensors", "bitsandbytes", "opencv-python",
        "Pillow", "PyYAML",
    ]
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "NOT_INSTALLED"
    return versions


def validate_config(
    feasibility: dict[str, Any], models: dict[str, Any], scene: dict[str, Any], backend: str
) -> None:
    if feasibility.get("status") != "ready_for_manual_gpu_launch":
        raise ValueError("Feasibility config is not approved for manual launch")
    if models.get("status") != "frozen_for_single_image_feasibility_v2":
        raise ValueError("Model config is not frozen for feasibility")
    if backend not in feasibility.get("backends", []):
        raise ValueError(f"Unsupported backend: {backend}")
    if scene.get("status") != "frozen_for_genai_baselines":
        raise ValueError("Scene policy is not frozen")
    accepted_key = "accepted" if backend == "sdxl" else "accepted_for_feasibility"
    if not models[backend].get(accepted_key):
        raise ValueError(f"{backend} is not accepted for feasibility")
    if models.get("release_gate", {}).get("canonical_generation_approved"):
        raise ValueError("Feasibility config must not approve canonical generation")
    shared = models["shared"]
    class_id = int(shared["target_class_id"])
    expected_name = scene["class_policy"]["names"].get(class_id)
    if expected_name != shared["target_class_name"]:
        raise ValueError("Target class ID/name disagrees with the frozen scene policy")
    assigned = scene["class_to_scene_families"].get(class_id, [])
    if shared["scene_family"] not in assigned:
        raise ValueError("Feasibility scene is not assigned to the target class")
    if feasibility.get("samples_per_backend") != 1:
        raise ValueError("This runner is restricted to one image per backend")
    if not feasibility.get("same_control_for_both_backends"):
        raise ValueError("Both backends must use the same control layout")


def select_real_class_mask(config: dict[str, Any], class_id: int) -> dict[str, str]:
    silhouette = config["silhouette_source"]
    manifest_path = resolve_repo_path(silhouette["audit_manifest"])
    candidates: list[dict[str, str]] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["class_id"]) == class_id and row["status"] == "accepted":
                candidates.append(row)
    if not candidates:
        raise ValueError(f"No accepted real silhouette for class {class_id}")
    candidates.sort(key=lambda row: row["asset_id"])
    index = int(silhouette["selection_seed"]) % len(candidates)
    return candidates[index]


def draw_control(
    config: dict[str, Any], width: int, height: int, class_id: int
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    layout = config["control_layout"]
    canvas_value = int(layout["canvas_value"])
    table_value = int(layout["table_value"])
    object_value = int(layout["object_value"])
    proxy = np.full((height, width), canvas_value, dtype=np.uint8)
    table = np.asarray(layout["table_polygon"], dtype=np.int32)
    cv2.fillPoly(proxy, [table], table_value)

    # Only the binary SAM3 silhouette is reused. No source RGB/RGBA pixels enter
    # the generated scene or the ControlNet condition.
    x1, y1, x2, y2 = [int(v) for v in layout["target_box_xyxy"]]
    selected = select_real_class_mask(config, class_id)
    mask_path = resolve_repo_path(selected["mask_path"])
    source_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if source_mask is None:
        raise ValueError(f"Cannot read silhouette mask: {mask_path}")
    binary = np.where(source_mask > 0, 255, 0).astype(np.uint8)
    ys, xs = np.where(binary > 0)
    if xs.size == 0 or ys.size == 0:
        raise ValueError(f"Empty silhouette mask: {mask_path}")
    binary = binary[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    box_w, box_h = x2 - x1, y2 - y1
    scale = min(box_w / binary.shape[1], box_h / binary.shape[0])
    resized_w = max(1, round(binary.shape[1] * scale))
    resized_h = max(1, round(binary.shape[0] * scale))
    resized = cv2.resize(binary, (resized_w, resized_h), interpolation=cv2.INTER_NEAREST)
    paste_x = x1 + (box_w - resized_w) // 2
    paste_y = y1 + (box_h - resized_h) // 2
    region = proxy[paste_y:paste_y + resized_h, paste_x:paste_x + resized_w]
    region[resized > 0] = object_value

    blurred = cv2.GaussianBlur(
        proxy, (0, 0), sigmaX=float(layout["pre_canny_blur_sigma"])
    )
    low, high = [int(value) for value in layout["canny_thresholds"]]
    canny = cv2.Canny(blurred, low, high)
    proxy_rgb = cv2.cvtColor(proxy, cv2.COLOR_GRAY2RGB)
    canny_rgb = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
    metadata = {
        "source_type": "real_class_sam3_binary_mask",
        "asset_id": selected["asset_id"],
        "class_id": class_id,
        "mask_path": str(mask_path.relative_to(REPO_ROOT)),
        "mask_sha256": sha256_file(mask_path),
        "audit_manifest": str(resolve_repo_path(config["silhouette_source"]["audit_manifest"]).relative_to(REPO_ROOT)),
        "audit_manifest_sha256": sha256_file(resolve_repo_path(config["silhouette_source"]["audit_manifest"])),
        "selection_seed": int(config["silhouette_source"]["selection_seed"]),
        "real_pixels_reused": False,
        "rendered_box_xyxy": [paste_x, paste_y, paste_x + resized_w, paste_y + resized_h],
    }
    return Image.fromarray(proxy_rgb), Image.fromarray(canny_rgb), metadata


def resolve_remote_revisions(models: dict[str, Any], backend: str) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for component in ("base_model", "controlnet"):
        item = models[backend][component]
        info = model_info(item["id"], revision=item["revision"])
        if info.sha != item["revision"]:
            raise RuntimeError(
                f"Resolved {component} revision {info.sha} does not match frozen {item['revision']}"
            )
        resolved[component] = info.sha
    return resolved


def preflight(
    backend: str,
    feasibility_path: Path,
    models_path: Path,
    scene_path: Path,
    feasibility: dict[str, Any],
    models: dict[str, Any],
    scene: dict[str, Any],
    output_dir: Path,
    require_gpu: bool,
) -> dict[str, Any]:
    validate_config(feasibility, models, scene, backend)
    versions = installed_versions()
    mismatches = {
        name: {"expected": expected, "actual": versions.get(name)}
        for name, expected in EXPECTED_PACKAGES.items()
        if versions.get(name) != expected
    }
    if backend == "qwen" and versions["bitsandbytes"] != "0.50.2":
        mismatches["bitsandbytes"] = {
            "expected": "0.50.2", "actual": versions["bitsandbytes"]
        }
    if mismatches:
        raise RuntimeError(f"Package-version mismatch: {json.dumps(mismatches, indent=2)}")

    state = git_state()
    if state["dirty"]:
        raise RuntimeError("Git worktree is dirty; commit the frozen feasibility code before launch")
    if output_dir.exists():
        raise FileExistsError(f"Output already exists and will not be overwritten: {output_dir}")
    if require_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not visible in this process")
        if torch.cuda.device_count() <= 0:
            raise RuntimeError("No logical CUDA device is visible")
        if not torch.cuda.is_bf16_supported() and backend == "qwen":
            raise RuntimeError("Qwen feasibility requires BF16 support")

    usage = shutil.disk_usage(REPO_ROOT)
    hub_cache = Path(HF_HUB_CACHE)
    hub_storage_root = hub_cache
    while not hub_storage_root.exists():
        hub_storage_root = hub_storage_root.parent
    hub_usage = shutil.disk_usage(hub_storage_root)
    mem_available_kib = 0
    with Path("/proc/meminfo").open("r", encoding="utf-8") as handle:
        for row in handle:
            if row.startswith("MemAvailable:"):
                mem_available_kib = int(row.split()[1])
                break
    return {
        "backend": backend,
        "status": "ready",
        "git": state,
        "packages": versions,
        "config_sha256": {
            "feasibility": sha256_file(feasibility_path),
            "models": sha256_file(models_path),
            "scene_policy": sha256_file(scene_path),
            "script": sha256_file(Path(__file__).resolve()),
        },
        "available_disk_gib": round(usage.free / 1024**3, 2),
        "huggingface_cache": str(hub_cache),
        "huggingface_cache_filesystem_free_gib": round(hub_usage.free / 1024**3, 2),
        "available_ram_gib": round(mem_available_kib / 1024**2, 2),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def load_pipeline(backend: str, models: dict[str, Any], logical_gpu: int):
    from diffusers import (
        ControlNetModel,
        PipelineQuantizationConfig,
        QwenImageControlNetModel,
        QwenImageControlNetPipeline,
        StableDiffusionXLControlNetPipeline,
    )

    cfg = models[backend]
    base = cfg["base_model"]
    control = cfg["controlnet"]
    if backend == "sdxl":
        controlnet = ControlNetModel.from_pretrained(
            control["id"], revision=control["revision"],
            torch_dtype=torch.float16, variant=cfg["variant"], use_safetensors=True,
        )
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            base["id"], revision=base["revision"], controlnet=controlnet,
            torch_dtype=torch.float16, variant=cfg["variant"], use_safetensors=True,
        )
    else:
        quant = cfg["quantization"]
        quantization_config = PipelineQuantizationConfig(
            quant_backend=quant["backend"],
            quant_kwargs={
                "load_in_4bit": quant["load_in_4bit"],
                "bnb_4bit_quant_type": quant["bnb_4bit_quant_type"],
                "bnb_4bit_compute_dtype": torch.bfloat16,
            },
            components_to_quantize=list(quant["components"]),
        )
        controlnet = QwenImageControlNetModel.from_pretrained(
            control["id"], revision=control["revision"], torch_dtype=torch.bfloat16,
        )
        pipe = QwenImageControlNetPipeline.from_pretrained(
            base["id"], revision=base["revision"], controlnet=controlnet,
            torch_dtype=torch.bfloat16, quantization_config=quantization_config,
        )
    pipe.enable_model_cpu_offload(gpu_id=logical_gpu)
    if hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    return pipe


def generate(
    backend: str, pipe, control: Image.Image, models: dict[str, Any], logical_gpu: int
) -> Image.Image:
    shared = models["shared"]
    cfg = models[backend]
    generator = torch.Generator(device="cpu").manual_seed(int(shared["seed"]))
    common = {
        "prompt": shared["prompt"],
        "negative_prompt": shared["negative_prompt"],
        "height": int(shared["output_height"]),
        "width": int(shared["output_width"]),
        "num_inference_steps": int(shared["inference_steps"]),
        "controlnet_conditioning_scale": float(shared["controlnet_conditioning_scale"]),
        "generator": generator,
    }
    if backend == "sdxl":
        result = pipe(
            image=control, guidance_scale=float(cfg["guidance_scale"]), **common
        )
    else:
        result = pipe(
            control_image=control, true_cfg_scale=float(cfg["true_cfg_scale"]), **common
        )
    return result.images[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("sdxl", "qwen"), required=True)
    parser.add_argument("--gpu", type=int, default=0, help="Logical CUDA index after CUDA_VISIBLE_DEVICES")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    feasibility_path = resolve_repo_path(args.config)
    feasibility = load_yaml(feasibility_path)
    models_path = resolve_repo_path(feasibility["models_config"])
    scene_path = resolve_repo_path(feasibility["scene_policy"])
    models = load_yaml(models_path)
    scene = load_yaml(scene_path)
    output_dir = resolve_repo_path(feasibility["output_root"]) / args.backend

    report = preflight(
        args.backend, feasibility_path, models_path, scene_path,
        feasibility, models, scene, output_dir, require_gpu=not args.preflight_only,
    )
    report["remote_revisions"] = resolve_remote_revisions(models, args.backend)
    if args.preflight_only:
        print(json.dumps(report, indent=2))
        return

    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    started_utc = utc_now()
    proxy, control, silhouette_metadata = draw_control(
        feasibility, int(models["shared"]["output_width"]),
        int(models["shared"]["output_height"]), int(models["shared"]["target_class_id"]),
    )
    control_path = output_dir / feasibility["output_files"]["control_image"]
    proxy_path = output_dir / feasibility["output_files"]["proxy_layout"]
    output_path = output_dir / feasibility["output_files"]["generated_image"]
    record_path = output_dir / feasibility["output_files"]["run_record"]
    resolved_path = output_dir / feasibility["output_files"]["resolved_config"]
    control.save(control_path)
    proxy.save(proxy_path)
    with resolved_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {"feasibility": feasibility, "models": models, "scene_policy": scene},
            handle, sort_keys=False, allow_unicode=True,
        )

    report.update({
        "format_version": 1,
        "started_utc": started_utc,
        "logical_gpu": args.gpu,
        "gpu_name": torch.cuda.get_device_name(args.gpu),
        "gpu_compute_capability": ".".join(
            str(part) for part in torch.cuda.get_device_capability(args.gpu)
        ),
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "python": platform.python_version(),
        "target_class_id": models["shared"]["target_class_id"],
        "target_class_name": models["shared"]["target_class_name"],
        "scene_family": models["shared"]["scene_family"],
        "seed": models["shared"]["seed"],
        "prompt": models["shared"]["prompt"],
        "negative_prompt": models["shared"]["negative_prompt"],
        "control_sha256": sha256_file(control_path),
        "proxy_layout_sha256": sha256_file(proxy_path),
        "silhouette_source": silhouette_metadata,
        "canonical_degradation_applied": False,
        "annotation_performed": False,
        "canonical_dataset_modified": False,
    })
    try:
        torch.cuda.reset_peak_memory_stats(args.gpu)
        load_started = time.monotonic()
        pipe = load_pipeline(args.backend, models, args.gpu)
        load_seconds = time.monotonic() - load_started
        inference_started = time.monotonic()
        generated = generate(args.backend, pipe, control, models, args.gpu)
        inference_seconds = time.monotonic() - inference_started
        generated.save(output_path)
        torch.cuda.synchronize(args.gpu)
        report.update({
            "status": "feasibility_generated_pending_human_review",
            "model_load_seconds": round(load_seconds, 3),
            "inference_seconds": round(inference_seconds, 3),
            "peak_allocated_gib": round(
                torch.cuda.max_memory_allocated(args.gpu) / 1024**3, 3
            ),
            "peak_reserved_gib": round(
                torch.cuda.max_memory_reserved(args.gpu) / 1024**3, 3
            ),
            "output_sha256": sha256_file(output_path),
            "output": str(output_path.relative_to(REPO_ROOT)),
        })
    except Exception as error:
        report.update({
            "status": "feasibility_failed",
            "error_type": type(error).__name__,
            "error": str(error),
        })
        raise
    finally:
        report["completed_utc"] = utc_now()
        report["wall_time_seconds"] = round(time.monotonic() - started, 3)
        if torch.cuda.is_available():
            report["peak_allocated_gib"] = round(
                torch.cuda.max_memory_allocated(args.gpu) / 1024**3, 3
            )
            report["peak_reserved_gib"] = round(
                torch.cuda.max_memory_reserved(args.gpu) / 1024**3, 3
            )
        with record_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
