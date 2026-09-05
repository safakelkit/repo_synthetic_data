"""Generate a non-training all-class, all-scene full-scene GenAI pilot.

The selected configuration defines the class/scene schedule. The runner uses
only binary SAM3 mask geometry for Canny ControlNet input, not source RGB/RGBA
pixels. It deliberately creates no YOLO labels: human review and the future
localization/QC policy decide which pilot design can be promoted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/generation/genai_remaining_classes_pilot_v4.yaml"
FEASIBILITY_MODULE = REPO_ROOT / "src/generation/run_full_scene_feasibility.py"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        result = yaml.safe_load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"YAML root must be mapping: {path}")
    return result


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feasibility_module():
    spec = importlib.util.spec_from_file_location("genai_feasibility", FEASIBILITY_MODULE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load feasibility helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_masks(manifest: Path, class_id: int) -> list[dict[str, str]]:
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["class_id"]) == class_id and row["status"] == "accepted"]
    if not rows:
        raise ValueError(f"No accepted masks for class {class_id}")
    return sorted(rows, key=lambda row: row["asset_id"])


def rotate_binary(mask: np.ndarray, angle: float) -> np.ndarray:
    height, width = mask.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_width, new_height = int(height * sin + width * cos), int(height * cos + width * sin)
    matrix[0, 2] += new_width / 2 - width / 2
    matrix[1, 2] += new_height / 2 - height / 2
    return cv2.warpAffine(mask, matrix, (new_width, new_height), flags=cv2.INTER_NEAREST)


def draw_semantic_overlay(
    proxy: np.ndarray,
    class_id: int,
    sample_index: int,
    box_xyxy: list[int],
    config: dict[str, Any],
) -> None:
    """Add class-disambiguating Canny geometry without using source RGB/RGBA.

    A binary object silhouette cannot distinguish some visually generic classes
    (for example, phone versus battery, or aerosol can versus bottle).  These
    overlays add only simple class-defining edges to the synthetic ControlNet
    condition. They are pilot-only and are recorded in the configuration.
    """
    overlays = config.get("semantic_control_overlays", {})
    specification = overlays.get(class_id)
    if not specification:
        return
    if isinstance(specification, list):
        specification = specification[sample_index % len(specification)]
    x1, y1, x2, y2 = box_xyxy
    width, height = x2 - x1, y2 - y1
    edge = int(config["control_layout"].get("semantic_edge_value", 32))
    thickness = max(2, round(min(width, height) * float(specification.get("thickness_ratio", 0.025))))

    def point(x_ratio: float, y_ratio: float) -> tuple[int, int]:
        return round(x1 + width * x_ratio), round(y1 + height * y_ratio)

    for line in specification.get("lines", []):
        cv2.line(proxy, point(*line[0]), point(*line[1]), edge, thickness=thickness, lineType=cv2.LINE_AA)
    for rectangle in specification.get("rectangles", []):
        (left, top), (right, bottom) = rectangle
        cv2.rectangle(proxy, point(left, top), point(right, bottom), edge, thickness=thickness, lineType=cv2.LINE_AA)
    for circle in specification.get("circles", []):
        center, radius_ratio = circle
        cv2.circle(proxy, point(*center), max(2, round(min(width, height) * radius_ratio)), edge, thickness=thickness, lineType=cv2.LINE_AA)


def build_control(config: dict[str, Any], row: dict[str, str], sample_index: int, scene_name: str) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    width, height = [int(value) for value in config["output_size"]]
    layout = config["control_layout"]
    proxy = np.full((height, width), int(layout["canvas_value"]), dtype=np.uint8)
    scene_layout = config["scene_layouts"][scene_name]
    support = np.asarray(scene_layout["support_polygon"], dtype=np.int32)
    cv2.fillPoly(proxy, [support], int(layout["support_value"]))
    mask_path = repo_path(row["mask_path"])
    original = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise ValueError(f"Unreadable mask: {mask_path}")
    ys, xs = np.where(original > 0)
    if xs.size == 0:
        raise ValueError(f"Empty mask: {mask_path}")
    binary = np.where(original[ys.min():ys.max() + 1, xs.min():xs.max() + 1] > 0, 255, 0).astype(np.uint8)
    class_rotations = config.get("class_rotation_degrees", {}).get(int(row["class_id"]))
    rotations = class_rotations if class_rotations is not None else (-24, -8, 8, 24)
    angle = float(rotations[sample_index % len(rotations)])
    binary = rotate_binary(binary, angle)
    x1, y1, x2, y2 = [int(value) for value in scene_layout["target_box_xyxy"]]
    box_width, box_height = x2 - x1, y2 - y1
    scale = min(box_width / binary.shape[1], box_height / binary.shape[0]) * (0.82 + 0.06 * (sample_index % 3))
    target_width, target_height = max(1, round(binary.shape[1] * scale)), max(1, round(binary.shape[0] * scale))
    binary = cv2.resize(binary, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    offset_x = (-28, 0, 28)[sample_index % 3]
    offset_y = (-16, 0, 16)[(sample_index // 3) % 3]
    paste_x = max(0, min(width - target_width, x1 + (box_width - target_width) // 2 + offset_x))
    paste_y = max(0, min(height - target_height, y1 + (box_height - target_height) // 2 + offset_y))
    proxy[paste_y:paste_y + target_height, paste_x:paste_x + target_width][binary > 0] = int(layout["target_value"])
    rendered_box = [paste_x, paste_y, paste_x + target_width, paste_y + target_height]
    draw_semantic_overlay(proxy, int(row["class_id"]), sample_index, rendered_box, config)
    blurred = cv2.GaussianBlur(proxy, (0, 0), sigmaX=float(layout["pre_canny_blur_sigma"]))
    low, high = [int(value) for value in layout["canny_thresholds"]]
    canny = cv2.Canny(blurred, low, high)
    return Image.fromarray(cv2.cvtColor(proxy, cv2.COLOR_GRAY2RGB)), Image.fromarray(cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)), {
        "asset_id": row["asset_id"], "mask_path": str(mask_path.relative_to(REPO_ROOT)),
        "mask_sha256": sha256(mask_path), "rotation_degrees": angle,
        "scene_layout": scene_name,
        "rendered_box_xyxy": rendered_box,
    }


def prompt_for(config: dict[str, Any], scene: dict[str, Any], target: str, class_id: int, sample_index: int) -> str:
    description = str(scene["description"]).rstrip(".")
    phrase = config.get("class_target_phrases", {}).get(class_id, target.lower())
    if isinstance(phrase, list):
        phrase = phrase[sample_index % len(phrase)]
    return f"{config['prompt']['prefix']} {description}. {config['prompt']['suffix'].format(target=phrase)}"


def negative_prompt_for(config: dict[str, Any], class_id: int) -> str:
    base = str(config["prompt"]["negative"]).rstrip().rstrip(",")
    additions = config.get("class_negative_additions", {}).get(class_id, [])
    return ", ".join([base, *[str(value) for value in additions]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("sdxl", "qwen"), required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = repo_path(args.config)
    config = load_yaml(config_path)
    models_path = repo_path(config["models_config"])
    scene_path = repo_path(config["scene_policy"])
    generation_path = repo_path(config["generation_policy"])
    models = load_yaml(models_path)
    scene_policy = load_yaml(scene_path)
    helper = feasibility_module()
    output_dir = repo_path(config["output_root"]) / args.backend
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing pilot: {output_dir}")
    if config["status"] != "ready_for_manual_gpu_launch":
        raise ValueError("Pilot config is not approved for manual launch")
    if args.backend not in config["backends"]:
        raise ValueError("Unsupported backend")
    versions = helper.installed_versions()
    mismatches = {
        name: {"expected": expected, "actual": versions.get(name)}
        for name, expected in helper.EXPECTED_PACKAGES.items()
        if versions.get(name) != expected
    }
    if args.backend == "qwen" and versions.get("bitsandbytes") != "0.50.2":
        mismatches["bitsandbytes"] = {
            "expected": "0.50.2", "actual": versions.get("bitsandbytes")
        }
    if mismatches:
        raise RuntimeError(f"Package-version mismatch: {json.dumps(mismatches, indent=2)}")
    git = helper.git_state()
    if git["dirty"]:
        raise RuntimeError("Git worktree is dirty; commit the frozen pilot code before launch")
    if not torch.cuda.is_available() and not args.preflight_only:
        raise RuntimeError("CUDA is not visible")
    if not args.preflight_only and not 0 <= args.gpu < torch.cuda.device_count():
        raise ValueError(
            f"Logical GPU index {args.gpu} is invalid; visible device count is "
            f"{torch.cuda.device_count()}"
        )
    manifest_path = repo_path(config["silhouette_source"]["audit_manifest"])
    schedule: list[dict[str, Any]] = []
    selected_classes = config.get("pilot_scope", {}).get("class_ids")
    selected_ids = {int(value) for value in selected_classes} if selected_classes is not None else None
    for class_id, target in scene_policy["class_policy"]["names"].items():
        if selected_ids is not None and int(class_id) not in selected_ids:
            continue
        for scene_name in scene_policy["class_to_scene_families"][class_id]:
            schedule.append({"class_id": int(class_id), "target": target, "scene_name": scene_name})
    expected = int(config["pilot_scope"]["expected_images_per_backend"])
    if len(schedule) != expected:
        raise ValueError(f"Expected {expected} samples, got {len(schedule)}")
    if args.preflight_only:
        print(json.dumps({"status": "ready", "backend": args.backend, "samples": len(schedule), "output": str(output_dir.relative_to(REPO_ROOT)), "git": git, "packages": versions, "config_sha256": sha256(config_path), "mask_manifest_sha256": sha256(manifest_path)}, indent=2))
        return
    remote_revisions = helper.resolve_remote_revisions(models, args.backend)
    run_started_utc = helper.utc_now()
    run_started = time.monotonic()
    model_load_started = time.monotonic()
    pipe = helper.load_pipeline(args.backend, models, args.gpu)
    model_load_seconds = round(time.monotonic() - model_load_started, 3)
    output_dir.mkdir(parents=True)
    (output_dir / "images").mkdir()
    (output_dir / "controls").mkdir()
    records: list[dict[str, Any]] = []
    for index, sample in enumerate(schedule):
        rows = read_masks(manifest_path, sample["class_id"])
        preferred = config["silhouette_source"].get("preferred_stems", {}).get(sample["class_id"])
        if preferred:
            stem = preferred[index % len(preferred)]
            matches = [candidate for candidate in rows if candidate["stem"] == stem]
            if len(matches) != 1:
                raise ValueError(f"Preferred silhouette is not unique/accepted: class={sample['class_id']} stem={stem}")
            row = matches[0]
        else:
            row = rows[(int(config["seed"]) + int(config["silhouette_source"]["selection_seed_offset"]) + index) % len(rows)]
        proxy, control, silhouette = build_control(config, row, index, sample["scene_name"])
        prompt = prompt_for(config, scene_policy["scene_families"][sample["scene_name"]], sample["target"], sample["class_id"], index)
        generator = torch.Generator(device="cpu").manual_seed(int(config["seed"]) + index)
        class_scale = config.get("class_controlnet_conditioning_scale", {}).get(sample["class_id"], config["controlnet_conditioning_scale"])
        width, height = [int(value) for value in config["output_size"]]
        common = {"prompt": prompt, "negative_prompt": negative_prompt_for(config, sample["class_id"]), "height": height, "width": width, "num_inference_steps": int(config["inference_steps"]), "controlnet_conditioning_scale": float(class_scale), "generator": generator}
        started = time.monotonic()
        result = pipe(image=control, guidance_scale=float(models["sdxl"]["guidance_scale"]), **common) if args.backend == "sdxl" else pipe(control_image=control, true_cfg_scale=float(models["qwen"]["true_cfg_scale"]), **common)
        torch.cuda.synchronize(args.gpu)
        image_name = f"{index:03d}_c{sample['class_id']:02d}_{sample['scene_name']}.png"
        output_path = output_dir / "images" / image_name
        proxy_path = output_dir / "controls" / f"{index:03d}_proxy.png"
        control_path = output_dir / "controls" / f"{index:03d}_canny.png"
        result.images[0].save(output_path)
        proxy.save(proxy_path)
        control.save(control_path)
        records.append({**sample, "index": index, "seed": int(config["seed"]) + index, "prompt": prompt, "negative_prompt": common["negative_prompt"], "controlnet_conditioning_scale": float(class_scale), "output": str(output_path.relative_to(REPO_ROOT)), "output_sha256": sha256(output_path), "silhouette": silhouette, "proxy_sha256": sha256(proxy_path), "control_sha256": sha256(control_path), "inference_seconds": round(time.monotonic() - started, 3), "annotation_performed": False, "degradation_applied": False, "training_use_forbidden": True})
        print(json.dumps(records[-1], ensure_ascii=False))
    torch.cuda.synchronize(args.gpu)
    manifest = {
        "format_version": 2,
        "pilot_id": config["pilot_id"],
        "backend": args.backend,
        "status": "generated_pending_human_review",
        "started_utc": run_started_utc,
        "completed_utc": helper.utc_now(),
        "wall_time_seconds": round(time.monotonic() - run_started, 3),
        "model_load_seconds": model_load_seconds,
        "git": git,
        "packages": versions,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_gpu": args.gpu,
        "gpu_name": torch.cuda.get_device_name(args.gpu),
        "gpu_compute_capability": ".".join(str(part) for part in torch.cuda.get_device_capability(args.gpu)),
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated(args.gpu) / 1024**3, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved(args.gpu) / 1024**3, 3),
        "remote_revisions": remote_revisions,
        "models": {
            "base_model": models[args.backend]["base_model"],
            "controlnet": models[args.backend]["controlnet"],
            "pipeline_class": models[args.backend]["pipeline_class"],
        },
        "config_sha256": {
            "pilot": sha256(config_path),
            "models": sha256(models_path),
            "scene_policy": sha256(scene_path),
            "generation_policy": sha256(generation_path),
            "mask_manifest": sha256(manifest_path),
            "script": sha256(Path(__file__).resolve()),
        },
        "records": records,
    }
    with (output_dir / "pilot_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
