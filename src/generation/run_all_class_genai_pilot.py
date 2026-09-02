"""Generate a non-training all-class, all-scene full-scene GenAI pilot.

This runner creates 64 images per backend: one for every frozen class/scene
assignment.  It uses only binary SAM3 mask geometry for Canny ControlNet input,
not source RGB/RGBA pixels.  It deliberately creates no YOLO labels: human
review and the future localization/QC policy decide which pilot design can be
promoted to canonical generation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/generation/genai_all_class_pilot_v2.yaml"
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
    angle = (-24, -8, 8, 24)[sample_index % 4]
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
    blurred = cv2.GaussianBlur(proxy, (0, 0), sigmaX=float(layout["pre_canny_blur_sigma"]))
    low, high = [int(value) for value in layout["canny_thresholds"]]
    canny = cv2.Canny(blurred, low, high)
    return Image.fromarray(cv2.cvtColor(proxy, cv2.COLOR_GRAY2RGB)), Image.fromarray(cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)), {
        "asset_id": row["asset_id"], "mask_path": str(mask_path.relative_to(REPO_ROOT)),
        "mask_sha256": sha256(mask_path), "rotation_degrees": angle,
        "scene_layout": scene_name,
        "rendered_box_xyxy": [paste_x, paste_y, paste_x + target_width, paste_y + target_height],
    }


def prompt_for(config: dict[str, Any], scene: dict[str, Any], target: str, class_id: int) -> str:
    description = str(scene["description"]).rstrip(".")
    phrase = config.get("class_target_phrases", {}).get(class_id, target.lower())
    return f"{config['prompt']['prefix']} {description}. {config['prompt']['suffix'].format(target=phrase)}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("sdxl", "qwen"), required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = repo_path(args.config)
    config = load_yaml(config_path)
    models = load_yaml(repo_path(config["models_config"]))
    scene_policy = load_yaml(repo_path(config["scene_policy"]))
    output_dir = repo_path(config["output_root"]) / args.backend
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing pilot: {output_dir}")
    if config["status"] != "ready_for_manual_gpu_launch":
        raise ValueError("Pilot config is not approved for manual launch")
    if args.backend not in config["backends"]:
        raise ValueError("Unsupported backend")
    if not torch.cuda.is_available() and not args.preflight_only:
        raise RuntimeError("CUDA is not visible")
    manifest_path = repo_path(config["silhouette_source"]["audit_manifest"])
    schedule: list[dict[str, Any]] = []
    for class_id, target in scene_policy["class_policy"]["names"].items():
        for scene_name in scene_policy["class_to_scene_families"][class_id]:
            schedule.append({"class_id": int(class_id), "target": target, "scene_name": scene_name})
    expected = int(config["pilot_scope"]["expected_images_per_backend"])
    if len(schedule) != expected:
        raise ValueError(f"Expected {expected} samples, got {len(schedule)}")
    if args.preflight_only:
        print(json.dumps({"status": "ready", "backend": args.backend, "samples": len(schedule), "output": str(output_dir.relative_to(REPO_ROOT)), "config_sha256": sha256(config_path), "mask_manifest_sha256": sha256(manifest_path)}, indent=2))
        return
    output_dir.mkdir(parents=True)
    (output_dir / "images").mkdir()
    (output_dir / "controls").mkdir()
    helper = feasibility_module()
    pipe = helper.load_pipeline(args.backend, models, args.gpu)
    records: list[dict[str, Any]] = []
    for index, sample in enumerate(schedule):
        rows = read_masks(manifest_path, sample["class_id"])
        preferred = config["silhouette_source"].get("preferred_stems", {}).get(sample["class_id"])
        if preferred:
            stem = preferred[index % 4]
            matches = [candidate for candidate in rows if candidate["stem"] == stem]
            if len(matches) != 1:
                raise ValueError(f"Preferred silhouette is not unique/accepted: class={sample['class_id']} stem={stem}")
            row = matches[0]
        else:
            row = rows[(int(config["seed"]) + int(config["silhouette_source"]["selection_seed_offset"]) + index) % len(rows)]
        proxy, control, silhouette = build_control(config, row, index, sample["scene_name"])
        prompt = prompt_for(config, scene_policy["scene_families"][sample["scene_name"]], sample["target"], sample["class_id"])
        generator = torch.Generator(device="cpu").manual_seed(int(config["seed"]) + index)
        common = {"prompt": prompt, "negative_prompt": config["prompt"]["negative"], "height": 1024, "width": 1024, "num_inference_steps": int(config["inference_steps"]), "controlnet_conditioning_scale": float(config["controlnet_conditioning_scale"]), "generator": generator}
        started = time.monotonic()
        result = pipe(image=control, guidance_scale=float(models["sdxl"]["guidance_scale"]), **common) if args.backend == "sdxl" else pipe(control_image=control, true_cfg_scale=float(models["qwen"]["true_cfg_scale"]), **common)
        image_name = f"{index:03d}_c{sample['class_id']:02d}_{sample['scene_name']}.png"
        result.images[0].save(output_dir / "images" / image_name)
        proxy.save(output_dir / "controls" / f"{index:03d}_proxy.png")
        control.save(output_dir / "controls" / f"{index:03d}_canny.png")
        records.append({**sample, "index": index, "seed": int(config["seed"]) + index, "prompt": prompt, "output": str((output_dir / "images" / image_name).relative_to(REPO_ROOT)), "silhouette": silhouette, "control_sha256": sha256(output_dir / "controls" / f"{index:03d}_canny.png"), "inference_seconds": round(time.monotonic() - started, 3), "annotation_performed": False, "degradation_applied": False, "training_use_forbidden": True})
        print(json.dumps(records[-1], ensure_ascii=False))
    with (output_dir / "pilot_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"backend": args.backend, "status": "generated_pending_human_review", "records": records}, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
