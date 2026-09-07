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
DEFAULT_CONFIG = REPO_ROOT / "configs/generation/sdxl_generation.yaml"
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


def internal_object_edges(
    row: dict[str, str],
    original_mask: np.ndarray,
    crop_bounds: tuple[int, int, int, int],
    angle: float,
    target_size: tuple[int, int],
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract non-textural object edges from the accepted RGB crop.

    This is condition-map construction only.  The RGB crop is converted to a
    blurred, masked edge map and is never passed to SDXL or composited into an
    output image.
    """
    source_path = repo_path(row["rgb_path"])
    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if source is None or source.shape[:2] != original_mask.shape:
        raise ValueError(f"Unreadable or misaligned source RGB crop: {source_path}")
    y1, y2, x1, x2 = crop_bounds
    source = source[y1:y2, x1:x2]
    mask = np.where(original_mask[y1:y2, x1:x2] > 0, 255, 0).astype(np.uint8)
    settings = config["internal_edge_control"]
    # A bilateral pass preserves large cap/base seams; the subsequent Gaussian
    # pass deliberately removes printed labels and fine source texture.
    filtered = cv2.bilateralFilter(
        source,
        int(settings.get("bilateral_diameter", 7)),
        float(settings.get("bilateral_sigma_color", 45)),
        float(settings.get("bilateral_sigma_space", 45)),
    )
    filtered = cv2.GaussianBlur(
        filtered, (0, 0), sigmaX=float(settings.get("gaussian_sigma", 3.0))
    )
    grayscale = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    low, high = [int(value) for value in settings.get("canny_thresholds", [24, 72])]
    edges = cv2.Canny(grayscale, low, high)
    erosion = max(1, int(settings.get("mask_erosion_px", 2)))
    inside = cv2.erode(mask, np.ones((erosion * 2 + 1, erosion * 2 + 1), np.uint8))
    edges = cv2.bitwise_and(edges, inside)
    edges = rotate_binary(edges, angle)
    edges = cv2.resize(edges, target_size, interpolation=cv2.INTER_NEAREST)
    if int(settings.get("dilate_px", 0)):
        radius = int(settings["dilate_px"])
        edges = cv2.dilate(edges, np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8))
    return edges, {
        "source_type": "accepted_rgb_masked_blurred_internal_edges_only",
        "rgb_path": str(source_path.relative_to(REPO_ROOT)),
        "rgb_sha256": sha256(source_path),
        "gaussian_sigma": float(settings.get("gaussian_sigma", 3.0)),
        "canny_thresholds": [low, high],
        "mask_erosion_px": erosion,
    }


def draw_scene_structure(
    proxy: np.ndarray,
    config: dict[str, Any],
    scene_name: str,
    sample_index: int,
    target_box_xyxy: list[int],
) -> dict[str, Any] | None:
    """Draw fixed room architecture into the Canny proxy.

    The configured primitives describe large installed fixtures such as beds,
    cabinets, windows, sinks and doors.  They intentionally avoid small loose
    objects that could be mistaken for one of the detection classes.
    """
    variants = config.get("scene_structure_variants", {}).get(scene_name, [])
    if not variants:
        return None
    variant = variants[sample_index % len(variants)]
    height, width = proxy.shape
    before = proxy.copy()
    default_value = int(config.get("scene_structure_value", 72))
    default_thickness = int(config.get("scene_structure_thickness_px", 7))

    def point(value: list[float]) -> tuple[int, int]:
        return (
            int(round(float(value[0]) * (width - 1))),
            int(round(float(value[1]) * (height - 1))),
        )

    for primitive in variant.get("primitives", []):
        kind = str(primitive["type"])
        value = int(primitive.get("value", default_value))
        thickness = int(primitive.get("thickness", default_thickness))
        if kind == "line":
            cv2.line(proxy, point(primitive["points"][0]), point(primitive["points"][1]), value, thickness, cv2.LINE_AA)
        elif kind == "polyline":
            points = np.asarray([point(item) for item in primitive["points"]], dtype=np.int32)
            cv2.polylines(proxy, [points], bool(primitive.get("closed", False)), value, thickness, cv2.LINE_AA)
        elif kind == "rectangle":
            cv2.rectangle(proxy, point(primitive["xyxy"][:2]), point(primitive["xyxy"][2:]), value, thickness, cv2.LINE_AA)
        elif kind == "ellipse":
            center = point(primitive["center"])
            axes = (
                max(1, int(round(float(primitive["axes"][0]) * width))),
                max(1, int(round(float(primitive["axes"][1]) * height))),
            )
            cv2.ellipse(proxy, center, axes, 0, 0, 360, value, thickness, cv2.LINE_AA)
        else:
            raise ValueError(f"Unsupported scene structure primitive: {kind}")

    # Keep the object placement area free of architectural control edges. This
    # preserves the successful target silhouette behavior for every class.
    padding = int(config.get("scene_structure_target_clearance_px", 28))
    x1, y1, x2, y2 = [int(value) for value in target_box_xyxy]
    x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
    x2, y2 = min(width, x2 + padding), min(height, y2 + padding)
    proxy[y1:y2, x1:x2] = before[y1:y2, x1:x2]
    return {
        "variant": str(variant.get("name", sample_index % len(variants))),
        "variant_index": sample_index % len(variants),
        "primitive_count": len(variant.get("primitives", [])),
        "target_clearance_xyxy": [x1, y1, x2, y2],
    }


def build_control(config: dict[str, Any], row: dict[str, str] | None, sample_index: int, scene_name: str) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    width, height = [int(value) for value in config["output_size"]]
    class_id = int(row["class_id"]) if row is not None else None
    class_layouts = config.get("class_control_layout_overrides", {})
    layout = {
        **config["control_layout"],
        **(class_layouts.get(class_id, {}) if class_id is not None else {}),
    }
    proxy = np.full((height, width), int(layout["canvas_value"]), dtype=np.uint8)
    class_scenes = config.get("class_scene_layout_overrides", {})
    scene_layout = {
        **config["scene_layouts"][scene_name],
        **(
            class_scenes.get(class_id, {}).get(scene_name, {})
            if class_id is not None
            else {}
        ),
    }
    variation_rng = np.random.default_rng(int(config.get("seed", 0)) + sample_index * 104729)
    support = np.asarray(scene_layout["support_polygon"], dtype=np.int32).copy()
    support_jitter = int(layout.get("support_jitter_px", 0))
    if support_jitter:
        support += variation_rng.integers(-support_jitter, support_jitter + 1, size=support.shape)
        support[:, 0] = np.clip(support[:, 0], 0, width - 1)
        support[:, 1] = np.clip(support[:, 1], 0, height - 1)
    mode = config.get("target_conditioning_mode", "silhouette")
    if mode != "target_only":
        cv2.fillPoly(proxy, [support], int(layout["support_value"]))
        scene_structure = draw_scene_structure(
            proxy,
            config,
            scene_name,
            sample_index,
            [int(value) for value in scene_layout["target_box_xyxy"]],
        )
    else:
        scene_structure = None
    if mode == "scene_only":
        silhouette = {
            "source_type": "none_scene_only_control",
            "asset_id": None,
            "mask_path": None,
            "mask_sha256": None,
            "rotation_degrees": None,
            "scene_layout": scene_name,
            "rendered_box_xyxy": None,
            "support_polygon": support.tolist(),
            "scene_structure": scene_structure,
        }
    elif mode in ("silhouette", "target_only"):
        if row is None:
            raise ValueError("Silhouette conditioning requires an object-mask row")
        mask_path = repo_path(row["mask_path"])
        original = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if original is None:
            raise ValueError(f"Unreadable mask: {mask_path}")
        ys, xs = np.where(original > 0)
        if xs.size == 0:
            raise ValueError(f"Empty mask: {mask_path}")
        crop_bounds = (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)
        binary = np.where(original[crop_bounds[0]:crop_bounds[1], crop_bounds[2]:crop_bounds[3]] > 0, 255, 0).astype(np.uint8)
        class_rotations = config.get("class_rotation_degrees", {}).get(int(row["class_id"]))
        rotations = class_rotations if class_rotations is not None else (-24, -8, 8, 24)
        angle = float(rotations[int(variation_rng.integers(0, len(rotations)))])
        binary = rotate_binary(binary, angle)
        x1, y1, x2, y2 = [int(value) for value in scene_layout["target_box_xyxy"]]
        box_width, box_height = x2 - x1, y2 - y1
        scale_range = layout.get("target_scale_factor")
        factor = float(variation_rng.uniform(*scale_range)) if scale_range else 0.82 + 0.06 * (sample_index % 3)
        scale = min(box_width / binary.shape[1], box_height / binary.shape[0]) * factor
        target_width, target_height = max(1, round(binary.shape[1] * scale)), max(1, round(binary.shape[0] * scale))
        binary = cv2.resize(binary, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        center_jitter = layout.get("target_center_jitter_px")
        if center_jitter:
            offset_x = int(variation_rng.integers(-int(center_jitter[0]), int(center_jitter[0]) + 1))
            offset_y = int(variation_rng.integers(-int(center_jitter[1]), int(center_jitter[1]) + 1))
        else:
            offset_x = (-28, 0, 28)[sample_index % 3]
            offset_y = (-16, 0, 16)[(sample_index // 3) % 3]
        paste_x = max(0, min(width - target_width, x1 + (box_width - target_width) // 2 + offset_x))
        paste_y = max(0, min(height - target_height, y1 + (box_height - target_height) // 2 + offset_y))
        proxy[paste_y:paste_y + target_height, paste_x:paste_x + target_width][binary > 0] = int(layout["target_value"])
        rendered_box = [paste_x, paste_y, paste_x + target_width, paste_y + target_height]
        draw_semantic_overlay(proxy, int(row["class_id"]), sample_index, rendered_box, config)
        internal_edges = None
        internal_metadata = None
        internal_classes = {int(value) for value in config.get("internal_edge_control", {}).get("class_ids", [])}
        if int(row["class_id"]) in internal_classes:
            internal_edges, internal_metadata = internal_object_edges(
                row, original, crop_bounds, angle, (target_width, target_height), config
            )
        silhouette = {
            "source_type": (
                "real_class_sam3_binary_mask_target_only"
                if mode == "target_only"
                else "real_class_sam3_binary_mask"
            ),
            "asset_id": row["asset_id"],
            "mask_path": str(mask_path.relative_to(REPO_ROOT)),
            "mask_sha256": sha256(mask_path),
            "rotation_degrees": angle,
            "scene_layout": scene_name,
            "rendered_box_xyxy": rendered_box,
            "support_polygon": support.tolist(),
            "scene_structure": scene_structure,
            "target_scale_factor": round(factor, 6),
            "target_center_offset_xy": [offset_x, offset_y],
            "internal_edges": internal_metadata,
        }
    else:
        raise ValueError(f"Unsupported target_conditioning_mode: {mode}")
    blurred = cv2.GaussianBlur(proxy, (0, 0), sigmaX=float(layout["pre_canny_blur_sigma"]))
    low, high = [int(value) for value in layout["canny_thresholds"]]
    canny = cv2.Canny(blurred, low, high)
    if mode in ("silhouette", "target_only") and internal_edges is not None:
        canny[paste_y:paste_y + target_height, paste_x:paste_x + target_width] = cv2.bitwise_or(
            canny[paste_y:paste_y + target_height, paste_x:paste_x + target_width], internal_edges
        )
    return Image.fromarray(cv2.cvtColor(proxy, cv2.COLOR_GRAY2RGB)), Image.fromarray(cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)), silhouette


def scene_prompt_context(
    config: dict[str, Any], scene_name: str, scene: dict[str, Any], sample_index: int
) -> dict[str, str]:
    """Select a deterministic, scene-compatible background description."""
    base = str(scene["description"]).rstrip(".")
    profiles = config.get("scene_prompt_profiles", {}).get(scene_name, [])
    if not profiles:
        return {"base": base, "profile": "", "description": base}
    profile = str(profiles[sample_index % len(profiles)]).rstrip(".")
    max_words = config.get("scene_prompt_profile_max_words")
    if max_words is not None:
        profile = " ".join(profile.split()[:int(max_words)])
    description = f"{base}. {profile}" if config.get("scene_prompt_include_base", True) else profile
    return {"base": base, "profile": profile, "description": description}


def prompt_for(config: dict[str, Any], scene: dict[str, Any], target: str, class_id: int, sample_index: int) -> str:
    description = str(scene["description"]).rstrip(".")
    phrase = config.get("class_target_phrases", {}).get(class_id, target.lower())
    if isinstance(phrase, list):
        phrase = phrase[sample_index % len(phrase)]
    class_templates = config.get("class_prompt_templates", {})
    if class_id in class_templates:
        return str(class_templates[class_id]).format(target=phrase, scene=description)
    variation = config.get("prompt_variation", {})
    details = []
    for offset, key in enumerate(("lighting", "camera", "material", "clutter")):
        choices = variation.get(key, [])
        if choices:
            details.append(str(choices[(sample_index + offset * 3) % len(choices)]))
    suffix = config['prompt']['suffix'].format(target=phrase)
    extra = f" {'; '.join(details)}." if details else ""
    if config.get("prompt_order", "scene_first") == "target_first":
        target_first = str(config["prompt"].get("target_first_template", "{target}")).format(
            target=phrase
        )
        return f"{target_first}. {description}. {suffix} {config['prompt']['prefix']}{extra}"
    return f"{config['prompt']['prefix']} {description}.{extra} {suffix}"


def negative_prompt_for(
    config: dict[str, Any], class_id: int, scene_name: str | None = None
) -> str:
    base = str(config["prompt"]["negative"]).rstrip().rstrip(",")
    additions = config.get("class_negative_additions", {}).get(class_id, [])
    scene_additions = (
        config.get("scene_negative_additions", {}).get(scene_name, [])
        if scene_name is not None
        else []
    )
    return ", ".join(
        [base, *[str(value) for value in additions], *[str(value) for value in scene_additions]]
    )


def main() -> None:
    raise RuntimeError(
        "This legacy pilot entry point is retired. Run "
        "src/generation/generate_sdxl_dataset.py instead."
    )
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
    repeats = int(config.get("pilot_scope", {}).get("images_per_class_scene", 1))
    if repeats < 1:
        raise ValueError("images_per_class_scene must be positive")
    for class_id, target in scene_policy["class_policy"]["names"].items():
        if selected_ids is not None and int(class_id) not in selected_ids:
            continue
        for repeat_index in range(repeats):
            for scene_name in scene_policy["class_to_scene_families"][class_id]:
                schedule.append({
                    "class_id": int(class_id), "target": target,
                    "scene_name": scene_name, "repeat_index": repeat_index,
                })
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
        if config.get("target_conditioning_mode", "silhouette") == "scene_only":
            row = None
        else:
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
        scale_schedule = config.get("class_controlnet_conditioning_scales", {}).get(sample["class_id"])
        if scale_schedule is not None:
            class_scale = scale_schedule[sample["repeat_index"] % len(scale_schedule)]
        width, height = [int(value) for value in config["output_size"]]
        common = {"prompt": prompt, "negative_prompt": negative_prompt_for(config, sample["class_id"]), "height": height, "width": width, "num_inference_steps": int(config["inference_steps"]), "controlnet_conditioning_scale": float(class_scale), "generator": generator}
        if args.backend == "sdxl":
            common["control_guidance_start"] = float(config.get("control_guidance_start", 0.0))
            common["control_guidance_end"] = float(config.get("control_guidance_end", 1.0))
        started = time.monotonic()
        result = pipe(image=control, guidance_scale=float(config.get("guidance_scale", models["sdxl"]["guidance_scale"])), **common) if args.backend == "sdxl" else pipe(control_image=control, true_cfg_scale=float(models["qwen"]["true_cfg_scale"]), **common)
        torch.cuda.synchronize(args.gpu)
        image_name = f"{index:03d}_c{sample['class_id']:02d}_{sample['scene_name']}.png"
        output_path = output_dir / "images" / image_name
        proxy_path = output_dir / "controls" / f"{index:03d}_proxy.png"
        control_path = output_dir / "controls" / f"{index:03d}_canny.png"
        result.images[0].save(output_path)
        proxy.save(proxy_path)
        control.save(control_path)
        records.append({**sample, "index": index, "seed": int(config["seed"]) + index, "prompt": prompt, "negative_prompt": common["negative_prompt"], "controlnet_conditioning_scale": float(class_scale), "control_guidance_start": common.get("control_guidance_start"), "control_guidance_end": common.get("control_guidance_end"), "output": str(output_path.relative_to(REPO_ROOT)), "output_sha256": sha256(output_path), "silhouette": silhouette, "proxy_sha256": sha256(proxy_path), "control_sha256": sha256(control_path), "inference_seconds": round(time.monotonic() - started, 3), "annotation_performed": False, "degradation_applied": False, "training_use_forbidden": True})
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
