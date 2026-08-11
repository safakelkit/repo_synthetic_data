from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import random
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from tqdm import tqdm


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/generation/copy_paste_v1.yaml"
DEFAULT_SIZE_TEMPLATES = REPO_ROOT / "data/processed/object_size_analysis/train_box_templates.csv"


def repo_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def repository_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cached_file_sha256(path: Path, cache: dict[Path, str]) -> str:
    if path not in cache:
        cache[path] = file_sha256(path)
    return cache[path]


def code_revision() -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": revision, "working_tree_dirty": dirty}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "working_tree_dirty": None}


def software_versions() -> dict[str, str]:
    packages = ("numpy", "opencv-python", "PyYAML", "tqdm", "ultralytics")
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def collect_backgrounds(background_root: str | Path) -> list[Path]:
    root = repo_path(background_root)
    paths: list[Path] = []
    for ext in IMAGE_EXTS:
        paths.extend(root.rglob(f"*{ext}"))
    return sorted(paths)


def collect_object_bank(object_bank_root: str | Path) -> dict[int, list[Path]]:
    root = repo_path(object_bank_root)
    bank: dict[int, list[Path]] = {}

    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir() or not class_dir.name[:2].isdigit():
            continue

        class_id = int(class_dir.name.split("_")[0])
        rgba_dir = class_dir / "rgba"

        if not rgba_dir.exists():
            continue

        crops = sorted(rgba_dir.glob("*.png"))
        if crops:
            bank[class_id] = crops

    return bank


def validate_object_bank_against_audit(
    object_bank: dict[int, list[Path]],
    audit_manifest_path: str | Path,
) -> None:
    manifest_path = repo_path(audit_manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Required object-bank audit manifest not found: {manifest_path}"
        )
    with open(manifest_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    accepted = {
        repository_relative(repo_path(row["rgba_path"]))
        for row in rows
        if row["status"] == "accepted"
    }
    bank_paths = {
        repository_relative(path)
        for class_paths in object_bank.values()
        for path in class_paths
    }
    if bank_paths != accepted:
        raise ValueError(
            "Object bank and accepted audit manifest differ: "
            f"bank_only={len(bank_paths - accepted)}, "
            f"manifest_only={len(accepted - bank_paths)}"
        )


def build_balanced_asset_schedule(
    assets: list[Path],
    required: int,
    rng: random.Random,
) -> list[Path]:
    """Cycle through every class asset before reusing one."""
    if not assets:
        raise ValueError("At least one object asset is required")
    schedule: list[Path] = []
    while len(schedule) < required:
        cycle = assets.copy()
        rng.shuffle(cycle)
        schedule.extend(cycle)
    return schedule[:required]


def build_balanced_class_schedule(
    class_ids: list[int],
    num_images: int,
    seed: int,
) -> list[int]:
    """Return shuffled class blocks with one sample per class in every block."""
    class_ids = sorted(class_ids)
    if not class_ids:
        raise ValueError("At least one class is required")
    if num_images % len(class_ids) != 0:
        raise ValueError(
            f"num_images={num_images} must be divisible by {len(class_ids)} "
            "for exact class balance"
        )

    schedule_rng = random.Random(seed)
    schedule: list[int] = []
    for _ in range(num_images // len(class_ids)):
        block = class_ids.copy()
        schedule_rng.shuffle(block)
        schedule.extend(block)
    return schedule


def read_rgba(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim != 3 or img.shape[2] != 4:
        return None
    return img


def is_valid_rgba_object(
    rgba: np.ndarray,
    alpha_threshold: int = 20,
    minimum_visible_pixels: int = 40,
    minimum_dimension_pixels: int = 8,
    minimum_visible_ratio: float = 0.04,
) -> bool:
    alpha = rgba[:, :, 3]
    visible = alpha > alpha_threshold

    visible_area = int(visible.sum())
    total_area = alpha.shape[0] * alpha.shape[1]

    if total_area <= 0:
        return False

    area_ratio = visible_area / total_area
    h, w = alpha.shape[:2]

    if visible_area < minimum_visible_pixels:
        return False

    if h < minimum_dimension_pixels or w < minimum_dimension_pixels:
        return False

    if area_ratio < minimum_visible_ratio:
        return False

    return True


def trim_transparent_padding(
    rgba: np.ndarray,
    alpha_threshold: int = 20,
) -> np.ndarray | None:
    visible = rgba[:, :, 3] > alpha_threshold
    ys, xs = np.nonzero(visible)
    if len(xs) == 0:
        return None
    return rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def visible_bbox(
    rgba: np.ndarray,
    alpha_threshold: int = 20,
) -> tuple[int, int, int, int] | None:
    visible = rgba[:, :, 3] > alpha_threshold
    ys, xs = np.nonzero(visible)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def load_size_templates(
    templates_path: str | Path,
    lower_quantile: float = 0.10,
    upper_quantile: float = 0.90,
) -> dict[int, list[dict[str, Any]]]:
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("Size quantiles must satisfy 0 <= lower < upper <= 1")

    with open(templates_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        class_id = int(row["class_id"])
        grouped.setdefault(class_id, []).append(
            {
                "source_label": row["source_label"],
                "annotation_index": int(row["annotation_index"]),
                "area": float(row["area"]),
            }
        )

    filtered: dict[int, list[dict[str, Any]]] = {}
    for class_id, class_rows in grouped.items():
        areas = np.asarray([row["area"] for row in class_rows], dtype=np.float64)
        lower = float(np.quantile(areas, lower_quantile))
        upper = float(np.quantile(areas, upper_quantile))
        filtered[class_id] = [row for row in class_rows if lower <= row["area"] <= upper]
        if not filtered[class_id]:
            raise ValueError(f"No usable size templates for class {class_id}")
    return filtered


def resize_rgba_to_target_area(
    rgba: np.ndarray,
    target_area: float,
    background_width: int,
    background_height: int,
    maximum_dimension_ratio: float = 0.90,
    alpha_threshold: int = 20,
) -> tuple[np.ndarray, float] | None:
    rgba = trim_transparent_padding(rgba, alpha_threshold=alpha_threshold)
    if rgba is None:
        return None

    height, width = rgba.shape[:2]
    target_pixels = target_area * background_width * background_height
    scale = float(np.sqrt(target_pixels / max(1, width * height)))
    new_width = max(4, int(round(width * scale)))
    new_height = max(4, int(round(height * scale)))

    # Permit naturally large classes such as laptops while preventing a crop
    # from covering essentially the complete background.
    if (
        new_width > maximum_dimension_ratio * background_width
        or new_height > maximum_dimension_ratio * background_height
    ):
        return None

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(rgba, (new_width, new_height), interpolation=interpolation)
    resized_bbox = visible_bbox(resized, alpha_threshold=alpha_threshold)
    if resized_bbox is None:
        return None
    realized_area = bbox_area(resized_bbox) / (background_width * background_height)
    return resized, realized_area


def paste_rgba(
    background_bgr: np.ndarray,
    rgba: np.ndarray,
    x: int,
    y: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    bg = background_bgr.copy()
    h, w = rgba.shape[:2]
    bg_h, bg_w = bg.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bg_w, x + w)
    y2 = min(bg_h, y + h)

    if x2 <= x1 or y2 <= y1:
        return bg, (0, 0, 0, 0)

    crop_x1 = x1 - x
    crop_y1 = y1 - y
    crop_x2 = crop_x1 + (x2 - x1)
    crop_y2 = crop_y1 + (y2 - y1)

    obj = rgba[crop_y1:crop_y2, crop_x1:crop_x2, :3]
    alpha = rgba[crop_y1:crop_y2, crop_x1:crop_x2, 3] / 255.0
    alpha = alpha[..., None]

    roi = bg[y1:y2, x1:x2]
    blended = (alpha * obj + (1.0 - alpha) * roi).astype(np.uint8)
    bg[y1:y2, x1:x2] = blended

    return bg, (x1, y1, x2, y2)


def bbox_area(bbox: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_to_yolo(
    bbox: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1

    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0

    return xc / img_w, yc / img_h, bw / img_w, bh / img_h


def sample_surface_position(
    bg_w: int,
    bg_h: int,
    obj_w: int,
    obj_h: int,
    zones: list[tuple[float, float, float, float]] | None = None,
    max_attempts: int = 50,
) -> tuple[int, int] | None:
    """
    Controlled random placement.

    Instead of placing objects anywhere, this samples from likely lower/middle
    surface regions. It is still not semantic segmentation, but it greatly
    reduces ceiling/wall/floating placements.
    """
    zones = zones or [
        # lower central surface / floor / bed / table region
        (0.15, 0.72, 0.85, 0.93),

        # middle-lower area, useful for bed/table surfaces
        (0.25, 0.62, 0.75, 0.84),

        # slightly wider lower band
        (0.10, 0.78, 0.90, 0.94),
    ]

    for _ in range(max_attempts):
        x1r, y1r, x2r, y2r = random.choice(zones)

        x_min = int(x1r * bg_w)
        x_max = int(x2r * bg_w) - obj_w

        bottom_min = int(y1r * bg_h)
        bottom_max = int(y2r * bg_h)

        if x_max <= x_min:
            continue

        if bottom_max <= bottom_min:
            continue

        object_bottom_y = random.randint(bottom_min, bottom_max)
        y = object_bottom_y - obj_h
        x = random.randint(x_min, x_max)

        if y < 0 or y + obj_h > bg_h:
            continue

        if x < 0 or x + obj_w > bg_w:
            continue

        return x, y

    return None


def generate_dataset(
    object_bank_root: str = "data/processed/object_bank_sam3",
    background_root: str = "data/backgrounds/places365_subset",
    output_root: str = "data/synthetic/cp_v1_seed42",
    num_images: int = 2048,
    seed: int = 42,
    size_templates_path: str | Path = DEFAULT_SIZE_TEMPLATES,
    generator_version: str = "cp_v1",
    source_config_path: str | Path | None = None,
    object_audit_manifest_path: str | Path | None = None,
    lower_size_quantile: float = 0.10,
    upper_size_quantile: float = 0.90,
    maximum_object_dimension_ratio: float = 0.90,
    placement_zones: list[tuple[float, float, float, float]] | None = None,
    placement_attempts: int = 50,
    generation_attempts_per_image: int = 50,
    asset_schedule_seed: int | None = None,
    alpha_threshold: int = 20,
    minimum_visible_pixels: int = 40,
    minimum_crop_dimension_pixels: int = 8,
    minimum_visible_crop_ratio: float = 0.04,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    object_bank_root = repo_path(object_bank_root)
    background_root = repo_path(background_root)
    size_templates_path = repo_path(size_templates_path)
    output_root = repo_path(output_root)
    images_out = output_root / "images"
    labels_out = output_root / "labels"

    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output dataset already contains files: {output_root}. "
            "Choose a new versioned output path."
        )

    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    backgrounds = collect_backgrounds(background_root)
    object_bank = collect_object_bank(object_bank_root)
    size_templates = load_size_templates(
        size_templates_path,
        lower_quantile=lower_size_quantile,
        upper_quantile=upper_size_quantile,
    )

    if not backgrounds:
        raise ValueError(f"No backgrounds found in {background_root}")

    if not object_bank:
        raise ValueError(f"No object crops found in {object_bank_root}")
    if object_audit_manifest_path is None:
        raise ValueError("An accepted object-bank audit manifest is required")
    validate_object_bank_against_audit(object_bank, object_audit_manifest_path)

    print(f"Backgrounds found: {len(backgrounds)}")
    expected_class_ids = list(range(16))
    if sorted(object_bank) != expected_class_ids:
        raise ValueError(
            "Object bank must contain exactly class IDs 0 through 15; "
            f"found {sorted(object_bank)}"
        )

    class_schedule = build_balanced_class_schedule(
        class_ids=expected_class_ids,
        num_images=num_images,
        seed=seed,
    )
    images_per_class = num_images // len(expected_class_ids)
    if asset_schedule_seed is None:
        asset_schedule_seed = seed + 1
    asset_rng = random.Random(asset_schedule_seed)
    asset_schedules = {
        class_id: build_balanced_asset_schedule(
            object_bank[class_id], images_per_class, asset_rng
        )
        for class_id in expected_class_ids
    }
    saved_per_class = {class_id: 0 for class_id in expected_class_ids}

    print(f"Object classes found: {len(object_bank)}")

    metadata: list[dict[str, Any]] = []
    seen_image_hashes: set[str] = set()
    source_hash_cache: dict[Path, str] = {}

    saved_count = 0
    attempt_count = 0
    max_attempts = num_images * generation_attempts_per_image

    progress = tqdm(total=num_images, desc="Generating copy-paste dataset")

    while saved_count < num_images and attempt_count < max_attempts:
        attempt_count += 1

        bg_path = random.choice(backgrounds)
        bg = cv2.imread(str(bg_path), cv2.IMREAD_COLOR)

        if bg is None:
            continue

        bg_h, bg_w = bg.shape[:2]

        # Retries retain the scheduled class, guaranteeing exact quotas.
        class_id = class_schedule[saved_count]
        class_sequence_index = saved_per_class[class_id]
        obj_path = asset_schedules[class_id][class_sequence_index]
        rgba = read_rgba(obj_path)

        if rgba is None:
            continue

        if not is_valid_rgba_object(
            rgba,
            alpha_threshold=alpha_threshold,
            minimum_visible_pixels=minimum_visible_pixels,
            minimum_dimension_pixels=minimum_crop_dimension_pixels,
            minimum_visible_ratio=minimum_visible_crop_ratio,
        ):
            continue

        size_template = random.choice(size_templates[class_id])
        resized_result = resize_rgba_to_target_area(
            rgba=rgba,
            target_area=size_template["area"],
            background_width=bg_w,
            background_height=bg_h,
            maximum_dimension_ratio=maximum_object_dimension_ratio,
            alpha_threshold=alpha_threshold,
        )
        if resized_result is None:
            continue
        rgba, realized_area = resized_result

        if not is_valid_rgba_object(
            rgba,
            alpha_threshold=alpha_threshold,
            minimum_visible_pixels=minimum_visible_pixels,
            minimum_dimension_pixels=minimum_crop_dimension_pixels,
            minimum_visible_ratio=minimum_visible_crop_ratio,
        ):
            continue

        obj_h, obj_w = rgba.shape[:2]

        if obj_w >= bg_w or obj_h >= bg_h:
            continue

        position = sample_surface_position(
            bg_w=bg_w,
            bg_h=bg_h,
            obj_w=obj_w,
            obj_h=obj_h,
            zones=placement_zones,
            max_attempts=placement_attempts,
        )

        if position is None:
            continue

        x, y = position

        bg_pasted, pasted_crop_bbox = paste_rgba(bg, rgba, x, y)

        local_visible_bbox = visible_bbox(rgba, alpha_threshold=alpha_threshold)
        if local_visible_bbox is None:
            continue
        vx1, vy1, vx2, vy2 = local_visible_bbox
        final_bbox = (x + vx1, y + vy1, x + vx2, y + vy2)

        if bbox_area(pasted_crop_bbox) <= 0 or bbox_area(final_bbox) <= 0:
            continue

        x_c, y_c, w, h = bbox_to_yolo(final_bbox, bg_w, bg_h)
        if not (
            0.0 <= x_c <= 1.0
            and 0.0 <= y_c <= 1.0
            and 0.0 < w <= 1.0
            and 0.0 < h <= 1.0
        ):
            continue

        label_line = f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"

        image_name = f"copypaste_{saved_count:06d}.jpg"
        label_name = f"copypaste_{saved_count:06d}.txt"

        image_path = images_out / image_name
        if not cv2.imwrite(str(image_path), bg_pasted):
            continue

        image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if image_sha256 in seen_image_hashes:
            image_path.unlink()
            continue
        seen_image_hashes.add(image_sha256)

        with open(labels_out / label_name, "w", encoding="utf-8") as f:
            f.write(label_line + "\n")

        metadata.append(
            {
                "image_id": saved_count,
                "image": repository_relative(image_path),
                "label": repository_relative(labels_out / label_name),
                "image_sha256": image_sha256,
                "generator_version": generator_version,
                "generator_seed": seed,
                "primary_class_id": class_id,
                "background": repository_relative(bg_path),
                "background_sha256": cached_file_sha256(bg_path, source_hash_cache),
                "background_width": bg_w,
                "background_height": bg_h,
                "placement_xy": [x, y],
                "degradations": [],
                "qc_status": "automated_checks_passed_pending_dataset_review",
                "object": {
                    "class_id": class_id,
                    "object_path": repository_relative(obj_path),
                    "object_sha256": cached_file_sha256(obj_path, source_hash_cache),
                    "class_sequence_index": class_sequence_index,
                    "bbox_xyxy": list(final_bbox),
                    "size_template": size_template,
                    "target_normalized_area": size_template["area"],
                    "realized_normalized_area": realized_area,
                },
            }
        )

        saved_count += 1
        saved_per_class[class_id] += 1
        progress.update(1)

    progress.close()

    with open(output_root / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    generation_config = {
        "format_version": 2,
        "method": "copy_paste",
        "generator_version": generator_version,
        "seed": seed,
        "requested_images": num_images,
        "saved_images": saved_count,
        "dataset_role": "canonical_production_candidate",
        "code_revision": code_revision(),
        "software_versions": software_versions(),
        "object_bank_root": repository_relative(object_bank_root),
        "object_audit_manifest": repository_relative(repo_path(object_audit_manifest_path)),
        "object_audit_manifest_sha256": file_sha256(repo_path(object_audit_manifest_path)),
        "object_assets": sum(len(paths) for paths in object_bank.values()),
        "object_asset_sampling": "deterministic_shuffled_cycles_without_reuse_until_exhausted",
        "object_asset_schedule_seed": asset_schedule_seed,
        "background_root": repository_relative(background_root),
        "background_images": len(backgrounds),
        "size_templates_path": repository_relative(size_templates_path),
        "size_templates_sha256": file_sha256(size_templates_path),
        "size_sampling": "observed_INSP-DET_train_area_within_class_quantile_range",
        "size_lower_quantile": lower_size_quantile,
        "size_upper_quantile": upper_size_quantile,
        "maximum_object_dimension_ratio": maximum_object_dimension_ratio,
        "alpha_visibility_threshold": alpha_threshold,
        "minimum_visible_pixels": minimum_visible_pixels,
        "minimum_crop_dimension_pixels": minimum_crop_dimension_pixels,
        "minimum_visible_crop_ratio": minimum_visible_crop_ratio,
        "class_sampling": "exact_balanced_shuffled_blocks",
        "class_ids": expected_class_ids,
        "images_per_class": num_images // len(expected_class_ids),
        "class_block_size": len(expected_class_ids),
        "objects_per_image": 1,
        "placement": "predefined_lower_surface_zones",
        "placement_zones_xyxy_normalized": placement_zones,
        "placement_attempts": placement_attempts,
        "generation_attempts_per_image": generation_attempts_per_image,
        "degradations": [],
        "release_status": "generated_candidate_pending_dataset_qc",
    }
    if source_config_path is not None:
        resolved_config = repo_path(source_config_path)
        generation_config["source_config"] = repository_relative(resolved_config)
        generation_config["source_config_sha256"] = file_sha256(resolved_config)
    with open(output_root / "generation_config.json", "w", encoding="utf-8") as f:
        json.dump(generation_config, f, indent=2, ensure_ascii=False)

    print(f"Saved {saved_count}/{num_images} images after {attempt_count} attempts.")
    print(f"Output: {output_root}")

    if saved_count != num_images:
        raise RuntimeError(
            f"Generation incomplete: saved {saved_count}/{num_images}. "
            "The partial output is not a valid dataset and must not be used for training."
        )

def create_subset_manifests(full_root: Path, manifest_dir: Path) -> None:
    """Create YOLO image lists without duplicating generated image files."""
    splits = [512, 1024, 1536, 2048]

    full_root = repo_path(full_root)
    manifest_dir = repo_path(manifest_dir)
    images = sorted((full_root / "images").glob("*.jpg"))
    if len(images) != max(splits):
        raise ValueError(f"Expected exactly 2048 images, found {len(images)}")

    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_records: list[dict[str, Any]] = []
    for n in splits:
        missing_labels = [
            image for image in images[:n]
            if not (full_root / "labels" / f"{image.stem}.txt").is_file()
        ]
        if missing_labels:
            raise ValueError(
                f"Missing labels for {len(missing_labels)} images; "
                f"first missing label: {missing_labels[0].stem}.txt"
            )

        manifest_path = manifest_dir / f"CP-B{n:04d}.txt"
        with open(manifest_path, "w", encoding="utf-8") as f:
            for image in images[:n]:
                # Ultralytics resolves paths beginning with ./ relative to the
                # manifest, so the dataset remains portable with the repo.
                f.write(f"./../images/{image.name}\n")

        manifest_records.append(
            {
                "experiment_id": f"CP-B{n:04d}",
                "images": n,
                "manifest": repository_relative(manifest_path),
                "sha256": file_sha256(manifest_path),
            }
        )

        print(f"Created subset manifest: {manifest_path}")

    with open(manifest_dir / "manifest_checksums.json", "w", encoding="utf-8") as f:
        json.dump(manifest_records, f, indent=2, ensure_ascii=False)


def load_generation_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid generation config: {config_path}")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the canonical cut-paste dataset")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config_path = repo_path(args.config)
    config = load_generation_config(config_path)
    production_config = config["production"]

    if not config["release_gate"]["approved"]:
        raise RuntimeError(
            "Production generation is blocked: placement, degradation, and QC "
            "values must be frozen and release_gate.approved set to true."
        )
    required_statuses = {
        "placement.status": (config["placement"]["status"], "approved_before_generation"),
        "degradation.status": (config["degradation"]["status"], "frozen_implemented"),
        "quality_control.status": (
            config["quality_control"]["status"],
            "frozen_implemented",
        ),
        "inputs.background_target_class_audit_status": (
            config["inputs"]["background_target_class_audit_status"],
            "complete",
        ),
    }
    unresolved = [
        f"{name}={actual!r} (expected {expected!r})"
        for name, (actual, expected) in required_statuses.items()
        if actual != expected
    ]
    if unresolved:
        raise RuntimeError("Production generation has unresolved gates: " + "; ".join(unresolved))
    if config["release_gate"].get("require_clean_git_worktree", True):
        revision = code_revision()
        if revision["commit"] is None or revision["working_tree_dirty"]:
            raise RuntimeError(
                "Production generation requires a committed, clean Git worktree "
                "so its exact source revision can be recovered."
            )

    output_root = repo_path(production_config["output_root"])
    placement_zones = [tuple(map(float, zone)) for zone in config["placement"]["zones_xyxy_normalized"]]
    asset_schedule_seed = int(config["seed"]) + int(
        config["allocation"]["object_asset_schedule_seed_offset"]
    )
    generate_dataset(
        object_bank_root=config["inputs"]["object_bank_root"],
        object_audit_manifest_path=config["inputs"]["object_audit_manifest"],
        background_root=config["inputs"]["background_root"],
        output_root=str(output_root),
        num_images=int(production_config["num_images"]),
        seed=int(config["seed"]),
        size_templates_path=config["sizing"]["templates_path"],
        generator_version=str(config["version"]),
        source_config_path=config_path,
        lower_size_quantile=float(config["sizing"]["lower_quantile"]),
        upper_size_quantile=float(config["sizing"]["upper_quantile"]),
        maximum_object_dimension_ratio=float(config["sizing"]["maximum_object_dimension_ratio"]),
        placement_zones=placement_zones,
        placement_attempts=int(config["placement"]["attempts_per_object"]),
        generation_attempts_per_image=int(config["quality_control"]["generation_attempts_per_image"]),
        asset_schedule_seed=asset_schedule_seed,
        alpha_threshold=int(config["sizing"]["alpha_visibility_threshold"]),
        minimum_visible_pixels=int(config["sizing"]["minimum_visible_pixels"]),
        minimum_crop_dimension_pixels=int(config["sizing"]["minimum_crop_dimension_pixels"]),
        minimum_visible_crop_ratio=float(config["sizing"]["minimum_visible_crop_ratio"]),
    )

    create_subset_manifests(
        full_root=output_root,
        manifest_dir=output_root / "manifests",
    )

if __name__ == "__main__":
    main()
