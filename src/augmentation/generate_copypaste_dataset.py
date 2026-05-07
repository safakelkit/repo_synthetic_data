from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import shutil
import cv2
import numpy as np
from tqdm import tqdm


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def collect_backgrounds(background_root: str | Path) -> list[Path]:
    root = Path(background_root)
    paths: list[Path] = []
    for ext in IMAGE_EXTS:
        paths.extend(root.rglob(f"*{ext}"))
    return sorted(paths)


def collect_object_bank(object_bank_root: str | Path) -> dict[str, list[Path]]:
    root = Path(object_bank_root)
    bank: dict[str, list[Path]] = {}

    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue

        class_id = class_dir.name.split("_")[0]
        rgba_dir = class_dir / "rgba"

        if not rgba_dir.exists():
            continue

        crops = sorted(rgba_dir.glob("*.png"))
        if crops:
            bank[class_id] = crops

    return bank


def sample_class_uniform(object_bank: dict[str, list[Path]]) -> str:
    return random.choice(list(object_bank.keys()))


def read_rgba(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim != 3 or img.shape[2] != 4:
        return None
    return img


def is_valid_rgba_object(rgba: np.ndarray) -> bool:
    alpha = rgba[:, :, 3]
    visible = alpha > 20

    visible_area = int(visible.sum())
    total_area = alpha.shape[0] * alpha.shape[1]

    if total_area <= 0:
        return False

    area_ratio = visible_area / total_area
    h, w = alpha.shape[:2]

    if visible_area < 40:
        return False

    if h < 8 or w < 8:
        return False

    if area_ratio < 0.04:
        return False

    if area_ratio > 0.95:
        return False

    return True


def sample_scale(class_id: str) -> float:
    cid = int(class_id)

    # Thin / small objects can be slightly larger.
    if cid in [1, 2, 4, 9, 10, 12]:
        return random.uniform(0.025, 0.080)

    # Naturally larger objects.
    if cid in [14, 15]:
        return random.uniform(0.020, 0.055)

    return random.uniform(0.025, 0.070)


def resize_rgba(rgba: np.ndarray, scale: float) -> np.ndarray:
    h, w = rgba.shape[:2]
    new_w = max(4, int(w * scale))
    new_h = max(4, int(h * scale))
    return cv2.resize(rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)


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
) -> tuple[int, int] | None:
    """
    Controlled random placement.

    Instead of placing objects anywhere, this samples from likely lower/middle
    surface regions. It is still not semantic segmentation, but it greatly
    reduces ceiling/wall/floating placements.
    """
    zones = [
        # lower central surface / floor / bed / table region
        (0.15, 0.72, 0.85, 0.93),

        # middle-lower area, useful for bed/table surfaces
        (0.25, 0.62, 0.75, 0.84),

        # slightly wider lower band
        (0.10, 0.78, 0.90, 0.94),
    ]

    for _ in range(50):
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
    output_root: str = "data/synthetic/copypaste_v1_50",
    num_images: int = 50,
    seed: int = 42,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    output_root = Path(output_root)
    images_out = output_root / "images"
    labels_out = output_root / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    backgrounds = collect_backgrounds(background_root)
    object_bank = collect_object_bank(object_bank_root)

    if not backgrounds:
        raise ValueError(f"No backgrounds found in {background_root}")

    if not object_bank:
        raise ValueError(f"No object crops found in {object_bank_root}")

    print(f"Backgrounds found: {len(backgrounds)}")
    print(f"Object classes found: {len(object_bank)}")

    metadata: list[dict[str, Any]] = []

    saved_count = 0
    attempt_count = 0
    max_attempts = num_images * 50

    progress = tqdm(total=num_images, desc="Generating copy-paste dataset")

    while saved_count < num_images and attempt_count < max_attempts:
        attempt_count += 1

        bg_path = random.choice(backgrounds)
        bg = cv2.imread(str(bg_path), cv2.IMREAD_COLOR)

        if bg is None:
            continue

        bg_h, bg_w = bg.shape[:2]

        class_id = sample_class_uniform(object_bank)
        obj_path = random.choice(object_bank[class_id])
        rgba = read_rgba(obj_path)

        if rgba is None:
            continue

        if not is_valid_rgba_object(rgba):
            continue

        scale = sample_scale(class_id)
        rgba = resize_rgba(rgba, scale)

        if not is_valid_rgba_object(rgba):
            continue

        obj_h, obj_w = rgba.shape[:2]

        if obj_w >= bg_w or obj_h >= bg_h:
            continue

        position = sample_surface_position(
            bg_w=bg_w,
            bg_h=bg_h,
            obj_w=obj_w,
            obj_h=obj_h,
        )

        if position is None:
            continue

        x, y = position

        bg_pasted, final_bbox = paste_rgba(bg, rgba, x, y)

        if bbox_area(final_bbox) <= 0:
            continue

        x_c, y_c, w, h = bbox_to_yolo(final_bbox, bg_w, bg_h)

        label_line = f"{int(class_id)} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"

        image_name = f"copypaste_{saved_count:06d}.jpg"
        label_name = f"copypaste_{saved_count:06d}.txt"

        cv2.imwrite(str(images_out / image_name), bg_pasted)

        with open(labels_out / label_name, "w", encoding="utf-8") as f:
            f.write(label_line + "\n")

        metadata.append(
            {
                "image": str(images_out / image_name),
                "label": str(labels_out / label_name),
                "background": str(bg_path),
                "object": {
                    "class_id": int(class_id),
                    "object_path": str(obj_path),
                    "bbox_xyxy": list(final_bbox),
                    "scale": scale,
                },
            }
        )

        saved_count += 1
        progress.update(1)

    progress.close()

    with open(output_root / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Saved {saved_count}/{num_images} images after {attempt_count} attempts.")
    print(f"Output: {output_root}")

    if saved_count < num_images:
        print(
            "Warning: Could not generate the requested number of images. "
            "Consider increasing max_attempts, relaxing object filters, or checking object/background folders."
        )

def create_subsets(full_root: Path, base_root: Path) -> None:
    splits = [500, 1000, 1500, 2215]

    images = sorted((full_root / "images").glob("*.jpg"))
    labels = sorted((full_root / "labels").glob("*.txt"))

    for n in splits:
        dst = base_root / f"copypaste_v2_{n}"
        images_out = dst / "images"
        labels_out = dst / "labels"

        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        for i in range(n):
            shutil.copy(images[i], images_out / images[i].name)
            shutil.copy(labels[i], labels_out / labels[i].name)

        print(f"Created subset: {dst}")

def main() -> None:
    FULL_OUTPUT = "data/synthetic/copypaste_v3_full_2215"

    # 1. Generate FULL dataset
    generate_dataset(
        object_bank_root="data/processed/object_bank_sam3",
        background_root="data/backgrounds/places365_subset",
        output_root=FULL_OUTPUT,
        num_images=2215,
        seed=42, #43
    )

    # 2. Create subsets
    create_subsets(
        full_root=Path(FULL_OUTPUT),
        base_root=Path("data/synthetic"),
    )

if __name__ == "__main__":
    main()