from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()

def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(repo_path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_class_names(data_yaml: str | Path) -> dict[int, str]:
    data = load_yaml(data_yaml)
    names = data["names"]

    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}

    raise ValueError("Unsupported 'names' format in YAML.")


def sanitize_class_name(class_name: str) -> str:
    return class_name.replace(" ", "_").replace("/", "_")


def yolo_to_xyxy(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    x1 = int((x_center - width / 2.0) * img_w)
    y1 = int((y_center - height / 2.0) * img_h)
    x2 = int((x_center + width / 2.0) * img_w)
    y2 = int((y_center + height / 2.0) * img_h)

    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))

    return x1, y1, x2, y2


def find_image_for_label(label_path: Path, images_dir: Path) -> Path | None:
    stem = label_path.stem
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def parse_yolo_label_file(label_path: Path) -> list[dict[str, float]]:
    annotations: list[dict[str, float]] = []

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        class_id, x_center, y_center, width, height = parts
        annotations.append(
            {
                "class_id": int(float(class_id)),
                "x_center": float(x_center),
                "y_center": float(y_center),
                "width": float(width),
                "height": float(height),
            }
        )

    return annotations


def add_padding(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    img_w: int,
    img_h: int,
    pad_ratio: float = 0.08,
) -> tuple[int, int, int, int]:
    bw = x2 - x1
    bh = y2 - y1

    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)

    px1 = max(0, x1 - pad_x)
    py1 = max(0, y1 - pad_y)
    px2 = min(img_w, x2 + pad_x)
    py2 = min(img_h, y2 + pad_y)

    return px1, py1, px2, py2


def make_grabcut_mask(
    crop_bgr: np.ndarray,
    inner_rect_ratio: float = 0.1,
    iterations: int = 5,
) -> np.ndarray:
    h, w = crop_bgr.shape[:2]

    if h < 4 or w < 4:
        return np.ones((h, w), dtype=np.uint8) * 255

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    margin_x = max(1, int(w * inner_rect_ratio))
    margin_y = max(1, int(h * inner_rect_ratio))

    rect_x = margin_x
    rect_y = margin_y
    rect_w = max(1, w - 2 * margin_x)
    rect_h = max(1, h - 2 * margin_y)

    if rect_w <= 1 or rect_h <= 1:
        return np.ones((h, w), dtype=np.uint8) * 255

    rect = (rect_x, rect_y, rect_w, rect_h)

    try:
        cv2.grabCut(
            crop_bgr,
            mask,
            rect,
            bgd_model,
            fgd_model,
            iterations,
            cv2.GC_INIT_WITH_RECT,
        )
    except cv2.error:
        return np.ones((h, w), dtype=np.uint8) * 255

    binary_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype("uint8")

    kernel = np.ones((3, 3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    return binary_mask


def make_rgba_cutout(crop_bgr: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(crop_bgr)
    alpha = binary_mask
    return cv2.merge([b, g, r, alpha])


def mask_bbox(binary_mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return x1, y1, x2, y2


def extract_object_crops_with_masks(
    data_yaml: str = "configs/data_insp.yaml",
    split: str = "train",
    output_dir: str = "data/processed/object_bank",
    min_crop_size: int = 8,
    pad_ratio: float = 0.08,
    grabcut_iters: int = 5,
    min_mask_area_ratio: float = 0.03,
    save_metadata: bool = True,
    image_limit: int | None = None,
) -> dict[str, Any]:
    data = load_yaml(data_yaml)
    class_names = get_class_names(data_yaml)

    data_yaml_path = repo_path(data_yaml)
    dataset_root = repo_path(data["path"]) if data.get("path") else data_yaml_path.parent
    images_dir = (dataset_root / data[split]).resolve()
    labels_dir = Path(str(images_dir).replace(f"{os.sep}images", f"{os.sep}labels"))

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    output_root = repo_path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for class_id, class_name in class_names.items():
        class_dir = output_root / f"{class_id:02d}_{sanitize_class_name(class_name)}"
        (class_dir / "rgb").mkdir(parents=True, exist_ok=True)
        (class_dir / "mask").mkdir(parents=True, exist_ok=True)
        (class_dir / "rgba").mkdir(parents=True, exist_ok=True)

    label_files = sorted(labels_dir.glob("*.txt"))

    total_available = len(label_files)
    if image_limit is not None:
        label_files = label_files[:image_limit]

    summary: dict[str, Any] = {
        "data_yaml": str(data_yaml),
        "split": split,
        "dataset_root": str(dataset_root),
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "output_dir": str(output_root),
        "total_label_files_available": total_available,
        "total_label_files_used": len(label_files),
        "image_limit": image_limit,
        "total_objects_seen": 0,
        "total_crops_saved": 0,
        "skipped_missing_images": 0,
        "skipped_invalid_boxes": 0,
        "skipped_small_masks": 0,
        "per_class_saved": {
            str(class_id): {
                "class_name": class_name,
                "count": 0,
            }
            for class_id, class_name in class_names.items()
        },
    }

    crop_index_per_class: dict[int, int] = {class_id: 0 for class_id in class_names.keys()}
    metadata_rows: list[dict[str, Any]] = []

    progress = tqdm(label_files, desc=f"Extracting {split} object bank", unit="image")

    for label_path in progress:
        image_path = find_image_for_label(label_path, images_dir)
        if image_path is None:
            summary["skipped_missing_images"] += 1
            progress.set_postfix(saved=summary["total_crops_saved"], skipped=summary["skipped_missing_images"])
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            summary["skipped_missing_images"] += 1
            progress.set_postfix(saved=summary["total_crops_saved"], skipped=summary["skipped_missing_images"])
            continue

        img_h, img_w = image.shape[:2]
        annotations = parse_yolo_label_file(label_path)

        for ann_idx, ann in enumerate(annotations):
            summary["total_objects_seen"] += 1

            class_id = ann["class_id"]
            if class_id not in class_names:
                continue

            x1, y1, x2, y2 = yolo_to_xyxy(
                x_center=ann["x_center"],
                y_center=ann["y_center"],
                width=ann["width"],
                height=ann["height"],
                img_w=img_w,
                img_h=img_h,
            )

            if x2 <= x1 or y2 <= y1:
                summary["skipped_invalid_boxes"] += 1
                continue

            bw = x2 - x1
            bh = y2 - y1
            if bw < min_crop_size or bh < min_crop_size:
                summary["skipped_invalid_boxes"] += 1
                continue

            px1, py1, px2, py2 = add_padding(
                x1, y1, x2, y2,
                img_w=img_w,
                img_h=img_h,
                pad_ratio=pad_ratio,
            )

            crop_bgr = image[py1:py2, px1:px2]
            if crop_bgr.size == 0:
                summary["skipped_invalid_boxes"] += 1
                continue

            binary_mask = make_grabcut_mask(
                crop_bgr=crop_bgr,
                inner_rect_ratio=0.1,
                iterations=grabcut_iters,
            )

            mask_area = int((binary_mask > 0).sum())
            crop_area = binary_mask.shape[0] * binary_mask.shape[1]
            area_ratio = mask_area / max(crop_area, 1)

            if area_ratio < min_mask_area_ratio:
                summary["skipped_small_masks"] += 1
                continue

            rgba_cutout = make_rgba_cutout(crop_bgr, binary_mask)
            tight_box = mask_bbox(binary_mask)

            class_name = class_names[class_id]
            class_dir = output_root / f"{class_id:02d}_{sanitize_class_name(class_name)}"

            crop_index_per_class[class_id] += 1
            crop_idx = crop_index_per_class[class_id]

            stem = f"{image_path.stem}_obj_{ann_idx:02d}_crop_{crop_idx:06d}"

            rgb_path = class_dir / "rgb" / f"{stem}.png"
            mask_path = class_dir / "mask" / f"{stem}.png"
            rgba_path = class_dir / "rgba" / f"{stem}.png"

            ok_rgb = cv2.imwrite(str(rgb_path), crop_bgr)
            ok_mask = cv2.imwrite(str(mask_path), binary_mask)
            ok_rgba = cv2.imwrite(str(rgba_path), rgba_cutout)

            if not (ok_rgb and ok_mask and ok_rgba):
                summary["skipped_invalid_boxes"] += 1
                continue

            summary["total_crops_saved"] += 1
            summary["per_class_saved"][str(class_id)]["count"] += 1

            if save_metadata:
                metadata_rows.append(
                    {
                        "rgb_path": str(rgb_path),
                        "mask_path": str(mask_path),
                        "rgba_path": str(rgba_path),
                        "source_image": str(image_path),
                        "source_label": str(label_path),
                        "class_id": class_id,
                        "class_name": class_name,
                        "original_bbox_xyxy": [x1, y1, x2, y2],
                        "padded_bbox_xyxy": [px1, py1, px2, py2],
                        "bbox_yolo": [
                            ann["x_center"],
                            ann["y_center"],
                            ann["width"],
                            ann["height"],
                        ],
                        "crop_shape_hw": [int(crop_bgr.shape[0]), int(crop_bgr.shape[1])],
                        "mask_area": mask_area,
                        "mask_area_ratio": area_ratio,
                        "tight_mask_bbox_xyxy": list(tight_box) if tight_box is not None else None,
                    }
                )

        progress.set_postfix(
            saved=summary["total_crops_saved"],
            objects=summary["total_objects_seen"],
            small_masks=summary["skipped_small_masks"],
        )

    if save_metadata:
        metadata_path = output_root / f"crop_metadata_{split}.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_rows, f, indent=2, ensure_ascii=False)
        summary["metadata_path"] = str(metadata_path)

    summary_path = output_root / f"crop_summary_{split}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== Object Bank Extraction Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    return summary


def main() -> None:
    extract_object_crops_with_masks(
        data_yaml="configs/data_insp.yaml",
        split="train",
        output_dir="data/processed/object_bank_test",
        min_crop_size=8,
        pad_ratio=0.08,
        grabcut_iters=5,
        min_mask_area_ratio=0.03,
        save_metadata=True,
        image_limit=300,  # set None for full run
    )


if __name__ == "__main__":
    main()
