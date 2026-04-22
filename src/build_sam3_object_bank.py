#load source image -> read YOLO labels -> convert YOLO box to pixel xyxy -> run SAM3 with that box as a positive box prompt -> get mask, score and refined box
#save: RGB crop, mask, RGBA cutout, metadata

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import Sam3Model, Sam3Processor


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
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


def ensure_dir_structure(output_root: Path, class_names: dict[int, str]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    for class_id, class_name in class_names.items():
        class_dir = output_root / f"{class_id:02d}_{sanitize_class_name(class_name)}"
        (class_dir / "rgb").mkdir(parents=True, exist_ok=True)
        (class_dir / "mask").mkdir(parents=True, exist_ok=True)
        (class_dir / "rgba").mkdir(parents=True, exist_ok=True)


def build_rgba_from_mask(crop_bgr: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(crop_bgr)
    alpha = binary_mask
    return cv2.merge([b, g, r, alpha])


def crop_from_mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return x1, y1, x2, y2


def select_best_sam_result(
    result: dict[str, Any],
    prompt_box_xyxy: list[int],
) -> tuple[np.ndarray | None, list[float] | None, float | None]:
    masks = result.get("masks", [])
    boxes = result.get("boxes", [])
    scores = result.get("scores", [])

    if len(masks) == 0:
        return None, None, None

    px1, py1, px2, py2 = prompt_box_xyxy
    prompt_area = max((px2 - px1) * (py2 - py1), 1)

    best_idx = None
    best_value = -1.0

    for i, mask in enumerate(masks):
        box = boxes[i] if i < len(boxes) else prompt_box_xyxy
        score = float(scores[i]) if i < len(scores) else 0.0

        bx1, by1, bx2, by2 = [float(v) for v in box]
        pred_area = max((bx2 - bx1) * (by2 - by1), 1.0)

        area_ratio = min(pred_area, prompt_area) / max(pred_area, prompt_area)

        value = 0.7 * score + 0.3 * area_ratio
        if value > best_value:
            best_value = value
            best_idx = i

    assert best_idx is not None
    best_mask = masks[best_idx]
    best_box = boxes[best_idx] if best_idx < len(boxes) else prompt_box_xyxy
    best_score = float(scores[best_idx]) if best_idx < len(scores) else None

    return best_mask, [float(v) for v in best_box], best_score


def extract_sam3_object_bank(
    data_yaml: str = "configs/data_insp.yaml",
    split: str = "train",
    model_name: str = "facebook/sam3",
    output_dir: str = "data/processed/object_bank_sam3_test",
    image_limit: int | None = 100,
    score_threshold: float = 0.30,
    mask_threshold: float = 0.50,
    min_mask_area_ratio: float = 0.01,
    save_metadata: bool = True,
    device: str | None = None,
) -> dict[str, Any]:
    data = load_yaml(data_yaml)
    class_names = get_class_names(data_yaml)

    dataset_root = Path(data["path"])
    images_dir = dataset_root / data[split]
    labels_dir = dataset_root / data[split].replace("images", "labels")

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    output_root = Path(output_dir)
    ensure_dir_structure(output_root, class_names)

    if device is None:
        device = "cpu"

    print(f"Loading SAM3 model from: {model_name}")
    print(f"Using device: {device}")

    processor = Sam3Processor.from_pretrained(model_name)
    model = Sam3Model.from_pretrained(model_name).to(device)
    model.eval()

    label_files = sorted(labels_dir.glob("*.txt"))
    total_available = len(label_files)
    if image_limit is not None:
        label_files = label_files[:image_limit]

    summary: dict[str, Any] = {
        "data_yaml": str(data_yaml),
        "split": split,
        "model_name": model_name,
        "device": device,
        "dataset_root": str(dataset_root),
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "output_dir": str(output_root),
        "total_label_files_available": total_available,
        "total_label_files_used": len(label_files),
        "image_limit": image_limit,
        "score_threshold": score_threshold,
        "mask_threshold": mask_threshold,
        "min_mask_area_ratio": min_mask_area_ratio,
        "total_objects_seen": 0,
        "total_crops_saved": 0,
        "skipped_missing_images": 0,
        "skipped_empty_masks": 0,
        "skipped_small_masks": 0,
        "skipped_low_score": 0,
        "per_class_saved": {
            str(class_id): {
                "class_name": class_name,
                "count": 0,
            }
            for class_id, class_name in class_names.items()
        },
    }

    crop_index_per_class: dict[int, int] = {class_id: 0 for class_id in class_names}
    metadata_rows: list[dict[str, Any]] = []

    progress = tqdm(label_files, desc=f"Building SAM3 object bank ({split})", unit="image")

    for label_path in progress:
        image_path = find_image_for_label(label_path, images_dir)
        if image_path is None:
            summary["skipped_missing_images"] += 1
            continue

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            summary["skipped_missing_images"] += 1
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        img_h, img_w = image_bgr.shape[:2]
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

            prompt_box = [x1, y1, x2, y2]

            inputs = processor(
                images=pil_image,
                input_boxes=[[prompt_box]],
                input_boxes_labels=[[1]],
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            processed = processor.post_process_instance_segmentation(
                outputs,
                threshold=score_threshold,
                mask_threshold=mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist(),
            )[0]

            best_mask, best_box, best_score = select_best_sam_result(processed, prompt_box)

            if best_mask is None:
                summary["skipped_empty_masks"] += 1
                continue

            if best_score is not None and best_score < score_threshold:
                summary["skipped_low_score"] += 1
                continue

            # best_mask is expected at original image resolution after post-processing
            mask_uint8 = (np.array(best_mask) > 0).astype(np.uint8) * 255

            # Restrict to bbox neighborhood to prevent unrelated spillover
            constrained_mask = np.zeros_like(mask_uint8)
            constrained_mask[y1:y2, x1:x2] = mask_uint8[y1:y2, x1:x2]

            tight = crop_from_mask_bbox(constrained_mask)
            if tight is None:
                summary["skipped_empty_masks"] += 1
                continue

            tx1, ty1, tx2, ty2 = tight
            crop_bgr = image_bgr[ty1:ty2, tx1:tx2]
            crop_mask = constrained_mask[ty1:ty2, tx1:tx2]

            if crop_bgr.size == 0 or crop_mask.size == 0:
                summary["skipped_empty_masks"] += 1
                continue

            mask_area = int((crop_mask > 0).sum())
            crop_area = crop_mask.shape[0] * crop_mask.shape[1]
            area_ratio = mask_area / max(crop_area, 1)

            if area_ratio < min_mask_area_ratio:
                summary["skipped_small_masks"] += 1
                continue

            rgba_cutout = build_rgba_from_mask(crop_bgr, crop_mask)

            class_name = class_names[class_id]
            class_dir = output_root / f"{class_id:02d}_{sanitize_class_name(class_name)}"

            crop_index_per_class[class_id] += 1
            crop_idx = crop_index_per_class[class_id]

            stem = f"{image_path.stem}_obj_{ann_idx:02d}_crop_{crop_idx:06d}"
            rgb_path = class_dir / "rgb" / f"{stem}.png"
            mask_path = class_dir / "mask" / f"{stem}.png"
            rgba_path = class_dir / "rgba" / f"{stem}.png"

            ok_rgb = cv2.imwrite(str(rgb_path), crop_bgr)
            ok_mask = cv2.imwrite(str(mask_path), crop_mask)
            ok_rgba = cv2.imwrite(str(rgba_path), rgba_cutout)

            if not (ok_rgb and ok_mask and ok_rgba):
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
                        "prompt_bbox_xyxy": prompt_box,
                        "tight_mask_bbox_xyxy": [tx1, ty1, tx2, ty2],
                        "sam_box_xyxy": best_box,
                        "sam_score": best_score,
                        "bbox_yolo": [
                            ann["x_center"],
                            ann["y_center"],
                            ann["width"],
                            ann["height"],
                        ],
                        "crop_shape_hw": [int(crop_bgr.shape[0]), int(crop_bgr.shape[1])],
                        "mask_area": mask_area,
                        "mask_area_ratio": area_ratio,
                    }
                )

        progress.set_postfix(
            saved=summary["total_crops_saved"],
            objects=summary["total_objects_seen"],
            low_score=summary["skipped_low_score"],
            empty=summary["skipped_empty_masks"],
        )

    if save_metadata:
        metadata_path = output_root / f"crop_metadata_{split}.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_rows, f, indent=2, ensure_ascii=False)
        summary["metadata_path"] = str(metadata_path)

    summary_path = output_root / f"crop_summary_{split}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== SAM3 Object Bank Extraction Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    return summary


def main() -> None:
    extract_sam3_object_bank(
        data_yaml="configs/data_insp.yaml",
        split="train",
        model_name="facebook/sam3",
        output_dir="data/processed/object_bank_sam3_test",
        image_limit=100,
        score_threshold=0.30,
        mask_threshold=0.50,
        min_mask_area_ratio=0.01,
        save_metadata=True,
        device=None,
    )


if __name__ == "__main__":
    main()