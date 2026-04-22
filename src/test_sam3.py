from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor


MODEL_NAME = "facebook/sam3"


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


def parse_first_yolo_box(label_path: str | Path) -> tuple[int, float, float, float, float]:
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, x_center, y_center, width, height = parts
            return (
                int(float(class_id)),
                float(x_center),
                float(y_center),
                float(width),
                float(height),
            )
    raise ValueError(f"No valid YOLO box found in: {label_path}")


def select_best_mask(result: dict, prompt_box_xyxy: list[int]) -> tuple[np.ndarray, list[float], float]:
    masks = result.get("masks", [])
    boxes = result.get("boxes", [])
    scores = result.get("scores", [])

    if len(masks) == 0:
        raise ValueError("SAM3 returned no masks.")

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
    best_mask = np.array(masks[best_idx])
    best_box = [float(v) for v in boxes[best_idx]] if best_idx < len(boxes) else prompt_box_xyxy
    best_score = float(scores[best_idx]) if best_idx < len(scores) else 0.0
    return best_mask, best_box, best_score


def main() -> None:
    # Change these two paths to a real pair from your dataset
    image_path = "data/raw/insp-det/train/images/1b23f767-11d9-472e-bc0f-c848e07803e5.jpg"
    label_path = "data/raw/insp-det/train/labels/1b23f767-11d9-472e-bc0f-c848e07803e5.txt"

    output_dir = Path("runs/sam3_single_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cpu"
    print(f"Using device: {device}")

    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img_h, img_w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    class_id, x_center, y_center, width, height = parse_first_yolo_box(label_path)
    x1, y1, x2, y2 = yolo_to_xyxy(x_center, y_center, width, height, img_w, img_h)
    prompt_box = [x1, y1, x2, y2]

    print(f"class_id: {class_id}")
    print(f"prompt_box: {prompt_box}")

    processor = Sam3Processor.from_pretrained(MODEL_NAME)
    model = Sam3Model.from_pretrained(MODEL_NAME).to(device)
    model.eval()

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
        threshold=0.3,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]

    best_mask, best_box, best_score = select_best_mask(processed, prompt_box)

    binary_mask = (best_mask > 0).astype(np.uint8) * 255

    constrained_mask = np.zeros_like(binary_mask)
    constrained_mask[y1:y2, x1:x2] = binary_mask[y1:y2, x1:x2]

    overlay = image_bgr.copy()
    overlay[constrained_mask > 0] = (
        0.6 * overlay[constrained_mask > 0] + 0.4 * np.array([0, 255, 0])
    ).astype(np.uint8)

    vis = overlay.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

    bx1, by1, bx2, by2 = [int(v) for v in best_box]
    cv2.rectangle(vis, (bx1, by1), (bx2, by2), (255, 0, 0), 2)

    crop = image_bgr[y1:y2, x1:x2]
    crop_mask = constrained_mask[y1:y2, x1:x2]

    b, g, r = cv2.split(crop)
    rgba = cv2.merge([b, g, r, crop_mask])

    cv2.imwrite(str(output_dir / "original.png"), image_bgr)
    cv2.imwrite(str(output_dir / "mask.png"), constrained_mask)
    cv2.imwrite(str(output_dir / "overlay.png"), overlay)
    cv2.imwrite(str(output_dir / "visualization.png"), vis)
    cv2.imwrite(str(output_dir / "crop_rgb.png"), crop)
    cv2.imwrite(str(output_dir / "crop_mask.png"), crop_mask)
    cv2.imwrite(str(output_dir / "crop_rgba.png"), rgba)

    print(f"best_score: {best_score:.4f}")
    print(f"best_box: {best_box}")
    print(f"Saved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()