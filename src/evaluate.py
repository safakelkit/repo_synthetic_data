from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ultralytics import YOLO


DATASETS = {
    "insp_det": "configs/data_insp.yaml",
    "insp_mot_det_easy": "configs/data_insp_mot_easy.yaml",
    "insp_mot_det_hard": "configs/data_insp_mot_hard.yaml",
}

# Update this path if your trained model is stored elsewhere.
DEFAULT_MODEL_PATH = "runs/detect/runs/train/initial_yolo11n/weights/best.pt"

# Where evaluation outputs will be saved.
OUTPUT_DIR = Path("runs/evaluation")
OUTPUT_JSON = OUTPUT_DIR / "evaluation_results.json"


def safe_float(value: Any) -> float | None:
    """Convert value to float if possible, otherwise return None."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_class_names(model: YOLO) -> list[str]:
    """
    Extract class names from the loaded YOLO model.
    Falls back to class indices as strings if needed.
    """
    names = getattr(model.model, "names", None)

    if isinstance(names, dict):
        return [str(names[i]) for i in sorted(names.keys())]

    if isinstance(names, list):
        return [str(name) for name in names]

    return []


def extract_overall_metrics(results: Any) -> dict[str, float | None]:
    """
    Extract overall detection metrics from Ultralytics results object.
    """
    box = results.box

    metrics = {
        "precision": safe_float(getattr(box, "mp", None)),
        "recall": safe_float(getattr(box, "mr", None)),
        "map50": safe_float(getattr(box, "map50", None)),
        "map50_95": safe_float(getattr(box, "map", None)),
    }
    return metrics


def extract_per_class_metrics(results: Any, class_names: list[str]) -> list[dict[str, float | str | None]]:
    """
    Extract per-class metrics from Ultralytics results object.

    Notes:
    - `results.box.ap50` is typically per-class AP@50
    - `results.box.maps` is typically per-class mAP@50-95
    - Per-class precision/recall are not always exposed in a stable way across versions,
      so we only include them if available.
    """
    box = results.box

    ap50_list = getattr(box, "ap50", None)
    maps_list = getattr(box, "maps", None)

    # Some versions may not expose per-class precision/recall directly.
    precision_list = getattr(box, "p", None)
    recall_list = getattr(box, "r", None)

    per_class_results: list[dict[str, float | str | None]] = []

    num_classes = len(class_names)
    if num_classes == 0 and maps_list is not None:
        num_classes = len(maps_list)
        class_names = [str(i) for i in range(num_classes)]

    for idx in range(num_classes):
        class_result = {
            "class_id": idx,
            "class_name": class_names[idx] if idx < len(class_names) else str(idx),
            "precision": safe_float(precision_list[idx]) if precision_list is not None and idx < len(precision_list) else None,
            "recall": safe_float(recall_list[idx]) if recall_list is not None and idx < len(recall_list) else None,
            "ap50": safe_float(ap50_list[idx]) if ap50_list is not None and idx < len(ap50_list) else None,
            "map50_95": safe_float(maps_list[idx]) if maps_list is not None and idx < len(maps_list) else None,
        }
        per_class_results.append(class_result)

    return per_class_results


def print_summary(dataset_name: str, overall: dict[str, float | None], per_class: list[dict[str, float | str | None]]) -> None:
    """Print clean human-readable evaluation summary."""
    print("\n" + "=" * 70)
    print(f"Dataset: {dataset_name}")
    print("=" * 70)
    print(f"Precision : {overall['precision']}")
    print(f"Recall    : {overall['recall']}")
    print(f"mAP50     : {overall['map50']}")
    print(f"mAP50-95  : {overall['map50_95']}")

    print("\nPer-class metrics:")
    for row in per_class:
        print(
            f"  [{row['class_id']:>2}] {row['class_name']:<15} "
            f"AP50: {row['ap50']} | "
            f"mAP50-95: {row['map50_95']} | "
            f"P: {row['precision']} | "
            f"R: {row['recall']}"
        )


def evaluate_dataset(model: YOLO, dataset_name: str, dataset_yaml: str, imgsz: int = 640, device: int | str = 0) -> dict[str, Any]:
    """
    Evaluate a model on one dataset yaml and return structured metrics.
    """
    print(f"\nRunning evaluation on {dataset_name} using {dataset_yaml} ...")

    results = model.val(
        data=dataset_yaml,
        split="val",
        imgsz=imgsz,
        device=device,
        plots=True,
        project="runs/evaluation/val",
        name=dataset_name,
        exist_ok=True,
    )

    class_names = extract_class_names(model)
    overall_metrics = extract_overall_metrics(results)
    per_class_metrics = extract_per_class_metrics(results, class_names)

    return {
        "dataset_name": dataset_name,
        "dataset_yaml": dataset_yaml,
        "overall": overall_metrics,
        "per_class": per_class_metrics,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = DEFAULT_MODEL_PATH
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    all_results: dict[str, Any] = {
        "model_path": model_path,
        "datasets": {},
    }

    for dataset_name, dataset_yaml in DATASETS.items():
        dataset_result = evaluate_dataset(
            model=model,
            dataset_name=dataset_name,
            dataset_yaml=dataset_yaml,
            imgsz=640,
            device=0,
        )

        all_results["datasets"][dataset_name] = dataset_result
        print_summary(
            dataset_name=dataset_name,
            overall=dataset_result["overall"],
            per_class=dataset_result["per_class"],
        )

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"Saved evaluation results to: {OUTPUT_JSON}")
    print("=" * 70)


if __name__ == "__main__":
    main()