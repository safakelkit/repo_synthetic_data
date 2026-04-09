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


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_class_names(model: YOLO) -> list[str]:
    names = getattr(model.model, "names", None)
    if isinstance(names, dict):
        return [str(names[i]) for i in sorted(names.keys())]
    if isinstance(names, list):
        return [str(x) for x in names]
    return []


def extract_overall_metrics(results: Any) -> dict[str, float | None]:
    box = results.box
    return {
        "precision": safe_float(getattr(box, "mp", None)),
        "recall": safe_float(getattr(box, "mr", None)),
        "map50": safe_float(getattr(box, "map50", None)),
        "map50_95": safe_float(getattr(box, "map", None)),
    }


def extract_per_class_metrics(results: Any, class_names: list[str]) -> list[dict[str, Any]]:
    box = results.box
    ap50_list = getattr(box, "ap50", None)
    maps_list = getattr(box, "maps", None)
    precision_list = getattr(box, "p", None)
    recall_list = getattr(box, "r", None)

    num_classes = len(class_names)
    rows = []

    for idx in range(num_classes):
        rows.append(
            {
                "class_id": idx,
                "class_name": class_names[idx],
                "precision": safe_float(precision_list[idx]) if precision_list is not None else None,
                "recall": safe_float(recall_list[idx]) if recall_list is not None else None,
                "ap50": safe_float(ap50_list[idx]) if ap50_list is not None else None,
                "map50_95": safe_float(maps_list[idx]) if maps_list is not None else None,
            }
        )
    return rows


def evaluate_single_dataset(
    model: YOLO,
    dataset_name: str,
    dataset_yaml: str,
    imgsz: int = 640,
    device: int | str = 0,
    plots: bool = False,
    project: str = "runs/evaluation/val",
) -> dict[str, Any]:
    results = model.val(
        data=dataset_yaml,
        split="val",
        imgsz=imgsz,
        device=device,
        plots=plots,
        project=project,
        name=dataset_name,
        exist_ok=True,
        verbose=True,
    )

    class_names = extract_class_names(model)

    return {
        "dataset_name": dataset_name,
        "dataset_yaml": dataset_yaml,
        "overall": extract_overall_metrics(results),
        "per_class": extract_per_class_metrics(results, class_names),
    }


def evaluate_model(
    model_path: str,
    imgsz: int = 640,
    device: int | str = 0,
    plots: bool = False,
    save_json_path: str | None = None,
) -> dict[str, Any]:
    model = YOLO(model_path)

    all_results: dict[str, Any] = {
        "model_path": model_path,
        "datasets": {},
    }

    for dataset_name, dataset_yaml in DATASETS.items():
        dataset_result = evaluate_single_dataset(
            model=model,
            dataset_name=dataset_name,
            dataset_yaml=dataset_yaml,
            imgsz=imgsz,
            device=device,
            plots=plots,
        )
        all_results["datasets"][dataset_name] = dataset_result

    if save_json_path is not None:
        save_path = Path(save_json_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    return all_results


def main() -> None:
    model_path = "runs/detect/runs/train/initial_yolo11n/weights/best.pt"

    results = evaluate_model(
        model_path=model_path,
        imgsz=640,
        device=0,
        plots=True,
        save_json_path="runs/evaluation/evaluation_results.json",
    )

    print(json.dumps(results["datasets"], indent=2))


if __name__ == "__main__":
    main()