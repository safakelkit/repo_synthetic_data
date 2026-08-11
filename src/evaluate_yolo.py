from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from ultralytics import YOLO


DATASETS = {
    "insp_det": ("configs/data_insp.yaml", "test"),
    "insp_mot_det_easy": ("configs/data_insp_mot_easy.yaml", "test"),
    "insp_mot_det_hard": ("configs/data_insp_mot_hard.yaml", "test"),
}
REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


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
    ap_class_index = getattr(box, "ap_class_index", None)

    num_classes = len(class_names)

    rows = [
        {
            "class_id": idx,
            "class_name": class_names[idx],
            "precision": None,
            "recall": None,
            "ap50": None,
            "map50_95": None,
        }
        for idx in range(num_classes)
    ]

    if ap_class_index is None:
        usable_len = min(
            [
                len(x)
                for x in [precision_list, recall_list, ap50_list]
                if x is not None
            ],
            default=0,
        )

        for idx in range(min(num_classes, usable_len)):
            rows[idx]["precision"] = (
                safe_float(precision_list[idx]) if precision_list is not None else None
            )
            rows[idx]["recall"] = (
                safe_float(recall_list[idx]) if recall_list is not None else None
            )
            rows[idx]["ap50"] = (
                safe_float(ap50_list[idx]) if ap50_list is not None else None
            )
            rows[idx]["map50_95"] = (
                safe_float(maps_list[idx])
                if maps_list is not None and idx < len(maps_list)
                else None
            )

        return rows

    for metric_idx, class_id_raw in enumerate(ap_class_index):
        class_id = int(class_id_raw)

        if class_id < 0 or class_id >= num_classes:
            continue

        rows[class_id]["precision"] = (
            safe_float(precision_list[metric_idx])
            if precision_list is not None and metric_idx < len(precision_list)
            else None
        )
        rows[class_id]["recall"] = (
            safe_float(recall_list[metric_idx])
            if recall_list is not None and metric_idx < len(recall_list)
            else None
        )
        rows[class_id]["ap50"] = (
            safe_float(ap50_list[metric_idx])
            if ap50_list is not None and metric_idx < len(ap50_list)
            else None
        )

        rows[class_id]["map50_95"] = (
            safe_float(maps_list[class_id])
            if maps_list is not None and class_id < len(maps_list)
            else None
        )

    return rows


def evaluate_single_dataset(
    model: YOLO,
    dataset_name: str,
    dataset_yaml: str,
    split: str,
    imgsz: int = 640,
    device: int | str = 0,
    plots: bool = True,
    project: str = "runs/evaluation/default",
    conf: float = 0.001,
    iou: float = 0.7,
    max_det: int = 300,
) -> dict[str, Any]:
    results = model.val(
        data=str(repo_path(dataset_yaml)),
        split=split,
        imgsz=imgsz,
        device=device,
        plots=plots,
        conf=conf,
        iou=iou,
        max_det=max_det,
        # Keep output rooted in this repository, independent of cwd and the
        # Ultralytics global runs_dir setting.
        project=str(repo_path(project)),
        name=dataset_name,
        exist_ok=True,
        verbose=True,
    )

    class_names = extract_class_names(model)

    return {
        "dataset_name": dataset_name,
        "dataset_yaml": dataset_yaml,
        "split": split,
        "evaluation_settings": {
            "imgsz": imgsz,
            "conf": conf,
            "iou": iou,
            "max_det": max_det,
            "device": device,
        },
        "overall": extract_overall_metrics(results),
        "per_class": extract_per_class_metrics(results, class_names),
    }


def evaluate_model(
    model_path: str,
    imgsz: int = 640,
    device: int | str = 0,
    plots: bool = True,
    save_json_path: str | None = None,
    eval_project: str | None = None,
) -> dict[str, Any]:
    resolved_model_path = repo_path(model_path)
    model = YOLO(str(resolved_model_path))

    model_name = resolved_model_path.parent.parent.name

    if eval_project is None:
        eval_project = str(repo_path(f"runs/evaluation/{model_name}"))
    else:
        eval_project = str(repo_path(eval_project))

    all_results: dict[str, Any] = {
        "model_path": str(resolved_model_path),
        "eval_project": eval_project,
        "datasets": {},
    }

    for dataset_name, (dataset_yaml, split) in DATASETS.items():
        dataset_result = evaluate_single_dataset(
            model=model,
            dataset_name=dataset_name,
            dataset_yaml=dataset_yaml,
            split=split,
            imgsz=imgsz,
            device=device,
            plots=plots,
            project=eval_project,
        )
        all_results["datasets"][dataset_name] = dataset_result

    if save_json_path is not None:
        save_path = repo_path(save_json_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    return all_results


def main() -> None:
    if len(sys.argv) < 2:
        raise ValueError("Usage: python src/evaluate_yolo.py <model_path> [output_json_path]")

    model_path = sys.argv[1]
    model_name = Path(model_path).parent.parent.name

    if len(sys.argv) >= 3:
        save_path = sys.argv[2]
    else:
        save_path = f"runs/evaluation/{model_name}_results.json"

    eval_project = str(repo_path(f"runs/evaluation/{model_name}"))

    results = evaluate_model(
        model_path=model_path,
        imgsz=640,
        device=0,
        plots=True,
        save_json_path=save_path,
        eval_project=eval_project,
    )

    print(json.dumps(results["datasets"], indent=2))


if __name__ == "__main__":
    main()
