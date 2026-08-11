from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_YAML = REPO_ROOT / "configs/data_insp.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/processed/object_size_analysis"
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def load_dataset_config(path: Path) -> tuple[Path, dict[int, str]]:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    dataset_root = Path(config["path"]) if config.get("path") else path.parent
    if not dataset_root.is_absolute():
        dataset_root = (path.parent / dataset_root).resolve()
    train_images = (dataset_root / config["train"]).resolve()
    labels_dir = Path(str(train_images).replace(f"{os.sep}images", f"{os.sep}labels"))
    names = {int(class_id): str(name) for class_id, name in config["names"].items()}
    return labels_dir, names


def read_training_boxes(labels_dir: Path, class_names: dict[int, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        for annotation_index, line in enumerate(label_path.read_text().splitlines()):
            if not line.strip():
                continue
            values = line.split()
            if len(values) != 5:
                raise ValueError(f"Invalid YOLO row in {label_path}:{annotation_index + 1}")
            class_id = int(values[0])
            x_center, y_center, width, height = map(float, values[1:])
            if class_id not in class_names:
                raise ValueError(f"Unknown class {class_id} in {label_path}")
            if not all(0.0 <= value <= 1.0 for value in (x_center, y_center, width, height)):
                raise ValueError(f"Non-normalized box in {label_path}:{annotation_index + 1}")
            if width <= 0.0 or height <= 0.0:
                raise ValueError(f"Empty box in {label_path}:{annotation_index + 1}")

            rows.append(
                {
                    "source_label": str(label_path.relative_to(REPO_ROOT)),
                    "annotation_index": annotation_index,
                    "class_id": class_id,
                    "class_name": class_names[class_id],
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height,
                    "area": width * height,
                    "aspect_ratio": width / height,
                }
            )
    return rows


def describe(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    result = {
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }
    result.update(
        {
            f"p{int(quantile * 100):02d}": float(np.quantile(array, quantile))
            for quantile in QUANTILES
        }
    )
    return result


def analyze_sizes(
    data_yaml: Path = DEFAULT_DATA_YAML,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    labels_dir, class_names = load_dataset_config(data_yaml)
    rows = read_training_boxes(labels_dir, class_names)
    output_dir.mkdir(parents=True, exist_ok=True)

    templates_path = output_dir / "train_box_templates.csv"
    with open(templates_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    per_class: dict[str, Any] = {}
    for class_id, class_name in class_names.items():
        class_rows = [row for row in rows if row["class_id"] == class_id]
        per_class[str(class_id)] = {
            "class_name": class_name,
            "box_count": len(class_rows),
            "source_image_count": len({row["source_label"] for row in class_rows}),
            "width": describe([row["width"] for row in class_rows]),
            "height": describe([row["height"] for row in class_rows]),
            "area": describe([row["area"] for row in class_rows]),
            "aspect_ratio": describe([row["aspect_ratio"] for row in class_rows]),
        }

    summary = {
        "format_version": 1,
        "source_data_yaml": str(data_yaml.relative_to(REPO_ROOT)),
        "source_split": "INSP-DET train",
        "label_files": len(list(labels_dir.glob("*.txt"))),
        "total_boxes": len(rows),
        "class_counts": dict(sorted(Counter(row["class_id"] for row in rows).items())),
        "quantiles": list(QUANTILES),
        "generator_sampling_rule": {
            "metric": "normalized_bbox_area",
            "range": "class-specific p10 to p90",
            "sampling": "sample an observed training-box area inside the range",
            "target_data_used": False,
        },
        "per_class": per_class,
        "templates_file": str(templates_path.relative_to(REPO_ROOT)),
    }
    summary_path = output_dir / "object_size_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze INSP-DET train object sizes")
    parser.add_argument("--data-yaml", type=Path, default=DEFAULT_DATA_YAML)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    summary = analyze_sizes(args.data_yaml.resolve(), args.output_dir.resolve())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
