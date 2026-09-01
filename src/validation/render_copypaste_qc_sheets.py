from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = REPO_ROOT / "data/synthetic/cp_v1_seed42"
SEVERITY_ORDER = {"clean": 0, "light": 1, "medium": 2, "heavy": 3}


def repo_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def read_box(label: Path, width: int, height: int) -> tuple[int, int, int, int]:
    _, xc, yc, bw, bh = label.read_text(encoding="utf-8").split()
    xc, yc, bw, bh = map(float, (xc, yc, bw, bh))
    return (
        round((xc - bw / 2) * width),
        round((yc - bh / 2) * height),
        round((xc + bw / 2) * width),
        round((yc + bh / 2) * height),
    )


def tile(row: dict[str, str], dataset_root: Path, size: int = 320) -> np.ndarray:
    image_path = repo_path(row["image"])
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    height, width = image.shape[:2]
    x1, y1, x2, y2 = read_box(dataset_root / "labels" / f"{image_path.stem}.txt", width, height)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), max(2, round(min(width, height) / 300)))
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    canvas = np.full((size + 42, size, 3), 255, np.uint8)
    canvas[:size] = resized
    cv2.putText(canvas, image_path.stem, (5, size + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, row["severity"], (5, size + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Render class-wise copy-paste QC contact sheets")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--tile-size", type=int, default=320)
    args = parser.parse_args()

    dataset_root = repo_path(str(args.dataset_root))
    sample = dataset_root / "manual_review_sample.csv"
    output = dataset_root / "manual_review_sheets"
    with sample.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output.mkdir(parents=True, exist_ok=True)
    class_ids = sorted({int(row["class_id"]) for row in rows})
    for class_id in class_ids:
        selected = sorted(
            (row for row in rows if int(row["class_id"]) == class_id),
            key=lambda row: (SEVERITY_ORDER[row["severity"]], row["image"]),
        )
        tiles = [tile(row, dataset_root, args.tile_size) for row in selected]
        columns = 4
        rows_count = math.ceil(len(tiles) / columns)
        sheet = np.vstack([np.hstack(tiles[index:index + columns]) for index in range(0, rows_count * columns, columns)])
        destination = output / f"class_{class_id:02d}.jpg"
        if not cv2.imwrite(str(destination), sheet):
            raise RuntimeError(f"Could not write {destination}")
    print(f"Wrote {len(class_ids)} QC sheets to {output.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
