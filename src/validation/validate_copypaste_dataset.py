from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/quality/copy_paste_qc_v1.yaml"


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid YAML mapping: {path}")
    return value


def parse_label(path: Path) -> tuple[int, list[float]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"Expected exactly one annotation: {path}")
    parts = lines[0].split()
    if len(parts) != 5:
        raise ValueError(f"Invalid YOLO annotation: {path}")
    class_id = int(parts[0])
    values = [float(value) for value in parts[1:]]
    x, y, width, height = values
    if not (0 <= class_id < 16 and 0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
        raise ValueError(f"Out-of-range YOLO annotation: {path}")
    if x - width / 2 < -1e-6 or y - height / 2 < -1e-6 or x + width / 2 > 1 + 1e-6 or y + height / 2 > 1 + 1e-6:
        raise ValueError(f"YOLO box extends outside image: {path}")
    return class_id, values


def validate(config_path: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    root = repo_path(config["dataset_root"])
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = int(config["expected_images"])
    images = sorted((root / "images").glob("*.jpg"))
    labels = sorted((root / "labels").glob("*.txt"))
    if not (len(images) == len(labels) == len(metadata) == expected):
        raise ValueError(f"Count mismatch: images={len(images)}, labels={len(labels)}, metadata={len(metadata)}")

    class_counts: Counter[int] = Counter()
    severity_by_class: Counter[tuple[int, str]] = Counter()
    hashes: set[str] = set()
    ordered: list[tuple[int, str]] = []
    for index, record in enumerate(metadata):
        image = repo_path(record["image"])
        label = repo_path(record["label"])
        if image != images[index] or label != labels[index]:
            raise ValueError(f"Metadata ordering/path mismatch at index {index}")
        decoded = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if decoded is None or decoded.shape[:2] != (int(record["background_height"]), int(record["background_width"])):
            raise ValueError(f"Unreadable image or changed dimensions: {image}")
        digest = sha256(image)
        if digest != record["image_sha256"] or digest in hashes:
            raise ValueError(f"Image hash mismatch or exact duplicate: {image}")
        hashes.add(digest)
        class_id, values = parse_label(label)
        if class_id != int(record["primary_class_id"]):
            raise ValueError(f"Metadata/label class mismatch: {label}")
        realized = float(record["object"]["realized_normalized_area"])
        labelled_area = values[2] * values[3]
        tolerance = float(config["automatic"]["maximum_target_area_relative_error"])
        if abs(labelled_area - realized) / max(realized, 1e-12) > tolerance:
            raise ValueError(f"Realized/label area mismatch: {label}")
        severity = record["degradation_severity"]
        if severity == "clean" and record["degradations"]:
            raise ValueError(f"Clean image has degradation metadata: {image}")
        if severity != "clean" and not record["degradations"]:
            raise ValueError(f"Degraded image lacks operations: {image}")
        class_counts[class_id] += 1
        severity_by_class[(class_id, severity)] += 1
        ordered.append((class_id, severity))

    per_class = int(config["expected_images_per_class"])
    if class_counts != Counter({class_id: per_class for class_id in range(int(config["classes"]))}):
        raise ValueError(f"Final class balance failed: {class_counts}")
    base = config["severity_per_class_per_32"]
    for prefix in config["expected_prefixes"]:
        factor = int(prefix) // 512
        prefix_counts = Counter(ordered[: int(prefix)])
        expected_counts = Counter({(class_id, severity): int(count) * factor for class_id in range(16) for severity, count in base.items()})
        if prefix_counts != expected_counts:
            raise ValueError(f"Class/severity balance failed at prefix {prefix}")
        manifest = root / "manifests" / f"CP-B{int(prefix):04d}.txt"
        paths = [
            (manifest.parent / line.strip()).resolve()
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(paths) != int(prefix) or any(not path.is_file() for path in paths):
            raise ValueError(f"Invalid subset manifest: {manifest}")

    review_config = config["manual_review"]
    review_rng = random.Random(int(config["seed"]))
    selected: list[dict[str, Any]] = []
    count = int(review_config["images_per_class_per_severity"])
    for class_id in range(16):
        for severity in base:
            candidates = [record for record in metadata if int(record["primary_class_id"]) == class_id and record["degradation_severity"] == severity]
            for record in review_rng.sample(candidates, count):
                selected.append({
                    "image": record["image"],
                    "class_id": class_id,
                    "severity": severity,
                    "review_status": "pending",
                    "reviewer_note": "",
                })
    review_path = root / "manual_review_sample.csv"
    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)

    result = {
        "format_version": 1,
        "status": "automatic_qc_passed_manual_review_pending",
        "config": config_path.resolve().relative_to(REPO_ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "images": expected,
        "unique_image_hashes": len(hashes),
        "class_counts": dict(sorted(class_counts.items())),
        "severity_counts": {f"{class_id}:{severity}": count for (class_id, severity), count in sorted(severity_by_class.items())},
        "manual_review_sample": review_path.resolve().relative_to(REPO_ROOT).as_posix(),
        "manual_review_rows": len(selected),
    }
    (root / "qc_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the canonical cut-paste dataset")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(validate(repo_path(args.config)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
