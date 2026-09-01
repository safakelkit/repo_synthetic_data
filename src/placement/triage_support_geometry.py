from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/placement/support_geometry_full_review_v1.yaml"
EXTRA_FIELDS = ["triage_status", "triage_reasons"]


def repo_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Invalid YAML mapping: {path}")
    return value


def triage_row(row: dict[str, str], rules: dict[str, Any]) -> tuple[str, list[str]]:
    if row["derivation_status"] != "derived":
        return "rejected", ["no_valid_region"]
    area = float(row["region_area_ratio"])
    anchors = int(row["anchor_count"])
    reasons: list[str] = []
    if area < float(rules["minimum_area_ratio"]):
        reasons.append("area_below_conservative_minimum")
    if area > float(rules["maximum_area_ratio"]):
        reasons.append("area_above_conservative_maximum")
    if anchors < int(rules["minimum_anchors"]):
        reasons.append("too_few_anchor_candidates")
    return ("needs_review", reasons) if reasons else ("candidate_accept", [])


def stratified_sample(rows: list[dict[str, str]], count: int, seed: int) -> list[dict[str, str]]:
    if len(rows) <= count:
        return sorted(rows, key=lambda row: row["background_path"])
    ordered = sorted(rows, key=lambda row: (float(row["region_area_ratio"]), row["background_path"]))
    # Evenly spaced area ranks expose both tails and the distribution center.
    ranks = np.linspace(0, len(ordered) - 1, count).round().astype(int).tolist()
    selected = [ordered[index] for index in ranks]
    rng = random.Random(seed)
    rng.shuffle(selected)
    return selected


def draw_review_tile(row: dict[str, str], tile_size: int) -> np.ndarray:
    background = cv2.imread(str(repo_path(row["background_path"])), cv2.IMREAD_COLOR)
    region = cv2.imread(str(repo_path(row["region_path"])), cv2.IMREAD_GRAYSCALE)
    if background is None or region is None or background.shape[:2] != region.shape:
        raise RuntimeError(f"Could not load review inputs for {row['background_path']}")
    binary = region > 0
    color = np.asarray((40, 180, 240) if row["triage_status"] == "needs_review" else (60, 190, 70))
    overlay = background.copy()
    overlay[binary] = (0.55 * overlay[binary] + 0.45 * color).astype(np.uint8)
    for x, y in json.loads(row["anchor_points_xy"]):
        cv2.circle(overlay, (int(x), int(y)), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(overlay, (int(x), int(y)), 5, color.tolist(), 1, cv2.LINE_AA)
    tile = cv2.resize(overlay, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
    canvas = np.full((tile_size + 42, tile_size, 3), 255, dtype=np.uint8)
    canvas[:tile_size] = tile
    cv2.putText(canvas, Path(row["background_path"]).stem[:30], (4, tile_size + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 0, 0), 1, cv2.LINE_AA)
    label = f"{row['support_type']} {float(row['region_area_ratio']):.3f} a={row['anchor_count']}"
    cv2.putText(canvas, label, (4, tile_size + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def write_sheet(rows: list[dict[str, str]], destination: Path, tile_size: int, columns: int) -> None:
    tiles = [draw_review_tile(row, tile_size) for row in rows]
    if not tiles:
        return
    row_count = math.ceil(len(tiles) / columns)
    tiles.extend([np.full_like(tiles[0], 255)] * (row_count * columns - len(tiles)))
    sheet = np.vstack([np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)])
    if not cv2.imwrite(str(destination), sheet):
        raise RuntimeError(f"Could not write review sheet: {destination}")


def run(config_path: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    manifest_path = repo_path(config["input_manifest"])
    run_summary_path = repo_path(config["input_run_summary"])
    output_root = repo_path(config["output_root"])
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Review output is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    if sha256(manifest_path) != run_summary["region_manifest_sha256"]:
        raise ValueError("Geometry manifest checksum differs from run summary")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])

    counts: Counter[str] = Counter()
    for row in rows:
        status, reasons = triage_row(row, config["triage"][row["support_type"]])
        row["triage_status"] = status
        row["triage_reasons"] = json.dumps(reasons)
        counts[f"{row['support_type']}:{status}"] += 1

    triage_manifest = output_root / "triaged_support_regions.csv"
    with triage_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + EXTRA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    sample_config = config["visual_sample"]
    sample_rows: list[dict[str, str]] = []
    for support_index, support_type in enumerate(config["triage"]):
        for group_index, group in enumerate(sample_config["groups"]):
            candidates = [row for row in rows if row["support_type"] == support_type and row["triage_status"] == group]
            sample = stratified_sample(
                candidates,
                int(sample_config["per_support_per_group"]),
                int(config["seed"]) + support_index * 100 + group_index,
            )
            sample_rows.extend(sample)
            write_sheet(
                sample,
                output_root / f"review_{support_type}_{group}.jpg",
                int(sample_config["tile_size"]),
                int(sample_config["columns"]),
            )
    sample_manifest = output_root / "visual_review_sample.csv"
    with sample_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + EXTRA_FIELDS)
        writer.writeheader()
        writer.writerows(sample_rows)

    full_sheet_root = output_root / "all_candidate_sheets"
    full_sheet_root.mkdir(parents=True, exist_ok=True)
    full_sheet_count = 0
    for support_type in sample_config.get("full_sheet_supports", []):
        candidates = sorted(
            (
                row for row in rows
                if row["support_type"] == support_type
                and row["triage_status"] == "candidate_accept"
            ),
            key=lambda row: row["background_path"],
        )
        chunk_size = int(sample_config["full_sheet_chunk_size"])
        for start in range(0, len(candidates), chunk_size):
            chunk = candidates[start:start + chunk_size]
            destination = full_sheet_root / f"{support_type}_{start // chunk_size:03d}.jpg"
            write_sheet(
                chunk,
                destination,
                int(sample_config["tile_size"]),
                int(sample_config["columns"]),
            )
            full_sheet_count += 1

    summary = {
        "format_version": 1,
        "status": "triage_complete_visual_review_pending",
        "config_path": relative(config_path),
        "config_sha256": sha256(config_path),
        "input_manifest": relative(manifest_path),
        "input_manifest_sha256": sha256(manifest_path),
        "triage_manifest": relative(triage_manifest),
        "triage_manifest_sha256": sha256(triage_manifest),
        "row_count": len(rows),
        "counts": dict(sorted(counts.items())),
        "visual_sample_rows": len(sample_rows),
        "visual_sample_manifest": relative(sample_manifest),
        "visual_sample_manifest_sha256": sha256(sample_manifest),
        "full_candidate_sheet_count": full_sheet_count,
        "production_use_approved": False,
    }
    (output_root / "triage_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Conservatively triage full support geometry")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(run(repo_path(args.config)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
