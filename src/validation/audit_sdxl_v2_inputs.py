"""Audit SDXL-v2 background prompts and deterministic source-asset diversity."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import importlib.util
import json
from pathlib import Path
import re
from typing import Any

from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR_PATH = REPO_ROOT / "src/generation/generate_sdxl_dataset.py"
RUNNER_PATH = REPO_ROOT / "src/generation/run_sdxl_depth_background_experiment.py"
DEFAULT_CONFIG = REPO_ROOT / "configs/generation/experiments/sdxl_v2_background_source_review.yaml"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def checkerboard(size: tuple[int, int], step: int = 12) -> Image.Image:
    image = Image.new("RGB", size, (235, 235, 235))
    draw = ImageDraw.Draw(image)
    for y in range(0, size[1], step):
        for x in range(0, size[0], step):
            if (x // step + y // step) % 2:
                draw.rectangle((x, y, x + step - 1, y + step - 1), fill=(205, 205, 205))
    return image


def make_source_sheets(rows: list[dict[str, str]], output: Path, title: str) -> list[Path]:
    thumb, caption, columns = 180, 34, 8
    outputs = []
    for page, start in enumerate(range(0, len(rows), 32), start=1):
        selected = rows[start:start + 32]
        row_count = (len(selected) + columns - 1) // columns
        sheet = Image.new("RGB", (thumb * columns, 30 + (thumb + caption) * row_count), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((6, 7), f"{title} | page {page}", fill="black")
        for local_index, row in enumerate(selected):
            index = start + local_index
            tile = checkerboard((thumb, thumb))
            with Image.open(REPO_ROOT / row["rgba_path"]).convert("RGBA") as source:
                source.thumbnail((thumb - 12, thumb - 12), Image.Resampling.LANCZOS)
                tile.paste(
                    source,
                    ((thumb - source.width) // 2, (thumb - source.height) // 2),
                    source,
                )
            x = (local_index % columns) * thumb
            y = 30 + (local_index // columns) * (thumb + caption)
            sheet.paste(tile, (x, y))
            draw.text((x + 4, y + thumb + 3), f"{index:02d} {row['stem'][:21]}", fill="black")
        output.parent.mkdir(parents=True, exist_ok=True)
        page_output = output.with_name(f"{output.stem}_p{page:02d}{output.suffix}")
        sheet.save(page_output)
        outputs.append(page_output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    generator = load_module("sdxl_generator", GENERATOR_PATH)
    runner = load_module("sdxl_v2_runner", RUNNER_PATH)
    helper = generator.load_helpers()
    config_path = helper.repo_path(args.config)
    config, _ = generator.load_generation_config(helper, config_path)
    plate_rows = runner.plate_records(config)

    source_config = config["silhouette_source"]
    attempts = int(source_config["projected_samples_per_class"])
    minimum_unique = int(source_config["minimum_unique_per_class"])
    maximum_reuse = int(source_config["maximum_reuse_per_asset"])
    manifest_path = helper.repo_path(source_config["audit_manifest"])
    names = helper.load_yaml(helper.repo_path(config["scene_policy"]))["class_policy"]["names"]
    source_reports: list[dict[str, Any]] = []
    source_rows_for_sheets: dict[int, list[dict[str, str]]] = {}
    source_candidates_per_class = int(config["source_review_gate"]["candidates_per_class"])
    required_accepted_per_class = int(config["source_review_gate"]["required_accepted_per_class"])

    for class_id in sorted(int(value) for value in config["class_ids"]):
        accepted_rows = helper.read_masks(manifest_path, class_id)
        ordered_rows = generator.ordered_mask_rows(config, accepted_rows, class_id)
        selected = [ordered_rows[attempt % len(ordered_rows)] for attempt in range(attempts)]
        counts = Counter(row["asset_id"] for row in selected)
        report = {
            "class_id": class_id,
            "class_name": names[class_id],
            "projected_samples": attempts,
            "accepted_pool_size": len(accepted_rows),
            "unique_sources": len(counts),
            "maximum_reuse": max(counts.values()),
        }
        if report["unique_sources"] < minimum_unique:
            raise RuntimeError(f"Source diversity minimum failed: {report}")
        if report["maximum_reuse"] > maximum_reuse:
            raise RuntimeError(f"Source reuse cap failed: {report}")
        source_reports.append(report)
        candidates = ordered_rows[:min(source_candidates_per_class, len(ordered_rows))]
        if len(candidates) < required_accepted_per_class:
            raise RuntimeError(f"Insufficient source review candidates for class {class_id}")
        source_rows_for_sheets[class_id] = candidates

    scene_counts = Counter(row["scene_name"] for row in plate_rows)
    expected_candidates = int(config["background_review_gate"]["candidate_count"])
    if len(plate_rows) != expected_candidates:
        raise RuntimeError(f"Expected {expected_candidates} plate candidates, got {len(plate_rows)}")
    if set(scene_counts.values()) != {int(config["plate_variants_per_scene"])}:
        raise RuntimeError(f"Unbalanced plate prompt grid: {scene_counts}")
    if len({row["prompt"] for row in plate_rows}) != len(plate_rows):
        raise RuntimeError("Background prompt grid contains duplicates")

    output_root = helper.repo_path(config["output_root"])
    sheet_root = output_root / "source_review_sheets"
    for report in source_reports:
        class_id = int(report["class_id"])
        make_source_sheets(
            source_rows_for_sheets[class_id],
            sheet_root / f"c{class_id:02d}_{safe_name(report['class_name'])}.png",
            f"c{class_id:02d} {report['class_name']} | {len(source_rows_for_sheets[class_id])} candidates",
        )

    source_review_path = output_root / "source_review.csv"
    with source_review_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "class_id", "class_name", "candidate_index", "asset_id", "stem",
            "rgba_path", "review_status", "semantic_form", "pose_mode", "review_reason",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for report in source_reports:
            class_id = int(report["class_id"])
            for candidate_index, row in enumerate(source_rows_for_sheets[class_id]):
                writer.writerow({
                    "class_id": class_id,
                    "class_name": report["class_name"],
                    "candidate_index": candidate_index,
                    "asset_id": row["asset_id"],
                    "stem": row["stem"],
                    "rgba_path": row["rgba_path"],
                    "review_status": "pending",
                    "semantic_form": "",
                    "pose_mode": "",
                    "review_reason": "",
                })

    result = {
        "status": "passed",
        "training_use_forbidden": True,
        "config": str(config_path.relative_to(REPO_ROOT)),
        "background_candidates": len(plate_rows),
        "background_candidates_per_scene": dict(sorted(scene_counts.items())),
        "background_acceptance_target": int(config["background_review_gate"]["required_accepted_count"]),
        "source_projection": source_reports,
        "source_candidates_per_class": {
            report["class_name"]: len(source_rows_for_sheets[int(report["class_id"])])
            for report in source_reports
        },
        "source_required_accepted_per_class": required_accepted_per_class,
        "source_review_csv": str(source_review_path.relative_to(REPO_ROOT)),
        "source_review_sheets": str(sheet_root.relative_to(REPO_ROOT)),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "input_diversity_preflight.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
