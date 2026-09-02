from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/generation/genai_shared_v1.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "data/processed/genai_v1"


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def expand_bbox(
    bbox: list[int], width: int, height: int, margin_ratio: float, minimum_margin: int
) -> list[int]:
    x1, y1, x2, y2 = map(int, bbox)
    margin_x = max(minimum_margin, round((x2 - x1) * margin_ratio))
    margin_y = max(minimum_margin, round((y2 - y1) * margin_ratio))
    return [
        max(0, x1 - margin_x),
        max(0, y1 - margin_y),
        min(width, x2 + margin_x),
        min(height, y2 + margin_y),
    ]


def normalized_bbox(bbox: list[int], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = bbox
    return [x1 / width, y1 / height, x2 / width, y2 / height]


PROMPT_OBJECTS = {
    "Lighter": "a cigarette lighter",
    "Matches": "a box of matches",
    "Scissors": "a pair of scissors",
    "Pliers": "a pair of pliers",
    "Knife": "a knife",
    "Shaver": "an electric shaver",
    "Hammer": "a hammer",
    "Cigarettes": "a pack of cigarettes",
    "Saw": "a hand saw",
    "Screwdriver": "a screwdriver",
    "Wrench": "a wrench",
    "Aerosol can": "an aerosol can",
    "Battery": "a battery",
    "Alcohol": "a bottle of alcohol",
    "Mobile phone": "a mobile phone",
    "Laptop": "a laptop computer",
}


def class_prompt(class_name: str, support_type: str) -> str:
    support = "the bed" if support_type == "bed_top" else "a dining table"
    object_phrase = PROMPT_OBJECTS[class_name]
    return (
        f"A realistic indoor surveillance-camera photograph containing exactly "
        f"{object_phrase}, naturally placed on {support}. Keep the room, camera viewpoint, and all "
        "content outside the masked region unchanged. The object is fully visible, "
        "photorealistic, correctly scaled, and contains no text or watermark."
    )


def build_plan(config_path: Path, output_root: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    metadata_path = repo_path(config["source"]["copy_paste_metadata"])
    generation_config_path = repo_path(
        config["source"]["copy_paste_generation_config"]
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = int(config["allocation"]["canonical_images"])
    if len(metadata) != expected:
        raise ValueError(f"Expected {expected} source records, found {len(metadata)}")

    data_config = load_yaml(REPO_ROOT / "configs/data_insp.yaml")
    class_names = {int(key): value for key, value in data_config["names"].items()}
    margin_ratio = float(config["conditioning"]["bbox_margin_ratio"])
    minimum_margin = int(
        config["conditioning"]["minimum_bbox_margin_source_pixels"]
    )
    sd_offset = int(config["seeds"]["stable_diffusion_offset"])
    qwen_offset = int(config["seeds"]["qwen_offset"])

    records: list[dict[str, Any]] = []
    for expected_id, source in enumerate(metadata):
        image_id = int(source["image_id"])
        if image_id != expected_id:
            raise ValueError(
                f"Source metadata ordering mismatch: expected {expected_id}, got {image_id}"
            )
        class_id = int(source["primary_class_id"])
        width = int(source["background_width"])
        height = int(source["background_height"])
        intended_bbox = [int(value) for value in source["object"]["bbox_xyxy"]]
        inpaint_bbox = expand_bbox(
            intended_bbox, width, height, margin_ratio, minimum_margin
        )
        background_path = repo_path(source["background"])
        object_path = repo_path(source["object"]["object_path"])
        if not background_path.is_file() or not object_path.is_file():
            raise FileNotFoundError(
                f"Missing source asset for image_id={image_id}: "
                f"{background_path}, {object_path}"
            )
        if sha256(background_path) != source["background_sha256"]:
            raise ValueError(f"Background hash mismatch for image_id={image_id}")
        if sha256(object_path) != source["object"]["object_sha256"]:
            raise ValueError(f"Object hash mismatch for image_id={image_id}")

        records.append(
            {
                "image_id": image_id,
                "class_id": class_id,
                "class_name": class_names[class_id],
                "background": source["background"],
                "background_sha256": source["background_sha256"],
                "object_rgba": source["object"]["object_path"],
                "object_sha256": source["object"]["object_sha256"],
                "conditioning_source": "reconstructed_undegraded_copy_paste_composite",
                "support_type": source["support_type"],
                "support_region": source["support_region"],
                "support_region_sha256": source["support_region_sha256"],
                "source_width": width,
                "source_height": height,
                "intended_bbox_xyxy": intended_bbox,
                "intended_bbox_normalized_xyxy": normalized_bbox(
                    intended_bbox, width, height
                ),
                "inpaint_bbox_xyxy": inpaint_bbox,
                "inpaint_bbox_normalized_xyxy": normalized_bbox(
                    inpaint_bbox, width, height
                ),
                "prompt": class_prompt(class_names[class_id], source["support_type"]),
                "negative_prompt": (
                    "multiple objects, duplicate object, wrong object, deformed, floating, "
                    "cropped, text, watermark, logo, illustration, cartoon"
                ),
                "stable_diffusion_seed": sd_offset + image_id,
                "qwen_seed": qwen_offset + image_id,
                "degradation_severity": source["degradation_severity"],
                "degradations": source["degradations"],
                "final_annotation_policy": "sam3_mask_after_generation",
            }
        )

    classes = int(config["allocation"]["classes"])
    per_class = int(config["allocation"]["images_per_class"])
    for prefix in config["allocation"]["nested_prefixes"]:
        counts = Counter(record["class_id"] for record in records[: int(prefix)])
        expected_per_class = int(prefix) // classes
        if counts != Counter({class_id: expected_per_class for class_id in range(classes)}):
            raise ValueError(f"Class balance failed at prefix {prefix}: {counts}")
    full_counts = Counter(record["class_id"] for record in records)
    if full_counts != Counter({class_id: per_class for class_id in range(classes)}):
        raise ValueError(f"Canonical class balance failed: {full_counts}")

    pilot_per_class = int(config["allocation"]["pilot_images_per_class"])
    pilot_ids: list[int] = []
    pilot_counts: Counter[int] = Counter()
    for record in records:
        class_id = int(record["class_id"])
        if pilot_counts[class_id] < pilot_per_class:
            pilot_ids.append(int(record["image_id"]))
            pilot_counts[class_id] += 1
    if len(pilot_ids) != classes * pilot_per_class:
        raise ValueError(f"Pilot selection incomplete: {pilot_counts}")

    output_root.mkdir(parents=True, exist_ok=True)
    plan_path = output_root / "shared_conditioning_plan.jsonl"
    plan_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    pilot_path = output_root / "pilot_image_ids.json"
    pilot_path.write_text(json.dumps(pilot_ids, indent=2) + "\n", encoding="utf-8")
    summary = {
        "format_version": 1,
        "status": "shared_plan_ready_backend_smoke_tests_pending",
        "config": config_path.relative_to(REPO_ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "source_metadata": metadata_path.relative_to(REPO_ROOT).as_posix(),
        "source_metadata_sha256": sha256(metadata_path),
        "source_generation_config_sha256": sha256(generation_config_path),
        "records": len(records),
        "class_counts": dict(sorted(full_counts.items())),
        "nested_prefixes": config["allocation"]["nested_prefixes"],
        "pilot_records": len(pilot_ids),
        "pilot_class_counts": dict(sorted(pilot_counts.items())),
        "plan": plan_path.relative_to(REPO_ROOT).as_posix(),
        "plan_sha256": sha256(plan_path),
        "pilot_ids": pilot_path.relative_to(REPO_ROOT).as_posix(),
        "pilot_ids_sha256": sha256(pilot_path),
    }
    summary_path = output_root / "shared_conditioning_plan_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the paired SDXL/Qwen GenAI conditioning plan"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build_plan(repo_path(args.config), repo_path(args.output)), indent=2))


if __name__ == "__main__":
    main()
