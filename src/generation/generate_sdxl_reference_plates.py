"""Generate review-gated SDXL img2img plates from approved real supports.

Low transformation strengths preserve the perspective and support geometry of
human-reviewed Places365 backgrounds. Outputs remain experiment-only until a
new review explicitly accepts their realism, geometry, and target-free status.
Target-test images are never inputs to this stage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/generation/experiments/sdxl_reference_plate_pilot_v1.yaml"
HELPERS = REPO_ROOT / "src/generation/run_all_class_genai_pilot.py"


def load_helpers():
    spec = importlib.util.spec_from_file_location("generation_helpers", HELPERS)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load generation helpers: {HELPERS}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def select_sources(helper, config: dict[str, Any]) -> list[dict[str, Any]]:
    source = config["source"]
    eligible_path = helper.repo_path(source["eligible_backgrounds"])
    support_path = helper.repo_path(source["reviewed_support_regions"])
    required = str(source["required_status"])
    eligible = {
        (row["background_path"], support_type): row
        for row in read_csv(eligible_path)
        if row["eligibility_status"] == required
        for support_type in row["support_types"].split(";")
    }
    reviewed = {
        (row["background_path"], row["support_type"]): row
        for row in read_csv(support_path)
        if row["review_status"] == required and row["region_path"]
    }
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    count = int(source["samples_per_support_type"])
    selection_seed = int(source["selection_seed"])
    for support_type in source["support_types"]:
        candidates = []
        for key, eligibility in eligible.items():
            if key[1] != support_type or key not in reviewed:
                continue
            candidates.append((eligibility, reviewed[key]))
        candidates.sort(key=lambda pair: hashlib.sha256(
            f"{selection_seed}:{support_type}:{pair[0]['background_sha256']}".encode()
        ).digest())
        if len(candidates) < count:
            raise ValueError(f"Only {len(candidates)} approved sources for {support_type}; need {count}")
        for eligibility, support in candidates[:count]:
            background = helper.repo_path(eligibility["background_path"])
            region = helper.repo_path(support["region_path"])
            if eligibility["background_sha256"] != helper.sha256(background):
                raise ValueError(f"Background hash mismatch: {background}")
            if support["region_sha256"] != helper.sha256(region):
                raise ValueError(f"Support-region hash mismatch: {region}")
            if eligibility["background_sha256"] in seen:
                raise ValueError(f"Source reused across pilot: {background}")
            seen.add(eligibility["background_sha256"])
            records.append({
                "index": len(records), "support_type": support_type,
                "source": eligibility["background_path"],
                "source_sha256": eligibility["background_sha256"],
                "source_category": support["category"],
                "source_region": support["region_path"],
                "source_region_sha256": support["region_sha256"],
                "source_anchor_points_xy": json.loads(support["anchor_points_xy"]),
                "source_region_area_ratio": float(support["region_area_ratio"]),
            })
    max_reuse = int(source["maximum_source_reuse"])
    reuse_counts = {digest: sum(row["source_sha256"] == digest for row in records) for digest in seen}
    if max(reuse_counts.values(), default=0) > max_reuse:
        raise ValueError(f"maximum_source_reuse={max_reuse} violated")
    return records


def validate_config(config: dict[str, Any], records: list[dict[str, Any]]) -> None:
    if config.get("status") != "ready_for_experiment":
        raise RuntimeError("Reference-plate pilot is not launch-ready")
    generation = config["generation"]
    strengths = [float(value) for value in generation["strengths"]]
    if not strengths or not all(0.0 < value <= 0.35 for value in strengths):
        raise ValueError("Reference strengths must stay in the geometry-preserving range (0, 0.35]")
    if set(generation["prompts"]) != set(config["source"]["support_types"]):
        raise ValueError("Exactly one prompt is required for every support type")
    # A conservative word-count guard catches obviously overlong prompts in
    # CPU-only preflight; exact CLIP token lengths are checked after loading.
    for prompt in [*generation["prompts"].values(), generation["negative_prompt"]]:
        if len(str(prompt).split()) > 55:
            raise ValueError("Prompt exceeds the 55-word preflight safety limit")
    expected = len(config["source"]["support_types"]) * int(config["source"]["samples_per_support_type"])
    if len(records) != expected:
        raise ValueError(f"Expected {expected} records, got {len(records)}")


def load_pipeline(models: dict[str, Any], gpu: int):
    from diffusers import StableDiffusionXLImg2ImgPipeline

    base = models["sdxl"]["base_model"]
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        base["id"], revision=base["revision"], torch_dtype=torch.float16,
        variant=models["sdxl"].get("variant"), use_safetensors=True,
    )
    pipe.enable_model_cpu_offload(gpu_id=gpu)
    pipe.vae.enable_slicing()
    return pipe


def validate_prompts(pipe, config: dict[str, Any]) -> None:
    prompts = [*config["generation"]["prompts"].values(), config["generation"]["negative_prompt"]]
    for name in ("tokenizer", "tokenizer_2"):
        tokenizer = getattr(pipe, name)
        limit = int(tokenizer.model_max_length)
        for prompt in prompts:
            length = len(tokenizer(prompt, truncation=False)["input_ids"])
            if length > limit:
                raise ValueError(f"{name} prompt length {length} exceeds {limit}: {prompt}")


def crop_to_support(
    source: Image.Image, region: Image.Image, crop_config: dict[str, Any] | None,
) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int]]:
    """Return a square context crop centered on the reviewed support region."""
    full_box = (0, 0, source.width, source.height)
    if not crop_config or not crop_config.get("enabled", False):
        return source, region, full_box
    bbox = region.getbbox()
    if bbox is None:
        raise ValueError("Reviewed support region is empty")
    support_width, support_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    side = math.ceil(max(
        max(support_width, support_height) * float(crop_config["context_scale"]),
        min(source.size) * float(crop_config["minimum_source_fraction"]),
    ))
    side = min(side, source.width, source.height)
    center_x, center_y = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    left = max(0, min(round(center_x - side / 2), source.width - side))
    top = max(0, min(round(center_y - side / 2), source.height - side))
    crop_box = (left, top, left + side, top + side)
    return source.crop(crop_box), region.crop(crop_box), crop_box


def contact_sheet(items: list[tuple[Path, str]], output: Path) -> None:
    thumb, caption, columns = 256, 34, 4
    rows = (len(items) + columns - 1) // columns
    canvas = Image.new("RGB", (thumb * columns, (thumb + caption) * rows), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (path, label) in enumerate(items):
        with Image.open(path).convert("RGB") as source:
            source.thumbnail((thumb, thumb))
            x = index % columns * thumb
            y = index // columns * (thumb + caption)
            canvas.paste(source, (x, y))
        draw.text((x + 4, y + thumb + 4), label, fill="black")
    canvas.save(output)


def generate(
    helper, config: dict[str, Any], config_path: Path, models: dict[str, Any],
    records: list[dict[str, Any]], gpu: int,
) -> None:
    if not torch.cuda.is_available() or not 0 <= gpu < torch.cuda.device_count():
        raise RuntimeError(f"Logical CUDA device {gpu} is unavailable")
    output_dir = helper.repo_path(config["output_root"])
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    (output_dir / "images").mkdir(parents=True)
    (output_dir / "support_masks").mkdir()
    (output_dir / "source_crops").mkdir()
    pipe = load_pipeline(models, gpu)
    validate_prompts(pipe, config)
    generation = config["generation"]
    width, height = int(generation["width"]), int(generation["height"])
    strengths = [float(value) for value in generation["strengths"]]
    generated = []
    for position, record in enumerate(records, start=1):
        source_path = helper.repo_path(record["source"])
        region_path = helper.repo_path(record["source_region"])
        with Image.open(source_path).convert("RGB") as raw_source:
            source_size = raw_source.size
            source = raw_source.copy()
        with Image.open(region_path).convert("L") as raw_region:
            if raw_region.size != source_size:
                raise ValueError(f"Region/source size mismatch for {source_path}")
            region_source = raw_region.copy()
        source, region_source, crop_box = crop_to_support(
            source, region_source, generation.get("support_crop")
        )
        initial = source.resize((width, height), Image.Resampling.LANCZOS)
        region = region_source.resize((width, height), Image.Resampling.NEAREST)
        strength = strengths[record["index"] % len(strengths)]
        seed = int(generation["seed"]) + record["index"]
        started = time.monotonic()
        image = pipe(
            prompt=generation["prompts"][record["support_type"]],
            negative_prompt=generation["negative_prompt"], image=initial,
            strength=strength, width=width, height=height,
            num_inference_steps=int(generation["inference_steps"]),
            guidance_scale=float(generation["guidance_scale"]),
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).images[0]
        torch.cuda.synchronize(gpu)
        stem = f"r{record['index']:03d}_{record['support_type']}"
        image_path = output_dir / "images" / f"{stem}.png"
        mask_path = output_dir / "support_masks" / f"{stem}.png"
        crop_path = output_dir / "source_crops" / f"{stem}.png"
        image.save(image_path); region.save(mask_path); initial.save(crop_path)
        sx, sy = width / source.width, height / source.height
        anchors = [
            [round((x - crop_box[0]) * sx), round((y - crop_box[1]) * sy)]
            for x, y in record["source_anchor_points_xy"]
            if crop_box[0] <= x < crop_box[2] and crop_box[1] <= y < crop_box[3]
        ]
        source_pixels = np.asarray(initial, dtype=np.float32)
        output_pixels = np.asarray(image, dtype=np.float32)
        normalized_mae = float(np.abs(output_pixels - source_pixels).mean() / 255.0)
        generated.append({
            **record, "source_size": list(source_size), "source_crop_xyxy": list(crop_box),
            "source_crop_size": list(source.size), "output_size": [width, height],
            "seed": seed, "strength": strength,
            "prompt": generation["prompts"][record["support_type"]],
            "negative_prompt": generation["negative_prompt"],
            "output": str(image_path.relative_to(REPO_ROOT)), "output_sha256": helper.sha256(image_path),
            "source_crop": str(crop_path.relative_to(REPO_ROOT)),
            "source_crop_sha256": helper.sha256(crop_path),
            "support_mask": str(mask_path.relative_to(REPO_ROOT)), "support_mask_sha256": helper.sha256(mask_path),
            "anchor_points_xy": anchors, "normalized_source_output_mae": round(normalized_mae, 6),
            "inference_seconds": round(time.monotonic() - started, 3),
            "training_use_forbidden": True,
        })
        print(f"[reference plate {position}/{len(records)}] {image_path.name}", flush=True)
    paired_items = []
    for row in generated:
        paired_items.extend([
            (helper.repo_path(row["source_crop"]), f"SOURCE {row['index']:03d} {row['support_type']}"),
            (helper.repo_path(row["output"]), f"OUTPUT {row['index']:03d} s={row['strength']:.2f}"),
        ])
    contact_sheet(paired_items, output_dir / "source_output_contact_sheet.png")
    contact_sheet(
        [(helper.repo_path(row["output"]), f"{row['index']:03d} {row['support_type']} s={row['strength']:.2f}") for row in generated],
        output_dir / "contact_sheet.png",
    )
    write_json(output_dir / "manifest.json", {
        "status": "generated_pending_human_review", "training_use_forbidden": True,
        "target_test_used": False, "source_dataset": config["source"]["dataset_id"],
        "source_dataset_revision": config["source"]["dataset_revision"],
        "config": str(config_path.relative_to(REPO_ROOT)),
        "config_sha256": helper.sha256(config_path),
        "model_id": models["sdxl"]["base_model"]["id"],
        "model_revision": models["sdxl"]["base_model"]["revision"],
        "review_gate": config["review_gate"], "records": generated,
    })
    fields = ["index", "support_type", "image_path", "review_status", "organic_realism",
              "physical_geometry", "source_geometry_preserved", "support_surface_preserved",
              "meaningful_source_difference", "target_free", "duplicate_free", "review_reason"]
    with (output_dir / "review.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in generated:
            writer.writerow({"index": row["index"], "support_type": row["support_type"],
                             "image_path": row["output"], "review_status": "pending",
                             **{field: "pending" for field in fields[4:-1]}, "review_reason": ""})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    helper = load_helpers()
    config_path = helper.repo_path(args.config)
    config = helper.load_yaml(config_path)
    records = select_sources(helper, config)
    validate_config(config, records)
    report = {
        "status": "ready", "samples": len(records),
        "counts": {kind: sum(row["support_type"] == kind for row in records)
                   for kind in config["source"]["support_types"]},
        "unique_sources": len({row["source_sha256"] for row in records}),
        "target_test_used": False, "config_sha256": helper.sha256(config_path),
    }
    if args.preflight_only:
        print(json.dumps(report, indent=2)); return
    models = helper.load_yaml(helper.repo_path(config["models_config"]))
    generate(helper, config, config_path, models, records, args.gpu)


if __name__ == "__main__":
    main()
