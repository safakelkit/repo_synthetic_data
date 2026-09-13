"""Harmonize diverse real-support composites with localized SDXL inpainting.

The initialization stage supplies reviewed real backgrounds, train-bounded
object sizes, source-image-balanced identities, support-valid placement, and
YOLO labels. This stage changes only a padded target region. Outputs remain
training-forbidden until visual and structural review succeeds.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/generation/sdxl_hybrid_initialization_512_v1.yaml"
COPYPASTE_GENERATOR = REPO_ROOT / "src/augmentation/generate_copypaste_dataset.py"


def load_copypaste_module():
    spec = importlib.util.spec_from_file_location("copypaste_generator", COPYPASTE_GENERATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {COPYPASTE_GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_initializations(cp, config: dict[str, Any]) -> list[dict[str, Any]]:
    root = cp.repo_path(config["production"]["output_root"])
    metadata_path = root / "metadata.json"
    generation_path = root / "generation_config.json"
    if not metadata_path.is_file() or not generation_path.is_file():
        raise FileNotFoundError("Generate the hybrid initialization dataset before harmonization")
    records = read_json(metadata_path)
    generation = read_json(generation_path)
    expected = int(config["production"]["num_images"])
    if len(records) != expected or generation["saved_images"] != expected:
        raise ValueError(f"Expected {expected} complete initializations")
    if generation["object_asset_sampling_unit"] != "source_image":
        raise ValueError("Hybrid initializations must balance originating train images")
    if generation["background_sampling"] != "balanced_low_reuse":
        raise ValueError("Hybrid initializations must use balanced backgrounds")
    if not generation["compositing_masks_saved"]:
        raise ValueError("Hybrid initializations require exact compositing masks")
    source_counts: dict[int, Counter[str]] = {}
    for record in records:
        class_id = int(record["primary_class_id"])
        source_counts.setdefault(class_id, Counter())[record["object_source_image"]] += 1
        for key in ("image", "label", "compositing_mask", "support_region"):
            if not cp.repo_path(record[key]).is_file():
                raise FileNotFoundError(f"Missing initialization input: {record[key]}")
        if cp.file_sha256(cp.repo_path(record["image"])) != record["image_sha256"]:
            raise ValueError(f"Initialization image hash mismatch: {record['image']}")
        if cp.file_sha256(cp.repo_path(record["compositing_mask"])) != record["compositing_mask_sha256"]:
            raise ValueError(f"Compositing-mask hash mismatch: {record['compositing_mask']}")
    expected_per_class = expected // 16
    if set(Counter(int(row["primary_class_id"]) for row in records).values()) != {expected_per_class}:
        raise ValueError("Initialization classes are not exactly balanced")
    if min(len(counts) for counts in source_counts.values()) != expected_per_class:
        raise ValueError("A source train image repeats inside the 512-image experiment")
    return records


def load_pipeline(models: dict[str, Any], gpu: int):
    from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline

    base = models["sdxl"]["base_model"]
    control = models["sdxl"]["controlnet"]
    controlnet = ControlNetModel.from_pretrained(
        control["id"], revision=control["revision"], torch_dtype=torch.float16,
        variant="fp16", use_safetensors=True,
    )
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        base["id"], revision=base["revision"], controlnet=controlnet,
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    )
    pipe.enable_model_cpu_offload(gpu_id=gpu)
    pipe.vae.enable_slicing()
    return pipe


def validate_prompts(pipe, prompts: list[str]) -> None:
    for name in ("tokenizer", "tokenizer_2"):
        tokenizer = getattr(pipe, name)
        limit = int(tokenizer.model_max_length)
        for prompt in prompts:
            length = len(tokenizer(prompt, truncation=False)["input_ids"])
            if length > limit:
                raise ValueError(f"{name} prompt length {length} exceeds {limit}: {prompt}")


def target_controls(mask: Image.Image, padding: int) -> tuple[Image.Image, Image.Image]:
    binary = np.where(np.asarray(mask.convert("L")) > 8, 255, 0).astype(np.uint8)
    canny = cv2.Canny(cv2.GaussianBlur(binary, (0, 0), sigmaX=0.8), 32, 96)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (padding * 2 + 1, padding * 2 + 1))
    inpaint = cv2.dilate(binary, kernel)
    inpaint = cv2.GaussianBlur(inpaint, (0, 0), sigmaX=4)
    return Image.fromarray(inpaint), Image.fromarray(cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB))


def make_contact_sheet(records: list[dict[str, Any]], output: Path) -> None:
    selected = records[:32]
    thumb, caption, samples_per_row = 320, 42, 2
    sample_width = thumb * 2
    rows = (len(selected) + samples_per_row - 1) // samples_per_row
    sheet = Image.new("RGB", (sample_width * samples_per_row, (thumb + caption) * rows), "white")
    draw = ImageDraw.Draw(sheet)
    for index, record in enumerate(selected):
        x = index % samples_per_row * sample_width
        y = index // samples_per_row * (thumb + caption)
        with Image.open(REPO_ROOT / record["source_initialization"]).convert("RGB") as source:
            sheet.paste(source.resize((thumb, thumb), Image.Resampling.LANCZOS), (x, y))
        with Image.open(REPO_ROOT / record["output"]).convert("RGB") as result:
            result = result.resize((thumb, thumb), Image.Resampling.LANCZOS)
            label = (REPO_ROOT / record["label"]).read_text(encoding="utf-8").strip().split()
            if len(label) == 5:
                _, cx, cy, width, height = label
                cx, cy, width, height = map(float, (cx, cy, width, height))
                box = (
                    int((cx - width / 2) * thumb), int((cy - height / 2) * thumb),
                    int((cx + width / 2) * thumb), int((cy + height / 2) * thumb),
                )
                ImageDraw.Draw(result).rectangle(box, outline=(0, 255, 0), width=3)
            sheet.paste(result, (x + thumb, y))
        draw.text(
            (x + 4, y + thumb + 4),
            f"c{record['class_id']:02d} {record['support_type']} | init / SDXL + label",
            fill="black",
        )
    sheet.save(output)


def harmonize(
    cp, config: dict[str, Any], config_path: Path, initializations: list[dict[str, Any]],
    gpu: int, limit: int, output_name: str,
) -> None:
    if not torch.cuda.is_available() or not 0 <= gpu < torch.cuda.device_count():
        raise RuntimeError(f"Logical CUDA device {gpu} is unavailable")
    harmonization = config["harmonization"]
    output_root = cp.repo_path(harmonization["output_root"]) / output_name
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")
    images_dir, labels_dir, controls_dir = (
        output_root / "images", output_root / "labels", output_root / "controls"
    )
    images_dir.mkdir(parents=True); labels_dir.mkdir(); controls_dir.mkdir()
    models = cp.load_generation_config(cp.repo_path(harmonization["model_config"]))
    pipe = load_pipeline(models, gpu)
    width, height = map(int, harmonization["output_size"])
    prompts_by_class = {int(key): value for key, value in harmonization["class_prompts"].items()}
    poses = {
        "lying": "lying flat and fully supported",
        "upright": "upright with its base touching the surface",
        "asset_preserved_bottom_contact": "open with its lower base fully supported",
    }
    prompts = [
        harmonization["prompt_template"].format(
            target=prompts_by_class[class_id], pose=pose, support=support.replace("_", " ")
        )
        for class_id in sorted(prompts_by_class)
        for pose in poses.values()
        for support in ("bed_top", "dining_table_top")
    ]
    validate_prompts(pipe, [*prompts, harmonization["negative_prompt"]])
    strengths = {int(key): float(value) for key, value in harmonization["strength_by_class"].items()}
    canny_scales = {int(key): float(value) for key, value in harmonization["canny_scale_by_class"].items()}
    records = []
    for position, source_record in enumerate(initializations[:limit], start=1):
        class_id = int(source_record["primary_class_id"])
        with Image.open(cp.repo_path(source_record["image"])).convert("RGB") as source:
            initial = source.resize((width, height), Image.Resampling.LANCZOS)
        with Image.open(cp.repo_path(source_record["compositing_mask"])).convert("L") as source:
            alpha = source.resize((width, height), Image.Resampling.NEAREST)
        mask, canny = target_controls(alpha, int(harmonization["mask_padding_px"]))
        prompt = harmonization["prompt_template"].format(
            target=prompts_by_class[class_id], pose=poses[source_record["placement_mode"]],
            support=source_record["support_type"].replace("_", " "),
        )
        seed = int(config["seed"]) + int(harmonization["seed_offset"]) + int(source_record["image_id"])
        strength = strengths.get(class_id, float(harmonization["default_strength"]))
        canny_scale = canny_scales.get(class_id, float(harmonization["default_canny_scale"]))
        started = time.monotonic()
        image = pipe(
            prompt=prompt, negative_prompt=harmonization["negative_prompt"],
            image=initial, mask_image=mask, control_image=canny,
            strength=strength, width=width, height=height,
            num_inference_steps=int(harmonization["inference_steps"]),
            guidance_scale=float(harmonization["guidance_scale"]),
            controlnet_conditioning_scale=canny_scale,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).images[0]
        torch.cuda.synchronize(gpu)
        stem = f"hybrid_{source_record['image_id']:06d}"
        image_path = images_dir / f"{stem}.png"
        label_path = labels_dir / f"{stem}.txt"
        mask_path = controls_dir / f"{stem}_mask.png"
        canny_path = controls_dir / f"{stem}_canny.png"
        image.save(image_path); mask.save(mask_path); canny.save(canny_path)
        label_path.write_text(cp.repo_path(source_record["label"]).read_text(encoding="utf-8"), encoding="utf-8")
        records.append({
            "index": int(source_record["image_id"]), "class_id": class_id,
            "source_initialization": source_record["image"],
            "source_initialization_sha256": source_record["image_sha256"],
            "object_source_image": source_record["object_source_image"],
            "object_source_rgba": source_record["object"]["object_path"],
            "background": source_record["background"], "support_type": source_record["support_type"],
            "target_area": source_record["object"]["realized_normalized_area"],
            "rotation_degrees": source_record["rotation_degrees"], "seed": seed,
            "strength": strength, "canny_scale": canny_scale, "prompt": prompt,
            "output": str(image_path.relative_to(REPO_ROOT)), "output_sha256": cp.file_sha256(image_path),
            "label": str(label_path.relative_to(REPO_ROOT)), "mask": str(mask_path.relative_to(REPO_ROOT)),
            "canny": str(canny_path.relative_to(REPO_ROOT)),
            "inference_seconds": round(time.monotonic() - started, 3),
            "training_use_forbidden": True,
        })
        print(f"[hybrid {position}/{limit}] {stem} c{class_id:02d}", flush=True)
    make_contact_sheet(records, output_root / "contact_sheet.png")
    write_json(output_root / "manifest.json", {
        "status": "generated_pending_human_review", "training_use_forbidden": True,
        "target_test_used": False, "config": str(config_path.relative_to(REPO_ROOT)),
        "config_sha256": cp.file_sha256(config_path), "records": records,
    })
    fields = ["index", "class_id", "image_path", "review_status", "object_identity",
              "organic_background", "physical_contact", "annotation_box", "target_count",
              "artifact_free", "review_reason"]
    with (output_root / "review.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in records:
            writer.writerow({"index": row["index"], "class_id": row["class_id"],
                             "image_path": row["output"], "review_status": "pending",
                             **{field: "pending" for field in fields[4:-1]}, "review_reason": ""})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--output-name", default="pilot_16")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    cp = load_copypaste_module()
    config_path = cp.repo_path(args.config)
    config = cp.load_generation_config(config_path)
    records = validate_initializations(cp, config)
    if not 1 <= args.limit <= len(records):
        raise ValueError(f"--limit must be in [1, {len(records)}]")
    selected = records[:args.limit]
    report = {
        "status": "ready", "samples": len(selected),
        "class_counts": dict(sorted(Counter(row["primary_class_id"] for row in selected).items())),
        "unique_backgrounds": len({row["background_sha256"] for row in selected}),
        "unique_object_source_images": len({row["object_source_image"] for row in selected}),
        "target_test_used": False,
    }
    if args.preflight_only:
        print(json.dumps(report, indent=2)); return
    harmonize(cp, config, config_path, records, args.gpu, args.limit, args.output_name)


if __name__ == "__main__":
    main()
