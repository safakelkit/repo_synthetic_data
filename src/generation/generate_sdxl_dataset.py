"""Generate the frozen SDXL acceptance dataset.

The established classes use full-scene Canny generation. Aerosol uses a
target-free scene plate followed by source-initialized, Canny-controlled
inpainting so its appearance and physical scale remain recognizable.
"""

from __future__ import annotations

import argparse
import csv
import copy
import gc
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/generation/sdxl_generation.yaml"
HELPERS = REPO_ROOT / "src/generation/run_all_class_genai_pilot.py"


def load_helpers():
    spec = importlib.util.spec_from_file_location("generation_helpers", HELPERS)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load generation helpers: {HELPERS}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def schedule(config: dict[str, Any], scenes: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for repeat in range(int(config["images_per_class_scene"])):
        for scene_slot in range(4):
            for class_id in range(16):
                rows.append({
                    "index": len(rows),
                    "class_id": class_id,
                    "class_name": scenes["class_policy"]["names"][class_id],
                    "scene_slot": scene_slot,
                    "scene_name": scenes["class_to_scene_families"][class_id][scene_slot],
                    "class_attempt_index": repeat * 4 + scene_slot,
                })
    if len(rows) != int(config["expected_images"]):
        raise ValueError(f"Expected {config['expected_images']} images, scheduled {len(rows)}")
    return rows


def ordered_mask_rows(config: dict[str, Any], rows: list[dict[str, str]], class_id: int):
    """Order one class pool according to the configured deterministic policy."""
    source_config = config["silhouette_source"]
    strategy = str(source_config.get("selection_strategy", "preferred_or_sequential_cycle"))
    preferred_by_class = source_config.get("preferred_stems") or {}
    preferred = preferred_by_class.get(class_id)
    if strategy == "preferred_or_sequential_cycle" and preferred:
        ordered = []
        for stem in preferred:
            matches = [row for row in rows if row["stem"] == stem]
            if len(matches) != 1:
                raise ValueError(f"Preferred silhouette is not uniquely accepted: {stem}")
            ordered.append(matches[0])
        return ordered
    offset = int(source_config["selection_seed_offset"])
    if strategy == "preferred_or_sequential_cycle":
        start = (int(config["seed"]) + offset) % len(rows)
        return rows[start:] + rows[:start]
    if strategy != "balanced_seeded_cycle":
        raise ValueError(f"Unknown silhouette selection strategy: {strategy}")

    # A stable hash permutation prevents manifest ordering from correlating with
    # class attempt, while cycling through the complete accepted class pool
    # before reusing an asset.  This keeps generation reproducible without
    # collapsing identity-preserving classes onto one preferred prototype.
    seed = int(config["seed"]) + offset + class_id
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{seed}:{row['asset_id']}".encode("utf-8")
        ).digest(),
    )


def select_mask(helper, config: dict[str, Any], manifest: Path, sample: dict[str, Any]):
    rows = helper.read_masks(manifest, sample["class_id"])
    ordered = ordered_mask_rows(config, rows, int(sample["class_id"]))
    return ordered[int(sample["class_attempt_index"]) % len(ordered)]


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key == "base_config":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_generation_config(helper, config_path: Path) -> tuple[dict[str, Any], Path | None]:
    def resolve(path: Path, ancestry: set[Path]) -> dict[str, Any]:
        resolved_path = path.resolve()
        if resolved_path in ancestry:
            chain = " -> ".join(str(item) for item in [*ancestry, resolved_path])
            raise ValueError(f"Cyclic generation-config inheritance: {chain}")
        override = helper.load_yaml(path)
        base_reference = override.get("base_config")
        if base_reference is None:
            return override
        base_path = helper.repo_path(base_reference)
        return merge_config(resolve(base_path, ancestry | {resolved_path}), override)

    override = helper.load_yaml(config_path)
    base_reference = override.get("base_config")
    return resolve(config_path, set()), helper.repo_path(base_reference) if base_reference else None


def release_pipeline(pipe) -> None:
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


def validate_clip_prompts(pipe, prompts: list[str]) -> None:
    for name in ("tokenizer", "tokenizer_2"):
        tokenizer = getattr(pipe, name)
        limit = int(tokenizer.model_max_length)
        for prompt in prompts:
            length = len(tokenizer(prompt, truncation=False)["input_ids"])
            if length > limit:
                raise ValueError(f"{name} prompt length {length} exceeds {limit}: {prompt}")


def load_base_pipeline(models: dict[str, Any], gpu: int):
    from diffusers import StableDiffusionXLPipeline

    model = models["sdxl"]
    base = model["base_model"]
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base["id"], revision=base["revision"], torch_dtype=torch.float16,
        variant=model["variant"], use_safetensors=True,
    )
    pipe.enable_model_cpu_offload(gpu_id=gpu)
    pipe.vae.enable_slicing()
    return pipe


def load_inpaint_pipeline(models: dict[str, Any], gpu: int):
    from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline

    model = models["sdxl"]
    base, control = model["base_model"], model["controlnet"]
    controlnet = ControlNetModel.from_pretrained(
        control["id"], revision=control["revision"], torch_dtype=torch.float16,
        variant=model["variant"], use_safetensors=True,
    )
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        base["id"], revision=base["revision"], controlnet=controlnet,
        torch_dtype=torch.float16, variant=model["variant"], use_safetensors=True,
    )
    pipe.enable_model_cpu_offload(gpu_id=gpu)
    pipe.vae.enable_slicing()
    return pipe


def keep_largest_canny_components(control: Image.Image, count: int) -> Image.Image:
    if count <= 0:
        return control
    array = np.asarray(control.convert("L"))
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (array > 127).astype(np.uint8), connectivity=8
    )
    ranked = sorted(
        range(1, component_count),
        key=lambda label: int(stats[label, cv2.CC_STAT_AREA]),
        reverse=True,
    )
    retained = np.isin(labels, ranked[:count]).astype(np.uint8) * 255
    return Image.fromarray(retained)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--class-id", type=int, action="append", dest="class_ids")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require num-shards >= 1 and a valid shard-index")

    helper = load_helpers()
    feasibility = helper.feasibility_module()
    config_path = helper.repo_path(args.config)
    config, base_config_path = load_generation_config(helper, config_path)
    models_path = helper.repo_path(config["models_config"])
    scenes_path = helper.repo_path(config["scene_policy"])
    manifest_path = helper.repo_path(config["silhouette_source"]["audit_manifest"])
    models = helper.load_yaml(models_path)
    scenes = helper.load_yaml(scenes_path)
    full_schedule = schedule(config, scenes)
    if args.class_ids:
        available = {int(row["class_id"]) for row in full_schedule}
        requested = set(args.class_ids)
        unknown = sorted(requested - available)
        if unknown:
            raise ValueError(f"Unknown class ids: {unknown}")
        full_schedule = [row for row in full_schedule if int(row["class_id"]) in requested]
    shard_schedule = [row for row in full_schedule if row["index"] % args.num_shards == args.shard_index]
    versions = feasibility.installed_versions()
    mismatches = {name: {"expected": expected, "actual": versions.get(name)} for name, expected in feasibility.EXPECTED_PACKAGES.items() if versions.get(name) != expected}
    if mismatches:
        raise RuntimeError(f"Package-version mismatch: {json.dumps(mismatches, indent=2)}")
    git = feasibility.git_state()
    if git["dirty"]:
        raise RuntimeError("Git worktree is dirty; commit the frozen pipeline before launch")
    output_root = helper.repo_path(config["output_root"])
    output_dir = output_root / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    preflight = {"status": "ready", "pipeline_id": config["pipeline_id"], "total": len(full_schedule), "shard_total": len(shard_schedule), "output": str(output_dir.relative_to(REPO_ROOT)), "git": git, "packages": versions, "config_sha256": helper.sha256(config_path), "base_config_sha256": helper.sha256(base_config_path) if base_config_path else None, "mask_manifest_sha256": helper.sha256(manifest_path)}
    if args.preflight_only:
        print(json.dumps(preflight, indent=2))
        return
    if config["status"] != "ready_for_acceptance_test":
        raise RuntimeError("Pipeline is not released for an acceptance test")
    if not torch.cuda.is_available() or not 0 <= args.gpu < torch.cuda.device_count():
        raise RuntimeError(f"Logical CUDA device {args.gpu} is unavailable")

    started_utc = feasibility.utc_now()
    started = time.monotonic()
    images_dir = output_dir / "images"
    controls_dir = output_dir / "controls"
    images_dir.mkdir(parents=True)
    controls_dir.mkdir()
    records = []
    review_rows = []
    modes = {int(key): value for key, value in config["generation_modes"]["class_overrides"].items()}
    scales = {int(key): value for key, value in config["controlnet_conditioning_scale"]["class_overrides"].items()}
    width, height = config["output_size"]
    aerosol_mode = "source_initialized_inpaint"
    standard_schedule = [
        sample for sample in shard_schedule
        if modes.get(sample["class_id"], config["generation_modes"]["default"]) != aerosol_mode
    ]
    aerosol_schedule = [
        sample for sample in shard_schedule
        if modes.get(sample["class_id"], config["generation_modes"]["default"]) == aerosol_mode
    ]
    completed = 0

    pipe = feasibility.load_pipeline("sdxl", models, args.gpu) if standard_schedule else None
    reference_config = config.get("reference_conditioning")
    reference_class_ids: set[int] = set()
    if reference_config and pipe is not None:
        reference_class_ids = {int(value) for value in reference_config["class_ids"]}
        adapter = reference_config["adapter"]
        pipe.load_ip_adapter(
            adapter["repo_id"], subfolder=adapter["subfolder"],
            weight_name=adapter["weight_name"], image_encoder_folder=adapter["image_encoder_folder"],
        )
        pipe.set_ip_adapter_scale(float(reference_config["scale"]))

    for sample in standard_schedule:
        mode = modes.get(sample["class_id"], config["generation_modes"]["default"])
        base_scene = scenes["scene_families"][sample["scene_name"]]
        context = helper.scene_prompt_context(
            config, sample["scene_name"], base_scene, sample["index"]
        )
        scene = {**base_scene, "description": context["description"]}
        if mode == "text_only_full_scene":
            control_config = {**config, "target_conditioning_mode": "scene_only"}
            _, control, condition = helper.build_control(control_config, None, sample["index"], sample["scene_name"])
            variants = config["aerosol_full_scene"]["prompt_variants"]
            prompt = variants[sample["class_attempt_index"] % len(variants)].format(
                scene=scene["description"].rstrip("."),
                background=context["description"].rstrip("."),
            )
            negative = config["aerosol_full_scene"]["negative_prompt"]
        else:
            row = select_mask(helper, config, manifest_path, sample)
            _, control, condition = helper.build_control(config, row, sample["index"], sample["scene_name"])
            prompt = helper.prompt_for(config, scene, sample["class_name"], sample["class_id"], sample["index"])
            negative = helper.negative_prompt_for(
                config, sample["class_id"], sample["scene_name"]
            )
        reference_image = None
        reference_metadata = None
        if sample["class_id"] in reference_class_ids:
            if mode == "text_only_full_scene":
                raise ValueError("Reference conditioning requires a selected object asset")
            reference_path = helper.repo_path(row["rgba_path"])
            with Image.open(reference_path).convert("RGBA") as source:
                reference_image = Image.new("RGBA", source.size, "white")
                reference_image.alpha_composite(source)
            reference_metadata = {
                "source_type": "sam3_rgba_ip_adapter_reference",
                "asset_id": row["asset_id"],
                "rgba_path": str(reference_path.relative_to(REPO_ROOT)),
                "rgba_sha256": helper.sha256(reference_path),
                "scale": float(reference_config["scale"]),
            }
        scale = float(scales.get(sample["class_id"], config["controlnet_conditioning_scale"]["default"]))
        seed = int(config["seed"]) + sample["index"]
        generator = torch.Generator(device="cpu").manual_seed(seed)
        inference_started = time.monotonic()
        call = {
            "prompt": prompt,
            "negative_prompt": negative,
            "image": control,
            "width": width,
            "height": height,
            "num_inference_steps": int(config["inference_steps"]),
            "guidance_scale": float(config["guidance_scale"]),
            "controlnet_conditioning_scale": scale,
            "generator": generator,
        }
        if reference_image is not None:
            call["ip_adapter_image"] = reference_image
        result = pipe(**call).images[0]
        torch.cuda.synchronize(args.gpu)
        name = f"g{sample['index']:05d}_c{sample['class_id']:02d}_{sample['scene_name']}.png"
        image_path = images_dir / name
        control_path = controls_dir / name.replace(".png", "_canny.png")
        result.save(image_path)
        control.save(control_path)
        record = {**sample, "generation_mode": mode, "seed": seed, "background_profile": context["profile"], "background_description": context["description"], "prompt": prompt, "negative_prompt": negative, "controlnet_conditioning_scale": scale, "condition": condition, "reference_condition": reference_metadata, "output": str(image_path.relative_to(REPO_ROOT)), "output_sha256": helper.sha256(image_path), "control_sha256": helper.sha256(control_path), "inference_seconds": round(time.monotonic() - inference_started, 3), "annotation_performed": False, "degradation_applied": False, "training_use_forbidden": True}
        records.append(record)
        review_rows.append({"index": sample["index"], "class_id": sample["class_id"], "class_name": sample["class_name"], "scene_name": sample["scene_name"], "generation_mode": mode, "image_path": record["output"], "review_status": "pending", "review_reason": ""})
        completed += 1
        print(f"[{completed}/{len(shard_schedule)}] {name} {record['inference_seconds']:.3f}s", flush=True)

    if pipe is not None:
        release_pipeline(pipe)
        pipe = None

    if aerosol_schedule:
        aerosol = config["aerosol_inpaint"]
        plates_dir = controls_dir / "aerosol_plates"
        plates_dir.mkdir()
        plate_pipe = load_base_pipeline(models, args.gpu)
        plate_prompts = [aerosol["backgrounds"][sample["scene_name"]]["prompt"] for sample in aerosol_schedule]
        validate_clip_prompts(plate_pipe, [*plate_prompts, aerosol["plate_negative_prompt"]])
        plate_metadata: dict[int, dict[str, Any]] = {}
        for sample in aerosol_schedule:
            background = aerosol["backgrounds"][sample["scene_name"]]
            plate_seed = int(background["seed"])
            plate_started = time.monotonic()
            plate = plate_pipe(
                prompt=background["prompt"], negative_prompt=aerosol["plate_negative_prompt"],
                width=width, height=height,
                num_inference_steps=int(aerosol["plate_inference_steps"]),
                guidance_scale=float(aerosol["plate_guidance_scale"]),
                generator=torch.Generator(device="cpu").manual_seed(plate_seed),
            ).images[0]
            torch.cuda.synchronize(args.gpu)
            plate_path = plates_dir / f"g{sample['index']:05d}_{sample['scene_name']}.png"
            plate.save(plate_path)
            plate_metadata[sample["index"]] = {
                "path": str(plate_path.relative_to(REPO_ROOT)),
                "sha256": helper.sha256(plate_path),
                "seed": plate_seed,
                "prompt": background["prompt"],
                "inference_seconds": round(time.monotonic() - plate_started, 3),
            }
            print(f"[aerosol plate] {plate_path.name}", flush=True)
        release_pipeline(plate_pipe)
        plate_pipe = None

        inpaint_pipe = load_inpaint_pipeline(models, args.gpu)
        aerosol_prompts = [
            aerosol["prompt_template"].format(scene=aerosol["scene_descriptions"][sample["scene_name"]])
            for sample in aerosol_schedule
        ]
        validate_clip_prompts(inpaint_pipe, [*aerosol_prompts, aerosol["negative_prompt"]])
        target_control_config = copy.deepcopy(config)
        target_control_config["target_conditioning_mode"] = "target_only"
        for sample in aerosol_schedule:
            row = select_mask(helper, config, manifest_path, sample)
            _, control, condition = helper.build_control(
                target_control_config, row, sample["index"], sample["scene_name"]
            )
            control = keep_largest_canny_components(
                control, int(aerosol["canny_keep_largest_components"])
            )
            plate_path = helper.repo_path(plate_metadata[sample["index"]]["path"])
            with Image.open(plate_path).convert("RGB") as source:
                plate = source.copy()
            render_x1, render_y1, render_x2, render_y2 = condition["rendered_box_xyxy"]
            rgba_path = helper.repo_path(row["rgba_path"])
            with Image.open(rgba_path).convert("RGBA") as source:
                target = source.resize(
                    (render_x2-render_x1, render_y2-render_y1), Image.Resampling.LANCZOS
                )
            canvas = plate.convert("RGBA")
            canvas.alpha_composite(target, dest=(render_x1, render_y1))
            initial_image = canvas.convert("RGB")
            padding = int(aerosol["mask_padding_px"])
            x1, y1 = max(0, render_x1-padding), max(0, render_y1-padding)
            x2, y2 = min(width, render_x2+padding), min(height, render_y2+padding)
            mask_array = np.zeros((height, width), dtype=np.uint8)
            cv2.rectangle(mask_array, (x1, y1), (x2, y2), 255, thickness=-1)
            mask_array = cv2.GaussianBlur(mask_array, (0, 0), sigmaX=5)
            mask = Image.fromarray(mask_array)
            prompt = aerosol["prompt_template"].format(
                scene=aerosol["scene_descriptions"][sample["scene_name"]]
            )
            seed = int(config["seed"]) + sample["index"]
            inference_started = time.monotonic()
            result = inpaint_pipe(
                prompt=prompt, negative_prompt=aerosol["negative_prompt"],
                image=initial_image, mask_image=mask, control_image=control,
                strength=float(aerosol["strength"]), width=width, height=height,
                num_inference_steps=int(config["inference_steps"]),
                guidance_scale=float(config["guidance_scale"]),
                controlnet_conditioning_scale=float(aerosol["canny_scale"]),
                generator=torch.Generator(device="cpu").manual_seed(seed),
            ).images[0]
            torch.cuda.synchronize(args.gpu)
            name = f"g{sample['index']:05d}_c{sample['class_id']:02d}_{sample['scene_name']}.png"
            image_path = images_dir / name
            control_path = controls_dir / name.replace(".png", "_canny.png")
            mask_path = controls_dir / name.replace(".png", "_mask.png")
            init_path = controls_dir / name.replace(".png", "_init.png")
            result.save(image_path)
            control.save(control_path)
            mask.save(mask_path)
            initial_image.save(init_path)
            record = {
                **sample, "generation_mode": aerosol_mode, "seed": seed,
                "background_profile": "aerosol_target_free_plate",
                "background_description": plate_metadata[sample["index"]]["prompt"],
                "prompt": prompt, "negative_prompt": aerosol["negative_prompt"],
                "controlnet_conditioning_scale": float(aerosol["canny_scale"]),
                "condition": condition,
                "reference_condition": {
                    "source_type": "sam3_rgba_diffusion_initialization",
                    "asset_id": row["asset_id"],
                    "rgba_path": str(rgba_path.relative_to(REPO_ROOT)),
                    "rgba_sha256": helper.sha256(rgba_path),
                    "strength": float(aerosol["strength"]),
                },
                "plate": plate_metadata[sample["index"]], "mask_xyxy": [x1, y1, x2, y2],
                "output": str(image_path.relative_to(REPO_ROOT)),
                "output_sha256": helper.sha256(image_path),
                "control_sha256": helper.sha256(control_path),
                "inference_seconds": round(time.monotonic() - inference_started, 3),
                "annotation_performed": False, "degradation_applied": False,
                "training_use_forbidden": True,
            }
            records.append(record)
            review_rows.append({"index": sample["index"], "class_id": sample["class_id"], "class_name": sample["class_name"], "scene_name": sample["scene_name"], "generation_mode": aerosol_mode, "image_path": record["output"], "review_status": "pending", "review_reason": ""})
            completed += 1
            print(f"[{completed}/{len(shard_schedule)}] {name} {record['inference_seconds']:.3f}s", flush=True)
        release_pipeline(inpaint_pipe)
        inpaint_pipe = None

    records.sort(key=lambda row: int(row["index"]))
    review_rows.sort(key=lambda row: int(row["index"]))

    manifest = {"format_version": 2, "pipeline_id": config["pipeline_id"], "status": "generated_pending_review", "started_utc": started_utc, "completed_utc": feasibility.utc_now(), "wall_time_seconds": round(time.monotonic() - started, 3), "git": git, "packages": versions, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu_name": torch.cuda.get_device_name(args.gpu), "peak_allocated_gib": round(torch.cuda.max_memory_allocated(args.gpu) / 1024**3, 3), "models": {"base": models["sdxl"]["base_model"], "controlnet": models["sdxl"]["controlnet"]}, "config_sha256": {"generation": helper.sha256(config_path), "base_generation": helper.sha256(base_config_path) if base_config_path else None, "models": helper.sha256(models_path), "scenes": helper.sha256(scenes_path), "mask_manifest": helper.sha256(manifest_path), "script": helper.sha256(Path(__file__).resolve())}, "records": records}
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    with (output_dir / "review.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)


if __name__ == "__main__":
    main()
