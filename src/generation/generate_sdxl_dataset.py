"""Generate the clean, one-pass SDXL acceptance dataset.

Every output is synthesized as one complete image. Classes with accepted Canny
profiles keep those profiles; Aerosol uses a concise text-only full-scene branch
until a semantic layout controller is adopted. No compositing or inpainting is
performed here.
"""

from __future__ import annotations

import argparse
import csv
import copy
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/generation/sdxl_generation_v1.yaml"
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


def select_mask(helper, config: dict[str, Any], manifest: Path, sample: dict[str, Any]):
    rows = helper.read_masks(manifest, sample["class_id"])
    preferred = config["silhouette_source"].get("preferred_stems", {}).get(sample["class_id"])
    if preferred:
        stem = preferred[sample["class_attempt_index"] % len(preferred)]
        matches = [row for row in rows if row["stem"] == stem]
        if len(matches) != 1:
            raise ValueError(f"Preferred silhouette is not uniquely accepted: {stem}")
        return matches[0]
    offset = int(config["silhouette_source"]["selection_seed_offset"])
    return rows[(int(config["seed"]) + offset + sample["class_attempt_index"]) % len(rows)]


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
    pipe = feasibility.load_pipeline("sdxl", models, args.gpu)
    reference_config = config.get("reference_conditioning")
    reference_class_ids: set[int] = set()
    if reference_config:
        reference_class_ids = {int(value) for value in reference_config["class_ids"]}
        adapter = reference_config["adapter"]
        pipe.load_ip_adapter(
            adapter["repo_id"],
            subfolder=adapter["subfolder"],
            weight_name=adapter["weight_name"],
            image_encoder_folder=adapter["image_encoder_folder"],
        )
        pipe.set_ip_adapter_scale(float(reference_config["scale"]))
    images_dir = output_dir / "images"
    controls_dir = output_dir / "controls"
    images_dir.mkdir(parents=True)
    controls_dir.mkdir()
    records = []
    review_rows = []
    modes = {int(key): value for key, value in config["generation_modes"]["class_overrides"].items()}
    scales = {int(key): value for key, value in config["controlnet_conditioning_scale"]["class_overrides"].items()}
    width, height = config["output_size"]
    for position, sample in enumerate(shard_schedule, start=1):
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
        print(f"[{position}/{len(shard_schedule)}] {name} {record['inference_seconds']:.3f}s", flush=True)

    manifest = {"format_version": 2, "pipeline_id": config["pipeline_id"], "status": "generated_pending_review", "started_utc": started_utc, "completed_utc": feasibility.utc_now(), "wall_time_seconds": round(time.monotonic() - started, 3), "git": git, "packages": versions, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu_name": torch.cuda.get_device_name(args.gpu), "peak_allocated_gib": round(torch.cuda.max_memory_allocated(args.gpu) / 1024**3, 3), "models": {"base": models["sdxl"]["base_model"], "controlnet": models["sdxl"]["controlnet"]}, "config_sha256": {"generation": helper.sha256(config_path), "base_generation": helper.sha256(base_config_path) if base_config_path else None, "models": helper.sha256(models_path), "scenes": helper.sha256(scenes_path), "mask_manifest": helper.sha256(manifest_path), "script": helper.sha256(Path(__file__).resolve())}, "records": records}
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    with (output_dir / "review.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)


if __name__ == "__main__":
    main()
