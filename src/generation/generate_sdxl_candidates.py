"""Generate the frozen SDXL full-scene acceptance test or production candidates.

This stage creates clean candidate images and traceable ControlNet conditions.
It deliberately creates no training labels and applies no degradation. A later
review/annotation/finalization stage selects exactly balanced accepted images.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/generation/sdxl_canonical_v1.yaml"
PILOT_MODULE = REPO_ROOT / "src/generation/run_all_class_genai_pilot.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_schedule(config: dict[str, Any], scene: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    names = {int(key): value for key, value in scene["class_policy"]["names"].items()}
    assignments = {int(key): value for key, value in scene["class_to_scene_families"].items()}
    schedule: list[dict[str, Any]] = []
    if mode == "test":
        repeats = int(config["test"]["images_per_class_scene"])
        for repeat in range(repeats):
            for scene_slot in range(4):
                for class_id in range(16):
                    schedule.append({
                        "class_id": class_id,
                        "target": names[class_id],
                        "scene_name": assignments[class_id][scene_slot],
                        "scene_slot": scene_slot,
                        "quota_block": None,
                        "attempt_in_cell": repeat,
                        "class_attempt_index": repeat * 4 + scene_slot,
                    })
        expected = int(config["test"]["expected_candidates"])
    else:
        production = config["production"]
        blocks = int(production["class_scene_blocks"])
        default_attempts = int(production["standard_attempts_per_class_scene_block"])
        overrides = {int(key): int(value) for key, value in production.get("class_attempts_per_class_scene_block", {}).items()}
        class_ordinals = Counter()
        for block in range(blocks):
            for scene_slot in range(4):
                max_attempts = max([default_attempts, *overrides.values()])
                for attempt in range(max_attempts):
                    for class_id in range(16):
                        if attempt >= overrides.get(class_id, default_attempts):
                            continue
                        schedule.append({
                            "class_id": class_id,
                            "target": names[class_id],
                            "scene_name": assignments[class_id][scene_slot],
                            "scene_slot": scene_slot,
                            "quota_block": block,
                            "attempt_in_cell": attempt,
                            "class_attempt_index": class_ordinals[class_id],
                        })
                        class_ordinals[class_id] += 1
        expected = int(production["expected_candidates"])
    if len(schedule) != expected:
        raise ValueError(f"Expected {expected} {mode} candidates, scheduled {len(schedule)}")
    for global_index, sample in enumerate(schedule):
        sample["global_index"] = global_index
    return schedule


def choose_row(pilot, config: dict[str, Any], manifest_path: Path, sample: dict[str, Any]) -> dict[str, str]:
    class_id = sample["class_id"]
    rows = pilot.read_masks(manifest_path, class_id)
    preferred = config["silhouette_source"].get("preferred_stems", {}).get(class_id)
    if preferred:
        stem = preferred[sample["class_attempt_index"] % len(preferred)]
        matches = [row for row in rows if row["stem"] == stem]
        if len(matches) != 1:
            raise ValueError(f"Preferred silhouette not unique/accepted: class={class_id} stem={stem}")
        return matches[0]
    index = (
        int(config["seed"])
        + int(config["silhouette_source"]["selection_seed_offset"])
        + int(sample["class_attempt_index"])
    ) % len(rows)
    return rows[index]


def class_scale(config: dict[str, Any], sample: dict[str, Any]) -> float:
    value = config["class_controlnet_conditioning_scale"][sample["class_id"]]
    if isinstance(value, list):
        value = value[sample["class_attempt_index"] % len(value)]
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("test", "production"), default="test")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--aerosol-scene-only-recheck", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require num-shards >= 1 and 0 <= shard-index < num-shards")

    pilot = load_module(PILOT_MODULE, "genai_pilot_helpers")
    helper = pilot.feasibility_module()
    config_path = pilot.repo_path(args.config)
    config = pilot.load_yaml(config_path)
    models_path = pilot.repo_path(config["models_config"])
    scene_path = pilot.repo_path(config["scene_policy"])
    generation_path = pilot.repo_path(config["generation_policy"])
    degradation_path = pilot.repo_path(config["degradation_policy"])
    manifest_path = pilot.repo_path(config["silhouette_source"]["audit_manifest"])
    models = pilot.load_yaml(models_path)
    scene = pilot.load_yaml(scene_path)

    if config["status"] != "ready_for_canonical_acceptance_test":
        raise ValueError("Canonical config is not frozen for its acceptance test")
    if args.mode == "production" and not config["production"]["approved"]:
        raise RuntimeError("Production is locked until the canonical acceptance test and annotation preflight pass")
    git = helper.git_state()
    if git["dirty"]:
        raise RuntimeError("Git worktree is dirty; commit the frozen pipeline before launch")
    versions = helper.installed_versions()
    mismatches = {
        name: {"expected": expected, "actual": versions.get(name)}
        for name, expected in helper.EXPECTED_PACKAGES.items()
        if versions.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(f"Package-version mismatch: {json.dumps(mismatches, indent=2)}")
    schedule = build_schedule(config, scene, args.mode)
    if args.aerosol_scene_only_recheck:
        if args.mode != "test":
            raise ValueError("Aerosol scene-only recheck is only valid in test mode")
        recheck = config["test"]["aerosol_scene_only_recheck"]
        if not recheck["enabled"]:
            raise RuntimeError("Aerosol scene-only recheck is disabled")
        schedule = [row for row in schedule if row["class_id"] == int(recheck["class_id"])]
        if len(schedule) != int(recheck["expected_candidates"]):
            raise ValueError("Unexpected Aerosol recheck schedule size")
    shard_schedule = [row for row in schedule if row["global_index"] % args.num_shards == args.shard_index]
    output_root = pilot.repo_path(config["output_roots"][args.mode])
    if args.aerosol_scene_only_recheck:
        output_root = output_root / config["test"]["aerosol_scene_only_recheck"]["output_subdirectory"]
    output_dir = output_root / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")

    preflight = {
        "status": "ready",
        "mode": args.mode,
        "backend": "sdxl",
        "total_candidates": len(schedule),
        "shard_candidates": len(shard_schedule),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "aerosol_scene_only_recheck": args.aerosol_scene_only_recheck,
        "output": str(output_dir.relative_to(REPO_ROOT)),
        "git": git,
        "packages": versions,
        "config_sha256": pilot.sha256(config_path),
        "mask_manifest_sha256": pilot.sha256(manifest_path),
    }
    if args.preflight_only:
        print(json.dumps(preflight, indent=2))
        return
    if not torch.cuda.is_available() or not 0 <= args.gpu < torch.cuda.device_count():
        raise RuntimeError(f"Logical CUDA device {args.gpu} is unavailable")

    remote_revisions = helper.resolve_remote_revisions(models, "sdxl")
    run_started_utc = helper.utc_now()
    run_started = time.monotonic()
    load_started = time.monotonic()
    pipe = helper.load_pipeline("sdxl", models, args.gpu)
    model_load_seconds = round(time.monotonic() - load_started, 3)
    images_dir = output_dir / "images"
    controls_dir = output_dir / "controls"
    images_dir.mkdir(parents=True)
    controls_dir.mkdir()
    records: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    width, height = [int(value) for value in config["output_size"]]

    for shard_position, sample in enumerate(shard_schedule, start=1):
        index = int(sample["global_index"])
        scene_only = args.aerosol_scene_only_recheck
        row = None if scene_only else choose_row(pilot, config, manifest_path, sample)
        control_config = config
        if scene_only:
            control_config = dict(config)
            control_config["target_conditioning_mode"] = "scene_only"
        proxy, control, condition = pilot.build_control(control_config, row, index, sample["scene_name"])
        prompt = pilot.prompt_for(config, scene["scene_families"][sample["scene_name"]], sample["target"], sample["class_id"], index)
        negative = pilot.negative_prompt_for(config, sample["class_id"])
        scale = (float(config["test"]["aerosol_scene_only_recheck"]["controlnet_conditioning_scale"])
                 if scene_only else class_scale(config, sample))
        seed = int(config["seed"]) + index
        generator = torch.Generator(device="cpu").manual_seed(seed)
        started = time.monotonic()
        result = pipe(
            prompt=prompt, negative_prompt=negative, image=control,
            height=height, width=width,
            num_inference_steps=int(config["inference_steps"]),
            controlnet_conditioning_scale=scale,
            guidance_scale=float(config["guidance_scale"]), generator=generator,
        )
        torch.cuda.synchronize(args.gpu)
        name = f"g{index:05d}_c{sample['class_id']:02d}_{sample['scene_name']}.png"
        output_path = images_dir / name
        proxy_path = controls_dir / f"g{index:05d}_proxy.png"
        canny_path = controls_dir / f"g{index:05d}_canny.png"
        result.images[0].save(output_path)
        proxy.save(proxy_path)
        control.save(canny_path)
        record = {
            **sample, "seed": seed, "prompt": prompt, "negative_prompt": negative,
            "controlnet_conditioning_scale": scale,
            "condition": condition,
            "output": str(output_path.relative_to(REPO_ROOT)),
            "output_sha256": pilot.sha256(output_path),
            "proxy_sha256": pilot.sha256(proxy_path),
            "control_sha256": pilot.sha256(canny_path),
            "inference_seconds": round(time.monotonic() - started, 3),
            "annotation_performed": False,
            "degradation_applied": False,
            "training_use_forbidden": True,
        }
        records.append(record)
        review_rows.append({
            "global_index": index, "class_id": sample["class_id"],
            "class_name": sample["target"], "scene_name": sample["scene_name"],
            "quota_block": sample["quota_block"], "attempt_in_cell": sample["attempt_in_cell"],
            "image_path": record["output"], "review_status": "pending",
            "review_reason": "", "annotation_status": "pending",
        })
        print(f"[{shard_position}/{len(shard_schedule)}] {name} {record['inference_seconds']:.3f}s", flush=True)

    manifest = {
        "format_version": 1,
        "pipeline_id": config["pipeline_id"],
        "status": "generated_pending_review_and_annotation",
        "mode": args.mode,
        "started_utc": run_started_utc,
        "completed_utc": helper.utc_now(),
        "wall_time_seconds": round(time.monotonic() - run_started, 3),
        "model_load_seconds": model_load_seconds,
        "git": git,
        "packages": versions,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_gpu": args.gpu,
        "gpu_name": torch.cuda.get_device_name(args.gpu),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated(args.gpu) / 1024**3, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved(args.gpu) / 1024**3, 3),
        "remote_revisions": remote_revisions,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "aerosol_scene_only_recheck": args.aerosol_scene_only_recheck,
        "config_sha256": {
            "canonical": pilot.sha256(config_path),
            "models": pilot.sha256(models_path),
            "scene_policy": pilot.sha256(scene_path),
            "generation_policy": pilot.sha256(generation_path),
            "degradation_policy": pilot.sha256(degradation_path),
            "mask_manifest": pilot.sha256(manifest_path),
            "script": pilot.sha256(Path(__file__).resolve()),
        },
        "records": records,
    }
    with (output_dir / "candidate_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    with (output_dir / "review_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)


if __name__ == "__main__":
    main()
