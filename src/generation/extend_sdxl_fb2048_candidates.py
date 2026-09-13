"""Append review-gated retry candidates to an existing SDXL-FB2048 pilot pool."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_SCRIPT = REPO_ROOT / "src/generation/run_sdxl_fb2048_pipeline.py"
DEFAULT_CONFIG = REPO_ROOT / "configs/generation/sdxl_fb2048_v1.yaml"


def load_pipeline_module():
    spec = importlib.util.spec_from_file_location("sdxl_fb2048", PIPELINE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import pipeline: {PIPELINE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_review(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_review_atomic(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def next_round(records: list[dict[str, Any]], class_id: int) -> int:
    return sum(int(row["class_id"]) == class_id for row in records)


def retry_background_prompt(config: dict[str, Any], scene_name: str, attempt: int) -> str:
    settings = config["background_generation"]
    return settings["retry_prompt_template"].format(
        scene=scene_name.replace("_", " "),
        support=settings["support_profiles"][attempt % len(settings["support_profiles"])],
        layout=settings["layout_profiles"][(attempt // 2) % len(settings["layout_profiles"])],
        wear=settings["wear_profiles"][(attempt // 3) % len(settings["wear_profiles"])],
        occupancy=settings["occupancy_profiles"][(attempt // 5) % len(settings["occupancy_profiles"])],
        camera=settings["camera_profiles"][attempt % len(settings["camera_profiles"])],
        lighting=settings["lighting_profiles"][(attempt // 2) % len(settings["lighting_profiles"])],
    )


def retry_foreground_prompt(config: dict[str, Any], class_id: int, attempt: int) -> str:
    settings = config["foreground_generation"]
    prompts = {int(key): value for key, value in settings["class_prompts"].items()}
    guidance = {int(key): value for key, value in settings.get("retry_guidance", {}).items()}
    variation = settings["variation_phrases"][attempt % len(settings["variation_phrases"])]
    prompt = settings["prompt_template"].format(target=prompts[class_id], variation=variation)
    if class_id in guidance:
        prompt = f"{prompt} {guidance[class_id]}."
    return prompt


def append_review_rows(
    existing: list[dict[str, str]], records: list[dict[str, Any]], reason: str,
) -> list[dict[str, str]]:
    rows = list(existing)
    for record in records:
        rows.append({
            "candidate_id": record["candidate_id"],
            "class_id": str(record["class_id"]),
            "class_name": record["class_name"],
            "scene_name": record.get("scene_name", ""),
            "image_path": record["image_path"],
            "review_status": "pending",
            "review_reason": reason,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pool", choices=("backgrounds", "foregrounds"), required=True)
    parser.add_argument("--class-id", type=int, action="append", required=True, dest="class_ids")
    parser.add_argument("--count-per-class", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.count_per_class < 1:
        raise ValueError("--count-per-class must be positive")
    class_ids = sorted(set(args.class_ids))
    if any(class_id not in range(16) for class_id in class_ids):
        raise ValueError(f"Class IDs must be in [0, 15]: {class_ids}")

    fb = load_pipeline_module()
    config_path = fb.repo_path(args.config)
    config = fb.load_yaml(config_path)
    scenes = fb.load_yaml(fb.repo_path(config["scene_policy"]))
    models = fb.load_yaml(fb.repo_path(config["models_config"]))
    root = fb.repo_path(config["output_root"]) / "pilot" / args.pool
    manifest_path, review_path = root / "manifest.json", root / "review.csv"
    if not manifest_path.is_file() or not review_path.is_file():
        raise FileNotFoundError(f"Complete the initial {args.pool} stage first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest["records"]
    fields, review_rows = read_review(review_path)
    if set(row["candidate_id"] for row in review_rows) != set(row["candidate_id"] for row in records):
        raise ValueError("Existing review and manifest candidate IDs do not match")

    prompts = []
    assignments = {int(key): value for key, value in scenes["class_to_scene_families"].items()}
    for class_id in class_ids:
        start = next_round(records, class_id)
        for offset in range(args.count_per_class):
            attempt = start + offset
            scene_name = assignments[class_id][attempt % len(assignments[class_id])]
            prompts.append(
                retry_background_prompt(config, scene_name, attempt)
                if args.pool == "backgrounds"
                else retry_foreground_prompt(config, class_id, attempt)
            )

    if fb.git_state()["dirty"]:
        raise RuntimeError("Commit retry code/config before launch")
    if not torch.cuda.is_available() or not 0 <= args.gpu < torch.cuda.device_count():
        raise RuntimeError(f"Logical CUDA device {args.gpu} is unavailable")
    pipe = fb.load_base_pipeline(models, args.gpu)
    negative = config["background_generation" if args.pool == "backgrounds" else "foreground_generation"]["negative_prompt"]
    fb.validate_prompts(pipe, prompts + [negative])
    if args.preflight_only:
        fb.release(pipe)
        print(json.dumps({"status": "ready", "pool": args.pool, "class_ids": class_ids,
                          "new_candidates": len(prompts), "runtime_prompts_validated": True}, indent=2))
        return

    names = {int(key): value for key, value in scenes["class_policy"]["names"].items()}
    width, height = map(int, config["output_size"])
    added = []
    with fb.stage_lock(root.parent, f"{args.pool}_extension"):
        prompt_index = 0
        for class_id in class_ids:
            start = next_round(records, class_id)
            for offset in range(args.count_per_class):
                attempt = start + offset
                candidate_id = f"{'bg' if args.pool == 'backgrounds' else 'fg'}_c{class_id:02d}_retry_{attempt:04d}"
                scene_name = assignments[class_id][attempt % len(assignments[class_id])]
                prompt = prompts[prompt_index]; prompt_index += 1
                seed_offset = 500000 if args.pool == "backgrounds" else 600000
                seed = int(config["seed"]) + seed_offset + class_id * 10000 + attempt
                path = root / "images" / f"{candidate_id}.png"
                resumed = fb.verified_existing_image(path, (width, height))
                started = time.monotonic()
                if not resumed:
                    image = pipe(
                        prompt=prompt, negative_prompt=negative, width=width, height=height,
                        num_inference_steps=int(config["inference_steps"]),
                        guidance_scale=float(config["guidance_scale"]),
                        generator=torch.Generator(device="cpu").manual_seed(seed),
                    ).images[0]
                    fb.save_image_atomic(image, path)
                record = {"candidate_id": candidate_id, "class_id": class_id,
                          "class_name": names[class_id], "image_path": str(path.relative_to(REPO_ROOT)),
                          "sha256": fb.sha256(path), "seed": seed, "prompt": prompt,
                          "seconds": 0.0 if resumed else round(time.monotonic() - started, 3),
                          "resumed_existing": resumed, "retry": True}
                if args.pool == "backgrounds":
                    record.update({"planned_class_id": class_id,
                                   "scene_slot": attempt % len(assignments[class_id]),
                                   "scene_name": scene_name})
                added.append(record)
                print(f"[{args.pool} retry {len(added)}/{len(prompts)}] {candidate_id}", flush=True)
        records.extend(added)
        manifest["records"] = records
        manifest.setdefault("extensions", []).append({
            "git": fb.git_state(), "script_sha256": fb.sha256(Path(__file__).resolve()),
            "config_sha256": fb.sha256(config_path), "class_ids": class_ids,
            "count_per_class": args.count_per_class, "candidate_ids": [row["candidate_id"] for row in added],
        })
        fb.write_json(manifest_path, manifest)
        reason = "review_retry_support_surface" if args.pool == "backgrounds" else "review_retry_object_identity"
        write_review_atomic(review_path, fields, append_review_rows(review_rows, added, reason))
    fb.release(pipe)


if __name__ == "__main__":
    main()
