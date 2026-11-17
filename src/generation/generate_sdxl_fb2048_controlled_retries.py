"""Append synthetic-Canny SDXL foreground retries for geometry-sensitive classes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_SCRIPT = REPO_ROOT / "src/generation/run_sdxl_fb2048_pipeline.py"
EXTENSION_SCRIPT = REPO_ROOT / "src/generation/extend_sdxl_fb2048_candidates.py"
DEFAULT_CONFIG = REPO_ROOT / "configs/generation/sdxl_fb2048_v2.yaml"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_control_pipeline(models: dict[str, Any], gpu: int):
    from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
    model = models["sdxl"]
    control = model["controlnet"]
    controlnet = ControlNetModel.from_pretrained(
        control["id"], revision=control["revision"], dtype=torch.float16,
        variant=model["variant"], use_safetensors=True,
    )
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        model["base_model"]["id"], revision=model["base_model"]["revision"],
        controlnet=controlnet, dtype=torch.float16, variant=model["variant"],
        use_safetensors=True,
    )
    pipe.enable_model_cpu_offload(gpu_id=gpu)
    pipe.vae.enable_slicing()
    return pipe


def draw_scissors(canvas: np.ndarray) -> None:
    # Closed, joined blades above the pivot; two distinct lower finger loops.
    cv2.fillConvexPoly(canvas, np.array([[510, 490], [470, 210], [485, 205], [530, 490]], np.int32), 255)
    cv2.fillConvexPoly(canvas, np.array([[525, 490], [535, 210], [550, 205], [540, 490]], np.int32), 255)
    cv2.line(canvas, (510, 500), (420, 660), 255, 34)
    cv2.line(canvas, (535, 500), (605, 660), 255, 34)
    for center, angle in (((390, 730), -25), ((630, 730), 25)):
        cv2.ellipse(canvas, center, (88, 112), angle, 0, 360, 255, -1)
        cv2.ellipse(canvas, center, (50, 70), angle, 0, 360, 0, -1)
    cv2.circle(canvas, (522, 500), 23, 255, -1)
    cv2.circle(canvas, (522, 500), 9, 0, -1)


def draw_pliers(canvas: np.ndarray) -> None:
    pivot = (512, 455)
    cv2.fillConvexPoly(canvas, np.array([[497, 455], [475, 285], [497, 255], [511, 385]], np.int32), 255)
    cv2.fillConvexPoly(canvas, np.array([[527, 455], [527, 385], [540, 255], [562, 285]], np.int32), 255)
    cv2.line(canvas, (501, 465), (365, 810), 255, 66)
    cv2.line(canvas, (523, 465), (660, 810), 255, 66)
    cv2.circle(canvas, pivot, 27, 255, -1)
    cv2.circle(canvas, pivot, 10, 0, -1)


def draw_cigarette_pack(canvas: np.ndarray) -> None:
    cv2.rectangle(canvas, (350, 330), (675, 790), 255, -1)
    cv2.rectangle(canvas, (350, 245), (675, 430), 255, -1)
    cv2.fillConvexPoly(canvas, np.array([[350, 430], [505, 500], [675, 430]], np.int32), 255)
    for x in range(375, 655, 45):
        cv2.rectangle(canvas, (x, 270), (x + 25, 420), 0, -1)
        cv2.circle(canvas, (x + 12, 275), 12, 255, -1)


def draw_hacksaw(canvas: np.ndarray) -> None:
    cv2.line(canvas, (265, 360), (745, 360), 255, 42)
    cv2.line(canvas, (265, 360), (245, 650), 255, 42)
    cv2.line(canvas, (745, 360), (745, 650), 255, 42)
    # A separate thin blade with visible downward teeth, anchored at both ends.
    cv2.line(canvas, (245, 662), (745, 662), 255, 9)
    for x in range(265, 730, 18):
        cv2.line(canvas, (x, 665), (x + 9, 678), 255, 3)
    cv2.circle(canvas, (252, 647), 12, 180, -1)
    cv2.circle(canvas, (739, 647), 12, 180, -1)
    handle = np.array([[706, 510], [852, 570], [845, 770], [710, 720]], np.int32)
    cv2.fillConvexPoly(canvas, handle, 255)
    hole = np.array([[748, 590], [815, 615], [810, 705], [750, 685]], np.int32)
    cv2.fillConvexPoly(canvas, hole, 0)


def draw_battery(canvas: np.ndarray) -> None:
    cv2.rectangle(canvas, (390, 345), (635, 735), 255, -1)
    cv2.rectangle(canvas, (395, 325), (630, 355), 180, -1)
    # Face-wrap seams and a printed-label region give SDXL material cues.
    cv2.rectangle(canvas, (412, 410), (613, 672), 170, -1)
    cv2.rectangle(canvas, (412, 410), (613, 672), 255, 3)
    cv2.line(canvas, (403, 365), (403, 715), 150, 4)
    cv2.line(canvas, (622, 365), (622, 715), 150, 4)
    cv2.putText(canvas, "9V", (443, 555), cv2.FONT_HERSHEY_SIMPLEX,
                2.0, 90, 5, cv2.LINE_AA)
    cv2.circle(canvas, (455, 313), 32, 255, -1)
    cv2.circle(canvas, (565, 313), 21, 255, -1)
    cv2.circle(canvas, (455, 313), 17, 0, -1)


def draw_aerosol(canvas: np.ndarray) -> None:
    cv2.rectangle(canvas, (405, 345), (620, 805), 255, -1)
    cv2.ellipse(canvas, (512, 345), (108, 45), 0, 180, 360, 255, -1)
    cv2.rectangle(canvas, (485, 260), (540, 335), 255, -1)
    cv2.circle(canvas, (538, 284), 8, 0, -1)
    cv2.line(canvas, (410, 795), (615, 795), 90, 5)


DRAWERS = {2: draw_scissors, 3: draw_pliers, 7: draw_cigarette_pack,
           8: draw_hacksaw, 11: draw_aerosol, 12: draw_battery}


def make_control(class_id: int, attempt: int, size: tuple[int, int]) -> Image.Image:
    width, height = size
    if (width, height) != (1024, 1024):
        raise ValueError("Frozen controlled pilot geometry requires 1024x1024")
    silhouette = np.zeros((height, width), dtype=np.uint8)
    DRAWERS[class_id](silhouette)
    angle = (-6.0, 6.0)[attempt % 2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    silhouette = cv2.warpAffine(silhouette, matrix, (width, height), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    control = cv2.Canny(cv2.GaussianBlur(silhouette, (0, 0), 1.0), 32, 96)
    return Image.fromarray(cv2.cvtColor(control, cv2.COLOR_GRAY2RGB))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--class-id", type=int, action="append",
                        help="Limit this retry to configured class IDs; repeat for multiple classes")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-experimental", action="store_true")
    args = parser.parse_args()
    fb = load_module("sdxl_fb2048", PIPELINE_SCRIPT)
    extension = load_module("sdxl_fb2048_extension", EXTENSION_SCRIPT)
    config_path = fb.repo_path(args.config)
    config = fb.load_yaml(config_path)
    config["_config_sha256"] = fb.sha256(config_path)
    config["_models_sha256"] = fb.sha256(fb.repo_path(config["models_config"]))
    scenes = fb.load_yaml(fb.repo_path(config["scene_policy"]))
    models = fb.load_yaml(fb.repo_path(config["models_config"]))
    settings = config["foreground_generation"]["controlled_retry"]
    if settings.get("status") != "approved" and not args.allow_experimental:
        raise RuntimeError(
            "Controlled retry is experimental; run its separate pilot with --allow-experimental"
        )
    configured_class_ids = {int(value) for value in settings["class_ids"]}
    if not configured_class_ids.issubset(DRAWERS):
        raise ValueError("A controlled retry class lacks a frozen geometry drawer")
    class_ids = sorted(set(args.class_id)) if args.class_id else sorted(configured_class_ids)
    if not set(class_ids).issubset(configured_class_ids):
        raise ValueError(f"Requested classes must be configured controlled retries: {class_ids}")
    count = int(settings["candidates_per_class"])
    names = {int(key): value for key, value in scenes["class_policy"]["names"].items()}
    identity_prompts = {int(key): value for key, value in settings["identity_prompts"].items()}
    class_negatives = {
        int(key): value
        for key, value in config["foreground_generation"]["class_negative_prompts"].items()
    }
    control_scales = {int(key): float(value) for key, value in
                      settings["controlnet_conditioning_scale_by_class"].items()}
    if configured_class_ids != set(identity_prompts) or configured_class_ids != set(control_scales):
        raise ValueError("Each controlled retry class needs its own prompt and Canny scale")
    if any(not 0.0 < control_scales[class_id] <= 1.0 for class_id in class_ids):
        raise ValueError("Controlled retry Canny scales must be in (0, 1]")
    prompts = [identity_prompts[class_id] for class_id in class_ids]
    if fb.git_state()["dirty"]:
        raise RuntimeError("Commit controlled retry code/config before launch")
    if not torch.cuda.is_available() or not 0 <= args.gpu < torch.cuda.device_count():
        raise RuntimeError(f"Logical CUDA device {args.gpu} is unavailable")
    pipe = load_control_pipeline(models, args.gpu)
    negative_by_class = {
        class_id: f"{settings['negative_prompt']}, {class_negatives[class_id]}"
        for class_id in class_ids
    }
    fb.validate_prompts(pipe, prompts + list(negative_by_class.values()))
    if args.preflight_only:
        fb.release(pipe)
        print(json.dumps({"status": "ready", "class_ids": class_ids,
                          "new_candidates": len(class_ids) * count, "synthetic_canny_only": True,
                          "runtime_prompts_validated": True}, indent=2))
        return

    root = fb.repo_path(config["output_root"]) / "pilot" / "foregrounds"
    manifest_path, review_path = root / "manifest.json", root / "review.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest["records"]
    fields, review_rows = extension.read_review(review_path)
    if set(row["candidate_id"] for row in review_rows) != set(row["candidate_id"] for row in records):
        raise ValueError("Existing review and manifest candidate IDs do not match")
    width, height = map(int, config["output_size"])
    attempts_by_class = {
        class_id: extension.next_attempts(records, class_id, count)
        for class_id in class_ids
    }
    added = []
    with fb.stage_lock(root.parent, "foregrounds_controlled_extension"):
        for class_id in class_ids:
            for attempt in attempts_by_class[class_id]:
                candidate_id = f"fg_c{class_id:02d}_control_{attempt:04d}"
                prompt = identity_prompts[class_id]
                seed = int(config["seed"]) + 700000 + class_id * 10000 + attempt
                image_path = root / "images" / f"{candidate_id}.png"
                control_path = root / "controls" / f"{candidate_id}.png"
                foreground_guidance = float(config["foreground_generation"]["guidance_scale"])
                generation_spec = {
                    "pipeline_id": config["pipeline_id"], "stage": "foreground_controlled_retry",
                    "candidate_id": candidate_id, "config_sha256": config["_config_sha256"],
                    "models_sha256": config["_models_sha256"], "seed": seed,
                    "size": [width, height], "steps": int(config["inference_steps"]),
                    "prompt": prompt, "negative_prompt": negative_by_class[class_id],
                    "guidance_scale": foreground_guidance,
                    "controlnet_conditioning_scale": control_scales[class_id],
                }
                resumed = fb.verified_existing_image(image_path, (width, height), generation_spec)
                started = time.monotonic()
                control = make_control(class_id, attempt, (width, height))
                if not resumed:
                    result = pipe(
                        prompt=prompt, negative_prompt=negative_by_class[class_id], image=control,
                        width=width, height=height, num_inference_steps=int(config["inference_steps"]),
                        guidance_scale=foreground_guidance,
                        controlnet_conditioning_scale=control_scales[class_id],
                        generator=torch.Generator(device="cpu").manual_seed(seed),
                    ).images[0]
                    fb.save_image_atomic(result, image_path, generation_spec)
                fb.save_image_atomic(control, control_path)
                added.append({
                    "candidate_id": candidate_id, "class_id": class_id,
                    "class_name": names[class_id], "image_path": str(image_path.relative_to(REPO_ROOT)),
                    "sha256": fb.sha256(image_path), "seed": seed, "attempt": attempt,
                    "prompt": prompt,
                    "control_path": str(control_path.relative_to(REPO_ROOT)),
                    "control_sha256": fb.sha256(control_path), "control_source": "synthetic_programmatic_canny",
                    "real_pixels_reused": False, "controlled_retry": True, "retry": True,
                    "negative_prompt": negative_by_class[class_id],
                    "guidance_scale": foreground_guidance,
                    "controlnet_conditioning_scale": control_scales[class_id],
                    "generation_fingerprint": fb.generation_fingerprint(generation_spec),
                    "seconds": 0.0 if resumed else round(time.monotonic() - started, 3),
                    "resumed_existing": resumed,
                })
                print(f"[controlled retry {len(added)}/{len(class_ids) * count}] {candidate_id}", flush=True)
        fb.release(pipe)
        fb.apply_person_screen(config, added, args.gpu)
        records.extend(added)
        manifest["records"] = records
        manifest.setdefault("extensions", []).append({
            "git": fb.git_state(), "script_sha256": fb.sha256(Path(__file__).resolve()),
            "config_sha256": fb.sha256(config_path), "class_ids": class_ids,
            "count_per_class": count, "control_source": "synthetic_programmatic_canny",
            "candidate_ids": [row["candidate_id"] for row in added],
        })
        fb.write_json(manifest_path, manifest)
        rows = extension.append_review_rows(
            review_rows, added, "review_controlled_object_identity", fields,
        )
        extension.write_review_atomic(review_path, fields, rows)
    fb.release(None)


if __name__ == "__main__":
    main()
