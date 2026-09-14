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
    cv2.fillConvexPoly(canvas, np.array([[505, 520], [305, 220], [485, 500]], np.int32), 255)
    cv2.fillConvexPoly(canvas, np.array([[530, 520], [720, 225], [550, 505]], np.int32), 255)
    for center, angle in (((390, 690), -25), ((590, 690), 25)):
        cv2.ellipse(canvas, center, (120, 82), angle, 0, 360, 255, -1)
        cv2.ellipse(canvas, center, (78, 43), angle, 0, 360, 0, -1)
    cv2.line(canvas, (455, 625), (510, 515), 255, 45)
    cv2.line(canvas, (535, 515), (525, 625), 255, 45)
    cv2.circle(canvas, (520, 515), 25, 255, -1)


def draw_cigarette_pack(canvas: np.ndarray) -> None:
    cv2.rectangle(canvas, (350, 330), (675, 790), 255, -1)
    cv2.rectangle(canvas, (350, 245), (675, 430), 255, -1)
    cv2.fillConvexPoly(canvas, np.array([[350, 430], [505, 500], [675, 430]], np.int32), 255)
    for x in range(375, 655, 45):
        cv2.rectangle(canvas, (x, 270), (x + 25, 420), 0, -1)
        cv2.circle(canvas, (x + 12, 275), 12, 255, -1)


def draw_hacksaw(canvas: np.ndarray) -> None:
    cv2.line(canvas, (280, 300), (735, 300), 255, 45)
    cv2.line(canvas, (280, 300), (230, 650), 255, 45)
    cv2.line(canvas, (735, 300), (785, 625), 255, 45)
    cv2.line(canvas, (235, 650), (780, 650), 255, 18)
    handle = np.array([[700, 545], [865, 575], [850, 770], [705, 735]], np.int32)
    cv2.fillConvexPoly(canvas, handle, 255)
    hole = np.array([[750, 605], [820, 620], [812, 710], [752, 700]], np.int32)
    cv2.fillConvexPoly(canvas, hole, 0)


def draw_battery(canvas: np.ndarray) -> None:
    cv2.rectangle(canvas, (390, 275), (635, 790), 255, -1)
    cv2.rectangle(canvas, (410, 250), (615, 305), 255, -1)
    cv2.ellipse(canvas, (455, 247), (42, 24), 0, 0, 360, 255, -1)
    cv2.ellipse(canvas, (570, 247), (30, 18), 0, 0, 360, 255, -1)


DRAWERS = {2: draw_scissors, 7: draw_cigarette_pack, 8: draw_hacksaw, 12: draw_battery}


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
    class_ids = [int(value) for value in settings["class_ids"]]
    if set(class_ids) != set(DRAWERS):
        raise ValueError("Controlled retry classes and frozen geometry drawers disagree")
    count = int(settings["candidates_per_class"])
    names = {int(key): value for key, value in scenes["class_policy"]["names"].items()}
    targets = {int(key): value for key, value in config["foreground_generation"]["class_prompts"].items()}
    guidance = {int(key): value for key, value in config["foreground_generation"]["retry_guidance"].items()}
    prompts = [
        settings["prompt_template"].format(target=targets[class_id], guidance=guidance[class_id])
        for class_id in class_ids for _ in range(count)
    ]
    if fb.git_state()["dirty"]:
        raise RuntimeError("Commit controlled retry code/config before launch")
    if not torch.cuda.is_available() or not 0 <= args.gpu < torch.cuda.device_count():
        raise RuntimeError(f"Logical CUDA device {args.gpu} is unavailable")
    pipe = load_control_pipeline(models, args.gpu)
    fb.validate_prompts(pipe, prompts + [settings["negative_prompt"]])
    if args.preflight_only:
        fb.release(pipe)
        print(json.dumps({"status": "ready", "class_ids": class_ids,
                          "new_candidates": len(prompts), "synthetic_canny_only": True,
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
    added = []
    with fb.stage_lock(root.parent, "foregrounds_controlled_extension"):
        for class_id in class_ids:
            start = extension.next_round(records, class_id)
            for offset in range(count):
                attempt = start + offset
                candidate_id = f"fg_c{class_id:02d}_control_{attempt:04d}"
                prompt = settings["prompt_template"].format(
                    target=targets[class_id], guidance=guidance[class_id])
                seed = int(config["seed"]) + 700000 + class_id * 10000 + attempt
                image_path = root / "images" / f"{candidate_id}.png"
                control_path = root / "controls" / f"{candidate_id}.png"
                foreground_guidance = float(config["foreground_generation"]["guidance_scale"])
                generation_spec = {
                    "pipeline_id": config["pipeline_id"], "stage": "foreground_controlled_retry",
                    "candidate_id": candidate_id, "config_sha256": config["_config_sha256"],
                    "models_sha256": config["_models_sha256"], "seed": seed,
                    "size": [width, height], "steps": int(config["inference_steps"]),
                    "prompt": prompt, "negative_prompt": settings["negative_prompt"],
                    "guidance_scale": foreground_guidance,
                    "controlnet_conditioning_scale": float(settings["controlnet_conditioning_scale"]),
                }
                resumed = fb.verified_existing_image(image_path, (width, height), generation_spec)
                started = time.monotonic()
                control = make_control(class_id, attempt, (width, height))
                if not resumed:
                    result = pipe(
                        prompt=prompt, negative_prompt=settings["negative_prompt"], image=control,
                        width=width, height=height, num_inference_steps=int(config["inference_steps"]),
                        guidance_scale=foreground_guidance,
                        controlnet_conditioning_scale=float(settings["controlnet_conditioning_scale"]),
                        generator=torch.Generator(device="cpu").manual_seed(seed),
                    ).images[0]
                    fb.save_image_atomic(result, image_path, generation_spec)
                fb.save_image_atomic(control, control_path)
                added.append({
                    "candidate_id": candidate_id, "class_id": class_id,
                    "class_name": names[class_id], "image_path": str(image_path.relative_to(REPO_ROOT)),
                    "sha256": fb.sha256(image_path), "seed": seed, "prompt": prompt,
                    "control_path": str(control_path.relative_to(REPO_ROOT)),
                    "control_sha256": fb.sha256(control_path), "control_source": "synthetic_programmatic_canny",
                    "real_pixels_reused": False, "controlled_retry": True, "retry": True,
                    "negative_prompt": settings["negative_prompt"],
                    "guidance_scale": foreground_guidance,
                    "generation_fingerprint": fb.generation_fingerprint(generation_spec),
                    "seconds": 0.0 if resumed else round(time.monotonic() - started, 3),
                    "resumed_existing": resumed,
                })
                print(f"[controlled retry {len(added)}/{len(prompts)}] {candidate_id}", flush=True)
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
