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
DEFAULT_CONFIG = REPO_ROOT / "configs/generation/sdxl_fb2048_v1.yaml"


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
    cv2.ellipse(canvas, (385, 690), (105, 68), -25, 0, 360, 255, 18)
    cv2.ellipse(canvas, (590, 690), (105, 68), 25, 0, 360, 255, 18)
    cv2.line(canvas, (455, 625), (505, 520), 255, 18)
    cv2.line(canvas, (530, 520), (525, 625), 255, 18)
    cv2.circle(canvas, (518, 520), 22, 255, 8)
    cv2.line(canvas, (505, 500), (325, 235), 255, 20)
    cv2.line(canvas, (535, 500), (700, 245), 255, 20)
    cv2.line(canvas, (325, 235), (495, 520), 255, 5)
    cv2.line(canvas, (700, 245), (540, 520), 255, 5)


def draw_cigarette_pack(canvas: np.ndarray) -> None:
    cv2.rectangle(canvas, (350, 245), (675, 790), 255, 18)
    cv2.line(canvas, (350, 400), (675, 400), 255, 12)
    cv2.line(canvas, (350, 400), (500, 480), 255, 8)
    cv2.line(canvas, (675, 400), (525, 480), 255, 8)
    cv2.rectangle(canvas, (455, 525), (570, 640), 255, 8)


def draw_hacksaw(canvas: np.ndarray) -> None:
    cv2.line(canvas, (285, 300), (735, 300), 255, 18)
    cv2.line(canvas, (285, 300), (230, 650), 255, 18)
    cv2.line(canvas, (735, 300), (790, 630), 255, 18)
    cv2.line(canvas, (240, 650), (765, 650), 255, 10)
    for x in range(255, 760, 24):
        cv2.line(canvas, (x, 650), (x + 10, 665), 255, 4)
    cv2.rectangle(canvas, (700, 560), (860, 760), 255, 18)
    cv2.rectangle(canvas, (745, 605), (820, 710), 255, 12)


def draw_battery(canvas: np.ndarray) -> None:
    cv2.rectangle(canvas, (390, 260), (635, 790), 255, 18)
    cv2.rectangle(canvas, (410, 280), (615, 760), 255, 5)
    cv2.ellipse(canvas, (455, 250), (38, 22), 0, 0, 360, 255, 10)
    cv2.ellipse(canvas, (570, 250), (27, 17), 0, 0, 360, 255, 9)
    cv2.line(canvas, (405, 410), (620, 410), 255, 6)


DRAWERS = {2: draw_scissors, 7: draw_cigarette_pack, 8: draw_hacksaw, 12: draw_battery}


def make_control(class_id: int, attempt: int, size: tuple[int, int]) -> Image.Image:
    width, height = size
    if (width, height) != (1024, 1024):
        raise ValueError("Frozen controlled pilot geometry requires 1024x1024")
    canvas = np.zeros((height, width), dtype=np.uint8)
    DRAWERS[class_id](canvas)
    angle = (-6.0, 6.0)[attempt % 2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    canvas = cv2.warpAffine(canvas, matrix, (width, height), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    fb = load_module("sdxl_fb2048", PIPELINE_SCRIPT)
    extension = load_module("sdxl_fb2048_extension", EXTENSION_SCRIPT)
    config_path = fb.repo_path(args.config)
    config = fb.load_yaml(config_path)
    scenes = fb.load_yaml(fb.repo_path(config["scene_policy"]))
    models = fb.load_yaml(fb.repo_path(config["models_config"]))
    settings = config["foreground_generation"]["controlled_retry"]
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
                resumed = fb.verified_existing_image(image_path, (width, height))
                started = time.monotonic()
                control = make_control(class_id, attempt, (width, height))
                if not resumed:
                    result = pipe(
                        prompt=prompt, negative_prompt=settings["negative_prompt"], image=control,
                        width=width, height=height, num_inference_steps=int(config["inference_steps"]),
                        guidance_scale=float(config["guidance_scale"]),
                        controlnet_conditioning_scale=float(settings["controlnet_conditioning_scale"]),
                        generator=torch.Generator(device="cpu").manual_seed(seed),
                    ).images[0]
                    fb.save_image_atomic(result, image_path)
                fb.save_image_atomic(control, control_path)
                added.append({
                    "candidate_id": candidate_id, "class_id": class_id,
                    "class_name": names[class_id], "image_path": str(image_path.relative_to(REPO_ROOT)),
                    "sha256": fb.sha256(image_path), "seed": seed, "prompt": prompt,
                    "control_path": str(control_path.relative_to(REPO_ROOT)),
                    "control_sha256": fb.sha256(control_path), "control_source": "synthetic_programmatic_canny",
                    "real_pixels_reused": False, "controlled_retry": True, "retry": True,
                    "seconds": 0.0 if resumed else round(time.monotonic() - started, 3),
                    "resumed_existing": resumed,
                })
                print(f"[controlled retry {len(added)}/{len(prompts)}] {candidate_id}", flush=True)
        records.extend(added)
        manifest["records"] = records
        manifest.setdefault("extensions", []).append({
            "git": fb.git_state(), "script_sha256": fb.sha256(Path(__file__).resolve()),
            "config_sha256": fb.sha256(config_path), "class_ids": class_ids,
            "count_per_class": count, "control_source": "synthetic_programmatic_canny",
            "candidate_ids": [row["candidate_id"] for row in added],
        })
        fb.write_json(manifest_path, manifest)
        rows = extension.append_review_rows(review_rows, added, "review_controlled_object_identity")
        extension.write_review_atomic(review_path, fields, rows)
    fb.release(pipe)


if __name__ == "__main__":
    main()
