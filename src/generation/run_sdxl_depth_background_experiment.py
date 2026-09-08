"""Isolated SDXL experiment: organic scene depth plus target-only Canny.

The experiment has three resumable stages. ``plates`` synthesizes target-free
rooms, ``depth`` estimates their geometry, and ``final`` uses two ControlNets
to synthesize one coherent image. Nothing from this runner is training-ready.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/generation/experiments/sdxl_depth_background_pilot.yaml"
GENERATOR = REPO_ROOT / "src/generation/generate_sdxl_dataset.py"


def load_generator():
    spec = importlib.util.spec_from_file_location("canonical_generator", GENERATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load generator helpers: {GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def validate_environment(helper, feasibility, config: dict[str, Any], require_clean: bool) -> dict[str, Any]:
    versions = feasibility.installed_versions()
    mismatches = {
        name: {"expected": expected, "actual": versions.get(name)}
        for name, expected in feasibility.EXPECTED_PACKAGES.items()
        if versions.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(f"Package-version mismatch: {json.dumps(mismatches, indent=2)}")
    git = feasibility.git_state()
    if require_clean and git["dirty"]:
        raise RuntimeError("Commit the isolated experiment before GPU generation")
    if config.get("status") != "ready_for_experiment":
        raise RuntimeError("Experiment configuration is not launch-ready")
    if len(config["scene_plate_prompts"]) != 8:
        raise ValueError("Expected eight scene families")
    for scene, prompts in config["scene_plate_prompts"].items():
        if len(prompts) != int(config["plate_variants_per_scene"]):
            raise ValueError(f"Scene {scene} has an unexpected number of variants")
    return {"git": git, "packages": versions}


def require_gpu(gpu: int) -> None:
    if not torch.cuda.is_available() or not 0 <= gpu < torch.cuda.device_count():
        raise RuntimeError(f"Logical CUDA device {gpu} is unavailable")


def load_base_pipeline(models: dict[str, Any], gpu: int):
    from diffusers import StableDiffusionXLPipeline

    base = models["sdxl"]["base_model"]
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base["id"], revision=base["revision"], torch_dtype=torch.float16,
        variant=models["sdxl"].get("variant"), use_safetensors=True,
    )
    pipe.enable_model_cpu_offload(gpu_id=gpu)
    pipe.enable_vae_slicing()
    return pipe


def plate_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    index = 0
    for scene_name, prompts in config["scene_plate_prompts"].items():
        for variant, prompt in enumerate(prompts):
            records.append({
                "index": index,
                "scene_name": scene_name,
                "variant": variant,
                "seed": int(config["plate_seed"]) + index,
                "prompt": prompt,
            })
            index += 1
    return records


def make_contact_sheet(paths: list[Path], output: Path, labels: list[str]) -> None:
    thumb = 320
    caption = 36
    canvas = Image.new("RGB", (thumb * 4, (thumb + caption) * 4), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (path, label) in enumerate(zip(paths, labels)):
        with Image.open(path).convert("RGB") as source:
            source.thumbnail((thumb, thumb))
            x = (index % 4) * thumb
            y = (index // 4) * (thumb + caption)
            canvas.paste(source, (x, y))
            draw.text((x + 5, y + thumb + 4), label, fill="black")
    canvas.save(output)


def run_plates(helper, feasibility, config: dict[str, Any], models: dict[str, Any], gpu: int, output_root: Path) -> None:
    output_dir = output_root / "plates"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    pipe = load_base_pipeline(models, gpu)
    records = []
    for position, record in enumerate(plate_records(config), start=1):
        started = time.monotonic()
        image = pipe(
            prompt=record["prompt"], negative_prompt=config["plate_negative_prompt"],
            width=int(config["output_size"][0]), height=int(config["output_size"][1]),
            num_inference_steps=int(config["plate_inference_steps"]),
            guidance_scale=float(config["plate_guidance_scale"]),
            generator=torch.Generator(device="cpu").manual_seed(record["seed"]),
        ).images[0]
        torch.cuda.synchronize(gpu)
        path = output_dir / f"{record['scene_name']}_v{record['variant']}.png"
        image.save(path)
        records.append({**record, "output": str(path.relative_to(REPO_ROOT)),
                        "sha256": helper.sha256(path),
                        "inference_seconds": round(time.monotonic() - started, 3)})
        print(f"[plates {position}/16] {path.name}", flush=True)
    paths = [REPO_ROOT / row["output"] for row in records]
    make_contact_sheet(paths, output_dir / "contact_sheet.png",
                       [f"{r['scene_name']} v{r['variant']}" for r in records])
    write_json(output_dir / "manifest.json", {
        "status": "generated_pending_visual_review", "training_use_forbidden": True,
        "records": records,
    })


def run_depth(helper, config: dict[str, Any], gpu: int, output_root: Path) -> None:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    plates_dir = output_root / "plates"
    plate_manifest = json.loads((plates_dir / "manifest.json").read_text(encoding="utf-8"))
    output_dir = output_root / "depth"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    model_spec = config["depth_estimator"]
    processor = AutoImageProcessor.from_pretrained(model_spec["id"], revision=model_spec["revision"])
    model = AutoModelForDepthEstimation.from_pretrained(
        model_spec["id"], revision=model_spec["revision"], torch_dtype=torch.float16
    ).to(f"cuda:{gpu}").eval()
    records = []
    for position, plate in enumerate(plate_manifest["records"], start=1):
        plate_path = REPO_ROOT / plate["output"]
        with Image.open(plate_path).convert("RGB") as image:
            inputs = processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(f"cuda:{gpu}", dtype=torch.float16)
            with torch.inference_mode():
                prediction = model(pixel_values=pixel_values).predicted_depth
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1), size=(image.height, image.width),
                mode="bicubic", align_corners=False,
            ).squeeze().float().cpu().numpy()
        low, high = np.percentile(prediction, [2, 98])
        normalized = np.clip((prediction - low) / max(high - low, 1e-6), 0, 1)
        depth = np.uint8(normalized * 255)
        depth_rgb = Image.fromarray(np.repeat(depth[:, :, None], 3, axis=2))
        path = output_dir / Path(plate["output"]).name
        depth_rgb.save(path)
        records.append({"scene_name": plate["scene_name"], "variant": plate["variant"],
                        "plate": plate["output"], "output": str(path.relative_to(REPO_ROOT)),
                        "sha256": helper.sha256(path)})
        print(f"[depth {position}/16] {path.name}", flush=True)
    make_contact_sheet([REPO_ROOT / row["output"] for row in records], output_dir / "contact_sheet.png",
                       [f"{r['scene_name']} v{r['variant']}" for r in records])
    write_json(output_dir / "manifest.json", {"status": "ready_for_controlnet", "records": records})


def compact_schedule(generator, config: dict[str, Any], scenes: dict[str, Any]) -> list[dict[str, Any]]:
    allowed = {int(value) for value in config["class_ids"]}
    return [row for row in generator.schedule(config, scenes) if int(row["class_id"]) in allowed]


def clear_target_depth(depth: Image.Image, box: list[int], padding: int) -> Image.Image:
    array = np.asarray(depth.convert("RGB")).copy()
    height, width = array.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = max(0, x1-padding), max(0, y1-padding), min(width, x2+padding), min(height, y2+padding)
    ring = array[max(0, y2-24):min(height, y2+24), x1:x2]
    fill = np.median(ring.reshape(-1, 3), axis=0).astype(np.uint8) if ring.size else np.array([127]*3, np.uint8)
    array[y1:y2, x1:x2] = fill
    return Image.fromarray(array)


def run_final(generator, helper, config: dict[str, Any], models: dict[str, Any], scenes: dict[str, Any], gpu: int, output_root: Path) -> None:
    from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

    depth_manifest = json.loads((output_root / "depth/manifest.json").read_text(encoding="utf-8"))
    depth_lookup = {(r["scene_name"], int(r["variant"])): r for r in depth_manifest["records"]}
    output_dir = output_root / "final"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    (output_dir / "images").mkdir(parents=True)
    (output_dir / "controls").mkdir()
    base = models["sdxl"]["base_model"]
    canny_spec = models["sdxl"]["controlnet"]
    depth_spec = config["depth_controlnet"]
    controls = [
        ControlNetModel.from_pretrained(depth_spec["id"], revision=depth_spec["revision"],
                                        torch_dtype=torch.float16, variant="fp16", use_safetensors=True),
        ControlNetModel.from_pretrained(canny_spec["id"], revision=canny_spec["revision"],
                                        torch_dtype=torch.float16, variant="fp16", use_safetensors=True),
    ]
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        base["id"], revision=base["revision"], controlnet=controls,
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    )
    pipe.enable_model_cpu_offload(gpu_id=gpu)
    pipe.enable_vae_slicing()
    manifest_path = helper.repo_path(config["silhouette_source"]["audit_manifest"])
    target_control_config = {**config, "target_conditioning_mode": "target_only"}
    scales = {int(k): v for k, v in config["controlnet_conditioning_scale"]["class_overrides"].items()}
    records = []
    schedule = compact_schedule(generator, config, scenes)
    for position, sample in enumerate(schedule, start=1):
        row = generator.select_mask(helper, config, manifest_path, sample)
        _, canny, condition = helper.build_control(target_control_config, row, sample["index"], sample["scene_name"])
        variant = int(sample["class_attempt_index"]) % int(config["plate_variants_per_scene"])
        depth_row = depth_lookup[(sample["scene_name"], variant)]
        with Image.open(REPO_ROOT / depth_row["output"]) as source:
            depth = clear_target_depth(source, condition["rendered_box_xyxy"], int(config["depth_target_clearance_px"]))
        prompt = f"{config['target_prompts'][sample['class_id']]} {config['scene_plate_prompts'][sample['scene_name']][variant]}"
        canny_scale = float(scales.get(sample["class_id"], config["controlnet_conditioning_scale"]["default"]))
        seed = int(config["seed"]) + int(sample["index"])
        started = time.monotonic()
        result = pipe(
            prompt=prompt, negative_prompt=config["final_negative_prompt"], image=[depth, canny],
            width=int(config["output_size"][0]), height=int(config["output_size"][1]),
            num_inference_steps=int(config["inference_steps"]), guidance_scale=float(config["guidance_scale"]),
            controlnet_conditioning_scale=[float(depth_spec["conditioning_scale"]), canny_scale],
            control_guidance_start=[float(depth_spec["guidance_start"]), float(config["canny_guidance_start"])],
            control_guidance_end=[float(depth_spec["guidance_end"]), float(config["canny_guidance_end"])],
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).images[0]
        torch.cuda.synchronize(gpu)
        name = f"g{sample['index']:05d}_c{sample['class_id']:02d}_{sample['scene_name']}.png"
        image_path, canny_path, depth_path = output_dir/"images"/name, output_dir/"controls"/name.replace(".png", "_canny.png"), output_dir/"controls"/name.replace(".png", "_depth.png")
        result.save(image_path); canny.save(canny_path); depth.save(depth_path)
        records.append({**sample, "plate_variant": variant, "seed": seed, "prompt": prompt,
                        "negative_prompt": config["final_negative_prompt"], "condition": condition,
                        "depth_scale": float(depth_spec["conditioning_scale"]), "canny_scale": canny_scale,
                        "output": str(image_path.relative_to(REPO_ROOT)), "sha256": helper.sha256(image_path),
                        "inference_seconds": round(time.monotonic()-started, 3), "training_use_forbidden": True})
        print(f"[final {position}/{len(schedule)}] {name}", flush=True)
    make_contact_sheet([REPO_ROOT/r["output"] for r in records], output_dir/"contact_sheet.png",
                       [f"c{r['class_id']} {r['scene_name']}" for r in records])
    write_json(output_dir/"manifest.json", {"status": "generated_pending_human_review", "records": records})
    with (output_dir/"review.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["index", "class_id", "class_name", "scene_name", "image_path", "review_status", "review_reason"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in records:
            writer.writerow({"index": row["index"], "class_id": row["class_id"], "class_name": row["class_name"],
                             "scene_name": row["scene_name"], "image_path": row["output"],
                             "review_status": "pending", "review_reason": ""})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--stage", choices=("plates", "depth", "final"), default="plates")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    generator = load_generator(); helper = generator.load_helpers(); feasibility = helper.feasibility_module()
    config_path = helper.repo_path(args.config)
    config, base_config_path = generator.load_generation_config(helper, config_path)
    models = helper.load_yaml(helper.repo_path(config["models_config"]))
    scenes = helper.load_yaml(helper.repo_path(config["scene_policy"]))
    environment = validate_environment(helper, feasibility, config, require_clean=not args.preflight_only)
    output_root = helper.repo_path(config["output_root"])
    report = {"status": "ready", "stage": args.stage, "samples": 16 if args.stage != "final" else 12,
              "output_root": str(output_root.relative_to(REPO_ROOT)), "environment": environment,
              "config_sha256": helper.sha256(config_path),
              "base_config_sha256": helper.sha256(base_config_path) if base_config_path else None}
    if args.preflight_only:
        print(json.dumps(report, indent=2)); return
    require_gpu(args.gpu)
    if args.stage == "plates": run_plates(helper, feasibility, config, models, args.gpu, output_root)
    elif args.stage == "depth": run_depth(helper, config, args.gpu, output_root)
    else: run_final(generator, helper, config, models, scenes, args.gpu, output_root)


if __name__ == "__main__":
    main()
