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

    base = models["sdxl"]["base_model"]
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base["id"], revision=base["revision"], torch_dtype=torch.float16,
        variant=models["sdxl"].get("variant"), use_safetensors=True,
    )
    pipe.enable_model_cpu_offload(gpu_id=gpu)
    pipe.vae.enable_slicing()
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
    rows = max(1, (len(paths) + 3) // 4)
    canvas = Image.new("RGB", (thumb * 4, (thumb + caption) * rows), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (path, label) in enumerate(zip(paths, labels)):
        with Image.open(path).convert("RGB") as source:
            source.thumbnail((thumb, thumb))
            x = (index % 4) * thumb
            y = (index // 4) * (thumb + caption)
            canvas.paste(source, (x, y))
            draw.text((x + 5, y + thumb + 4), label, fill="black")
    canvas.save(output)


def run_plates(
    helper, feasibility, config: dict[str, Any], models: dict[str, Any], gpu: int,
    output_root: Path, records_to_generate: list[dict[str, Any]], output_name: str,
) -> None:
    output_dir = output_root / output_name
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    pipe = load_base_pipeline(models, gpu)
    records = []
    for position, record in enumerate(records_to_generate, start=1):
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
        print(f"[plates {position}/{len(records_to_generate)}] {path.name}", flush=True)
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
        print(f"[depth {position}/{len(plate_manifest['records'])}] {path.name}", flush=True)
    make_contact_sheet([REPO_ROOT / row["output"] for row in records], output_dir / "contact_sheet.png",
                       [f"{r['scene_name']} v{r['variant']}" for r in records])
    write_json(output_dir / "manifest.json", {"status": "ready_for_controlnet", "records": records})


def compact_schedule(generator, config: dict[str, Any], scenes: dict[str, Any]) -> list[dict[str, Any]]:
    allowed = {int(value) for value in config["class_ids"]}
    return [row for row in generator.schedule(config, scenes) if int(row["class_id"]) in allowed]


def clear_target_depth(depth: Image.Image, box: list[int], padding: int) -> Image.Image:
    if padding < 0:
        return depth.convert("RGB").copy()
    array = np.asarray(depth.convert("RGB")).copy()
    height, width = array.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = max(0, x1-padding), max(0, y1-padding), min(width, x2+padding), min(height, y2+padding)
    ring = array[max(0, y2-24):min(height, y2+24), x1:x2]
    fill = np.median(ring.reshape(-1, 3), axis=0).astype(np.uint8) if ring.size else np.array([127]*3, np.uint8)
    array[y1:y2, x1:x2] = fill
    return Image.fromarray(array)


def run_final(generator, helper, config: dict[str, Any], models: dict[str, Any], scenes: dict[str, Any], gpu: int, output_root: Path, final_name: str, sample_indices: set[int] | None) -> None:
    from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

    depth_manifest = json.loads((output_root / "depth/manifest.json").read_text(encoding="utf-8"))
    depth_lookup = {(r["scene_name"], int(r["variant"])): r for r in depth_manifest["records"]}
    output_dir = output_root / final_name
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
    pipe.vae.enable_slicing()
    manifest_path = helper.repo_path(config["silhouette_source"]["audit_manifest"])
    target_control_config = {**config, "target_conditioning_mode": "target_only"}
    scales = {int(k): v for k, v in config["controlnet_conditioning_scale"]["class_overrides"].items()}
    records = []
    schedule = compact_schedule(generator, config, scenes)
    if sample_indices is not None:
        schedule = [row for row in schedule if int(row["index"]) in sample_indices]
        missing = sample_indices - {int(row["index"]) for row in schedule}
        if missing:
            raise ValueError(f"Requested sample indices are unavailable: {sorted(missing)}")
    for position, sample in enumerate(schedule, start=1):
        row = generator.select_mask(helper, config, manifest_path, sample)
        _, canny, condition = helper.build_control(target_control_config, row, sample["index"], sample["scene_name"])
        variant = int(sample["class_attempt_index"]) % int(config["plate_variants_per_scene"])
        depth_row = depth_lookup[(sample["scene_name"], variant)]
        with Image.open(REPO_ROOT / depth_row["output"]) as source:
            depth = clear_target_depth(source, condition["rendered_box_xyxy"], int(config["depth_target_clearance_px"]))
        prompt = f"{config['target_prompts'][sample['class_id']]} {config['final_scene_prompts'][sample['scene_name']]}"
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


def run_img2img(generator, helper, config: dict[str, Any], models: dict[str, Any], scenes: dict[str, Any], gpu: int, output_root: Path, final_name: str, sample_indices: set[int] | None) -> None:
    """Preserve an organic plate while Canny introduces the target object."""
    from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline

    output_dir = output_root / final_name
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    (output_dir / "images").mkdir(parents=True)
    (output_dir / "controls").mkdir()
    base = models["sdxl"]["base_model"]
    canny_spec = models["sdxl"]["controlnet"]
    controlnet = ControlNetModel.from_pretrained(
        canny_spec["id"], revision=canny_spec["revision"], torch_dtype=torch.float16,
        variant="fp16", use_safetensors=True,
    )
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        base["id"], revision=base["revision"], controlnet=controlnet,
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    )
    pipe.enable_model_cpu_offload(gpu_id=gpu)
    pipe.vae.enable_slicing()
    manifest_path = helper.repo_path(config["silhouette_source"]["audit_manifest"])
    target_control_config = {**config, "target_conditioning_mode": "target_only"}
    scale_config = config["img2img_controlnet_conditioning_scale"]
    scales = {int(k): v for k, v in scale_config["class_overrides"].items()}
    schedule = compact_schedule(generator, config, scenes)
    if sample_indices is not None:
        schedule = [row for row in schedule if int(row["index"]) in sample_indices]
        missing = sample_indices - {int(row["index"]) for row in schedule}
        if missing:
            raise ValueError(f"Requested sample indices are unavailable: {sorted(missing)}")
    records = []
    for position, sample in enumerate(schedule, start=1):
        row = generator.select_mask(helper, config, manifest_path, sample)
        _, canny, condition = helper.build_control(target_control_config, row, sample["index"], sample["scene_name"])
        variant = int(sample["class_attempt_index"]) % int(config["plate_variants_per_scene"])
        plate_path = output_root / "plates" / f"{sample['scene_name']}_v{variant}.png"
        with Image.open(plate_path).convert("RGB") as source:
            plate = source.copy()
        prompt = f"{config['target_prompts'][sample['class_id']]} {config['final_scene_prompts'][sample['scene_name']]}"
        canny_scale = float(scales.get(sample["class_id"], scale_config["default"]))
        seed = int(config["seed"]) + int(sample["index"])
        started = time.monotonic()
        result = pipe(
            prompt=prompt, negative_prompt=config["final_negative_prompt"], image=plate,
            control_image=canny, strength=float(config["img2img_strength"]),
            width=int(config["output_size"][0]), height=int(config["output_size"][1]),
            num_inference_steps=int(config["inference_steps"]), guidance_scale=float(config["guidance_scale"]),
            controlnet_conditioning_scale=canny_scale,
            control_guidance_start=float(config["canny_guidance_start"]),
            control_guidance_end=float(config["canny_guidance_end"]),
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).images[0]
        torch.cuda.synchronize(gpu)
        name = f"g{sample['index']:05d}_c{sample['class_id']:02d}_{sample['scene_name']}.png"
        image_path = output_dir / "images" / name
        canny_path = output_dir / "controls" / name.replace(".png", "_canny.png")
        result.save(image_path)
        canny.save(canny_path)
        records.append({**sample, "plate_variant": variant, "plate": str(plate_path.relative_to(REPO_ROOT)),
                        "seed": seed, "prompt": prompt, "condition": condition,
                        "strength": float(config["img2img_strength"]), "canny_scale": canny_scale,
                        "output": str(image_path.relative_to(REPO_ROOT)), "sha256": helper.sha256(image_path),
                        "inference_seconds": round(time.monotonic()-started, 3), "training_use_forbidden": True})
        print(f"[img2img {position}/{len(schedule)}] {name}", flush=True)
    make_contact_sheet([REPO_ROOT/r["output"] for r in records], output_dir/"contact_sheet.png",
                       [f"c{r['class_id']} {r['scene_name']}" for r in records])
    write_json(output_dir/"manifest.json", {"status": "generated_pending_human_review", "records": records})


def run_direct_aerosol(generator, helper, config: dict[str, Any], models: dict[str, Any], scenes: dict[str, Any], gpu: int, output_root: Path, final_name: str) -> None:
    """Generate class 11 directly as a coherent full scene without ControlNet."""
    output_dir = output_root / final_name
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    (output_dir / "images").mkdir(parents=True)
    pipe = load_base_pipeline(models, gpu)
    aerosol_prompts = [
        config["direct_aerosol_prompt"].format(scene=scene.lower().rstrip("."))
        for scene in config["final_scene_prompts"].values()
    ]
    validate_clip_prompts(pipe, [*aerosol_prompts, config["direct_aerosol_negative_prompt"]])
    schedule = [row for row in compact_schedule(generator, config, scenes) if int(row["class_id"]) == 11]
    records = []
    for position, sample in enumerate(schedule, start=1):
        scene = config["final_scene_prompts"][sample["scene_name"]]
        prompt = config["direct_aerosol_prompt"].format(scene=scene.lower().rstrip("."))
        seed = int(config["seed"]) + int(sample["index"])
        started = time.monotonic()
        result = pipe(
            prompt=prompt, negative_prompt=config["direct_aerosol_negative_prompt"],
            width=int(config["output_size"][0]), height=int(config["output_size"][1]),
            num_inference_steps=int(config["inference_steps"]), guidance_scale=float(config["guidance_scale"]),
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).images[0]
        torch.cuda.synchronize(gpu)
        name = f"g{sample['index']:05d}_c11_{sample['scene_name']}.png"
        image_path = output_dir / "images" / name
        result.save(image_path)
        records.append({**sample, "seed": seed, "prompt": prompt,
                        "output": str(image_path.relative_to(REPO_ROOT)), "sha256": helper.sha256(image_path),
                        "inference_seconds": round(time.monotonic()-started, 3), "training_use_forbidden": True})
        print(f"[direct aerosol {position}/{len(schedule)}] {name}", flush=True)
    make_contact_sheet([REPO_ROOT/r["output"] for r in records], output_dir/"contact_sheet.png",
                       [r["scene_name"] for r in records])
    write_json(output_dir/"manifest.json", {"status": "generated_pending_human_review", "records": records})


def run_inpaint_aerosol(generator, helper, config: dict[str, Any], models: dict[str, Any], scenes: dict[str, Any], gpu: int, output_root: Path, final_name: str, sample_indices: set[int] | None) -> None:
    """Diffuse aerosol into a plate region while preserving installed scenery."""
    from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline

    output_dir = output_root / final_name
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    (output_dir / "images").mkdir(parents=True)
    (output_dir / "controls").mkdir()
    base = models["sdxl"]["base_model"]
    canny_spec = models["sdxl"]["controlnet"]
    controlnet = ControlNetModel.from_pretrained(
        canny_spec["id"], revision=canny_spec["revision"], torch_dtype=torch.float16,
        variant="fp16", use_safetensors=True,
    )
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        base["id"], revision=base["revision"], controlnet=controlnet,
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    )
    pipe.enable_model_cpu_offload(gpu_id=gpu)
    pipe.vae.enable_slicing()
    aerosol_prompts = [
        config["direct_aerosol_prompt"].format(scene=scene.lower().rstrip("."))
        for scene in config["final_scene_prompts"].values()
    ]
    validate_clip_prompts(pipe, [*aerosol_prompts, config["direct_aerosol_negative_prompt"]])
    manifest_path = helper.repo_path(config["silhouette_source"]["audit_manifest"])
    target_control_config = {**config, "target_conditioning_mode": "target_only"}
    schedule = [row for row in compact_schedule(generator, config, scenes) if int(row["class_id"]) == 11]
    if sample_indices is not None:
        schedule = [row for row in schedule if int(row["index"]) in sample_indices]
        missing = sample_indices - {int(row["index"]) for row in schedule}
        if missing:
            raise ValueError(f"Requested aerosol sample indices are unavailable: {sorted(missing)}")
    records = []
    for position, sample in enumerate(schedule, start=1):
        row = generator.select_mask(helper, config, manifest_path, sample)
        _, canny, condition = helper.build_control(target_control_config, row, sample["index"], sample["scene_name"])
        keep_components = int(config.get("inpaint_canny_keep_largest_components", 0))
        if keep_components:
            canny_array = np.asarray(canny.convert("L"))
            component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
                (canny_array > 127).astype(np.uint8), connectivity=8
            )
            ranked = sorted(range(1, component_count), key=lambda label: int(stats[label, cv2.CC_STAT_AREA]), reverse=True)
            retained = np.isin(labels, ranked[:keep_components]).astype(np.uint8) * 255
            canny = Image.fromarray(retained)
        variant = int(sample["class_attempt_index"]) % int(config["plate_variants_per_scene"])
        plate_path = output_root / "plates" / f"{sample['scene_name']}_v{variant}.png"
        with Image.open(plate_path).convert("RGB") as source:
            plate = source.copy()
        initial_image = plate
        render_x1, render_y1, render_x2, render_y2 = condition["rendered_box_xyxy"]
        if bool(config.get("inpaint_source_initialization", False)):
            rgba_path = helper.repo_path(row["rgba_path"])
            with Image.open(rgba_path).convert("RGBA") as source:
                target = source.resize(
                    (render_x2-render_x1, render_y2-render_y1), Image.Resampling.LANCZOS
                )
            canvas = plate.convert("RGBA")
            canvas.alpha_composite(target, dest=(render_x1, render_y1))
            initial_image = canvas.convert("RGB")
        mask_array = np.zeros((plate.height, plate.width), dtype=np.uint8)
        padding = int(config["inpaint_mask_padding_px"])
        x1, y1, x2, y2 = render_x1, render_y1, render_x2, render_y2
        x1, y1 = max(0, x1-padding), max(0, y1-padding)
        x2, y2 = min(plate.width, x2+padding), min(plate.height, y2+padding)
        cv2.rectangle(mask_array, (x1, y1), (x2, y2), 255, thickness=-1)
        mask_array = cv2.GaussianBlur(mask_array, (0, 0), sigmaX=5)
        mask = Image.fromarray(mask_array)
        scene = config["final_scene_prompts"][sample["scene_name"]]
        prompt = config["direct_aerosol_prompt"].format(scene=scene.lower().rstrip("."))
        seed = int(config["seed"]) + int(sample["index"])
        started = time.monotonic()
        result = pipe(
            prompt=prompt, negative_prompt=config["direct_aerosol_negative_prompt"],
            image=initial_image, mask_image=mask, control_image=canny,
            strength=float(config["inpaint_strength"]),
            width=plate.width, height=plate.height,
            num_inference_steps=int(config["inference_steps"]), guidance_scale=float(config["guidance_scale"]),
            controlnet_conditioning_scale=float(config["inpaint_canny_scale"]),
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).images[0]
        torch.cuda.synchronize(gpu)
        name = f"g{sample['index']:05d}_c11_{sample['scene_name']}.png"
        image_path = output_dir / "images" / name
        canny_path = output_dir / "controls" / name.replace(".png", "_canny.png")
        mask_path = output_dir / "controls" / name.replace(".png", "_mask.png")
        init_path = output_dir / "controls" / name.replace(".png", "_init.png")
        result.save(image_path); canny.save(canny_path); mask.save(mask_path)
        if initial_image is not plate:
            initial_image.save(init_path)
        records.append({**sample, "plate_variant": variant, "seed": seed, "prompt": prompt,
                        "mask_xyxy": [x1, y1, x2, y2], "condition": condition,
                        "source_initialization": bool(config.get("inpaint_source_initialization", False)),
                        "output": str(image_path.relative_to(REPO_ROOT)), "sha256": helper.sha256(image_path),
                        "inference_seconds": round(time.monotonic()-started, 3), "training_use_forbidden": True})
        print(f"[inpaint aerosol {position}/{len(schedule)}] {name}", flush=True)
    make_contact_sheet([REPO_ROOT/r["output"] for r in records], output_dir/"contact_sheet.png",
                       [r["scene_name"] for r in records])
    write_json(output_dir/"manifest.json", {"status": "generated_pending_human_review", "records": records})


def rotate_rgba_like_control(source: Image.Image, angle: float) -> Image.Image:
    """Crop and rotate RGBA with the same geometry used by ``rotate_binary``."""
    rgba = np.asarray(source.convert("RGBA"))
    alpha = rgba[:, :, 3]
    ys, xs = np.where(alpha > 0)
    if not xs.size:
        raise ValueError("Source RGBA has an empty alpha channel")
    rgba = rgba[ys.min():ys.max()+1, xs.min():xs.max()+1]
    height, width = rgba.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_width = int(height * sin + width * cos)
    new_height = int(height * cos + width * sin)
    matrix[0, 2] += new_width / 2 - width / 2
    matrix[1, 2] += new_height / 2 - height / 2
    rotated = cv2.warpAffine(
        rgba, matrix, (new_width, new_height), flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0),
    )
    return Image.fromarray(rotated, mode="RGBA")


def crop_rgba(source: Image.Image) -> Image.Image:
    """Crop transparent margins without changing the object's camera pose."""
    rgba = np.asarray(source.convert("RGBA"))
    ys, xs = np.where(rgba[:, :, 3] > 0)
    if not xs.size:
        raise ValueError("Source RGBA has an empty alpha channel")
    return Image.fromarray(rgba[ys.min():ys.max()+1, xs.min():xs.max()+1], mode="RGBA")


def _unit(vector: list[float]) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-6:
        raise ValueError(f"Zero-length placement axis: {vector}")
    return value / norm


def warp_flat_rgba(
    source: Image.Image,
    canvas_size: tuple[int, int],
    placement: dict[str, Any],
    long_px: float,
) -> tuple[Image.Image, dict[str, Any]]:
    """Lay an RGBA crop onto a photographed support plane using a homography."""
    target = crop_rgba(source)
    if target.height > target.width:
        target = target.transpose(Image.Transpose.ROTATE_90)
    rgba = np.asarray(target)
    height, width = rgba.shape[:2]
    center = np.asarray(placement["center_xy"], dtype=np.float32)
    axis_u = _unit(placement["surface_axis_u"])
    axis_v = _unit(placement["surface_axis_v"])
    compression = float(placement.get("depth_compression", 0.55))
    short_px = np.clip(long_px * height / max(width, 1) * compression, 22.0, long_px * 0.58)
    half_u, half_v = axis_u * (long_px / 2.0), axis_v * (short_px / 2.0)
    quad = np.asarray([
        center - half_u - half_v,
        center + half_u - half_v,
        center + half_u + half_v,
        center - half_u + half_v,
    ], dtype=np.float32)
    source_quad = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source_quad, quad)
    canvas_width, canvas_height = canvas_size
    warped = cv2.warpPerspective(
        rgba, matrix, (canvas_width, canvas_height), flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0),
    )
    return Image.fromarray(warped, mode="RGBA"), {
        "mode": "lying_flat_homography",
        "center_xy": center.round(2).tolist(),
        "destination_quad_xy": quad.round(2).tolist(),
        "surface_axis_u": axis_u.round(6).tolist(),
        "surface_axis_v": axis_v.round(6).tolist(),
        "long_px": round(float(long_px), 3),
        "short_px": round(float(short_px), 3),
        "depth_compression": compression,
    }


def place_supported_upright_rgba(
    source: Image.Image,
    canvas_size: tuple[int, int],
    placement: dict[str, Any],
    target_height: float,
) -> tuple[Image.Image, dict[str, Any]]:
    """Resize an upright/hinged object and pin its bottom edge to the support."""
    target = crop_rgba(source)
    scale = float(target_height) / target.height
    width = max(1, round(target.width * scale))
    height = max(1, round(target.height * scale))
    max_width = int(placement.get("upright_max_width_px", 320))
    if width > max_width:
        scale = max_width / target.width
        width, height = max_width, max(1, round(target.height * scale))
    target = target.resize((width, height), Image.Resampling.LANCZOS)
    canvas_width, canvas_height = canvas_size
    anchor_x, baseline_y = [
        int(round(v)) for v in placement.get("upright_anchor_xy", placement["center_xy"])
    ]
    paste_x = int(np.clip(anchor_x - width // 2, 0, canvas_width - width))
    paste_y = int(np.clip(baseline_y - height, 0, canvas_height - height))
    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    canvas.alpha_composite(target, dest=(paste_x, paste_y))
    return canvas, {
        "mode": "upright_bottom_anchored",
        "support_anchor_xy": [anchor_x, baseline_y],
        "rendered_box_xyxy": [paste_x, paste_y, paste_x + width, paste_y + height],
        "target_height_px": height,
    }


def pose_condition(
    config: dict[str, Any], source: Image.Image, class_id: int,
    scene_name: str, variant: int, canvas_size: tuple[int, int],
) -> tuple[Image.Image, Image.Image, Image.Image, dict[str, Any]]:
    """Build a pose-aware source layer, alpha mask and target-only Canny."""
    modes = {int(key): str(value) for key, value in config["pose_class_modes"].items()}
    sizes = {int(key): float(value) for key, value in config["pose_target_size_px"].items()}
    mode = modes[class_id]
    placement = dict(config["pose_plane_placements"][scene_name][variant])
    depth_overrides = {
        int(key): float(value)
        for key, value in config.get("pose_depth_compression_overrides", {}).items()
    }
    if class_id in depth_overrides:
        placement["depth_compression"] = depth_overrides[class_id]
    offsets = {
        int(key): [float(v) for v in value]
        for key, value in config.get("pose_anchor_offset_xy", {}).items()
    }
    if class_id in offsets:
        anchor = placement.get("upright_anchor_xy", placement["center_xy"])
        placement["upright_anchor_xy"] = [
            float(anchor[0]) + offsets[class_id][0],
            float(anchor[1]) + offsets[class_id][1],
        ]
    anchor_overrides = config.get("pose_anchor_xy_overrides", {})
    class_overrides = anchor_overrides.get(class_id, anchor_overrides.get(str(class_id), {}))
    scene_overrides = class_overrides.get(scene_name, {})
    anchor_override = scene_overrides.get(variant, scene_overrides.get(str(variant)))
    if anchor_override is not None:
        placement["upright_anchor_xy"] = [float(anchor_override[0]), float(anchor_override[1])]
    if mode == "flat":
        layer, metadata = warp_flat_rgba(source, canvas_size, placement, sizes[class_id])
    elif mode in ("upright", "hinged"):
        layer, metadata = place_supported_upright_rgba(
            source, canvas_size, placement, sizes[class_id]
        )
        metadata["mode"] = "hinged_bottom_anchored" if mode == "hinged" else metadata["mode"]
    else:
        raise ValueError(f"Unsupported pose mode for class {class_id}: {mode}")

    rgba = np.asarray(layer)
    alpha = rgba[:, :, 3]
    binary = np.where(alpha > 8, 255, 0).astype(np.uint8)
    if not np.any(binary):
        raise ValueError(f"Pose rendering produced an empty object for class {class_id}")
    canny = cv2.Canny(cv2.GaussianBlur(binary, (0, 0), sigmaX=0.8), 32, 96)
    internal_classes = {
        int(value) for value in config.get("pose_internal_edge_class_ids", [])
    }
    if class_id in internal_classes:
        gray = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2GRAY)
        internal = cv2.Canny(cv2.GaussianBlur(gray, (0, 0), sigmaX=0.8), 55, 140)
        internal[cv2.dilate(binary, np.ones((3, 3), np.uint8)) == 0] = 0
        canny = cv2.bitwise_or(canny, internal)
    padding = int(config.get("pose_mask_padding_px", 20))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (padding * 2 + 1, padding * 2 + 1))
    mask = cv2.dilate(binary, kernel)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=4)
    ys, xs = np.where(binary > 0)
    metadata["alpha_box_xyxy"] = [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)]
    metadata["class_pose_mode"] = mode
    return layer, Image.fromarray(mask), Image.fromarray(cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)), metadata


def composite_pose_layer(plate: Image.Image, layer: Image.Image, mode: str) -> Image.Image:
    """Add a restrained contact shadow before compositing the real RGBA pixels."""
    canvas = plate.convert("RGBA")
    alpha = np.asarray(layer)[:, :, 3]
    if mode == "flat":
        shadow = cv2.GaussianBlur(alpha, (0, 0), sigmaX=5)
        shifted = np.zeros_like(shadow)
        shifted[4:, :] = shadow[:-4, :]
        opacity = 0.28
    else:
        ys, xs = np.where(alpha > 8)
        shifted = np.zeros_like(alpha)
        center = (int(round((xs.min() + xs.max()) / 2)), int(ys.max()))
        half_width = max(9, int(round((xs.max() - xs.min()) * 0.38)))
        half_height = max(3, int(round(half_width * 0.13)))
        cv2.ellipse(shifted, center, (half_width, half_height), 0, 0, 360, 255, -1)
        shifted = cv2.GaussianBlur(shifted, (0, 0), sigmaX=4)
        opacity = 0.34
    shadow_layer = np.zeros((*shifted.shape, 4), dtype=np.uint8)
    shadow_layer[:, :, 3] = np.uint8(shifted.astype(np.float32) * opacity)
    canvas = Image.alpha_composite(canvas, Image.fromarray(shadow_layer, mode="RGBA"))
    return Image.alpha_composite(canvas, layer).convert("RGB")


def run_inpaint_all(generator, helper, config: dict[str, Any], models: dict[str, Any], scenes: dict[str, Any], gpu: int, output_root: Path, final_name: str, sample_indices: set[int] | None) -> None:
    """Insert every class into an organic target-free plate using target-only Canny."""
    from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline

    output_dir = output_root / final_name
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    (output_dir / "images").mkdir(parents=True)
    (output_dir / "controls").mkdir()
    base = models["sdxl"]["base_model"]
    canny_spec = models["sdxl"]["controlnet"]
    controlnet = ControlNetModel.from_pretrained(
        canny_spec["id"], revision=canny_spec["revision"], torch_dtype=torch.float16,
        variant="fp16", use_safetensors=True,
    )
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        base["id"], revision=base["revision"], controlnet=controlnet,
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    )
    pipe.enable_model_cpu_offload(gpu_id=gpu)
    pipe.vae.enable_slicing()

    target_prompts = {int(key): value for key, value in config["inpaint_all_target_prompts"].items()}
    template = str(config["inpaint_all_prompt_template"])
    prompts = [
        template.format(target=target_prompts[class_id], scene=scene.lower().rstrip("."))
        for class_id in sorted(target_prompts)
        for scene in config["final_scene_prompts"].values()
    ]
    validate_clip_prompts(pipe, [*prompts, config["inpaint_all_negative_prompt"]])

    manifest_path = helper.repo_path(config["silhouette_source"]["audit_manifest"])
    target_control_config = {**config, "target_conditioning_mode": "target_only"}
    schedule = compact_schedule(generator, config, scenes)
    if sample_indices is not None:
        schedule = [row for row in schedule if int(row["index"]) in sample_indices]
        missing = sample_indices - {int(row["index"]) for row in schedule}
        if missing:
            raise ValueError(f"Requested sample indices are unavailable: {sorted(missing)}")

    strengths = {int(key): float(value) for key, value in config.get("inpaint_all_strength_overrides", {}).items()}
    paddings = {int(key): int(value) for key, value in config.get("inpaint_all_mask_padding_overrides", {}).items()}
    canny_scales = {int(key): float(value) for key, value in config.get("inpaint_all_canny_scale_overrides", {}).items()}
    component_limits = {int(key): int(value) for key, value in config.get("inpaint_all_keep_components_overrides", {}).items()}
    records = []
    for position, sample in enumerate(schedule, start=1):
        class_id = int(sample["class_id"])
        row = generator.select_mask(helper, config, manifest_path, sample)
        variant = int(sample["class_attempt_index"]) % int(config["plate_variants_per_scene"])
        sample_control_config = target_control_config
        plate_boxes = config.get("inpaint_all_plate_target_boxes", {})
        if class_id != 11 and sample["scene_name"] in plate_boxes:
            target_box = plate_boxes[sample["scene_name"]][variant]
            class_scene_overrides = {
                int(key): value
                for key, value in target_control_config.get("class_scene_layout_overrides", {}).items()
            }
            class_override = dict(class_scene_overrides.get(class_id, {}))
            class_override[sample["scene_name"]] = {
                **class_override.get(sample["scene_name"], {}),
                "target_box_xyxy": target_box,
            }
            class_scene_overrides[class_id] = class_override
            sample_control_config = {
                **target_control_config,
                "class_scene_layout_overrides": class_scene_overrides,
            }
        _, canny, condition = helper.build_control(
            sample_control_config, row, sample["index"], sample["scene_name"]
        )
        keep_components = component_limits.get(
            class_id, int(config.get("inpaint_all_keep_components_default", 0))
        )
        if keep_components:
            canny_array = np.asarray(canny.convert("L"))
            component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
                (canny_array > 127).astype(np.uint8), connectivity=8
            )
            ranked = sorted(
                range(1, component_count),
                key=lambda label: int(stats[label, cv2.CC_STAT_AREA]), reverse=True,
            )
            canny = Image.fromarray(
                np.isin(labels, ranked[:keep_components]).astype(np.uint8) * 255
            )

        plate_path = output_root / "plates" / f"{sample['scene_name']}_v{variant}.png"
        with Image.open(plate_path).convert("RGB") as source:
            plate = source.copy()
        render_x1, render_y1, render_x2, render_y2 = condition["rendered_box_xyxy"]
        rgba_path = helper.repo_path(row["rgba_path"])
        with Image.open(rgba_path).convert("RGBA") as source:
            target = rotate_rgba_like_control(source, float(condition["rotation_degrees"]))
            target = target.resize(
                (render_x2-render_x1, render_y2-render_y1), Image.Resampling.LANCZOS
            )
        canvas = plate.convert("RGBA")
        canvas.alpha_composite(target, dest=(render_x1, render_y1))
        initial_image = canvas.convert("RGB")

        padding = paddings.get(class_id, int(config["inpaint_all_mask_padding_default_px"]))
        x1, y1 = max(0, render_x1-padding), max(0, render_y1-padding)
        x2, y2 = min(plate.width, render_x2+padding), min(plate.height, render_y2+padding)
        mask_array = np.zeros((plate.height, plate.width), dtype=np.uint8)
        cv2.rectangle(mask_array, (x1, y1), (x2, y2), 255, thickness=-1)
        mask_array = cv2.GaussianBlur(mask_array, (0, 0), sigmaX=5)
        mask = Image.fromarray(mask_array)

        scene = config["final_scene_prompts"][sample["scene_name"]]
        prompt = template.format(target=target_prompts[class_id], scene=scene.lower().rstrip("."))
        seed = int(config["seed"]) + int(sample["index"])
        strength = strengths.get(class_id, float(config["inpaint_all_strength_default"]))
        canny_scale = canny_scales.get(class_id, float(config["inpaint_all_canny_scale_default"]))
        started = time.monotonic()
        result = pipe(
            prompt=prompt, negative_prompt=config["inpaint_all_negative_prompt"],
            image=initial_image, mask_image=mask, control_image=canny,
            strength=strength, width=plate.width, height=plate.height,
            num_inference_steps=int(config["inference_steps"]),
            guidance_scale=float(config["guidance_scale"]),
            controlnet_conditioning_scale=canny_scale,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).images[0]
        torch.cuda.synchronize(gpu)

        name = f"g{sample['index']:05d}_c{class_id:02d}_{sample['scene_name']}.png"
        image_path = output_dir / "images" / name
        canny_path = output_dir / "controls" / name.replace(".png", "_canny.png")
        mask_path = output_dir / "controls" / name.replace(".png", "_mask.png")
        init_path = output_dir / "controls" / name.replace(".png", "_init.png")
        result.save(image_path)
        canny.save(canny_path)
        mask.save(mask_path)
        initial_image.save(init_path)
        records.append({
            **sample, "plate_variant": variant, "plate": str(plate_path.relative_to(REPO_ROOT)),
            "seed": seed, "prompt": prompt, "negative_prompt": config["inpaint_all_negative_prompt"],
            "condition": condition, "mask_xyxy": [x1, y1, x2, y2],
            "source_initialization": True, "source_rgba": str(rgba_path.relative_to(REPO_ROOT)),
            "strength": strength, "canny_scale": canny_scale,
            "output": str(image_path.relative_to(REPO_ROOT)), "sha256": helper.sha256(image_path),
            "inference_seconds": round(time.monotonic()-started, 3), "training_use_forbidden": True,
        })
        print(f"[inpaint all {position}/{len(schedule)}] {name}", flush=True)

    make_contact_sheet(
        [REPO_ROOT/r["output"] for r in records], output_dir/"contact_sheet.png",
        [f"c{r['class_id']:02d} {r['class_name']} | {r['scene_name']}" for r in records],
    )
    write_json(output_dir/"manifest.json", {
        "status": "generated_pending_human_review", "training_use_forbidden": True,
        "records": records,
    })
    with (output_dir/"review.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["index", "class_id", "class_name", "scene_name", "image_path", "review_status", "review_reason"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({
                "index": row["index"], "class_id": row["class_id"], "class_name": row["class_name"],
                "scene_name": row["scene_name"], "image_path": row["output"],
                "review_status": "pending", "review_reason": "",
            })


def build_pose_sample(
    generator, helper, config: dict[str, Any], output_root: Path,
    sample: dict[str, Any], manifest_path: Path,
) -> dict[str, Any]:
    """Render one deterministic pose-aware initialization package."""
    class_id = int(sample["class_id"])
    variant = int(sample["class_attempt_index"]) % int(config["plate_variants_per_scene"])
    row = generator.select_mask(helper, config, manifest_path, sample)
    plate_path = output_root / "plates" / f"{sample['scene_name']}_v{variant}.png"
    rgba_path = helper.repo_path(row["rgba_path"])
    with Image.open(plate_path).convert("RGB") as source:
        plate = source.copy()
    with Image.open(rgba_path).convert("RGBA") as source:
        layer, mask, canny, pose = pose_condition(
            config, source, class_id, sample["scene_name"], variant, plate.size
        )
    mode = pose["class_pose_mode"]
    initial_image = composite_pose_layer(plate, layer, mode)
    return {
        "sample": sample, "class_id": class_id, "variant": variant, "row": row,
        "plate": plate, "plate_path": plate_path, "rgba_path": rgba_path,
        "layer": layer, "mask": mask, "canny": canny, "pose": pose,
        "initial_image": initial_image,
    }


def pose_schedule(generator, config: dict[str, Any], scenes: dict[str, Any], sample_indices: set[int] | None) -> list[dict[str, Any]]:
    schedule = compact_schedule(generator, config, scenes)
    if sample_indices is not None:
        schedule = [row for row in schedule if int(row["index"]) in sample_indices]
        missing = sample_indices - {int(row["index"]) for row in schedule}
        if missing:
            raise ValueError(f"Requested sample indices are unavailable: {sorted(missing)}")
    return schedule


def run_pose_preview(
    generator, helper, config: dict[str, Any], scenes: dict[str, Any],
    output_root: Path, final_name: str, sample_indices: set[int] | None,
) -> None:
    """Render source initializations without loading a diffusion model."""
    output_dir = output_root / final_name
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    (output_dir / "images").mkdir(parents=True)
    (output_dir / "controls").mkdir()
    manifest_path = helper.repo_path(config["silhouette_source"]["audit_manifest"])
    records = []
    schedule = pose_schedule(generator, config, scenes, sample_indices)
    for position, sample in enumerate(schedule, start=1):
        package = build_pose_sample(generator, helper, config, output_root, sample, manifest_path)
        class_id = package["class_id"]
        name = f"g{sample['index']:05d}_c{class_id:02d}_{sample['scene_name']}.png"
        image_path = output_dir / "images" / name
        package["initial_image"].save(image_path)
        package["canny"].save(output_dir / "controls" / name.replace(".png", "_canny.png"))
        package["mask"].save(output_dir / "controls" / name.replace(".png", "_mask.png"))
        package["layer"].save(output_dir / "controls" / name.replace(".png", "_layer.png"))
        records.append({
            **sample, "plate_variant": package["variant"],
            "plate": str(package["plate_path"].relative_to(REPO_ROOT)),
            "source_rgba": str(package["rgba_path"].relative_to(REPO_ROOT)),
            "pose": package["pose"], "output": str(image_path.relative_to(REPO_ROOT)),
            "sha256": helper.sha256(image_path), "training_use_forbidden": True,
        })
        print(f"[pose preview {position}/{len(schedule)}] {name}", flush=True)
    make_contact_sheet(
        [REPO_ROOT / row["output"] for row in records], output_dir / "contact_sheet.png",
        [f"c{r['class_id']:02d} {r['pose']['class_pose_mode']} | {r['scene_name']}" for r in records],
    )
    write_json(output_dir / "manifest.json", {
        "status": "source_initialization_preview_only", "training_use_forbidden": True,
        "records": records,
    })


def run_inpaint_pose(
    generator, helper, config: dict[str, Any], models: dict[str, Any], scenes: dict[str, Any],
    gpu: int, output_root: Path, final_name: str, sample_indices: set[int] | None,
) -> None:
    """Inpaint pose-aware real-object initializations into reviewed organic plates."""
    from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline

    output_dir = output_root / final_name
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    (output_dir / "images").mkdir(parents=True)
    (output_dir / "controls").mkdir()
    base = models["sdxl"]["base_model"]
    canny_spec = models["sdxl"]["controlnet"]
    controlnet = ControlNetModel.from_pretrained(
        canny_spec["id"], revision=canny_spec["revision"], torch_dtype=torch.float16,
        variant="fp16", use_safetensors=True,
    )
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        base["id"], revision=base["revision"], controlnet=controlnet,
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    )
    pipe.enable_model_cpu_offload(gpu_id=gpu)
    pipe.vae.enable_slicing()

    target_prompts = {int(key): value for key, value in config["inpaint_all_target_prompts"].items()}
    pose_phrases = {str(key): str(value) for key, value in config["pose_prompt_phrases"].items()}
    template = str(config["pose_prompt_template"])
    prompts = [
        template.format(
            target=target_prompts[class_id], pose=pose_phrases[config["pose_class_modes"][class_id]],
            scene=scene.lower().rstrip("."),
        )
        for class_id in sorted(target_prompts)
        for scene in config["final_scene_prompts"].values()
    ]
    validate_clip_prompts(pipe, [*prompts, config["pose_negative_prompt"]])
    manifest_path = helper.repo_path(config["silhouette_source"]["audit_manifest"])
    schedule = pose_schedule(generator, config, scenes, sample_indices)
    mode_strengths = {str(k): float(v) for k, v in config["pose_strength_by_mode"].items()}
    mode_scales = {str(k): float(v) for k, v in config["pose_canny_scale_by_mode"].items()}
    class_strengths = {int(k): float(v) for k, v in config.get("pose_strength_overrides", {}).items()}
    class_scales = {int(k): float(v) for k, v in config.get("pose_canny_scale_overrides", {}).items()}
    records = []
    for position, sample in enumerate(schedule, start=1):
        package = build_pose_sample(generator, helper, config, output_root, sample, manifest_path)
        class_id = package["class_id"]
        mode = package["pose"]["class_pose_mode"]
        scene = config["final_scene_prompts"][sample["scene_name"]]
        prompt = template.format(
            target=target_prompts[class_id], pose=pose_phrases[mode], scene=scene.lower().rstrip(".")
        )
        strength = class_strengths.get(class_id, mode_strengths[mode])
        canny_scale = class_scales.get(class_id, mode_scales[mode])
        seed = int(config["seed"]) + int(sample["index"])
        started = time.monotonic()
        result = pipe(
            prompt=prompt, negative_prompt=config["pose_negative_prompt"],
            image=package["initial_image"], mask_image=package["mask"],
            control_image=package["canny"], strength=strength,
            width=package["plate"].width, height=package["plate"].height,
            num_inference_steps=int(config["inference_steps"]),
            guidance_scale=float(config["guidance_scale"]),
            controlnet_conditioning_scale=canny_scale,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).images[0]
        torch.cuda.synchronize(gpu)
        name = f"g{sample['index']:05d}_c{class_id:02d}_{sample['scene_name']}.png"
        image_path = output_dir / "images" / name
        result.save(image_path)
        package["initial_image"].save(output_dir / "controls" / name.replace(".png", "_init.png"))
        package["canny"].save(output_dir / "controls" / name.replace(".png", "_canny.png"))
        package["mask"].save(output_dir / "controls" / name.replace(".png", "_mask.png"))
        records.append({
            **sample, "plate_variant": package["variant"],
            "plate": str(package["plate_path"].relative_to(REPO_ROOT)), "seed": seed,
            "prompt": prompt, "negative_prompt": config["pose_negative_prompt"],
            "pose": package["pose"], "source_initialization": True,
            "source_rgba": str(package["rgba_path"].relative_to(REPO_ROOT)),
            "strength": strength, "canny_scale": canny_scale,
            "output": str(image_path.relative_to(REPO_ROOT)), "sha256": helper.sha256(image_path),
            "inference_seconds": round(time.monotonic() - started, 3),
            "training_use_forbidden": True,
        })
        print(f"[inpaint pose {position}/{len(schedule)}] {name}", flush=True)
    make_contact_sheet(
        [REPO_ROOT / row["output"] for row in records], output_dir / "contact_sheet.png",
        [f"c{r['class_id']:02d} {r['class_name']} | {r['scene_name']}" for r in records],
    )
    write_json(output_dir / "manifest.json", {
        "status": "generated_pending_human_review", "training_use_forbidden": True,
        "records": records,
    })
    with (output_dir / "review.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["index", "class_id", "class_name", "scene_name", "image_path", "review_status", "review_reason"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in records:
            writer.writerow({
                "index": row["index"], "class_id": row["class_id"], "class_name": row["class_name"],
                "scene_name": row["scene_name"], "image_path": row["output"],
                "review_status": "pending", "review_reason": "",
            })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--stage",
        choices=("plates", "depth", "final", "img2img", "direct_aerosol", "inpaint_aerosol",
                 "inpaint_all", "pose_preview", "inpaint_pose"),
        default="plates",
    )
    parser.add_argument("--final-name", default="final")
    parser.add_argument(
        "--plate-key", action="append", default=[],
        help="Generate only one scene variant, formatted as scene_name:variant; repeatable",
    )
    parser.add_argument(
        "--plate-output-name", default="plates",
        help="Output directory below output_root for the plates stage",
    )
    parser.add_argument("--sample-index", type=int, action="append", dest="sample_indices")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    generator = load_generator(); helper = generator.load_helpers(); feasibility = helper.feasibility_module()
    config_path = helper.repo_path(args.config)
    config, base_config_path = generator.load_generation_config(helper, config_path)
    models = helper.load_yaml(helper.repo_path(config["models_config"]))
    scenes = helper.load_yaml(helper.repo_path(config["scene_policy"]))
    environment = validate_environment(helper, feasibility, config, require_clean=not args.preflight_only)
    output_root = helper.repo_path(config["output_root"])
    all_plate_records = plate_records(config)
    requested_plate_keys = set(args.plate_key)
    known_plate_keys = {
        f"{row['scene_name']}:{row['variant']}" for row in all_plate_records
    }
    unknown_plate_keys = requested_plate_keys - known_plate_keys
    if unknown_plate_keys:
        raise ValueError(f"Unknown plate keys: {sorted(unknown_plate_keys)}")
    selected_plate_records = [
        row for row in all_plate_records
        if not requested_plate_keys
        or f"{row['scene_name']}:{row['variant']}" in requested_plate_keys
    ]
    stage_samples = {
        "plates": len(selected_plate_records), "depth": len(all_plate_records),
        "final": len(compact_schedule(generator, config, scenes)),
        "img2img": len(compact_schedule(generator, config, scenes)), "direct_aerosol": 4,
        "inpaint_aerosol": 4, "inpaint_all": len(compact_schedule(generator, config, scenes)),
        "pose_preview": len(compact_schedule(generator, config, scenes)),
        "inpaint_pose": len(compact_schedule(generator, config, scenes)),
    }
    report = {"status": "ready", "stage": args.stage, "samples": stage_samples[args.stage],
              "output_root": str(output_root.relative_to(REPO_ROOT)), "environment": environment,
              "config_sha256": helper.sha256(config_path),
              "base_config_sha256": helper.sha256(base_config_path) if base_config_path else None}
    if args.preflight_only:
        print(json.dumps(report, indent=2)); return
    if args.stage != "pose_preview":
        require_gpu(args.gpu)
    if args.stage == "plates":
        run_plates(
            helper, feasibility, config, models, args.gpu, output_root,
            selected_plate_records, args.plate_output_name,
        )
    elif args.stage == "depth": run_depth(helper, config, args.gpu, output_root)
    elif args.stage == "final":
        run_final(generator, helper, config, models, scenes, args.gpu, output_root,
                  args.final_name, set(args.sample_indices) if args.sample_indices else None)
    elif args.stage == "img2img":
        run_img2img(generator, helper, config, models, scenes, args.gpu, output_root,
                    args.final_name, set(args.sample_indices) if args.sample_indices else None)
    elif args.stage == "direct_aerosol":
        run_direct_aerosol(generator, helper, config, models, scenes, args.gpu,
                           output_root, args.final_name)
    elif args.stage == "inpaint_aerosol":
        run_inpaint_aerosol(generator, helper, config, models, scenes, args.gpu,
                            output_root, args.final_name,
                            set(args.sample_indices) if args.sample_indices else None)
    elif args.stage == "inpaint_all":
        run_inpaint_all(generator, helper, config, models, scenes, args.gpu,
                        output_root, args.final_name,
                        set(args.sample_indices) if args.sample_indices else None)
    elif args.stage == "pose_preview":
        run_pose_preview(generator, helper, config, scenes, output_root, args.final_name,
                         set(args.sample_indices) if args.sample_indices else None)
    else:
        run_inpaint_pose(generator, helper, config, models, scenes, args.gpu,
                         output_root, args.final_name,
                         set(args.sample_indices) if args.sample_indices else None)


if __name__ == "__main__":
    main()
