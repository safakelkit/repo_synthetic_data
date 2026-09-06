"""Generate the training-forbidden two-stage SDXL Aerosol pilot."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFilter


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/generation/genai_aerosol_two_stage_pilot_v8.yaml"
HELPERS = REPO_ROOT / "src/generation/run_all_class_genai_pilot.py"


def load_helpers():
    spec = importlib.util.spec_from_file_location("genai_helpers", HELPERS)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load helpers: {HELPERS}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def padded_mask(size: tuple[int, int], box: list[int], padding_ratio: float, blur: int) -> Image.Image:
    width, height = size
    x1, y1, x2, y2 = box
    padding = round(max(x2 - x1, y2 - y1) * padding_ratio)
    bounds = (max(0, x1 - padding), max(0, y1 - padding), min(width, x2 + padding), min(height, y2 + padding))
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle(bounds, fill=255)
    return mask.filter(ImageFilter.GaussianBlur(blur)) if blur else mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    helper = load_helpers()
    feasibility = helper.feasibility_module()
    config_path = helper.repo_path(args.config)
    config = helper.load_yaml(config_path)
    models_path = helper.repo_path(config["models_config"])
    scene_path = helper.repo_path(config["scene_policy"])
    models = helper.load_yaml(models_path)
    scenes = helper.load_yaml(scene_path)
    manifest_path = helper.repo_path(config["silhouette_source"]["audit_manifest"])
    rows = helper.read_masks(manifest_path, 11)
    selected = {}
    for stem in config["silhouette_source"]["preferred_stems"]:
        matches = [row for row in rows if row["stem"] == stem]
        if len(matches) != 1:
            raise ValueError(f"Aerosol silhouette is not uniquely accepted: {stem}")
        selected[stem] = matches[0]

    scene_names = scenes["class_to_scene_families"][11]
    schedule = [
        {"index": repeat * len(scene_names) + slot, "repeat": repeat, "scene": name, "stem": config["silhouette_source"]["preferred_stems"][slot]}
        for repeat in range(int(config["images_per_scene"]))
        for slot, name in enumerate(scene_names)
    ]
    if len(schedule) != int(config["expected_images"]):
        raise ValueError("Two-stage schedule size does not match expected_images")
    git = feasibility.git_state()
    versions = feasibility.installed_versions()
    mismatches = {name: {"expected": expected, "actual": versions.get(name)} for name, expected in feasibility.EXPECTED_PACKAGES.items() if versions.get(name) != expected}
    if mismatches:
        raise RuntimeError(f"Package-version mismatch: {json.dumps(mismatches, indent=2)}")
    if git["dirty"]:
        raise RuntimeError("Git worktree is dirty; commit the frozen pilot before launch")
    output_dir = helper.repo_path(config["output_root"]) / "sdxl"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    preflight = {"status": "ready", "samples": len(schedule), "output": str(output_dir.relative_to(REPO_ROOT)), "git": git, "packages": versions, "config_sha256": helper.sha256(config_path), "mask_manifest_sha256": helper.sha256(manifest_path)}
    if args.preflight_only:
        print(json.dumps(preflight, indent=2))
        return
    if not torch.cuda.is_available() or not 0 <= args.gpu < torch.cuda.device_count():
        raise RuntimeError(f"Logical CUDA device {args.gpu} is unavailable")

    from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline

    started = time.monotonic()
    run_started_utc = feasibility.utc_now()
    base_pipe = feasibility.load_pipeline("sdxl", models, args.gpu)
    output_dir.mkdir(parents=True)
    for name in ("scenes", "images", "masks", "controls"):
        (output_dir / name).mkdir()

    scene_records = []
    scene_images = {}
    for sample in schedule:
        scene_config = {
            **config,
            "output_size": config["output_size"],
            "seed": config["seed"],
            "target_conditioning_mode": "scene_only",
            "control_layout": config["scene_control_layout"],
        }
        _, control, condition = helper.build_control(scene_config, None, sample["index"], sample["scene"])
        description = scenes["scene_families"][sample["scene"]]["description"]
        prompt = config["stage_1_scene"]["prompt"].format(scene=description.rstrip("."))
        seed = int(config["seed"]) + sample["index"]
        generator = torch.Generator(device="cpu").manual_seed(seed)
        generated = base_pipe(prompt=prompt, negative_prompt=config["stage_1_scene"]["negative_prompt"], image=control, width=config["output_size"][0], height=config["output_size"][1], num_inference_steps=int(config["inference_steps"]), guidance_scale=float(config["guidance_scale"]), controlnet_conditioning_scale=float(config["stage_1_scene"]["controlnet_conditioning_scale"]), generator=generator).images[0]
        scene_path_out = output_dir / "scenes" / f"{sample['index']:03d}_{sample['scene']}.png"
        control_path = output_dir / "controls" / f"{sample['index']:03d}_scene_canny.png"
        generated.save(scene_path_out)
        control.save(control_path)
        scene_images[sample["index"]] = generated
        scene_records.append({**sample, "seed": seed, "prompt": prompt, "condition": condition, "scene_output": str(scene_path_out.relative_to(REPO_ROOT)), "scene_sha256": helper.sha256(scene_path_out), "scene_control_sha256": helper.sha256(control_path)})

    del base_pipe
    gc.collect()
    torch.cuda.empty_cache()
    sdxl = models["sdxl"]
    controlnet = ControlNetModel.from_pretrained(sdxl["controlnet"]["id"], revision=sdxl["controlnet"]["revision"], torch_dtype=torch.float16, variant="fp16")
    inpaint = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(sdxl["inpaint_model"]["id"], revision=sdxl["inpaint_model"]["revision"], controlnet=controlnet, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    inpaint.enable_model_cpu_offload(gpu_id=args.gpu)
    records = []
    width, height = config["output_size"]
    for sample, scene_record in zip(schedule, scene_records):
        target_config = {
            **config,
            "output_size": config["output_size"],
            "seed": int(config["seed"]) + 10000,
            "target_conditioning_mode": "target_only",
            "control_layout": config["target_control_layout"],
            "class_rotation_degrees": {11: [0]},
        }
        _, target_control, condition = helper.build_control(target_config, selected[sample["stem"]], sample["index"], sample["scene"])
        mask = padded_mask((width, height), condition["rendered_box_xyxy"], float(config["stage_2_inpaint"]["mask_padding_ratio"]), int(config["stage_2_inpaint"]["mask_blur_radius"]))
        prompt = config["stage_2_inpaint"]["prompt_variants"][sample["repeat"]]
        seed = int(config["seed"]) + 10000 + sample["index"]
        generator = torch.Generator(device="cpu").manual_seed(seed)
        result = inpaint(prompt=prompt, negative_prompt=config["stage_2_inpaint"]["negative_prompt"], image=scene_images[sample["index"]], mask_image=mask, control_image=target_control, width=width, height=height, strength=float(config["stage_2_inpaint"]["strength"]), num_inference_steps=int(config["inference_steps"]), guidance_scale=float(config["guidance_scale"]), controlnet_conditioning_scale=float(config["stage_2_inpaint"]["controlnet_conditioning_scale"]), control_guidance_start=float(config["stage_2_inpaint"]["control_guidance_start"]), control_guidance_end=float(config["stage_2_inpaint"]["control_guidance_end"]), generator=generator).images[0]
        final = Image.composite(result, scene_images[sample["index"]], mask)
        output_path = output_dir / "images" / f"{sample['index']:03d}_c11_{sample['scene']}.png"
        mask_path = output_dir / "masks" / f"{sample['index']:03d}_inpaint_mask.png"
        control_path = output_dir / "controls" / f"{sample['index']:03d}_target_canny.png"
        final.save(output_path)
        mask.save(mask_path)
        target_control.save(control_path)
        records.append({**scene_record, "stage_2_seed": seed, "stage_2_prompt": prompt, "target_condition": condition, "output": str(output_path.relative_to(REPO_ROOT)), "output_sha256": helper.sha256(output_path), "mask_sha256": helper.sha256(mask_path), "target_control_sha256": helper.sha256(control_path), "annotation_performed": False, "degradation_applied": False, "training_use_forbidden": True})
        print(f"[{sample['index'] + 1}/{len(schedule)}] {output_path.name}", flush=True)

    manifest = {"format_version": 1, "pilot_id": config["pilot_id"], "status": "generated_pending_human_review", "started_utc": run_started_utc, "completed_utc": feasibility.utc_now(), "wall_time_seconds": round(time.monotonic() - started, 3), "git": git, "packages": versions, "gpu_name": torch.cuda.get_device_name(args.gpu), "peak_allocated_gib": round(torch.cuda.max_memory_allocated(args.gpu) / 1024**3, 3), "models": {"base": sdxl["base_model"], "controlnet": sdxl["controlnet"], "inpaint": sdxl["inpaint_model"]}, "config_sha256": helper.sha256(config_path), "records": records}
    with (output_dir / "pilot_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
