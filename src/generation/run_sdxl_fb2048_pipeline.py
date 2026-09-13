"""Generate the review-gated SDXL-FB2048 foreground/background pipeline.

Stages deliberately stop at human review boundaries.  Run ``backgrounds`` and
``foregrounds``, mark their review CSV rows accepted, run ``segment``, review
the masks, and only then run ``compose``.  Production stays locked until the
all-class pilot is approved and the config is explicitly revised.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import subprocess
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/generation/sdxl_fb2048_v1.yaml"
CLASS_COUNT = 16


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def git_state() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip())
    return {"revision": revision, "dirty": dirty}


def release(pipe: Any) -> None:
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def validate_prompts(pipe: Any, prompts: list[str]) -> None:
    for tokenizer_name in ("tokenizer", "tokenizer_2"):
        tokenizer = getattr(pipe, tokenizer_name)
        limit = int(tokenizer.model_max_length)
        for prompt in prompts:
            length = len(tokenizer(prompt, truncation=False)["input_ids"])
            if length > limit:
                raise ValueError(f"{tokenizer_name} prompt length {length} exceeds {limit}: {prompt}")


def background_prompt(config: dict[str, Any], scene_name: str, attempt: int) -> str:
    settings = config["background_generation"]
    template = settings["prompt_templates"][(attempt // 4) % len(settings["prompt_templates"])]
    return template.format(
        scene=scene_name.replace("_", " "),
        support=settings["support_profiles"][attempt % len(settings["support_profiles"])],
        layout=settings["layout_profiles"][(attempt // 2) % len(settings["layout_profiles"])],
        wear=settings["wear_profiles"][(attempt // 3) % len(settings["wear_profiles"])],
        occupancy=settings["occupancy_profiles"][(attempt // 5) % len(settings["occupancy_profiles"])],
        camera=settings["camera_profiles"][attempt % len(settings["camera_profiles"])],
        lighting=settings["lighting_profiles"][(attempt // 2) % len(settings["lighting_profiles"])],
    )


def background_prompts(config: dict[str, Any], scenes: dict[str, Any]) -> list[str]:
    settings = config["background_generation"]
    maximum_attempts = max(
        int(config["production"]["background_candidates"]) // CLASS_COUNT,
        int(config["pilot"]["background_candidates"]) // CLASS_COUNT,
    )
    return [
        background_prompt(config, scene_name, attempt)
        for attempt in range(maximum_attempts)
        for scene_name in scenes["scene_families"]
    ] + [settings["negative_prompt"]]


def foreground_prompts(config: dict[str, Any]) -> list[str]:
    settings = config["foreground_generation"]
    return [
        settings["prompt_template"].format(target=target, variation=variation)
        for target in settings["class_prompts"].values()
        for variation in settings["variation_phrases"]
    ] + [settings["negative_prompt"]]


def harmonization_prompts(config: dict[str, Any], scenes: dict[str, Any]) -> list[str]:
    settings = config["harmonization"]
    return [
        settings["prompt_template"].format(
            target=target, scene=scene_name.replace("_", " "))
        for target in config["foreground_generation"]["class_prompts"].values()
        for scene_name in scenes["scene_families"]
    ] + [settings["negative_prompt"]]


def validate_config(config: dict[str, Any], scenes: dict[str, Any], mode: str) -> dict[str, int]:
    if config.get("status") != "ready_for_all_class_pilot":
        raise ValueError("SDXL-FB2048 config is not released for the all-class pilot")
    if config.get("training_use_forbidden") is not True or config.get("target_test_used") is not False:
        raise ValueError("Pilot must be training-forbidden and target-test independent")
    if len(config["foreground_generation"]["class_prompts"]) != CLASS_COUNT:
        raise ValueError("Exactly 16 immutable class prompts are required")
    if len(scenes["class_policy"]["names"]) != CLASS_COUNT:
        raise ValueError("Scene policy must contain exactly 16 classes")
    policy = config[mode]
    if mode == "production" and policy.get("locked", True):
        raise RuntimeError("Production is locked until the all-class pilot is reviewed")
    final = int(policy["final_images"])
    if mode == "production" and final != 2048:
        raise ValueError("Production must contain exactly 2,048 images")
    return {
        "final": final,
        "background_candidates": int(policy["background_candidates"]),
        "accepted_backgrounds": int(policy["accepted_backgrounds"]),
        "foreground_candidates_per_class": int(policy["foreground_candidates_per_class"]),
        "accepted_foregrounds_per_class": int(policy["accepted_foregrounds_per_class"]),
    }


def load_base_pipeline(models: dict[str, Any], gpu: int):
    from diffusers import StableDiffusionXLPipeline
    model = models["sdxl"]
    base = model["base_model"]
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base["id"], revision=base["revision"], dtype=torch.float16,
        variant=model["variant"], use_safetensors=True,
    )
    pipe.enable_model_cpu_offload(gpu_id=gpu)
    pipe.vae.enable_slicing()
    return pipe


def load_harmonizer(models: dict[str, Any], gpu: int):
    from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline
    model = models["sdxl"]
    control = model["controlnet"]
    controlnet = ControlNetModel.from_pretrained(
        control["id"], revision=control["revision"], dtype=torch.float16,
        variant=model["variant"], use_safetensors=True,
    )
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        model["base_model"]["id"], revision=model["base_model"]["revision"],
        controlnet=controlnet, dtype=torch.float16,
        variant=model["variant"], use_safetensors=True,
    )
    pipe.enable_model_cpu_offload(gpu_id=gpu)
    pipe.vae.enable_slicing()
    return pipe


def require_launch_ready(gpu: int, output: Path, allow_partial: bool = False) -> None:
    state = git_state()
    if state["dirty"]:
        raise RuntimeError("Commit the frozen pilot before GPU generation")
    if (output / "manifest.json").exists():
        raise FileExistsError(f"Stage is already complete: {output}")
    if output.exists() and not allow_partial:
        raise FileExistsError(f"Refusing to overwrite existing stage: {output}")
    if not torch.cuda.is_available() or not 0 <= gpu < torch.cuda.device_count():
        raise RuntimeError(f"Logical CUDA device {gpu} is unavailable")


@contextmanager
def stage_lock(root: Path, stage: str):
    root.mkdir(parents=True, exist_ok=True)
    lock = root / f".{stage}.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Stage lock already exists; another {stage} process may be active: {lock}") from exc
    try:
        os.write(descriptor, f"pid={os.getpid()}\nstarted={time.time()}\n".encode())
        os.close(descriptor)
        yield
    finally:
        lock.unlink(missing_ok=True)


def save_image_atomic(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.partial{path.suffix}")
    image.save(temporary)
    temporary.replace(path)


def verified_existing_image(path: Path, expected_size: tuple[int, int]) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            image.load()
            if image.size != expected_size:
                raise ValueError(f"Partial image has wrong size: {path} ({image.size})")
    except Exception as exc:
        raise ValueError(f"Partial image is not a valid resumable output: {path}") from exc
    return True


def write_review(path: Path, records: list[dict[str, Any]], kind: str) -> None:
    fields = ["candidate_id", "class_id", "class_name", "scene_name", "image_path",
              "review_status", "review_reason"]
    temporary = path.with_name(f".{path.name}.partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({
                "candidate_id": row["candidate_id"],
                "class_id": row.get("class_id", ""),
                "class_name": row.get("class_name", ""),
                "scene_name": row.get("scene_name", ""),
                "image_path": row["image_path"],
                "review_status": "pending",
                "review_reason": f"review_{kind}",
            })
    temporary.replace(path)


def accepted_records(manifest_path: Path, review_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.is_file() or not review_path.is_file():
        raise FileNotFoundError(f"Missing manifest or manual review file: {manifest_path}, {review_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError(f"Invalid manifest records: {manifest_path}")
    by_id = {str(row["candidate_id"]): row for row in records}
    if len(by_id) != len(records):
        raise ValueError(f"Duplicate candidate IDs in manifest: {manifest_path}")
    with review_path.open("r", encoding="utf-8", newline="") as handle:
        review = list(csv.DictReader(handle))
    review_ids = [row["candidate_id"] for row in review]
    if len(set(review_ids)) != len(review_ids) or set(review_ids) != set(by_id):
        raise ValueError("Review candidate IDs must match the immutable manifest exactly")
    accepted_ids = [
        row["candidate_id"] for row in review
        if row["review_status"].strip().lower() == "accepted"
    ]
    return [by_id[candidate_id] for candidate_id in accepted_ids]


def require_unique_files(records: list[dict[str, Any]], path_key: str, label: str) -> None:
    hashes = [sha256(repo_path(row[path_key])) for row in records]
    if len(set(hashes)) != len(hashes):
        raise ValueError(f"Exact duplicate detected among accepted {label}")


def generate_backgrounds(config: dict[str, Any], scenes: dict[str, Any], counts: dict[str, int], root: Path, gpu: int) -> None:
    output = root / "backgrounds"
    require_launch_ready(gpu, output, allow_partial=True)
    images = output / "images"
    pipe = load_base_pipeline(load_yaml(repo_path(config["models_config"])), gpu)
    settings = config["background_generation"]
    names = {int(key): value for key, value in scenes["class_policy"]["names"].items()}
    assignments = {int(key): value for key, value in scenes["class_to_scene_families"].items()}
    width, height = map(int, config["output_size"])
    records = []
    validate_prompts(pipe, background_prompts(config, scenes))
    images.mkdir(parents=True)
    base, remainder = divmod(counts["background_candidates"], CLASS_COUNT)
    schedule = [
        (class_id, attempt)
        for class_id in range(CLASS_COUNT)
        for attempt in range(base + (1 if class_id < remainder else 0))
    ]
    if len(schedule) != counts["background_candidates"]:
        raise AssertionError("Background candidate schedule has the wrong size")
    for index, (class_id, attempt) in enumerate(schedule):
        scene_slot = attempt % len(assignments[class_id])
        scene_name = assignments[class_id][scene_slot]
        prompt = background_prompt(config, scene_name, attempt)
        seed = int(config["seed"]) + index
        path = images / f"bg_{index:05d}_{scene_name}.png"
        resumed = verified_existing_image(path, (width, height))
        started = time.monotonic()
        if not resumed:
            image = pipe(
                prompt=prompt, negative_prompt=settings["negative_prompt"], width=width, height=height,
                num_inference_steps=int(config["inference_steps"]), guidance_scale=float(config["guidance_scale"]),
                generator=torch.Generator(device="cpu").manual_seed(seed),
            ).images[0]
            save_image_atomic(image, path)
        records.append({"candidate_id": f"bg_c{class_id:02d}_{attempt:04d}",
                        "class_id": class_id, "class_name": names[class_id],
                        "planned_class_id": class_id, "scene_slot": scene_slot,
                        "scene_name": scene_name,
                        "image_path": str(path.relative_to(REPO_ROOT)), "sha256": sha256(path),
                        "seed": seed, "prompt": prompt,
                        "seconds": 0.0 if resumed else round(time.monotonic() - started, 3),
                        "resumed_existing": resumed})
        action = "resume" if resumed else "generate"
        print(f"[background {index + 1}/{counts['background_candidates']}] {action} {path.name}", flush=True)
    release(pipe)
    write_json(output / "manifest.json", {"status": "pending_manual_review", "git": git_state(),
               "config_sha256": config["_config_sha256"], "models_sha256": config["_models_sha256"],
               "script_sha256": sha256(Path(__file__).resolve()), "records": records})
    write_review(output / "review.csv", records, "target_free_realistic_unique_background")


def foreground_schedule(config: dict[str, Any], counts: dict[str, int], mode: str) -> list[tuple[int, int]]:
    rows = []
    for class_id in range(CLASS_COUNT):
        count = counts["foreground_candidates_per_class"]
        if mode == "production" and class_id == 11:
            count = int(config["production"]["aerosol_foreground_candidates"])
        rows.extend((class_id, attempt) for attempt in range(count))
    return rows


def generate_foregrounds(config: dict[str, Any], scenes: dict[str, Any], counts: dict[str, int], root: Path, gpu: int, mode: str) -> None:
    output = root / "foregrounds"
    require_launch_ready(gpu, output, allow_partial=True)
    images = output / "images"
    pipe = load_base_pipeline(load_yaml(repo_path(config["models_config"])), gpu)
    settings = config["foreground_generation"]
    width, height = map(int, config["output_size"])
    prompts = {int(key): value for key, value in settings["class_prompts"].items()}
    names = {int(key): value for key, value in scenes["class_policy"]["names"].items()}
    records = []
    schedule = foreground_schedule(config, counts, mode)
    validate_prompts(pipe, foreground_prompts(config))
    images.mkdir(parents=True)
    for index, (class_id, attempt) in enumerate(schedule):
        prompt = settings["prompt_template"].format(
            target=prompts[class_id],
            variation=settings["variation_phrases"][attempt % len(settings["variation_phrases"])],
        )
        seed = int(config["seed"]) + 100000 + class_id * 10000 + attempt
        path = images / f"fg_c{class_id:02d}_{attempt:04d}.png"
        resumed = verified_existing_image(path, (width, height))
        started = time.monotonic()
        if not resumed:
            image = pipe(
                prompt=prompt, negative_prompt=settings["negative_prompt"], width=width, height=height,
                num_inference_steps=int(config["inference_steps"]), guidance_scale=float(config["guidance_scale"]),
                generator=torch.Generator(device="cpu").manual_seed(seed),
            ).images[0]
            save_image_atomic(image, path)
        records.append({"candidate_id": f"fg_c{class_id:02d}_{attempt:04d}", "class_id": class_id,
                        "class_name": names[class_id], "image_path": str(path.relative_to(REPO_ROOT)),
                        "sha256": sha256(path), "seed": seed, "prompt": prompt,
                        "seconds": 0.0 if resumed else round(time.monotonic() - started, 3),
                        "resumed_existing": resumed})
        action = "resume" if resumed else "generate"
        print(f"[foreground {index + 1}/{len(schedule)}] {action} {path.name}", flush=True)
    release(pipe)
    write_json(output / "manifest.json", {"status": "pending_manual_review", "git": git_state(),
               "config_sha256": config["_config_sha256"], "models_sha256": config["_models_sha256"],
               "script_sha256": sha256(Path(__file__).resolve()), "records": records})
    write_review(output / "review.csv", records, "single_correct_complete_unique_object")


def best_sam_mask(processed: dict[str, Any], shape: tuple[int, int]) -> tuple[np.ndarray, float]:
    masks = processed.get("masks")
    scores = processed.get("scores")
    if masks is None or len(masks) == 0:
        raise ValueError("SAM3 returned no mask")
    best = int(torch.as_tensor(scores).argmax().item()) if scores is not None and len(scores) else 0
    mask = masks[best]
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask).squeeze()
    if mask.shape != shape:
        mask = cv2.resize(mask.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8), float(scores[best]) if scores is not None and len(scores) else 0.0


def segment_foregrounds(config: dict[str, Any], counts: dict[str, int], root: Path, gpu: int) -> None:
    from transformers import Sam3Model, Sam3Processor
    source = root / "foregrounds"
    accepted = accepted_records(source / "manifest.json", source / "review.csv")
    require_unique_files(accepted, "image_path", "foreground candidates")
    per_class = Counter(int(row["class_id"]) for row in accepted)
    needed = counts["accepted_foregrounds_per_class"]
    missing = {class_id: needed - per_class[class_id] for class_id in range(CLASS_COUNT) if per_class[class_id] < needed}
    if missing:
        raise ValueError(f"Insufficient manually accepted foregrounds by class: {missing}")
    output = root / "segmented_foregrounds"
    require_launch_ready(gpu, output)
    (output / "masks").mkdir(parents=True)
    (output / "rgba").mkdir()
    settings = config["segmentation"]
    processor = Sam3Processor.from_pretrained(settings["model_id"])
    model = Sam3Model.from_pretrained(settings["model_id"]).to(f"cuda:{gpu}").eval()
    records = []
    for index, row in enumerate(accepted):
        path = repo_path(row["image_path"])
        with Image.open(path).convert("RGB") as image:
            rgb = image.copy()
        width, height = rgb.size
        margin = int(settings["prompt_box_margin_px"])
        inputs = processor(images=rgb, input_boxes=[[[margin, margin, width-margin, height-margin]]],
                           input_boxes_labels=[[1]], return_tensors="pt").to(f"cuda:{gpu}")
        with torch.no_grad():
            outputs = model(**inputs)
        processed = processor.post_process_instance_segmentation(
            outputs, threshold=float(settings["score_threshold"]),
            mask_threshold=float(settings["mask_threshold"]),
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]
        mask, score = best_sam_mask(processed, (height, width))
        components, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if components <= 1:
            raise ValueError(f"Empty SAM3 mask: {path}")
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (labels == largest).astype(np.uint8)
        ys, xs = np.where(mask > 0)
        area_ratio = float(mask.mean())
        clearance = int(settings["border_clearance_px"])
        valid = (float(settings["min_mask_area_ratio"]) <= area_ratio <= float(settings["max_mask_area_ratio"])
                 and xs.min() >= clearance and ys.min() >= clearance
                 and xs.max() < width-clearance and ys.max() < height-clearance)
        candidate_id = row["candidate_id"]
        mask_path = output / "masks" / f"{candidate_id}.png"
        rgba_path = output / "rgba" / f"{candidate_id}.png"
        Image.fromarray(mask * 255).save(mask_path)
        rgba = np.dstack((np.asarray(rgb), mask * 255)).astype(np.uint8)
        Image.fromarray(rgba).save(rgba_path)
        records.append({"candidate_id": candidate_id, "class_id": int(row["class_id"]),
                        "class_name": row["class_name"], "source_path": row["image_path"],
                        "mask_path": str(mask_path.relative_to(REPO_ROOT)),
                        "rgba_path": str(rgba_path.relative_to(REPO_ROOT)), "sam_score": score,
                        "mask_area_ratio": area_ratio, "automatic_geometry_valid": valid})
        print(f"[segment {index + 1}/{len(accepted)}] {candidate_id} valid={valid}", flush=True)
    del model, processor
    release(None)
    write_json(output / "manifest.json", {"status": "pending_mask_review", "git": git_state(),
               "config_sha256": config["_config_sha256"], "models_sha256": config["_models_sha256"],
               "script_sha256": sha256(Path(__file__).resolve()), "records": records})
    write_review(output / "review.csv", [dict(row, image_path=row["rgba_path"], scene_name="") for row in records], "mask_and_rgba")


def target_controls(mask: Image.Image, padding: int) -> tuple[Image.Image, Image.Image]:
    binary = (np.asarray(mask.convert("L")) > 0).astype(np.uint8) * 255
    canny = cv2.Canny(cv2.GaussianBlur(binary, (0, 0), 0.8), 32, 96)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (padding * 2 + 1, padding * 2 + 1))
    # Diffusion may create contact shadow and repair the surrounding seam, but
    # it must never regenerate the accepted foreground or invalidate its mask.
    inpaint = cv2.dilate(binary, kernel)
    inpaint[binary > 0] = 0
    inpaint = cv2.GaussianBlur(inpaint, (0, 0), 4)
    inpaint[binary > 0] = 0
    return Image.fromarray(inpaint), Image.fromarray(cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB))


def color_match_foreground(
    foreground: Image.Image, background: Image.Image, x: int, y: int, strength: float,
) -> Image.Image:
    rgba = np.asarray(foreground.convert("RGBA")).copy()
    alpha = rgba[:, :, 3] > 8
    if not alpha.any() or strength <= 0:
        return foreground
    background_rgb = np.asarray(background.convert("RGB"))
    patch = background_rgb[y:y + foreground.height, x:x + foreground.width]
    if patch.size == 0:
        return foreground
    object_lab = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2LAB).astype(np.float32)
    patch_lab = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB).astype(np.float32)
    object_mean = object_lab[alpha].mean(axis=0)
    patch_mean = patch_lab.reshape(-1, 3).mean(axis=0)
    shift = (patch_mean - object_mean) * float(strength)
    object_lab[alpha] = np.clip(object_lab[alpha] + shift, 0, 255)
    rgba[:, :, :3] = cv2.cvtColor(object_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgba)


def compose(config: dict[str, Any], scenes: dict[str, Any], counts: dict[str, int], root: Path, gpu: int) -> None:
    backgrounds = accepted_records(root / "backgrounds/manifest.json", root / "backgrounds/review.csv")
    foregrounds = accepted_records(
        root / "segmented_foregrounds/manifest.json", root / "segmented_foregrounds/review.csv"
    )
    require_unique_files(backgrounds, "image_path", "backgrounds")
    require_unique_files(foregrounds, "rgba_path", "segmented foregrounds")
    invalid_geometry = [row["candidate_id"] for row in foregrounds if not row["automatic_geometry_valid"]]
    if invalid_geometry:
        raise ValueError(f"Accepted foregrounds failed automatic mask geometry: {invalid_geometry}")
    if len(backgrounds) != counts["accepted_backgrounds"]:
        raise ValueError(f"Require exactly {counts['accepted_backgrounds']} accepted backgrounds")
    backgrounds_per_class = Counter(int(row["class_id"]) for row in backgrounds)
    per_class = Counter(int(row["class_id"]) for row in foregrounds)
    needed = counts["accepted_foregrounds_per_class"]
    if set(per_class) != set(range(CLASS_COUNT)) or set(per_class.values()) != {needed}:
        raise ValueError(f"Require exactly {needed} accepted segmented foregrounds per class: {dict(per_class)}")
    if set(backgrounds_per_class) != set(range(CLASS_COUNT)) or set(backgrounds_per_class.values()) != {needed}:
        raise ValueError(f"Require exactly {needed} accepted backgrounds per planned class: {dict(backgrounds_per_class)}")
    if counts["final"] == 2048:
        scene_counts = Counter((int(row["class_id"]), row["scene_name"]) for row in backgrounds)
        assignments = {int(key): value for key, value in scenes["class_to_scene_families"].items()}
        wrong = {
            (class_id, scene): scene_counts[class_id, scene]
            for class_id, scene_names in assignments.items() for scene in scene_names
            if scene_counts[class_id, scene] != 32
        }
        if wrong:
            raise ValueError(f"Production requires 32 accepted backgrounds per class-scene pair: {wrong}")
    output = root / "final"
    require_launch_ready(gpu, output)
    for directory in (output / "images", output / "labels", output / "masks", output / "controls"):
        directory.mkdir(parents=True, exist_ok=True)
    segment_by_id = {row["candidate_id"]: row for row in foregrounds}
    settings = config["composition"]
    harmonization = config["harmonization"]
    class_prompts = {int(key): value for key, value in config["foreground_generation"]["class_prompts"].items()}
    rng = random.Random(int(config["seed"]) + 300000)
    pipe = load_harmonizer(load_yaml(repo_path(config["models_config"])), gpu)
    validate_prompts(pipe, harmonization_prompts(config, scenes))
    records = []
    foregrounds.sort(key=lambda row: (int(row["candidate_id"].rsplit("_", 1)[1]), int(row["class_id"])))
    background_pools = {
        class_id: sorted(
            (row for row in backgrounds if int(row["class_id"]) == class_id),
            key=lambda row: row["candidate_id"],
        )
        for class_id in range(CLASS_COUNT)
    }
    width, height = map(int, config["output_size"])
    for index, foreground_row in enumerate(foregrounds):
        class_id = int(foreground_row["class_id"])
        background_row = background_pools[class_id].pop(0)
        segment = segment_by_id[foreground_row["candidate_id"]]
        with Image.open(repo_path(background_row["image_path"])).convert("RGB") as item:
            background = item.resize((width, height), Image.Resampling.LANCZOS)
        with Image.open(repo_path(segment["rgba_path"])).convert("RGBA") as item:
            foreground = item.copy()
        bbox = foreground.getchannel("A").getbbox()
        if bbox is None:
            raise ValueError(f"Empty accepted alpha: {segment['rgba_path']}")
        foreground = foreground.crop(bbox)
        low, high = map(float, settings["object_height_fraction"])
        target_h = round(height * rng.uniform(low, high))
        target_w = max(1, round(foreground.width * target_h / foreground.height))
        if target_w > width * 0.48:
            target_w = round(width * 0.48); target_h = round(foreground.height * target_w / foreground.width)
        foreground = foreground.resize((target_w, target_h), Image.Resampling.LANCZOS)
        scene_name = background_row["scene_name"]
        center_x, support_y = map(float, settings["placement_centers"][scene_name])
        x = round(width * (center_x + rng.uniform(-float(settings["horizontal_jitter_fraction"]), float(settings["horizontal_jitter_fraction"]))) - target_w / 2)
        y = round(height * (support_y + rng.uniform(-float(settings["vertical_jitter_fraction"]), float(settings["vertical_jitter_fraction"]))) - target_h)
        x, y = min(max(x, 0), width-target_w), min(max(y, 0), height-target_h)
        foreground = color_match_foreground(
            foreground, background, x, y,
            float(harmonization["foreground_color_match_strength"]),
        )
        canvas = background.convert("RGBA"); canvas.alpha_composite(foreground, dest=(x, y))
        initial = canvas.convert("RGB")
        mask_canvas = Image.new("L", (width, height), 0); mask_canvas.paste(foreground.getchannel("A"), (x, y))
        inpaint, canny = target_controls(mask_canvas, int(harmonization["mask_padding_px"]))
        prompt = harmonization["prompt_template"].format(
            target=class_prompts[class_id], scene=scene_name.replace("_", " "))
        seed = int(config["seed"]) + 400000 + index
        result = pipe(
            prompt=prompt, negative_prompt=harmonization["negative_prompt"], image=initial,
            mask_image=inpaint, control_image=canny, strength=float(harmonization["strength"]),
            width=width, height=height, num_inference_steps=int(config["inference_steps"]),
            guidance_scale=float(config["guidance_scale"]),
            controlnet_conditioning_scale=float(harmonization["canny_scale"]),
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).images[0]
        # Restore the color-matched accepted object exactly after diffusion;
        # only its generated-background neighborhood was allowed to change.
        result_canvas = result.convert("RGBA")
        result_canvas.alpha_composite(foreground, dest=(x, y))
        result = result_canvas.convert("RGB")
        final_bbox = mask_canvas.getbbox()
        if final_bbox is None:
            raise ValueError("Final retained mask is empty")
        x1, y1, x2, y2 = final_bbox
        label = f"{class_id} {(x1+x2)/(2*width):.8f} {(y1+y2)/(2*height):.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}\n"
        stem = f"fb_{index:05d}_c{class_id:02d}"
        image_path, label_path = output / "images" / f"{stem}.png", output / "labels" / f"{stem}.txt"
        mask_path, control_path = output / "masks" / f"{stem}.png", output / "controls" / f"{stem}.png"
        result.save(image_path); label_path.write_text(label, encoding="utf-8")
        mask_canvas.save(mask_path); canny.save(control_path)
        records.append({"index": index, "class_id": class_id, "class_name": foreground_row["class_name"],
                        "scene_name": scene_name, "background_candidate_id": background_row["candidate_id"],
                        "foreground_candidate_id": foreground_row["candidate_id"], "seed": seed,
                        "image_path": str(image_path.relative_to(REPO_ROOT)),
                        "label_path": str(label_path.relative_to(REPO_ROOT)),
                        "mask_path": str(mask_path.relative_to(REPO_ROOT)), "bbox_xyxy": list(final_bbox),
                        "image_sha256": sha256(image_path), "training_use_forbidden": True})
        print(f"[compose {index + 1}/{counts['final']}] {stem}", flush=True)
    release(pipe)
    if len(records) != counts["final"]:
        raise ValueError(f"Expected {counts['final']} final records, got {len(records)}")
    write_json(output / "manifest.json", {"status": "generated_pending_final_review",
               "training_use_forbidden": True, "target_test_used": False, "git": git_state(),
               "config_sha256": config["_config_sha256"], "models_sha256": config["_models_sha256"],
               "script_sha256": sha256(Path(__file__).resolve()), "records": records})
    write_review(output / "review.csv", [dict(row, candidate_id=f"final_{row['index']:05d}") for row in records], "final_identity_contact_realism_annotation")


def runtime_preflight(config: dict[str, Any], scenes: dict[str, Any], stage: str, gpu: int) -> None:
    if git_state()["dirty"]:
        raise RuntimeError("Commit the pipeline before runtime preflight")
    if not torch.cuda.is_available() or not 0 <= gpu < torch.cuda.device_count():
        raise RuntimeError(f"Logical CUDA device {gpu} is unavailable")
    models = load_yaml(repo_path(config["models_config"]))
    if stage in {"backgrounds", "foregrounds"}:
        pipe = load_base_pipeline(models, gpu)
        prompts = background_prompts(config, scenes) if stage == "backgrounds" else foreground_prompts(config)
        validate_prompts(pipe, prompts)
        release(pipe)
    elif stage == "segment":
        from transformers import Sam3Model, Sam3Processor
        settings = config["segmentation"]
        processor = Sam3Processor.from_pretrained(settings["model_id"])
        model = Sam3Model.from_pretrained(settings["model_id"]).to(f"cuda:{gpu}").eval()
        del model, processor
        release(None)
    else:
        pipe = load_harmonizer(models, gpu)
        validate_prompts(pipe, harmonization_prompts(config, scenes))
        release(pipe)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("pilot", "production"), default="pilot")
    parser.add_argument("--stage", choices=("backgrounds", "foregrounds", "segment", "compose"), required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = repo_path(args.config)
    config = load_yaml(config_path)
    config["_config_sha256"] = sha256(config_path)
    config["_models_sha256"] = sha256(repo_path(config["models_config"]))
    scenes = load_yaml(repo_path(config["scene_policy"]))
    counts = validate_config(config, scenes, args.mode)
    root = repo_path(config["output_root"]) / args.mode
    report = {"status": "ready", "pipeline_id": config["pipeline_id"], "mode": args.mode,
              "stage": args.stage, "counts": counts, "output_root": str(root.relative_to(REPO_ROOT)),
              "config_sha256": sha256(config_path), "git": git_state(),
              "review_gates": config["quality_gates"]}
    if args.preflight_only:
        runtime_preflight(config, scenes, args.stage, args.gpu)
        report["runtime_models_and_prompts_validated"] = True
        print(json.dumps(report, indent=2)); return
    with stage_lock(root, args.stage):
        if args.stage == "backgrounds":
            generate_backgrounds(config, scenes, counts, root, args.gpu)
        elif args.stage == "foregrounds":
            generate_foregrounds(config, scenes, counts, root, args.gpu, args.mode)
        elif args.stage == "segment":
            segment_foregrounds(config, counts, root, args.gpu)
        else:
            compose(config, scenes, counts, root, args.gpu)


if __name__ == "__main__":
    main()
