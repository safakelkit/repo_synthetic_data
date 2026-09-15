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
from PIL import Image, PngImagePlugin


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/generation/sdxl_fb2048_v2.yaml"
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


def deterministic_choice(values: list[Any], key: str) -> tuple[int, Any]:
    if not values:
        raise ValueError(f"Empty background variation axis: {key}")
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    index = int.from_bytes(digest[:8], "big") % len(values)
    return index, values[index]


def background_conditioning(
    config: dict[str, Any], scene_name: str, class_id: int, attempt: int,
) -> dict[str, Any]:
    settings = config["background_generation"]
    base_key = f"{config['seed']}:background:{class_id}:{attempt}:{scene_name}"
    selected: dict[str, Any] = {"scene_context": settings["scene_contexts"][scene_name]}
    for axis, values in (
        ("scene_recipe", settings["scene_recipes"][scene_name]),
        ("support", settings["support_profiles"]),
        ("composition", settings["composition_profiles"]),
        ("capture", settings["capture_profiles"]),
        ("lighting", settings["lighting_profiles"]),
        ("materials", settings["material_profiles"]),
        ("guidance_scale", settings["guidance_scales"]),
    ):
        index, value = deterministic_choice(values, f"{base_key}:{axis}")
        selected[axis] = value
        selected[f"{axis}_index"] = index
    return {
        "prompt": settings["prompt_template"].format(**selected),
        "prompt_2": settings["prompt_2_template"].format(**selected),
        "negative_prompt": settings["negative_prompt"],
        "negative_prompt_2": settings["negative_prompt_2"],
        "guidance_scale": float(selected["guidance_scale"]),
        "variation_indices": {
            key.removesuffix("_index"): value
            for key, value in selected.items() if key.endswith("_index")
        },
    }


def scheduled_background_conditionings(
    config: dict[str, Any], scenes: dict[str, Any], count: int,
) -> list[dict[str, Any]]:
    assignments = {int(key): value for key, value in scenes["class_to_scene_families"].items()}
    base, remainder = divmod(count, CLASS_COUNT)
    return [
        background_conditioning(
            config, assignments[class_id][attempt % len(assignments[class_id])], class_id, attempt,
        )
        for class_id in range(CLASS_COUNT)
        for attempt in range(base + (1 if class_id < remainder else 0))
    ]


def validate_background_conditionings(pipe: Any, conditionings: list[dict[str, Any]]) -> None:
    for tokenizer_name, positive_key, negative_key in (
        ("tokenizer", "prompt", "negative_prompt"),
        ("tokenizer_2", "prompt_2", "negative_prompt_2"),
    ):
        tokenizer = getattr(pipe, tokenizer_name)
        limit = int(tokenizer.model_max_length)
        prompts = [row[positive_key] for row in conditionings]
        prompts.extend(dict.fromkeys(row[negative_key] for row in conditionings))
        for prompt in prompts:
            length = len(tokenizer(prompt, truncation=False)["input_ids"])
            if length > limit:
                raise ValueError(f"{tokenizer_name} prompt length {length} exceeds {limit}: {prompt}")


def foreground_prompt(config: dict[str, Any], class_id: int, attempt: int) -> str:
    settings = config["foreground_generation"]
    prompts = {int(key): value for key, value in settings["class_prompts"].items()}
    constraints = {int(key): value for key, value in settings["class_constraints"].items()}
    variants = {int(key): value for key, value in settings["class_variants"].items()}[class_id]
    conditions = settings["condition_profiles"]
    viewpoints = settings["viewpoint_profiles"]
    cycle = len(viewpoints) * len(variants) * len(conditions)
    if attempt >= cycle:
        raise ValueError(f"Foreground class {class_id} supports only {cycle} unique prompt combinations")
    viewpoint = viewpoints[attempt % len(viewpoints)]
    variant = variants[(attempt // len(viewpoints)) % len(variants)]
    condition = conditions[(attempt // (len(viewpoints) * len(variants))) % len(conditions)]
    return settings["prompt_template"].format(
        target=prompts[class_id], constraint=constraints[class_id],
        variant=variant, condition=condition, viewpoint=viewpoint,
    )


def foreground_negative_prompt(config: dict[str, Any], class_id: int) -> str:
    settings = config["foreground_generation"]
    class_negatives = {
        int(key): value for key, value in settings["class_negative_prompts"].items()
    }
    return f"{settings['negative_prompt']}, {class_negatives[class_id]}"


def pilot_foreground_attempts(config: dict[str, Any], class_id: int, count: int) -> list[int]:
    """Select sparse prompt-cycle positions that cover every foreground axis."""
    settings = config["foreground_generation"]
    variants = {int(key): value for key, value in settings["class_variants"].items()}[class_id]
    viewpoints = settings["viewpoint_profiles"]
    conditions = settings["condition_profiles"]
    required = max(len(variants), len(viewpoints), len(conditions))
    if count < required:
        raise ValueError(
            f"Pilot foreground class {class_id} needs at least {required} candidates "
            "to cover every variation axis"
        )
    cycle = len(viewpoints) * len(variants) * len(conditions)
    if count > cycle:
        raise ValueError(f"Pilot foreground class {class_id} supports at most {cycle} candidates")

    attempts = []
    for sample in range(count):
        viewpoint_index = (sample + 3 * class_id) % len(viewpoints)
        variant_index = sample % len(variants)
        condition_index = (sample + class_id) % len(conditions)
        attempt = viewpoint_index + len(viewpoints) * (
            variant_index + len(variants) * condition_index
        )
        if attempt not in attempts:
            attempts.append(attempt)

    # Counts above the marginal-coverage minimum are filled deterministically.
    if len(attempts) < count:
        attempts.extend(attempt for attempt in range(cycle) if attempt not in attempts)
    return attempts[:count]


def foreground_prompts(config: dict[str, Any]) -> list[str]:
    settings = config["foreground_generation"]
    return [
        foreground_prompt(config, class_id, attempt)
        for class_id in range(CLASS_COUNT)
        for attempt in range(
            len(settings["viewpoint_profiles"])
            * len({int(key): value for key, value in settings["class_variants"].items()}[class_id])
            * len(settings["condition_profiles"])
        )
    ] + [foreground_negative_prompt(config, class_id) for class_id in range(CLASS_COUNT)]


def harmonization_prompts(config: dict[str, Any], scenes: dict[str, Any]) -> list[str]:
    settings = config["harmonization"]
    return [
        settings["prompt_template"].format(
            target=target, scene=scene_name.replace("_", " "))
        for target in config["foreground_generation"]["class_prompts"].values()
        for scene_name in scenes["scene_families"]
    ] + [settings["negative_prompt"]]


def validate_config(config: dict[str, Any], scenes: dict[str, Any], mode: str) -> dict[str, int]:
    if config.get("status") != "ready_for_hardened_all_class_pilot":
        raise ValueError("SDXL-FB2048 config is not released for the all-class pilot")
    if config.get("training_use_forbidden") is not True or config.get("target_test_used") is not False:
        raise ValueError("Pilot must be training-forbidden and target-test independent")
    foreground = config["foreground_generation"]
    expected_classes = set(range(CLASS_COUNT))
    for key in ("class_prompts", "class_constraints", "class_variants"):
        values = {int(class_id): value for class_id, value in foreground[key].items()}
        if set(values) != expected_classes or any(not value for value in values.values()):
            raise ValueError(f"Foreground {key} must cover all 16 classes")
    negative_classes = {int(class_id) for class_id in foreground["class_negative_prompts"]}
    if negative_classes != expected_classes:
        raise ValueError("Foreground class_negative_prompts must cover all 16 classes")
    route_classes = [
        int(class_id)
        for class_ids in foreground["generation_routes"].values()
        for class_id in class_ids
    ]
    if len(route_classes) != CLASS_COUNT or set(route_classes) != expected_classes:
        raise ValueError("Foreground generation routes must partition classes 0..15")
    controlled_classes = {
        int(class_id) for class_id in foreground["controlled_retry"]["class_ids"]
    }
    routed_controlled = {
        int(class_id) for class_id in foreground["generation_routes"]["controlled_geometry_pilot"]
    }
    retry_guidance_classes = {int(class_id) for class_id in foreground["retry_guidance"]}
    if controlled_classes != routed_controlled or retry_guidance_classes != routed_controlled:
        raise ValueError("Controlled routes, class IDs, and retry guidance must agree")
    strict_text_classes = {
        int(class_id) for class_id in foreground["generation_routes"]["text_to_image_strict_review"]
    }
    for plan_name, plan in foreground.get("focused_retry_plans", {}).items():
        plan_classes = [int(class_id) for class_id in plan.get("class_ids", [])]
        if (plan.get("status") != "ready" or plan.get("pool") != "foregrounds"
                or not plan_classes or len(plan_classes) != len(set(plan_classes))
                or not set(plan_classes).issubset(strict_text_classes)
                or int(plan.get("candidates_per_class", 0)) < 1):
            raise ValueError(f"Invalid focused foreground retry plan: {plan_name}")
    if len(scenes["class_policy"]["names"]) != CLASS_COUNT:
        raise ValueError("Scene policy must contain exactly 16 classes")
    background = config["background_generation"]
    scene_names = set(scenes["scene_families"])
    recipe_names = set(background["scene_recipes"])
    context_names = set(background["scene_contexts"])
    if recipe_names != scene_names or context_names != scene_names:
        raise ValueError(
            "Background scene recipes/contexts do not match scene policy; "
            f"recipe_missing={sorted(scene_names - recipe_names)}, "
            f"context_missing={sorted(scene_names - context_names)}"
        )
    required_axes = (
        "support_profiles", "composition_profiles", "capture_profiles", "lighting_profiles",
        "material_profiles", "guidance_scales",
    )
    if any(not background.get(axis) for axis in required_axes):
        raise ValueError("Every background variation axis must contain at least one value")
    production_conditionings = scheduled_background_conditionings(
        config, scenes, int(config["production"]["background_candidates"]),
    )
    prompt_pairs = {(row["prompt"], row["prompt_2"]) for row in production_conditionings}
    if len(prompt_pairs) < int(0.99 * len(production_conditionings)):
        raise ValueError("Production background schedule has insufficient prompt-pair diversity")
    assignments = {int(key): value for key, value in scenes["class_to_scene_families"].items()}
    pilot_per_class = int(config["pilot"]["background_candidates"]) // CLASS_COUNT
    uncovered = {
        class_id: scene_names[pilot_per_class:]
        for class_id, scene_names in assignments.items() if pilot_per_class < len(scene_names)
    }
    if uncovered:
        raise ValueError(f"Pilot does not cover every assigned background scene: {uncovered}")
    viewpoints = len(foreground["viewpoint_profiles"])
    conditions = len(foreground["condition_profiles"])
    pilot_foreground_count = int(config["pilot"]["foreground_candidates_per_class"])
    if config["pilot"].get("foreground_sampling") != "stratified_axes":
        raise ValueError("Pilot foreground sampling must use stratified_axes")
    minimum_pilot_count = max(
        viewpoints, conditions,
        max(len(foreground["class_variants"][class_id]) for class_id in expected_classes),
    )
    if pilot_foreground_count < minimum_pilot_count:
        raise ValueError(
            f"Pilot needs at least {minimum_pilot_count} foreground candidates per class "
            "to cover every variation axis"
        )
    foreground_cycles = {
        class_id: viewpoints * conditions * len(foreground["class_variants"][class_id])
        for class_id in expected_classes
    }
    required_attempts = {
        class_id: (int(config["production"]["aerosol_foreground_candidates"])
                   if class_id == 11 else int(config["production"]["foreground_candidates_per_class"]))
        for class_id in expected_classes
    }
    if foreground_cycles != required_attempts:
        raise ValueError(
            f"Foreground prompt cycles must exactly match production candidate counts: "
            f"cycles={foreground_cycles}, required={required_attempts}"
        )
    production_foreground_prompts = [
        foreground_prompt(config, class_id, attempt)
        for class_id, count in required_attempts.items() for attempt in range(count)
    ]
    if len(set(production_foreground_prompts)) != len(production_foreground_prompts):
        raise ValueError("Production foreground schedule contains repeated prompts")
    size_classes = {int(key) for key in config["composition"]["class_object_height_fraction"]}
    if size_classes != expected_classes:
        raise ValueError("Class-specific object size ranges must cover all 16 classes")
    for class_id, limits in config["composition"]["class_object_height_fraction"].items():
        low, high = map(float, limits)
        if not 0 < low < high < 1:
            raise ValueError(f"Invalid class-specific size range for {class_id}: {limits}")
    review_checks = config["quality_gates"]["required_manual_checks"]
    unknown_review_fields = {
        field for fields in review_checks.values() for field in fields if field not in REVIEW_FIELDS
    }
    if unknown_review_fields:
        raise ValueError(f"Unknown required manual review fields: {sorted(unknown_review_fields)}")
    person_screen = config["quality_gates"]["person_screen"]
    person_model = repo_path(person_screen["model"])
    if not person_model.is_file() or sha256(person_model) != person_screen["model_sha256"]:
        raise ValueError("Frozen person-screen model is missing or has the wrong hash")
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


def generation_fingerprint(specification: dict[str, Any]) -> str:
    encoded = json.dumps(specification, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def save_image_atomic(
    image: Image.Image, path: Path, generation_spec: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.partial{path.suffix}")
    if generation_spec is not None and path.suffix.lower() == ".png":
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("generation_fingerprint", generation_fingerprint(generation_spec))
        image.save(temporary, pnginfo=metadata)
    else:
        image.save(temporary)
    temporary.replace(path)


def verified_existing_image(
    path: Path, expected_size: tuple[int, int], generation_spec: dict[str, Any] | None = None,
) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            image.load()
            if image.size != expected_size:
                raise ValueError(f"Partial image has wrong size: {path} ({image.size})")
            if generation_spec is not None:
                expected = generation_fingerprint(generation_spec)
                actual = image.info.get("generation_fingerprint")
                if actual != expected:
                    raise ValueError(
                        f"Partial image provenance mismatch: {path} "
                        f"(actual={actual!r}, expected={expected})"
                    )
    except Exception as exc:
        raise ValueError(f"Partial image is not a valid resumable output: {path}: {exc}") from exc
    return True


REVIEW_FIELDS = [
    "candidate_id", "class_id", "class_name", "scene_name", "image_path",
    "review_status", "review_reason", "target_free", "identity_correct",
    "single_object", "no_human", "photorealistic", "support_valid",
    "support_x1", "support_y1", "support_x2", "support_y2", "mask_correct",
    "placement_valid", "annotation_valid",
    "auto_person_detected", "auto_person_confidence",
]


def write_review(path: Path, records: list[dict[str, Any]], kind: str) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for row in records:
            review_row = {field: "" for field in REVIEW_FIELDS}
            review_row.update({
                "candidate_id": row["candidate_id"],
                "class_id": row.get("class_id", ""),
                "class_name": row.get("class_name", ""),
                "scene_name": row.get("scene_name", ""),
                "image_path": row["image_path"],
                "review_status": "pending",
                "review_reason": f"review_{kind}",
                "auto_person_detected": str(bool(row.get("auto_person_detected", False))).lower(),
                "auto_person_confidence": row.get("auto_person_confidence", 0.0),
            })
            if row.get("auto_person_detected", False):
                review_row.update({"review_status": "rejected", "no_human": "no",
                                   "review_reason": "automatic_person_detection"})
            writer.writerow(review_row)
    temporary.replace(path)


def accepted_records(
    manifest_path: Path, review_path: Path, required_checks: list[str] | tuple[str, ...] = (),
) -> list[dict[str, Any]]:
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
    allowed_statuses = {"pending", "accepted", "rejected"}
    invalid_statuses = {
        row["candidate_id"]: row.get("review_status", "") for row in review
        if row.get("review_status", "").strip().lower() not in allowed_statuses
    }
    if invalid_statuses:
        raise ValueError(f"Invalid review status values: {invalid_statuses}")
    accepted = []
    for review_row in review:
        if review_row["review_status"].strip().lower() != "accepted":
            continue
        failed = [key for key in required_checks if review_row.get(key, "").strip().lower() != "yes"]
        if failed:
            raise ValueError(
                f"Accepted candidate {review_row['candidate_id']} lacks required yes checks: {failed}"
            )
        record = dict(by_id[review_row["candidate_id"]])
        if record.get("auto_person_detected", False):
            raise ValueError(
                f"Candidate {review_row['candidate_id']} cannot override automatic person rejection"
            )
        record["review"] = dict(review_row)
        accepted.append(record)
    return accepted


def require_unique_files(records: list[dict[str, Any]], path_key: str, label: str) -> None:
    hashes = [sha256(repo_path(row[path_key])) for row in records]
    if len(set(hashes)) != len(hashes):
        raise ValueError(f"Exact duplicate detected among accepted {label}")


def require_manifest_hashes(
    records: list[dict[str, Any]], path_key: str, hash_key: str, label: str,
) -> None:
    mismatches = []
    for row in records:
        path = repo_path(row[path_key])
        if not path.is_file() or sha256(path) != row.get(hash_key):
            mismatches.append(str(row.get("candidate_id", path)))
    if mismatches:
        raise ValueError(f"Manifest/file hash mismatch among {label}: {mismatches[:20]}")


def difference_hash(path: Path) -> int:
    with Image.open(path) as source:
        if "A" in source.getbands():
            rgba = source.convert("RGBA")
            image = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("L")
        else:
            image = source.convert("L")
        pixels = np.asarray(image.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.flat:
        value = (value << 1) | int(bit)
    return value


def require_perceptual_uniqueness(
    records: list[dict[str, Any]], path_key: str, label: str, max_distance: int,
    group_key: str | None = None,
) -> None:
    groups: dict[Any, list[tuple[str, int]]] = {}
    for row in records:
        group = row.get(group_key) if group_key else "all"
        groups.setdefault(group, []).append(
            (str(row["candidate_id"]), difference_hash(repo_path(row[path_key])))
        )
    collisions = []
    for values in groups.values():
        for index, (left_id, left_hash) in enumerate(values):
            for right_id, right_hash in values[index + 1:]:
                distance = (left_hash ^ right_hash).bit_count()
                if distance <= max_distance:
                    collisions.append((left_id, right_id, distance))
                    if len(collisions) >= 20:
                        break
            if len(collisions) >= 20:
                break
        if len(collisions) >= 20:
            break
    if collisions:
        raise ValueError(f"Perceptual near-duplicates detected among {label}: {collisions}")


def reviewed_support_bbox(row: dict[str, Any], width: int, height: int) -> tuple[int, int, int, int]:
    review = row.get("review", {})
    try:
        normalized = tuple(float(review[key]) for key in ("support_x1", "support_y1", "support_x2", "support_y2"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing reviewed support coordinates for {row['candidate_id']}") from exc
    x1, y1, x2, y2 = normalized
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError(f"Invalid normalized support box for {row['candidate_id']}: {normalized}")
    if x2 - x1 < 0.12 or y2 - y1 < 0.03:
        raise ValueError(f"Reviewed support box is too small for {row['candidate_id']}: {normalized}")
    return round(x1 * width), round(y1 * height), round(x2 * width), round(y2 * height)


def apply_person_screen(config: dict[str, Any], records: list[dict[str, Any]], gpu: int | str) -> None:
    from ultralytics import YOLO
    settings = config["quality_gates"]["person_screen"]
    model_path = repo_path(settings["model"])
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing frozen person-screen model: {model_path}")
    actual_model_hash = sha256(model_path)
    if actual_model_hash != settings["model_sha256"]:
        raise ValueError(
            f"Person-screen model hash mismatch: actual={actual_model_hash}, "
            f"expected={settings['model_sha256']}"
        )
    model = YOLO(str(model_path))
    paths = [str(repo_path(row["image_path"])) for row in records]
    results = model.predict(
        source=paths, classes=[int(settings["class_id"])],
        conf=float(settings["confidence"]), device=gpu, verbose=False, stream=True,
    )
    for row, result in zip(records, results, strict=True):
        confidences = result.boxes.conf.detach().cpu().tolist() if result.boxes is not None else []
        row["auto_person_confidence"] = round(max(confidences, default=0.0), 6)
        row["auto_person_detected"] = bool(confidences)


def generate_backgrounds(config: dict[str, Any], scenes: dict[str, Any], counts: dict[str, int], root: Path, gpu: int) -> None:
    output = root / "backgrounds"
    require_launch_ready(gpu, output, allow_partial=True)
    images = output / "images"
    pipe = load_base_pipeline(load_yaml(repo_path(config["models_config"])), gpu)
    names = {int(key): value for key, value in scenes["class_policy"]["names"].items()}
    assignments = {int(key): value for key, value in scenes["class_to_scene_families"].items()}
    width, height = map(int, config["output_size"])
    records = []
    conditionings = scheduled_background_conditionings(config, scenes, counts["background_candidates"])
    validate_background_conditionings(pipe, conditionings)
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
        conditioning = conditionings[index]
        seed = int(config["seed"]) + index
        path = images / f"bg_{index:05d}_{scene_name}.png"
        candidate_id = f"bg_c{class_id:02d}_{attempt:04d}"
        generation_spec = {
            "pipeline_id": config["pipeline_id"], "stage": "background", "candidate_id": candidate_id,
            "config_sha256": config["_config_sha256"], "models_sha256": config["_models_sha256"],
            "seed": seed, "size": [width, height], "steps": int(config["inference_steps"]),
            "prompt": conditioning["prompt"], "prompt_2": conditioning["prompt_2"],
            "negative_prompt": conditioning["negative_prompt"],
            "negative_prompt_2": conditioning["negative_prompt_2"],
            "guidance_scale": conditioning["guidance_scale"],
        }
        resumed = verified_existing_image(path, (width, height), generation_spec)
        started = time.monotonic()
        if not resumed:
            image = pipe(
                prompt=conditioning["prompt"], prompt_2=conditioning["prompt_2"],
                negative_prompt=conditioning["negative_prompt"],
                negative_prompt_2=conditioning["negative_prompt_2"], width=width, height=height,
                num_inference_steps=int(config["inference_steps"]),
                guidance_scale=conditioning["guidance_scale"],
                generator=torch.Generator(device="cpu").manual_seed(seed),
            ).images[0]
            save_image_atomic(image, path, generation_spec)
        records.append({"candidate_id": candidate_id,
                        "class_id": class_id, "class_name": names[class_id],
                        "planned_class_id": class_id, "scene_slot": scene_slot,
                        "scene_name": scene_name,
                        "image_path": str(path.relative_to(REPO_ROOT)), "sha256": sha256(path),
                        "seed": seed, "prompt": conditioning["prompt"],
                        "prompt_2": conditioning["prompt_2"],
                        "negative_prompt": conditioning["negative_prompt"],
                        "negative_prompt_2": conditioning["negative_prompt_2"],
                        "guidance_scale": conditioning["guidance_scale"],
                        "generation_fingerprint": generation_fingerprint(generation_spec),
                        "variation_indices": conditioning["variation_indices"],
                        "seconds": 0.0 if resumed else round(time.monotonic() - started, 3),
                        "resumed_existing": resumed})
        action = "resume" if resumed else "generate"
        print(f"[background {index + 1}/{counts['background_candidates']}] {action} {path.name}", flush=True)
    release(pipe)
    apply_person_screen(config, records, gpu)
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
        attempts = (
            pilot_foreground_attempts(config, class_id, count)
            if mode == "pilot"
            else range(count)
        )
        rows.extend((class_id, attempt) for attempt in attempts)
    return rows


def generate_foregrounds(config: dict[str, Any], scenes: dict[str, Any], counts: dict[str, int], root: Path, gpu: int, mode: str) -> None:
    output = root / "foregrounds"
    require_launch_ready(gpu, output, allow_partial=True)
    images = output / "images"
    pipe = load_base_pipeline(load_yaml(repo_path(config["models_config"])), gpu)
    settings = config["foreground_generation"]
    width, height = map(int, config["output_size"])
    names = {int(key): value for key, value in scenes["class_policy"]["names"].items()}
    records = []
    schedule = foreground_schedule(config, counts, mode)
    validate_prompts(pipe, foreground_prompts(config))
    images.mkdir(parents=True)
    for index, (class_id, attempt) in enumerate(schedule):
        prompt = foreground_prompt(config, class_id, attempt)
        negative_prompt = foreground_negative_prompt(config, class_id)
        seed = int(config["seed"]) + 100000 + class_id * 10000 + attempt
        path = images / f"fg_c{class_id:02d}_{attempt:04d}.png"
        candidate_id = f"fg_c{class_id:02d}_{attempt:04d}"
        generation_spec = {
            "pipeline_id": config["pipeline_id"], "stage": "foreground", "candidate_id": candidate_id,
            "config_sha256": config["_config_sha256"], "models_sha256": config["_models_sha256"],
            "seed": seed, "size": [width, height], "steps": int(config["inference_steps"]),
            "prompt": prompt, "negative_prompt": negative_prompt,
            "guidance_scale": float(settings["guidance_scale"]),
        }
        resumed = verified_existing_image(path, (width, height), generation_spec)
        started = time.monotonic()
        if not resumed:
            image = pipe(
                prompt=prompt, negative_prompt=negative_prompt, width=width, height=height,
                num_inference_steps=int(config["inference_steps"]),
                guidance_scale=float(settings["guidance_scale"]),
                generator=torch.Generator(device="cpu").manual_seed(seed),
            ).images[0]
            save_image_atomic(image, path, generation_spec)
        records.append({"candidate_id": candidate_id, "class_id": class_id,
                        "class_name": names[class_id], "image_path": str(path.relative_to(REPO_ROOT)),
                        "sha256": sha256(path), "seed": seed, "attempt": attempt, "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "guidance_scale": float(settings["guidance_scale"]),
                        "generation_fingerprint": generation_fingerprint(generation_spec),
                        "seconds": 0.0 if resumed else round(time.monotonic() - started, 3),
                        "resumed_existing": resumed})
        action = "resume" if resumed else "generate"
        print(f"[foreground {index + 1}/{len(schedule)}] {action} {path.name}", flush=True)
    release(pipe)
    apply_person_screen(config, records, gpu)
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
    checks = config["quality_gates"]["required_manual_checks"]["foregrounds"]
    accepted = accepted_records(source / "manifest.json", source / "review.csv", checks)
    require_manifest_hashes(accepted, "image_path", "sha256", "foreground candidates")
    require_unique_files(accepted, "image_path", "foreground candidates")
    require_perceptual_uniqueness(
        accepted, "image_path", "foreground candidates",
        int(config["quality_gates"]["perceptual_hash"]["foreground_max_hamming_distance_within_class"]),
        group_key="class_id",
    )
    per_class = Counter(int(row["class_id"]) for row in accepted)
    needed = counts["accepted_foregrounds_per_class"]
    wrong = {class_id: per_class[class_id] for class_id in range(CLASS_COUNT) if per_class[class_id] != needed}
    if wrong:
        raise ValueError(f"Require exactly {needed} accepted foregrounds per class: {wrong}")
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
                        "rgba_path": str(rgba_path.relative_to(REPO_ROOT)),
                        "mask_sha256": sha256(mask_path), "rgba_sha256": sha256(rgba_path),
                        "sam_score": score,
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


def write_nested_manifests(output: Path, records: list[dict[str, Any]], prefixes: list[int]) -> None:
    if not prefixes:
        return
    subset_root = output / "subsets"
    subset_root.mkdir()
    for prefix in map(int, prefixes):
        if prefix <= 0 or prefix > len(records) or prefix % CLASS_COUNT:
            raise ValueError(f"Invalid nested prefix: {prefix}")
        subset = records[:prefix]
        expected = prefix // CLASS_COUNT
        counts = Counter(int(row["class_id"]) for row in subset)
        if set(counts) != set(range(CLASS_COUNT)) or set(counts.values()) != {expected}:
            raise ValueError(f"Nested prefix {prefix} is not class-balanced: {dict(counts)}")
        write_json(subset_root / f"fb_{prefix}.json", {
            "status": "generated_pending_final_review",
            "images": prefix,
            "images_per_class": expected,
            "records": subset,
        })


def compose(config: dict[str, Any], scenes: dict[str, Any], counts: dict[str, int], root: Path, gpu: int) -> None:
    checks = config["quality_gates"]["required_manual_checks"]
    backgrounds = accepted_records(
        root / "backgrounds/manifest.json", root / "backgrounds/review.csv", checks["backgrounds"],
    )
    foregrounds = accepted_records(
        root / "segmented_foregrounds/manifest.json", root / "segmented_foregrounds/review.csv",
        checks["segmented_foregrounds"],
    )
    require_unique_files(backgrounds, "image_path", "backgrounds")
    require_unique_files(foregrounds, "rgba_path", "segmented foregrounds")
    require_manifest_hashes(backgrounds, "image_path", "sha256", "backgrounds")
    require_manifest_hashes(foregrounds, "rgba_path", "rgba_sha256", "segmented foregrounds")
    phash = config["quality_gates"]["perceptual_hash"]
    require_perceptual_uniqueness(
        backgrounds, "image_path", "backgrounds", int(phash["background_max_hamming_distance"]),
    )
    require_perceptual_uniqueness(
        foregrounds, "rgba_path", "segmented foregrounds",
        int(phash["foreground_max_hamming_distance_within_class"]), group_key="class_id",
    )
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
    if counts["final"] < 2048:
        missing_scenes = set(scenes["scene_families"]) - {row["scene_name"] for row in backgrounds}
        if missing_scenes:
            raise ValueError(
                f"Pilot composition must cover every global scene family; missing={sorted(missing_scenes)}"
            )
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
    foreground_pools = {
        class_id: sorted(
            (row for row in foregrounds if int(row["class_id"]) == class_id),
            key=lambda row: row["candidate_id"],
        )
        for class_id in range(CLASS_COUNT)
    }
    foregrounds = [
        foreground_pools[class_id][round_index]
        for round_index in range(needed)
        for class_id in range(CLASS_COUNT)
    ]
    background_pools = {
        class_id: sorted(
            (row for row in backgrounds if int(row["class_id"]) == class_id),
            key=lambda row: row["candidate_id"],
        )
        for class_id in range(CLASS_COUNT)
    }
    width, height = map(int, config["output_size"])
    size_ranges = {int(key): value for key, value in settings["class_object_height_fraction"].items()}
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
        low, high = map(float, size_ranges[class_id])
        target_h = round(height * rng.uniform(low, high))
        target_w = max(1, round(foreground.width * target_h / foreground.height))
        sx1, sy1, sx2, sy2 = reviewed_support_bbox(background_row, width, height)
        support_margin = round((sx2 - sx1) * float(settings["support_edge_margin_fraction"]))
        maximum_width = max(1, sx2 - sx1 - 2 * support_margin)
        if target_w > maximum_width:
            target_w = maximum_width
            target_h = max(1, round(foreground.height * target_w / foreground.width))
        foreground = foreground.resize((target_w, target_h), Image.Resampling.LANCZOS)
        scene_name = background_row["scene_name"]
        minimum_x = sx1 + support_margin
        maximum_x = max(minimum_x, sx2 - support_margin - target_w)
        x = rng.randint(minimum_x, maximum_x)
        contact_y = rng.randint(sy1, sy2)
        y = contact_y - target_h
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
        records.append({"candidate_id": f"final_{index:05d}", "index": index,
                        "class_id": class_id, "class_name": foreground_row["class_name"],
                        "scene_name": scene_name, "background_candidate_id": background_row["candidate_id"],
                        "foreground_candidate_id": foreground_row["candidate_id"], "seed": seed,
                        "reviewed_support_xyxy": [sx1, sy1, sx2, sy2],
                        "image_path": str(image_path.relative_to(REPO_ROOT)),
                        "label_path": str(label_path.relative_to(REPO_ROOT)),
                        "mask_path": str(mask_path.relative_to(REPO_ROOT)),
                        "control_path": str(control_path.relative_to(REPO_ROOT)),
                        "bbox_xyxy": list(final_bbox),
                        "image_sha256": sha256(image_path), "label_sha256": sha256(label_path),
                        "mask_sha256": sha256(mask_path), "control_sha256": sha256(control_path),
                        "training_use_forbidden": True})
        print(f"[compose {index + 1}/{counts['final']}] {stem}", flush=True)
    release(pipe)
    if len(records) != counts["final"]:
        raise ValueError(f"Expected {counts['final']} final records, got {len(records)}")
    apply_person_screen(config, records, gpu)
    require_unique_files(records, "image_path", "final images")
    require_manifest_hashes(records, "image_path", "image_sha256", "final images")
    require_manifest_hashes(records, "label_path", "label_sha256", "final labels")
    require_manifest_hashes(records, "mask_path", "mask_sha256", "final masks")
    require_manifest_hashes(records, "control_path", "control_sha256", "final controls")
    write_json(output / "manifest.json", {"status": "generated_pending_final_review",
               "training_use_forbidden": True, "target_test_used": False, "git": git_state(),
               "config_sha256": config["_config_sha256"], "models_sha256": config["_models_sha256"],
               "script_sha256": sha256(Path(__file__).resolve()), "records": records})
    write_review(output / "review.csv", records, "final_identity_contact_realism_annotation")
    write_nested_manifests(output, records, config["production"]["nested_prefixes"] if counts["final"] == 2048 else [])


def finalize_review(config: dict[str, Any], counts: dict[str, int], root: Path) -> None:
    if git_state()["dirty"]:
        raise RuntimeError("Commit the reviewed pipeline before finalization")
    output = root / "final"
    release_path = output / "release_manifest.json"
    if release_path.exists():
        raise FileExistsError(f"Final review is already finalized: {release_path}")
    checks = config["quality_gates"]["required_manual_checks"]["final"]
    accepted = accepted_records(output / "manifest.json", output / "review.csv", checks)
    if len(accepted) != counts["final"]:
        raise ValueError(f"Require all {counts['final']} final images accepted, got {len(accepted)}")
    require_unique_files(accepted, "image_path", "accepted final images")
    require_manifest_hashes(accepted, "image_path", "image_sha256", "accepted final images")
    require_manifest_hashes(accepted, "label_path", "label_sha256", "accepted final labels")
    require_manifest_hashes(accepted, "mask_path", "mask_sha256", "accepted final masks")
    require_manifest_hashes(accepted, "control_path", "control_sha256", "accepted final controls")
    require_perceptual_uniqueness(
        accepted, "image_path", "accepted final images",
        int(config["quality_gates"]["perceptual_hash"]["final_max_hamming_distance"]),
    )
    width, height = map(int, config["output_size"])
    for row in accepted:
        label_path = repo_path(row["label_path"])
        values = label_path.read_text(encoding="utf-8").strip().split()
        if len(values) != 5 or int(values[0]) != int(row["class_id"]):
            raise ValueError(f"Invalid final label identity: {label_path}")
        coordinates = list(map(float, values[1:]))
        if any(not 0 <= value <= 1 for value in coordinates) or coordinates[2] <= 0 or coordinates[3] <= 0:
            raise ValueError(f"Invalid normalized final label: {label_path}")
        x1, y1, x2, y2 = map(int, row["bbox_xyxy"])
        expected = [(x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                    (x2 - x1) / width, (y2 - y1) / height]
        if any(abs(actual - wanted) > 1e-7 for actual, wanted in zip(coordinates, expected, strict=True)):
            raise ValueError(f"Final label does not match retained mask box: {label_path}")
        with Image.open(repo_path(row["image_path"])) as image:
            if image.size != (width, height):
                raise ValueError(f"Invalid final image size: {row['image_path']}")
    write_json(release_path, {
        "status": "pilot_accepted" if counts["final"] < 2048 else "production_accepted",
        "training_use_forbidden": bool(config["training_use_forbidden"]),
        "config_sha256": config["_config_sha256"],
        "models_sha256": config["_models_sha256"],
        "records": accepted,
    })


def runtime_preflight(
    config: dict[str, Any], scenes: dict[str, Any], counts: dict[str, int], stage: str, gpu: int,
) -> None:
    if git_state()["dirty"]:
        raise RuntimeError("Commit the pipeline before runtime preflight")
    if not torch.cuda.is_available() or not 0 <= gpu < torch.cuda.device_count():
        raise RuntimeError(f"Logical CUDA device {gpu} is unavailable")
    models = load_yaml(repo_path(config["models_config"]))
    if stage in {"backgrounds", "foregrounds"}:
        pipe = load_base_pipeline(models, gpu)
        if stage == "backgrounds":
            validate_background_conditionings(
                pipe, scheduled_background_conditionings(config, scenes, counts["background_candidates"]),
            )
        else:
            validate_prompts(pipe, foreground_prompts(config))
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
    parser.add_argument(
        "--stage", choices=("backgrounds", "foregrounds", "segment", "compose", "finalize"), required=True,
    )
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
    if args.preflight_only and args.stage == "finalize":
        raise ValueError("Finalize has no model preflight; run it only after completing final review")
    if args.preflight_only:
        runtime_preflight(config, scenes, counts, args.stage, args.gpu)
        report["runtime_models_and_prompts_validated"] = True
        print(json.dumps(report, indent=2)); return
    with stage_lock(root, args.stage):
        if args.stage == "backgrounds":
            generate_backgrounds(config, scenes, counts, root, args.gpu)
        elif args.stage == "foregrounds":
            generate_foregrounds(config, scenes, counts, root, args.gpu, args.mode)
        elif args.stage == "segment":
            segment_foregrounds(config, counts, root, args.gpu)
        elif args.stage == "compose":
            compose(config, scenes, counts, root, args.gpu)
        else:
            finalize_review(config, counts, root)


if __name__ == "__main__":
    main()
