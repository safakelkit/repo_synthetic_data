from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import transformers
import yaml
from huggingface_hub import HfApi
from PIL import Image
from transformers import Sam3Model, Sam3Processor


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/placement/support_masks_sam3_v2.yaml"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MANIFEST_FIELDS = [
    "background_path",
    "background_sha256",
    "category",
    "prompt",
    "instance_index",
    "raw_instance_index",
    "score",
    "bbox_xyxy",
    "mask_area_pixels",
    "mask_area_ratio",
    "mask_path",
    "mask_sha256",
    "proposal_status",
    "review_status",
    "reviewer_note",
    "model_id",
    "model_revision",
]


def repo_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def repository_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config: {path}")
    return config


def resolve_model_revision(model_id: str, configured_revision: str) -> str:
    requested = None if configured_revision == "resolve_exact_sha_at_pilot_start" else configured_revision
    try:
        model_info = HfApi().model_info(model_id, revision=requested)
    except Exception as exc:
        raise RuntimeError(
            f"Could not resolve the exact revision for gated model {model_id!r}. "
            "Confirm network access and run `hf auth login` in env_sam3."
        ) from exc
    if not model_info.sha:
        raise RuntimeError(f"Hugging Face returned no commit SHA for {model_id!r}")
    return str(model_info.sha)


def collect_backgrounds(
    background_root: Path,
    categories: dict[str, list[str]],
    mode: str,
    images_per_category: int,
    seed: int,
) -> dict[str, list[Path]]:
    selected: dict[str, list[Path]] = {}
    for category_index, category in enumerate(categories):
        category_root = background_root / category
        images = sorted(
            path
            for path in category_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not images:
            raise FileNotFoundError(f"No background images found in {category_root}")
        if mode == "pilot":
            if len(images) < images_per_category:
                raise ValueError(
                    f"Category {category!r} has {len(images)} images, "
                    f"fewer than requested pilot count {images_per_category}"
                )
            rng = random.Random(seed + category_index)
            images = sorted(rng.sample(images, images_per_category))
        selected[category] = images
    return selected


def prepare_output_root(output_root: Path) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output is not empty: {output_root}. Move or explicitly remove the "
            "previous pilot before creating another traceable candidate."
        )
    (output_root / "masks").mkdir(parents=True, exist_ok=True)
    (output_root / "overlays").mkdir(parents=True, exist_ok=True)
    (output_root / "contact_sheets").mkdir(parents=True, exist_ok=True)


def tensor_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def binary_mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.logical_and(first, second).sum())
    union = int(np.logical_or(first, second).sum())
    return intersection / union if union else 0.0


def build_mask_candidates(
    masks: Any,
    boxes: Any,
    scores: Any,
    image_area: int,
    minimum_area_ratio: float,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for raw_instance_index, raw_mask in enumerate(masks):
        binary_mask = np.squeeze(tensor_to_numpy(raw_mask)) > 0
        mask_area = int(binary_mask.sum())
        area_ratio = mask_area / max(image_area, 1)
        if mask_area == 0 or area_ratio < minimum_area_ratio:
            continue
        candidates.append(
            {
                "raw_instance_index": raw_instance_index,
                "binary_mask": binary_mask,
                "mask_area": mask_area,
                "area_ratio": area_ratio,
                "box": tensor_to_numpy(boxes[raw_instance_index]).reshape(-1).tolist(),
                "score": float(
                    tensor_to_numpy(scores[raw_instance_index]).reshape(-1)[0]
                ),
            }
        )
    return candidates


def deduplicate_mask_candidates(
    candidates: list[dict[str, Any]],
    iou_threshold: float,
) -> tuple[list[dict[str, Any]], int]:
    kept: list[dict[str, Any]] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (-item["score"], item["raw_instance_index"]),
    ):
        if any(
            binary_mask_iou(candidate["binary_mask"], other["binary_mask"])
            > iou_threshold
            for other in kept
        ):
            continue
        kept.append(candidate)
    return kept, len(candidates) - len(kept)


def safe_prompt_name(prompt: str) -> str:
    return "_".join(prompt.lower().replace("/", " ").split())


def stable_color(prompt: str) -> np.ndarray:
    raw = hashlib.sha256(prompt.encode("utf-8")).digest()
    return np.asarray([64 + raw[0] % 192, 64 + raw[1] % 192, 64 + raw[2] % 192])


def draw_overlay(
    image_bgr: np.ndarray,
    prompt_masks: dict[str, list[np.ndarray]],
) -> np.ndarray:
    overlay = image_bgr.copy()
    for prompt_index, (prompt, masks) in enumerate(prompt_masks.items()):
        color = stable_color(prompt)
        if not masks:
            label = f"{prompt}: none"
        else:
            union = np.logical_or.reduce([mask > 0 for mask in masks])
            overlay[union] = (0.55 * overlay[union] + 0.45 * color).astype(np.uint8)
            label = f"{prompt}: {len(masks)}"
        y = 14 + 14 * prompt_index
        cv2.rectangle(overlay, (2, y - 10), (8, y - 4), color.tolist(), -1)
        cv2.putText(
            overlay,
            label,
            (12, y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            label,
            (12, y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return overlay


def write_contact_sheet(
    overlay_paths: list[Path],
    destination: Path,
    columns: int = 5,
) -> None:
    tiles: list[np.ndarray] = []
    for path in overlay_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read generated overlay: {path}")
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        canvas = np.full((286, 256, 3), 255, dtype=np.uint8)
        canvas[:256] = image
        cv2.putText(
            canvas,
            path.stem[:34],
            (5, 276),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        tiles.append(canvas)

    rows = (len(tiles) + columns - 1) // columns
    blank = np.full_like(tiles[0], 255)
    tiles.extend([blank] * (rows * columns - len(tiles)))
    sheet_rows = [np.hstack(tiles[index:index + columns]) for index in range(0, len(tiles), columns)]
    sheet = np.vstack(sheet_rows)
    if not cv2.imwrite(str(destination), sheet):
        raise RuntimeError(f"Could not write contact sheet: {destination}")


def propose_masks(
    config_path: Path,
    mode: str,
    gpu: int,
) -> None:
    config = load_config(config_path)
    if mode == "full" and not config["full"]["approved"]:
        raise RuntimeError("Full support-mask preprocessing is blocked until the pilot is reviewed.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible. Run this command from the GPU-enabled host terminal.")
    if gpu < 0 or gpu >= torch.cuda.device_count():
        raise ValueError(f"GPU index {gpu} is invalid; visible devices: {torch.cuda.device_count()}")

    model_config = config["model"]
    model_id = str(model_config["id"])
    model_revision = resolve_model_revision(model_id, str(model_config["revision"]))
    device = f"cuda:{gpu}"

    background_root = repo_path(config["inputs"]["background_root"])
    categories = config["inputs"]["categories"]
    selected = collect_backgrounds(
        background_root=background_root,
        categories=categories,
        mode=mode,
        images_per_category=int(config["pilot"]["images_per_category"]),
        seed=int(config["seed"]),
    )
    output_root = repo_path(config[mode]["output_root"])
    prepare_output_root(output_root)

    processor = Sam3Processor.from_pretrained(model_id, revision=model_revision)
    model = Sam3Model.from_pretrained(model_id, revision=model_revision).to(device)
    model.eval()

    score_threshold = float(model_config["score_threshold"])
    mask_threshold = float(model_config["mask_threshold"])
    minimum_area_ratio = float(model_config["minimum_mask_area_ratio"])
    deduplication_config = config.get("proposal_deduplication", {})
    deduplication_enabled = bool(deduplication_config.get("enabled", False))
    deduplication_iou_threshold = float(
        deduplication_config.get("mask_iou_threshold", 1.0)
    )
    rows: list[dict[str, Any]] = []
    overlay_paths: dict[str, list[Path]] = {category: [] for category in categories}
    raw_proposal_count = 0
    deduplicated_proposal_count = 0

    for category, image_paths in selected.items():
        for image_path in image_paths:
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise RuntimeError(f"Could not read background: {image_path}")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            image_area = image_bgr.shape[0] * image_bgr.shape[1]
            background_sha256 = file_sha256(image_path)
            prompt_masks: dict[str, list[np.ndarray]] = {}

            for prompt in categories[category]:
                inputs = processor(images=pil_image, text=prompt, return_tensors="pt").to(device)
                with torch.inference_mode():
                    outputs = model(**inputs)
                result = processor.post_process_instance_segmentation(
                    outputs,
                    threshold=score_threshold,
                    mask_threshold=mask_threshold,
                    target_sizes=inputs.get("original_sizes").tolist(),
                )[0]

                masks = result.get("masks", [])
                boxes = result.get("boxes", [])
                scores = result.get("scores", [])
                prompt_masks[prompt] = []
                candidates = build_mask_candidates(
                    masks=masks,
                    boxes=boxes,
                    scores=scores,
                    image_area=image_area,
                    minimum_area_ratio=minimum_area_ratio,
                )
                raw_proposal_count += len(candidates)
                removed_count = 0
                if deduplication_enabled:
                    candidates, removed_count = deduplicate_mask_candidates(
                        candidates,
                        iou_threshold=deduplication_iou_threshold,
                    )
                deduplicated_proposal_count += removed_count

                for instance_index, candidate in enumerate(candidates):
                    binary_mask = candidate["binary_mask"]
                    mask_area = candidate["mask_area"]
                    area_ratio = candidate["area_ratio"]
                    mask_uint8 = binary_mask.astype(np.uint8) * 255
                    prompt_masks[prompt].append(mask_uint8)

                    mask_dir = output_root / "masks" / category / image_path.stem
                    mask_dir.mkdir(parents=True, exist_ok=True)
                    mask_path = mask_dir / f"{safe_prompt_name(prompt)}_{instance_index:02d}.png"
                    if not cv2.imwrite(str(mask_path), mask_uint8):
                        raise RuntimeError(f"Could not write mask: {mask_path}")

                    rows.append(
                        {
                            "background_path": repository_relative(image_path),
                            "background_sha256": background_sha256,
                            "category": category,
                            "prompt": prompt,
                            "instance_index": instance_index,
                            "raw_instance_index": candidate["raw_instance_index"],
                            "score": f"{candidate['score']:.8f}",
                            "bbox_xyxy": json.dumps(
                                [float(value) for value in candidate["box"]]
                            ),
                            "mask_area_pixels": mask_area,
                            "mask_area_ratio": f"{area_ratio:.8f}",
                            "mask_path": repository_relative(mask_path),
                            "mask_sha256": file_sha256(mask_path),
                            "proposal_status": "proposed",
                            "review_status": config["review"]["initial_status"],
                            "reviewer_note": "",
                            "model_id": model_id,
                            "model_revision": model_revision,
                        }
                    )

                if not candidates:
                    rows.append(
                        {
                            "background_path": repository_relative(image_path),
                            "background_sha256": background_sha256,
                            "category": category,
                            "prompt": prompt,
                            "instance_index": "",
                            "raw_instance_index": "",
                            "score": "",
                            "bbox_xyxy": "",
                            "mask_area_pixels": 0,
                            "mask_area_ratio": "0.00000000",
                            "mask_path": "",
                            "mask_sha256": "",
                            "proposal_status": "no_proposal",
                            "review_status": config["review"]["initial_status"],
                            "reviewer_note": "",
                            "model_id": model_id,
                            "model_revision": model_revision,
                        }
                    )

            overlay = draw_overlay(image_bgr, prompt_masks)
            overlay_dir = output_root / "overlays" / category
            overlay_dir.mkdir(parents=True, exist_ok=True)
            overlay_path = overlay_dir / f"{image_path.stem}.jpg"
            if not cv2.imwrite(str(overlay_path), overlay):
                raise RuntimeError(f"Could not write overlay: {overlay_path}")
            overlay_paths[category].append(overlay_path)

    manifest_path = output_root / "support_region_proposals.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    for category, paths in overlay_paths.items():
        write_contact_sheet(
            paths,
            output_root / "contact_sheets" / f"{category}.jpg",
        )

    summary = {
        "format_version": 2,
        "status": "pilot_pending_human_review" if mode == "pilot" else "full_pending_human_review",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "seed": int(config["seed"]),
        "model_id": model_id,
        "model_revision": model_revision,
        "model_license": model_config["license"],
        "device": device,
        "gpu_name": torch.cuda.get_device_name(gpu),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "opencv": cv2.__version__,
        "config_path": repository_relative(config_path),
        "config_sha256": file_sha256(config_path),
        "script_path": repository_relative(Path(__file__)),
        "script_sha256": file_sha256(Path(__file__)),
        "backgrounds_per_category": {
            category: len(paths) for category, paths in selected.items()
        },
        "proposal_deduplication": {
            "enabled": deduplication_enabled,
            "strategy": deduplication_config.get("strategy", "none"),
            "mask_iou_threshold": deduplication_iou_threshold,
            "raw_proposals_after_area_filter": raw_proposal_count,
            "removed_duplicate_proposals": deduplicated_proposal_count,
            "retained_proposals": raw_proposal_count - deduplicated_proposal_count,
        },
        "proposal_rows": len(rows),
        "manifest_path": repository_relative(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
    }
    with (output_root / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("No copy-paste images were generated. Review the contact sheets and proposal manifest.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Propose SAM3 semantic support masks for human review"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    propose_masks(repo_path(args.config), args.mode, args.gpu)


if __name__ == "__main__":
    main()
