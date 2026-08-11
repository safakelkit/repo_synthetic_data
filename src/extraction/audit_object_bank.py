from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BANK = REPO_ROOT / "data/processed/object_bank_sam3"
SOURCE_IMAGES = REPO_ROOT / "data/raw/insp-det/train/images"
SOURCE_LABELS = REPO_ROOT / "data/raw/insp-det/train/labels"
ASSET_KINDS = ("rgb", "mask", "rgba")


def load_metadata(bank_root: Path) -> dict[tuple[int, str], dict[str, Any]]:
    metadata_path = bank_root / "crop_metadata_train.json"
    if not metadata_path.is_file():
        return {}

    with open(metadata_path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    return {
        (int(row["class_id"]), Path(row["rgba_path"]).name): row
        for row in rows
    }


def reconstruct_provenance(stem: str, class_id: int) -> dict[str, Any]:
    match = re.match(r"^(?P<source>.+)_obj_(?P<index>\d+)_crop_\d+$", stem)
    if not match:
        return {"status": "unresolved", "reason": "unparseable_asset_stem"}

    source_stem = match.group("source")
    annotation_index = int(match.group("index"))
    image_matches = sorted(path for path in SOURCE_IMAGES.glob(f"{source_stem}.*") if path.is_file())
    label_path = SOURCE_LABELS / f"{source_stem}.txt"
    if len(image_matches) != 1 or not label_path.is_file():
        return {"status": "unresolved", "reason": "source_file_not_found"}

    annotations = [line.split() for line in label_path.read_text().splitlines() if line.strip()]
    if annotation_index >= len(annotations):
        return {"status": "unresolved", "reason": "annotation_index_out_of_range"}

    annotation_class = int(float(annotations[annotation_index][0]))
    provenance = {
        "source_image": str(image_matches[0]),
        "source_label": str(label_path),
        "source_annotation_index": annotation_index,
        "source_annotation_class": annotation_class,
    }
    if annotation_class != class_id:
        return {
            **provenance,
            "status": "mismatch",
            "reason": "source_annotation_class_mismatch",
        }
    return {**provenance, "status": "reconstructed", "reason": ""}


def read_image(path: Path, flags: int) -> np.ndarray | None:
    if not path.is_file():
        return None
    return cv2.imread(str(path), flags)


def tight_bbox(binary: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(binary)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_asset(
    class_id: int,
    class_name: str,
    stem: str,
    paths: dict[str, Path],
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    rejection_reasons: list[str] = []
    warnings: list[str] = []

    for kind in ASSET_KINDS:
        if not paths[kind].is_file():
            rejection_reasons.append(f"missing_{kind}")

    rgb = read_image(paths["rgb"], cv2.IMREAD_COLOR)
    mask = read_image(paths["mask"], cv2.IMREAD_GRAYSCALE)
    rgba = read_image(paths["rgba"], cv2.IMREAD_UNCHANGED)

    if paths["rgb"].is_file() and rgb is None:
        rejection_reasons.append("unreadable_rgb")
    if paths["mask"].is_file() and mask is None:
        rejection_reasons.append("unreadable_mask")
    if paths["rgba"].is_file() and rgba is None:
        rejection_reasons.append("unreadable_rgba")

    height = width = visible_pixels = 0
    visible_ratio = alpha_mask_iou = alpha_mask_agreement = 0.0
    padding_ratio = 1.0
    bbox = None
    mask_unique_values = ""
    touches_border = False

    if rgba is not None and (rgba.ndim != 3 or rgba.shape[2] != 4):
        rejection_reasons.append("rgba_not_four_channel")
        rgba = None

    shapes = [image.shape[:2] for image in (rgb, mask, rgba) if image is not None]
    if shapes and any(shape != shapes[0] for shape in shapes[1:]):
        rejection_reasons.append("shape_mismatch")

    if rgba is not None:
        height, width = rgba.shape[:2]
        if height < 8 or width < 8:
            rejection_reasons.append("crop_too_small")

        alpha_binary = rgba[:, :, 3] > 20
        visible_pixels = int(alpha_binary.sum())
        visible_ratio = visible_pixels / max(1, height * width)
        bbox = tight_bbox(alpha_binary)

        if visible_pixels < 40 or bbox is None:
            rejection_reasons.append("empty_or_tiny_alpha")
        else:
            x1, y1, x2, y2 = bbox
            tight_area = (x2 - x1) * (y2 - y1)
            padding_ratio = 1.0 - tight_area / (height * width)
            touches_border = x1 == 0 or y1 == 0 or x2 == width or y2 == height
            if padding_ratio > 0.50:
                warnings.append("high_transparent_padding")

    if mask is not None:
        values = np.unique(mask)
        mask_unique_values = ";".join(str(int(value)) for value in values[:20])
        if len(values) > 20:
            mask_unique_values += ";..."
        if not np.any(mask > 20):
            rejection_reasons.append("empty_mask")
        if not set(int(value) for value in values).issubset({0, 255}):
            warnings.append("non_binary_mask_values")

    if rgba is not None and mask is not None and rgba.shape[:2] == mask.shape[:2]:
        alpha_binary = rgba[:, :, 3] > 20
        mask_binary = mask > 20
        intersection = int(np.logical_and(alpha_binary, mask_binary).sum())
        union = int(np.logical_or(alpha_binary, mask_binary).sum())
        alpha_mask_iou = intersection / union if union else 1.0
        alpha_mask_agreement = float(np.mean(alpha_binary == mask_binary))
        if alpha_mask_iou < 0.95:
            rejection_reasons.append("alpha_mask_iou_below_0_95")

    if rgb is not None and rgba is not None and rgb.shape[:2] == rgba.shape[:2]:
        visible = rgba[:, :, 3] > 20
        if np.any(visible):
            difference = np.abs(
                rgb[visible].astype(np.int16) - rgba[:, :, :3][visible].astype(np.int16)
            )
            if float(difference.mean()) > 1.0:
                warnings.append("rgb_rgba_visible_pixels_differ")

    provenance = (
        {
            "status": "metadata",
            "reason": "",
            "source_image": metadata["source_image"],
            "source_label": metadata["source_label"],
            "source_annotation_index": int(re.search(r"_obj_(\d+)_", stem).group(1)),
            "source_annotation_class": class_id,
        }
        if metadata
        else reconstruct_provenance(stem, class_id)
    )
    if provenance["status"] in {"unresolved", "mismatch"}:
        rejection_reasons.append(provenance["reason"])

    rejection_reasons = sorted(set(rejection_reasons))
    warnings = sorted(set(warnings))
    rgba_hash = sha256(paths["rgba"]) if paths["rgba"].is_file() else ""

    return {
        "asset_id": f"{class_id:02d}_{stem}",
        "class_id": class_id,
        "class_name": class_name,
        "stem": stem,
        "rgb_path": str(paths["rgb"].relative_to(REPO_ROOT)),
        "mask_path": str(paths["mask"].relative_to(REPO_ROOT)),
        "rgba_path": str(paths["rgba"].relative_to(REPO_ROOT)),
        "source_image": provenance.get("source_image", ""),
        "source_label": provenance.get("source_label", ""),
        "source_annotation_index": provenance.get("source_annotation_index", ""),
        "source_annotation_class": provenance.get("source_annotation_class", ""),
        "provenance_status": provenance["status"],
        "sam_score": metadata.get("sam_score", "") if metadata else "",
        "metadata_matched": metadata is not None,
        "width": width,
        "height": height,
        "visible_pixels": visible_pixels,
        "visible_ratio": visible_ratio,
        "tight_bbox_xyxy": list(bbox) if bbox else "",
        "transparent_padding_ratio": padding_ratio,
        "alpha_touches_border": touches_border,
        "mask_unique_values": mask_unique_values,
        "alpha_mask_iou": alpha_mask_iou,
        "alpha_mask_agreement": alpha_mask_agreement,
        "rgba_sha256": rgba_hash,
        "exact_duplicate_group": "",
        "warnings": ";".join(warnings),
        "status": "rejected" if rejection_reasons else "accepted",
        "rejection_reason": ";".join(rejection_reasons),
    }


def mark_exact_duplicates(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["rgba_sha256"]:
            groups[(row["class_id"], row["rgba_sha256"])].append(row)

    duplicate_index = 0
    for group in groups.values():
        if len(group) < 2:
            continue
        duplicate_index += 1
        group_id = f"exact_{duplicate_index:04d}"
        for duplicate in group:
            duplicate["exact_duplicate_group"] = group_id
            warnings = set(filter(None, duplicate["warnings"].split(";")))
            warnings.add("exact_duplicate_rgba")
            duplicate["warnings"] = ";".join(sorted(warnings))


def make_contact_sheets(
    rows: list[dict[str, Any]],
    output_dir: Path,
    samples_per_class: int,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    tile_size = 180
    columns = 5

    for class_id in range(16):
        candidates = [
            row for row in rows
            if row["class_id"] == class_id and row["status"] == "accepted"
        ]
        selected = rng.sample(candidates, min(samples_per_class, len(candidates)))
        sheet_rows = max(1, (len(selected) + columns - 1) // columns)
        sheet = np.full((sheet_rows * tile_size, columns * tile_size, 3), 235, np.uint8)

        for index, row in enumerate(selected):
            rgba = cv2.imread(str(REPO_ROOT / row["rgba_path"]), cv2.IMREAD_UNCHANGED)
            if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
                continue
            alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
            checker = np.full(rgba.shape[:2] + (3,), 220, np.uint8)
            checker[::16, :] = 190
            checker[:, ::16] = 190
            composite = (rgba[:, :, :3] * alpha + checker * (1 - alpha)).astype(np.uint8)
            scale = min((tile_size - 34) / composite.shape[1], (tile_size - 34) / composite.shape[0])
            resized = cv2.resize(
                composite,
                (max(1, int(composite.shape[1] * scale)), max(1, int(composite.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
            row_index, column_index = divmod(index, columns)
            y0 = row_index * tile_size + 20 + (tile_size - 34 - resized.shape[0]) // 2
            x0 = column_index * tile_size + (tile_size - resized.shape[1]) // 2
            sheet[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
            cv2.putText(
                sheet,
                row["stem"][-18:],
                (column_index * tile_size + 4, row_index * tile_size + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.33,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )

        class_name = selected[0]["class_name"] if selected else f"class_{class_id}"
        output_path = output_dir / f"{class_id:02d}_{class_name.replace(' ', '_')}.jpg"
        cv2.imwrite(str(output_path), sheet)


def audit_object_bank(
    bank_root: Path,
    samples_per_class: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    metadata_index = load_metadata(bank_root)
    rows: list[dict[str, Any]] = []

    class_dirs = sorted(
        path
        for path in bank_root.iterdir()
        if path.is_dir() and re.match(r"^\d{2}_", path.name)
    )
    for class_dir in class_dirs:
        class_id = int(class_dir.name.split("_")[0])
        class_name = class_dir.name.split("_", 1)[1].replace("_", " ")
        stems: set[str] = set()
        for kind in ASSET_KINDS:
            stems.update(path.stem for path in (class_dir / kind).glob("*.png"))

        for stem in sorted(stems):
            paths = {kind: class_dir / kind / f"{stem}.png" for kind in ASSET_KINDS}
            rows.append(
                audit_asset(
                    class_id,
                    class_name,
                    stem,
                    paths,
                    metadata_index.get((class_id, f"{stem}.png")),
                )
            )

    mark_exact_duplicates(rows)

    manifest_path = bank_root / "asset_audit_manifest.csv"
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    per_class: dict[str, Any] = {}
    for class_id in range(16):
        class_rows = [row for row in rows if row["class_id"] == class_id]
        per_class[str(class_id)] = {
            "class_name": class_rows[0]["class_name"] if class_rows else "",
            "total": len(class_rows),
            "accepted": sum(row["status"] == "accepted" for row in class_rows),
            "rejected": sum(row["status"] == "rejected" for row in class_rows),
            "metadata_matched": sum(bool(row["metadata_matched"]) for row in class_rows),
            "warning_counts": dict(
                Counter(
                    warning
                    for row in class_rows
                    for warning in row["warnings"].split(";")
                    if warning
                )
            ),
            "rejection_counts": dict(
                Counter(
                    reason
                    for row in class_rows
                    for reason in row["rejection_reason"].split(";")
                    if reason
                )
            ),
        }

    visual_review_path = bank_root / "asset_visual_sample_review.json"
    visual_review = None
    if visual_review_path.is_file():
        with open(visual_review_path, "r", encoding="utf-8") as f:
            visual_review = json.load(f)

    summary = {
        "format_version": 1,
        "bank_root": str(bank_root.relative_to(REPO_ROOT)),
        "audit_seed": seed,
        "visual_samples_per_class": samples_per_class,
        "total_assets": len(rows),
        "accepted": sum(row["status"] == "accepted" for row in rows),
        "rejected": sum(row["status"] == "rejected" for row in rows),
        "metadata_entries": len(metadata_index),
        "metadata_matched": sum(bool(row["metadata_matched"]) for row in rows),
        "provenance_counts": dict(Counter(row["provenance_status"] for row in rows)),
        "exact_duplicate_assets": sum(bool(row["exact_duplicate_group"]) for row in rows),
        "warning_counts": dict(
            Counter(
                warning
                for row in rows
                for warning in row["warnings"].split(";")
                if warning
            )
        ),
        "rejection_counts": dict(
            Counter(
                reason
                for row in rows
                for reason in row["rejection_reason"].split(";")
                if reason
            )
        ),
        "per_class": per_class,
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "visual_review_status": "sampled_10_per_class" if visual_review else "pending",
        "visual_reviewed": visual_review.get("reviewed_assets", 0) if visual_review else 0,
        "visual_pass": visual_review.get("visual_pass", 0) if visual_review else 0,
        "visual_flag": visual_review.get("visual_flag", 0) if visual_review else 0,
        "visual_review_file": (
            str(visual_review_path.relative_to(REPO_ROOT)) if visual_review else ""
        ),
    }
    summary_path = bank_root / "asset_audit_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    make_contact_sheets(
        rows,
        bank_root / "asset_audit_visuals",
        samples_per_class=samples_per_class,
        seed=seed,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit SAM3 RGB/mask/RGBA object assets")
    parser.add_argument("--bank-root", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--samples-per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = audit_object_bank(
        bank_root=args.bank_root.resolve(),
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
