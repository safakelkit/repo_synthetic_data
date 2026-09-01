from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/placement/support_geometry_v1.yaml"
REGION_FIELDS = [
    "background_path", "background_sha256", "category", "support_type",
    "source_prompts", "source_mask_paths", "source_mask_sha256s",
    "region_path", "region_sha256", "region_area_pixels", "region_area_ratio",
    "anchor_points_xy", "anchor_count", "derivation_status", "review_status",
    "reviewer_note",
]


def repo_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Invalid YAML mapping: {path}")
    return value


def read_manifest(path: Path) -> dict[str, list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["background_path"]].append(row)
    return dict(grouped)


def validate_inputs(manifest: Path, summary: Path) -> dict[str, Any]:
    with summary.open("r", encoding="utf-8") as handle:
        run = json.load(handle)
    if sha256(manifest) != run["manifest_sha256"]:
        raise ValueError("Proposal manifest checksum differs from v2 run summary")
    return run


def load_prompt_masks(
    rows: list[dict[str, str]], prompts: list[str], minimum_score: float,
    maximum_instances: int,
) -> tuple[dict[str, np.ndarray], list[str], list[str]]:
    masks: dict[str, np.ndarray] = {}
    paths: list[str] = []
    checksums: list[str] = []
    for prompt in prompts:
        candidates = sorted(
            (
                row for row in rows
                if row["prompt"] == prompt and row["proposal_status"] == "proposed"
                and float(row["score"]) >= minimum_score
            ),
            key=lambda row: (-float(row["score"]), int(row["instance_index"])),
        )[:maximum_instances]
        prompt_union: np.ndarray | None = None
        for row in candidates:
            mask_path = repo_path(row["mask_path"])
            if sha256(mask_path) != row["mask_sha256"]:
                raise ValueError(f"Mask checksum mismatch: {mask_path}")
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"Could not read mask: {mask_path}")
            binary = mask > 0
            prompt_union = binary if prompt_union is None else (prompt_union | binary)
            paths.append(row["mask_path"])
            checksums.append(row["mask_sha256"])
        if prompt_union is not None:
            masks[prompt] = prompt_union
    return masks, paths, checksums


def consensus_mask(prompt_masks: dict[str, np.ndarray], minimum: int) -> np.ndarray | None:
    if len(prompt_masks) < minimum:
        return None
    stack = np.stack(list(prompt_masks.values()), axis=0).astype(np.uint8)
    return stack.sum(axis=0) >= minimum


def ellipse_kernel(radius: int) -> np.ndarray:
    size = max(1, 2 * radius + 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def clean_components(mask: np.ndarray, minimum_pixels: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_pixels:
            cleaned[labels == label] = 1
    return cleaned.astype(bool)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def keep_component_vertical_fraction(mask: np.ndarray, low: float, high: float) -> np.ndarray:
    output = np.zeros_like(mask, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    for label in range(1, count):
        x, y, width, height, _ = stats[label]
        start = y + int(round(height * low))
        stop = y + max(int(round(height * high)), int(round(height * low)) + 1)
        component = labels == label
        band = np.zeros_like(mask, dtype=bool)
        band[start:min(stop, y + height), x:x + width] = True
        output[component & band] = 1
    return output.astype(bool)


def derive_region(
    support_type: str, base: np.ndarray, spec: dict[str, Any], image_shape: tuple[int, int]
) -> np.ndarray:
    height, width = image_shape
    if support_type == "floor":
        low, high = spec["vertical_keep"]
        vertical = np.zeros_like(base, dtype=bool)
        vertical[int(height * low):int(height * high), :] = True
        result = base & vertical
        radius = max(1, int(round(min(height, width) * float(spec["erosion_ratio"]))))
        return cv2.erode(result.astype(np.uint8), ellipse_kernel(radius)) > 0

    low, high = spec["component_vertical_keep"]
    result = keep_component_vertical_fraction(base, float(low), float(high))
    if support_type == "bed_top":
        radius = max(1, int(round(min(height, width) * float(spec["erosion_ratio"]))))
        return cv2.erode(result.astype(np.uint8), ellipse_kernel(radius)) > 0

    # A dining-table anchor represents the visible upper support boundary, not
    # arbitrary pixels from the full table silhouette.
    band_height = max(2, int(round(height * float(spec["band_height_ratio"]))))
    boundary_band = np.zeros_like(result, dtype=np.uint8)
    for x in range(width):
        ys = np.flatnonzero(result[:, x])
        if ys.size:
            top = int(ys.min())
            boundary_band[top:min(height, top + band_height), x] = 1
    horizontal = max(1, int(round(width * float(spec["horizontal_erosion_ratio"]))))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * horizontal + 1, 3))
    return cv2.erode(boundary_band, kernel) > 0


def deterministic_anchors(
    mask: np.ndarray, maximum: int, minimum_spacing_ratio: float,
) -> list[tuple[int, int]]:
    height, width = mask.shape
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    minimum_spacing = max(2.0, min(height, width) * minimum_spacing_ratio)
    anchors: list[tuple[int, int]] = []
    work = distance.copy()
    for _ in range(maximum):
        _, maximum_value, _, maximum_location = cv2.minMaxLoc(work)
        if maximum_value <= 0:
            break
        x, y = maximum_location
        anchors.append((int(x), int(y)))
        cv2.circle(work, (x, y), int(math.ceil(minimum_spacing)), 0.0, -1)
    return anchors


def draw_geometry_overlay(
    image: np.ndarray, regions: list[tuple[str, np.ndarray, list[tuple[int, int]]]],
    footprint_specs: list[dict[str, Any]],
) -> np.ndarray:
    output = image.copy()
    colors = {"floor": (70, 180, 70), "bed_top": (210, 120, 40), "dining_table_top": (40, 120, 220)}
    for support_type, mask, anchors in regions:
        color = np.asarray(colors[support_type], dtype=np.uint8)
        output[mask] = (0.62 * output[mask] + 0.38 * color).astype(np.uint8)
        for index, (x, y) in enumerate(anchors):
            cv2.circle(output, (x, y), 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(output, (x, y), 5, colors[support_type], 1, cv2.LINE_AA)
            if index < len(footprint_specs):
                spec = footprint_specs[index]
                half_w = int(image.shape[1] * float(spec["width_ratio"]) / 2)
                half_h = int(image.shape[0] * float(spec["height_ratio"]) / 2)
                cv2.rectangle(output, (x - half_w, y - half_h), (x + half_w, y + half_h), colors[support_type], 1)
        cv2.putText(output, f"{support_type}: {len(anchors)} anchors", (8, 18 + 18 * list(colors).index(support_type)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors[support_type], 1, cv2.LINE_AA)
    return output


def write_contact_sheet(paths: list[Path], destination: Path, columns: int = 5) -> None:
    tiles = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        tile = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        canvas = np.full((282, 256, 3), 255, np.uint8)
        canvas[:256] = tile
        cv2.putText(canvas, path.stem[:34], (5, 274), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(canvas)
    rows = math.ceil(len(tiles) / columns)
    tiles.extend([np.full_like(tiles[0], 255)] * (rows * columns - len(tiles)))
    sheet = np.vstack([np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)])
    if not cv2.imwrite(str(destination), sheet):
        raise RuntimeError(f"Could not write contact sheet: {destination}")


def derive(config_path: Path, mode: str = "pilot") -> dict[str, Any]:
    config = load_yaml(config_path)
    if mode == "pilot":
        manifest_path = repo_path(config["inputs"]["proposal_manifest"])
        summary_path = repo_path(config["inputs"]["proposal_run_summary"])
        output_root = repo_path(config["output"]["root"])
    else:
        manifest_path = repo_path(config["inputs"]["full_proposal_manifest"])
        summary_path = repo_path(config["inputs"]["full_proposal_run_summary"])
        output_root = repo_path(config["output"]["full_root"])
    proposal_run = validate_inputs(manifest_path, summary_path)
    grouped = read_manifest(manifest_path)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Geometry output is not empty: {output_root}")
    for folder in ("regions", "overlays", "contact_sheets"):
        (output_root / folder).mkdir(parents=True, exist_ok=True)

    selection = config["proposal_selection"]
    geometry = config["geometry"]
    minimum_region_ratio = float(geometry["minimum_region_area_ratio"])
    rows_out: list[dict[str, Any]] = []
    overlays: dict[str, list[Path]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)

    for background_path_text, proposal_rows in sorted(grouped.items()):
        background_path = repo_path(background_path_text)
        image = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read background: {background_path}")
        height, width = image.shape[:2]
        image_area = height * width
        category = proposal_rows[0]["category"]
        if sha256(background_path) != proposal_rows[0]["background_sha256"]:
            raise ValueError(f"Background checksum mismatch: {background_path}")
        overlay_regions = []

        for support_type in ("floor", "bed_top", "dining_table_top"):
            spec = geometry[support_type]
            if "categories" in spec and category not in spec["categories"]:
                continue
            masks, source_paths, source_hashes = load_prompt_masks(
                proposal_rows, list(spec["prompts"]), float(selection["minimum_score"]),
                int(selection["maximum_instances_per_prompt"]),
            )
            base = consensus_mask(masks, int(spec["minimum_consensus"]))
            if base is None:
                region = np.zeros((height, width), dtype=bool)
            else:
                region = derive_region(support_type, base, spec, (height, width))
                margin = max(1, int(round(min(height, width) * float(geometry["boundary_margin_ratio"]))))
                region[:margin, :] = False
                region[-margin:, :] = False
                region[:, :margin] = False
                region[:, -margin:] = False
                region = clean_components(region, max(1, int(image_area * float(geometry["minimum_component_area_ratio"]))))
                if geometry.get("component_selection") == "largest_only":
                    region = keep_largest_component(region)
            area = int(region.sum())
            status = "derived" if area / image_area >= minimum_region_ratio else "no_valid_region"
            anchors = deterministic_anchors(region, int(config["anchors"]["maximum_per_region"]), float(config["anchors"]["minimum_spacing_ratio"])) if status == "derived" else []
            region_path_text = ""
            region_hash = ""
            if status == "derived":
                region_dir = output_root / "regions" / category / background_path.stem
                region_dir.mkdir(parents=True, exist_ok=True)
                region_path = region_dir / f"{support_type}.png"
                if not cv2.imwrite(str(region_path), region.astype(np.uint8) * 255):
                    raise RuntimeError(f"Could not write region: {region_path}")
                region_path_text = relative(region_path)
                region_hash = sha256(region_path)
                counts[support_type] += 1
                overlay_regions.append((support_type, region, anchors))
            rows_out.append({
                "background_path": background_path_text,
                "background_sha256": proposal_rows[0]["background_sha256"],
                "category": category,
                "support_type": support_type,
                "source_prompts": json.dumps(list(spec["prompts"])),
                "source_mask_paths": json.dumps(source_paths),
                "source_mask_sha256s": json.dumps(source_hashes),
                "region_path": region_path_text,
                "region_sha256": region_hash,
                "region_area_pixels": area,
                "region_area_ratio": f"{area / image_area:.8f}",
                "anchor_points_xy": json.dumps(anchors),
                "anchor_count": len(anchors),
                "derivation_status": status,
                "review_status": config["review"]["initial_status"],
                "reviewer_note": "",
            })

        overlay = draw_geometry_overlay(image, overlay_regions, config["anchors"]["footprint_sizes"])
        overlay_dir = output_root / "overlays" / category
        overlay_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = overlay_dir / f"{background_path.stem}.jpg"
        if not cv2.imwrite(str(overlay_path), overlay):
            raise RuntimeError(f"Could not write overlay: {overlay_path}")
        overlays[category].append(overlay_path)

    region_manifest = output_root / "support_regions.csv"
    with region_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGION_FIELDS)
        writer.writeheader()
        writer.writerows(rows_out)
    for category, paths in overlays.items():
        write_contact_sheet(paths, output_root / "contact_sheets" / f"{category}.jpg")

    summary = {
        "format_version": 1,
        "status": f"geometry_{mode}_pending_human_review",
        "mode": mode,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": relative(config_path), "config_sha256": sha256(config_path),
        "script_path": relative(Path(__file__)), "script_sha256": sha256(Path(__file__)),
        "proposal_manifest_path": relative(manifest_path), "proposal_manifest_sha256": sha256(manifest_path),
        "proposal_model_revision": proposal_run["model_revision"],
        "background_count": len(grouped), "manifest_rows": len(rows_out),
        "derived_regions_by_type": dict(sorted(counts.items())),
        "region_manifest_path": relative(region_manifest), "region_manifest_sha256": sha256(region_manifest),
    }
    with (output_root / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive conservative support geometry from SAM3 v2 proposals")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    args = parser.parse_args()
    summary = derive(repo_path(args.config), mode=args.mode)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("No copy-paste images were generated. Review geometry contact sheets and manifest.")


if __name__ == "__main__":
    main()
