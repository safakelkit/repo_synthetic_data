from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = REPO_ROOT / "data/processed/genai_v1/shared_conditioning_plan.jsonl"
DEFAULT_PILOT_IDS = REPO_ROOT / "data/processed/genai_v1/pilot_image_ids.json"
DEFAULT_CONFIG = REPO_ROOT / "configs/generation/genai_shared_v1.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "data/processed/genai_v1/conditioning_preview"


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def load_records(plan_path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for line in plan_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        records[int(record["image_id"])] = record
    return records


def reconstruct_initialization(record: dict[str, Any]) -> Image.Image:
    background = Image.open(repo_path(record["background"])).convert("RGB")
    rgba = Image.open(repo_path(record["object_rgba"])).convert("RGBA")
    x1, y1, x2, y2 = map(int, record["intended_bbox_xyxy"])
    target_width, target_height = x2 - x1, y2 - y1
    if target_width <= 0 or target_height <= 0:
        raise ValueError(f"Invalid intended bbox for image_id={record['image_id']}")
    rgba = rgba.resize((target_width, target_height), Image.Resampling.LANCZOS)
    composite = background.convert("RGBA")
    composite.alpha_composite(rgba, (x1, y1))
    return composite.convert("RGB")


def scale_bbox(record: dict[str, Any], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = record["inpaint_bbox_normalized_xyxy"]
    return (
        round(x1 * width),
        round(y1 * height),
        round(x2 * width),
        round(y2 * height),
    )


def make_mask(record: dict[str, Any], width: int, height: int) -> Image.Image:
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).rectangle(scale_bbox(record, width, height), fill=255)
    return mask


def make_canny(image: Image.Image, low: int, high: int) -> Image.Image:
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low, high)
    return Image.fromarray(np.repeat(edges[:, :, None], 3, axis=2), mode="RGB")


def make_panel(
    initialization: Image.Image, mask: Image.Image, canny: Image.Image, record: dict[str, Any]
) -> Image.Image:
    overlay = initialization.copy()
    red = Image.new("RGB", overlay.size, (255, 0, 0))
    overlay.paste(Image.blend(overlay, red, 0.35), mask=mask)
    panels = [initialization, overlay, canny]
    canvas = Image.new("RGB", (sum(x.width for x in panels), panels[0].height + 34), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 34))
        x += panel.width
    ImageDraw.Draw(canvas).text(
        (8, 8),
        f"id={record['image_id']} class={record['class_name']} support={record['support_type']}",
        fill="black",
    )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the shared GenAI initialization, inpaint masks, and controls"
    )
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--pilot-ids", type=Path, default=DEFAULT_PILOT_IDS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    plan_path = repo_path(args.plan)
    pilot_path = repo_path(args.pilot_ids)
    config_path = repo_path(args.config)
    output_root = repo_path(args.output)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {output_root}")

    records = load_records(plan_path)
    pilot_ids = [int(value) for value in json.loads(pilot_path.read_text(encoding="utf-8"))]
    config = load_yaml(config_path)
    width = int(config["conditioning"]["inference_width"])
    height = int(config["conditioning"]["inference_height"])
    sdxl = load_yaml(REPO_ROOT / "configs/generation/sdxl_controlnet_v1.yaml")
    low = int(sdxl["inference"]["canny_low_threshold"])
    high = int(sdxl["inference"]["canny_high_threshold"])

    manifest: list[dict[str, Any]] = []
    for image_id in pilot_ids:
        record = records[image_id]
        sample_root = output_root / f"sample_{image_id:06d}"
        sample_root.mkdir(parents=True, exist_ok=False)
        initialization = reconstruct_initialization(record).resize(
            (width, height), Image.Resampling.LANCZOS
        )
        mask = make_mask(record, width, height)
        canny = make_canny(initialization, low, high)
        files = {
            "initialization": sample_root / "initialization.png",
            "inpaint_mask": sample_root / "inpaint_mask.png",
            "canny_control": sample_root / "canny_control.png",
            "preview": sample_root / "preview.jpg",
        }
        initialization.save(files["initialization"])
        mask.save(files["inpaint_mask"])
        canny.save(files["canny_control"])
        make_panel(initialization, mask, canny, record).save(files["preview"], quality=92)
        manifest.append(
            {
                "image_id": image_id,
                "class_id": record["class_id"],
                "class_name": record["class_name"],
                "prompt": record["prompt"],
                "inpaint_bbox_1024_xyxy": list(scale_bbox(record, width, height)),
                "files": {
                    key: path.relative_to(REPO_ROOT).as_posix() for key, path in files.items()
                },
                "sha256": {key: sha256(path) for key, path in files.items()},
            }
        )

    manifest_path = output_root / "conditioning_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    summary = {
        "format_version": 1,
        "status": "conditioning_preview_ready_for_human_review",
        "records": len(manifest),
        "plan_sha256": sha256(plan_path),
        "pilot_ids_sha256": sha256(pilot_path),
        "config_sha256": sha256(config_path),
        "manifest": manifest_path.relative_to(REPO_ROOT).as_posix(),
        "manifest_sha256": sha256(manifest_path),
        "no_model_inference_performed": True,
    }
    (output_root / "conditioning_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
