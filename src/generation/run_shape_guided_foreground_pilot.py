"""Small, isolated synthetic-shape SDXL pilot for scissors, pliers, and 9V battery.

No real object pixels enter the controls. Generated images are experimental and
must pass visual review before any training or canonical use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_BASE = ROOT / "data/synthetic"
MODEL_CONFIG = ROOT / "configs/generation/genai_models_v1.yaml"
CLASSES = {
    2: ("scissors", "A single pair of ordinary stainless steel household scissors, one central metal pivot, two broad tapered cutting blades crossing at the pivot, two separate oval finger loops connected to the blades, black plastic handles. Complete assembled tool, photographed alone on a plain light gray surface."),
    3: ("pliers", "One single electrician combination pliers tool. Short tapered serrated metal jaws above one round pivot; two much longer metal handles with blue rubber grips extend below the pivot. Closed jaws, complete assembled tool, photographed alone on a plain light gray surface."),
    12: ("battery", "Exactly one rectangular 9 volt PP3 alkaline battery, a compact box-shaped body with a flat top plate and exactly two unequal round snap terminals on that top plate. Ordinary retail battery, photographed alone upright on a plain light gray surface."),
}
NEGATIVE = "duplicate, group, extra tool, extra battery, scissors as needles, pruning shears, tongs, charger, lighter, four terminals, detached pieces, hand, person, cartoon, CGI, illustration, collage, text, watermark"


def control_image(class_id: int, size: int = 1024) -> Image.Image:
    """Draw object topology and internal structure, then convert it to soft Canny edges."""
    canvas = np.zeros((size, size), dtype=np.uint8)
    if class_id == 2:
        # The blade bases overlap at one pivot; broad triangular blades taper toward tips.
        cv2.fillConvexPoly(canvas, np.array([[493, 515], [325, 175], [360, 180], [532, 492]], np.int32), 255)
        cv2.fillConvexPoly(canvas, np.array([[510, 490], [665, 180], [700, 170], [532, 515]], np.int32), 255)
        cv2.line(canvas, (500, 510), (430, 630), 255, 55)
        cv2.line(canvas, (520, 510), (605, 630), 255, 55)
        for center, angle in [((385, 725), -20), ((635, 725), 20)]:
            cv2.ellipse(canvas, center, (110, 83), angle, 0, 360, 255, -1)
            cv2.ellipse(canvas, center, (64, 43), angle, 0, 360, 0, -1)
        cv2.circle(canvas, (510, 505), 24, 120, -1)
        cv2.circle(canvas, (510, 505), 10, 255, -1)
    elif class_id == 3:
        cv2.fillConvexPoly(canvas, np.array([[490, 445], [470, 305], [505, 270], [520, 437]], np.int32), 255)
        cv2.fillConvexPoly(canvas, np.array([[520, 437], [525, 270], [560, 305], [550, 445]], np.int32), 255)
        cv2.line(canvas, (505, 450), (355, 825), 255, 70)
        cv2.line(canvas, (535, 450), (675, 825), 255, 70)
        cv2.line(canvas, (360, 645), (355, 825), 170, 58)
        cv2.line(canvas, (670, 645), (675, 825), 170, 58)
        cv2.circle(canvas, (520, 445), 32, 100, -1)
        cv2.circle(canvas, (520, 445), 12, 255, -1)
        for y in range(315, 405, 22):
            cv2.line(canvas, (483, y), (503, y+8), 95, 4)
            cv2.line(canvas, (537, y+8), (557, y), 95, 4)
    elif class_id == 12:
        cv2.rectangle(canvas, (375, 295), (650, 805), 255, -1)
        cv2.rectangle(canvas, (385, 280), (640, 315), 175, -1)
        cv2.circle(canvas, (450, 265), 31, 255, -1)
        cv2.circle(canvas, (570, 265), 22, 255, -1)
        cv2.circle(canvas, (450, 265), 15, 80, -1)
        cv2.rectangle(canvas, (395, 370), (630, 690), 170, 3)
        cv2.line(canvas, (405, 385), (405, 675), 110, 4)
    else:
        raise ValueError(f"Unsupported class: {class_id}")
    edges = cv2.Canny(cv2.GaussianBlur(canvas, (5, 5), 0), 35, 90)
    return Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--class-id", type=int, action="append", choices=tuple(CLASSES))
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument("--run-name", default="sdxl_shape_guided_v1")
    args = parser.parse_args()
    if not args.run_name.startswith("sdxl_shape_guided_") or not args.run_name.replace("_", "").isalnum():
        raise ValueError("Run name must be an sdxl_shape_guided_* identifier")
    output = OUTPUT_BASE / args.run_name
    if args.samples_per_class < 1 or args.samples_per_class > 4:
        raise ValueError("Pilot samples per class must be 1..4")
    classes = sorted(set(args.class_id or CLASSES))
    if output.exists():
        raise FileExistsError(f"Pilot output already exists: {output}")
    if not torch.cuda.is_available() or args.gpu >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {args.gpu} is unavailable")
    model = yaml.safe_load(MODEL_CONFIG.read_text())["sdxl"]
    controlnet = ControlNetModel.from_pretrained(model["controlnet"]["id"],
        revision=model["controlnet"]["revision"], torch_dtype=torch.float16,
        variant=model["variant"], use_safetensors=True)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(model["base_model"]["id"],
        revision=model["base_model"]["revision"], controlnet=controlnet,
        torch_dtype=torch.float16, variant=model["variant"], use_safetensors=True)
    pipe.enable_model_cpu_offload(gpu_id=args.gpu)
    pipe.vae.enable_slicing()
    for cid in classes:
        prompt = CLASSES[cid][1]
        for tokenizer in (pipe.tokenizer, pipe.tokenizer_2):
            for value in (prompt, NEGATIVE):
                if len(tokenizer(value, truncation=False)["input_ids"]) > tokenizer.model_max_length:
                    raise ValueError(f"Prompt exceeds {tokenizer.model_max_length} tokens: {cid}")
    output.mkdir(parents=True)
    records = []
    for cid in classes:
        control = control_image(cid)
        control_path = output / f"c{cid:02d}_control.png"
        control.save(control_path)
        control_hash = sha256(control_path)
        for attempt in range(args.samples_per_class):
            seed = 83000 + cid * 100 + attempt
            image = pipe(prompt=CLASSES[cid][1], negative_prompt=NEGATIVE,
                         image=control, controlnet_conditioning_scale=0.55,
                         guidance_scale=5.0, num_inference_steps=30,
                         generator=torch.Generator(device="cpu").manual_seed(seed)).images[0]
            path = output / f"c{cid:02d}_{attempt:02d}.png"
            image.save(path)
            records.append({"class_id": cid, "class_name": CLASSES[cid][0],
                            "seed": seed, "prompt": CLASSES[cid][1],
                            "negative_prompt": NEGATIVE, "control_path": str(control_path.relative_to(ROOT)),
                            "control_sha256": control_hash,
                            "image_path": str(path.relative_to(ROOT)), "sha256": sha256(path),
                            "status": "pending_visual_review"})
            print(f"Generated {path}", flush=True)
    (output / "manifest.json").write_text(json.dumps({"status": "experimental_pending_visual_review",
        "model_config": str(MODEL_CONFIG.relative_to(ROOT)),
        "model_config_sha256": sha256(MODEL_CONFIG),
        "base_model_revision": model["base_model"]["revision"],
        "controlnet_revision": model["controlnet"]["revision"],
        "script_sha256": sha256(Path(__file__)), "gpu": args.gpu,
        "control_scale": 0.55, "steps": 30, "guidance_scale": 5.0,
        "records": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
