from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = REPO_ROOT / "data/processed/genai_v1/shared_conditioning_plan.jsonl"
PILOT_IDS_PATH = REPO_ROOT / "data/processed/genai_v1/pilot_image_ids.json"
CONDITIONING_ROOT = REPO_ROOT / "data/processed/genai_v1/conditioning_preview"
BACKEND_CONFIGS = {
    "sdxl": REPO_ROOT / "configs/generation/sdxl_controlnet_v1.yaml",
    "qwen": REPO_ROOT / "configs/generation/qwen_controlnet_v1.yaml",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def load_record(image_id: int) -> dict[str, Any]:
    for line in PLAN_PATH.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if int(value["image_id"]) == image_id:
            return value
    raise KeyError(f"image_id not found in shared plan: {image_id}")


def require_exact_revisions(backend: str) -> dict[str, Any]:
    path = REPO_ROOT / f"data/processed/genai_v1/{backend}_model_preflight.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Run preflight_genai_backend.py --backend {backend} before inference"
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "ready_for_smoke_test":
        raise RuntimeError(f"Preflight is not ready: {path}")
    for key in ("base_model", "controlnet"):
        revision = report["models"][key].get("resolved_revision")
        if not revision or len(revision) < 20:
            raise ValueError(f"Missing immutable revision for {key}: {path}")
    return report


def git_record() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip())
    return {"revision": revision, "dirty": dirty}


def version(package: str) -> str:
    return importlib.metadata.version(package)


def load_inputs(image_id: int) -> tuple[Image.Image, Image.Image, Image.Image, dict[str, str]]:
    root = CONDITIONING_ROOT / f"sample_{image_id:06d}"
    paths = {
        "initialization": root / "initialization.png",
        "inpaint_mask": root / "inpaint_mask.png",
        "canny_control": root / "canny_control.png",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing conditioning files: {missing}")
    return (
        Image.open(paths["initialization"]).convert("RGB"),
        Image.open(paths["inpaint_mask"]).convert("L"),
        Image.open(paths["canny_control"]).convert("RGB"),
        {key: sha256(path) for key, path in paths.items()},
    )


def generate_sdxl(
    config: dict[str, Any], preflight: dict[str, Any], record: dict[str, Any],
    init_image: Image.Image, mask: Image.Image, canny: Image.Image, gpu: int,
) -> Image.Image:
    from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline

    dtype = torch.float16
    controlnet = ControlNetModel.from_pretrained(
        config["controlnet"]["id"],
        revision=preflight["models"]["controlnet"]["resolved_revision"],
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        config["base_model"]["id"],
        revision=preflight["models"]["base_model"]["resolved_revision"],
        controlnet=controlnet,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipe.enable_model_cpu_offload(gpu_id=gpu)
    settings = config["inference"]
    generator = torch.Generator(device="cpu").manual_seed(
        int(record["stable_diffusion_seed"])
    )
    return pipe(
        prompt=record["prompt"],
        negative_prompt=record["negative_prompt"],
        image=init_image,
        mask_image=mask,
        control_image=canny,
        width=int(settings["width"]),
        height=int(settings["height"]),
        num_inference_steps=int(settings["num_inference_steps"]),
        guidance_scale=float(settings["guidance_scale"]),
        controlnet_conditioning_scale=float(settings["controlnet_conditioning_scale"]),
        strength=float(settings["strength"]),
        generator=generator,
    ).images[0]


def generate_qwen(
    config: dict[str, Any], preflight: dict[str, Any], record: dict[str, Any],
    init_image: Image.Image, mask: Image.Image, gpu: int,
) -> Image.Image:
    from diffusers import QwenImageControlNetInpaintPipeline, QwenImageControlNetModel

    dtype = torch.bfloat16
    controlnet = QwenImageControlNetModel.from_pretrained(
        config["controlnet"]["id"],
        revision=preflight["models"]["controlnet"]["resolved_revision"],
        torch_dtype=dtype,
    )
    pipe = QwenImageControlNetInpaintPipeline.from_pretrained(
        config["base_model"]["id"],
        revision=preflight["models"]["base_model"]["resolved_revision"],
        controlnet=controlnet,
        torch_dtype=dtype,
    )
    pipe.enable_model_cpu_offload(gpu_id=gpu)
    settings = config["inference"]
    generator = torch.Generator(device=f"cuda:{gpu}").manual_seed(
        int(record["qwen_seed"])
    )
    return pipe(
        prompt=record["prompt"],
        negative_prompt=record["negative_prompt"],
        control_image=init_image,
        control_mask=mask,
        width=int(settings["width"]),
        height=int(settings["height"]),
        num_inference_steps=int(settings["num_inference_steps"]),
        true_cfg_scale=float(settings["true_cfg_scale"]),
        controlnet_conditioning_scale=float(settings["controlnet_conditioning_scale"]),
        generator=generator,
    ).images[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exactly one GenAI backend smoke image")
    parser.add_argument("--backend", choices=tuple(BACKEND_CONFIGS), required=True)
    parser.add_argument("--image-id", type=int)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available() or args.gpu >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA logical GPU {args.gpu} is not available")
    pilot_ids = [int(x) for x in json.loads(PILOT_IDS_PATH.read_text(encoding="utf-8"))]
    image_id = pilot_ids[0] if args.image_id is None else args.image_id
    if image_id not in pilot_ids:
        raise ValueError(f"Smoke image_id must belong to frozen pilot IDs: {image_id}")

    config_path = BACKEND_CONFIGS[args.backend]
    config = load_yaml(config_path)
    preflight = require_exact_revisions(args.backend)
    record = load_record(image_id)
    initialization, mask, canny, input_hashes = load_inputs(image_id)
    output_root = REPO_ROOT / config["output"]["pilot_root"] / f"smoke_{image_id:06d}"
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite smoke output: {output_root}")
    output_root.mkdir(parents=True)

    torch.cuda.set_device(args.gpu)
    torch.cuda.reset_peak_memory_stats(args.gpu)
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    try:
        if args.backend == "sdxl":
            generated = generate_sdxl(
                config, preflight, record, initialization, mask, canny, args.gpu
            )
        else:
            generated = generate_qwen(
                config, preflight, record, initialization, mask, args.gpu
            )
        runtime = time.monotonic() - started
        image_path = output_root / "generated.png"
        generated.save(image_path)
        report = {
            "format_version": 1,
            "status": "smoke_generated_pending_review",
            "backend": args.backend,
            "image_id": image_id,
            "class_id": record["class_id"],
            "class_name": record["class_name"],
            "started_utc": started_utc,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "wall_time_seconds": runtime,
            "logical_gpu": args.gpu,
            "gpu_name": torch.cuda.get_device_name(args.gpu),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(args.gpu) / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(args.gpu) / 1024**3,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "diffusers": version("diffusers"),
            "transformers": version("transformers"),
            "config": config_path.relative_to(REPO_ROOT).as_posix(),
            "config_sha256": sha256(config_path),
            "models": preflight["models"],
            "git": git_record(),
            "prompt": record["prompt"],
            "negative_prompt": record["negative_prompt"],
            "input_sha256": input_hashes,
            "output": image_path.relative_to(REPO_ROOT).as_posix(),
            "output_sha256": sha256(image_path),
            "annotation_performed": False,
            "copy_paste_dataset_modified": False,
        }
        report_path = output_root / "smoke_report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    except Exception:
        (output_root / "FAILED.txt").write_text(
            f"started_utc={started_utc}\nfailed_utc={datetime.now(timezone.utc).isoformat()}\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
