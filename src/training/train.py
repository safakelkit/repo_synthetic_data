from __future__ import annotations

import argparse
import glob
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS, check_det_dataset, img2label_paths

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_TRAIN_KEYS = (
    "patience", "optimizer", "lr0", "lrf", "momentum", "weight_decay",
    "warmup_epochs", "warmup_momentum", "warmup_bias_lr", "box", "cls",
    "dfl", "close_mosaic", "amp", "cache", "rect", "multi_scale",
    "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
    "perspective", "flipud", "fliplr", "bgr", "mosaic", "mixup", "cutmix",
    "copy_paste",
)


def repo_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_yaml(path: str) -> dict[str, Any]:
    with open(repo_path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def git_state() -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return revision, bool(status)


def resolve_ultralytics_image_paths(img_path: str | list[str]) -> list[Path]:
    """Resolve training inputs exactly as the pinned Ultralytics release does."""
    files: list[str] = []
    for raw_path in img_path if isinstance(img_path, list) else [img_path]:
        path = Path(raw_path)
        if path.is_dir():
            files.extend(glob.glob(str(path / "**" / "*.*"), recursive=True))
        elif path.is_file():
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            parent = str(path.parent) + os.sep
            files.extend(
                line.replace("./", parent) if line.startswith("./") else line
                for line in lines
                if line.strip()
            )
        else:
            raise FileNotFoundError(f"Training input does not exist: {path}")
    return sorted(
        Path(file.replace("/", os.sep))
        for file in files
        if file.rpartition(".")[-1].lower() in IMG_FORMATS
    )


def validate_training_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    """Fail before training if image lists resolve incompletely or incorrectly."""
    image_paths = resolve_ultralytics_image_paths(dataset["train"])
    expected = int(dataset["expected_train_images"])
    if len(image_paths) != expected:
        raise ValueError(
            f"Training image count mismatch: expected {expected}, resolved {len(image_paths)}"
        )

    duplicate_paths = len(image_paths) - len({str(path.resolve()) for path in image_paths})
    if duplicate_paths:
        raise ValueError(f"Training inputs contain {duplicate_paths} duplicate image paths")

    missing_images = [path for path in image_paths if not path.is_file()]
    if missing_images:
        raise FileNotFoundError(
            f"Training inputs contain {len(missing_images)} missing images; "
            f"first missing image: {missing_images[0]}"
        )

    label_paths = [Path(path) for path in img2label_paths([str(path) for path in image_paths])]
    missing_labels = [path for path in label_paths if not path.is_file()]
    if missing_labels:
        raise FileNotFoundError(
            f"Training inputs contain {len(missing_labels)} missing labels; "
            f"first missing label: {missing_labels[0]}"
        )

    return {
        "expected_images": expected,
        "resolved_images": len(image_paths),
        "resolved_labels": len(label_paths),
        "duplicate_image_paths": duplicate_paths,
    }


def validate_training_preflight(
    model_path: str,
    data_yaml: str,
    train_cfg_path: str,
    run_name: str,
) -> dict[str, Any]:
    train_cfg = load_yaml(train_cfg_path)
    expected_version = str(train_cfg["ultralytics_version"])
    installed_version = importlib.metadata.version("ultralytics")
    if installed_version != expected_version:
        raise RuntimeError(
            f"Ultralytics version mismatch: expected {expected_version}, "
            f"found {installed_version}"
        )

    resolved_model = repo_path(model_path)
    if not resolved_model.is_file():
        raise FileNotFoundError(f"Model weights not found: {resolved_model}")

    resolved_data = repo_path(data_yaml)
    if not resolved_data.is_file():
        raise FileNotFoundError(f"Dataset config not found: {resolved_data}")
    dataset = check_det_dataset(str(resolved_data), autodownload=False)
    dataset_integrity = validate_training_dataset(dataset)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this terminal session")
    device = int(train_cfg["device"])
    if device < 0 or device >= torch.cuda.device_count():
        raise RuntimeError(
            f"Configured CUDA device {device} is unavailable; "
            f"visible device count is {torch.cuda.device_count()}"
        )

    project_dir = repo_path(train_cfg["project"])
    run_dir = project_dir / run_name
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")

    revision, dirty = git_state()
    if dirty:
        raise RuntimeError("Git working tree is dirty; commit the experiment code before training")

    return {
        "status": "ready",
        "code_revision": revision,
        "ultralytics": installed_version,
        "torch": torch.__version__,
        "cuda_device": device,
        "gpu_name": torch.cuda.get_device_name(device),
        "model": str(resolved_model),
        "data_yaml": str(resolved_data),
        "train": dataset["train"],
        "validation": dataset["val"],
        "dataset_integrity": dataset_integrity,
        "run_dir": str(run_dir),
    }


def train_yolo(
    model_path: str,
    data_yaml: str,
    train_cfg_path: str,
    epochs: int,
    run_name: str,
    resume: bool = False,
) -> dict[str, str]:
    train_cfg = load_yaml(train_cfg_path)
    missing_keys = [key for key in FROZEN_TRAIN_KEYS if key not in train_cfg]
    if missing_keys:
        raise ValueError(f"Training config is missing frozen values: {missing_keys}")
    frozen_args = {key: train_cfg[key] for key in FROZEN_TRAIN_KEYS}
    project_dir = repo_path(train_cfg["project"])
    run_dir = project_dir / run_name

    if run_dir.exists() and not resume:
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. "
            "Choose a new run name or explicitly resume the existing run."
        )

    if not resume:
        preflight = validate_training_preflight(
            model_path=model_path,
            data_yaml=data_yaml,
            train_cfg_path=train_cfg_path,
            run_name=run_name,
        )
        print(json.dumps(preflight, indent=2))

    model = YOLO(str(repo_path(model_path)))

    model.train(
        data=str(repo_path(data_yaml)),
        imgsz=train_cfg["imgsz"],
        epochs=epochs,
        batch=train_cfg["batch"],
        device=train_cfg["device"],
        workers=train_cfg["workers"],
        # Ultralytics may otherwise prepend its global runs_dir to this path.
        project=str(project_dir),
        name=run_name,
        exist_ok=resume,
        save_period=train_cfg.get("save_period", -1),
        resume=resume,
        seed=train_cfg.get("seed", 0),
        deterministic=train_cfg.get("deterministic", True),

        val=True,
        plots=True,
        **frozen_args,
    )

    if model.trainer is None:
        raise RuntimeError("Ultralytics training finished without a trainer state")
    save_dir = Path(model.trainer.save_dir)
    weights_dir = save_dir / "weights"

    return {
        "save_dir": str(save_dir),
        "best_model": str(weights_dir / "best.pt"),
        "last_model": str(weights_dir / "last.pt"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the frozen E000 real-only baseline")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate environment, data, GPU, Git state, and output path without training",
    )
    args = parser.parse_args()

    cfg_path = "configs/train_baseline.yaml"
    data_yaml = "configs/data_insp.yaml"
    train_cfg = load_yaml(cfg_path)

    if args.preflight_only:
        report = validate_training_preflight(
            model_path=train_cfg["model"],
            data_yaml=data_yaml,
            train_cfg_path=cfg_path,
            run_name=train_cfg["name"],
        )
        print(json.dumps(report, indent=2))
        return

    output = train_yolo(
        model_path=train_cfg["model"],
        data_yaml=data_yaml,
        train_cfg_path=cfg_path,
        epochs=train_cfg["epochs"],
        run_name=train_cfg["name"],
        resume=False,
    )

    print("Training finished.")
    print(output)


if __name__ == "__main__":
    main()
