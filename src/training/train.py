from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

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
    cfg_path = "configs/train_baseline.yaml"
    data_yaml = "configs/data_insp.yaml"
    train_cfg = load_yaml(cfg_path)

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
