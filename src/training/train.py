from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
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

    model = YOLO(model_path)

    results = model.train(
        data=data_yaml,
        imgsz=train_cfg["imgsz"],
        epochs=epochs,
        batch=train_cfg["batch"],
        device=train_cfg["device"],
        workers=train_cfg["workers"],
        project=train_cfg["project"],
        name=run_name,
        save_period=train_cfg.get("save_period", -1),
        resume=resume,

        # validation / output
        val=True,
        plots=True,

        # augmentation settings from YAML
        bgr=train_cfg.get("bgr", 0.0),
        fliplr=train_cfg.get("fliplr", 0.5),
        flipud=train_cfg.get("flipud", 0.0),
        hsv_v=train_cfg.get("hsv_v", 0.4),
        hsv_h=train_cfg.get("hsv_h", 0.015),
    )

    save_dir = Path(results.save_dir)
    weights_dir = save_dir / "weights"

    return {
        "save_dir": str(save_dir),
        "best_model": str(weights_dir / "best.pt"),
        "last_model": str(weights_dir / "last.pt"),
    }


def main() -> None:
    cfg_path = "configs/train_baseline.yaml"
    data_yaml = "configs/data_insp_cp_500.yaml"
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