import argparse
import json
from pathlib import Path

from train import load_yaml, repo_path, train_yolo, validate_training_preflight


def require_synthetic_manifest(data_yaml: str) -> None:
    data_config = load_yaml(data_yaml)
    dataset_root = (
        repo_path(data_config["path"])
        if data_config.get("path")
        else repo_path(data_yaml).parent
    )
    manifest_entries = [entry for entry in data_config["train"] if str(entry).endswith(".txt")]
    if len(manifest_entries) != 1:
        raise ValueError(f"Expected exactly one synthetic manifest in {data_yaml}")
    manifest_path = dataset_root / Path(manifest_entries[0])
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Synthetic manifest does not exist: {manifest_path}. "
            "Generate and approve the cut-paste release before training."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train frozen copy-paste baselines")
    parser.add_argument(
        "--experiment",
        choices=("512", "1024", "1536", "2048", "all"),
        required=True,
        help="Run one quantity, or all quantities sequentially",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate selected runs without starting training",
    )
    args = parser.parse_args()

    cfg_path = "configs/train_baseline.yaml"
    train_cfg = load_yaml(cfg_path)

    experiment_registry = {
        "512": ("configs/data_insp_cp_512.yaml", "CP-B0512_yolo11s_seed0"),
        "1024": ("configs/data_insp_cp_1024.yaml", "CP-B1024_yolo11s_seed0"),
        "1536": ("configs/data_insp_cp_1536.yaml", "CP-B1536_yolo11s_seed0"),
        "2048": ("configs/data_insp_cp_2048.yaml", "CP-B2048_yolo11s_seed0"),
    }
    experiments = (
        list(experiment_registry.values())
        if args.experiment == "all"
        else [experiment_registry[args.experiment]]
    )

    for data_yaml, _ in experiments:
        require_synthetic_manifest(data_yaml)

    if args.preflight_only:
        reports = [
            validate_training_preflight(
                model_path=train_cfg["model"],
                data_yaml=data_yaml,
                train_cfg_path=cfg_path,
                run_name=run_name,
            )
            for data_yaml, run_name in experiments
        ]
        print(json.dumps(reports, indent=2))
        return

    for data_yaml, run_name in experiments:
        print(f"\nStarting experiment: {run_name}")

        output = train_yolo(
            model_path=train_cfg["model"],
            data_yaml=data_yaml,
            train_cfg_path=cfg_path,
            epochs=train_cfg["epochs"],
            run_name=run_name,
            resume=False,
        )

        print("Finished:", run_name)
        print(output)


if __name__ == "__main__":
    main()
