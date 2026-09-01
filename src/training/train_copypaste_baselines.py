import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from train import load_yaml, repo_path, train_yolo, validate_training_preflight


MATRIX_STATUS = repo_path("runs/evaluation/copy_paste_matrix_status.json")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_matrix_status(state: dict) -> None:
    MATRIX_STATUS.parent.mkdir(parents=True, exist_ok=True)
    MATRIX_STATUS.write_text(json.dumps(state, indent=2), encoding="utf-8")


def evaluation_targets(run_name: str) -> tuple[Path, Path]:
    return (
        repo_path(f"runs/evaluation/{run_name}_results.json"),
        repo_path(f"runs/evaluation/{run_name}"),
    )


def require_evaluation_targets_absent(run_name: str) -> None:
    result_json, result_dir = evaluation_targets(run_name)
    collisions = [path for path in (result_json, result_dir) if path.exists()]
    if collisions:
        raise FileExistsError(f"Evaluation output already exists: {collisions}")


def evaluate_and_plot(best_model: str, run_name: str) -> str:
    result_json, _ = evaluation_targets(run_name)
    subprocess.run(
        [
            sys.executable,
            str(repo_path("src/evaluate_yolo.py")),
            best_model,
            str(result_json),
        ],
        cwd=repo_path("."),
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(repo_path("src/analysis/plot_results.py")),
            str(result_json),
        ],
        cwd=repo_path("."),
        check=True,
    )
    return str(result_json)


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
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="After each training run, evaluate best.pt on clean/easy/hard and render plots",
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
    if args.evaluate:
        for _, run_name in experiments:
            require_evaluation_targets_absent(run_name)

    reports = [
        validate_training_preflight(
            model_path=train_cfg["model"],
            data_yaml=data_yaml,
            train_cfg_path=cfg_path,
            run_name=run_name,
        )
        for data_yaml, run_name in experiments
    ]
    if args.preflight_only:
        print(json.dumps(reports, indent=2))
        return

    state = {
        "status": "running",
        "started_utc": utc_now(),
        "evaluate_after_training": args.evaluate,
        "code_revision": reports[0]["code_revision"],
        "environment": {
            "ultralytics": reports[0]["ultralytics"],
            "torch": reports[0]["torch"],
            "cuda_device": reports[0]["cuda_device"],
            "gpu_name": reports[0]["gpu_name"],
        },
        "experiments": [
            {"run_name": run_name, "data_yaml": data_yaml, "status": "pending"}
            for data_yaml, run_name in experiments
        ],
    }
    write_matrix_status(state)

    try:
        for index, (data_yaml, run_name) in enumerate(experiments):
            print(f"\nStarting experiment: {run_name}")
            state["experiments"][index]["status"] = "training"
            state["experiments"][index]["started_utc"] = utc_now()
            write_matrix_status(state)

            output = train_yolo(
                model_path=train_cfg["model"],
                data_yaml=data_yaml,
                train_cfg_path=cfg_path,
                epochs=train_cfg["epochs"],
                run_name=run_name,
                resume=False,
            )
            state["experiments"][index].update(output)
            state["experiments"][index]["status"] = "trained"
            write_matrix_status(state)

            if args.evaluate:
                state["experiments"][index]["status"] = "evaluating"
                write_matrix_status(state)
                state["experiments"][index]["evaluation_json"] = evaluate_and_plot(
                    output["best_model"], run_name
                )

            state["experiments"][index]["status"] = "complete"
            state["experiments"][index]["completed_utc"] = utc_now()
            write_matrix_status(state)
            print("Finished:", run_name)
            print(output)
    except Exception as error:
        state["status"] = "failed"
        state["failed_utc"] = utc_now()
        state["error"] = f"{type(error).__name__}: {error}"
        write_matrix_status(state)
        raise

    state["status"] = "complete"
    state["completed_utc"] = utc_now()
    write_matrix_status(state)


if __name__ == "__main__":
    main()
