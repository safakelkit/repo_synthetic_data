"""Train the paired clean and mixed-degradation SDXL quantity matrix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from train import load_yaml, repo_path, train_yolo, validate_training_preflight


EVALUATION_ROOT = repo_path("runs/evaluation/sdxl")
MATRIX_STATUS = EVALUATION_ROOT / "matrix_status.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_matrix_status(state: dict, status_path: Path) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def evaluation_targets(run_name: str) -> tuple[Path, Path]:
    variant = "clean" if run_name.startswith("SDXL-C") else "mixed_degradation"
    output_root = EVALUATION_ROOT / variant
    return (
        output_root / f"{run_name}_results.json",
        output_root / run_name,
    )


def require_evaluation_targets_absent(run_name: str) -> None:
    collisions = [path for path in evaluation_targets(run_name) if path.exists()]
    if collisions:
        raise FileExistsError(f"Evaluation output already exists: {collisions}")


def evaluate_and_plot(best_model: str, run_name: str) -> str:
    result_json, _ = evaluation_targets(run_name)
    subprocess.run(
        [sys.executable, str(repo_path("src/evaluate_yolo.py")), best_model, str(result_json)],
        cwd=repo_path("."),
        check=True,
    )
    subprocess.run(
        [sys.executable, str(repo_path("src/analysis/plot_results.py")), str(result_json)],
        cwd=repo_path("."),
        check=True,
    )
    return str(result_json)


def require_released_synthetic_manifest(data_yaml: str) -> dict:
    data_config = load_yaml(data_yaml)
    dataset_root = repo_path(data_yaml).parent
    entries = [Path(entry) for entry in data_config["train"] if str(entry).endswith(".txt")]
    if len(entries) != 1:
        raise ValueError(f"Expected exactly one synthetic image list in {data_yaml}")
    image_list = (dataset_root / entries[0]).resolve()
    manifest_path = image_list.parent / "manifest.json"
    if not image_list.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Synthetic release files are incomplete for {data_yaml}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("training_use_forbidden", True):
        raise RuntimeError(f"Synthetic dataset is not approved for training: {manifest_path}")
    if manifest.get("status") != "approved_for_training":
        raise RuntimeError(f"Unexpected synthetic release status in {manifest_path}")
    blocked = sum(bool(row.get("training_use_forbidden", True)) for row in manifest["records"])
    if blocked:
        raise RuntimeError(f"Synthetic manifest still blocks {blocked} records: {manifest_path}")
    pending = sum(
        row.get("post_degradation_visibility_qc") not in (None, "passed")
        for row in manifest["records"]
    )
    if pending:
        raise RuntimeError(f"Synthetic manifest has {pending} pending visibility checks")
    synthetic_lines = [
        line.strip() for line in image_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_total = int(data_config["expected_train_images"])
    expected_synthetic = expected_total - 2215
    if len(synthetic_lines) != expected_synthetic or len(set(synthetic_lines)) != len(synthetic_lines):
        raise RuntimeError(
            f"Synthetic list count/uniqueness mismatch in {image_list}: "
            f"expected {expected_synthetic}, got {len(synthetic_lines)}"
        )
    return {
        "data_yaml": data_yaml,
        "manifest": str(manifest_path),
        "manifest_status": manifest["status"],
        "synthetic_images": len(synthetic_lines),
        "expected_train_images": expected_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=(
            "clean-512", "clean-1024", "clean-1536", "clean-2048",
            "mixed-512", "mixed-1024", "mixed-1536", "mixed-2048",
            "clean-ablation", "clean-all", "mixed-all", "all",
        ),
        required=True,
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--dataset-preflight-only", action="store_true",
        help="Validate released manifests and list counts without requiring CUDA.",
    )
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    train_cfg_path = "configs/train_baseline.yaml"
    train_cfg = load_yaml(train_cfg_path)
    registry = {
        "clean-512": ("configs/data_insp_sdxl_clean_512.yaml", "SDXL-C0512_yolo11s_seed0"),
        "clean-1024": ("configs/data_insp_sdxl_clean_1024.yaml", "SDXL-C1024_yolo11s_seed0"),
        "clean-1536": ("configs/data_insp_sdxl_clean_1536.yaml", "SDXL-C1536_yolo11s_seed0"),
        "clean-2048": ("configs/data_insp_sdxl_clean_2048.yaml", "SDXL-C2048_yolo11s_seed0"),
        "mixed-512": ("configs/data_insp_sdxl_mixed_512.yaml", "SDXL-M0512_yolo11s_seed0"),
        "mixed-1024": ("configs/data_insp_sdxl_mixed_1024.yaml", "SDXL-M1024_yolo11s_seed0"),
        "mixed-1536": ("configs/data_insp_sdxl_mixed_1536.yaml", "SDXL-M1536_yolo11s_seed0"),
        "mixed-2048": ("configs/data_insp_sdxl_mixed_2048.yaml", "SDXL-M2048_yolo11s_seed0"),
    }
    if args.experiment == "all":
        experiments = list(registry.values())
    elif args.experiment == "clean-ablation":
        experiments = [registry["clean-512"], registry["clean-2048"]]
    elif args.experiment == "clean-all":
        experiments = [registry[f"clean-{quantity}"] for quantity in (512, 1024, 1536, 2048)]
    elif args.experiment == "mixed-all":
        experiments = [registry[f"mixed-{quantity}"] for quantity in (512, 1024, 1536, 2048)]
    else:
        experiments = [registry[args.experiment]]

    dataset_reports = [
        require_released_synthetic_manifest(data_yaml) for data_yaml, _ in experiments
    ]
    if args.evaluate:
        for _, run_name in experiments:
            require_evaluation_targets_absent(run_name)
    if args.dataset_preflight_only:
        print(json.dumps(dataset_reports, indent=2))
        return
    status_path = (
        EVALUATION_ROOT / "clean_ablation_status.json"
        if args.experiment == "clean-ablation"
        else MATRIX_STATUS
    )
    reports = [
        validate_training_preflight(
            model_path=train_cfg["model"], data_yaml=data_yaml,
            train_cfg_path=train_cfg_path, run_name=run_name,
        )
        for data_yaml, run_name in experiments
    ]
    if args.preflight_only:
        print(json.dumps(reports, indent=2))
        return

    state = {
        "status": "running", "started_utc": utc_now(),
        "evaluate_after_training": args.evaluate,
        "code_revision": reports[0]["code_revision"],
        "environment": {
            "ultralytics": reports[0]["ultralytics"], "torch": reports[0]["torch"],
            "cuda_device": reports[0]["cuda_device"], "gpu_name": reports[0]["gpu_name"],
        },
        "experiments": [
            {"run_name": run_name, "data_yaml": data_yaml, "status": "pending"}
            for data_yaml, run_name in experiments
        ],
    }
    write_matrix_status(state, status_path)
    try:
        for index, (data_yaml, run_name) in enumerate(experiments):
            state["experiments"][index].update(status="training", started_utc=utc_now())
            write_matrix_status(state, status_path)
            output = train_yolo(
                model_path=train_cfg["model"], data_yaml=data_yaml,
                train_cfg_path=train_cfg_path, epochs=train_cfg["epochs"],
                run_name=run_name, resume=False,
            )
            state["experiments"][index].update(output, status="trained")
            write_matrix_status(state, status_path)
            if args.evaluate:
                state["experiments"][index]["status"] = "evaluating"
                write_matrix_status(state, status_path)
                state["experiments"][index]["evaluation_json"] = evaluate_and_plot(
                    output["best_model"], run_name
                )
            state["experiments"][index].update(status="complete", completed_utc=utc_now())
            write_matrix_status(state, status_path)
    except BaseException as error:
        state["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        state["failed_utc"] = utc_now()
        state["error"] = f"{type(error).__name__}: {error}"
        write_matrix_status(state, status_path)
        raise
    state.update(status="complete", completed_utc=utc_now())
    write_matrix_status(state, status_path)


if __name__ == "__main__":
    main()
