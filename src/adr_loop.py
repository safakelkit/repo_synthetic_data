from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from train import train_yolo, load_yaml
from evaluate import evaluate_model
from adr.difficulty import compute_difficulty_scores, rank_difficulties
from adr.weights import build_sampling_weights, save_weights_json


def save_json(data: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run_adr_loop(
    train_cfg_path: str = "configs/train_baseline.yaml",
    data_yaml: str = "configs/data_insp.yaml",
    total_epochs: int = 2,
    cycle_epochs: int = 1,
    base_run_name: str = "adr_yolo11n",
    metric_key: str = "map50_95",
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
    weighting_method: str = "softmax",
    temperature: float = 0.7,
) -> dict[str, Any]:
    """
    Main ADR loop:

    cycle:
        1) train for cycle_epochs
        2) evaluate last.pt
        3) compute difficulty scores
        4) compute sampling weights
        5) save ADR artifacts
        6) continue next cycle from previous last.pt

    Notes:
    - Uses last.pt for continuity
    - Starts each cycle as a fresh train call with updated model_path
    - Does NOT yet modify dataset or generate synthetic data
    """

    if total_epochs <= 0:
        raise ValueError("total_epochs must be > 0")

    if cycle_epochs <= 0:
        raise ValueError("cycle_epochs must be > 0")

    train_cfg = load_yaml(train_cfg_path)
    imgsz = train_cfg["imgsz"]
    device = train_cfg["device"]

    root_output_dir = Path("runs/adr") / base_run_name
    root_output_dir.mkdir(parents=True, exist_ok=True)

    current_model_path = train_cfg["model"]
    completed_epochs = 0
    cycle_index = 0

    history: list[dict[str, Any]] = []

    while completed_epochs < total_epochs:
        cycle_index += 1
        epochs_this_cycle = min(cycle_epochs, total_epochs - completed_epochs)

        cycle_name = f"{base_run_name}_cycle_{cycle_index:02d}"
        cycle_dir = root_output_dir / cycle_name
        cycle_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 70)
        print(f"ADR CYCLE {cycle_index}")
        print(f"Start model: {current_model_path}")
        print(f"Train epochs this cycle: {epochs_this_cycle}")
        print("=" * 70)

        train_output = train_yolo(
            model_path=current_model_path,
            data_yaml=data_yaml,
            train_cfg_path=train_cfg_path,
            epochs=epochs_this_cycle,
            run_name=cycle_name,
            resume=False,
        )

        last_model_path = train_output["last_model"]
        best_model_path = train_output["best_model"]

        print(f"\nTraining finished for cycle {cycle_index}.")
        print(f"Last model: {last_model_path}")
        print(f"Best model: {best_model_path}")

        evaluation_json_path = cycle_dir / "evaluation_results.json"
        evaluation_results = evaluate_model(
            model_path=last_model_path,
            imgsz=imgsz,
            device=device,
            plots=False,
            save_json_path=str(evaluation_json_path),
        )

        print(f"Evaluation saved to: {evaluation_json_path}")

        difficulty_scores = compute_difficulty_scores(
            evaluation_results=evaluation_results,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            metric_key=metric_key,
        )

        difficulty_json_path = cycle_dir / "difficulty_scores.json"
        save_json(difficulty_scores, difficulty_json_path)

        _, sampling_weights = build_sampling_weights(
            evaluation_results=evaluation_results,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            metric_key=metric_key,
            weighting_method=weighting_method,
            temperature=temperature,
        )

        weights_json_path = cycle_dir / "sampling_weights.json"
        save_weights_json(sampling_weights, weights_json_path)

        ranked_difficulties = rank_difficulties(difficulty_scores)

        print("\nTop 5 hardest classes:")
        for class_name, score in ranked_difficulties[:5]:
            print(f"  {class_name:15s} -> {score:.6f}")

        cycle_summary = {
            "cycle_index": cycle_index,
            "epochs_this_cycle": epochs_this_cycle,
            "completed_epochs": completed_epochs + epochs_this_cycle,
            "train_output": train_output,
            "evaluation_json": str(evaluation_json_path),
            "difficulty_json": str(difficulty_json_path),
            "weights_json": str(weights_json_path),
            "top_5_hardest": [
                {"class_name": class_name, "difficulty": score}
                for class_name, score in ranked_difficulties[:5]
            ],
            "metric_key": metric_key,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "weighting_method": weighting_method,
            "temperature": temperature,
        }

        history.append(cycle_summary)

        history_json_path = root_output_dir / "adr_history.json"
        save_json({"cycles": history}, history_json_path)

        print(f"\nCycle {cycle_index} artifacts saved in: {cycle_dir}")
        print(f"ADR history updated: {history_json_path}")

        current_model_path = last_model_path
        completed_epochs += epochs_this_cycle

        # Placeholder for future Step C:
        # Here we will later:
        # 1) read sampling_weights
        # 2) decide which classes/conditions to generate
        # 3) add new synthetic images
        # 4) optionally update data YAML or train folder

    final_summary = {
        "base_run_name": base_run_name,
        "total_epochs": total_epochs,
        "cycle_epochs": cycle_epochs,
        "num_cycles": cycle_index,
        "final_model_path": current_model_path,
        "history_file": str(root_output_dir / "adr_history.json"),
        "root_output_dir": str(root_output_dir),
    }

    final_summary_path = root_output_dir / "final_summary.json"
    save_json(final_summary, final_summary_path)

    print("\n" + "=" * 70)
    print("ADR LOOP FINISHED")
    print("=" * 70)
    print(json.dumps(final_summary, indent=2))

    return final_summary


def main() -> None:
    run_adr_loop(
        train_cfg_path="configs/train_baseline.yaml",
        data_yaml="configs/data_insp.yaml",
        total_epochs=2,
        cycle_epochs=1,
        base_run_name="adr_yolo11n_test",
        metric_key="map50_95",
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        weighting_method="softmax",
        temperature=0.7,
    )


if __name__ == "__main__":
    main()