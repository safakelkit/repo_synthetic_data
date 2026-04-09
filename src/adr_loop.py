from __future__ import annotations

import json
from pathlib import Path

from train import train_yolo
from evaluate import evaluate_model
from adr.difficulty import compute_difficulty_scores, rank_difficulties
from adr.weights import normalize_weights


TRAIN_CFG_PATH = "configs/train_baseline.yaml"
REAL_DATA_YAML = "configs/data_insp.yaml"

INITIAL_MODEL = "yolo11n.pt"

TOTAL_EPOCHS = 25
EVAL_INTERVAL = 5

RUN_ROOT = Path("runs/adr_loop")
RUN_ROOT.mkdir(parents=True, exist_ok=True)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    current_model = INITIAL_MODEL
    current_data_yaml = REAL_DATA_YAML

    num_cycles = TOTAL_EPOCHS // EVAL_INTERVAL

    for cycle_idx in range(num_cycles):
        cycle_num = cycle_idx + 1
        epoch_end = cycle_num * EVAL_INTERVAL

        print("=" * 80)
        print(f"ADR Cycle {cycle_num}/{num_cycles} | Training up to epoch {epoch_end}")
        print("=" * 80)

        run_name = f"adr_cycle_{cycle_num:02d}"

        # 1. Train for EVAL_INTERVAL epochs
        train_output = train_yolo(
            model_path=current_model,
            data_yaml=current_data_yaml,
            train_cfg_path=TRAIN_CFG_PATH,
            epochs=EVAL_INTERVAL,
            run_name=run_name,
            resume=False,
        )

        # Continue from last checkpoint in next cycle
        current_model = train_output["last_model"]

        # 2. Evaluate automatically
        eval_json_path = RUN_ROOT / f"cycle_{cycle_num:02d}" / "evaluation_results.json"

        evaluation_results = evaluate_model(
            model_path=train_output["best_model"],
            imgsz=640,
            device=0,
            plots=True,
            save_json_path=str(eval_json_path),
        )

        # 3. Compute difficulty scores automatically
        difficulty_scores = compute_difficulty_scores(
            evaluation_results=evaluation_results,
            alpha=0.6,
            beta=0.4,
            metric_key="map50_95",
        )

        ranked = rank_difficulties(difficulty_scores)

        # 4. Convert difficulty -> sampling weights
        sampling_weights = normalize_weights(difficulty_scores)

        # 5. Save ADR state for this cycle
        adr_state = {
            "cycle_num": cycle_num,
            "epoch_end": epoch_end,
            "train_output": train_output,
            "difficulty_scores": difficulty_scores,
            "difficulty_ranking": ranked,
            "sampling_weights": sampling_weights,
        }

        adr_state_path = RUN_ROOT / f"cycle_{cycle_num:02d}" / "adr_state.json"
        save_json(adr_state, adr_state_path)

        print(f"\nCycle {cycle_num} finished.")
        print("Top difficult classes:")
        for class_name, score in ranked[:5]:
            print(f"  {class_name}: {score:.4f}")

        print("\nSampling weights:")
        for class_name, weight in sorted(sampling_weights.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {class_name}: {weight:.4f}")

    print("\nADR loop completed.")


if __name__ == "__main__":
    main()