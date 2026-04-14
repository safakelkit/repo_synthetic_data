# will compute the difficulty score -> difficulty(c) = α * (1 - S_hard(c) + β * (S_clean(c) - S_hard(c)) + γ * (S_clean(c) - S_easy(c))

from __future__ import annotations

from typing import Any


def _per_class_to_dict(
    per_class_rows: list[dict[str, Any]],
    metric_key: str
) -> dict[str, float]:
    out: dict[str, float] = {}

    for row in per_class_rows:
        class_name = row["class_name"]
        value = row.get(metric_key, None)

        if value is None:
            value = 0.0

        out[class_name] = float(value)

    return out


def compute_difficulty_scores(
    evaluation_results: dict[str, Any],
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
    metric_key: str = "map50_95",
) -> dict[str, float]:
    datasets = evaluation_results["datasets"]

    clean_rows = datasets["insp_det"]["per_class"]
    easy_rows = datasets["insp_mot_det_easy"]["per_class"]
    hard_rows = datasets["insp_mot_det_hard"]["per_class"]

    clean_scores = _per_class_to_dict(clean_rows, metric_key)
    easy_scores = _per_class_to_dict(easy_rows, metric_key)
    hard_scores = _per_class_to_dict(hard_rows, metric_key)

    difficulty_scores: dict[str, float] = {}

    for class_name, s_clean in clean_scores.items():
        s_easy = easy_scores.get(class_name, 0.0)
        s_hard = hard_scores.get(class_name, 0.0)

        difficulty = (
            alpha * (1.0 - s_hard)
            + beta * (s_clean - s_hard)
            + gamma * (s_clean - s_easy)
        )

        difficulty_scores[class_name] = float(max(difficulty, 0.0))

    return difficulty_scores


def rank_difficulties(difficulty_scores: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(difficulty_scores.items(), key=lambda x: x[1], reverse=True)

if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        raise ValueError("Usage: python difficulty.py <evaluation_json_path>")

    input_path = sys.argv[1]

    with open(input_path, "r") as f:
        data = json.load(f)

    scores = compute_difficulty_scores(data)

    print("\n=== Difficulty Scores ===")
    for k, v in scores.items():
        print(f"{k}: {v:.4f}")