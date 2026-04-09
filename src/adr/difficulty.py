#will compute the difficulty score -> difficulty(c) = ⍺ * (1 - S_hard(c)) + β * (S_clean(c) - S_hard(c))

from __future__ import annotations

from typing import Any


def _per_class_to_dict(per_class_rows: list[dict[str, Any]], metric_key: str) -> dict[str, float]:
    out = {}
    for row in per_class_rows:
        class_name = row["class_name"]
        value = row.get(metric_key, None)
        if value is None:
            continue
        out[class_name] = float(value)
    return out


def compute_difficulty_scores(
    evaluation_results: dict[str, Any],
    alpha: float = 0.6,
    beta: float = 0.4,
    metric_key: str = "map50_95",
) -> dict[str, float]:
    datasets = evaluation_results["datasets"]

    clean_rows = datasets["insp_det"]["per_class"]
    hard_rows = datasets["insp_mot_det_hard"]["per_class"]

    clean_scores = _per_class_to_dict(clean_rows, metric_key)
    hard_scores = _per_class_to_dict(hard_rows, metric_key)

    difficulty_scores: dict[str, float] = {}

    for class_name, ap_clean in clean_scores.items():
        ap_hard = hard_scores.get(class_name, 0.0)

        difficulty = alpha * (1.0 - ap_clean) + beta * (ap_clean - ap_hard)
        difficulty_scores[class_name] = float(max(difficulty, 0.0))

    return difficulty_scores


def rank_difficulties(difficulty_scores: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(difficulty_scores.items(), key=lambda x: x[1], reverse=True)