from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

from .difficulty import compute_difficulty_scores, rank_difficulties


def normalize_weights(difficulty_scores: dict[str, float]) -> dict[str, float]:
    """
    Linear normalization:
        w_i = d_i / sum(d)

    Keeps proportional differences, but is less selective than softmax.
    """
    total = sum(difficulty_scores.values())

    if total <= 0:
        n = len(difficulty_scores)
        if n == 0:
            return {}
        return {k: 1.0 / n for k in difficulty_scores}

    return {k: v / total for k, v in difficulty_scores.items()}


def softmax_weights(
    difficulty_scores: dict[str, float],
    temperature: float = 0.7,
) -> dict[str, float]:
    """
    Softmax normalization:
        w_i = exp(d_i / T) / sum_j exp(d_j / T)

    Lower temperature => more focus on hardest classes
    Higher temperature => more uniform distribution
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    if not difficulty_scores:
        return {}

    scaled = {k: v / temperature for k, v in difficulty_scores.items()}
    max_val = max(scaled.values())

    exp_scores = {k: math.exp(v - max_val) for k, v in scaled.items()}
    total = sum(exp_scores.values())

    if total <= 0:
        n = len(exp_scores)
        return {k: 1.0 / n for k in exp_scores}

    return {k: v / total for k, v in exp_scores.items()}


def load_evaluation_results(json_path: str | Path) -> dict[str, Any]:
    """
    Load evaluation JSON file.
    """
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"Evaluation JSON not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_weights_json(weights: dict[str, float], output_path: str | Path) -> None:
    """
    Save sampling weights to JSON.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2, ensure_ascii=False)


def build_sampling_weights(
    evaluation_results: dict[str, Any],
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
    metric_key: str = "map50_95",
    weighting_method: str = "softmax",
    temperature: float = 0.7,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    End-to-end pipeline:
    evaluation_results -> difficulty scores -> sampling weights

    Returns:
        (difficulty_scores, sampling_weights)
    """
    difficulty_scores = compute_difficulty_scores(
        evaluation_results=evaluation_results,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        metric_key=metric_key,
    )

    if weighting_method == "softmax":
        weights = softmax_weights(difficulty_scores, temperature=temperature)
    elif weighting_method == "normalize":
        weights = normalize_weights(difficulty_scores)
    else:
        raise ValueError(
            "weighting_method must be either 'softmax' or 'normalize'"
        )

    return difficulty_scores, weights


def main() -> None:
    if len(sys.argv) < 2:
        raise ValueError(
            "Usage: python weights.py <evaluation_json_path> [output_json_path]"
        )

    input_json = Path(sys.argv[1])

    if len(sys.argv) >= 3:
        output_json = Path(sys.argv[2])
    else:
        output_json = input_json.parent / "sampling_weights.json"

    evaluation_results = load_evaluation_results(input_json)

    difficulty_scores, sampling_weights = build_sampling_weights(
        evaluation_results=evaluation_results,
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        metric_key="map50_95",
        weighting_method="softmax",
        temperature=0.7,
    )

    print("\n=== Difficulty Ranking ===")
    for class_name, score in rank_difficulties(difficulty_scores):
        print(f"{class_name:15s} -> {score:.6f}")

    print("\n=== Sampling Weights ===")
    for class_name, weight in sorted(
        sampling_weights.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"{class_name:15s} -> {weight:.6f}")

    save_weights_json(sampling_weights, output_json)
    print(f"\nSampling weights saved to: {output_json}")


if __name__ == "__main__":
    main()