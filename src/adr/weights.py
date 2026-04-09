#will convert difficulty scores into probabilities (Softmax Function?)

from __future__ import annotations

import math


def normalize_weights(difficulty_scores: dict[str, float]) -> dict[str, float]:
    total = sum(difficulty_scores.values())
    if total <= 0:
        n = len(difficulty_scores)
        if n == 0:
            return {}
        return {k: 1.0 / n for k in difficulty_scores}

    return {k: v / total for k, v in difficulty_scores.items()}


def softmax_weights(difficulty_scores: dict[str, float], temperature: float = 1.0) -> dict[str, float]:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    scaled = {k: v / temperature for k, v in difficulty_scores.items()}
    max_val = max(scaled.values()) if scaled else 0.0

    exp_scores = {k: math.exp(v - max_val) for k, v in scaled.items()}
    total = sum(exp_scores.values())

    if total <= 0:
        n = len(exp_scores)
        if n == 0:
            return {}
        return {k: 1.0 / n for k in exp_scores}

    return {k: v / total for k, v in exp_scores.items()}