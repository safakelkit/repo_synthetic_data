from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DATASET_DISPLAY_NAMES = {
    "insp_det": "INSP-DET",
    "insp_mot_det_easy": "INSP-MOT-DET Easy",
    "insp_mot_det_hard": "INSP-MOT-DET Hard",
}


def load_results(json_path: Path) -> dict[str, Any]:
    if not json_path.exists():
        raise FileNotFoundError(f"Evaluation JSON not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def safe_value(x: Any) -> float:
    if x is None:
        return np.nan
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def get_dataset_order(results: dict[str, Any]) -> list[str]:
    datasets = results.get("datasets", {})
    preferred_order = ["insp_det", "insp_mot_det_easy", "insp_mot_det_hard"]
    existing = [d for d in preferred_order if d in datasets]
    remaining = [d for d in datasets.keys() if d not in existing]
    return existing + remaining


def extract_overall_table(results: dict[str, Any]) -> dict[str, dict[str, float]]:
    datasets = results["datasets"]
    dataset_order = get_dataset_order(results)

    table: dict[str, dict[str, float]] = {}
    for dataset_name in dataset_order:
        overall = datasets[dataset_name]["overall"]
        table[dataset_name] = {
            "precision": safe_value(overall.get("precision")),
            "recall": safe_value(overall.get("recall")),
            "map50": safe_value(overall.get("map50")),
            "map50_95": safe_value(overall.get("map50_95")),
        }
    return table


def extract_per_class_table(results: dict[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    """
    Returns:
        {
            dataset_name: {
                class_name: {
                    "ap50": ...,
                    "map50_95": ...,
                    "precision": ...,
                    "recall": ...
                }
            }
        }
    """
    datasets = results["datasets"]
    dataset_order = get_dataset_order(results)

    out: dict[str, dict[str, dict[str, float]]] = {}

    for dataset_name in dataset_order:
        per_class = datasets[dataset_name]["per_class"]
        class_dict: dict[str, dict[str, float]] = {}

        for row in per_class:
            class_name = str(row["class_name"])
            class_dict[class_name] = {
                "ap50": safe_value(row.get("ap50")),
                "map50_95": safe_value(row.get("map50_95")),
                "precision": safe_value(row.get("precision")),
                "recall": safe_value(row.get("recall")),
            }

        out[dataset_name] = class_dict

    return out


def get_all_class_names(per_class_table: dict[str, dict[str, dict[str, float]]]) -> list[str]:
    all_names = set()
    for dataset_dict in per_class_table.values():
        all_names.update(dataset_dict.keys())
    return sorted(all_names)


def plot_overall_metrics(results: dict[str, Any], output_dir: Path) -> None:
    overall_table = extract_overall_table(results)
    dataset_order = list(overall_table.keys())
    dataset_labels = [DATASET_DISPLAY_NAMES.get(d, d) for d in dataset_order]

    metric_names = ["precision", "recall", "map50", "map50_95"]
    metric_labels = ["Precision", "Recall", "mAP50", "mAP50-95"]

    x = np.arange(len(dataset_labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (metric_name, metric_label) in enumerate(zip(metric_names, metric_labels)):
        values = [overall_table[d][metric_name] for d in dataset_order]
        ax.bar(x + (i - 1.5) * width, values, width, label=metric_label)

    ax.set_title("Overall Performance Comparison Across Datasets")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels, rotation=15)
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "overall_metrics_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_metric(results: dict[str, Any], output_dir: Path, metric_key: str, filename: str, title: str) -> None:
    per_class_table = extract_per_class_table(results)
    dataset_order = get_dataset_order(results)
    dataset_labels = [DATASET_DISPLAY_NAMES.get(d, d) for d in dataset_order]
    class_names = get_all_class_names(per_class_table)

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(18, 7))

    for i, dataset_name in enumerate(dataset_order):
        values = []
        for class_name in class_names:
            value = per_class_table.get(dataset_name, {}).get(class_name, {}).get(metric_key, np.nan)
            values.append(value)

        ax.bar(x + (i - 1) * width, values, width, label=dataset_labels[i])

    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel(metric_key)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_domain_drops(results: dict[str, Any], metric_key: str = "map50_95") -> tuple[list[str], np.ndarray, np.ndarray]:
    per_class_table = extract_per_class_table(results)

    if "insp_det" not in per_class_table:
        raise ValueError("insp_det not found in evaluation results. Cannot compute domain gap.")

    clean = per_class_table["insp_det"]
    easy = per_class_table.get("insp_mot_det_easy", {})
    hard = per_class_table.get("insp_mot_det_hard", {})

    class_names = sorted(clean.keys())

    drop_easy = []
    drop_hard = []

    for class_name in class_names:
        clean_val = clean[class_name].get(metric_key, np.nan)
        easy_val = easy.get(class_name, {}).get(metric_key, np.nan)
        hard_val = hard.get(class_name, {}).get(metric_key, np.nan)

        clean_val = safe_value(clean_val)
        easy_val = safe_value(easy_val)
        hard_val = safe_value(hard_val)

        drop_easy.append(clean_val - easy_val if not np.isnan(clean_val) and not np.isnan(easy_val) else np.nan)
        drop_hard.append(clean_val - hard_val if not np.isnan(clean_val) and not np.isnan(hard_val) else np.nan)

    return class_names, np.array(drop_easy), np.array(drop_hard)


def plot_domain_gap(results: dict[str, Any], output_dir: Path, metric_key: str = "map50_95") -> None:
    class_names, drop_easy, drop_hard = compute_domain_drops(results, metric_key=metric_key)

    x = np.arange(len(class_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(18, 7))
    ax.bar(x - width / 2, drop_easy, width, label="Clean → Easy Drop")
    ax.bar(x + width / 2, drop_hard, width, label="Clean → Hard Drop")

    ax.set_title(f"Per-Class Domain Gap ({metric_key})")
    ax.set_xlabel("Class")
    ax.set_ylabel("Performance Drop")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / f"domain_gap_{metric_key}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_hard_domain_drop_ranking(results: dict[str, Any], output_dir: Path, metric_key: str = "map50_95") -> None:
    class_names, _, drop_hard = compute_domain_drops(results, metric_key=metric_key)

    valid_items = [
        (class_name, drop)
        for class_name, drop in zip(class_names, drop_hard)
        if not np.isnan(drop)
    ]

    valid_items.sort(key=lambda x: x[1], reverse=True)

    ranked_names = [item[0] for item in valid_items]
    ranked_scores = [item[1] for item in valid_items]

    fig, ax = plt.subplots(figsize=(12, 8))
    y = np.arange(len(ranked_names))

    ax.barh(y, ranked_scores)
    ax.set_yticks(y)
    ax.set_yticklabels(ranked_names)
    ax.invert_yaxis()
    ax.set_title(f"Clean → Hard Per-Class Drop Ranking ({metric_key})")
    ax.set_xlabel("Performance Drop")
    ax.set_ylabel("Class")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / f"hard_domain_drop_ranking_{metric_key}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_domain_gap_csv(results: dict[str, Any], output_dir: Path, metric_key: str = "map50_95") -> None:
    class_names, drop_easy, drop_hard = compute_domain_drops(results, metric_key=metric_key)

    lines = ["class_name,drop_easy,drop_hard\n"]
    for class_name, d_easy, d_hard in zip(class_names, drop_easy, drop_hard):
        easy_str = "" if np.isnan(d_easy) else str(float(d_easy))
        hard_str = "" if np.isnan(d_hard) else str(float(d_hard))
        lines.append(f"{class_name},{easy_str},{hard_str}\n")

    with open(output_dir / f"domain_gap_{metric_key}.csv", "w", encoding="utf-8") as f:
        f.writelines(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot one saved three-domain evaluation result")
    parser.add_argument("results_json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    input_json = args.results_json.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else input_json.parent / f"{input_json.stem}_plots"
    )
    ensure_output_dir(output_dir)

    results = load_results(input_json)

    plot_overall_metrics(results, output_dir)

    plot_per_class_metric(
        results=results,
        output_dir=output_dir,
        metric_key="ap50",
        filename="per_class_ap50_comparison.png",
        title="Per-Class AP50 Comparison Across Datasets",
    )

    plot_per_class_metric(
        results=results,
        output_dir=output_dir,
        metric_key="map50_95",
        filename="per_class_map50_95_comparison.png",
        title="Per-Class mAP50-95 Comparison Across Datasets",
    )

    plot_domain_gap(
        results=results,
        output_dir=output_dir,
        metric_key="map50_95",
    )

    plot_hard_domain_drop_ranking(
        results=results,
        output_dir=output_dir,
        metric_key="map50_95",
    )

    save_domain_gap_csv(
        results=results,
        output_dir=output_dir,
        metric_key="map50_95",
    )

    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
