"""Compare real-only, copy-paste, and mixed-degradation SDXL evaluations."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ROOT = REPO_ROOT / "runs/evaluation"
COMPARISON_ROOT = EVALUATION_ROOT / "comparisons"
DOMAINS = (
    ("clean", "insp_det", "INSP-DET clean"),
    ("easy", "insp_mot_det_easy", "INSP-MOT-DET easy"),
    ("hard", "insp_mot_det_hard", "INSP-MOT-DET hard"),
)
SERIES = (
    ("CP-B", "Copy-paste", EVALUATION_ROOT / "cut_paste"),
    ("SDXL-M", "SDXL mixed degradation", EVALUATION_ROOT / "sdxl/mixed_degradation"),
)
QUANTITIES = (512, 1024, 1536, 2048)


def result_path(root: Path, prefix: str, quantity: int) -> Path:
    return root / f"{prefix}{quantity:04d}_yolo11s_seed0_results.json"


def metrics(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing evaluation result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))["datasets"]


def main() -> None:
    baseline_path = EVALUATION_ROOT / "baseline/E000_results.json"
    baseline = metrics(baseline_path)
    baseline_row = {
        "method": "E000", "method_label": "Real-only baseline",
        "synthetic_images": 0,
        "result_json": baseline_path.relative_to(REPO_ROOT).as_posix(),
    }
    for short, key, _ in DOMAINS:
        value = float(baseline[key]["overall"]["map50_95"])
        aerosol = next(x for x in baseline[key]["per_class"] if x["class_id"] == 11)
        baseline_row[f"{short}_map50_95"] = value
        baseline_row[f"{short}_delta_vs_e000"] = 0.0
        baseline_row[f"{short}_aerosol_map50_95"] = float(aerosol["map50_95"])
        baseline_row[f"{short}_aerosol_delta_vs_e000"] = 0.0
    rows = [baseline_row]
    per_class_rows = []

    def append_per_class(method: str, label: str, quantity: int, datasets: dict) -> None:
        for short, key, _ in DOMAINS:
            for item in datasets[key]["per_class"]:
                per_class_rows.append({
                    "method": method,
                    "method_label": label,
                    "synthetic_images": quantity,
                    "domain": short,
                    "class_id": int(item["class_id"]),
                    "class_name": item["class_name"],
                    "precision": item["precision"],
                    "recall": item["recall"],
                    "ap50": item["ap50"],
                    "map50_95": item["map50_95"],
                })

    append_per_class("E000", "Real-only baseline", 0, baseline)
    for prefix, label, root in SERIES:
        for quantity in QUANTITIES:
            path = result_path(root, prefix, quantity)
            datasets = metrics(path)
            row = {
                "method": prefix, "method_label": label,
                "synthetic_images": quantity,
                "result_json": path.relative_to(REPO_ROOT).as_posix(),
            }
            for short, key, _ in DOMAINS:
                value = float(datasets[key]["overall"]["map50_95"])
                base_value = float(baseline[key]["overall"]["map50_95"])
                aerosol = next(x for x in datasets[key]["per_class"] if x["class_id"] == 11)
                base_aerosol = next(x for x in baseline[key]["per_class"] if x["class_id"] == 11)
                row[f"{short}_map50_95"] = value
                row[f"{short}_delta_vs_e000"] = value - base_value
                row[f"{short}_aerosol_map50_95"] = float(aerosol["map50_95"])
                row[f"{short}_aerosol_delta_vs_e000"] = (
                    float(aerosol["map50_95"]) - float(base_aerosol["map50_95"])
                )
            rows.append(row)
            append_per_class(prefix, label, quantity, datasets)

    COMPARISON_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = COMPARISON_ROOT / "augmentation_matrix_overall.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    per_class_path = COMPARISON_ROOT / "augmentation_matrix_per_class.csv"
    with per_class_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_class_rows[0]))
        writer.writeheader()
        writer.writerows(per_class_rows)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True)
    colors = {"CP-B": "#777777", "SDXL-M": "#b52b65"}
    for col, (short, _, title) in enumerate(DOMAINS):
        for prefix, label, _ in SERIES:
            selected = [row for row in rows if row["method"] == prefix]
            axes[0, col].plot(
                QUANTITIES, [row[f"{short}_map50_95"] for row in selected],
                marker="o", label=label, color=colors[prefix],
            )
            axes[1, col].plot(
                QUANTITIES, [row[f"{short}_aerosol_map50_95"] for row in selected],
                marker="o", label=label, color=colors[prefix],
            )
        axes[0, col].set_title(title)
        axes[1, col].set_title(f"{title} — Aerosol can")
        axes[1, col].set_xlabel("Synthetic training images")
        for row in range(2):
            axes[row, col].set_ylabel("mAP50-95")
            axes[row, col].grid(alpha=0.25)
    axes[0, 0].legend()
    fig.tight_layout()
    plot_path = COMPARISON_ROOT / "augmentation_matrix_summary.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    print(json.dumps({
        "overall_rows": len(rows), "per_class_rows": len(per_class_rows),
        "overall_csv": str(csv_path), "per_class_csv": str(per_class_path),
        "plot": str(plot_path),
    }, indent=2))


if __name__ == "__main__":
    main()
