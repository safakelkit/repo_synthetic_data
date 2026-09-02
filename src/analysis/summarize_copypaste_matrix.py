from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ROOT = REPO_ROOT / "runs/evaluation"
EXPERIMENTS = (
    ("E000", 0, EVALUATION_ROOT / "E000_results.json"),
    ("CP-B0512", 512, EVALUATION_ROOT / "CP-B0512_yolo11s_seed0_results.json"),
    ("CP-B1024", 1024, EVALUATION_ROOT / "CP-B1024_yolo11s_seed0_results.json"),
    ("CP-B1536", 1536, EVALUATION_ROOT / "CP-B1536_yolo11s_seed0_results.json"),
    ("CP-B2048", 2048, EVALUATION_ROOT / "CP-B2048_yolo11s_seed0_results.json"),
)
DOMAINS = (
    ("clean", "insp_det", "INSP-DET clean"),
    ("easy", "insp_mot_det_easy", "INSP-MOT-DET easy"),
    ("hard", "insp_mot_det_hard", "INSP-MOT-DET hard"),
)


def load_rows() -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    for experiment_id, synthetic_images, path in EXPERIMENTS:
        if not path.is_file():
            raise FileNotFoundError(f"Missing evaluation result: {path}")
        datasets = json.loads(path.read_text(encoding="utf-8"))["datasets"]
        row: dict[str, str | int | float] = {
            "experiment_id": experiment_id,
            "synthetic_images": synthetic_images,
            "result_json": path.relative_to(REPO_ROOT).as_posix(),
        }
        for short_name, dataset_key, _ in DOMAINS:
            row[f"{short_name}_map50_95"] = float(
                datasets[dataset_key]["overall"]["map50_95"]
            )
        rows.append(row)

    baseline = rows[0]
    for row in rows:
        for short_name, _, _ in DOMAINS:
            metric = f"{short_name}_map50_95"
            row[f"{short_name}_delta_vs_e000"] = float(row[metric]) - float(
                baseline[metric]
            )
    return rows


def write_csv(rows: list[dict[str, str | int | float]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_plot(rows: list[dict[str, str | int | float]], path: Path) -> None:
    quantities = [int(row["synthetic_images"]) for row in rows]
    fig, (absolute_ax, delta_ax) = plt.subplots(1, 2, figsize=(13, 5.2))
    colors = {"clean": "#3366cc", "easy": "#e69138", "hard": "#b52b65"}

    for short_name, _, label in DOMAINS:
        absolute = [float(row[f"{short_name}_map50_95"]) for row in rows]
        delta = [float(row[f"{short_name}_delta_vs_e000"]) for row in rows]
        absolute_ax.plot(quantities, absolute, marker="o", linewidth=2, label=label,
                         color=colors[short_name])
        delta_ax.plot(quantities, delta, marker="o", linewidth=2, label=label,
                      color=colors[short_name])

    absolute_ax.set_title("Absolute test mAP50-95")
    absolute_ax.set_ylabel("mAP50-95")
    delta_ax.set_title("Change relative to E000")
    delta_ax.set_ylabel("Absolute mAP50-95 change")
    delta_ax.axhline(0, color="black", linewidth=1, linestyle="--")
    for axis in (absolute_ax, delta_ax):
        axis.set_xlabel("Number of cut-paste images")
        axis.set_xticks(quantities)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("Cut-paste quantity response (YOLO11s, detector seed 0)")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = load_rows()
    csv_path = EVALUATION_ROOT / "copy_paste_matrix_summary.csv"
    plot_path = EVALUATION_ROOT / "copy_paste_matrix_summary.png"
    write_csv(rows, csv_path)
    render_plot(rows, plot_path)
    print(csv_path)
    print(plot_path)


if __name__ == "__main__":
    main()
