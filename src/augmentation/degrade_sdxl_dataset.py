"""Derive a deterministic mixed-degradation dataset from clean SDXL images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import yaml

from generate_copypaste_dataset import (
    apply_degradations,
    build_degradation_schedule,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY = REPO_ROOT / "configs/generation/genai_degradation_v1.yaml"


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def stored_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def make_contact_sheet(records: list[dict[str, Any]], output: Path) -> None:
    from PIL import Image, ImageDraw

    severity_order = ("clean", "light", "medium", "heavy")
    selected: list[dict[str, Any]] = []
    for severity in severity_order:
        candidates = [row for row in records if row["degradation_severity"] == severity]
        selected.extend(candidates[:16])
    thumb, caption, columns = 320, 38, 4
    rows = (len(selected) + columns - 1) // columns
    sheet = Image.new("RGB", (thumb * columns, (thumb + caption) * rows), "white")
    draw = ImageDraw.Draw(sheet)
    for position, record in enumerate(selected):
        with Image.open(REPO_ROOT / record["output"]).convert("RGB") as image:
            image.thumbnail((thumb, thumb))
            x = position % columns * thumb
            y = position // columns * (thumb + caption)
            sheet.paste(image, (x, y))
        draw.text(
            (x + 4, y + thumb + 3),
            f"g{int(record['index']):05d} c{int(record['class_id']):02d} {record['degradation_severity']}",
            fill="black",
        )
    sheet.save(output)


def validate_source(records: list[dict[str, Any]], policy: dict[str, Any]) -> Counter[int]:
    if not records:
        raise ValueError("Source manifest contains no records")
    counts = Counter(int(row["class_id"]) for row in records)
    if set(counts) != set(range(16)):
        raise ValueError(f"Expected classes 0-15, found {sorted(counts)}")
    block_size = sum(int(value) for value in policy["per_class_per_32_images"].values())
    invalid = {class_id: count for class_id, count in counts.items() if count % block_size}
    if invalid:
        raise ValueError(f"Per-class counts must divide into {block_size}-image blocks: {invalid}")
    for row in records:
        source = repo_path(row["output"])
        if not source.is_file():
            raise FileNotFoundError(source)
        expected = row.get("sha256") or row.get("output_sha256")
        if expected and sha256(source) != expected:
            raise ValueError(f"Source hash mismatch: {source}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--seed", type=int, default=60000)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    source_manifest_path = repo_path(args.source_manifest)
    output_root = repo_path(args.output_root)
    policy_path = repo_path(args.policy)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    policy = load_yaml(policy_path)
    if policy.get("status") != "frozen_equal_to_executed_copy_paste_degradation_v1_pending_genai_integration":
        raise RuntimeError("The requested GenAI degradation policy is not frozen")
    if not policy.get("apply_to_complete_generated_image"):
        raise ValueError("Degradation must be applied to the complete image")
    records = sorted(source_manifest["records"], key=lambda row: int(row["index"]))
    class_counts = validate_source(records, policy)
    if output_root.exists() and not args.preflight_only:
        raise FileExistsError(f"Refusing to overwrite {output_root}")

    degradation_seed = int(args.seed) + int(policy["seed_offset"])
    report = {
        "status": "ready",
        "source_records": len(records),
        "class_counts": dict(sorted(class_counts.items())),
        "degradation_seed": degradation_seed,
        "source_manifest": stored_path(source_manifest_path),
        "source_manifest_sha256": sha256(source_manifest_path),
        "policy": stored_path(policy_path),
        "policy_sha256": sha256(policy_path),
        "output_root": stored_path(output_root),
    }
    if args.preflight_only:
        print(json.dumps(report, indent=2))
        return

    (output_root / "images").mkdir(parents=True)
    source_labels = source_manifest_path.parent / "labels"
    if source_labels.is_dir():
        (output_root / "labels").mkdir()
    schedules = {
        class_id: build_degradation_schedule(
            count, policy["per_class_per_32_images"], degradation_seed + class_id
        )
        for class_id, count in class_counts.items()
    }
    class_positions: defaultdict[int, int] = defaultdict(int)
    rng = random.Random(degradation_seed)
    output_records: list[dict[str, Any]] = []
    severity_counts: Counter[str] = Counter()
    for position, source_record in enumerate(records, start=1):
        class_id = int(source_record["class_id"])
        class_position = class_positions[class_id]
        class_positions[class_id] += 1
        severity = schedules[class_id][class_position]
        source_path = repo_path(source_record["output"])
        destination = output_root / "images" / source_path.name
        if severity == "clean":
            shutil.copy2(source_path, destination)
            operations: list[dict[str, Any]] = []
        else:
            image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Cannot decode source image: {source_path}")
            degraded, operations = apply_degradations(image, severity, policy, rng)
            if not cv2.imwrite(str(destination), degraded):
                raise RuntimeError(f"Cannot write degraded image: {destination}")
        if source_labels.is_dir():
            source_label = source_labels / f"{source_path.stem}.txt"
            if not source_label.is_file():
                raise FileNotFoundError(source_label)
            shutil.copy2(source_label, output_root / "labels" / source_label.name)
        severity_counts[severity] += 1
        output_records.append({
            **source_record,
            "clean_source": stored_path(source_path),
            "clean_source_sha256": sha256(source_path),
            "output": stored_path(destination),
            "sha256": sha256(destination),
            "degradation_applied": severity != "clean",
            "degradation_severity": severity,
            "degradations": operations,
            "annotation_geometry_changed": False,
            "post_degradation_visibility_qc": "pending",
            "training_use_forbidden": True,
        })
        if position % 64 == 0 or position == len(records):
            print(f"[degradation {position}/{len(records)}]", flush=True)

    make_contact_sheet(output_records, output_root / "contact_sheet_by_severity.png")
    manifest = {
        "status": "generated_pending_post_degradation_visibility_qc",
        "training_use_forbidden": True,
        "source_manifest": report["source_manifest"],
        "source_manifest_sha256": report["source_manifest_sha256"],
        "degradation_policy": report["policy"],
        "degradation_policy_sha256": report["policy_sha256"],
        "degradation_seed": degradation_seed,
        "code_revision": git_revision(),
        "severity_counts": dict(severity_counts),
        "records": output_records,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_root / "degradation_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "index", "class_id", "class_name", "degradation_severity",
            "clean_source", "output", "post_degradation_visibility_qc",
        ])
        writer.writeheader()
        for row in output_records:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    print(json.dumps({**report, "severity_counts": dict(severity_counts)}, indent=2))


if __name__ == "__main__":
    main()
