from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/placement/support_geometry_full_decisions_v2.yaml"


def repo_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Invalid YAML mapping: {path}")
    return value


def finalize(config_path: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    input_path = repo_path(config["input_manifest"])
    output_path = repo_path(config["output_manifest"])
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    for field in ("review_status", "reviewer_note"):
        if field not in fields:
            fields.append(field)

    rejected = {
        (item["background_stem"], item["support_type"]): item["reason"]
        for item in config["policy"]["rejected_regions"]
    }
    used_rejections: set[tuple[str, str]] = set()
    counts: Counter[str] = Counter()
    for row in rows:
        key = (Path(row["background_path"]).stem, row["support_type"])
        triage_status = row.get("triage_status", "candidate_accept")
        if row["derivation_status"] == "no_valid_region":
            row["review_status"] = config["policy"]["no_valid_region"]
            row["reviewer_note"] = "automatic_no_valid_region"
        elif row["support_type"] in set(config["policy"].get("disabled_support_types", [])):
            row["review_status"] = "rejected"
            row["reviewer_note"] = "support_type_disabled_due_to_2d_occlusion_risk"
        elif triage_status == "needs_review" and config["policy"].get("reject_needs_review", False):
            row["review_status"] = "rejected"
            row["reviewer_note"] = "conservative_rejection_of_automatic_risk_group"
        elif triage_status == "rejected":
            row["review_status"] = "rejected"
            row["reviewer_note"] = "automatic_triage_rejection"
        elif key in rejected:
            row["review_status"] = "rejected"
            row["reviewer_note"] = rejected[key]
            used_rejections.add(key)
        else:
            row["review_status"] = config["policy"]["derived_default"]
            row["reviewer_note"] = config["policy"].get(
                "accepted_note", "accepted_after_complete_candidate_sheet_review"
            )
        counts[f"{row['support_type']}:{row['review_status']}"] += 1

    missing = set(rejected) - used_rejections
    if missing:
        raise ValueError(f"Review rules did not match manifest rows: {sorted(missing)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "format_version": 2,
        "status": config.get("status", "full_geometry_review_complete"),
        "review_config": relative(config_path),
        "review_config_sha256": sha256(config_path),
        "input_manifest": relative(input_path),
        "input_manifest_sha256": sha256(input_path),
        "output_manifest": relative(output_path),
        "output_manifest_sha256": sha256(output_path),
        "row_count": len(rows),
        "counts": dict(sorted(counts.items())),
        "review_evidence": config.get("review_evidence", {}),
        "production_use_approved": bool(config["production_use"]["approved"]),
    }
    summary_path = output_path.with_name("review_summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the versioned geometry pilot review")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(finalize(repo_path(args.config)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
