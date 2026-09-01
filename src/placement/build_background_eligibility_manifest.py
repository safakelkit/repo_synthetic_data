from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/placement/background_eligibility_v1.yaml"


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid YAML mapping: {path}")
    return value


def build(config_path: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    source = repo_path(config["source_manifest"])
    output = repo_path(config["output_manifest"])
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    accepted = [row for row in rows if row["review_status"] == "accepted"]
    by_background: dict[str, dict[str, str]] = {}
    for row in accepted:
        path = repo_path(row["background_path"])
        if not path.is_file() or sha256(path) != row["background_sha256"]:
            raise ValueError(f"Background integrity failure: {path}")
        record = by_background.setdefault(
            row["background_path"],
            {
                "background_path": row["background_path"],
                "background_sha256": row["background_sha256"],
                "support_types": "",
                "eligibility_status": "accepted",
                "evidence": ";".join(config["eligibility"]["evidence"]),
                "review_interpretation": config["eligibility"]["interpretation"],
            },
        )
        supports = set(filter(None, record["support_types"].split(";")))
        supports.add(row["support_type"])
        record["support_types"] = ";".join(sorted(supports))
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["background_path", "background_sha256", "support_types", "eligibility_status", "evidence", "review_interpretation"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(by_background.values(), key=lambda row: row["background_path"]))
    result = {
        "format_version": 1,
        "status": "background_eligibility_complete",
        "config": config_path.resolve().relative_to(REPO_ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "source_manifest": source.resolve().relative_to(REPO_ROOT).as_posix(),
        "source_manifest_sha256": sha256(source),
        "output_manifest": output.resolve().relative_to(REPO_ROOT).as_posix(),
        "output_manifest_sha256": sha256(output),
        "eligible_backgrounds": len(by_background),
        "limitation": config["eligibility"]["limitation"],
    }
    output.with_name("background_eligibility_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the reviewed copy-paste background eligibility manifest")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(build(repo_path(args.config)), indent=2))


if __name__ == "__main__":
    main()
