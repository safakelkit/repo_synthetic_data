"""Derive an experimental 75% clean / 25% light FB2048 set after final review.

The accepted clean images remain untouched. This output stays training-forbidden
until its own visibility review and a predeclared detector comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_copypaste_dataset import apply_degradations

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY = ROOT / "configs/generation/fb_light25_experimental.yaml"


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def severity_for(class_id: int, class_position: int) -> str:
    """Exactly one light image per four instances in every class and prefix."""
    return "light" if (class_position + class_id) % 4 == 0 else "clean"


def validate_release(path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "production_accepted":
        raise ValueError("Only a fully accepted production clean set may be degraded")
    records = manifest.get("records", [])
    if len(records) != 2048 or Counter(int(r["class_id"]) for r in records) != Counter({i: 128 for i in range(16)}):
        raise ValueError("Source must contain exactly 128 accepted images per class")
    if len({r["image_path"] for r in records}) != len(records):
        raise ValueError("Source image paths are not unique")
    for row in records:
        for key, hash_key in (("image_path", "image_sha256"), ("label_path", "label_sha256")):
            source = resolve(row[key])
            if not source.is_file() or sha256(source) != row[hash_key]:
                raise ValueError(f"Source file/hash mismatch: {source}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-release", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    source_path, policy_path, output = map(resolve, (args.source_release, args.policy, args.output_root))
    synthetic_root = ROOT / "data/synthetic"
    if not output.is_relative_to(synthetic_root) or output == synthetic_root:
        raise ValueError("Derived output must be a new directory under data/synthetic")
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    if policy.get("status") != "experimental_pending_visibility_review" or policy.get("clean_fraction") != 0.75 or policy.get("light_fraction") != 0.25:
        raise ValueError("Expected the frozen experimental 75/25 light policy")
    records = validate_release(source_path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if args.preflight_only:
        print(json.dumps({"status": "ready", "images": 2048, "clean": 1536, "light": 512,
                          "training_use_forbidden": True}, indent=2))
        return
    (output / "images").mkdir(parents=True)
    (output / "labels").mkdir()
    rng = random.Random(int(policy["seed"]))
    class_positions: Counter[int] = Counter()
    output_records = []
    for row in records:
        class_id = int(row["class_id"])
        severity = severity_for(class_id, class_positions[class_id])
        class_positions[class_id] += 1
        source_image = resolve(row["image_path"])
        source_label = resolve(row["label_path"])
        image_path = output / "images" / source_image.name
        label_path = output / "labels" / source_label.name
        if severity == "clean":
            shutil.copy2(source_image, image_path)
            operations = []
        else:
            image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Cannot decode {source_image}")
            result, operations = apply_degradations(image, "light", {"levels": {"light": policy["light_profile"]}}, rng)
            if not cv2.imwrite(str(image_path), result):
                raise RuntimeError(f"Cannot write {image_path}")
        shutil.copy2(source_label, label_path)
        output_records.append({
            "candidate_id": row["candidate_id"], "class_id": class_id,
            "severity": severity, "operations": operations,
            "clean_source_path": row["image_path"], "clean_source_sha256": row["image_sha256"],
            "image_path": str(image_path.relative_to(ROOT)), "image_sha256": sha256(image_path),
            "label_path": str(label_path.relative_to(ROOT)), "label_sha256": sha256(label_path),
        })
    distribution = Counter(row["severity"] for row in output_records)
    if distribution != Counter({"clean": 1536, "light": 512}):
        raise AssertionError(f"Unexpected severity allocation: {distribution}")
    temporary = output / ".manifest.json.partial"
    temporary.write_text(json.dumps({"status": "pending_visibility_review", "training_use_forbidden": True,
        "source_release_sha256": sha256(source_path), "policy_sha256": sha256(policy_path),
        "distribution": dict(distribution), "records": output_records}, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "manifest.json")
    print(f"Generated {len(output_records)} experimental images in {output}")


if __name__ == "__main__":
    main()
