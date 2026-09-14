from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = REPO_ROOT / "src/generation/run_sdxl_fb2048_pipeline.py"


def load_pipeline():
    spec = importlib.util.spec_from_file_location("sdxl_fb2048_test", PIPELINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import SDXL-FB2048 pipeline")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PipelineStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fb = load_pipeline()
        cls.config = cls.fb.load_yaml(REPO_ROOT / "configs/generation/sdxl_fb2048_v2.yaml")
        cls.scenes = cls.fb.load_yaml(REPO_ROOT / "configs/generation/genai_scene_policy_v1.yaml")

    def test_production_foreground_prompts_are_unique(self) -> None:
        prompts = []
        for class_id in range(16):
            count = (self.config["production"]["aerosol_foreground_candidates"]
                     if class_id == 11 else self.config["production"]["foreground_candidates_per_class"])
            prompts.extend(self.fb.foreground_prompt(self.config, class_id, attempt)
                           for attempt in range(count))
        self.assertEqual(2592, len(prompts))
        self.assertEqual(len(prompts), len(set(prompts)))

    def test_pilot_foregrounds_cover_every_variation_axis(self) -> None:
        counts = self.fb.validate_config(self.config, self.scenes, "pilot")
        schedule = self.fb.foreground_schedule(self.config, counts, "pilot")
        settings = self.config["foreground_generation"]
        viewpoints = len(settings["viewpoint_profiles"])
        conditions = len(settings["condition_profiles"])
        variants_by_class = {
            int(key): value for key, value in settings["class_variants"].items()
        }
        self.assertEqual(16 * 8, len(schedule))
        self.assertEqual(len(schedule), len(set(schedule)))
        prompts = [self.fb.foreground_prompt(self.config, class_id, attempt)
                   for class_id, attempt in schedule]
        self.assertEqual(len(prompts), len(set(prompts)))
        for class_id in range(16):
            attempts = [attempt for scheduled_class, attempt in schedule
                        if scheduled_class == class_id]
            variants = len(variants_by_class[class_id])
            self.assertEqual(len(attempts), len(set(attempts)))
            self.assertEqual(set(range(viewpoints)), {
                attempt % viewpoints for attempt in attempts
            })
            self.assertEqual(set(range(variants)), {
                (attempt // viewpoints) % variants for attempt in attempts
            })
            self.assertEqual(set(range(conditions)), {
                (attempt // (viewpoints * variants)) % conditions for attempt in attempts
            })

    def test_pilot_covers_every_assigned_scene(self) -> None:
        count = int(self.config["pilot"]["background_candidates"])
        per_class = count // 16
        assignments = {int(key): value for key, value in self.scenes["class_to_scene_families"].items()}
        self.assertTrue(all(per_class >= len(scene_names) for scene_names in assignments.values()))
        self.assertEqual(count, len(self.fb.scheduled_background_conditionings(
            self.config, self.scenes, count,
        )))

    def test_resume_requires_matching_generation_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.png"
            first = {"seed": 1, "prompt": "first"}
            self.fb.save_image_atomic(Image.new("RGB", (32, 32), "white"), path, first)
            self.assertTrue(self.fb.verified_existing_image(path, (32, 32), first))
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                self.fb.verified_existing_image(path, (32, 32), {"seed": 2, "prompt": "second"})

    def test_manual_checks_and_automatic_person_rejection_are_hard_gates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.png"
            Image.new("RGB", (32, 32), "white").save(image_path)
            record = {
                "candidate_id": "fg_0", "image_path": str(image_path),
                "auto_person_detected": False,
            }
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"records": [record]}), encoding="utf-8")
            review = root / "review.csv"
            row = {field: "" for field in self.fb.REVIEW_FIELDS}
            row.update({"candidate_id": "fg_0", "image_path": str(image_path),
                        "review_status": "accepted", "identity_correct": "yes"})
            with review.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fb.REVIEW_FIELDS)
                writer.writeheader(); writer.writerow(row)
            with self.assertRaisesRegex(ValueError, "lacks required yes checks"):
                self.fb.accepted_records(manifest, review, ["identity_correct", "no_human"])

            row["no_human"] = "yes"
            record["auto_person_detected"] = True
            manifest.write_text(json.dumps({"records": [record]}), encoding="utf-8")
            with review.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fb.REVIEW_FIELDS)
                writer.writeheader(); writer.writerow(row)
            with self.assertRaisesRegex(ValueError, "cannot override automatic person rejection"):
                self.fb.accepted_records(manifest, review, ["identity_correct", "no_human"])

    def test_reviewed_support_region_is_validated(self) -> None:
        row = {"candidate_id": "bg_0", "review": {
            "support_x1": "0.2", "support_y1": "0.5",
            "support_x2": "0.8", "support_y2": "0.7",
        }}
        self.assertEqual((200, 500, 800, 700), self.fb.reviewed_support_bbox(row, 1000, 1000))
        row["review"]["support_x2"] = "1.2"
        with self.assertRaisesRegex(ValueError, "Invalid normalized support box"):
            self.fb.reviewed_support_bbox(row, 1000, 1000)

    def test_nested_manifests_require_balanced_prefixes(self) -> None:
        records = [
            {"candidate_id": f"final_{index:03d}", "class_id": index % 16}
            for index in range(32)
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.fb.write_nested_manifests(output, records, [16, 32])
            manifest = json.loads((output / "subsets/fb_16.json").read_text(encoding="utf-8"))
            self.assertEqual(16, manifest["images"])
            self.assertEqual(1, manifest["images_per_class"])

        unbalanced = sorted(records, key=lambda row: row["class_id"])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "not class-balanced"):
                self.fb.write_nested_manifests(Path(directory), unbalanced, [16])


if __name__ == "__main__":
    unittest.main()
