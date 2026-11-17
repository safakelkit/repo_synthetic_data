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
CONTROLLED_PATH = REPO_ROOT / "src/generation/generate_sdxl_fb2048_controlled_retries.py"


def load_pipeline():
    spec = importlib.util.spec_from_file_location("sdxl_fb2048_test", PIPELINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import SDXL-FB2048 pipeline")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_controlled():
    spec = importlib.util.spec_from_file_location("sdxl_fb2048_controlled_test", CONTROLLED_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import controlled SDXL-FB2048 pipeline")
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

    def test_foreground_negatives_are_class_specific_and_routes_partition_classes(self) -> None:
        negatives = [self.fb.foreground_negative_prompt(self.config, class_id)
                     for class_id in range(16)]
        self.assertEqual(16, len(set(negatives)))
        constraints = self.config["foreground_generation"]["class_constraints"].values()
        self.assertTrue(all(" no " not in f" {value.lower()} " for value in constraints))
        routes = self.config["foreground_generation"]["generation_routes"]
        routed = [int(class_id) for class_ids in routes.values() for class_id in class_ids]
        self.assertEqual(list(range(16)), sorted(routed))

    def test_controlled_retry_has_geometry_and_prompt_for_each_target_class(self) -> None:
        controlled = load_controlled()
        retry = self.config["foreground_generation"]["controlled_retry"]
        configured = set(retry["class_ids"])
        self.assertEqual(configured, set(retry["identity_prompts"]))
        self.assertEqual(configured, set(retry["controlnet_conditioning_scale_by_class"]))
        self.assertTrue(configured.issubset(controlled.DRAWERS))
        for class_id in configured:
            self.assertIn("one", retry["identity_prompts"][class_id].lower())
            self.assertLessEqual(retry["controlnet_conditioning_scale_by_class"][class_id], 0.70)
            control = controlled.make_control(class_id, 0, (1024, 1024))
            self.assertEqual((1024, 1024), control.size)
            extrema = control.convert("L").getextrema()
            self.assertEqual((0, 255), extrema)

    def test_saw_blade_teeth_and_battery_label_are_in_synthetic_control(self) -> None:
        import numpy as np

        controlled = load_controlled()
        saw = np.zeros((1024, 1024), dtype=np.uint8)
        battery = np.zeros((1024, 1024), dtype=np.uint8)
        controlled.draw_hacksaw(saw)
        controlled.draw_battery(battery)
        self.assertGreater(np.count_nonzero(saw[671:681, 265:730]), 100)
        self.assertGreater(len(np.unique(battery[410:672, 412:613])), 2)

    def test_failed_control_fallback_plan_targets_strict_text_classes(self) -> None:
        settings = self.config["foreground_generation"]
        plan = settings["focused_retry_plans"]["failed_control_text_fallback"]
        self.assertEqual("ready", plan["status"])
        self.assertEqual("foregrounds", plan["pool"])
        self.assertEqual({3, 8, 12}, set(plan["class_ids"]))
        self.assertTrue(set(plan["class_ids"]).issubset(
            settings["generation_routes"]["text_to_image_strict_review"]
        ))

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

    def test_production_background_schedule_covers_independent_axes(self) -> None:
        count = int(self.config["production"]["background_candidates"])
        self.fb.require_background_schedule_diversity(self.config, self.scenes, count)
        altered = dict(self.config)
        altered["background_generation"] = dict(self.config["background_generation"])
        altered["background_generation"]["support_profiles"] = [
            self.config["background_generation"]["support_profiles"][0]
        ]
        with self.assertRaisesRegex(ValueError, "collapses"):
            self.fb.require_background_schedule_diversity(altered, self.scenes, count)

    def test_backgrounds_require_accepted_segmented_foregrounds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                self.fb.require_foreground_gate_before_backgrounds(
                    self.config, self.fb.validate_config(self.config, self.scenes, "pilot"),
                    Path(directory),
                )

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

    def test_review_cannot_change_manifest_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            review = root / "review.csv"
            record = {"candidate_id": "fg_0", "class_id": 2,
                      "class_name": "Scissors", "scene_name": "", "image_path": "original.png"}
            manifest.write_text(json.dumps({"records": [record]}), encoding="utf-8")
            row = {field: "" for field in self.fb.REVIEW_FIELDS}
            row.update({"candidate_id": "fg_0", "class_id": "2", "class_name": "Scissors",
                        "image_path": "other.png", "review_status": "pending"})
            with review.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fb.REVIEW_FIELDS)
                writer.writeheader(); writer.writerow(row)
            with self.assertRaisesRegex(ValueError, "metadata differs from manifest"):
                self.fb.accepted_records(manifest, review)

    def test_segmented_review_uses_rgba_manifest_image_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, review = root / "manifest.json", root / "review.csv"
            record = {"candidate_id": "fg_0", "class_id": 2,
                      "class_name": "Scissors", "image_path": "rgba/fg_0.png"}
            manifest.write_text(json.dumps({"records": [record]}), encoding="utf-8")
            row = {field: "" for field in self.fb.REVIEW_FIELDS}
            row.update({"candidate_id": "fg_0", "class_id": "2", "class_name": "Scissors",
                        "image_path": "rgba/fg_0.png", "review_status": "accepted",
                        "mask_correct": "yes"})
            with review.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fb.REVIEW_FIELDS)
                writer.writeheader(); writer.writerow(row)
            self.assertEqual(1, len(self.fb.accepted_records(manifest, review, ["mask_correct"])))

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
