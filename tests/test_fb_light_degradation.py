from __future__ import annotations

import importlib.util
import tempfile
import unittest
from collections import Counter
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "src/augmentation/derive_fb_light_degradation.py"


class LightDegradationPolicyTests(unittest.TestCase):
    def test_nested_quantities_keep_class_and_severity_balance(self) -> None:
        spec = importlib.util.spec_from_file_location("fb_light_test", SCRIPT)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for per_class in (32, 64, 96, 128):
            counts = Counter(module.severity_for(class_id, position)
                             for class_id in range(16) for position in range(per_class))
            self.assertEqual({"clean": 12 * per_class, "light": 4 * per_class}, dict(counts))

    def test_unreleased_source_is_rejected(self) -> None:
        spec = importlib.util.spec_from_file_location("fb_light_test", SCRIPT)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "release.json"
            source.write_text('{"status":"generated_pending_final_review","records":[]}')
            with self.assertRaisesRegex(ValueError, "fully accepted production"):
                module.validate_release(source)


if __name__ == "__main__":
    unittest.main()
