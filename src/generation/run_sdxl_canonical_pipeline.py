"""Run clean SDXL canonical generation followed by mixed degradation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import run_sdxl_depth_background_experiment as experiment


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/generation/sdxl_canonical_2048.yaml"
DEGRADER = REPO_ROOT / "src/augmentation/degrade_sdxl_dataset.py"
DEFAULT_POLICY = REPO_ROOT / "configs/generation/genai_degradation_v1.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--clean-name", default="canonical_clean_2048_v1")
    parser.add_argument("--degraded-name", default="canonical_mixed_degradation_2048_v1")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    generator = experiment.load_generator()
    helper = generator.load_helpers()
    config_path = helper.repo_path(args.config)
    config, _ = generator.load_generation_config(helper, config_path)
    output_root = helper.repo_path(config["output_root"])
    clean_root = output_root / args.clean_name
    degraded_root = output_root / args.degraded_name
    missing_plates = [
        output_root / "plates" / f"{scene}_v{variant}.png"
        for scene in config["scene_plate_prompts"]
        for variant in range(int(config["plate_variants_per_scene"]))
        if not (output_root / "plates" / f"{scene}_v{variant}.png").is_file()
    ]
    if missing_plates:
        raise FileNotFoundError(f"Missing reviewed background plates: {missing_plates[:3]}")
    collisions = [path for path in (clean_root, degraded_root) if path.exists()]
    if collisions:
        raise FileExistsError(f"Refusing to overwrite canonical outputs: {collisions}")

    generation_command = [
        sys.executable,
        str(REPO_ROOT / "src/generation/run_sdxl_depth_background_experiment.py"),
        "--config", str(config_path), "--stage", "inpaint_pose",
        "--gpu", str(args.gpu), "--final-name", args.clean_name,
    ]
    if args.preflight_only:
        subprocess.run([*generation_command, "--preflight-only"], cwd=REPO_ROOT, check=True)
        print(f"clean_output={clean_root}")
        print(f"degraded_output={degraded_root}")
        return

    subprocess.run(generation_command, cwd=REPO_ROOT, check=True)
    subprocess.run(
        [
            sys.executable, str(DEGRADER),
            "--source-manifest", str(clean_root / "manifest.json"),
            "--output-root", str(degraded_root),
            "--policy", str(args.policy), "--seed", str(config["seed"]),
        ],
        cwd=REPO_ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()

