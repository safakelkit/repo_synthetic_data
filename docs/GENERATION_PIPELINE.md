# Active GenAI Generation Pipeline

This file describes the executed and approved canonical SDXL pipeline. Failed
development pilots do not define the current implementation.

## Method

- SDXL inpainting and Canny ControlNet revisions are fixed by the canonical
  configuration and recorded in every manifest.
- Thirty-two reviewed structured background plates provide semantically valid,
  diverse correctional-facility interiors with furniture, fixtures, depth, and
  support surfaces.
- Class-specific source initialization and target-only Canny geometry preserve
  identity while allowing the model to integrate the object into the scene.
  Background Canny edges are excluded from target conditioning.
- Pose selection uses support geometry and scene perspective. YOLO boxes are
  written from the realized pose alpha box rather than inferred from prompt text.
- Aerosol uses an upright cylindrical spray-can initialization and explicit
  cap/nozzle language. Its 128 canonical outputs passed visual and box review.
- Generation first writes a clean 2,048-image dataset. Degradation is then
  applied to the complete rendered image, keeping label geometry unchanged.

## Canonical release

- Clean root: `data/synthetic/sdxl_background_diversity_experiment/canonical_clean_2048_v1/`
- Mixed root: `data/synthetic/sdxl_background_diversity_experiment/canonical_mixed_degradation_2048_v1/`
- Images: 2,048; 128/class; 1024x1024 RGB; all decoded successfully.
- Nested subsets: 512/1,024/1,536/2,048 with 32/64/96/128 images per class.
- Mixed schedule: 512 clean, 768 light, 512 medium, 256 heavy; every class has
  exact 32/48/32/16 counts.
- All 1,536 non-clean outputs differ from their sources; all 512 clean members
  remain unchanged. Labels are identical across paired clean/mixed images.
- Researcher visual approval plus stratified independent review and complete
  structural QC released the datasets for training on 2026-09-09.

Source files:

- `configs/generation/sdxl_generation.yaml`
- `src/generation/run_sdxl_depth_background_experiment.py`
- `configs/generation/genai_degradation_v1.yaml`
- `src/augmentation/degrade_sdxl_dataset.py`

Degradation-only detector matrix preflight:

```bash
CUDA_VISIBLE_DEVICES=1 ../env_sam3/bin/python src/training/train_sdxl_baselines.py --experiment mixed-all --preflight-only --evaluate
```

Training and three-domain evaluation:

```bash
CUDA_VISIBLE_DEVICES=1 ../env_sam3/bin/python src/training/train_sdxl_baselines.py --experiment mixed-all --evaluate
```

Evaluation artifacts are grouped under `runs/evaluation/sdxl/mixed_degradation/`.
