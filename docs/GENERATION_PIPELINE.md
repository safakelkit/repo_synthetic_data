# GenAI Generation Pipelines

This file separates the executed SDXL baseline from the prospective
foreground/background-separated follow-up. Planned work is not canonical data
and has no detector result until generation, QC, release, and training finish.

## Executed SDXL baseline — preserved

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

## Active prospective method — SDXL-FB2048 v2

The next SDXL experiment will generate foregrounds and backgrounds separately,
then compose them with exact masks. All final scene pixels originate from GenAI;
real INSP-DET and Places365 pixels are not pasted into this method.

### Fixed construction

1. Generate a surplus pool of target-free indoor backgrounds at 1024x1024.
2. Accept exactly 2,048 backgrounds: one unique background per final image.
3. Generate isolated foreground candidates from scratch with SDXL, accepting
   exactly 128 valid instances for each of the 16 fixed classes.
4. Extract and retain an alpha mask for every foreground. Reject wrong-class,
   malformed, duplicate, truncated, or unsegmentable candidates.
5. Pair foregrounds and backgrounds one-to-one under the frozen context matrix;
   sample pose and scale without using clean/easy/hard test results.
6. Compose from the retained mask and apply localized diffusion harmonization
   for lighting, edge, shadow, and physical-contact consistency.
7. Derive the YOLO box from the final retained visible mask, never from prompt
   text or an expected control rectangle.

### Diversity and quality gates

- Final quantity is 2,048, not 2,024, so every class has exactly 128 targets.
- Pixel hash uniqueness alone is insufficient. Background layout/scene
  similarity and foreground appearance/shape similarity must also pass frozen
  near-duplicate thresholds.
- Minimum candidate surplus is 25% (at least 2,560 backgrounds and 160 object
  candidates per ordinary class); Aerosol retains a 50% surplus because its
  cap/nozzle identity has historically produced more failures.
- Each final image contains one primary project-class target. Any additional
  project-class instance must be annotated or the image rejected.
- Nested 512/1,024/1,536/2,048 prefixes remain exactly class-balanced.
- Production remains blocked until same-size all-class pilots validate object
  identity, background realism, segmentation, compositing, harmonization, and
  annotation integrity.

### Method naming boundary

The existing real-background + real-INSP-object + localized-SDXL code is a
`CP+SDXL harmonization` hybrid ablation. It may be evaluated separately, but it
must not be reported as SDXL-FB2048, pure GenAI data, or a replacement for the
executed SDXL baseline.
