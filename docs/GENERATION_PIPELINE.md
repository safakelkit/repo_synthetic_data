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

### Implemented pilot workflow

The pilot configuration is `configs/generation/sdxl_fb2048_v1.yaml`; the runner
is `src/generation/run_sdxl_fb2048_pipeline.py`. It generates 32 background and
32 foreground candidates, then requires explicit review decisions before SAM3
segmentation and again before the 16-image composition stage. Exact duplicate
files, invalid mask geometry, incorrect class counts, review/manifest drift,
and existing output directories stop the run. Production is locked in config.

Run each preflight on its assigned GPU. Unlike a structural dry run, this loads
the real model/tokenizers and checks every scheduled prompt against both CLIP
77-token limits before creating output directories:

```bash
CUDA_VISIBLE_DEVICES=0 ../env_sam3/bin/python src/generation/run_sdxl_fb2048_pipeline.py --stage backgrounds --gpu 0 --preflight-only
CUDA_VISIBLE_DEVICES=1 ../env_sam3/bin/python src/generation/run_sdxl_fb2048_pipeline.py --stage foregrounds --gpu 0 --preflight-only
```

Remove `--preflight-only` only for the current stage. After background and
foreground generation, edit only `review_status` and `review_reason` in their
review CSV files. After segmentation, review the RGBA/mask rows in the same
way. Every generated final remains training-forbidden pending final review.
Background and foreground generation use separate stage locks, atomically write
images, and reuse valid partial images after an interruption. They can therefore
run concurrently on different GPUs and safely resume with the same command.

The initial pool is extended without deleting accepted candidates by
`src/generation/extend_sdxl_fb2048_candidates.py`. Extensions preserve existing
review decisions, append new immutable manifest records, use new deterministic
seeds, and add only the new candidates as `pending`. Background retries use an
explicit lower-third support-surface prompt; foreground retries may use frozen
class-specific corrective wording. The extension preflight loads the real SDXL
tokenizers before producing any image.

Background diversity is not seed-only. The pilot and production schedule now
crosses eight choices each for support type, room layout, wear, occupancy cues,
camera, and lighting while retaining target-free review. This avoids reverting
to the sterile empty-room pattern in the 2,560-candidate production run.

For geometry-sensitive classes that failed text-only retries, the controlled
runner is `src/generation/generate_sdxl_fb2048_controlled_retries.py`. It draws
non-photographic Canny geometry for Scissors, Cigarettes, Saw, and Battery and
uses SDXL ControlNet to generate all final pixels. No real object pixels or real
silhouettes enter these controls. Outputs append to the same immutable candidate
manifest and remain pending review.
