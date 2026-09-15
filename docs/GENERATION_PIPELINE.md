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

The pilot configuration is `configs/generation/sdxl_fb2048_v2.yaml`; the runner
is `src/generation/run_sdxl_fb2048_pipeline.py`. It generates 64 background and
128 foreground candidates (8 per class, stratified across every viewpoint,
condition profile, and class-specific variant), then requires explicit review
decisions before SAM3
segmentation and again before the 16-image composition stage. Exact duplicate
files, invalid mask geometry, incorrect class counts, review/manifest drift,
and existing output directories stop the run. Production is locked in config.
The 64-background pilot covers all four scene assignments for every class.

Run each preflight on its assigned GPU. Unlike a structural dry run, this loads
the real model/tokenizers and checks every scheduled prompt against both CLIP
77-token limits before creating output directories:

```bash
CUDA_VISIBLE_DEVICES=0 ../env_sam3/bin/python src/generation/run_sdxl_fb2048_pipeline.py --stage backgrounds --gpu 0 --preflight-only
CUDA_VISIBLE_DEVICES=1 ../env_sam3/bin/python src/generation/run_sdxl_fb2048_pipeline.py --stage foregrounds --gpu 0 --preflight-only
```

Remove `--preflight-only` only for the current stage. After background and
foreground generation, fill the required manual check columns and edit
`review_status`/`review_reason` in their review CSV files. Accepted backgrounds
also require a normalized safe support rectangle (`support_x1`, `support_y1`,
`support_x2`, `support_y2`). After segmentation, review the RGBA/mask rows in
the same way. Every generated final remains training-forbidden pending final review.
Background and foreground generation use separate stage locks, atomically write
images, and reuse partial images only when their embedded generation fingerprint
matches the exact config, model, seed, prompt, and inference parameters. They can
run concurrently on different GPUs and safely resume with the same command.

The frozen COCO YOLO11s checkpoint screens every generated background,
foreground, and final image for people. A detection is an irreversible automatic
rejection for that candidate; manual `no_human=yes` remains required because a
person detector may miss isolated hands or small figures.

Foreground retries are class-routed from the v2 pilot evidence. Every class now
has a positive-only geometry description and a class-specific negative prompt.
Simple classes remain text-to-image; ambiguous classes use text-to-image with
strict review; Scissors, Pliers, Cigarettes, Saw, and Battery enter a separate
synthetic-Canny pilot. Automatic person rejection is retained, while object
count remains a manual hard gate because generic component counts cannot safely
separate true duplicates from articulated parts, packaged contents, or shadows.

The initial pool is extended without deleting accepted candidates by
`src/generation/extend_sdxl_fb2048_candidates.py`. Extensions preserve existing
review decisions, append new immutable manifest records, use new deterministic
seeds, and add only the new candidates as `pending`. Background retries use an
explicit lower-third support-surface prompt; foreground retries may use frozen
class-specific corrective wording. The extension preflight loads the real SDXL
tokenizers before producing any image.

Background diversity is not seed-only. Scene-specific recipes cross independent
support, composition, camera, lighting, material, and guidance axes. Foreground
diversity is also not seed-only: class-specific physical variants cross eight
viewpoints and four condition profiles, producing 160 unique prompts per normal
class and 192 for Aerosol.

Composition uses the reviewer-marked safe support rectangle for each accepted
background and class-specific object-size ranges. Fixed scene-wide placement
centers are no longer used. Exact hashes, within-class foreground dHash checks,
background dHash checks, immutable file hashes, and structured manual acceptance
all gate composition. Production output is interleaved by class and writes
balanced 512/1,024/1,536/2,048 subset manifests.

For geometry-sensitive classes that failed text-only retries, the experimental controlled
runner is `src/generation/generate_sdxl_fb2048_controlled_retries.py`. It draws
filled proxy silhouettes and derives non-photographic Canny geometry for
Scissors, Cigarettes, Saw, and Battery. It
uses SDXL ControlNet to generate all final pixels. No real object pixels or real
silhouettes enter these controls. Outputs append to the same immutable candidate
manifest and remain pending review. Because the earlier line-control version
failed several identity checks, this path requires an explicit separate pilot
and the `--allow-experimental` flag; it is not production-approved.

After every final review row is accepted with all required checks, run the
`finalize` stage. It revalidates image hashes, labels, class identity, perceptual
uniqueness, and review completeness before writing `release_manifest.json`.
