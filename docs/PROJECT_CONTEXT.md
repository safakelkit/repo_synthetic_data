# Project Context

## Project

- **Title:** Synthetic Data Generation for Robust Object Detection
- **Project:** INSP-MOT-DET within MAIJA
- **Researcher:** Safa Kelkit
- **Supervisor:** Prof. Martin Kampel
- **Detector:** pretrained YOLO11s for the complete experiment matrix
- **Current status:** ready to begin the real-only and accepted cut-paste baseline training matrix

## Problem

A detector trained on clean INSP-DET images generalizes poorly to video-derived INSP-MOT-DET images, especially the hard split. The observed domain gap is associated with motion blur, low spatial resolution, lighting/exposure variation, compression/video artifacts, reduced visibility, different object instances, and limited source-domain intra-class diversity.

The primary objective is to increase mAP50-95 on both INSP-MOT-DET easy and
INSP-MOT-DET hard using synthetic training data. Clean INSP-DET performance must
be reported alongside those gains and kept within an acceptable degradation
tolerance.

## Current research questions

1. How does class-balanced cut-paste data affect clean, easy, and hard detection performance as the synthetic-data quantity increases?
2. How do class-balanced Stable Diffusion + ControlNet and Qwen + ControlNet data compare with the real-only and cut-paste baselines at equal quantities?
3. Which non-ADR generator and data quantity provides the best target-domain improvement without unacceptable clean-domain degradation?

ADR effectiveness is a later research question. It must be discussed and decided only after the current baselines are complete.

## Datasets

| Domain | Role | Reported size |
|---|---|---:|
| INSP-DET train | Real source training set | 2,215 images |
| INSP-DET validation | Source validation and official checkpoint selection | 276 images |
| INSP-DET test | Clean source-domain evaluation | 278 images |
| INSP-MOT-DET easy test | Target-domain evaluation | 352 images |
| INSP-MOT-DET hard test | Target-domain evaluation | 457 images |

These counts were verified against the repository on 2026-08-10. The INSP-DET
test split contains 278 image files and 278 matching YOLO label files, with 546
valid annotation rows and no unmatched files. The earlier count of 228 was a
documentation error.

## Classes

The detector has 16 classes in this order:

1. Lighter
2. Matches
3. Scissors
4. Pliers
5. Knife
6. Shaver
7. Hammer
8. Cigarettes
9. Saw
10. Screwdriver
11. Wrench
12. Aerosol can
13. Battery
14. Alcohol
15. Mobile phone
16. Laptop

Treat the repository dataset configuration as the executable source of truth. Report any mismatch before changing class IDs.

## Available object assets

Class-specific objects have been collected with SAM3. Available forms include:

- object masks;
- RGB crops;
- RGBA crops.

The completed audit accepted all 2,750 researcher-cleaned assets. It found no
integrity, alpha/mask agreement, exact-duplicate, or class-provenance failure.

## Active non-ADR experimental pipeline

### Stage 0 - Real-only baseline

Train pretrained YOLO11s on the original 2,215-image INSP-DET training set with no synthetic images. This is experiment `E000` and the reference for every comparison.

### Stage 1 - Cut-paste without ADR

Use the existing cut-paste pipeline to create four class-balanced datasets:

- `CP-B0512`: 2,215 real + 512 cut-paste images;
- `CP-B1024`: 2,215 real + 1,024 cut-paste images;
- `CP-B1536`: 2,215 real + 1,536 cut-paste images;
- `CP-B2048`: 2,215 real + 2,048 cut-paste images.

The generated baseline implements the frozen hard-domain degradation
distribution and records every sampled operation in per-image metadata.

The production placement method is context-aware rather than fixed-band random
placement. Automatically infer candidate support surfaces for each background,
validate the proposals on a deterministic category-stratified sample, and have
a human accept or reject proposed regions before generation. Store accepted
regions in a versioned manifest. Lying objects must fit inside a support region;
upright objects use their bottom alpha-contact region; table-like surfaces use
a support/top boundary rather than arbitrary pixels in the full semantic mask.
The exact segmentation model/revision and class-to-orientation/support mapping
must be frozen before generation and may not be tuned on either target test set.

The frozen placement front-end is `facebook/sam3` at immutable commit
`3c879f39826c281e95690f02c7821c4de09afae7`. A deterministic 30-background
pilot validated v2 proposal deduplication and conservative floor, bed-top, and
dining-table-top geometry. Review accepted 45 regions across 28 backgrounds and
rejected invalid or semantically unsafe regions. The 16-class lying/upright/
asset-preserved policy is versioned, and the generator now reads only accepted
regions. Full-pool SAM3 preprocessing subsequently retained 8,969 proposals
across all 1,166 backgrounds. Geometry-v1 was a superseded development pass;
geometry-v2 full review retained 527 production-eligible
regions (306 bed tops and 221 table tops) across 527 backgrounds. Floor
placement is disabled because 2D masks cannot resolve foreground occlusion or
scene depth. The accepted manifest is now the generator's placement source.
The generated `cp_v1_seed42` images are the accepted simple cut-paste baseline.

The frozen cut-paste degradation-v1 distribution keeps 25% of each class clean
and assigns 37.5% light, 25% medium, and 12.5% heavy corruption. Blur,
downscale-upscale, brightness/contrast, JPEG, and Gaussian sensor noise are
sampled after compositing from the versioned ranges in
`configs/generation/copy_paste_v1.yaml`. These values may not be tuned using
easy or hard test results.

QC-v1 validates every generated image, label, metadata record, hash, subset
manifest, class allocation, and severity allocation. A deterministic 256-image
sample (four per class×severity cell) also receives a documented visual review
and explicit dataset-level research disposition before training.

`cp_v1_seed42` contains 2,048 traceable images and passed all automatic QC-v1
checks. Its stratified review found that many objects do not conform to
support-plane perspective and that upright objects/laptops can have implausible
support-relative scale or contact. Before observing detector results, the
researcher accepted these artifacts as limitations of the simple cut-paste
baseline. The dataset may be used for the fixed experiment matrix; the target
test results may not be used to revise this generator and rerun the comparison.

Copy-paste production uses only the 527 geometry-v2-reviewed backgrounds. This
pool combines the researcher's earlier background inspection with the complete
608-candidate contact-sheet review. No target-class instance was confirmed in
the accepted pool; very small or occluded instances remain a documented visual-
review limitation and are checked again during QC-v1.

### Stage 2 - Stable Diffusion + ControlNet without ADR

Create four new class-balanced datasets:

- `SD-B0512`;
- `SD-B1024`;
- `SD-B1536`;
- `SD-B2048`.

Use ControlNet to control object placement as specified by the research plan. Freeze and record the exact model, prompts, conditioning, placement, annotation, seed, and quality-control configuration.

### Stage 3 - Qwen + ControlNet without ADR

Create four new class-balanced datasets:

- `QW-B0512`;
- `QW-B1024`;
- `QW-B1536`;
- `QW-B2048`.

Use ControlNet to control object placement as specified by the research plan. Verify the exact technical integration before full generation and record the same provenance and quality information required for Stable Diffusion.

### Meaning of without ADR

Without ADR means that generation uses no detector-feedback mechanism. Each of
the 16 classes receives exactly the same number of primary-target images. The
four budgets contain 32, 64, 96, and 128 images per class. Each shuffled block
of 16 contains one image from every class, and all smaller datasets are nested
prefixes of the canonical 2,048-image dataset.

## Active experiment matrix

| ID | Generator | ADR | Synthetic images | Total training images |
|---|---|---|---:|---:|
| E000 | None | No | 0 | 2,215 |
| CP-B0512 | Cut-paste | No | 512 | 2,727 |
| CP-B1024 | Cut-paste | No | 1,024 | 3,239 |
| CP-B1536 | Cut-paste | No | 1,536 | 3,751 |
| CP-B2048 | Cut-paste | No | 2,048 | 4,263 |
| SD-B0512 | Stable Diffusion + ControlNet | No | 512 | 2,727 |
| SD-B1024 | Stable Diffusion + ControlNet | No | 1,024 | 3,239 |
| SD-B1536 | Stable Diffusion + ControlNet | No | 1,536 | 3,751 |
| SD-B2048 | Stable Diffusion + ControlNet | No | 2,048 | 4,263 |
| QW-B0512 | Qwen + ControlNet | No | 512 | 2,727 |
| QW-B1024 | Qwen + ControlNet | No | 1,024 | 3,239 |
| QW-B1536 | Qwen + ControlNet | No | 1,536 | 3,751 |
| QW-B2048 | Qwen + ControlNet | No | 2,048 | 4,263 |

This is 13 active configurations per seed.

## Evaluation protocol

- Official checkpoint: Ultralytics `best.pt`, selected only by INSP-DET validation.
- Training environment: Ultralytics 8.4.46, checked by the training preflight before every run.
- Primary metric: mAP50-95.
- Supporting metrics: mAP50, precision, recall, and class-wise AP.
- Evaluate official checkpoints on INSP-DET test, INSP-MOT-DET easy test, and INSP-MOT-DET hard test.
- Do not use target test results to select checkpoints, prompts, generator settings, stopping criteria, or preferred runs during the active non-ADR phase.
- Compare equal synthetic quantities using equivalent initialization, seed policy, training budgets, and evaluation settings.
- Report clean-domain performance alongside all target-domain improvements.
- Use multiple seeds for central claims when computationally feasible.

## Deferred ADR phase

ADR is not part of the current implementation or experiment queue. Its inactive
implementation was removed from the active source tree after verification that
commit `8d44582` preserves the deleted files. Keep only the following proposed
score in documentation for later discussion:

```text
difficulty(c) = alpha * (1 - S_hard(c))
              + beta  * (S_clean(c) - S_hard(c))
              + gamma * (S_clean(c) - S_easy(c))
```

The metric definitions, weights, normalization, probability conversion, validation/feedback data, and scientific evaluation protocol remain unresolved. Do not implement or run ADR until all current baseline experiments are complete and the researcher explicitly approves the next phase.

VLM-based failure analysis, scene understanding, and condition-aware ADR remain out of scope. Synthetic generators may still create blur, low resolution, lighting variation, compression, occlusion, and varied backgrounds without claiming that a controller inferred those conditions.

## Reproducibility requirements

Each run must record its experiment ID, configuration, seed, code revision, package versions, dataset identifiers, initialization, training budget, generator configuration, synthetic manifest, selected checkpoint, hardware, and evaluation outputs.

Each generated image must record its primary target class, generator and version, source assets where applicable, prompt/conditioning where applicable, transformations, placement information, seed, annotation method, and quality-control status.

Do not claim that planned components are implemented until repository inspection and a verified run confirm them.
Use `METHODOLOGY_TRACEABILITY.md` to map paper statements to exact values,
source functions, configs, and evidence status.
