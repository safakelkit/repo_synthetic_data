# Project Context

## Objective

The INSP-MOT-DET project within MAIJA evaluates whether class-balanced
synthetic data can improve pretrained YOLO11s mAP50-95 on both video-derived
INSP-MOT-DET easy and hard without unacceptable loss on clean INSP-DET.

Active non-ADR generators:

1. context-constrained cut-paste;
2. Stable Diffusion XL + Canny ControlNet full-scene generation;
3. Qwen-Image + ControlNet full-scene generation.

ADR and detector-feedback-driven generation are deferred until all baselines
are complete.

## Data and taxonomy

| Split | Role | Images | Boxes |
|---|---|---:|---:|
| INSP-DET train | real training | 2,215 | 4,435 |
| INSP-DET validation | `best.pt` selection | 276 | 530 |
| INSP-DET test | clean reporting | 278 | 546 |
| INSP-MOT-DET easy test | target reporting | 352 | 750 |
| INSP-MOT-DET hard test | target reporting | 457 | 1,172 |

The fixed class order is: Lighter, Matches, Scissors, Pliers, Knife, Shaver,
Hammer, Cigarettes, Saw, Screwdriver, Wrench, Aerosol can, Battery, Alcohol,
Mobile phone, Laptop. Class IDs 0--15 may not be renamed, merged, reordered, or
removed. Hard has no Lighter annotations, so hard class-0 AP is unavailable.

## Frozen detector protocol

- pretrained YOLO11s only;
- 60 epochs, image size 640, global batch 16;
- detector seed 0 and deterministic training for the initial matrix;
- AdamW and all augmentation values frozen in `configs/train_baseline.yaml`;
- source-validation `best.pt` is the only official checkpoint;
- clean, easy, and hard tests are reporting-only;
- one independent single-GPU run per RTX 3090, no DDP.

Exact optimizer, augmentation, environment, and evaluation values are recorded
once in `METHODOLOGY_TRACEABILITY.md`.

## Experiment matrix

Every generator uses nested class-balanced budgets of 512, 1,024, 1,536, and
2,048 synthetic images: 32, 64, 96, and 128 primary targets per class. Together
with E000 this gives 13 configurations per detector seed.

| Prefix | Generator | Status |
|---|---|---|
| E000 | real only | complete, seed 0 |
| CP-B* | cut-paste | four quantities complete, seed 0 |
| SD-B* | SDXL + ControlNet | canonical data not generated |
| QW-B* | Qwen + ControlNet | canonical data not generated |

## Cut-paste baseline

The accepted canonical dataset contains 2,048 images stored once, with four
nested manifests. It uses all 2,750 audited SAM3 object assets, class-specific
INSP-DET-train size distributions, and 527 human-reviewed bed/table support
regions. Its fixed degradation mix is 25% clean, 37.5% light, 25% medium, and
12.5% heavy. Automatic QC passed; the researcher accepted its visible
perspective, contact, orientation, and scale limitations before detector
results were observed.

The seed-0 cut-paste matrix improved both target splits at every quantity. The
response was non-monotonic: CP-B1536 was strongest on hard, CP-B2048 on easy,
and CP-B1024 best preserved clean performance. Exact results and hashes are in
`RESULTS_SUMMARY.md` and `EXPERIMENT_LOG.md`.

## Full-scene GenAI baseline

Both the background and target are newly generated; real background or RGBA
object pixels are not composited into the output. Binary SAM3 silhouettes may
provide non-photographic ControlNet geometry. Eight frozen MAIJA-aligned
correctional-facility scene families provide four compatible contexts per
class while sharing contexts across classes to reduce background shortcuts.

SDXL and Qwen passed single-image integration feasibility. SDXL then completed
two all-class and three targeted diagnostics. Their positive findings now form
one class-specific candidate pipeline; earlier strict pilot pass rates are
diagnostic history, not the active acceptance threshold. Qwen all-class work is
paused until the shared annotation/finalization path is proven with SDXL. No
pilot output is eligible for detector training.

The accepted SDXL candidate method combines the useful class-specific pilot
settings, deterministic scene/layout diversity, and a compact annotation-first
quality gate. A 64-image acceptance test is implemented and must be reviewed
before production. Production may then generate a modest surplus and retain
only recognizable, reliably annotatable images. Multiple instances of the same
class are allowed when all are annotated. The four dataset quantities
refer to accepted images, while generation attempts, acceptance rate, and cost
must also be reported.

## Current stage

1. Run and review the 64-image SDXL canonical acceptance test.
2. Preflight post-generation annotation and finalization.
3. Generate, annotate, degrade, and validate the 2,048-image SDXL canonical
   dataset and nested prefixes.
4. Train/evaluate SD-B*; then repeat the frozen process for Qwen.

## Validity boundaries

- Easy/hard results cannot tune prompts, models, controls, QC thresholds,
  checkpoints, stopping, or dataset acceptance.
- A requested class or expected ControlNet region is not automatically a label.
- Additional visible project-class objects must be annotated or the image
  rejected.
- Report clean-domain cost beside easy/hard gains and all valid configurations,
  not only the best result.
- Initial results use one detector seed; variance claims require additional
  predeclared seeds.
- Do not claim planned components as implemented without code and verified run
  evidence.
