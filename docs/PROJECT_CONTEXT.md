# Project Context

## Objective

The INSP-MOT-DET project within MAIJA evaluates whether class-balanced
synthetic data can improve pretrained YOLO11s mAP50-95 on both video-derived
INSP-MOT-DET easy and hard without unacceptable loss on clean INSP-DET.

Active non-ADR generators:

1. context-constrained cut-paste;
2. Stable Diffusion XL + Canny ControlNet full-scene generation;
3. Qwen-Image + ControlNet full-scene generation;
4. the planned SDXL foreground/background-separated full-synthetic follow-up.

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
| SDXL-M* | SDXL + ControlNet, mixed degradation | four quantities complete, seed 0 |
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

The final SDXL pipeline uses generated structured background plates,
class-specific source initialization and target-only Canny geometry, including
the validated upright spray-can treatment for Aerosol. Canonical clean generation produced
2,048 images and pose-derived YOLO labels. The paired mixed dataset applies
full-frame degradation without changing annotation geometry. Researcher review
and independent integrity/visibility QC approved both manifests for training.
Scene contexts overlap across classes to reduce background shortcuts. Qwen
model feasibility is established, but its canonical work remains pending.

## Current stage

1. Preserve the complete SDXL-M seed-0 result matrix without test-driven
   generator retuning.
2. Implement the predeclared full-synthetic SDXL follow-up: 2,048 unique
   generated backgrounds paired one-to-one with 2,048 generated isolated
   foregrounds, followed by mask-derived compositing and local harmonization.
3. Treat the real-background/real-object SDXL harmonization pipeline only as a
   separately named hybrid ablation; it does not replace the executed SDXL
   baseline or qualify as fully generated data.
4. Complete the planned Qwen baseline under the same detector protocol.
5. Decide additional detector seeds and the numerical clean-loss tolerance.

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
