# Synthetic Data Generation for Robust Object Detection

This repository studies whether synthetic training data can reduce the domain
gap from clean INSP-DET images to the video-derived INSP-MOT-DET easy and hard
test sets. The first phase establishes reproducible real-only and non-ADR
synthetic-data baselines. ADR is deferred until those baselines are complete.

The methodological source of truth is [`docs/PROJECT_CONTEXT.md`](docs/PROJECT_CONTEXT.md).
Accepted constraints and open decisions are in [`docs/DECISIONS.md`](docs/DECISIONS.md).
Exact paper values and their code/evidence mapping are in
[`docs/METHODOLOGY_TRACEABILITY.md`](docs/METHODOLOGY_TRACEABILITY.md).

## Current status

- Dataset paths, counts, labels, and class order have been audited.
- The active detector is configured as pretrained YOLO11s for 60 epochs with seed 0; no frozen-protocol run is complete yet.
- A planned canonical 2,048-image copy-paste dataset will supply exactly balanced 512/1,024/1,536/2,048 subsets through text manifests; it has not been generated yet.
- Context-aware placement is implemented and reviewed over the full pool. Geometry-v2 retains 527 accepted regions across 527 backgrounds (306 bed tops and 221 table tops); floor placement is disabled because 2D masks cannot model foreground occlusion or scene depth.
- Training selects `best.pt` using INSP-DET validation.
- Official evaluation uses `best.pt` on the INSP-DET, INSP-MOT-DET easy, and INSP-MOT-DET hard test splits.
- No current run satisfies the complete Phase 1 protocol yet.

## Repository layout

```text
configs/                 Dataset and training configurations
data/raw/                INSP datasets (local, ignored by Git)
data/processed/          SAM3 object bank (local, ignored by Git)
data/backgrounds/        Copy-paste backgrounds (local, ignored by Git)
data/synthetic/          Canonical generated datasets and manifests
docs/                    Research context, decisions, TODO, and results
runs/train/              Training outputs (generated, ignored by Git)
runs/evaluation/         Evaluation outputs (generated, ignored by Git)
src/augmentation/        Copy-paste generation
src/extraction/          Object-bank extraction
src/training/            Training entry points
src/evaluate_yolo.py     Three-domain evaluation
```

## Phase 1 commands

Run commands from the repository root with the main environment activated.
Expensive generation, preprocessing, and training commands are launched manually
by the researcher after their stated release/preflight checks pass.

```bash
# Full SAM3 proposals, geometry-v2 derivation, and review have completed. Do
# not rerun them unless creating a new version.

# Generate the canonical 2,048-image dataset and nested manifests.
# Run only after the reviewed changes are committed and the explicit release
# switch is approved.
python src/augmentation/generate_copypaste_dataset.py

# Real-only baseline
python src/training/train.py

# Copy-paste runs at 512, 1,024, 1,536, and 2,048 images
python src/training/train_copypaste_baselines.py

# Evaluate a selected source-validation checkpoint on all test domains
python src/evaluate_yolo.py runs/train/<run-name>/weights/best.pt
```

Each run name is unique and an existing run directory causes an error instead
of silently creating a suffixed or nested folder.

## Pre-generation status

No current copy-paste set exists. Placement and its reviewed support manifest
are complete; degradation-v1 and QC-v1 are frozen and implemented. Production
is restricted to 527 reviewed backgrounds. Generated-dataset evidence does not
exist yet. The release switch is approved, but generation still requires a
committed clean worktree and an explicit researcher-launched command.
