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
- Context-aware placement using automatically proposed, human-verified semantic support surfaces is accepted. SAM3 pilot v2 completed: semantic proposals are promising, but raw masks are not approved as final anchor regions. Deterministic support-geometry postprocessing is next.
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
# Run the revised v2 support-mask pilot on 10 deterministic backgrounds per category.
# This does not generate copy-paste images; the researcher launches it manually.
python src/placement/propose_support_masks_sam3.py --mode pilot --gpu 0

# Generate the canonical 2,048-image dataset and nested manifests.
# This remains blocked until all generation/QC release gates pass.
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

## Known blockers before expensive runs

No current copy-paste set exists. SAM3 pilot v1 was technically reviewed and
its local outputs were intentionally removed after recording the evidence. The
revised v2 pilot has run, but raw masks still require support-plane extraction
and anchor-overlay validation; the final placement policy and full manifest
are therefore not approved. QC specifications and exact
degradation operations/ranges also remain unresolved. The background pool
requires a target-class leakage audit. Production is intentionally blocked
until these requirements are resolved.
