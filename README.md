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
- The 2,048-image `cp_v1_seed42` canonical dataset passed complete automatic QC and was accepted by the researcher as a simple context-constrained cut-paste baseline with documented visual limitations.
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
src/extraction/          SAM3 object-bank construction and audit
src/placement/           Reproducible support-mask preprocessing
src/training/            Training entry points
src/validation/          Dataset QC and review-sheet rendering
src/evaluate_yolo.py     Three-domain evaluation
```

The preprocessing scripts remain in the repository because their large outputs
under `data/` are Git-ignored and must be reproducible. The obsolete GrabCut
object-extraction path has been removed; the active object bank is SAM3-based.

## Phase 1 commands

Run commands from the repository root with the main environment activated.
Expensive generation, preprocessing, and training commands are launched manually
by the researcher after their stated release/preflight checks pass.

```bash
# Full SAM3 proposals, geometry-v2 derivation, and review have completed. Do
# not rerun them unless creating a new version.

# The canonical dataset already exists; do not regenerate it for this matrix.
# Re-run its automatic QC only when integrity needs to be reconfirmed.
python src/validation/validate_copypaste_dataset.py

# Real-only baseline
python src/training/train.py --preflight-only
python src/training/train.py

# Copy-paste runs are launched individually (example: 512 images)
python src/training/train_copypaste_baselines.py --experiment 512 --preflight-only
python src/training/train_copypaste_baselines.py --experiment 512

# Evaluate a selected source-validation checkpoint on all test domains
python src/evaluate_yolo.py runs/train/<run-name>/weights/best.pt

# Plot one saved evaluation JSON
python src/analysis/plot_results.py runs/evaluation/<run-name>_results.json
```

Each run name is unique and an existing run directory causes an error instead
of silently creating a suffixed or nested folder.

Every training entry point verifies the pinned Ultralytics version, model and
dataset paths, CUDA visibility, selected device, clean Git revision, and unused
run directory before starting. With `CUDA_VISIBLE_DEVICES=<physical-index>`,
the config's `device: 0` refers to that single exposed GPU.

## Copy-paste release status

`data/synthetic/cp_v1_seed42` contains a traceable 2,048-image dataset from
commit `6c14f12`. All automatic QC checks passed, including exact class/severity
balance, labels, hashes, manifests, and duplicate checks. The 256-image
class-by-severity review found frequent orientation, perspective,
support-depth-scale, and contact-integration artifacts. Before seeing detector
results, the researcher accepted these as limitations of the deliberately
simple cut-paste baseline. The dataset is approved for the fixed Phase 1 matrix;
its generator must not be tuned using easy/hard test results.
