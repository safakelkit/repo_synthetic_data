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
- The frozen-protocol E000 real-only YOLO11s seed-0 run is complete: clean/easy/hard test mAP50-95 is 0.6884/0.4143/0.1110.
- The complete seed-0 cut-paste quantity matrix is valid and complete. All four quantities improve both target splits; CP-B1536 is best on hard (0.1618), CP-B2048 is best on easy (0.4707), and CP-B1024 best preserves clean performance (0.6867).
- The 2,048-image `cp_v1_seed42` canonical dataset passed complete automatic QC and was accepted by the researcher as a simple context-constrained cut-paste baseline with documented visual limitations.
- Context-aware placement is implemented and reviewed over the full pool. Geometry-v2 retains 527 accepted regions across 527 backgrounds (306 bed tops and 221 table tops); floor placement is disabled because 2D masks cannot model foreground occlusion or scene depth.
- Training selects `best.pt` using INSP-DET validation.
- Official evaluation uses `best.pt` on the INSP-DET, INSP-MOT-DET easy, and INSP-MOT-DET hard test splits.
- E000 and all four accepted cut-paste quantity runs are complete. The next phase is a full-scene GenAI baseline in which Stable Diffusion and Qwen generate both the MAIJA-aligned background and the target object. Both exact model pairs completed v2 feasibility; active v3 replaces the semantically weak hand-drawn target proxy with class-matched binary SAM3 silhouettes. Annotation, QC, and canonical generation remain open.

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

# One copy-paste run (example: 512 images)
python src/training/train_copypaste_baselines.py --experiment 512 --preflight-only
python src/training/train_copypaste_baselines.py --experiment 512

# Full sequential overnight matrix: train 512 -> 1024 -> 1536 -> 2048;
# evaluate best.pt and render plots after each run.
python src/training/train_copypaste_baselines.py --experiment all --evaluate --preflight-only
python src/training/train_copypaste_baselines.py --experiment all --evaluate

# Evaluate a selected source-validation checkpoint on all test domains
python src/evaluate_yolo.py runs/train/<run-name>/weights/best.pt

# Plot one saved evaluation JSON
python src/analysis/plot_results.py runs/evaluation/<run-name>_results.json
```

Each run name is unique and an existing run directory causes an error instead
of silently creating a suffixed or nested folder.

Every training entry point verifies the pinned Ultralytics version, model and
dataset paths, exact expected image/label count, duplicate paths, CUDA
visibility, selected device, clean Git revision, and unused run directory
before starting. With `CUDA_VISIBLE_DEVICES=<physical-index>`, the config's
`device: 0` refers to that single exposed GPU.

The sequential matrix is fail-fast: an error stops later runs and is recorded
in `runs/evaluation/copy_paste_matrix_status.json`. Existing training or
evaluation outputs are never silently overwritten.

## GenAI baseline status

The accepted GenAI direction is complete-scene generation rather than reuse of
the cut-paste backgrounds or RGBA composites. The 16-class taxonomy remains
fixed. The frozen v1 policy is grounded in MAIJA's correctional-facility and
detention-room-search scope. It contains eight scene families spanning
detention living spaces, controlled property inspection, and supervised operational
spaces. Every class is assigned four semantically compatible families. A
property-inspection scene is shared by all classes, while the remaining scenes
overlap across several classes to limit background shortcuts.

A one-image feasibility runner validates the exact SDXL and Qwen model pairs
on the same Canny condition without creating training data. V2 proved both
model integrations, but its hand-drawn scissors proxy remained semantically
weak. Active v3 computes Canny from an accepted class-matched SAM3 binary mask,
without transferring source RGB/RGBA pixels, and uses control scale 0.8. Run
and review both v3 outputs before
expanding to the deterministic all-class prompt/control and SAM3 annotation/QC
pilot. Canonical 2,048-image generation remains blocked.

Canonical controls will not reuse this one feasibility layout. Every generated
sample must use a distinct deterministic combination of scene geometry,
camera, target placement/scale/orientation, lighting, clutter, prompt wording,
and seed. After clean generation and annotation, each backend receives the
same frozen degradation-v1 distribution and ranges used by cut-paste.

```bash
python src/generation/run_full_scene_feasibility.py --help
```

The runner is intentionally limited to one local image per backend and refuses
a dirty Git worktree or an existing output directory. Install the pinned GenAI
requirements, run `--preflight-only`, and then launch SDXL and Qwen manually on
an available GPU. Run the two first measurements sequentially to avoid mixing
CPU-offload, download, and storage contention in the evidence.

## Copy-paste release status

`data/synthetic/cp_v1_seed42` contains a traceable 2,048-image dataset from
commit `6c14f12`. All automatic QC checks passed, including exact class/severity
balance, labels, hashes, manifests, and duplicate checks. The 256-image
class-by-severity review found frequent orientation, perspective,
support-depth-scale, and contact-integration artifacts. Before seeing detector
results, the researcher accepted these as limitations of the deliberately
simple cut-paste baseline. The dataset is approved for the fixed Phase 1 matrix;
its generator must not be tuned using easy/hard test results.
