# Synthetic Data Generation for Robust Object Detection

This repository studies whether class-balanced synthetic training data improves
YOLO11s mAP50-95 on both INSP-MOT-DET easy and hard while limiting performance
loss on clean INSP-DET. The active comparison is real-only versus cut-paste,
Stable Diffusion + ControlNet, and Qwen + ControlNet. ADR is deferred.

## Current status

- E000 and the four seed-0 cut-paste runs are complete.
- Every cut-paste quantity improved both target splits over E000.
- The 2,048-image cut-paste dataset is accepted as a deliberately simple
  baseline with documented realism limitations.
- The GenAI pipeline has been restarted around one-pass full-image generation.
  Fifteen successful SDXL/Canny class profiles are retained; Aerosol is tested
  as a concise text-only full-scene branch. Production remains locked.
- No GenAI pilot or acceptance-test image is training data. Canonical SDXL
  production, annotation, degradation, QC, and detector training have not started.

| Run | Synthetic | Clean | Easy | Hard |
|---|---:|---:|---:|---:|
| E000 | 0 | 0.6884 | 0.4143 | 0.1110 |
| CP-B0512 | 512 | 0.6781 | 0.4556 | 0.1272 |
| CP-B1024 | 1,024 | 0.6867 | 0.4596 | 0.1312 |
| CP-B1536 | 1,536 | 0.6861 | 0.4551 | 0.1618 |
| CP-B2048 | 2,048 | 0.6753 | 0.4707 | 0.1463 |

## Documentation

Start with [docs/INDEX.md](docs/INDEX.md). In brief:

- `PROJECT_CONTEXT.md`: stable research scope and current stage;
- `TODO.md`: only active work and major completed milestones;
- `DECISIONS.md`: accepted methodology decisions;
- `METHODOLOGY_TRACEABILITY.md`: exact paper values and code/evidence mapping;
- `EXPERIMENT_LOG.md`: immutable run and pilot history;
- `RESULTS_SUMMARY.md`: verified detector results;
- `PAPER_OUTLINE.md`: manuscript structure;
- `PAPER_TECHNICAL_RECORD_PRIVATE.md`: comprehensive local paper source,
  intentionally ignored by Git.

## Repository layout

```text
configs/                 Dataset, generation, placement, and training configs
data/raw/                INSP datasets (local, ignored)
data/processed/          Object bank and preprocessing outputs (local, ignored)
data/backgrounds/        Cut-paste backgrounds (local, ignored)
data/synthetic/          Generated datasets and pilots (local, ignored)
docs/                    Research records
runs/                    Training/evaluation artifacts (local, ignored)
src/augmentation/        Cut-paste generation
src/generation/          Full-scene GenAI generation
src/placement/           Support-mask preprocessing
src/training/            Detector training
src/validation/          Dataset QC
src/evaluate_yolo.py     Three-domain evaluation
```

## Reproducibility rules

- Pretrained YOLO11s, 60 epochs, image size 640, batch 16, detector seed 0.
- Generator seed and detector seed are separate random processes.
- INSP-DET validation selects `best.pt`; clean/easy/hard test splits are for
  reporting and never tune generation or checkpoint selection.
- Synthetic quantities are nested, exactly class-balanced
  512/1,024/1,536/2,048-image sets.
- Expensive generation and training start only after preflight and are launched
  manually by the researcher on a selected RTX 3090.
- Existing output directories are never silently overwritten.

## Useful commands

Run from the repository root with the project environment active.

```bash
# Validate the existing canonical cut-paste dataset
python src/validation/validate_copypaste_dataset.py

# Training/evaluation preflight examples
python src/training/train.py --preflight-only
python src/training/train_copypaste_baselines.py --experiment 512 --preflight-only

# Evaluate a source-validation-selected checkpoint
python src/evaluate_yolo.py runs/train/<run-name>/weights/best.pt

# Preflight the restarted one-pass SDXL acceptance test
python src/generation/generate_sdxl_dataset.py --preflight-only
```

Exact commands, versions, hashes, and artifact paths belong in the traceability
and experiment records rather than being duplicated here.
