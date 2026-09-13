# Synthetic Data Generation for Robust Object Detection

This repository studies whether class-balanced synthetic training data improves
YOLO11s mAP50-95 on both INSP-MOT-DET easy and hard while limiting performance
loss on clean INSP-DET. The active comparison is real-only versus cut-paste,
Stable Diffusion + ControlNet, and Qwen + ControlNet. ADR is deferred.

## Current status

- E000, four cut-paste runs, and four mixed-degradation SDXL runs are complete
  for detector seed 0.
- The accepted SDXL dataset contains 2,048 class-balanced 1024x1024 images and
  nested 512/1,024/1,536/2,048 manifests. Its permanent degradation schedule is
  25% clean, 37.5% light, 25% medium, and 12.5% heavy.
- SDXL-M0512 is the only SDXL quantity that improves both clean and hard over
  E000. Increasing SDXL quantity reduces clean and hard performance; easy peaks
  at SDXL-M1536 but remains below every cut-paste run.
- SDXL improves Aerosol can across all three domains at several quantities, but
  does not resolve the hard-domain collapse for Matches, Pliers, Shaver, and
  Battery. Qwen remains pending.
- A prospective SDXL-FB2048 follow-up is now specified: generate 2,048 unique
  backgrounds and 2,048 isolated foreground objects from scratch, then use
  retained masks for one-to-one composition, localized harmonization, and exact
  annotation. It is planned work, not an executed result. The real-pixel hybrid
  harmonizer remains a separately named ablation.

| Run | Synthetic | Clean | Easy | Hard |
|---|---:|---:|---:|---:|
| E000 | 0 | 0.6884 | 0.4143 | 0.1110 |
| CP-B0512 | 512 | 0.6781 | 0.4556 | 0.1272 |
| CP-B1024 | 1,024 | 0.6867 | 0.4596 | 0.1312 |
| CP-B1536 | 1,536 | 0.6861 | 0.4551 | 0.1618 |
| CP-B2048 | 2,048 | 0.6753 | 0.4707 | 0.1463 |
| SDXL-M0512 | 512 | 0.6971 | 0.4091 | 0.1218 |
| SDXL-M1024 | 1,024 | 0.6846 | 0.4204 | 0.1182 |
| SDXL-M1536 | 1,536 | 0.6731 | 0.4335 | 0.1000 |
| SDXL-M2048 | 2,048 | 0.6646 | 0.4281 | 0.0909 |

## Documentation

Start with [docs/INDEX.md](docs/INDEX.md). In brief:

- `PROJECT_CONTEXT.md`: stable research scope and current stage;
- `TODO.md`: only active work and major completed milestones;
- `DECISIONS.md`: accepted methodology decisions;
- `METHODOLOGY_TRACEABILITY.md`: exact paper values and code/evidence mapping;
- `EXPERIMENT_LOG.md`: immutable run and pilot history;
- `RESULTS_SUMMARY.md`: verified detector results;
- `PAPER_OUTLINE.md`: manuscript structure.

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

# Preflight the degradation-only SDXL training matrix
python src/training/train_sdxl_baselines.py --experiment mixed-all --preflight-only --evaluate
```

Exact commands, versions, hashes, and artifact paths belong in the traceability
and experiment records rather than being duplicated here.
