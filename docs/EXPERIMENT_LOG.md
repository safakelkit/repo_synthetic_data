# Experiment Log

Add one entry per verified run. Do not overwrite earlier entries. Use `TBD` for missing facts and link raw artifacts instead of pasting terminal output. ADR fields are intentionally omitted from the active template because ADR is deferred.

## Active experiment registry

| ID | Generator | Synthetic count | Status |
|---|---|---:|---|
| E000 | None | 0 | Planned |
| CP-B0512 | Cut-paste | 512 | Planned |
| CP-B1024 | Cut-paste | 1,024 | Planned |
| CP-B1536 | Cut-paste | 1,536 | Planned |
| CP-B2048 | Cut-paste | 2,048 | Planned |
| SD-B0512 | Stable Diffusion + ControlNet | 512 | Planned |
| SD-B1024 | Stable Diffusion + ControlNet | 1,024 | Planned |
| SD-B1536 | Stable Diffusion + ControlNet | 1,536 | Planned |
| SD-B2048 | Stable Diffusion + ControlNet | 2,048 | Planned |
| QW-B0512 | Qwen + ControlNet | 512 | Planned |
| QW-B1024 | Qwen + ControlNet | 1,024 | Planned |
| QW-B1536 | Qwen + ControlNet | 1,536 | Planned |
| QW-B2048 | Qwen + ControlNet | 2,048 | Planned |

## E000 - Real-only baseline

- **Status:** Planned / Running / Complete / Invalid
- **Date:** TBD
- **Code revision:** TBD
- **Config:** `configs/train_baseline.yaml`
- **Seed:** 0 (initial matrix)
- **Model and initialization:** pretrained `yolo11s.pt`
- **Training data:** INSP-DET real only
- **Training budget:** 60 epochs, image size 640, batch 16
- **Checkpoint rule:** source-validation `best.pt`
- **Selected checkpoint:** TBD
- **INSP-DET mAP50-95:** TBD
- **INSP-MOT-DET easy mAP50-95:** TBD
- **INSP-MOT-DET hard mAP50-95:** TBD
- **Supporting/class-wise results:** TBD
- **Artifacts:** TBD
- **Main observation:** TBD
- **Problems or validity concerns:** TBD
- **Next action:** TBD

---

## Baseline experiment entry template

### EXPERIMENT_ID - Short descriptive name

- **Status:** Planned / Running / Complete / Invalid
- **Date:** YYYY-MM-DD
- **Code revision:** TBD
- **Training config:** `configs/train_baseline.yaml`
- **Detector seed:** 0 for the initial matrix
- **Generator seed:** 42 for synthetic generators unless superseded by a versioned config
- **Model and initialization:** pretrained `yolo11s.pt`
- **Generator:** None / Cut-paste / Stable Diffusion + ControlNet / Qwen + ControlNet
- **ADR:** No
- **Allocation:** None / Class-balanced
- **Real image count:** 2,215
- **Synthetic image count:** TBD
- **Total training image count:** TBD
- **Generator/config version:** TBD
- **Placement method:** semantic support masks with human verification
- **Segmentation model/revision:** TBD
- **Support-region manifest/checksum:** TBD
- **Class orientation/support policy version:** TBD
- **Support-mask pilot/QC evidence:** TBD
- **Synthetic manifest:** TBD
- **Class-allocation manifest:** TBD
- **Annotation/QC evidence:** TBD
- **Training budget:** 60 epochs, image size 640, batch 16
- **Checkpoint rule:** source-validation `best.pt`
- **Evaluation settings:** clean/easy/hard test; confidence 0.001; NMS IoU 0.7; max detections 300
- **Selected checkpoint:** TBD
- **INSP-DET mAP50-95:** TBD
- **INSP-MOT-DET easy mAP50-95:** TBD
- **INSP-MOT-DET hard mAP50-95:** TBD
- **mAP50, precision, recall, and class-wise results:** TBD
- **Clean-domain change versus E000:** TBD
- **Artifacts:** TBD
- **Main observation:** TBD
- **Problems or validity concerns:** TBD
- **Next action:** TBD

## Deferred ADR note

Do not add ADR runs to the active registry until all baseline experiments are complete and the researcher approves a finalized ADR protocol. The proposed score is preserved in `DECISIONS.md` and `PROJECT_CONTEXT.md`.

## Pre-generation placement pilot

- **Pilot ID:** SP-SAM3-P01
- **Status:** Complete; revision required before approval
- **Purpose:** Evaluate semantic support-region proposals; generates no detector-training images
- **Model:** `facebook/sam3`
- **Exact model revision:** `3c879f39826c281e95690f02c7821c4de09afae7`
- **License:** SAM License
- **Sampling:** 10 deterministic backgrounds per category; seed 42
- **Config:** `configs/placement/support_masks_sam3_v1.yaml`
- **Implementation:** `src/placement/propose_support_masks_sam3.py`
- **GPU/environment:** NVIDIA GeForce RTX 3090; Python 3.10.12; Torch 2.7.1+cu118; Transformers 5.5.4; OpenCV 4.13.0
- **Manifest:** `data/processed/background_support_masks/sam3_v1/pilot/support_region_proposals.csv`; SHA-256 `e1a4d939ade27424ac3445163820be8d98edd605ef8d1bbad8c639187b32ca6f`
- **Outputs:** 178 proposed masks, 39 no-proposal rows, 30 overlays, and 3 category contact sheets; integrity checks passed
- **Technical review:** Revise prompts and add duplicate suppression; do not approve full-pool preprocessing
- **Researcher decision:** Accept v1 revision finding; proceed with a clean v2 pilot
- **Artifact retention:** Local v1 masks/manifest/visualizations deleted by researcher decision after evidence was recorded; no v1 artifact is active

## SP-SAM3-P02 - Revised pre-generation placement pilot

- **Status:** Complete; semantic front-end promising, raw anchor masks rejected
- **Model revision:** `3c879f39826c281e95690f02c7821c4de09afae7`
- **Sampling:** Same deterministic 10 backgrounds per category as v1; seed 42
- **Revision:** Score-ordered within-prompt mask-IoU suppression at provisional 0.85; explicit bed/table/desk/nightstand surface prompts
- **Config:** `configs/placement/support_masks_sam3_v2.yaml`
- **Implementation:** `src/placement/propose_support_masks_sam3.py`
- **Outputs:** 259 retained masks, 30 no-proposal rows, 30 overlays, 3 contact sheets
- **Deduplication:** 275 raw area-filtered proposals; 16 removed; no retained within-prompt pair above mask IoU 0.85
- **Manifest:** `data/processed/background_support_masks/sam3_v2/pilot/support_region_proposals.csv`; SHA-256 `ff1c3db49b87898f85f29f21cf576c9dd38117a95c1d54a13201d4c9ae94e8ad`
- **Technical review:** Use SAM3 proposals as input to deterministic support-plane geometry; do not use raw masks directly for anchors
- **Researcher decision:** Pending
