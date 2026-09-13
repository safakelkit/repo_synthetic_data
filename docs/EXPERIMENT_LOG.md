# Experiment Log

Add one entry per verified run. Do not overwrite earlier entries. Use `TBD` for missing facts and link raw artifacts instead of pasting terminal output. ADR fields are intentionally omitted from the active template because ADR is deferred.

## Active experiment registry

| ID | Generator | Synthetic count | Status |
|---|---|---:|---|
| E000 | None | 0 | Complete |
| CP-B0512 | Cut-paste | 512 | Complete |
| CP-B1024 | Cut-paste | 1,024 | Complete |
| CP-B1536 | Cut-paste | 1,536 | Complete |
| CP-B2048 | Cut-paste | 2,048 | Complete |
| SDXL-M0512 | SDXL + ControlNet, mixed degradation | 512 | Complete |
| SDXL-M1024 | SDXL + ControlNet, mixed degradation | 1,024 | Complete |
| SDXL-M1536 | SDXL + ControlNet, mixed degradation | 1,536 | Complete |
| SDXL-M2048 | SDXL + ControlNet, mixed degradation | 2,048 | Complete |
| QW-B0512 | Qwen + ControlNet | 512 | Planned |
| QW-B1024 | Qwen + ControlNet | 1,024 | Planned |
| QW-B1536 | Qwen + ControlNet | 1,536 | Planned |
| QW-B2048 | Qwen + ControlNet | 2,048 | Planned |

## E000 - Real-only baseline

- **Status:** Complete
- **Date:** 2026-09-02
- **Code revision:** `30531abd3ee84a791348627d5e7daf7d4f535d66`
- **Config:** `configs/train_baseline.yaml`
- **Seed:** 0 (initial matrix)
- **Model and initialization:** pretrained `yolo11s.pt`
- **Training data:** INSP-DET real only
- **Training budget:** 60 epochs, image size 640, batch 16
- **Checkpoint rule:** source-validation `best.pt`
- **Selected checkpoint:** `runs/train/real_only_yolo11s_seed0/weights/best.pt`
- **Checkpoint SHA-256:** `9fcc2ca4ef62caf06bb1de76ace07ad41d622cfe41ccc1b68bd730d311392e82`
- **INSP-DET mAP50-95:** 0.688407
- **INSP-MOT-DET easy mAP50-95:** 0.414322
- **INSP-MOT-DET hard mAP50-95:** 0.111021
- **Supporting overall results (precision / recall / mAP50):** clean 0.882184 / 0.792164 / 0.845998; easy 0.755445 / 0.509286 / 0.536149; hard 0.265208 / 0.133092 / 0.137603
- **Source-validation result:** epoch 60; precision 0.86009; recall 0.74789; mAP50 0.79945; mAP50-95 0.65733
- **Training duration:** 2,111.68 seconds (0.587 hours)
- **Evaluation artifacts:** `runs/evaluation/baseline/E000_results.json`; `runs/evaluation/baseline/real_only_yolo11s_seed0/`; `runs/evaluation/baseline/E000_results_plots/`
- **Evaluation JSON SHA-256:** `4b0a0d8ed13c736030042317b0a8657038d92034614cafdcc2f14cb27e962783`
- **Main observation:** Relative to clean mAP50-95, easy drops 0.274086 (39.8%) and hard drops 0.577386 (83.9%), confirming the target-domain gap.
- **Class-wise observation:** Easy is strongest for Alcohol (0.7522), Shaver (0.7479), and Pliers (0.7143). Hard is strongest for Laptop (0.5327), Alcohol (0.3239), and Aerosol can (0.2330); Matches, Knife, and Shaver are 0.0. Hard Lighter is unavailable because the split has no class-0 annotations.
- **Problems or validity concerns:** None observed. Target test results are reporting evidence and must not be used to retune the frozen cut-paste generator.
- **Next action:** Use E000 as the fixed reference for the completed cut-paste matrix and future GenAI baselines.

## Verified cut-paste matrix — detector seed 0

All four runs used code revision `f72caff2f068aacebaeb0078b0b0e1461512bcea`,
pretrained YOLO11s, generator seed 42, 60 epochs, image size 640, batch 16,
and source-validation-selected `best.pt`. Ultralytics loaded the exact expected
2,727/3,239/3,751/4,263 training images with zero corrupt samples. The matrix
ran on one NVIDIA GeForce RTX 3090 using Ultralytics 8.4.46 and Torch
2.7.1+cu118. Results below are single-seed evidence.

| ID | Best epoch | Val mAP50-95 | Clean | Delta clean | Easy | Delta easy | Hard | Delta hard | Total seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CP-B0512 | 55 | 0.655150 | 0.678076 | -0.010331 | 0.455601 | +0.041279 | 0.127215 | +0.016194 | 2,208.64 |
| CP-B1024 | 54 | 0.682050 | 0.686670 | -0.001737 | 0.459600 | +0.045278 | 0.131242 | +0.020221 | 2,215.96 |
| CP-B1536 | 59 | 0.677100 | 0.686050 | -0.002357 | 0.455120 | +0.040799 | 0.161818 | +0.050797 | 2,262.81 |
| CP-B2048 | 49 | 0.665610 | 0.675311 | -0.013097 | 0.470741 | +0.056419 | 0.146286 | +0.035265 | 2,346.82 |

Checkpoint and evaluation evidence:

| ID | `best.pt` SHA-256 | Evaluation JSON SHA-256 |
|---|---|---|
| CP-B0512 | `8b0339e5b13197ba8f71556bd5d03ebb47f117f39e665e8259a7c33170a28dae` | `10f5b355d19a233c351ecbb2dbc3902eb64fcd75018cee52f1eb981b24c5545d` |
| CP-B1024 | `83643105bad327fbebce0aeba89e3728d63ff0f74e063bcf94ea8dda74ae317d` | `991eb361159d928cfdaeb1f34275583e047e59ee9d2e0e5bb887b531ece94977` |
| CP-B1536 | `475b0d0912b8877b0020c72235648172d5e5f2c7ed581e81f4d5d0711250db24` | `b231ffd6c65da93e74697e6c60b49c890294fd0168f976a2ae3a7412ef341e1a` |
| CP-B2048 | `4c2f4da413ecbe5a3460d3ad4051f28ccdab9e071700711e8920796d4d1d9dce` | `9de19abf705238a1497f2ecffa9a2e700c61c64caed1732d98e7f208880dd4dc` |

- **Artifacts:** `runs/train/CP-B*_yolo11s_seed0/`, `runs/evaluation/cut_paste/CP-B*_yolo11s_seed0_results.json`, domain evaluation folders, per-run plots, `runs/evaluation/cut_paste/matrix_summary.csv`, and `matrix_summary.png`.
- **Validity:** Matrix status is `complete`; every expected dataset scan reports zero corrupt samples; the log contains no traceback, runtime error, CUDA OOM, or NaN.
- **Main finding:** Every cut-paste quantity improves both target splits while decreasing clean mAP50-95. CP-B1024 gives the smallest clean loss; CP-B1536 gives the highest hard result; CP-B2048 gives the highest easy result. Quantity response is not monotonic.
- **Restriction:** These target-test results must not be used to revise `cp_v1_seed42` and rerun the same comparison. Additional detector seeds are required before strong variability or significance claims.

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
- **Generation mode:** None / Cut-paste composite / Complete-scene generation
- **Scene taxonomy and class-to-scene allocation:** TBD or N/A
- **Prompt and spatial-conditioning manifest/checksum:** TBD or N/A
- **Generation model(s), immutable revisions, and licenses:** TBD or N/A
- **Placement/support policy:** Cut-paste support-region version or GenAI
  spatial-control version; TBD or N/A
- **Segmentation/annotation model and revision:** TBD or N/A
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

## Cut-paste matrix execution rule

CP-B0512, CP-B1024, CP-B1536, and CP-B2048 execute sequentially in ascending
order. After each training run, `best.pt` is evaluated on clean/easy/hard and
plots are rendered before the next run begins. The fixed pipeline is fail-fast;
progress and errors are written to the Git-ignored
`runs/evaluation/cut_paste/matrix_status.json`. Metrics never alter later runs.

### Invalid CP-B0512 launch (2026-09-02)

- **Status:** Invalid and interrupted during epoch 2; no result is reportable.
- **Cause:** Nested manifest entries `./../images/...` were mis-expanded by Ultralytics 8.4.46. The loader marked all 512 synthetic images corrupt and retained only 2,215 real images (139 batches at batch size 16).
- **Detection:** The terminal batch count differed from the expected 171 batches and the training log reported `2215 images ... 512 corrupt`.
- **Correction:** Manifests now live at the dataset root with `./images/...` entries. Preflight requires exact expected image/label counts and rejects missing or duplicate paths.
- **Restriction:** The interrupted checkpoint, metrics, and plots must not be included in any comparison or paper result.

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
- **Researcher decision:** Accepted geometry-postprocessing plan; geometry-v1 executed

## SP-GEOM-P01 - Support-geometry postprocessing pilot

- **Status:** Complete and reviewed; full-pool preprocessing approved
- **Date:** 2026-08-28
- **Input:** Retained SAM3 v2 proposals on the same deterministic 30 backgrounds
- **Config:** `configs/placement/support_geometry_v1.yaml`
- **Implementation:** `src/placement/derive_support_geometry.py`
- **Outputs:** 60 manifest rows; 24 floor, 17 bed-top, and 9 dining-table-top candidate regions
- **Traceability:** raw geometry manifest SHA-256 `9111b3124e1bb16e7d21fe2b91d6db445ea3a350dc0c3f601f646a5607de4331`; reviewed manifest SHA-256 `883d666b97e826c50fe6df9d6915bedc100605e8c86128978fbfe1d18033b4ce`
- **Artifacts:** `data/processed/background_support_masks/sam3_v2/geometry_v1/` (Git-ignored)
- **Safety:** No SAM3 inference, copy-paste image, synthetic dataset, or detector training was produced
- **Review:** Accepted 24 floor, 12 bed-top, and 9 dining-table-top regions; rejected every no-valid-region row and five unsafe bed-top candidates
- **Next action:** Run the frozen proposal and geometry pipeline on all 1,166 backgrounds, then review the full manifest

## SP-SAM3-F01 - Full-background support proposals

- **Status:** Complete; integrity verified
- **Date:** 2026-08-29
- **Backgrounds:** 382 bedroom, 387 dining room, 397 hotel room; 1,166 total
- **Model:** `facebook/sam3` commit `3c879f39826c281e95690f02c7821c4de09afae7`; RTX 3090
- **Outputs:** 9,296 area-filtered proposals, 327 duplicates removed, 8,969 retained; 10,298 manifest rows
- **Manifest SHA-256:** `bcb346c5d74741dc7c6c16b521149d334d7849431d4af3c1645d61b09a5bba01`
- **Validity:** No copy-paste image or detector run was produced; all mask files and checksums passed

## SP-GEOM-F01 - Full-background geometry derivation (superseded)

- **Status:** Complete; superseded by largest-component geometry-v2
- **Date:** 2026-08-29
- **Input:** Verified SP-SAM3-F01 manifest
- **Outputs:** 2,332 decision rows; 911 floor, 625 bed-top, and 334 dining-table-top regions derived (1,870 total)
- **Manifest SHA-256:** `3751f84e27e428b61845c20d8f4884cf71a0f31669d1067b36cb33bea5ab0979`
- **Integrity:** Every generated region file, pixel count, and checksum passed
- **Visual finding:** Stratified overlays are generally useful; occasional false bed/bench/side-furniture regions require rejection
- **Disposition:** Preserved as development evidence; not used by the generator

## SP-GEOM-F02 - Full-background geometry-v2 review

- **Status:** Complete; accepted for cut-paste placement
- **Date:** 2026-09-01
- **Change:** Keep only the largest connected component and disable floor placement because 2D masks do not model foreground occlusion/depth
- **Derived regions:** 904 floor, 622 bed-top, 331 dining-table-top
- **Review protocol:** Conservative numeric triage; reject all risk-group rows; inspect every one of the 608 automatic bed/table candidates; reject explicit visual failures
- **Visual candidate failures:** bed 75/381 (19.7%), table 6/227 (2.6%), combined 81/608 (13.3%)
- **Accepted production pool:** 306 bed-top and 221 table-top regions across 527 backgrounds
- **Reviewed manifest SHA-256:** `758ed5959fcd40fd838e98be6c1b8beeb0a173bd56b7e47aee0fa1d7bfd0c702`
- **Safety:** No copy-paste image or detector training was produced

## CP-DEG-V1 - Frozen cut-paste degradation implementation

- **Status:** Implemented and evidenced in accepted `cp_v1_seed42`
- **Date:** 2026-09-01
- **Distribution per class/32:** 8 clean, 12 light, 8 medium, 4 heavy
- **Operations:** Gaussian/motion blur, downscale-upscale resolution loss, brightness/contrast, JPEG compression, and Gaussian sensor noise
- **Application:** Complete composite, after placement and before image encoding
- **Seed:** 44 (generator seed 42 + offset 2)
- **Verification:** Exact severity counts passed at every 32/64/96/128 per-class prefix; repeated calls with the same seed produced identical pixels and metadata
- **Evidence boundary:** Scheduling and transforms behaved as specified; visual compositing limitations are reported separately

## CP-QC-V1 - First canonical-candidate validation

- **Status:** Automatic pass; visual limitations reviewed and accepted for baseline use
- **Date:** 2026-09-01
- **Automatic coverage:** All 2,048 images plus labels, metadata, hashes, duplicates, geometry consistency, manifests, and class×severity prefix balance
- **Manual coverage:** 256 deterministic images, four per class×severity cell
- **Release rule:** Automatic pass plus complete stratified review and an explicit dataset-level researcher disposition before training
- **Candidate:** `data/synthetic/cp_v1_seed42`; generated from commit `6c14f12012f3c6c46be89b91dd87095f5730b08e`
- **Automatic result:** Pass; 2,048 unique images, 128 per class, exact `32/48/32/16` severity allocation per class
- **Manual finding:** Labels and degradation scheduling are generally sound; orientation/perspective, support-depth scale, and contact integration remain visible limitations, especially for long tools, aerosol/alcohol, and laptops.
- **Disposition:** Researcher accepted the dataset as a simple cut-paste baseline before detector results; approved for the fixed CP matrix without generator retuning

## CP-BG-V1 - Reviewed production-background eligibility

- **Status:** Complete
- **Date:** 2026-09-01
- **Input:** Full geometry-v2 candidate review plus prior researcher inspection
- **Eligible:** 527 backgrounds (306 bed-top, 221 table-top)
- **Manifest SHA-256:** `c15243e14a888d284c84e0bce66d46998f14437ac8a3eb573c712bb3e8161f09`
- **Finding:** No visible target instance confirmed in accepted backgrounds; small/occluded-instance risk remains a stated limitation
- **Safety:** No synthetic image or detector run was produced

## Verified SDXL mixed-degradation matrix — detector seed 0

- **Status:** Complete
- **Date:** 2026-09-09
- **Training code revision:** `bff18a34859364c4f851a511d282fbc0787c89cc`
- **Generator:** Canonical SDXL structured-background, source-initialized target-only Canny pipeline
- **Canonical data:** 2,048 images, 128/class, 1024x1024 RGB; nested 512/1,024/1,536/2,048 manifests
- **Degradation:** 512 clean, 768 light, 512 medium, 256 heavy; exact 32/48/32/16 per class
- **Dataset QC:** 4,096 clean/mixed files decoded; zero label/pair errors; 512 clean pairs unchanged; 1,536 degraded pairs changed; researcher and independent review passed
- **Detector:** pretrained YOLO11s; 60 epochs; image size 640; batch 16; detector seed 0; deterministic; frozen `configs/train_baseline.yaml`
- **Runtime:** physical GPU 1 exposed as logical CUDA device 0; NVIDIA GeForce RTX 3090; Ultralytics 8.4.46; Torch 2.7.1+cu118
- **UTC interval:** 2026-09-09T15:13:47.126728+00:00 to 2026-09-09T18:01:16.729623+00:00

| ID | Best epoch | Val mAP50-95 | Clean | Delta clean | Easy | Delta easy | Hard | Delta hard | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SDXL-M0512 | 60 | 0.678520 | 0.697123 | +0.008716 | 0.409061 | -0.005261 | 0.121801 | +0.010779 | 2,257.88 |
| SDXL-M1024 | 58 | 0.650530 | 0.684565 | -0.003843 | 0.420388 | +0.006066 | 0.118238 | +0.007217 | 2,360.33 |
| SDXL-M1536 | 60 | 0.658960 | 0.673075 | -0.015333 | 0.433494 | +0.019172 | 0.099994 | -0.011027 | 2,687.29 |
| SDXL-M2048 | 60 | 0.665780 | 0.664612 | -0.023796 | 0.428149 | +0.013827 | 0.090928 | -0.020093 | 2,744.09 |

| ID | `best.pt` SHA-256 | Evaluation JSON SHA-256 |
|---|---|---|
| SDXL-M0512 | `487fe13da0570f903ef5a40e0abffa5716c29291e752f99a4d7f31c3b3cbd754` | `5e048aced83812f4ed1f35adfcbbcad11fc1c5aa7a3fb1c4cf644e74ebb81369` |
| SDXL-M1024 | `ba6044a77ad820f451dd1229783c449a232dd2340976b0113b5918acee1af895` | `74f873072b14ec25833109fc2eae776d1b088f23aa334cf7d1dd77a22147be25` |
| SDXL-M1536 | `84fd32cec374df77afc9734e445ae31eed4d9229b074dcef04bf72b1d50d04b9` | `8bc2c5df13591f9c79779409cebdf5db2f9cf6074433792d828fed2fee572ccc` |
| SDXL-M2048 | `330461fa72ee671c935e8d49670c0c8adb2380a4947eec747d6a0acbaf9c5f19` | `76ef27adfdfc45075f4f6cf587eec3004e8ba4f92812c13bf39c802ed149ff88` |

- **Main finding:** SDXL quantity response is adverse beyond 512 for clean and hard. Easy peaks at 1,536 but remains below equal-size cut-paste. Only SDXL-M0512 improves both clean and hard over E000; none of the four improves both easy and hard while preserving clean.
- **Aerosol:** Every SDXL quantity exceeds E000 for Aerosol mAP50-95 on clean, easy, and hard. The best SDXL Aerosol values are clean 0.723839 (M1536), easy 0.678787 (M1536), and hard 0.280044 (M1024).
- **Persistent failures:** Matches, Pliers, Shaver, and Battery remain near zero on hard, as they already did under E000. This is a cross-method hard-domain failure rather than evidence limited to SDXL generation.
- **Artifacts:** `runs/train/SDXL-M*_yolo11s_seed0/`, `runs/evaluation/sdxl/mixed_degradation/`, and `runs/evaluation/comparisons/`.
- **Validity:** Single detector seed. Target results are report-only and may not retune the executed generator.

### Evaluation artifact organization (2026-09-10)

Evaluation artifacts were grouped under `baseline/`, `cut_paste/`,
`sdxl/mixed_degradation/`, and `comparisons/`. JSON `eval_project` fields and
matrix result paths were updated to the new locations, so the path-containing
E000 and CP evaluation JSON hashes changed without changing any metric. The
current hashes shown above are post-organization hashes; checkpoint hashes are
unchanged. Future runners write directly into the grouped layout.

## SDXL-HYB-P01 - Real-pixel localized harmonization ablation pilot

- **Status:** 16-class pilot visually accepted; full production stopped and not released
- **Date:** 2026-09-13
- **Method:** Real reviewed room background plus real INSP-DET training object,
  exact alpha-mask placement, target-only Canny, and localized SDXL inpainting
- **Pilot:** 16 images, one per class; object identity, organic background,
  physical contact, annotation box, target count, and artifact checks passed
- **Input candidate:** 512 balanced initializations; 32 distinct originating
  train images per class and 480 unique backgrounds in the final regenerated
  candidate; three visually unsafe support regions were excluded
- **Boundary:** This is `CP+SDXL harmonization`, not fully generated data. It is
  training-forbidden and does not replace the executed CP or SDXL baselines.
- **Stopped run:** A 512-image harmonization attempt was intentionally interrupted
  after 44 completed outputs, before a manifest or release, when the method was
  reclassified as an ablation and D031 became the active direction.
- **Target-test use:** None; pilot decisions used source-side visual QC only.
