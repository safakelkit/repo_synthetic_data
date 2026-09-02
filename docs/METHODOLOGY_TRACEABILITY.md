# Methodology and Code Traceability

This file maps paper-relevant methodological values to executable
configuration, implementation, and evidence. A value marked **planned** or
**provisional** must not be described as completed.

## Current evidence boundary

- **Implemented and verified:** dataset inventory, class order, SAM3 asset
  audit, source-only object-size analysis, balanced class schedule, manifest
  design, rooted outputs, and explicit training settings.
- **Implemented and evidenced:** cut-paste size sampling, annotation writing,
  duplicate rejection, provenance, degradation scheduling, and automatic QC on
  the 2,048-image `cp_v1_seed42` candidate.
- **Implemented and full-pool reviewed:** SAM3 v2 proposals and geometry-v2
  processed all 1,166 backgrounds. The generator consumes only 527 accepted
  bed/table regions; floor placement is disabled.
- **Generated and accepted with limitations:** QC-v1 automatic checks passed;
  the 256-image review documented systemic realism artifacts, which the
  researcher accepted as limitations of the simple cut-paste baseline before
  observing detector results.
- **Completed under the frozen protocol:** E000 and all four cut-paste quantity
  runs, including three-domain evaluation and combined response plots.
- **GenAI preparation complete:** one paired 2,048-record schedule and a
  deterministic 32-record initialization/mask/control preview.
- **Not yet run:** Stable Diffusion or Qwen model inference, GenAI annotation,
  GenAI dataset release, and the eight GenAI detector runs.

## Dataset and evaluation values

| Item | Frozen or verified value | Executable source | Evidence/status |
|---|---|---|---|
| Classes | 16, IDs 0--15 | `configs/data_insp.yaml` | Verified across dataset YAMLs |
| INSP-DET train | 2,215 images, 4,435 boxes | `configs/data_insp.yaml` | Verified locally |
| INSP-DET validation | 276 images, 530 boxes | `configs/data_insp.yaml` | Used for `best.pt` selection |
| INSP-DET test | 278 images, 546 boxes | `configs/data_insp.yaml` | Earlier 228 count corrected |
| INSP-MOT-DET easy test | 352 images, 750 boxes | `configs/data_insp_mot_easy.yaml` | Evaluation only |
| INSP-MOT-DET hard test | 457 images, 1,172 boxes | `configs/data_insp_mot_hard.yaml` | Evaluation only; class 0 absent |
| Official checkpoint | source-validation `best.pt` | `src/training/train.py::train_yolo` | Accepted |
| Evaluation domains | clean, easy, hard test | `src/evaluate_yolo.py::DATASETS` | Implemented |
| Primary metric | mAP50-95 | `src/evaluate_yolo.py::extract_overall_metrics` | Accepted |
| Supporting metrics | mAP50, precision, recall, per-class AP | `src/evaluate_yolo.py` | Implemented |
| Validation thresholds | confidence 0.001, NMS IoU 0.7, max detections 300 | `src/evaluate_yolo.py::evaluate_single_dataset` | Explicitly frozen |

Target easy/hard data do not participate in checkpoint selection, object-size
statistics, generation tuning, or run acceptance.

## Detector training values

The environment is pinned to Ultralytics 8.4.46 in `requirements.txt`. These
values were checked against that installed version before the first run and are explicit in
`configs/train_baseline.yaml`; `src/training/train.py::train_yolo` passes them.

| Group | Values |
|---|---|
| Model/budget | pretrained `yolo11s.pt`; image size 640; 60 epochs; batch 16 |
| Reproducibility | detector seed 0; deterministic true; workers 8 |
| Optimization | AdamW; lr0 0.0005; lrf 0.01; beta1 (`momentum`) 0.9; weight decay 0.0005 |
| Warmup/loss | warmup 3.0 epochs; warmup momentum 0.8; bias LR 0.0; box 7.5; cls 0.5; dfl 1.5 |
| Runtime | AMP true; cache false; rectangular batches false; multi-scale 0.0; patience 100 |
| Color | hsv_h 0.015; hsv_s 0.7; hsv_v 0.4; BGR swap 0.0 |
| Geometry | rotation 0.0; translation 0.1; scale 0.5; shear 0.0; perspective 0.0 |
| Flips | vertical 0.0; horizontal 0.5 |
| Mixing | mosaic 1.0, disabled for final 10 epochs; MixUp 0.0; CutMix 0.0; Ultralytics copy-paste 0.0 |

Albumentations is not installed in the main environment; optional
Albumentations blur or grayscale transforms must not be claimed.

Ultralytics `optimizer=auto` would ignore configured `lr0` and
`momentum`. For this 16-class, 60-epoch matrix it resolves to AdamW with
lr0 0.0005, beta1 0.9, and warmup bias LR 0.0. Those effective values are now
explicit, preventing dataset quantity or future library changes from silently
changing the optimizer.

YOLO11s is fixed for the complete experiment matrix; no detector model-size
ablation is planned. This keeps model capacity constant across generator and
synthetic-quantity comparisons.

## Cut-paste values and sources

The versioned source of truth is `configs/generation/copy_paste_v1.yaml`; the
entry point is `src/augmentation/generate_copypaste_dataset.py`. There is no
separate cut-paste pilot: the default command creates the canonical dataset once
the production gate is approved. It also requires a committed, clean Git
worktree so the recorded revision identifies the exact source used.

| Decision/value | Value | Implementation |
|---|---|---|
| Generator seed | 42, separate from detector seed | `generate_dataset` |
| Canonical release | 2,048 images, 128 per class | generation config `production` |
| Nested subsets | 512/1,024/1,536/2,048 = 32/64/96/128 per class | `build_balanced_class_schedule`, `create_subset_manifests` |
| Storage | one image/label copy; four prefix manifests | `create_subset_manifests` |
| Objects per image | one primary object | generation config `allocation` |
| Object assets | all 2,750 researcher-cleaned SAM3 RGBA assets eligible | `collect_object_bank`; audit manifest |
| Asset reuse | deterministic shuffled cycles with derived seed 43; no within-class reuse until its pool is exhausted | `build_balanced_asset_schedule` |
| Size source | all 4,435 INSP-DET train boxes; target data excluded | `src/analysis/analyze_object_sizes.py` |
| Size rule | observed class template within p10--p90 normalized box area | `load_size_templates` |
| Resize rule | background-relative; aspect preserved; each dimension <= 90% | `resize_rgba_to_target_area` |
| Placement family | automatic semantic support masks with human verification | Decision D017; generation config |
| First mask-model candidate | `facebook/sam3`; SAM License; HF commit `3c879f39826c281e95690f02c7821c4de09afae7` | run summary; support-mask config/script |
| Placement pilot sample | 10 deterministic backgrounds from each of bedroom, dining room, and hotel room; seed 42 | completed 2026-08-11; 30 backgrounds |
| Mask inference thresholds | score 0.30; mask 0.50; minimum area ratio 0.001; within-prompt IoU suppression 0.85 | `configs/placement/support_masks_sam3_v2.yaml`; pilot-approved for full proposals |
| Placement geometry | largest-component bed-top/table-top regions; floor disabled; footprint containment for lying objects; bottom contact for upright/asset-preserved objects | `support_geometry_v2.yaml`, full decisions v2, orientation policy v2; full-pool reviewed and generator-integrated |
| Annotation | visible alpha box converted to normalized YOLO xywh | `visible_bbox`, `bbox_to_yolo` |
| Output safety | nonempty output rejected; successful write required; exact composite duplicates rejected | `generate_dataset` |
| Provenance | asset/background paths, class, size template, realized area, placement, bbox, SHA-256, seed, QC status | output `metadata.json` |
| Dataset provenance | code revision/dirty state, package versions, input counts and hashes, source config hash | output `generation_config.json` |
| Manifest provenance | root-level portable `./images/...` paths, image count, and SHA-256 for each nested subset | output `manifest_checksums.json` |
| Training dataset gate | exact expected count, existing image/label pairs, zero duplicate paths, Ultralytics-compatible list resolution | `validate_training_dataset()` in `src/training/train.py` |
| Degradation mixture | 25% clean, 37.5% light, 25% medium, 12.5% heavy in every per-class 32-image block | generation config `degradation` |
| Degradation operations | Gaussian/motion blur, downscale-upscale, brightness/contrast, JPEG, Gaussian noise; severity-specific probabilities/ranges | `apply_degradations`; generation config `degradation.levels` |
| Degradation seed | generator seed + 2 = 44 | `build_degradation_schedule`; generation config `seed_offset` |
| Automatic dataset QC | all 2,048 images; paths/hashes/dimensions/labels/duplicates/area/prefix balance | executed `copy_paste_qc_v1.yaml`, active disposition `copy_paste_qc_v1_1.yaml`; `validate_copypaste_dataset.py` |
| Manual dataset QC | 4 images per class×severity cell = 256; complete review plus pre-results dataset-level researcher disposition | QC-v1.1 and generated `manual_review_sample.csv`/`manual_review_summary.json` |
| Eligible background pool | 527 reviewed backgrounds; 306 bed-top and 221 table-top supports | background eligibility v1; reviewed geometry-v2 manifest |

The three available background categories contain 1,166 images: bedroom 382,
dining room 387, and hotel room 397. The former three-band geometric heuristic
was rejected after qualitative inspection and is not approved for production.
The accepted replacement is a precomputed semantic support-region manifest with
review. The frozen proposal model is `facebook/sam3` at commit
`3c879f39826c281e95690f02c7821c4de09afae7`. The reviewed pilot accepted 45
regions across 28 backgrounds, froze the 16-class orientation policy, and
validated generator-side placement. Full-pool geometry-v2 review retained 306
bed-top and 221 table-top regions across 527 backgrounds. All floor regions are
disabled because 2D support masks cannot resolve foreground occlusion or depth.

Production is restricted to 527 backgrounds accepted by the complete support-
candidate review. The researcher had previously inspected the background pool;
the full contact-sheet pass provided a second visual screen and rejected the
confirmed laptop-containing example. This does not guarantee that tiny or
occluded target instances are absent, so that residual limitation must be
reported and checked again in the generated-dataset QC sample.

### Class-specific source area statistics

Percentages are normalized bounding-box area in INSP-DET train. Sampling uses
observed templates inside each class's p10--p90 interval.

| Class | Train boxes | p10 % | Median % | p90 % |
|---|---:|---:|---:|---:|
| Lighter | 303 | 0.16 | 0.86 | 7.45 |
| Matches | 91 | 0.60 | 2.26 | 6.93 |
| Scissors | 456 | 0.32 | 1.79 | 10.81 |
| Pliers | 166 | 0.53 | 2.13 | 12.83 |
| Knife | 451 | 0.58 | 2.41 | 11.16 |
| Shaver | 206 | 0.37 | 2.16 | 8.49 |
| Hammer | 97 | 1.28 | 4.34 | 34.07 |
| Cigarettes | 297 | 0.50 | 1.92 | 10.34 |
| Saw | 101 | 0.69 | 3.83 | 13.85 |
| Screwdriver | 348 | 0.25 | 1.40 | 11.12 |
| Wrench | 159 | 0.39 | 2.73 | 14.19 |
| Aerosol can | 469 | 0.38 | 2.39 | 9.79 |
| Battery | 379 | 0.05 | 0.41 | 2.52 |
| Alcohol | 397 | 0.55 | 2.61 | 12.68 |
| Mobile phone | 253 | 0.34 | 1.66 | 11.91 |
| Laptop | 262 | 2.56 | 11.68 | 38.28 |

## Object-bank audit evidence

`src/extraction/audit_object_bank.py` produced the local, Git-ignored evidence
under `data/processed/object_bank_sam3`. All 2,750 RGB/mask/RGBA triplets were
readable, dimension-matched, binary-mask consistent with alpha (IoU 1.0), and
free of exact RGBA duplicates. Direct metadata matched 1,564 assets; provenance
for 1,186 was reconstructed and class-verified. A deterministic 160-asset
visual sample produced 146 passes and 14 observations. Per the researcher's
manual cleaning decision, observations do not exclude assets.

## GenAI paired-plan values and sources

These values define the accepted shared experiment design. Backend model IDs
and numerical inference values remain provisional until smoke-test acceptance.

| Item | Current value | Executable source | Evidence/status |
|---|---|---|---|
| Shared records | 2,048; exactly 128/class | `genai_shared_v1.yaml`; `prepare_genai_plan.py` | Verified |
| Nested prefixes | 512/1,024/1,536/2,048; 32/64/96/128 per class | same | Verified |
| Backend pilot | same 32 records/backend; two/class | same | Verified selection |
| Sample pairing | same class, RGBA asset, background, support type, placement, and degradation assignment | CP metadata transformed by `prepare_genai_plan.py` | Verified hashes/paths |
| Initialization | undegraded RGBA composite reconstructed at the frozen intended box | `render_genai_conditioning.py::reconstruct_initialization` | 32 previews rendered |
| Inpaint region | intended box expanded 25% on each axis, minimum 4 source pixels | `expand_bbox`; shared config | Verified in preview |
| Inference resolution | 1,024 x 1,024 | shared config | Provisional pending model smoke |
| SD candidate seeds | `142000 + image_id` | shared config | Planned |
| Qwen candidate seeds | `242000 + image_id` | shared config | Planned |
| Prompt enhancement | disabled | backend configs | Planned |
| SD candidate | SDXL 1.0 base in the Diffusers ControlNet inpaint pipeline + SDXL Canny ControlNet | `sdxl_controlnet_v1.yaml` | Candidate; revisions unresolved |
| Qwen candidate | Qwen-Image + InstantX inpainting ControlNet | `qwen_controlnet_v1.yaml` | Candidate; community ControlNet; revisions unresolved |
| Final annotation | post-generation SAM3 target mask; intended bbox cannot be copied as label | shared config | Designed, not implemented |
| Degradation | reuse exact CP severity/operation assignment after generation/annotation acceptance | shared config | Designed, not implemented |

The initial composite is a control/reference input, not a GenAI output and not
an additional experiment. Stable Diffusion's candidate uses Canny edges of this
reference; the candidate Qwen pipeline uses its native image-plus-mask control.
The record identities are paired, but the backend-specific conditioning tensor
is not falsely claimed to be identical.

Local evidence (Git-ignored):

- `data/processed/genai_v1/shared_conditioning_plan_summary.json`;
- `data/processed/genai_v1/shared_conditioning_plan.jsonl`;
- `data/processed/genai_v1/pilot_image_ids.json`;
- `data/processed/genai_v1/conditioning_preview/conditioning_summary.json`;
- 32 preview panels under `conditioning_preview/sample_*/preview.jpg`.

Technical sources: Diffusers SDXL ControlNet inpainting API
(https://huggingface.co/docs/diffusers/main/api/pipelines/controlnet_sdxl),
official Qwen-Image model card (https://huggingface.co/Qwen/Qwen-Image), and
the InstantX community inpainting ControlNet model card
(https://huggingface.co/InstantX/Qwen-Image-ControlNet-Inpainting).

The GenAI dependency specification follows Diffusers 0.40.0 package metadata:
`huggingface-hub>=1.23.0,<2.0` and `safetensors>=0.8.0,<1.0`. These ranges also
satisfy pinned Transformers 5.5.4. Exact versions resolved in a run are
recorded by backend preflight rather than inferred from compatibility ranges.

## Generated dataset and experiment evidence

- `cp_v1_seed42` was generated from clean commit `6c14f12` with 2,048 unique
  images, 128 per class, and exact per-class `32/48/32/16`
  clean/light/medium/heavy counts. Automatic QC passed.
- The deterministic 256-image review found generally aligned labels and visibly
  distinct degradation tiers. Lying assets preserve
  their original 2D pose rather than conforming to the support plane; upright
  objects and laptops frequently show implausible scale or contact; perspective
  and contact shadows are absent. These are documented baseline limitations,
  not object-bank exclusions or automatic-integrity failures.
- The candidate used 497 unique backgrounds; 1,532/2,048 placements used
  `bed_top` and 516 used `dining_table_top`. This concentration and the lack of
  support-depth-conditioned scale are reported as methodological limitations.
- Acceptable clean-domain mAP50-95 decrease remains `TBD`.
- Detector seeds beyond the seed-0 initial matrix remain `TBD`.
- Stable Diffusion, Qwen, and ControlNet candidate configurations exist, but
  immutable revisions, memory behavior, inference values, and acceptance remain
  unresolved until smoke tests.
- The retained SAM3 asset metadata proves source images, classes, annotation
  indices, and SAM scores, but does not prove the exact model revision or full
  extraction command used for every historical crop. The current extraction
  source defaults (`facebook/sam3`, score 0.30, mask 0.50, minimum mask-area
  ratio 0.01) are code defaults, not yet recoverable run evidence.
- `src/background.py` names `ljnlonoljpiljm/places365-256px` as the background
  source and targets 500 images per configured category, but the historical
  download revision is not recorded. Generated metadata therefore records the
  SHA-256 of each background actually used.
- A paper run must record a committed code revision, environment/hardware,
  manifest checksums, checkpoint path, and evaluation JSON.
- The verified training environment uses Ultralytics 8.4.46 and has no broken
  package requirements. Training preflight enforces this exact version.

The seed-0 cut-paste matrix completed at code revision `f72caff`. Exact input
counts were 2,727/3,239/3,751/4,263 with zero corrupt samples. Relative to E000,
all four quantities improved easy by +0.041279/+0.045278/+0.040799/+0.056419
and hard by +0.016194/+0.020221/+0.050797/+0.035265, while clean changed by
-0.010331/-0.001737/-0.002357/-0.013097. The source evidence is the four
evaluation JSON files and checkpoint hashes in `EXPERIMENT_LOG.md`; the combined
paper-candidate evidence is `copy_paste_matrix_summary.csv` and `.png`. These
are single-detector-seed findings, not variance or significance estimates.

`cp_v1_seed42` is accepted for CP-B0512--CP-B2048 as a simple cut-paste
baseline. No realism-driven regeneration will be performed in this matrix. Any
future improved generator must receive a new version and cannot be selected or
tuned using easy/hard test results from the current matrix.

The four CP runs were launched in fixed ascending order by
`train_copypaste_baselines.py --experiment all --evaluate`. Each run is trained,
evaluated, and plotted before the next begins. The orchestration is fail-fast
and writes an ignored progress record; it never branches or changes parameters
based on observed metrics.

### Placement literature context

- Ghiasi et al., *Simple Copy-Paste is a Strong Data Augmentation Method for
  Instance Segmentation*, CVPR 2021:
  https://openaccess.thecvf.com/content/CVPR2021/html/Ghiasi_Simple_Copy-Paste_Is_a_Strong_Data_Augmentation_Method_for_Instance_CVPR_2021_paper.html
- Dvornik et al., *Modeling Visual Context is Key to Augmenting Object
  Detection Datasets*, ECCV 2018: https://arxiv.org/abs/1807.07428

These works motivate copy-paste and context-aware placement. The project's
semantic support-mask preprocessing and human region verification are a
reproducible engineering design, not a literature-mandated standard step.

SAM3 candidate source: https://huggingface.co/facebook/sam3 . The repository
records its dedicated SAM License from
https://huggingface.co/facebook/sam3/blob/main/LICENSE .
