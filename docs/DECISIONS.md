# Methodology Decisions

This file records decisions that Codex must preserve unless the researcher explicitly revises them.

## D001 - Current phase excludes ADR implementation

- **Status:** Accepted
- **Date:** 2026-08-10
- **Decision:** Complete all non-ADR baseline experiments before discussing or implementing ADR.
- **Current scope:** real-only, cut-paste, Stable Diffusion + ControlNet, and Qwen + ControlNet experiments.
- **Deferred scope:** ADR design, validation protocol, implementation, and experiments.

## D002 - Active experiment matrix

- **Status:** Accepted
- **Decision:** Run one real-only baseline and four synthetic-data quantities for each of three non-ADR generators.
- **Synthetic quantities:** 512, 1,024, 1,536, and 2,048 images.
- **Generators:** existing cut-paste pipeline, Stable Diffusion + ControlNet, and Qwen + ControlNet.
- **Total active configurations:** 13 per seed: one real-only plus 12 synthetic configurations.

## D003 - Non-ADR generation is class-balanced

- **Status:** Accepted
- **Decision:** Generate exactly the same number of primary-target images for each of the 16 classes, with no model-feedback mechanism.
- **Allocation:** Use budgets divisible by 16: 32, 64, 96, and 128 primary-target images per class, producing totals of 512, 1,024, 1,536, and 2,048.
- **Ordering:** Generate one image per class in each deterministically shuffled block of 16. The smaller quantities are nested prefixes of one canonical 2,048-image dataset.
- **Multi-class images:** The quota refers to the requested primary target class; all visible objects must still be annotated.
- **Background implication:** A background containing an existing target-class object cannot be used as an unlabeled negative; it must be excluded or fully annotated before production.

## D004 - Synthetic images represent the hard target domain

- **Status:** Accepted
- **Decision:** Synthetic images should include target-domain characteristics such as motion/Gaussian blur, resolution loss, lighting/exposure and contrast variation, compression artifacts, partial occlusion, and varied backgrounds.
- **Requirement:** Declare, version, and log the applicable generation or degradation configuration for every experiment.

## D005 - SAM3 object assets are available

- **Status:** Accepted
- **Decision:** The project has class-specific object assets collected with SAM3, including object masks, RGB crops, and RGBA crops.
- **Requirement:** Audit the assets for class identity, file integrity, duplicates, mask quality, and usable provenance before large-scale generation.
- **Researcher validation:** The current object bank was manually cleaned by the researcher before this audit. All assets currently retained in the bank are intentional and remain eligible for generation; automated visual observations do not independently remove assets.

## D006 - Both GenAI pipelines must be evaluated

- **Status:** Accepted
- **Decision:** Stable Diffusion + ControlNet and Qwen + ControlNet are separate experimental baselines. Do not silently select one and omit the other.
- **Placement control:** Both pipelines must use ControlNet for object placement as intended by the research plan. Verify technical compatibility and document the exact model and conditioning implementation before full generation.
- **Pilot rule:** Validate a small pilot for realism, class correctness, placement, annotation quality, duplicates, and leakage before generating full datasets.

## D007 - Official checkpoint uses source validation

- **Status:** Accepted
- **Decision:** Use Ultralytics `best.pt` selected by INSP-DET validation.
- **Reason:** INSP-MOT-DET easy and hard are target test sets, not checkpoint-selection sets during the active non-ADR phase.

## D008 - Metrics and fair comparison

- **Status:** Accepted
- **Primary success criterion:** Improve mAP50-95 on both INSP-MOT-DET easy and INSP-MOT-DET hard relative to E000, while respecting the clean-domain preservation rule in D011.
- **Decision:** Use mAP50-95 as the primary metric and report mAP50, precision, recall, and class-wise AP.
- **Decision:** Evaluate every official checkpoint on INSP-DET test, INSP-MOT-DET easy test, and INSP-MOT-DET hard test.
- **Fairness:** Paired comparisons must use equivalent initialization, seed policy, training budget, evaluation protocol, and synthetic-image count.

## D009 - Remove inactive ADR implementation files

- **Status:** Accepted for repository cleanup
- **Decision:** Remove existing ADR implementation files from the active codebase before baseline reruns because ADR is not part of the current experimental phase.
- **Safety procedure:** First inventory exact ADR-only files and references, preserve the score specification below, create a recoverable version-control checkpoint, verify that no shared baseline utility will be removed, then delete the ADR-only implementation and clean its imports, configuration entries, commands, tests, and documentation references.
- **Do not delete:** baseline code, synthetic generators, shared evaluation utilities, verified experiment results, or this preserved ADR specification.
- **Verification:** Run the baseline pipeline checks after cleanup and report every deleted path.

## D010 - Preserved ADR score for later discussion

- **Status:** Preserved; deferred and not approved for current implementation
- **Decision:** Keep the following proposed class-wise difficulty function in documentation so it is not lost:

```text
difficulty(c) = alpha * (1 - S_hard(c))
              + beta  * (S_clean(c) - S_hard(c))
              + gamma * (S_clean(c) - S_easy(c))
```

- `S_clean(c)`: proposed class-wise score for class `c` on the clean domain.
- `S_easy(c)`: proposed class-wise score for class `c` on the easy domain.
- `S_hard(c)`: proposed class-wise score for class `c` on the hard domain.
- **Unresolved:** exact metric, score scale, weights, normalization, probability conversion, target-feedback protocol, and test-set validity.
- **Restriction:** Codex must not implement or run ADR until the baseline experiments are complete and the researcher explicitly approves a finalized protocol.

## D011 - Clean-domain preservation

- **Status:** Accepted; threshold unresolved
- **Decision:** Always report INSP-DET performance alongside target-domain gains.
- **Open value:** acceptable INSP-DET mAP50-95 decrease relative to E000 = `TBD`.

## D012 - Direct canonical generation and production gate

- **Status:** Accepted and implemented as an execution safeguard
- **Decision:** Do not create a separate 160-image cut-paste pilot. Once the methodology is frozen, the default command creates the canonical 2,048-image dataset and its nested manifests.
- **Decision:** Generation remains blocked in the versioned config until placement, degradation, background-audit, and QC values are approved. The generated canonical set must be validated before detector training.
- **Source:** `configs/generation/copy_paste_v1.yaml` and `src/augmentation/generate_copypaste_dataset.py`.

## D013 - Explicit training and evaluation parameters

- **Status:** Accepted and implemented
- **Decision:** Pin Ultralytics 8.4.41 and write the critical optimizer, augmentation, reproducibility, and evaluation thresholds explicitly rather than relying on changeable library defaults.
- **Source:** `requirements.txt`, `configs/train_baseline.yaml`, `src/training/train.py`, and `src/evaluate_yolo.py`.
- **Paper record:** Exact values and code mappings are maintained in `METHODOLOGY_TRACEABILITY.md`.

## D014 - Freeze the effective optimizer

- **Status:** Accepted and implemented
- **Decision:** Replace `optimizer=auto` with the effective Ultralytics 8.4.41 choice for this matrix: AdamW, initial learning rate 0.0005, beta1 0.9, and warmup bias learning rate 0.0.
- **Reason:** `auto` ignores the YAML `lr0` and `momentum` values and derives them internally. Making the effective values explicit preserves the current behavior while preventing a library or experiment-size change from silently changing optimization.

## D015 - Detector architecture is YOLO11s

- **Status:** Accepted
- **Decision:** Use pretrained YOLO11s for the complete planned experiment matrix. Do not add a YOLO11 model-size ablation.
- **Reason:** Holding detector capacity fixed isolates the effects of generator type and synthetic-data quantity.

## D016 - Preserve clean synthetic images across nested cut-paste budgets

- **Status:** Accepted; operation ranges unresolved
- **Decision:** Assign each class and every nested 32-image allocation block to 25% clean composite, 37.5% light degradation, 25% medium degradation, and 12.5% heavy degradation.
- **Exact per-class allocation per 32 images:** 8 clean, 12 light, 8 medium, and 4 heavy.
- **Reason:** The allocation is exact and nested for CP-B0512 through CP-B2048, preserves a substantial clean synthetic component, and limits the heavy-degradation share.
- **Restriction:** Freeze generic degradation operations and ranges before generation. Do not tune them using easy/hard test performance.

## D017 - Context-aware cut-paste placement uses verified semantic support surfaces

- **Status:** Accepted; implementation and model validation pending
- **Date:** 2026-08-11
- **Decision:** Replace the provisional fixed normalized placement bands with automatically proposed semantic support-surface masks followed by human verification. The fixed-band implementation is not an approved production method.
- **Precomputation:** Infer support masks once for the background pool, store the masks and a versioned manifest, and reuse the accepted regions during deterministic generation. Record the segmentation model, exact revision, software environment, input/background checksum, mask checksum, accepted support classes, and reviewer status.
- **Pilot gate:** Before processing all backgrounds, evaluate the selected segmentation model on a deterministic, category-stratified background sample. Approve full preprocessing only if the proposed bed, floor, table/desk/counter, and other allowed support regions are sufficiently reliable for placement.
- **Placement geometry:** For objects intended to lie on a surface, require the transformed object footprint to fit inside an accepted support region. For upright objects, align the bottom alpha-contact region with a valid support anchor. Table-like objects require a top/support boundary rather than arbitrary pixels from the full table mask.
- **Human role:** Human review validates or rejects automatically proposed support regions; it does not manually choose every generated object coordinate. Generation samples deterministic anchors only from accepted regions.
- **Validity boundary:** Semantic masks constrain plausible 2D location but do not recover full 3D geometry, lighting, shadows, or physical interaction. These limitations must remain part of QC and paper reporting.
- **Restriction:** Select and record the exact segmentation model/revision and freeze the class-to-orientation/support policy before production. Do not tune either decision using INSP-MOT-DET easy or hard test performance.
- **Methodological context:** Random copy-paste is supported by *Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation* (CVPR 2021), while context-aware placement is supported by *Modeling Visual Context is Key to Augmenting Object Detection Datasets* (ECCV 2018). The semantic support-mask workflow is this project's reproducible implementation choice, not a mandatory standardized stage from either paper.
- **References:** https://openaccess.thecvf.com/content/CVPR2021/html/Ghiasi_Simple_Copy-Paste_Is_a_Strong_Data_Augmentation_Method_for_Instance_CVPR_2021_paper.html ; https://arxiv.org/abs/1807.07428
- **First pilot candidate:** Evaluate `facebook/sam3` through the Transformers text-prompt interface on 10 deterministically selected images from each of the three background categories. Resolve `main` to an immutable Hugging Face commit SHA before loading and record that SHA; never report mutable `main` as the model revision.
- **Candidate prompts:** bedroom/hotel room: `floor`, `bed`, `tabletop`, `desk surface`, `nightstand top`; dining room: `floor`, `tabletop`, `countertop`. These prompts and thresholds are pilot candidates, not frozen production values.
- **Pilot outputs:** Store each proposal mask, its score/box/checksum, no-proposal rows, combined overlays, category contact sheets, environment/model metadata, and a human `pending/accepted/rejected` field. Full background preprocessing remains code-blocked until pilot approval.
- **Model sources/license:** Official model card: https://huggingface.co/facebook/sam3 ; license: https://huggingface.co/facebook/sam3/blob/main/LICENSE
- **Pilot v1 evidence, 2026-08-11:** The run used immutable model revision `3c879f39826c281e95690f02c7821c4de09afae7` on an RTX 3090 and produced 178 masks plus 39 no-proposal rows across 30 backgrounds. File and manifest integrity checks passed.
- **Pilot v1 disposition:** Revision required; full-pool preprocessing remains unapproved. Floor and dining-room tabletop proposals were promising, but bedroom/hotel `tabletop` returned 0/20 coverage, `countertop` was semantically ambiguous, desk/nightstand prompts were inconsistent, bed masks represented complete bed silhouettes rather than verified support planes, and duplicate proposals require suppression.
- **Pilot v1 retention:** At the researcher's request, the Git-ignored v1 masks, manifest, overlays, contact sheets, and summaries were deleted after their hashes and findings were recorded. The versioned v1 config and documentary evidence remain; v1 is not an active dataset artifact.
- **Pilot v2 design:** Reuse the same deterministic 30 backgrounds and immutable SAM3 revision. Add score-ordered, within-prompt mask-IoU suppression at provisional threshold 0.85. Replace ambiguous prompts with `bed surface`, `top of bed`, `table surface`, `top of desk`, and `top of nightstand`; retain `floor` and dining-room `tabletop` as controls. These are pilot values, not production-approved values.
- **Pilot v2 evidence, 2026-08-11:** The run retained 259 of 275 area-filtered proposals after removing 16 within-prompt duplicates; no retained within-prompt pair exceeded mask IoU 0.85. All manifest/config/script and mask integrity checks passed.
- **Pilot v2 disposition:** Accept SAM3 as a promising semantic proposal front-end, but reject raw SAM3 masks as final anchor regions. Floors and dining-table regions were consistent; bed prompts found beds but not a reliable horizontal mattress plane, table masks could include non-support pixels, and desk/nightstand coverage remained low.
- **Next placement gate:** Do not run SAM3 again yet. Derive deterministic conservative support regions from the existing v2 proposals, visualize candidate anchors/footprints on the same 30 backgrounds, and obtain researcher approval before full-pool preprocessing.

## Open decisions

- Researcher acceptance of the v2 review; deterministic floor/bed/dining-table support-region postprocessing and anchor-overlay validation; then freeze the final support policy and class-to-orientation mapping.
- Exact Stable Diffusion, Qwen, and ControlNet models and versions.
- GenAI prompting, conditioning, placement, annotation, and quality-control procedures.
- Seed count and compute budget.
- Clean-domain tolerance.
- All ADR design and evaluation questions, after the active baseline phase is complete.
