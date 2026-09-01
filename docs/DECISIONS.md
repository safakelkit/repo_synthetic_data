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
- **Decision:** Generation remains blocked until placement, degradation, background eligibility, and QC are approved. Those methodological gates and the explicit release switch are now approved; the clean committed-worktree check and researcher-only manual launch still prevent accidental execution. The generated canonical set must be validated before detector training.
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

- **Status:** Accepted and implemented; exact operations/ranges frozen by D018
- **Decision:** Assign each class and every nested 32-image allocation block to 25% clean composite, 37.5% light degradation, 25% medium degradation, and 12.5% heavy degradation.
- **Exact per-class allocation per 32 images:** 8 clean, 12 light, 8 medium, and 4 heavy.
- **Reason:** The allocation is exact and nested for CP-B0512 through CP-B2048, preserves a substantial clean synthetic component, and limits the heavy-degradation share.
- **Restriction:** Do not tune the frozen operations or ranges using easy/hard test performance.

## D017 - Context-aware cut-paste placement uses verified semantic support surfaces

- **Status:** Accepted and implemented; full-pool review complete
- **Date:** 2026-08-11
- **Decision:** Replace the provisional fixed normalized placement bands with automatically proposed semantic support-surface masks followed by human verification. The fixed-band implementation is not an approved production method.
- **Precomputation:** Infer support masks once for the background pool, store the masks and a versioned manifest, and reuse the accepted regions during deterministic generation. Record the segmentation model, exact revision, software environment, input/background checksum, mask checksum, accepted support classes, and reviewer status.
- **Pilot gate:** The deterministic 30-background pilot is complete and approves full-pool proposal preprocessing with the frozen v2 configuration.
- **Placement geometry:** For objects intended to lie on a surface, require the transformed object footprint to fit inside an accepted support region. For upright objects, align the bottom alpha-contact region with a valid support anchor. Table-like objects require a top/support boundary rather than arbitrary pixels from the full table mask.
- **Human role:** Human review validates or rejects automatically proposed support regions; it does not manually choose every generated object coordinate. Generation samples deterministic anchors only from accepted regions.
- **Validity boundary:** Semantic masks constrain plausible 2D location but do not recover full 3D geometry, lighting, shadows, or physical interaction. These limitations must remain part of QC and paper reporting.
- **Frozen proposal model:** `facebook/sam3`, SAM License, commit `3c879f39826c281e95690f02c7821c4de09afae7`; score 0.30, mask 0.50, minimum mask-area ratio 0.001, within-prompt mask-IoU suppression 0.85.
- **Frozen geometry:** Geometry-v2 keeps only the largest connected component for each derived region. Production supports `bed_top` and `dining_table_top`; `floor` is disabled because a 2D floor mask cannot prevent pasted objects from appearing over foreground furniture or at implausible depth. Lying objects must fit inside the region, upright objects use bottom-alpha contact, and laptop uses asset-preserved bottom contact. Rotation is disabled. Exact class mapping is in `configs/placement/class_orientation_support_v2.yaml`.
- **Restriction:** Do not tune model, prompts, geometry, or orientation policy using INSP-MOT-DET easy or hard results.
- **Methodological context:** Random copy-paste is supported by *Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation* (CVPR 2021), while context-aware placement is supported by *Modeling Visual Context is Key to Augmenting Object Detection Datasets* (ECCV 2018). The semantic support-mask workflow is this project's reproducible implementation choice, not a mandatory standardized stage from either paper.
- **References:** https://openaccess.thecvf.com/content/CVPR2021/html/Ghiasi_Simple_Copy-Paste_Is_a_Strong_Data_Augmentation_Method_for_Instance_CVPR_2021_paper.html ; https://arxiv.org/abs/1807.07428
- **Model sources/license:** Official model card: https://huggingface.co/facebook/sam3 ; license: https://huggingface.co/facebook/sam3/blob/main/LICENSE
- **Pilot evidence:** V2 retained 259 of 275 area-filtered proposals after removing 16 duplicates. Geometry-v1 produced 24 floor, 17 bed-top, and 9 dining-table-top candidates. Review accepted 24 floor, 12 bed-top, and 9 table-top regions (45 total across 28 backgrounds) and rejected invalid or unsafe regions.
- **Implementation:** `propose_support_masks_sam3.py` -> `derive_support_geometry.py` -> `finalize_support_geometry_review.py` -> reviewed manifest consumed by `generate_copypaste_dataset.py`.
- **Full-pool evidence, 2026-08-29:** SAM3 processed 1,166 backgrounds and retained 8,969 masks. Geometry-v2 derived 904 floor, 622 bed-top, and 331 table-top regions after largest-component filtering. Conservative triage plus complete review of all 608 automatic bed/table candidates retained 306 bed-top and 221 table-top regions across 527 backgrounds. The reviewed manifest SHA-256 is `758ed5959fcd40fd838e98be6c1b8beeb0a173bd56b7e47aee0fa1d7bfd0c702`.
- **Review evidence:** Within the automatic candidate group, 75/381 bed regions (19.7%) and 6/227 table regions (2.6%) failed visual review, 81/608 combined (13.3%). All automatic risk-group rows and every floor row were rejected conservatively.

## D018 - Cut-paste degradation-v1 uses a fixed clean-to-heavy mixture

- **Status:** Accepted and implemented
- **Date:** 2026-09-01
- **Allocation:** Within every class-specific block of 32 synthetic images: 8 clean, 12 light, 8 medium, and 4 heavy. This remains exact at every 512-image experiment boundary.
- **Operations:** Blur (Gaussian or motion), downscale-upscale resolution loss, brightness/contrast, JPEG compression, and Gaussian sensor noise are applied after compositing so object and background share the same camera-like corruption.
- **Ranges:** The complete probabilities and numeric ranges are frozen in `configs/generation/copy_paste_v1.yaml`; every sampled operation and value is written to per-image metadata.
- **Determinism:** Degradation seed is generator seed + 2 (`44`). Non-clean severity always applies at least one operation.
- **Rationale:** Preserve 25% clean synthetic data while exposing the detector to varied common corruptions. The full CP-B2048 training set still contains 2,215 real images plus 512 clean synthetic images, so degraded samples are approximately 36% of all training images.
- **Restriction:** Do not alter the distribution or ranges based on INSP-MOT-DET easy/hard test results.
- **Context:** The chosen corruption families follow common-corruption robustness literature, including ImageNet-C and AugMix; their use here is a fixed project-specific distribution, not a reproduction of either benchmark.
- **References:** https://openreview.net/forum?id=Bygh9j09KX ; https://openreview.net/forum?id=S1gmrxHFvB

## D019 - Canonical cut-paste release requires automatic and stratified manual QC

- **Status:** Accepted and implemented; awaits generated-dataset execution
- **Date:** 2026-09-01
- **Automatic scope:** Validate all 2,048 images, labels, metadata links, dimensions, SHA-256 values, exact duplicates, normalized boxes, realized-vs-labelled area (maximum 5% relative difference), nested manifests, and exact class×severity counts at every experiment prefix.
- **Manual scope:** Deterministically sample four images from every class×severity cell, totaling 256 images. Review class identity, scale, support placement, alpha/blending, box alignment, degradation plausibility, and unlabeled target objects.
- **Release rule:** Automatic QC must pass and every sampled row must receive an explicit human pass/reject decision before detector training.
- **Implementation:** `configs/quality/copy_paste_qc_v1.yaml` and `src/validation/validate_copypaste_dataset.py`.

## D020 - Production backgrounds are restricted to the reviewed support pool

- **Status:** Accepted and implemented
- **Date:** 2026-09-01
- **Eligible pool:** 527 backgrounds, each with exactly one accepted geometry-v2 support region (306 bed-top, 221 dining-table-top).
- **Evidence:** The researcher's prior background inspection plus complete contact-sheet review of all 608 automatic bed/table candidates. The visibly confirmed laptop-containing background was rejected during that review.
- **Manifest:** `data/processed/background_support_masks/sam3_v2/full_geometry_v2/eligible_backgrounds.csv`, SHA-256 `c15243e14a888d284c84e0bce66d46998f14437ac8a3eb573c712bb3e8161f09`.
- **Interpretation:** No visible target-class instance was confirmed in the accepted pool. This is a human-review result, not an exhaustive object-detector guarantee; very small or occluded instances may escape visual review.
- **Enforcement:** The generator can only select backgrounds appearing through accepted support rows in the reviewed geometry manifest.

## Open decisions

- Exact Stable Diffusion, Qwen, and ControlNet models and versions.
- GenAI prompting, conditioning, placement, annotation, and quality-control procedures.
- Seed count and compute budget.
- Clean-domain tolerance.
- All ADR design and evaluation questions, after the active baseline phase is complete.
