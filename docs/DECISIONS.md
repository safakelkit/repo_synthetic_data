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
- **Decision:** Pin Ultralytics 8.4.46, matching the verified training environment before any frozen-protocol run, and write the critical optimizer, augmentation, reproducibility, and evaluation thresholds explicitly rather than relying on changeable library defaults.
- **Source:** `requirements.txt`, `configs/train_baseline.yaml`, `src/training/train.py`, and `src/evaluate_yolo.py`.
- **Paper record:** Exact values and code mappings are maintained in `METHODOLOGY_TRACEABILITY.md`.

## D014 - Freeze the effective optimizer

- **Status:** Accepted and implemented
- **Decision:** Replace `optimizer=auto` with the verified effective choice for this matrix: AdamW, initial learning rate 0.0005, beta1 0.9, and warmup bias learning rate 0.0. These explicit values were revalidated under Ultralytics 8.4.46.
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

- **Status:** Accepted, implemented, and executed
- **Date:** 2026-09-01
- **Automatic scope:** Validate all 2,048 images, labels, metadata links, dimensions, SHA-256 values, exact duplicates, normalized boxes, realized-vs-labelled area (maximum 5% relative difference), nested manifests, and exact class×severity counts at every experiment prefix.
- **Manual scope:** Deterministically sample four images from every class×severity cell, totaling 256 images. Review class identity, scale, support placement, alpha/blending, box alignment, degradation plausibility, and unlabeled target objects.
- **Release rule:** Automatic QC must pass; the complete stratified sample must be reviewed; and the researcher must record an explicit dataset-level accept/reject disposition before detector training.
- **Implementation:** Automatic thresholds and sampling originated in executed `copy_paste_qc_v1.yaml`; the pre-results researcher acceptance rule is versioned in active `configs/quality/copy_paste_qc_v1_1.yaml`. Both use `src/validation/validate_copypaste_dataset.py`.

## D020 - Production backgrounds are restricted to the reviewed support pool

- **Status:** Accepted and implemented
- **Date:** 2026-09-01
- **Eligible pool:** 527 backgrounds, each with exactly one accepted geometry-v2 support region (306 bed-top, 221 dining-table-top).
- **Evidence:** The researcher's prior background inspection plus complete contact-sheet review of all 608 automatic bed/table candidates. The visibly confirmed laptop-containing background was rejected during that review.
- **Manifest:** `data/processed/background_support_masks/sam3_v2/full_geometry_v2/eligible_backgrounds.csv`, SHA-256 `c15243e14a888d284c84e0bce66d46998f14437ac8a3eb573c712bb3e8161f09`.
- **Interpretation:** No visible target-class instance was confirmed in the accepted pool. This is a human-review result, not an exhaustive object-detector guarantee; very small or occluded instances may escape visual review.
- **Enforcement:** The generator can only select backgrounds appearing through accepted support rows in the reviewed geometry manifest.

## D021 - cp_v1 is accepted as a simple baseline with visual limitations

- **Status:** Accepted
- **Date:** 2026-09-01
- **Evidence:** The candidate generated from commit `6c14f12` contains 2,048 unique images and passed QC-v1 automatic checks. The complete deterministic 256-image class-by-severity visual sample was inspected.
- **Decision:** Accept `cp_v1_seed42` for the fixed CP-B0512--CP-B2048 matrix. This decision was made before observing detector results.
- **Limitations:** Lying assets can retain unsuitable 2D orientation, object scale is not conditioned on support depth, upright/laptop contact can be implausible, and no perspective/contact-shadow treatment integrates objects with the surface.
- **Rationale:** These limitations are intrinsic to the deliberately simple cut-paste baseline; solving them would substantially expand the method without guaranteed detector benefit. Automatic label/integrity checks passed, objects remain recognizable, and four nested quantities can measure whether artifacts help or harm detection.
- **Future work:** Any realism-enhanced generator must use a new version and repeat QC; do not overwrite `cp_v1_seed42`.
- **Restriction:** Do not use easy/hard test performance to tune these corrections.

## D022 - The cut-paste quantity matrix runs sequentially and fail-fast

- **Status:** Accepted and implemented
- **Date:** 2026-09-02
- **Order:** CP-B0512, CP-B1024, CP-B1536, CP-B2048.
- **Execution:** Train each run to completion, evaluate its source-validation `best.pt` on the three frozen test domains, render plots, then start the next quantity.
- **Safety:** Preflight validates all selected runs before the first starts. Existing train/evaluation outputs cause an error. Any training, evaluation, or plotting failure stops the matrix and records state in `runs/evaluation/copy_paste_matrix_status.json`.
- **Validity:** The pipeline does not inspect metrics or change later configurations; sequential evaluation is reporting only and does not create detector feedback.
- **Implementation:** `src/training/train_copypaste_baselines.py --experiment all --evaluate`.

## D023 - Dataset manifests must be validated with Ultralytics path semantics

- **Status:** Accepted and implemented
- **Date:** 2026-09-02
- **Incident:** The first CP-B0512 launch used nested manifest entries of the form `./../images/...`. Although these paths resolve with standard filesystem semantics, Ultralytics 8.4.46 replaces every `./` substring and produced invalid duplicated paths. All 512 synthetic images were ignored and the interrupted run loaded only the 2,215 real images; that attempt is invalid and must not be reported.
- **Decision:** Store the four manifests at the canonical dataset root and use `./images/...` entries. Do not duplicate image files.
- **Safety:** Dataset YAMLs declare the exact expected training-image count. Preflight resolves lists with the pinned Ultralytics semantics and requires the exact number of existing, unique image/label pairs before creating a run.
- **Expected counts:** 2,727; 3,239; 3,751; and 4,263 total training images for CP-B0512 through CP-B2048.
- **Implementation:** `create_subset_manifests()` and `validate_training_dataset()`.

## D024 - GenAI baselines generate complete MAIJA-aligned scenes

- **Status:** Method family, scene policy, and feasibility model pairs accepted; canonical implementation pending
- **Date:** 2026-09-02
- **Decision:** Stable Diffusion and Qwen baselines generate both the background and target object as new image content. They do not reuse a real Places365 background or paste an object-bank RGBA crop into the generated image.
- **Fixed taxonomy:** The existing 16 classes and IDs in `configs/data_insp.yaml` are immutable. GenAI generation may vary object appearance, subtype, viewpoint, and background, but must not add, remove, merge, rename, or reorder experimental classes.
- **Scene policy:** Backgrounds are not restricted to bedroom, hotel room, and dining room. They follow the frozen correctional-facility policy grounded in MAIJA's detention-room-search scenario and specified in D025.
- **Context balance:** A class may use physically plausible contexts, but no class may be tied to one unique background family. Multiple scene families must overlap across classes to reduce class prediction from background shortcuts. Freeze the class-to-scene allocation before target-test evaluation.
- **Spatial control:** ControlNet may receive a layout, contour, silhouette, depth, or other non-photographic spatial condition. It must not receive real background pixels or an RGBA object composite in the active full-scene method.
- **Allocation:** Preserve 2,048 images/backend, exactly 128 primary targets/class, with balanced nested prefixes of 512/1,024/1,536/2,048.
- **Annotation:** A requested class or control region is not automatically a valid label. Recover the visible generated target with SAM3 and reject absent, wrong-class, duplicate, malformed, implausibly scaled, or unlocalizable outputs. Any additional visible project-class instance must be annotated or the image rejected.
- **Pilot:** Before production, generate and inspect the same-size deterministic pilot for both backends, covering every class and the planned scene families. Record immutable model revisions, prompts, controls, seeds, runtime/VRAM, QC, and provenance.
- **Restriction:** Do not choose scene families, prompts, models, controls, or acceptance thresholds using INSP-MOT-DET easy/hard results.

## Open decisions

- Complete all-class prompting and spatial-conditioning templates after the single-image feasibility runs.
- Freeze SAM3 annotation thresholds and quality-control acceptance criteria.
- Seed count and compute budget.
- Clean-domain tolerance.
- All ADR design and evaluation questions, after the active baseline phase is complete.

## D025 - Frozen GenAI scene taxonomy and balanced context matrix

- **Status:** Accepted and frozen before GenAI generation
- **Date:** 2026-09-02
- **Source of truth:** `configs/generation/genai_scene_policy_v1.yaml`
- **Evidence:** The official MAIJA description places the mobile assistant in a correctional-facility context. The INSP dataset description narrows the object-recognition scenario to detention-room searches and reports varied indoor scenes, clutter, partial occlusion, and non-uniform lighting.
- **Scene families:** Cell sleeping area, cell desk area, cell storage area, cell wash area, communal day room, property-inspection station, correctional workshop, and maintenance/service area.
- **Domain weighting:** Five families represent detention living spaces, one is a controlled property-inspection reference, and two are supervised operational spaces. Assignment frequency gives each 512-image block 272/128/112 images from these three groups respectively.
- **Semantic matrix:** Every class is assigned exactly four compatible scene families. The property-inspection station is shared by all 16 classes; other families overlap across 4--8 classes. Each 32-image class block allocates eight images to each assigned family. Exact class balance is mandatory, but equal scene totals are deliberately not forced when that would introduce implausible class-scene pairs.
- **Class boundary:** Scene assignment never changes the existing class label. Subtype and appearance variation may be defined later only within the fixed semantic meaning of that class.
- **Validity boundary:** This matrix was designed without easy/hard test feedback and accepted by the researcher before generation. It must not be revised using target-test results.
- **Sources:** https://cvl.tuwien.ac.at/project/maija/ and https://repositum.tuwien.at/bitstream/20.500.12708/224661/1/Bernhart%20Costin%20-%202025%20-%20Real-Time%20Multi-Object%20Tracking%20under%20Resource...pdf

## D026 - Parallel runs use one independent experiment per GPU

- **Status:** Accepted
- **Date:** 2026-09-02
- **Hardware:** Three host-visible NVIDIA GeForce RTX 3090 GPUs are currently available; availability must be rechecked immediately before launch.
- **Decision:** Do not distribute one detector run across three GPUs. Launch up to three independent experiment processes concurrently, each restricted to one physical GPU with `CUDA_VISIBLE_DEVICES` and using logical `device: 0`.
- **Protocol preservation:** Every run retains global batch 16, detector seed 0, pretrained YOLO11s initialization, 60 epochs, and all other frozen settings. DDP is excluded from the active matrix because three-way distribution would require changing the batch protocol and execution mode used by E000 and cut-paste.
- **Isolation:** Every process must have a unique experiment ID, run directory, evaluation directory, progress record, and log. A failure in one process must not overwrite or invalidate another.
- **Resource caution:** Three runs may contend for CPU workers, RAM, and storage bandwidth. Measure utilization at launch; if worker count must change, freeze one common value for all GenAI detector experiments and document the deviation before results are observed.
- **Evidence:** Record physical GPU ID, logical GPU ID, GPU model, driver/CUDA environment, start/end UTC, wall time, and concurrent workloads for every run.
- **Generation:** Full-scene generation may also use one independent deterministic worker per GPU, provided sample-level seeds and output ownership make results invariant to worker scheduling.

## D027 - Frozen full-scene feasibility model pairs

- **Status:** Both model pairs accepted for continued pilot development; canonical production remains blocked
- **Date:** 2026-09-02
- **Source of truth:** `configs/generation/genai_models_v1.yaml`
- **Stable Diffusion:** `stabilityai/stable-diffusion-xl-base-1.0` at commit `462165984030d82259a11f4367a4eed129e94a7b` with `diffusers/controlnet-canny-sdxl-1.0` at `eb115a19a10d14909256db740ed109532ab1483c`; OpenRAIL++; FP16; no refiner.
- **Qwen:** `Qwen/Qwen-Image` at commit `75e0b4be04f60ec59a75f475837eced720f823b6` with `InstantX/Qwen-Image-ControlNet-Union` at `b13036f066d6dee7c20513e263d3d673055e9de8`; Apache-2.0; Canny control; BF16 compute.
- **Why original Qwen-Image:** The selected InstantX ControlNet explicitly documents compatibility with the original Qwen-Image base. Qwen-Image-2512 is therefore not substituted without separate compatibility evidence.
- **RTX 3090 evidence:** Qwen transformer and text encoder used bitsandbytes NF4 4-bit quantization while ControlNet remained BF16 with model CPU offload. V2 completed with 16.634/16.859 GiB peak allocated/reserved memory, 2,127.631 s model load, 142.715 s inference, and 2,270.781 s wall time.
- **Fair feasibility input:** Both backends receive the same programmatically drawn 1024x1024 Canny condition, class (`Scissors`, ID 2), scene (`property_inspection_station`), seed 42, 30 inference steps, and ControlNet scale 0.8. Model-specific guidance remains SDXL 5.0 and Qwen true-CFG 4.0.
- **Boundary:** A generated feasibility image is not training data and has no automatic YOLO annotation. Canonical generation stays blocked until both outputs are visually reviewed and the all-class annotation/QC pilot is approved.
- **Primary sources:** https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0 ; https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0 ; https://huggingface.co/Qwen/Qwen-Image ; https://huggingface.co/InstantX/Qwen-Image-ControlNet-Union ; https://huggingface.co/docs/diffusers/quantization/bitsandbytes .
- **V1 evidence:** SDXL loaded and generated successfully on an RTX 3090 at commit `957b8d9` (406.376 s wall time; 180.978 s inference; 7.735/10.379 GiB peak allocated/reserved). The single-line Canny drawing was interpreted as thin physical cords, so the output failed visual realism and is not training data.
- **V2 correction:** Keep the same models/class/scene/seed/steps, derive Canny from filled table and object proxy regions, and reduce control scale from 0.9 to 0.8. This tests solid-object boundaries rather than line-art copying.
- **V2 outcome:** SDXL produced a recognizable but imperfect scissors; Qwen followed the condition strongly but produced a forceps/hemostat-like object. Both runs proved technical integration, but the hand-drawn target proxy failed class-semantic review.

## D028 - GenAI diversity and degradation are controlled independently

- **Status:** Accepted design; canonical implementation and all-class pilot pending
- **Date:** 2026-09-02
- **Source of truth:** `configs/generation/genai_generation_policy_v1.yaml` and `configs/generation/genai_degradation_v1.yaml`
- **Diversity:** A single control image or layout must never be reused across the canonical dataset. Every class-scene block uses deterministic but distinct scene geometry, camera framing, target position/scale/orientation, lighting, clutter/material, prompt wording, and sample seed combinations. Each eight-image class-scene increment requires eight unique layouts and at least four class-specific shape/pose variants.
- **Fair comparison:** SDXL and Qwen receive the same class, scene, variation schedule, target geometry, and sample seed for corresponding canonical indices; their output pixels remain independently generated.
- **Clean-first rule:** Generate and validate the clean complete scene first. The one-image feasibility tests remain clean so degradation cannot hide a failed object or scene generator.
- **Canonical degradation:** After localization/annotation, apply the exact executed cut-paste degradation-v1 mixture and numerical ranges to the complete generated image: 25% clean, 37.5% light, 25% medium, and 12.5% heavy in every 32-image per-class block.
- **Operations:** Gaussian or motion blur, downscale-upscale resolution loss, brightness/contrast, JPEG compression, and Gaussian sensor noise; every non-clean image receives at least one operation.
- **Post-degradation QC:** Recheck target visibility and unchanged label geometry after degradation. Do not tune diversity or degradation using easy/hard results.

## D029 - GenAI ControlNet target geometry uses real class silhouettes

- **Status:** Accepted and implemented for v3 feasibility; all-class pilot validation pending
- **Date:** 2026-09-02
- **Source of truth:** `configs/generation/genai_feasibility_v3.yaml` and `src/generation/run_full_scene_feasibility.py`
- **Decision:** Construct the target portion of each Canny condition from a class-matched accepted SAM3 binary mask in the existing object bank. The background and target appearance remain newly generated.
- **Pixel boundary:** Reuse binary shape only. Source RGB/RGBA pixels never enter the condition or generated scene.
- **Coverage:** Real silhouettes are the default for all 16 classes. Geometry-critical hazard/tool classes have no programmatic-shape fallback.
- **Reproducibility:** Record asset ID, class ID, mask path and SHA-256, audit-manifest SHA-256, selection seed, rendered box, proxy hash, and Canny hash. Corresponding SDXL and Qwen samples receive the same condition.
- **Evidence:** All 16 class masks were selected and rendered successfully without GPU inference. The deterministic scissors case uses asset `02_IMG_0050121_obj_02_crop_000047`; its silhouette preserves blades, pivot, and asymmetric handles.
- **Boundary:** A correct source silhouette does not prove generated class identity. Post-generation semantic review, SAM3 localization, annotation, and QC remain mandatory.
