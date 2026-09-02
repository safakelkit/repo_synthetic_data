# Paper Outline

## Current writing boundary

The current experimental phase covers the real-only, cut-paste, Stable Diffusion + ControlNet, and Qwen + ControlNet baselines without ADR. Keep ADR as a planned later method; do not write ADR results or claims before its protocol is approved and experiments are complete.

## 1. Introduction

- Clean-to-video domain shift and practical motivation.
- Primary objective: improve mAP50-95 on both INSP-MOT-DET easy and hard while preserving acceptable clean-domain performance.
- Need for synthetic images that represent blur, low resolution, lighting variation, compression, occlusion, and background diversity.
- Motivation for comparing traditional and GenAI generation across equal data quantities.
- Contributions must match completed experiments.

## 2. Related Work

- Synthetic data for object detection.
- Copy-paste augmentation.
- Context-aware object placement and semantic support-surface constraints.
- Stable Diffusion, Qwen, and controlled image generation.
- ControlNet and spatial conditioning.
- Domain generalization under image/video degradation.
- Active Domain Randomization as planned later work.

## 3. Data and Problem Definition

- INSP-DET train/validation/test.
- INSP-MOT-DET easy/hard test splits.
- Sixteen-class label space.
- SAM3 masks, RGB crops, and RGBA crops.
- Observed target-domain characteristics.

## 4. Baseline Detector and Evaluation

- YOLO11s and the frozen training configuration in `METHODOLOGY_TRACEABILITY.md`.
- Real-only experiment E000.
- Source-validation selection of `best.pt`.
- mAP50-95 primary metric and supporting metrics.
- Clean/easy/hard evaluation protocol.

## 5. Synthetic Generation Baselines

### 5.1 Cut-paste

- SAM3 object assets and class-specific source-derived sizing.
- Automatically proposed, human-verified semantic support masks (**accepted;
  SAM3 v2 and geometry-v2 full-pool reviewed; 527 support regions accepted
  across 527 backgrounds, with floor placement excluded**).
- Exact mask-model revision, prompts, thresholds, pilot sampling, license, and
  human-review protocol.
- Footprint/contact-anchor placement rules and their 2D-geometry limitations.
- Fixed degradation-v1 clean/light/medium/heavy mixture, operation ranges, and post-composite application.
- Target-domain degradation design (**not yet frozen or implemented**).
- Annotation and provenance.

### 5.2 Stable Diffusion + ControlNet

- Shared method: paired controlled inpainting from the frozen background, RGBA object, placement, expanded mask, and degradation schedule.
- Candidate backend: SDXL inpainting with Diffusers Canny ControlNet; exact immutable revisions and inference values remain pending smoke-test acceptance.
- Prompts, conditioning, object placement, and pilot validation.
- Annotation and quality control.

### 5.3 Qwen + ControlNet

- Shared method and sample identities match Section 5.2.
- Candidate backend: Qwen-Image with the community InstantX inpainting ControlNet; explicitly distinguish this from an official Qwen release and freeze exact revisions only after feasibility testing.
- Prompts, conditioning, object placement, and pilot validation.
- Annotation and quality control.

### 5.4 Class-balanced allocation

- No detector feedback.
- Deterministic remainder handling across 16 classes.
- Quantities: 512, 1,024, 1,536, and 2,048, giving exactly 32, 64, 96, and 128 images per class.

## 6. Experimental Setup

- Thirteen configurations per seed.
- Equivalent training and evaluation budgets.
- Seeds, hardware, software, and dataset manifests.
- Synthetic provenance and annotation checks.
- Clean-domain preservation reporting.

## 7. Baseline Results

- Overall multi-domain results for all 13 configurations.
- Synthetic-quantity response for each generator.
- Equal-quantity generator comparisons.
- Per-class gains and losses.
- Clean-to-target trade-offs.
- Failed or invalid experiments.
- Current cut-paste seed-0 evidence: every quantity improves easy and hard over
  E000 while slightly reducing clean performance; the response is non-monotonic.
- Report CP-B1024 as the best clean-preserving trade-off, CP-B1536 as the
  highest hard result, and CP-B2048 as the highest easy result without treating
  one seed as statistical proof.

## 8. Baseline Analysis

- Which generator benefits easy and hard domains most?
- Does performance saturate or decline with more synthetic data?
- Does synthetic data harm clean-domain performance?
- Qualitative successes and failures.
- Synthetic realism, annotation quality, and placement limitations.

## 9. Deferred ADR Method

- ADR will be discussed only after baseline experiments are complete.
- Preserve the proposed class-wise difficulty function from `DECISIONS.md`.
- Resolve validation/feedback data, weights, normalization, probability conversion, cycle design, and scientific validity before implementation.
- Do not claim condition-aware control or scene understanding.

## 10. Limitations and Validity

- Synthetic label noise and realism.
- GenAI placement and conditioning reliability.
- Dataset size and class imbalance.
- Multiple-seed and compute limitations.
- Target test sets must not tune the active baseline methods.
- ADR validity questions remain deferred.

## 11. Conclusion

- State only supported baseline findings.
- Separate demonstrated results from the planned ADR phase.

## Planned tables and figures

- Dataset/domain summary.
- Thirteen-configuration experiment matrix.
- Synthetic generation examples from all three methods.
- Equal-quantity main-results table.
- Quantity-response plots per generator.
- Per-class clean/easy/hard comparison.
- Target gain versus clean-domain change.
- ADR diagram only when the later protocol is approved.
