# TODO

Keep this list current and update it only after verified work. ADR implementation is deferred until all baseline experiments are complete.

## Now - repository audit and cleanup

- [x] Locate training/evaluation entry points, dataset YAML files, class order, checkpoints, outputs, synthetic generators, SAM3 assets, and all ADR-related code and references.
- [x] Verify dataset paths, split sizes, YOLO11 variant, training configuration, and source-validation selection of `best.pt`.
- [x] Record the current commands/configurations and establish that no valid existing cut-paste experiment series is recoverable.
- [x] Confirm that the ADR score is preserved in `DECISIONS.md` and `PROJECT_CONTEXT.md`.
- [x] Verify the recoverable pre-cleanup checkpoint (`8d44582`) contains every removed ADR source file.
- [x] Produce an exact ADR deletion inventory and verify that no shared baseline or evaluation utility is included.
- [x] Delete ADR-only implementation files, callbacks, commands, configuration entries, imports, and ADR-specific tests from the active codebase.
- [x] Preserve baseline code, generators, shared evaluation utilities, verified experiment evidence, and the documented ADR specification during cleanup.
- [x] Run baseline static/import checks after cleanup; active ADR code references are absent and the deferred formula remains documentation-only.

## Completed cut-paste preparation

- [x] Verify and correct the INSP-DET test count to 278 matched image/label pairs.
- [x] Remove the obsolete generated synthetic datasets while preserving raw data, backgrounds, and the SAM3 object bank.
- [x] Set generator seed 42 and initial detector seed 0 as separate reproducible random processes.
- [x] Replace copied subset folders with one canonical dataset and nested manifests.
- [x] Freeze exact balanced quantities: CP-B0512, CP-B1024, CP-B1536, and CP-B2048.
- [x] Implement and verify deterministic balanced 16-class generation scheduling.
- [x] Audit all 2,750 SAM3 RGB/mask/RGBA triplets for integrity, mask agreement, exact duplicates, and provenance.
- [x] Reconstruct and verify provenance for the 1,186 assets missing direct metadata matches.
- [x] Record the researcher's decision that all manually cleaned object-bank assets remain eligible.
- [x] Calculate class-specific object-size distributions from all 4,435 INSP-DET train boxes.
- [x] Replace crop-relative resizing with background-relative class-specific p10--p90 sizing.
- [x] Add a versioned cut-paste config and remove the unused 160-image pilot stage.
- [x] Block full 2,048-image generation until placement, degradation, and QC release gates are approved.
- [x] Freeze critical Ultralytics 8.4.46 training and evaluation values explicitly.
- [x] Add a paper-facing methodology-to-code traceability record.
- [x] Implement per-image and dataset-level metadata, input/config hashes, code revision, software versions, and manifest checksums.
- [x] Restrict production to the 527 full-review-accepted backgrounds and record the researcher/contact-sheet target-leakage review with hashes and limitations.
- [x] Choose context-aware support-surface placement; reject the provisional fixed normalized bands for production.
- [x] Choose automatically proposed semantic support masks with human verification rather than manually selecting every anchor.
- [x] Select and record the exact semantic segmentation model, revision, license, environment, and accepted support-label mapping.
- [x] Select `facebook/sam3` as the first support-mask pilot candidate and record its SAM License and official source.
- [x] Implement the deterministic 10-images-per-category SAM3 proposal pilot, immutable revision capture, mask manifest, overlays, and contact sheets.
- [x] Manually launch SAM3 pilot v1 from the GPU-visible `env_sam3` terminal and record model SHA `3c879f39826c281e95690f02c7821c4de09afae7` and RTX 3090 metadata.
- [x] Complete structural/statistical and AI-assisted visual review of pilot v1; require revision and keep full-pool preprocessing blocked.
- [x] Obtain the researcher's acceptance of the pilot-v1 review and revised-pilot plan.
- [x] Delete v1's local masks, manifest, summaries, and visualizations after preserving hashes/findings in documentation.
- [x] Add provisional score-ordered mask-IoU deduplication (0.85) and revise bed/desk/nightstand/table prompt wording for v2.
- [x] Run SAM3 pilot v2 on the same deterministic 30 backgrounds; verify checksums, coverage, and mask-IoU deduplication.
- [x] Complete AI-assisted technical/visual review of v2; accept SAM3 as a proposal front-end but reject raw masks as final anchor regions.
- [x] Obtain the researcher's acceptance of the v2 review and geometry-postprocessing plan.
- [x] Implement conservative floor, bed-top, and dining-table-top support-region derivation from retained v2 masks.
- [x] Generate anchor/footprint overlays on the same 30 backgrounds without creating training images.
- [x] Complete researcher-authorized review of geometry-v1 regions/overlays and accept/reject pilot regions.
- [x] Freeze the 16-class orientation/support policy (lying, upright, or asset-preserved orientation) and contact/footprint rules.
- [x] Implement deterministic support-mask preprocessing and versioned manifests with checksums and reviewer status.
- [x] Validate automatic masks and geometry on a deterministic category-stratified sample before processing all 1,166 backgrounds.
- [x] Run frozen SAM3 v2 proposal preprocessing on all 1,166 backgrounds; verify 8,969 masks and manifest integrity.
- [x] Derive 1,870 full-pool support regions and verify every output/checksum.
- [x] Human-review all 608 full-pool bed/table candidate regions, reject the automatic risk groups and floor placement, and freeze 527 accepted regions.
- [x] Replace `sample_surface_position` with sampling from accepted support regions and test every class against the pilot manifest.
- [ ] Add per-attempt placement rejection-reason logging only if a future generator version is created.
- [x] Freeze and implement deterministic degradation-v1 operations, severity probabilities, ranges, and exact per-class schedules.
- [x] Freeze and implement QC-v1 for complete automatic validation and a deterministic 256-image class×severity manual review.
- [x] Approve the canonical cut-paste release switch after all pre-generation methodology gates passed.
- [x] Generate the first 2,048-image `cp_v1_seed42` canonical candidate and its four nested manifests.
- [x] Execute complete automatic QC-v1 and generate/review the deterministic 256-image class-by-severity sample.
- [x] Record the manual review's orientation, perspective, support-depth-scale, and contact-integration limitations.
- [x] Accept `cp_v1_seed42` as the simple cut-paste baseline before detector results are observed; defer realism improvements rather than expanding the current method.

## Phase 1 - reproduce the real-only and cut-paste baselines

- [x] Train E000 on the original 2,215 real images under the frozen 60-epoch YOLO11s protocol.
- [x] Evaluate E000 source-validation `best.pt` on the three official test domains and record class-wise results.
- [x] Verify the revised, versioned cut-paste pipeline after the generation specifications were frozen.
- [x] Generate the balanced canonical dataset and nested CP-B0512, CP-B1024, CP-B1536, and CP-B2048 manifests.
- [x] Validate its image counts, class allocation, annotations, provenance, duplicates, and manifests automatically.
- [x] Complete stratified visual review and approve the current dataset with documented baseline limitations.
- [x] Train each cut-paste configuration using the frozen training protocol.
- [x] Evaluate each cut-paste source-validation `best.pt` on INSP-DET test, easy test, and hard test.
- [x] Record the seed-0 cut-paste results and artifacts in `EXPERIMENT_LOG.md`, `RESULTS_SUMMARY.md`, and the private paper record.
- [x] Produce a combined cut-paste quantity-response CSV and paper-candidate plot.

## Phase 2 - Stable Diffusion + ControlNet baseline

- [x] Decide that the active GenAI method generates both background and object as new content while preserving the fixed 16-class taxonomy.
- [x] Draft, structurally validate, and freeze an eight-family MAIJA scene policy
  with four semantically compatible scenes per class and an all-class
  property-inspection reference scene.
- [x] Define and freeze MAIJA-aligned scene families and a cross-class context-allocation matrix that avoids one-class/one-background shortcuts.
- [x] Select and record the exact full-scene Stable Diffusion and ControlNet models, revisions, licenses, and feasibility compute strategy.
- [x] Define the shared single-image feasibility prompt, negative prompt, synthetic Canny condition, seed, and provenance.
- [x] Run SDXL feasibility v1 and record runtime/VRAM; reject its single-line Canny output as visually unrealistic while accepting model/hardware compatibility.
- [x] Run and visually review SDXL feasibility v2 using filled-proxy-derived Canny at control scale 0.8.
- [x] Replace the semantically weak programmatic target proxy with deterministic class-matched SAM3 binary silhouettes and verify all 16 classes.
- [x] Run and visually review SDXL feasibility v3 with the real scissors silhouette.
- [ ] Expand v3 into complete all-class prompt, subtype, scale, and spatial-control templates.
- [x] Freeze the GenAI canonical diversity rule: eight unique layouts per class-scene-prefix block, at least four shape/pose variants per class, and deterministic variation across camera, target, lighting, clutter, wording, and seed.
- [x] Freeze the GenAI canonical degradation schedule and ranges equal to executed copy-paste degradation-v1; keep feasibility images clean.
- [ ] Finalize post-generation SAM3 annotation and QC acceptance criteria.
- [ ] Generate and inspect a deterministic pilot covering all 16 classes and planned scene families.
- [ ] Resolve pilot failures before full-scale generation.
- [ ] Generate and validate SD-B0512, SD-B1024, SD-B1536, and SD-B2048.
- [ ] Train, evaluate, and record every valid configuration using the frozen protocol.

## Phase 3 - Qwen + ControlNet baseline

- [x] Apply the same fixed classes, quantities, and MAIJA scene/context policy as Stable Diffusion.
- [x] Select and record the exact full-scene Qwen and ControlNet models, revisions, licenses, compatibility basis, and proposed RTX 3090 memory strategy.
- [x] Define the shared single-image feasibility prompt, Canny condition, seed, and provenance.
- [x] Run Qwen feasibility v2 on RTX 3090 hardware using the same corrected control; measure offload, runtime, and peak VRAM.
- [x] Run and visually review Qwen feasibility v3 with the same real scissors silhouette used by SDXL.
- [ ] Finalize post-generation SAM3 annotation and QC acceptance criteria.
- [ ] Generate and inspect a deterministic all-class full-scene pilot; use
  binary SAM3 silhouettes as non-photographic ControlNet geometry and SAM3 separately for post-generation localization/annotation.
- [ ] Resolve pilot failures before full-scale generation.
- [ ] Generate and validate QW-B0512, QW-B1024, QW-B1536, and QW-B2048.
- [ ] Train, evaluate, and record every valid configuration using the frozen protocol.

## Phase 4 - baseline comparison and paper evidence

- [ ] Produce the complete 13-configuration results table.
- [ ] Compare every synthetic run with E000.
- [ ] Compare generators only at equal synthetic quantities.
- [ ] Analyze quantity response, class-wise changes, and clean-to-target trade-offs.
- [ ] Report all valid results and document failed or invalid runs.
- [ ] Update the paper outline with supported claims and missing evidence.

## Deferred - discuss and decide ADR after Phase 4

- [ ] Revisit the preserved difficulty function only after all baseline experiments are complete.
- [ ] Decide permitted validation/feedback data and resolve target-test validity.
- [ ] Finalize score metrics, weights, normalization, probability conversion, minimum class allocation, and cycle design.
- [ ] Decide which generators and quantities will receive ADR comparisons.
- [ ] Do not restore, implement, or run ADR without explicit researcher approval.

## Decisions still needed for the active phase

- [x] Freeze MAIJA-aligned scene families and cross-class context allocation.
- [x] Freeze exact full-scene Stable Diffusion, Qwen, and ControlNet models and immutable revisions for feasibility.
- [ ] Implement the frozen diversity/degradation policies, expand all-class prompt/conditioning templates, and finalize GenAI post-generation annotation/QC.
- [ ] Seed count and compute budget.
- [ ] Clean-domain mAP50-95 tolerance.
- [x] Fix the complete experiment matrix to pretrained YOLO11s; no model-size ablation.
- [x] Select semantic support-surface placement with human-verified automatic masks for cut-paste.

## Compute planning before generation and training

- [x] Confirm from the researcher's host terminal that CUDA is available and three GPUs are visible; Codex's isolated execution session has no GPU passthrough.
- [x] Require the researcher to launch every expensive preprocessing, generation, and training command manually after Codex reports the command and preflight status.
- [x] Align the detector protocol with the verified `env_sam3` Ultralytics 8.4.46 installation before any detector run.
- [x] Add training preflight checks for package version, CUDA/device visibility, dataset/model paths, clean Git state, and output collision.
- [x] Add explicit single-quantity selection to the copy-paste runner for safe parallel scheduling.
- [x] Add a fail-fast sequential 512→1024→1536→2048 pipeline with automatic three-domain evaluation, plotting, and progress state after each run.
- [x] Correct Ultralytics-incompatible nested manifest paths and require exact image/label counts in training preflight.
- [x] Confirm from the researcher that all three RTX 3090 GPUs are currently idle; recheck immediately before each launch.
- [x] Define the allocation plan: up to three independent single-GPU detector runs in parallel and deterministic per-GPU GenAI workers where memory permits.
- [ ] Record the exact GPU model, GPU ID, software/CUDA environment, run assignment, and wall-clock time for every paper experiment.
- [x] Keep one GPU per detector run and exclude DDP from the active experiment matrix so batch 16 and the E000/cut-paste execution mode remain unchanged.
- [ ] Benchmark generation throughput and VRAM for the selected Stable Diffusion, Qwen, and ControlNet implementations before launching full datasets.

## Scope boundaries

- No ADR implementation or experiments during the active baseline phase.
- No VLM-based failure analysis, detector-feedback-driven condition selection,
  or scene-aware ADR. Predeclared MAIJA scene prompting for non-adaptive GenAI
  baselines is allowed.
- Target easy and hard results do not tune checkpoints or generator settings during the active baseline phase.
