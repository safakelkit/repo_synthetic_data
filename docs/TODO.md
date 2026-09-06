# TODO

This file contains the active queue and milestone summary only. Run history is
in `EXPERIMENT_LOG.md`; accepted methodology is in `DECISIONS.md`; exact paper
values are in `METHODOLOGY_TRACEABILITY.md` and the private paper record.

## Active — SDXL baseline

- [x] Complete E000 and the four seed-0 cut-paste runs and evaluations.
- [x] Freeze the 16-class, quantity, detector, MAIJA scene, diversity, and
  degradation policies.
- [x] Verify SDXL/Qwen model integration and real-silhouette ControlNet input.
- [x] Run/review two 64-image SDXL all-class pilots.
- [x] Run/review the 28-image targeted SDXL v3 diagnostic; retain improvements
  for Matches, Shaver, and Mobile phone.
- [x] Combine successful class-specific pilot settings into the frozen SDXL
  candidate pipeline without using easy/hard feedback.
- [x] Prepare targeted SDXL v4 with subtype-consistent prompts/silhouettes and
  verify all 16 controls for the four remaining classes.
- [x] Extend the pilot manifest with frozen code/config/model hashes, package
  and GPU environment, runtime/VRAM, and per-image output/control hashes.
- [x] Run/review targeted SDXL v4: Pliers 4/4, Aerosol can 0/4, Battery
  2/4, Laptop 2/4; overall strict acceptance 8/16.
- [x] Prepare Aerosol-only v5 conditioning-scale diagnostic with four samples
  each at 0.45/0.60/0.75/0.90 and no misleading internal target edges.
- [x] Run/review Aerosol v5: 2/16 strict acceptance, both in one scene; reject
  further silhouette-Canny scale tuning.
- [x] Adopt compact annotation-first acceptance: reject absent/wrong,
  unrecognizable, unlocalizable, or incompletely labelled targets; do not reject
  merely for mild synthetic appearance or simple composition.
- [x] Define a fixed candidate-surplus policy and report attempts, class-wise
  acceptance rate, runtime, and rejection reasons.
- [x] Implement the deterministic 64-image SDXL canonical acceptance test and
  sharded production candidate scheduler.
- [x] Run/review the 64-image canonical acceptance test: 54/64 accepted
  (84.375%); 15 classes have usable yield; test images remain forbidden from training.
- [x] Run/review the four-image Aerosol scene-only recheck: 0/4; reject this
  fallback and stop SDXL prompt/ControlNet micro-tuning in the current architecture.
- [x] Run/review the eight-image SDXL Aerosol target-only Canny diagnostic. This
  tests a new conditioning decomposition (target edges only), not another scale
  sweep of the failed combined scene-and-target control.
- [ ] Validate a two-stage Aerosol path: generate the frozen scene first, then
  place/regenerate a target-only SDXL Aerosol inside a declared region while
  preserving scene pixels and recording the separate provenance of both stages.
- [ ] Choose and validate one method that can generate all 16 immutable classes;
  do not launch the blocked 2,592-candidate SDXL run.
- [ ] Implement post-generation localization/annotation and automatic QC.
- [x] Freeze compact acceptance criteria using pilot outputs only.
- [ ] Validate class-wise candidate yield before canonical production.
- [ ] Generate and validate the nested SD-B0512/1024/1536/2048 dataset.
- [ ] Train and evaluate all four SDXL configurations with the frozen YOLO11s
  protocol.

## Next — Qwen baseline

- [x] Freeze exact Qwen/ControlNet revisions and verify single-image execution
  on RTX 3090.
- [ ] Resume a small all-class Qwen pilot only after the shared annotation/QC
  method is frozen.
- [ ] Resolve Qwen-specific quality/runtime failures without changing the fair
  class, scene, quantity, and degradation policies.
- [ ] Generate/validate QW-B0512/1024/1536/2048.
- [ ] Train and evaluate all four Qwen configurations.

## Final baseline analysis

- [ ] Complete the 13-configuration table.
- [ ] Compare generators only at equal accepted synthetic quantities.
- [ ] Analyze easy/hard gains, clean cost, quantity response, class-wise changes,
  candidate yield, and generation cost.
- [ ] Repeat central comparisons with additional predeclared detector seeds if
  compute permits.
- [ ] Update the paper outline with supported claims, figures, limitations, and
  all failed/invalid runs.

## Open decisions

- [ ] Numerical clean-domain mAP50-95 loss tolerance.
- [ ] Additional detector seed count and compute budget.
- [x] Candidate surplus: 25% standard; 50% Aerosol until measured production yield.
- [ ] Exact annotation implementation and finalizer integrity checks.
- [ ] Capture physical GPU assignment, NVIDIA driver, CUDA/software environment,
  and wall time for every remaining paper run.

## Deferred ADR

- [ ] Revisit validation/feedback data, score definition, allocation, cycle
  design, generators, and quantities only after all non-ADR baselines finish.
- [ ] Do not restore or run ADR without explicit researcher approval.

## Completed milestone summary

- Repository paths, datasets, labels, manifests, and experiment output safety
  were audited and corrected.
- The SAM3 object bank (2,750 assets) and source-only object-size distributions
  were audited.
- Context-aware cut-paste placement was derived from all 1,166 backgrounds; 527
  support regions were accepted after full review.
- The canonical 2,048-image cut-paste dataset passed automatic QC and stratified
  visual review and was accepted with documented limitations.
- E000 and CP-B0512/1024/1536/2048 training, three-domain evaluation, plots,
  hashes, and paper records are complete.
- Obsolete active ADR code and superseded dataset/config paths were removed;
  recoverability and historical decisions remain documented.
