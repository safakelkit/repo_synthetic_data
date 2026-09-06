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
- [ ] Resolve the remaining Aerosol can zero-yield failure without using
  easy/hard feedback; preserve the accepted Pliers and viable 9V/Laptop rules.
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
- [ ] Test a different Aerosol conditioning design without changing the class,
  model pair, scene policy, or target-test boundary.
- [x] Prepare v6 scene-only Canny pilot so ControlNet constrains the MAIJA
  support surface while the prompt determines Aerosol identity.
- [ ] Run/review Aerosol v6 before deciding the canonical conditioning policy.
- [ ] Define a fixed candidate-surplus policy and report attempts, class-wise
  acceptance rate, runtime, and rejection reasons.
- [ ] Implement post-generation localization/annotation and automatic QC.
- [ ] Freeze acceptance thresholds and a limited manual-review protocol using
  pilot outputs only.
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
- [ ] Final candidate-surplus ratio after class-wise pilot yield is measured.
- [ ] Exact automatic annotation/QC thresholds and manual-review sample policy.
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
