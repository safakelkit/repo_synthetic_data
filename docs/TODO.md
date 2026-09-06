# TODO

This file contains the active queue and milestone summary only. Run history is
in `EXPERIMENT_LOG.md`; accepted methodology is in `DECISIONS.md`; exact paper
values are in `METHODOLOGY_TRACEABILITY.md` and the private paper record.

## Active — SDXL baseline

- [x] Reset failed generation versions, outputs, and detailed failure logs while
  preserving successful profiles and reusable methodology decisions.
- [x] Build and preflight the one-pass `sdxl_generation_v1` pipeline.
- [x] Preserve the successful class-specific SDXL/Canny profiles for 15 classes.
- [ ] Run and review the 64-image acceptance test; require at least 3/4 per
  class and at least 56/64 overall.
- [ ] Define the next Aerosol intervention only if its full-scene text branch
  fails the frozen acceptance rule.
- [ ] Implement post-generation localization, annotation, degradation, and QC.
- [ ] Generate and validate nested SD-B0512/1024/1536/2048 datasets.
- [ ] Train and evaluate all four SDXL configurations with frozen YOLO11s.

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
