# TODO

This file contains the active queue and milestone summary only. Run history is
in `EXPERIMENT_LOG.md`; accepted methodology is in `DECISIONS.md`; exact paper
values are in `METHODOLOGY_TRACEABILITY.md`.

## Active — SDXL full-synthetic foreground/background follow-up

- [x] Preserve the executed SDXL matrix and prohibit target-test-driven
  revision of its generator.
- [x] Implement and visually validate a real-background/real-object localized
  SDXL harmonization pilot; classify it as a hybrid ablation rather than a pure
  GenAI replacement.
- [ ] Freeze the v2 foreground and background prompt/control schedules before
  production; final pixels must not come from real object or background assets.
- [ ] Generate at least 2,560 background candidates and accept exactly 2,048
  genuinely distinct, target-free scenes (one per final image).
- [ ] Generate at least 2,592 isolated-object candidates, with a 50% candidate
  surplus for Aerosol, and accept exactly 128 foregrounds per class.
- [ ] Segment every accepted foreground, reject malformed/duplicate/wrong-class
  candidates, and retain the mask used to derive the final YOLO box.
- [ ] Pair accepted backgrounds and foregrounds one-to-one, use support-aware
  placement, and apply only localized diffusion harmonization.
- [ ] Verify 2,048 unique background layouts, 2,048 unique foreground instances,
  exact 128/class balance, labels, hashes, image integrity, and visual quality.
- [ ] Release nested 512/1,024/1,536/2,048 manifests only after the frozen QC
  gate passes; then run a separately named detector comparison.

## Completed — SDXL result analysis

- [x] Reset failed generation versions, outputs, and detailed failure logs while
  preserving successful profiles and reusable methodology decisions.
- [x] Build and preflight the historical one-pass `sdxl_generation_v1` pilot (superseded by the canonical pipeline).
- [x] Preserve the successful class-specific SDXL/Canny profiles for 15 classes.
- [x] Validate structured background generation, target-only Canny geometry,
  support-aware scale, and the upright spray-can intervention.
- [x] Generate and approve 2,048 clean SDXL images and pose-derived YOLO labels.
- [x] Derive and approve the paired mixed-degradation dataset with exact
  25/37.5/25/12.5% clean/light/medium/heavy allocation.
- [x] Train and evaluate SDXL-M0512/1024/1536/2048 with frozen YOLO11s.
- [x] Compare E000, cut-paste, and SDXL overall and per-class results.
- [ ] Determine whether the additional detector-seed experiment should focus
  on E000, CP-B1536, and SDXL-M0512 before making stability claims.
- [x] Analyze the hard-domain collapse for Matches, Pliers, Shaver, and Battery
  without retuning the executed SDXL generator on target-test feedback; the
  prospective remedy is broader source generation and is not selected from
  target-test outcomes.

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

- [ ] Complete the remaining Qwen configurations in the 13-configuration table.
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
- The canonical SDXL clean/mixed dataset passed complete structural QC and
  researcher review; all four mixed-degradation quantity runs and three-domain
  evaluations are complete.
- Evaluation artifacts are grouped by baseline, cut-paste, SDXL, and combined
  comparisons under `runs/evaluation/`.
- Obsolete active ADR code and superseded dataset/config paths were removed;
  recoverability and historical decisions remain documented.
