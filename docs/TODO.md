# TODO

This file contains the active queue and milestone summary only. Run history is
in `EXPERIMENT_LOG.md`; accepted methodology is in `DECISIONS.md`; exact paper
values are in `METHODOLOGY_TRACEABILITY.md`.

## Active — SDXL full-synthetic foreground/background follow-up

- [x] Preserve the executed SDXL matrix and prohibit target-test-driven
  revision of its generator.
- [x] Implement a review-gated all-class pilot runner for independent background
  generation, foreground generation, SAM3 masking, pairing, harmonization, and
  final mask-derived labels; keep production mechanically locked.
- [x] Generate the initial 32-background/32-foreground candidate pools; verify
  64/64 image integrity, unique hashes, resolution, and exact class counts.
- [x] Delete the old local GenAI synthetic images on researcher request while
  preserving cut-paste data and historical detector results.
- [x] Run an isolated shape-guided SDXL pilot: six initial images and two
  revised Pliers images. Scissors and 9V Battery improve in object count and
  broad silhouette; Pliers improves in the revised pilot but still has
  geometry/artifact defects.
- [ ] Refine and visually approve the three difficult foreground classes;
  regenerate a revised all-class pilot only after they pass.
- [x] Replace the production background prompt schedule with explicit support,
  layout, wear, occupancy, camera, and lighting axes; add a same-policy organic
  all-class retry before freezing production.
- [x] Add real-pixel-free synthetic Canny retry geometry for Scissors,
  Cigarettes, Saw, and Battery after text-only retries failed shape/count QC.
- [x] Harden the v2 implementation with provenance-bound resume, structured
  hard review gates, automatic person screening, reviewed support rectangles,
  class-specific scale, unique foreground prompt schedules, perceptual duplicate
  checks, balanced nested manifests, and a finalizer integrity gate.
- [x] Add a read-only source review audit and reject review rows whose class,
  scene, or image path differs from the immutable manifest.
- [x] Order the revised stages foreground identity → reviewed segmentation →
  diverse backgrounds → clean composition → optional light-only derivation.
  Background generation is gated by all-class foreground acceptance; the
  production scene schedule checks independent diversity axes.
- [x] Implement a separate experimental 75% clean / 25% light derivation that
  requires a reviewed 2,048-image clean release and preserves clean originals.
- [ ] Visually review any derived degradation set and compare its detector
  result against the clean set under a predeclared evaluation protocol.
- [x] Generate and inspect the historical v2 pilot candidate pool: 64
  backgrounds and 204 foregrounds. Its local files were subsequently deleted
  on researcher request; no candidate was manually accepted.
- [ ] Decide whether filled-proxy controlled retry earns approval from its
  separate experimental pilot; do not use it in production before approval.
- [ ] Compose and review the 16-image all-class pilot before changing the
  production lock.
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

- [x] Train and evaluate the clean-only SDXL-C0512/C2048 ablations; record
  their three-domain results alongside the mixed-degradation runs.
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
  without retuning the executed SDXL generator using target-test results; the
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
- [x] Implement mask-derived annotations and the FB2048 finalizer integrity
  checks; validate them on the composition pilot before production.
- [ ] Capture physical GPU assignment, NVIDIA driver, CUDA/software environment,
  and wall time for every remaining paper run.

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
- Superseded dataset/config paths were removed.
