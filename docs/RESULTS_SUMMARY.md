# Results Summary

Only add results produced by a verified run with a traceable configuration and artifact path. Keep preliminary report figures separate from reproducible project results. ADR results are not part of the current phase.

Placement preprocessing is not a detector result. The pilot geometry review
accepted 45 regions. Full geometry-v2 review later retained 527
production-eligible regions across 527 of the 1,166 backgrounds. None of these
steps created training images.

Degradation-v1 implementation is also preprocessing, not an experimental
result. Its exact schedules and deterministic image-transform tests passed, but
no degraded training image has yet been released.

## Preliminary report figures

The preliminary report states:

- INSP-DET: 71.66% mAP.
- INSP-MOT-DET hard: 9.93% mAP.

The report does not identify the precise mAP variant, run configuration, seed, or artifact path for these figures. They establish motivation only and must not be presented as verified mAP50-95 results until provenance is recovered.

## Verified primary results

| ID | Generator | Synthetic count | Seed(s) | INSP-DET mAP50-95 | Easy mAP50-95 | Hard mAP50-95 | Evidence |
|---|---|---:|---:|---:|---:|---:|---|
| E000 | None | 0 | 0 initial | TBD | TBD | TBD | TBD |
| CP-B0512 | Cut-paste | 512 | detector 0 / generator 42 | TBD | TBD | TBD | TBD |
| CP-B1024 | Cut-paste | 1,024 | detector 0 / generator 42 | TBD | TBD | TBD | TBD |
| CP-B1536 | Cut-paste | 1,536 | detector 0 / generator 42 | TBD | TBD | TBD | TBD |
| CP-B2048 | Cut-paste | 2,048 | detector 0 / generator 42 | TBD | TBD | TBD | TBD |
| SD-B0512 | Stable Diffusion + ControlNet | 512 | TBD | TBD | TBD | TBD | TBD |
| SD-B1024 | Stable Diffusion + ControlNet | 1,024 | TBD | TBD | TBD | TBD | TBD |
| SD-B1536 | Stable Diffusion + ControlNet | 1,536 | TBD | TBD | TBD | TBD | TBD |
| SD-B2048 | Stable Diffusion + ControlNet | 2,048 | TBD | TBD | TBD | TBD | TBD |
| QW-B0512 | Qwen + ControlNet | 512 | TBD | TBD | TBD | TBD | TBD |
| QW-B1024 | Qwen + ControlNet | 1,024 | TBD | TBD | TBD | TBD | TBD |
| QW-B1536 | Qwen + ControlNet | 1,536 | TBD | TBD | TBD | TBD | TBD |
| QW-B2048 | Qwen + ControlNet | 2,048 | TBD | TBD | TBD | TBD | TBD |

## Required comparisons

- Primary success criterion: improvement over E000 on both easy and hard mAP50-95, subject to the clean-domain preservation tolerance.
- Cut-paste results are not valid for the accepted methodology unless their
  dataset provenance identifies the verified semantic support-region manifest
  and class orientation/support policy used during placement.
- Every synthetic run versus E000.
- Quantity response within each generator: 512 versus 1,024 versus 1,536 versus 2,048.
- Cut-paste versus Stable Diffusion + ControlNet versus Qwen + ControlNet at each equal quantity.
- Target-domain gain versus INSP-DET change.
- Class-wise gains and losses for every completed run.

## Class-wise findings

TBD. Add only findings supported by saved class-wise evaluation outputs.

## Interpretation rules

- Do not compare unequal synthetic counts as evidence that one generator is better.
- Do not select checkpoints or generator settings using target test results.
- Report all completed valid configurations, not only the best result.
- Report failed and invalid runs in the experiment log.
- Report variability across seeds, not only the best seed.
- Report source-domain cost alongside target-domain gains.
- Do not make ADR claims during the current baseline phase.
