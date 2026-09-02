# Results Summary

Only add results produced by a verified run with a traceable configuration and artifact path. Keep preliminary report figures separate from reproducible project results. ADR results are not part of the current phase.

Placement preprocessing is not a detector result. The pilot geometry review
accepted 45 regions. Full geometry-v2 review later retained 527
production-eligible regions across 527 of the 1,166 backgrounds. None of these
steps created training images.

`cp_v1_seed42` is a dataset artifact, not a detector result. It contains 2,048
unique images, passed automatic integrity/QC, and was accepted before training
as a deliberately simple cut-paste baseline with documented placement,
perspective, scale, and contact limitations.

## Preliminary report figures

The preliminary report states:

- INSP-DET: 71.66% mAP.
- INSP-MOT-DET hard: 9.93% mAP.

The report does not identify the precise mAP variant, run configuration, seed, or artifact path for these figures. They establish motivation only and must not be presented as verified mAP50-95 results until provenance is recovered.

## Verified primary results

| ID | Generator | Synthetic count | Seed(s) | INSP-DET mAP50-95 | Easy mAP50-95 | Hard mAP50-95 | Evidence |
|---|---|---:|---:|---:|---:|---:|---|
| E000 | None | 0 | 0 initial | 0.688407 | 0.414322 | 0.111021 | `runs/evaluation/E000_results.json` |
| CP-B0512 | Cut-paste | 512 | detector 0 / generator 42 | 0.678076 | 0.455601 | 0.127215 | `runs/evaluation/CP-B0512_yolo11s_seed0_results.json` |
| CP-B1024 | Cut-paste | 1,024 | detector 0 / generator 42 | 0.686670 | 0.459600 | 0.131242 | `runs/evaluation/CP-B1024_yolo11s_seed0_results.json` |
| CP-B1536 | Cut-paste | 1,536 | detector 0 / generator 42 | 0.686050 | 0.455120 | 0.161818 | `runs/evaluation/CP-B1536_yolo11s_seed0_results.json` |
| CP-B2048 | Cut-paste | 2,048 | detector 0 / generator 42 | 0.675311 | 0.470741 | 0.146286 | `runs/evaluation/CP-B2048_yolo11s_seed0_results.json` |
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

For E000, easy-domain mAP50-95 is highest for Alcohol (0.7522), Shaver
(0.7479), Pliers (0.7143), Scissors (0.6309), and Aerosol can (0.6229). It is
lowest for Matches (0.0053), Battery (0.1610), and Lighter (0.1636).

Hard-domain mAP50-95 is highest for Laptop (0.5327), Alcohol (0.3239), Aerosol
can (0.2330), Hammer (0.2047), and Scissors (0.1395). Matches, Knife, and Shaver
are 0.0; Battery is 0.0002 and Screwdriver is 0.0014. Lighter is unavailable,
not zero, because the hard split contains no class-0 annotations.

The absolute mAP50-95 drop from clean is 0.2741 on easy and 0.5774 on hard
(39.8% and 83.9% relative, respectively). These E000 values are the fixed
reference for every synthetic-data comparison.

### Cut-paste quantity response

All four valid seed-0 cut-paste runs improve both target domains over E000.
Easy gains are +0.0413, +0.0453, +0.0408, and +0.0564; hard gains are
+0.0162, +0.0202, +0.0508, and +0.0353 for 512 through 2,048 synthetic images.
Clean changes are -0.0103, -0.0017, -0.0024, and -0.0131. CP-B1024 offers the
smallest clean cost, CP-B1536 has the largest hard gain, and CP-B2048 has the
largest easy gain. The response is not monotonic, so the current one-seed
matrix does not support a claim that more cut-paste data is always better.

Combined numeric and visual evidence is stored in
`runs/evaluation/copy_paste_matrix_summary.csv` and
`runs/evaluation/copy_paste_matrix_summary.png`.

## Interpretation rules

- Do not compare unequal synthetic counts as evidence that one generator is better.
- Do not select checkpoints or generator settings using target test results.
- Report all completed valid configurations, not only the best result.
- Report failed and invalid runs in the experiment log.
- Report variability across seeds, not only the best seed.
- Report source-domain cost alongside target-domain gains.
- Do not make ADR claims during the current baseline phase.
