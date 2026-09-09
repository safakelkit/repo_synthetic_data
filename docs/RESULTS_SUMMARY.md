# Results Summary

Only verified seed-0 runs with source-validation-selected `best.pt` are listed.
All models use pretrained YOLO11s, 60 epochs, image size 640, batch 16, and the
same clean/easy/hard reporting protocol. Target test results did not select
checkpoints or revise the executed generators.

## Preliminary report figures

The preliminary report states 71.66% mAP on INSP-DET and 9.93% mAP on
INSP-MOT-DET hard, but does not identify the exact metric, configuration, seed,
or artifact. These values establish motivation only.

## Verified primary results

| ID | Generator | Synthetic | Clean | Delta | Easy | Delta | Hard | Delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| E000 | Real only | 0 | 0.688407 | 0 | 0.414322 | 0 | 0.111021 | 0 |
| CP-B0512 | Cut-paste | 512 | 0.678076 | -0.010331 | 0.455601 | +0.041279 | 0.127215 | +0.016194 |
| CP-B1024 | Cut-paste | 1,024 | 0.686670 | -0.001737 | 0.459600 | +0.045278 | 0.131242 | +0.020221 |
| CP-B1536 | Cut-paste | 1,536 | 0.686050 | -0.002357 | 0.455120 | +0.040799 | 0.161818 | +0.050797 |
| CP-B2048 | Cut-paste | 2,048 | 0.675311 | -0.013097 | 0.470741 | +0.056419 | 0.146286 | +0.035265 |
| SDXL-M0512 | SDXL mixed degradation | 512 | 0.697123 | +0.008716 | 0.409061 | -0.005261 | 0.121801 | +0.010779 |
| SDXL-M1024 | SDXL mixed degradation | 1,024 | 0.684565 | -0.003843 | 0.420388 | +0.006066 | 0.118238 | +0.007217 |
| SDXL-M1536 | SDXL mixed degradation | 1,536 | 0.673075 | -0.015333 | 0.433494 | +0.019172 | 0.099994 | -0.011027 |
| SDXL-M2048 | SDXL mixed degradation | 2,048 | 0.664612 | -0.023796 | 0.428149 | +0.013827 | 0.090928 | -0.020093 |

Evidence:

- baseline: `runs/evaluation/baseline/`;
- cut-paste: `runs/evaluation/cut_paste/`;
- SDXL: `runs/evaluation/sdxl/mixed_degradation/`;
- combined tables and plot: `runs/evaluation/comparisons/`.

## Quantity response

Every cut-paste quantity improves easy and hard over E000, with a small clean
cost. CP-B1024 best preserves clean, CP-B1536 is strongest on hard, and
CP-B2048 is strongest on easy.

SDXL has a different response. SDXL-M0512 improves clean and hard but slightly
reduces easy. SDXL-M1024 improves easy and hard with a small clean cost.
SDXL-M1536 and M2048 improve easy while reducing both clean and hard. From 512
to 2,048 SDXL images, clean falls 0.032512 and hard falls 0.030872. Increasing
the executed SDXL mixture therefore does not monotonically improve robustness.
At 2,048 images, synthetic samples are 48.0% of the 4,263-image training set;
the fixed synthetic scene/appearance distribution may increasingly outweigh
the real-data distribution. This is an interpretation, not a causal result.

At every equal quantity cut-paste outperforms SDXL on easy. Cut-paste also
outperforms SDXL on hard at every quantity. SDXL-M0512 has the best clean result
of the complete matrix.

## Aerosol can

| ID | Clean | Easy | Hard |
|---|---:|---:|---:|
| E000 | 0.693934 | 0.622940 | 0.232962 |
| CP-B0512 | 0.701947 | 0.610061 | 0.262462 |
| CP-B1024 | 0.698306 | 0.601868 | 0.227695 |
| CP-B1536 | 0.716413 | 0.502200 | 0.265727 |
| CP-B2048 | 0.689066 | 0.630676 | 0.271762 |
| SDXL-M0512 | 0.723058 | 0.642239 | 0.276062 |
| SDXL-M1024 | 0.717595 | 0.657335 | 0.280044 |
| SDXL-M1536 | 0.723839 | 0.678787 | 0.263142 |
| SDXL-M2048 | 0.697802 | 0.654340 | 0.246270 |

The final spray-can intervention transferred successfully: SDXL exceeds E000
for Aerosol on all three domains at every quantity. It also exceeds equal-size
cut-paste on clean and easy at every quantity, and on hard at 512 and 1,024.

## Persistent class failures

Matches, Pliers, Shaver, and Battery are near zero mainly on the hard domain,
including before synthetic augmentation. E000 hard mAP50-95 is respectively
0.0000, 0.0035, 0.0000, and 0.0002. The hard split contains 69 Matches, 29
Pliers, 21 Shaver, and 276 Battery annotations, so Battery failure in particular
cannot be explained by a tiny evaluation count. Pliers and Shaver remain strong
on clean/easy; Matches already fails on easy; Battery is unstable on easy.
Neither generator resolves this hard-domain class collapse.

The complete 432-row method/quantity/domain/class table is
`runs/evaluation/comparisons/augmentation_matrix_per_class.csv`.

## Interpretation rules

- Compare generators at equal synthetic quantities.
- Report every completed configuration and clean-domain cost.
- Do not use easy/hard results to revise these executed generator versions.
- The initial matrix has one detector seed; variance and significance require
  additional predeclared seeds.
- Missing hard-domain Lighter AP is unavailable, not zero.
- ADR claims remain outside the current baseline phase.
