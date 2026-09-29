# 5-fold CV on the GPU (RTX 5070, 12 GB), branch `local-gpu`

All five folds trained and evaluated on the GPU with mixed precision; configuration identical to the
CPU run in `results/cv/` (same `scripts/run_cv.sh` arguments, same seed-42 patient folds).

| Path | Content |
|---|---|
| `fold0..fold4/` | per-fold post-processing selection (validation only) and test evaluation |
| `cv_summary.json` | pooled 63-patient summary: tumor Dice 0.548 (95% CI 0.484-0.607) |
| `clinical.json` | diameter / volume / lobe agreement on the out-of-fold predictions |
| `all_cases.json` | the 63 per-patient test rows, for paired comparisons |
| `replication.json` | earlier CPU-vs-GPU comparison of folds 1-4 |
| `*_MIXED_fold0cpu_folds1to4gpu.json` | superseded: pooled CPU fold 0 with GPU folds 1-4, before fold 0 was retrained on the GPU |

Checkpoints are in `runs/cv_gpu/fold*/`. The v2 (anatomical lung class) ablation on the same folds is in
`results/cv_gpu_anat/` and the paired v1-vs-v2 test in `results/compare_v1_v2_cv_gpu.json`.
