# Lungs: compact 3D U-Net for lung tumor segmentation with anatomy-level 3D reconstruction

A reproducible pipeline for chest CT that
1. segments lung tumors with a compact 3D U-Net (3.2 M parameters, trainable on a CPU or a single consumer GPU),
2. reconstructs the patient's anatomy in 3D (five lobes, airway tree with centreline and branch generations, pulmonary arteries and veins, expert and predicted tumor) as OBJ meshes in patient RAS millimetres plus a web bundle for the included viewer, and
3. evaluates everything with controls that expose false success: native-resolution metrics, patient-level bootstrap confidence intervals, an intensity-threshold baseline, an empty prediction, a shuffled-label control and a resampling ceiling, backed by an automated test suite.

> Research prototype. Not a medical device; it does not diagnose cancer.

## Results (MSD Task06, 63 labelled CTs, five-fold cross-validation)
| Model | Tumor Dice (95% CI) | Params |
|---|---|---|
| Compact 3D U-Net, default post-processing | 0.532 (0.461–0.597) | 3.2 M |
| Compact 3D U-Net, validation-selected post-processing | 0.579 | 3.2 M |
| nnU-Net 3d_fullres (250 epochs), same folds | 0.639 | 30.8 M |
| SwinUNETR (same 20-epoch budget), same folds | 0.472 | 62.2 M |

Three training runs of the compact model: 0.530 ± 0.009. All numbers come from the JSON files in `results/` (`results/cv_gpu_lungcrop/cv_summary.json`, `results/model_comparison.json`, `results/seed_compare_v1_v2.json`).

## Data
| Dataset | Use | Access |
|---|---|---|
| MSD Task06_Lung (63 labelled CTs, CC-BY-SA 4.0) | tumor training and testing | `python scripts/download_datasets.py msd` |
| TotalSegmentator models | lobes, trachea, airways, arteries, veins | `pip install totalsegmentator` |

## Setup (Linux or Windows + WSL2, NVIDIA GPU optional)
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # RTX 50-series needs CUDA 12.8+
pip install -r requirements.txt totalsegmentator
pytest -q
```

## Run everything (resumable)
```bash
scripts/run_all_gpu.sh
```
Downloads MSD Task06, preprocesses it, runs TotalSegmentator, the five-fold cross-validation, the anatomical-lung ablation and the reconstructions. Every step skips finished work; training resumes from `runs/*/last.pt`.

## Single steps
```bash
# cross-validation with the final settings
scripts/run_cv.sh
# reconstruct one patient (meshes + report + web bundle in data/recon/<case>/)
python scripts/reconstruct_case.py lung_014
# view the 3D viewer
python -m http.server -d viewer 8000     # open http://localhost:8000
# nnU-Net baseline on the same folds
cat scripts/nnunet/README.md
```

## Repository layout
| Path | Content |
|---|---|
| `lungseg/` | preprocessing, model, training, inference, metrics |
| `scripts/` | download, cross-validation, evaluation, statistics, reconstruction, viewer export |
| `tests/` | metric, split, geometry and inference tests |
| `runs/` | trained checkpoints and training logs |
| `results/` | evaluation outputs used for every reported number |
| `viewer/` | interactive three.js viewer with example reconstructions |

Raw data (`data/`) and prediction caches are not tracked; `scripts/run_all_gpu.sh` recreates them.

## License and data
Code: see repository. MSD Task06 data are CC-BY-SA 4.0 and are not redistributed here.
