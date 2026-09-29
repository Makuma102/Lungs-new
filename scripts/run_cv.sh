#!/bin/bash
# Resumable 5-fold cross-validation on all 63 MSD cases (v1 configuration).
# Each fold: train (resumes from last.pt) -> validation-only post-proc selection -> test evaluation.
# Override PREP/RUNS/RES to run another configuration, e.g. the anatomical-lung ablation:
#   PREP=data/prep_anat RUNS=runs/cv_gpu_anat RES=results/cv_gpu_anat scripts/run_cv.sh
# ARCH selects the network (ours, swinunetr, attention_unet, unet_monai); everything else stays identical:
#   ARCH=swinunetr RUNS=runs/cv_gpu_swinunetr RES=results/cv_gpu_swinunetr scripts/run_cv.sh
set -u
PREP=${PREP:-data/prep}; RUNS=${RUNS:-runs/cv}; RES=${RES:-results/cv}; ARCH=${ARCH:-ours}
# TRAIN_SEED changes only the training randomness (repeat runs on the same folds)
EXTRA=${TRAIN_SEED:+--train-seed $TRAIN_SEED}
for k in 0 1 2 3 4; do
  R=$RUNS/fold$k; O=$RES/fold$k; mkdir -p $R $O
  if [ ! -f $R/results.json ]; then
    python -m lungseg.train --arch $ARCH --data msd3d --root $PREP --dim 3 --isotropic --patch 64 96 96 --base 12 --batch 2 \
      --epochs 20 --samples-per-volume 8 --tumor-oversample 0.66 --lr 3e-3 --n-train 45 --n-val 5 --n-test 13 \
      --val-every 4 --seed 42 --n-folds 5 --fold $k $EXTRA --resume --out $R >> $R/train.log 2>&1 || { echo "fold $k train failed"; exit 1; }
  fi
  [ -f $O/postproc_selection.json ] || python scripts/select_postproc.py --ckpt $R/best.pt --out $O/postproc_selection.json >> $O/select.log 2>&1
  [ -f $O/test_eval.json ] || python scripts/evaluate_msd.py --ckpt $R/best.pt --out $O --postproc $O/postproc_selection.json >> $O/eval.log 2>&1
  echo "fold $k done $(date)"
done
echo CV_DONE
