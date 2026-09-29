"""GPU training time per fold for every model (results/train_time.json), for the cost table.

Our pipeline (ours, SwinUNETR, ...): sum of logged epoch times in runs/<run>/fold*/results.json.
nnU-Net: sum of "Epoch time" lines in its training logs (only folds with checkpoint_final.pth).

  python scripts/train_time.py --runs ours=runs/cv_gpu swinunetr=runs/cv_gpu_swinunetr \\
      --nnunet nnunet_work/results/Dataset006_Lung/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres
"""
import argparse
import glob
import json
import os
import re

import numpy as np


def summarise(hours):
    return {"folds": len(hours), "gpu_hours_per_fold": float(np.median(hours)), "gpu_hours_total": float(np.sum(hours))} if hours else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[], help="name=runs dir of our pipeline")
    ap.add_argument("--nnunet", default=None, help="nnU-Net trainer results folder")
    ap.add_argument("--out", default="results/train_time.json")
    a = ap.parse_args()
    out = {}
    for spec in a.runs:
        name, d = spec.split("=", 1)
        h = [sum(e["sec"] for e in json.load(open(f))["log"]) / 3600 for f in sorted(glob.glob(os.path.join(d, "fold[0-9]", "results.json")))]
        if summarise(h):
            out[name] = summarise(h)
    if a.nnunet:
        h = []
        for fd in sorted(glob.glob(os.path.join(a.nnunet, "fold_[0-9]"))):
            if not os.path.exists(os.path.join(fd, "checkpoint_final.pth")):
                continue
            secs = [float(x) for f in glob.glob(os.path.join(fd, "training_log*.txt")) for x in re.findall(r"Epoch time: ([0-9.]+) s", open(f).read())]
            h.append(sum(secs) / 3600)
        if summarise(h):
            out["nnunet"] = summarise(h)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
