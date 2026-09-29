"""Compare several models trained and tested on the identical 5 patient folds.

Each model is a directory with fold0..fold4/test_eval.json (our CV runs, or
scripts/nnunet/score_predictions.py output). Per-patient Dice is pooled over all
folds; every baseline is compared with the reference model patient by patient
(difference = baseline - reference, bootstrap 95% CI, two-sided Wilcoxon).
Optional cost figures come from measure_cost.py output and training logs.

  python scripts/compare_models.py --ref ours=results/cv_gpu \
      --models nnunet=results/nnunet swinunetr=results/cv_gpu_swinunetr --out results/model_comparison.json
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy import stats


def boot(d, n=5000, seed=0):
    d = np.asarray(d, float)
    m = np.random.default_rng(seed).choice(d, (n, len(d))).mean(1)
    return [float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def load(d):
    rows = {}
    for f in sorted(glob.glob(os.path.join(d, "fold[0-9]", "test_eval.json"))):
        for r in json.load(open(f))["cases"]:
            rows[r["case"]] = r["ours"]  # default output of that model (no test-time rule selection)
    return rows


def summary(rows):
    v = list(rows.values())
    tp = sum(r["tp"] for r in v); fn = sum(r["fn"] for r in v); fp = sum(r["fp"] for r in v)
    return {"n": len(v), "dice": boot([r["dice"] for r in v]), "median_dice": float(np.median([r["dice"] for r in v])),
            "hd95": boot([r["hd95"] for r in v if np.isfinite(r["hd95"])]),
            "lesion_sensitivity": tp / max(1, tp + fn), "fp_per_scan": fp / len(v),
            "case_detection_rate": float(np.mean([r["detected"] for r in v]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="name=dir of the reference model")
    ap.add_argument("--models", nargs="+", required=True, help="name=dir for each baseline")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rname, rdir = a.ref.split("=", 1)
    ref = load(rdir)
    out = {"reference": rname, "models": {rname: summary(ref)}, "paired": {}}
    for spec in a.models:
        name, d = spec.split("=", 1)
        rows = load(d)
        if len(rows) < 5:
            print(f"skip {name}: {len(rows)} patients scored"); continue
        common = sorted(set(rows) & set(ref))
        diff = np.array([rows[c]["dice"] - ref[c]["dice"] for c in common])
        p = float(stats.wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
        out["models"][name] = summary(rows)
        # case-level detection, paired: exact McNemar test on the discordant patients
        det_b = np.array([bool(rows[c]["detected"]) for c in common]); det_r = np.array([bool(ref[c]["detected"]) for c in common])
        only_b, only_r = int((det_b & ~det_r).sum()), int((~det_b & det_r).sum())
        mcn = float(stats.binomtest(only_b, only_b + only_r, 0.5).pvalue) if only_b + only_r else 1.0
        out["paired"][name] = {"n": len(common), "diff_ci": boot(diff), "wilcoxon_p": p,
                               "detected": [int(det_b.sum()), int(det_r.sum())], "detected_only_model": only_b, "detected_only_ref": only_r, "mcnemar_p": mcn,
                               "n_better": int((diff > 0).sum()), "n_worse": int((diff < 0).sum()),
                               "complete": len(common) == len(ref)}
        print(name, len(common), "patients: diff", [round(x, 3) for x in out["paired"][name]["diff_ci"]], "p", round(p, 4))
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
