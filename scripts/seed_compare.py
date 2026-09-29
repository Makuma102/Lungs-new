"""Seed-averaged comparison: average each patient's Dice over a model's runs, then one paired Wilcoxon over the 63 patients.

  python scripts/seed_compare.py --a v1=results/cv_gpu_lungcrop,results/cv_gpu_lungcrop_s1,results/cv_gpu_lungcrop_s2 \
      --b v2=results/cv_gpu_anat_lungcrop,results/cv_gpu_anat_lungcrop_s1,results/cv_gpu_anat_lungcrop_s2 --out results/seed_compare_v1_v2.json
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy import stats


def runs(spec):
    name, dirs = spec.split("=", 1)
    out = []
    for d in dirs.split(","):
        rows = {}
        for f in sorted(glob.glob(os.path.join(d, "fold[0-9]", "test_eval.json"))):
            for r in json.load(open(f))["cases"]:
                rows[r["case"]] = r["ours"]
        out.append(rows)
    return name, out


def boot(d, n=5000, seed=0):
    d = np.asarray(d, float)
    m = np.random.default_rng(seed).choice(d, (n, len(d))).mean(1)
    return [float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def pooled(rows):
    v = list(rows.values())
    return {"dice": float(np.mean([r["dice"] for r in v])), "fp_per_scan": sum(r["fp"] for r in v) / len(v),
            "detected": int(sum(bool(r["detected"]) for r in v))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--out", required=True)
    x = ap.parse_args()
    (na, ra), (nb, rb) = runs(x.a), runs(x.b)
    cases = sorted(set.intersection(*[set(r) for r in ra + rb]))
    ma = np.array([np.mean([r[c]["dice"] for r in ra]) for c in cases])
    mb = np.array([np.mean([r[c]["dice"] for r in rb]) for c in cases])
    d = mb - ma
    runs_a = [pooled(r) for r in ra]; runs_b = [pooled(r) for r in rb]
    sd = lambda rr: float(np.std([p["dice"] for p in rr], ddof=1)) if len(rr) > 1 else float("nan")
    out = {"n": len(cases), "names": [na, nb],
           "runs": {na: runs_a, nb: runs_b},
           "mean_over_runs": {na: [float(np.mean([p["dice"] for p in runs_a])), sd(runs_a)], nb: [float(np.mean([p["dice"] for p in runs_b])), sd(runs_b)]},
           "seed_averaged_paired": {"diff_ci": boot(d), "wilcoxon_p": float(stats.wilcoxon(d).pvalue), "n_better": int((d > 0).sum()), "n_worse": int((d < 0).sum())}}
    json.dump(out, open(x.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=1))


if __name__ == "__main__":
    main()
