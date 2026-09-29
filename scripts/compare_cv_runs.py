"""Run-to-run reproducibility: compare two cross-validation runs of the same
configuration (e.g. CPU fp32 vs GPU mixed precision) patient by patient, on
every fold that both runs have completed.

  python scripts/compare_cv_runs.py results/cv results/cv_gpu --names CPU GPU --out results/cv_gpu/replication.json
"""
import argparse
import json
import os

import numpy as np
from scipy import stats


def boot(d, n=5000, seed=0):
    d = np.asarray(d, float)
    m = np.random.default_rng(seed).choice(d, (n, len(d))).mean(1)
    return [float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--names", nargs=2, default=["A", "B"])
    ap.add_argument("--out", required=True)
    x = ap.parse_args()
    folds, rows, rules = [], [], []
    for k in range(10):
        fa, fb = (os.path.join(d, f"fold{k}", "test_eval.json") for d in (x.a, x.b))
        if not (os.path.exists(fa) and os.path.exists(fb)):
            continue
        A, B = json.load(open(fa)), json.load(open(fb))
        ca = {r["case"]: r for r in A["cases"]}; cb = {r["case"]: r for r in B["cases"]}
        assert set(ca) == set(cb), f"fold {k}: different patients"
        folds.append(k)
        for c in sorted(ca):
            rows.append({"fold": k, "case": c,
                         **{f"{key}_{n}": r[c][key]["dice"] for key in ("ours", "ours_sel") for n, r in zip(("a", "b"), (ca, cb))},
                         **{f"det_{n}": bool(r[c]["ours"]["detected"]) for n, r in zip(("a", "b"), (ca, cb))}})
        ra, rb = (json.load(open(os.path.join(d, f"fold{k}", "postproc_selection.json")))["selected"]["rule"] for d in (x.a, x.b))
        rules.append({"fold": k, "a": ra, "b": rb, "same": ra == rb})
    out = {"names": x.names, "folds": folds, "n_patients": len(rows), "rules": rules}
    for key in ("ours", "ours_sel"):
        a = np.array([r[f"{key}_a"] for r in rows]); b = np.array([r[f"{key}_b"] for r in rows]); d = b - a
        out[key] = {"mean_a": float(a.mean()), "mean_b": float(b.mean()), "diff_ci": boot(d),
                    "wilcoxon_p": float(stats.wilcoxon(b, a).pvalue) if np.any(d != 0) else 1.0,
                    "mean_abs_diff": float(np.abs(d).mean()), "max_abs_diff": float(np.abs(d).max()),
                    "frac_abs_diff_gt_0.1": float(np.mean(np.abs(d) > 0.1)),
                    "pearson_r": float(np.corrcoef(a, b)[0, 1])}
    da = np.array([r["det_a"] for r in rows]); db = np.array([r["det_b"] for r in rows])
    out["detection"] = {"both": int((da & db).sum()), "neither": int((~da & ~db).sum()), "discordant": int((da ^ db).sum())}
    out["rows"] = rows
    json.dump(out, open(x.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
