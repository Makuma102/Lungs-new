"""Population statistics of the anatomy reconstructions (data/recon/<case>/anatomy.json).

Reports lung and lobe volumes, airway-tree measures and vessel volumes, the lobe
that contains each expert tumor, and whether the out-of-fold predicted tumor was
placed in the same lobe. Only reconstructions made with a cross-validation
checkpoint count (so every predicted tumor is out of fold).

  python scripts/anatomy_stats.py --out results/anatomy63.json
"""
import argparse
import collections
import glob
import json
import os

import numpy as np

LOBES = ["lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right", "lung_upper_lobe_left", "lung_lower_lobe_left"]


def iqr(x):
    x = np.asarray(x, float)
    return {"median": float(np.median(x)), "q1": float(np.percentile(x, 25)), "q3": float(np.percentile(x, 75)), "n": int(len(x))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", default="data/recon")
    ap.add_argument("--ckpt-filter", default="runs/cv_gpu/", help="keep reconstructions made with these checkpoints")
    ap.add_argument("--out", default="results/anatomy63.json")
    a = ap.parse_args()
    reps = {}
    for f in sorted(glob.glob(os.path.join(a.recon, "*", "anatomy.json"))):
        r = json.load(open(f))
        if a.ckpt_filter in (r.get("pred_metrics") or {}).get("ckpt", ""):
            reps[os.path.basename(os.path.dirname(f))] = r
    st = lambda r, k: r["structures"].get(k, {}).get("volume_ml")
    lung = [sum(st(r, k) or 0 for k in LOBES) / 1000 for r in reps.values()]
    frac = {k: iqr([(st(r, k) or 0) / max(1e-6, sum(st(r, j) or 0 for j in LOBES)) for r in reps.values()]) for k in LOBES}
    aw = [r.get("airway_tree") or {} for r in reps.values()]
    gt_lobe = collections.Counter((r.get("tumor_gt") or {}).get("lobe") or "none" for r in reps.values())
    pairs = [((r.get("tumor_gt") or {}).get("lobe"), (r.get("tumor_pred") or {}).get("lobe")) for r in reps.values()]
    both = [(g, p) for g, p in pairs if g and p]
    out = {"n": len(reps), "cases": sorted(reps),
           "n_all_five_lobes": int(sum(all(st(r, k) for k in LOBES) for r in reps.values())),
           "lung_volume_L": iqr(lung), "lobe_fraction": frac,
           "airway_branchpoints": iqr([x["n_branchpoints"] for x in aw if "n_branchpoints" in x]),
           "airway_endpoints": iqr([x["n_endpoints"] for x in aw if "n_endpoints" in x]),
           "airway_max_generation": iqr([x["max_generation"] for x in aw if "max_generation" in x]),
           "airway_length_m": iqr([x["length_mm"] / 1000 for x in aw if "length_mm" in x]),
           "arteries_ml": iqr([st(r, "arteries") for r in reps.values() if st(r, "arteries")]),
           "veins_ml": iqr([st(r, "veins") for r in reps.values() if st(r, "veins")]),
           "tumor_lobe_gt": dict(gt_lobe),
           "lobe_agreement": {"n_with_both": len(both), "agree": int(sum(g == p for g, p in both)),
                              "rate": float(np.mean([g == p for g, p in both])) if both else None},
           "n_pred_missing": int(sum(1 for g, p in pairs if g and not p))}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("cases", "lobe_fraction")}, indent=1))


if __name__ == "__main__":
    main()
