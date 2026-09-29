"""Does the preprocessing crop depend on the tumour label? For every case compare the crop box built from
lung | tumour (current code) with the box from the label-free lung mask, and count tumour voxels outside
the label-free box. Writes results/crop_leak_check.json."""
import glob
import json
import os
import sys
from multiprocessing import Pool

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.expanduser("~/projects/Lungs"))
from lungseg.prep_msd import anatomical_lung, lungs_from_hu  # noqa: E402


def box(mask, zooms, shape):
    nz = np.argwhere(mask)
    m = np.ceil(10 / zooms).astype(int)
    return np.maximum(nz.min(0) - m, 0), np.minimum(nz.max(0) + m + 1, shape)


def check(img_path):
    case = os.path.basename(img_path).replace(".nii.gz", "")
    ct = nib.as_closest_canonical(nib.load(img_path))
    hu = np.asanyarray(ct.dataobj).astype(np.float32)
    tum = np.asanyarray(nib.as_closest_canonical(nib.load(img_path.replace("imagesTr", "labelsTr"))).dataobj) > 0
    zooms = np.array(ct.header.get_zooms()[:3], float)
    out = {"case": case, "tumor_vox": int(tum.sum())}
    for src in ("threshold", "anat"):
        lung = anatomical_lung(img_path) if src == "anat" else np.moveaxis(lungs_from_hu(np.moveaxis(hu, 2, 0)), 0, 2)
        lo1, hi1 = box(lung | tum, zooms, hu.shape)
        lo0, hi0 = box(lung, zooms, hu.shape)
        inside = np.zeros_like(tum); inside[lo0[0]:hi0[0], lo0[1]:hi0[1], lo0[2]:hi0[2]] = True
        out[src] = {"box_changed": bool((lo1 != lo0).any() or (hi1 != hi0).any()),
                    "tumor_vox_outside_labelfree_box": int((tum & ~inside).sum()),
                    "box_with_tumor": [lo1.tolist(), hi1.tolist()], "box_label_free": [lo0.tolist(), hi0.tolist()]}
    return out


if __name__ == "__main__":
    os.chdir(os.path.expanduser("~/projects/Lungs"))
    files = sorted(glob.glob("data/Task06_Lung/imagesTr/lung_*.nii.gz"))
    with Pool(2) as p:
        rows = p.map(check, files)
    summ = {src: {"n_box_changed": sum(r[src]["box_changed"] for r in rows),
                  "n_cases_tumor_outside": sum(r[src]["tumor_vox_outside_labelfree_box"] > 0 for r in rows),
                  "tumor_vox_outside_total": sum(r[src]["tumor_vox_outside_labelfree_box"] for r in rows),
                  "tumor_vox_total": sum(r["tumor_vox"] for r in rows),
                  "cases_changed": [r["case"] for r in rows if r[src]["box_changed"]]} for src in ("threshold", "anat")}
    json.dump({"summary": summ, "cases": rows}, open("results/crop_leak_check.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))
