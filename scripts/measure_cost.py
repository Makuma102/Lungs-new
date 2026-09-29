"""Computational cost of segmentation models, for the accuracy-vs-cost comparison.

Reports trainable parameters, forward GFLOPs for one training patch and for a
full sliding-window inference on a typical preprocessed CT (window count x
TTA), and optionally measured latency and peak GPU memory.

  python scripts/measure_cost.py                        # our model, FLOPs + params only
  python scripts/measure_cost.py --time --device cuda   # + latency / peak memory (use an idle GPU)
  python scripts/measure_cost.py --models ours swinunetr unet_monai attention_unet   # needs `pip install monai`

Writes results/cost.json. Timing numbers are only meaningful on an otherwise idle machine.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lungseg.model import ARCHS, build_model, count_parameters  # noqa: E402

PATCH = (64, 96, 96)          # training patch
INFER = (96, 128, 128)        # sliding-window patch at inference
TYPICAL = (207, 162, 215)     # measured median preprocessed CT over the 63 MSD cases (1.5 mm, lung crop), z,y,x


def build(name, plans=None):
    """Same configurations as training (lungseg.model.build_model), so costs match the trained baselines.
    'nnunet' is rebuilt from the nnU-Net plans file of our run (3d_fullres), with deep supervision off as at inference."""
    if name != "nnunet":
        return build_model(name, base=12, isotropic=True)
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    cm = PlansManager(plans).get_configuration("3d_fullres")
    return get_network_from_plans(cm.network_arch_class_name, cm.network_arch_init_kwargs, cm.network_arch_init_kwargs_req_import,
                                  1, 2, allow_init=True, deep_supervision=False)


def setting(name, plans=None):
    """Patch sizes, batch, typical volume and TTA each model actually uses."""
    if name != "nnunet":
        return {"patch": PATCH, "infer": INFER, "typical": TYPICAL, "batch": 2, "tta": 2}
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    cm = PlansManager(plans).get_configuration("3d_fullres")
    ps = tuple(cm.patch_size)
    # nnU-Net predicts on the native-resolution grid with 8-fold mirroring (3 axes) by default
    return {"patch": ps, "infer": ps, "typical": tuple(int(x) for x in cm.median_image_size_in_voxels), "batch": cm.batch_size, "tta": 8}


def train_step_mem(m, shape, batch, device):
    """Peak memory of one mixed-precision training step (forward + backward), as in training."""
    m.train(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    x = torch.randn(batch, 1, *shape, device=device)
    with torch.autocast("cuda", dtype=torch.float16):
        y = m(x)
        loss = (y[0] if isinstance(y, (list, tuple)) else y).float().mean()
    loss.backward(); torch.cuda.synchronize()
    m.zero_grad(set_to_none=True); m.eval()
    return torch.cuda.max_memory_allocated() / 1e9


def gflops(model, shape, device):
    x = torch.zeros(1, 1, *shape, device=device)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops() / 1e9


def n_windows(vol, patch, overlap=0.5):
    n = 1
    for s, p in zip(vol, patch):
        step = max(1, int(p * (1 - overlap)))
        n *= len(set(list(range(0, max(1, s - p + 1), step)) + [max(0, s - p)]))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=ARCHS + ("nnunet",), default=["ours"])
    ap.add_argument("--nnunet-plans", default="nnunet_work/preprocessed/Dataset006_Lung/nnUNetPlans.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads (e.g. 4 to match a 4-core machine)")
    ap.add_argument("--out", default="results/cost.json")
    a = ap.parse_args()
    out = {"patch": PATCH, "infer_patch": INFER, "typical_volume": TYPICAL, "device": a.device, "models": {}}
    if a.device == "cuda":
        out["gpu"] = torch.cuda.get_device_name(0)
    else:
        if a.threads:
            torch.set_num_threads(a.threads)
        cpu = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")] if os.path.exists("/proc/cpuinfo") else []
        out["cpu"] = cpu[0] if cpu else "unknown"; out["threads"] = torch.get_num_threads()
    for name in a.models:
        cfg = setting(name, a.nnunet_plans)
        m = build(name, a.nnunet_plans).to(a.device).eval()
        r = {"params_M": count_parameters(m) / 1e6, "setting": cfg,
             "gflops_train_patch": gflops(m, cfg["patch"], a.device),
             "gflops_infer_window": gflops(m, cfg["infer"], a.device)}
        nw = n_windows(cfg["typical"], cfg["infer"])
        r["windows_per_case"] = nw
        r["gflops_per_case_tta"] = r["gflops_infer_window"] * nw * cfg["tta"]
        if a.time:
            x = torch.randn(1, 1, *cfg["infer"], device=a.device)
            with torch.no_grad():
                for _ in range(2):
                    m(x)
                if a.device == "cuda":
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                ts = []
                for _ in range(5):
                    t0 = time.perf_counter(); m(x)
                    if a.device == "cuda":
                        torch.cuda.synchronize()
                    ts.append(time.perf_counter() - t0)
            r["sec_per_window"] = float(np.median(ts))
            r["est_sec_per_case_tta"] = r["sec_per_window"] * nw * cfg["tta"]
            if a.device == "cuda":
                r["peak_mem_GB_infer"] = torch.cuda.max_memory_allocated() / 1e9
                r["peak_mem_GB_train_step"] = train_step_mem(m, cfg["patch"], cfg["batch"], a.device)
        out["models"][name] = r
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
