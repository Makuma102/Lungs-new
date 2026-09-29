"""Model factory: every architecture builds, keeps the patch shape, and predicts a volume."""
import importlib.util

import numpy as np
import pytest
import torch

from lungseg.model import ARCHS, build_from_args, build_model
from lungseg.reconstruct3d import predict_volume

HAS_MONAI = importlib.util.find_spec("monai") is not None


@pytest.mark.parametrize("arch", ARCHS)
def test_build_model_output_shape(arch):
    if arch != "ours" and not HAS_MONAI:
        pytest.skip("MONAI not installed")
    m = build_model(arch).eval()
    with torch.no_grad():
        y = m(torch.zeros(1, 1, 64, 96, 96))
    assert y.shape == (1, 3, 64, 96, 96)


def test_build_from_args_defaults_to_ours():
    m = build_from_args({"base": 4, "isotropic": True})  # checkpoints saved before --arch existed
    assert type(m).__name__ == "SmallUNet3D"


@pytest.mark.skipif(not HAS_MONAI, reason="MONAI not installed")
def test_predict_volume_with_monai_baseline():
    m = build_model("attention_unet")
    labels, prob = predict_volume(m, np.random.rand(20, 30, 30).astype(np.float32))
    assert labels.shape == prob.shape == (20, 30, 30)


def test_unknown_arch_raises():
    with pytest.raises(ValueError):
        build_model("resnet")
