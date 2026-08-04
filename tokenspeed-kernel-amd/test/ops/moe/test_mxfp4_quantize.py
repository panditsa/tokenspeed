# Copyright (c) 2026 LightSeek Foundation

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused import (
    _quantize_mxfp4_activation,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.quantize import (
    quantize_mxfp4_activation_gluon,
)


def _is_gfx950() -> bool:
    return torch.cuda.is_available() and "gfx950" in getattr(
        torch.cuda.get_device_properties(0), "gcnArchName", ""
    )


pytestmark = pytest.mark.skipif(not _is_gfx950(), reason="requires gfx950")


def test_quantize_matches_existing_implementation() -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260802)
    hidden = torch.randn(
        32, 1024, device="cuda", dtype=torch.bfloat16, generator=generator
    )

    actual = quantize_mxfp4_activation_gluon(hidden)
    expected = _quantize_mxfp4_activation(hidden)

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


def test_quantize_accepts_zero_tokens() -> None:
    hidden = torch.empty((0, 1024), device="cuda", dtype=torch.bfloat16)

    values, scales = quantize_mxfp4_activation_gluon(hidden)

    assert values.shape == (0, 512)
    assert scales.shape == (1024, 0)
