# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Gluon MXFP4 activation quantization for gfx950."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon

MXFP4_BLOCK = 32
_M_SWIZZLE = 32
_K_SWIZZLE = 8


@gluon.jit
def quantize_mxfp4_tile(values):
    """Quantize ``[..., 32]`` groups to packed E2M1 and E8M0 scales."""
    block_m: gl.constexpr = values.shape[0]
    block_n: gl.constexpr = values.shape[1]
    groups: gl.constexpr = block_n // 32
    gl.static_assert(block_n % 32 == 0)

    values = values.to(gl.bfloat16).to(gl.float32).reshape((block_m, groups, 32))
    bits = values.to(gl.uint32, bitcast=True)
    abs_values = (bits & 0x7FFFFFFF).to(gl.float32, bitcast=True)
    amax = gl.max(abs_values, axis=2, keep_dims=True)
    rounded = (amax.to(gl.uint32, bitcast=True) + 0x200000) & 0x7F800000
    scale_i = gl.minimum(gl.maximum((rounded >> 23).to(gl.int32) - 2, 0), 254)
    scale = scale_i.to(gl.uint8).reshape((block_m, groups))

    inv_scale = ((254 - scale_i) << 23).to(gl.float32, bitcast=True)
    qbits = (values * inv_scale).to(gl.uint32, bitcast=True)
    sign = qbits & 0x80000000
    magnitude = qbits ^ sign
    magnitude_f32 = magnitude.to(gl.float32, bitcast=True)
    saturated = magnitude_f32 >= 6.0
    denormal = (not saturated) & (magnitude_f32 < 1.0)
    normal = not (saturated | denormal)

    denorm_bias_i: gl.constexpr = ((127 - 1) + (23 - 1) + 1) << 23
    denorm_bias: gl.constexpr = gl.cast(denorm_bias_i, gl.float32, bitcast=True)
    denorm_code = (magnitude_f32 + denorm_bias).to(gl.uint32, bitcast=True)
    denorm_code = (denorm_code - denorm_bias_i).to(gl.uint8)

    mantissa_odd = (magnitude >> 22) & 1
    normal_code = ((magnitude + 0xC11FFFFF + mantissa_odd) >> 22).to(gl.uint8)
    e2m1 = gl.full(values.shape, 0x7, gl.uint8, layout=values.type.layout)
    e2m1 = gl.where(normal, normal_code, e2m1)
    e2m1 = gl.where(denormal, denorm_code, e2m1)
    e2m1 |= (sign >> 28).to(gl.uint8)
    e2m1 = e2m1.reshape((block_m, groups, 16, 2))
    low, high = gl.split(e2m1)
    return low | (high << 4), scale


@gluon.jit
def store_cdna4_scale(
    ptr,
    scale,
    row,
    group,
    stride_k,
    stride_m,
    mask,
):
    """Store E8M0 scales in the layout consumed by CDNA4 scaled MFMA."""
    row_in_block = row % 32
    row_hi = row_in_block // 16
    row_lo = row_in_block % 16
    group_block = group // 8
    group_in_block = group % 8
    group_hi = group_in_block // 4
    group_lo = group_in_block % 4
    swizzled_group = (
        ((group_block * 4 + group_lo) * 16 + row_lo) * 2 + group_hi
    ) * 2 + row_hi
    gl.store(
        ptr
        + swizzled_group.to(gl.int64) * stride_k
        + (row // 32).to(gl.int64) * stride_m,
        scale,
        mask=mask,
    )


@gluon.jit
def _quantize_mxfp4_activation_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    rows,
    cols,
    stride_xm,
    stride_xk,
    stride_om,
    stride_ok,
    stride_sk,
    stride_sm,
):
    row = gl.program_id(0)
    group = gl.program_id(1)
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [1, 1], [1, 0])
    row_lane = gl.arange(0, 1, layout=gl.SliceLayout(1, layout))[:, None]
    offsets = group * 32 + gl.arange(0, 32, layout=gl.SliceLayout(0, layout))[None, :]
    values = gl.load(
        x_ptr + (row + row_lane).to(gl.int64) * stride_xm + offsets * stride_xk,
        mask=(row < rows) & (offsets < cols),
        other=0.0,
    )
    packed, scale = quantize_mxfp4_tile(values)
    packed = packed.reshape((1, 16))
    packed_row = gl.arange(0, 1, layout=gl.SliceLayout(1, packed.type.layout))[:, None]
    packed_offsets = (
        group * 16
        + gl.arange(0, 16, layout=gl.SliceLayout(0, packed.type.layout))[None, :]
    )
    gl.store(
        out_ptr
        + (row + packed_row).to(gl.int64) * stride_om
        + packed_offsets * stride_ok,
        packed,
        mask=row < rows,
    )
    scale_row = gl.arange(0, 1, layout=gl.SliceLayout(1, scale.type.layout))[:, None]
    scale_group = gl.arange(0, 1, layout=gl.SliceLayout(0, scale.type.layout))[None, :]
    store_cdna4_scale(
        scale_ptr,
        scale,
        row + scale_row,
        group + scale_group,
        stride_sk,
        stride_sm,
        (row < rows) & (group < (cols // 32)),
    )


def empty_cdna4_scale(rows: int, groups: int, device: torch.device) -> torch.Tensor:
    groups_padded = (groups + _K_SWIZZLE - 1) // _K_SWIZZLE * _K_SWIZZLE
    rows_padded = (rows + _M_SWIZZLE - 1) // _M_SWIZZLE * _M_SWIZZLE
    shape = (groups_padded * _M_SWIZZLE, rows_padded // _M_SWIZZLE)
    return torch.empty_strided(shape, (1, shape[0]), dtype=torch.uint8, device=device)


def quantize_mxfp4_activation_gluon(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.dtype != torch.bfloat16 or x.ndim != 2 or not x.is_contiguous():
        raise ValueError(
            "MXFP4 Gluon quantization requires contiguous rank-2 BF16 input"
        )
    rows, cols = map(int, x.shape)
    if cols % MXFP4_BLOCK:
        raise ValueError("MXFP4 Gluon quantization requires K divisible by 32")
    out = torch.empty((rows, cols // 2), dtype=torch.uint8, device=x.device)
    scale = empty_cdna4_scale(rows, cols // MXFP4_BLOCK, x.device)
    if rows == 0:
        return out, scale
    _quantize_mxfp4_activation_kernel[(rows, cols // MXFP4_BLOCK)](
        x,
        out,
        scale,
        rows,
        cols,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        scale.stride(0),
        scale.stride(1),
        num_warps=1,
    )
    return out, scale


__all__ = [
    "empty_cdna4_scale",
    "quantize_mxfp4_activation_gluon",
    "quantize_mxfp4_tile",
    "store_cdna4_scale",
]
