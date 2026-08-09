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

"""Route-direct gfx950 A4W4 warp decode experiments."""

from __future__ import annotations

import torch

from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.decode_kernels import (
    _cdna4_swizzled_mxfp4_scale_offset,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.quantize import (
    empty_cdna4_scale,
    quantize_mxfp4_activation_gluon,
    quantize_mxfp4_tile,
    store_cdna4_scale,
)

_LANES = gl.constexpr(64)
_STAGE1_BLOCK_N = 32
_STAGE1_BLOCK_KB = 1024
_STAGE1_NUM_WARPS = 16
_STAGE2_BLOCK_N = 16
_STAGE2_BLOCK_KB = 512
_STAGE2_NUM_WARPS = 8


@gluon.jit
def _stage1_a4w4_situ_warp_gemv_quant(
    hidden_ptr,
    hidden_scale_ptr,
    w13_ptr,
    w13_scale_ptr,
    inter_ptr,
    inter_scale_ptr,
    topk_ids_ptr,
    hidden_dim,
    intermediate_dim,
    stride_hm,
    stride_hk,
    stride_hslin,
    stride_hsnb,
    stride_we,
    stride_wk,
    stride_wn,
    stride_se,
    stride_wsn,
    stride_wsk,
    stride_im,
    stride_ik,
    stride_islin,
    stride_isnb,
    stride_idm,
    stride_ids,
    TOP_K: gl.constexpr,
    SITU_BETA: gl.constexpr,
    SITU_LINEAR_BETA: gl.constexpr,
    EXPERT_START: gl.constexpr,
    NUM_LOCAL_EXPERTS: gl.constexpr,
    W13_INTERLEAVED: gl.constexpr,
    NUM_PID_N: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_KB: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    route = gl.program_id(0) // NUM_PID_N
    pid_n = gl.program_id(0) % NUM_PID_N
    token = route // TOP_K
    slot = route % TOP_K
    expert = (
        gl.load(topk_ids_ptr + token * stride_idm + slot * stride_ids) - EXPERT_START
    )
    if expert < 0:
        return
    if expert >= NUM_LOCAL_EXPERTS:
        return

    layout: gl.constexpr = gl.BlockedLayout(
        [(BLOCK_N + NUM_WARPS - 1) // NUM_WARPS, BLOCK_KB // _LANES],
        [1, _LANES],
        [NUM_WARPS, 1],
        [1, 0],
    )
    n_layout: gl.constexpr = gl.SliceLayout(1, layout)
    k_layout: gl.constexpr = gl.SliceLayout(0, layout)
    expanded_layout: gl.constexpr = gl.BlockedLayout(
        [(BLOCK_N + NUM_WARPS - 1) // NUM_WARPS, (2 * BLOCK_KB) // _LANES],
        [1, _LANES],
        [NUM_WARPS, 1],
        [1, 0],
    )
    expanded_n_layout: gl.constexpr = gl.SliceLayout(1, expanded_layout)
    expanded_k_layout: gl.constexpr = gl.SliceLayout(0, expanded_layout)
    offs_n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=n_layout)
    expanded_offs_n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=expanded_n_layout)
    if W13_INTERLEAVED:
        gate_col = 2 * offs_n
        up_col = gate_col + 1
        expanded_gate_col = 2 * expanded_offs_n
        expanded_up_col = expanded_gate_col + 1
    else:
        gate_col = offs_n
        up_col = intermediate_dim + offs_n
        expanded_gate_col = expanded_offs_n
        expanded_up_col = intermediate_dim + expanded_offs_n

    packed_k = hidden_dim // 2
    hidden_row = token.to(gl.int64) * stride_hm
    w_expert = expert.to(gl.int64) * stride_we
    scale_expert = expert.to(gl.int64) * stride_se
    gate_acc = gl.zeros([BLOCK_N], gl.float32, expanded_n_layout)
    up_acc = gl.zeros([BLOCK_N], gl.float32, expanded_n_layout)

    for kb0 in range(0, packed_k, BLOCK_KB):
        offs_kb = kb0 + gl.arange(0, BLOCK_KB, layout=k_layout)
        expanded_k = 2 * kb0 + gl.arange(0, 2 * BLOCK_KB, layout=expanded_k_layout)
        packed_valid = offs_kb < packed_k
        expanded_valid = expanded_k < hidden_dim
        hidden_offsets = hidden_row + offs_kb.to(gl.int64) * stride_hk
        hidden_scale_offsets = _cdna4_swizzled_mxfp4_scale_offset(
            0,
            token,
            expanded_k // 32,
            stride_hslin,
            stride_hsnb,
        )
        hidden = gl.amd.cdna4.scaled_upcast(
            gl.amd.cdna4.buffer_load(
                ptr=hidden_ptr,
                offsets=hidden_offsets.to(gl.int32),
                mask=packed_valid,
                other=0,
            ),
            gl.amd.cdna4.buffer_load(
                ptr=hidden_scale_ptr,
                offsets=hidden_scale_offsets.to(gl.int32),
                mask=expanded_valid,
                other=0,
            ),
            gl.bfloat16,
            axis=0,
        )
        hidden = gl.convert_layout(hidden, expanded_k_layout)
        hidden = gl.convert_layout(hidden[None, :], expanded_layout)
        gate_offsets = (
            w_expert
            + offs_kb[None, :].to(gl.int64) * stride_wk
            + gate_col[:, None].to(gl.int64) * stride_wn
        )
        up_offsets = (
            w_expert
            + offs_kb[None, :].to(gl.int64) * stride_wk
            + up_col[:, None].to(gl.int64) * stride_wn
        )
        gate_scale_offsets = (
            scale_expert
            + expanded_gate_col[:, None].to(gl.int64) * stride_wsn
            + (expanded_k[None, :] // 32).to(gl.int64) * stride_wsk
        )
        up_scale_offsets = (
            scale_expert
            + expanded_up_col[:, None].to(gl.int64) * stride_wsn
            + (expanded_k[None, :] // 32).to(gl.int64) * stride_wsk
        )
        gate = gl.amd.cdna4.scaled_upcast(
            gl.amd.cdna4.buffer_load(
                ptr=w13_ptr,
                offsets=gate_offsets.to(gl.int32),
                mask=packed_valid[None, :],
                other=0,
            ),
            gl.amd.cdna4.buffer_load(
                ptr=w13_scale_ptr,
                offsets=gate_scale_offsets.to(gl.int32),
                mask=expanded_valid[None, :],
                other=0,
            ),
            gl.bfloat16,
            axis=1,
        )
        gate_acc += gl.sum(gate.to(gl.float32) * hidden, axis=1)
        up = gl.amd.cdna4.scaled_upcast(
            gl.amd.cdna4.buffer_load(
                ptr=w13_ptr,
                offsets=up_offsets.to(gl.int32),
                mask=packed_valid[None, :],
                other=0,
            ),
            gl.amd.cdna4.buffer_load(
                ptr=w13_scale_ptr,
                offsets=up_scale_offsets.to(gl.int32),
                mask=expanded_valid[None, :],
                other=0,
            ),
            gl.bfloat16,
            axis=1,
        )
        up_acc += gl.sum(up.to(gl.float32) * hidden, axis=1)

    gate = gate_acc.to(gl.bfloat16).to(gl.float32)
    up = up_acc.to(gl.bfloat16).to(gl.float32)
    gate = (
        SITU_BETA
        * gl.extra.libdevice.tanh(gate / SITU_BETA)
        * (1.0 / (1.0 + gl.exp(-gate)))
    )
    up = SITU_LINEAR_BETA * gl.extra.libdevice.tanh(up / SITU_LINEAR_BETA)
    activated = gl.permute((gate * up)[:, None], [1, 0])
    packed, scale = quantize_mxfp4_tile(activated)

    packed = packed.reshape((1, BLOCK_N // 2))
    packed_m = gl.arange(0, 1, layout=gl.SliceLayout(1, packed.type.layout))[:, None]
    packed_n = gl.arange(0, BLOCK_N // 2, layout=gl.SliceLayout(0, packed.type.layout))[
        None, :
    ]
    gl.store(
        inter_ptr
        + route.to(gl.int64) * stride_im
        + (pid_n * (BLOCK_N // 2) + packed_n).to(gl.int64) * stride_ik,
        packed,
        mask=packed_m == 0,
    )
    scale_m = gl.arange(0, 1, layout=gl.SliceLayout(1, scale.type.layout))[:, None]
    scale_n = gl.arange(0, BLOCK_N // 32, layout=gl.SliceLayout(0, scale.type.layout))[
        None, :
    ]
    store_cdna4_scale(
        inter_scale_ptr,
        scale,
        route + scale_m,
        pid_n * (BLOCK_N // 32) + scale_n,
        stride_islin,
        stride_isnb,
        scale_m == 0,
    )


def invoke_stage1_a4w4_situ_warp_decode_gluon(
    hidden: torch.Tensor,
    hidden_scale: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    inter: torch.Tensor,
    inter_scale: torch.Tensor,
    *,
    situ_beta: float = 4.0,
    situ_linear_beta: float = 25.0,
    expert_start: int = 0,
    w13_interleaved: bool = False,
    block_n: int = _STAGE1_BLOCK_N,
    block_kb: int = _STAGE1_BLOCK_KB,
    num_warps: int = _STAGE1_NUM_WARPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute linear-layout A4W4 W13, SiTU, and MXFP4 requantization."""
    if hidden.dtype != torch.uint8 or hidden_scale.dtype != torch.uint8:
        raise TypeError("A4W4 warp decode requires packed activation and scale bytes")
    if w13.dtype != torch.uint8 or w13_scale.dtype != torch.uint8:
        raise TypeError("A4W4 warp decode requires packed weight and scale bytes")
    if inter.dtype != torch.uint8 or inter_scale.dtype != torch.uint8:
        raise TypeError("A4W4 warp decode requires packed intermediate buffers")
    tokens, packed_hidden = map(int, hidden.shape)
    experts, two_intermediate, weight_packed_hidden = map(int, w13.shape)
    intermediate = two_intermediate // 2
    hidden_dim = packed_hidden * 2
    top_k = int(topk_ids.shape[1])
    if two_intermediate % 2 or weight_packed_hidden != packed_hidden:
        raise ValueError("linear W13 shape mismatch")
    if tuple(w13_scale.shape) != (experts, two_intermediate, hidden_dim // 32):
        raise ValueError("linear W13 scale shape mismatch")
    if tuple(inter.shape) != (tokens * top_k, intermediate // 2):
        raise ValueError("packed intermediate shape mismatch")
    if topk_ids.shape[0] != tokens:
        raise ValueError("top-k tensor shape mismatch")
    if block_n % 32 or intermediate % block_n:
        raise ValueError("stage1 output tile must divide N and be divisible by 32")
    if block_kb % 64:
        raise ValueError("stage1 packed-K tile must be lane aligned")
    if any(
        not tensor.is_contiguous()
        for tensor in (hidden, w13, w13_scale, topk_ids, inter)
    ):
        raise ValueError("A4W4 warp decode tensors must be contiguous")

    topk_ids = topk_ids.to(torch.int32)
    grid = tokens * top_k * (intermediate // block_n)
    _stage1_a4w4_situ_warp_gemv_quant[(grid,)](
        hidden,
        hidden_scale,
        w13,
        w13_scale,
        inter,
        inter_scale,
        topk_ids,
        hidden_dim,
        intermediate,
        hidden.stride(0),
        hidden.stride(1),
        hidden_scale.stride(0),
        hidden_scale.stride(1),
        w13.stride(0),
        w13.stride(2),
        w13.stride(1),
        w13_scale.stride(0),
        w13_scale.stride(1),
        w13_scale.stride(2),
        inter.stride(0),
        inter.stride(1),
        inter_scale.stride(0),
        inter_scale.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        TOP_K=top_k,
        SITU_BETA=float(situ_beta),
        SITU_LINEAR_BETA=float(situ_linear_beta),
        EXPERT_START=int(expert_start),
        NUM_LOCAL_EXPERTS=experts,
        W13_INTERLEAVED=w13_interleaved,
        NUM_PID_N=intermediate // block_n,
        BLOCK_N=block_n,
        BLOCK_KB=block_kb,
        NUM_WARPS=num_warps,
        num_warps=num_warps,
    )
    return inter, inter_scale


@gluon.jit
def _stage2_a4w4_warp_gemv_combine(
    inter_ptr,
    inter_scale_ptr,
    w2_ptr,
    w2_scale_ptr,
    out_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    hidden_dim,
    intermediate_dim,
    stride_ipm,
    stride_ipk,
    stride_islin,
    stride_isnb,
    stride_we,
    stride_wk,
    stride_wn,
    stride_se,
    stride_wsn,
    stride_wsk,
    stride_om,
    stride_on,
    stride_idm,
    stride_ids,
    stride_twm,
    stride_tws,
    TOP_K: gl.constexpr,
    EXPERT_START: gl.constexpr,
    NUM_LOCAL_EXPERTS: gl.constexpr,
    NUM_PID_N: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_KB: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    pid = gl.program_id(0)
    token = pid // NUM_PID_N
    pid_n = pid % NUM_PID_N
    layout: gl.constexpr = gl.BlockedLayout(
        [(BLOCK_N + NUM_WARPS - 1) // NUM_WARPS, BLOCK_KB // _LANES],
        [1, _LANES],
        [NUM_WARPS, 1],
        [1, 0],
    )
    n_layout: gl.constexpr = gl.SliceLayout(1, layout)
    k_layout: gl.constexpr = gl.SliceLayout(0, layout)
    expanded_layout: gl.constexpr = gl.BlockedLayout(
        [(BLOCK_N + NUM_WARPS - 1) // NUM_WARPS, (2 * BLOCK_KB) // _LANES],
        [1, _LANES],
        [NUM_WARPS, 1],
        [1, 0],
    )
    expanded_n_layout: gl.constexpr = gl.SliceLayout(1, expanded_layout)
    expanded_k_layout: gl.constexpr = gl.SliceLayout(0, expanded_layout)
    offs_n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=n_layout)
    expanded_offs_n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=expanded_n_layout)
    packed_k = intermediate_dim // 2
    acc = gl.zeros([BLOCK_N], gl.float32, expanded_n_layout)

    for slot in gl.static_range(0, TOP_K):
        expert = (
            gl.load(topk_ids_ptr + token * stride_idm + slot * stride_ids)
            - EXPERT_START
        )
        if (expert >= 0) & (expert < NUM_LOCAL_EXPERTS):
            route_weight = gl.load(
                topk_weights_ptr + token * stride_twm + slot * stride_tws
            ).to(gl.float32)
            row = token * TOP_K + slot
            inter_row = row.to(gl.int64) * stride_ipm
            w_expert = expert.to(gl.int64) * stride_we
            scale_expert = expert.to(gl.int64) * stride_se
            route_acc = gl.zeros([BLOCK_N], gl.float32, expanded_n_layout)
            for kb0 in range(0, packed_k, BLOCK_KB):
                offs_kb = kb0 + gl.arange(0, BLOCK_KB, layout=k_layout)
                expanded_k = 2 * kb0 + gl.arange(
                    0, 2 * BLOCK_KB, layout=expanded_k_layout
                )
                inter_packed = gl.amd.cdna4.buffer_load(
                    ptr=inter_ptr,
                    offsets=(inter_row + offs_kb * stride_ipk).to(gl.int32),
                )
                inter_scale_offsets = _cdna4_swizzled_mxfp4_scale_offset(
                    0,
                    row,
                    expanded_k // 32,
                    stride_islin,
                    stride_isnb,
                )
                inter = gl.amd.cdna4.scaled_upcast(
                    inter_packed,
                    gl.amd.cdna4.buffer_load(
                        ptr=inter_scale_ptr,
                        offsets=inter_scale_offsets.to(gl.int32),
                    ),
                    gl.bfloat16,
                    axis=0,
                )
                inter = gl.convert_layout(inter, expanded_k_layout)
                inter = gl.convert_layout(inter[None, :], expanded_layout)
                w_offsets = (
                    w_expert
                    + offs_kb[None, :].to(gl.int64) * stride_wk
                    + offs_n[:, None].to(gl.int64) * stride_wn
                )
                scale_offsets = (
                    scale_expert
                    + expanded_offs_n[:, None].to(gl.int64) * stride_wsn
                    + (expanded_k[None, :] // 32).to(gl.int64) * stride_wsk
                )
                weight = gl.amd.cdna4.scaled_upcast(
                    gl.amd.cdna4.buffer_load(
                        ptr=w2_ptr,
                        offsets=w_offsets.to(gl.int32),
                    ),
                    gl.amd.cdna4.buffer_load(
                        ptr=w2_scale_ptr,
                        offsets=scale_offsets.to(gl.int32),
                    ),
                    gl.bfloat16,
                    axis=1,
                )
                route_acc += gl.sum(weight.to(gl.float32) * inter, axis=1)
            acc += route_weight * route_acc.to(gl.bfloat16).to(gl.float32)

    gl.store(
        out_ptr + token * stride_om + expanded_offs_n * stride_on,
        acc.to(out_ptr.dtype.element_ty),
    )


def invoke_stage2_a4w4_warp_decode_gluon(
    inter: torch.Tensor,
    inter_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    out: torch.Tensor,
    *,
    expert_start: int = 0,
    block_n: int = _STAGE2_BLOCK_N,
    block_kb: int = _STAGE2_BLOCK_KB,
    num_warps: int = _STAGE2_NUM_WARPS,
) -> torch.Tensor:
    """Compute linear-layout A4W4 W2 with route-direct warp GEMV."""
    if inter.dtype != torch.uint8 or inter_scale.dtype != torch.uint8:
        raise TypeError("A4W4 warp decode requires packed activation and scale bytes")
    if w2.dtype != torch.uint8 or w2_scale.dtype != torch.uint8:
        raise TypeError("A4W4 warp decode requires packed weight and scale bytes")
    if out.dtype != torch.bfloat16 or out.ndim != 2:
        raise TypeError("A4W4 warp decode output must be rank-2 BF16")
    num_tokens, hidden_dim = map(int, out.shape)
    top_k = int(topk_ids.shape[1])
    num_experts, weight_hidden, packed_intermediate = map(int, w2.shape)
    intermediate_dim = packed_intermediate * 2
    if weight_hidden != hidden_dim:
        raise ValueError("W2 output dimension does not match the output tensor")
    if tuple(inter.shape) != (num_tokens * top_k, packed_intermediate):
        raise ValueError("packed intermediate shape mismatch")
    if tuple(w2_scale.shape) != (
        num_experts,
        hidden_dim,
        intermediate_dim // 32,
    ):
        raise ValueError("linear W2 scale shape mismatch")
    if topk_ids.shape != topk_weights.shape or topk_ids.shape[0] != num_tokens:
        raise ValueError("top-k tensor shape mismatch")
    if hidden_dim % block_n or packed_intermediate % block_kb:
        raise ValueError("A4W4 warp decode requires exact N and packed-K tiles")
    if any(
        not tensor.is_contiguous()
        for tensor in (inter, w2, w2_scale, topk_ids, topk_weights, out)
    ):
        raise ValueError("A4W4 warp decode tensors must be contiguous")

    topk_ids = topk_ids.to(torch.int32)
    grid = num_tokens * (hidden_dim // block_n)
    _stage2_a4w4_warp_gemv_combine[(grid,)](
        inter,
        inter_scale,
        w2,
        w2_scale,
        out,
        topk_ids,
        topk_weights,
        hidden_dim,
        intermediate_dim,
        inter.stride(0),
        inter.stride(1),
        inter_scale.stride(0),
        inter_scale.stride(1),
        w2.stride(0),
        w2.stride(2),
        w2.stride(1),
        w2_scale.stride(0),
        w2_scale.stride(1),
        w2_scale.stride(2),
        out.stride(0),
        out.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        TOP_K=top_k,
        EXPERT_START=int(expert_start),
        NUM_LOCAL_EXPERTS=num_experts,
        NUM_PID_N=hidden_dim // block_n,
        BLOCK_N=block_n,
        BLOCK_KB=block_kb,
        NUM_WARPS=num_warps,
        num_warps=num_warps,
    )
    return out


def gluon_a4w4_situ_warp_decode_ep_gfx950(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    situ_beta: float = 4.0,
    situ_linear_beta: float = 25.0,
    expert_start: int = 0,
    w13_interleaved: bool = False,
    stage1_block_n: int = _STAGE1_BLOCK_N,
    stage1_block_kb: int = _STAGE1_BLOCK_KB,
    stage1_num_warps: int = _STAGE1_NUM_WARPS,
    stage2_block_n: int = _STAGE2_BLOCK_N,
    stage2_block_kb: int = _STAGE2_BLOCK_KB,
    stage2_num_warps: int = _STAGE2_NUM_WARPS,
) -> torch.Tensor:
    """Run route-direct linear-layout A4W4 W13/SiTU/W2 warp decode."""
    if hidden_states.dtype != torch.bfloat16 or not hidden_states.is_contiguous():
        raise ValueError("A4W4 warp decode requires contiguous BF16 hidden states")
    tokens, hidden_dim = map(int, hidden_states.shape)
    top_k = int(topk_ids.shape[1])
    intermediate = int(w13.shape[1]) // 2
    if tuple(w2.shape) != (int(w13.shape[0]), hidden_dim, intermediate // 2):
        raise ValueError("linear W2 shape mismatch")

    hidden, hidden_scale = quantize_mxfp4_activation_gluon(hidden_states)
    inter = torch.empty(
        (tokens * top_k, intermediate // 2),
        dtype=torch.uint8,
        device=hidden_states.device,
    )
    inter_scale = empty_cdna4_scale(
        tokens * top_k, intermediate // 32, hidden_states.device
    )
    invoke_stage1_a4w4_situ_warp_decode_gluon(
        hidden,
        hidden_scale,
        w13,
        w13_scale,
        topk_ids,
        inter,
        inter_scale,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        expert_start=expert_start,
        w13_interleaved=w13_interleaved,
        block_n=stage1_block_n,
        block_kb=stage1_block_kb,
        num_warps=stage1_num_warps,
    )
    out = torch.empty_like(hidden_states)
    return invoke_stage2_a4w4_warp_decode_gluon(
        inter,
        inter_scale,
        w2,
        w2_scale,
        topk_ids,
        topk_weights,
        out,
        expert_start=expert_start,
        block_n=stage2_block_n,
        block_kb=stage2_block_kb,
        num_warps=stage2_num_warps,
    )


__all__ = [
    "gluon_a4w4_situ_warp_decode_ep_gfx950",
    "invoke_stage1_a4w4_situ_warp_decode_gluon",
    "invoke_stage2_a4w4_warp_decode_gluon",
]
