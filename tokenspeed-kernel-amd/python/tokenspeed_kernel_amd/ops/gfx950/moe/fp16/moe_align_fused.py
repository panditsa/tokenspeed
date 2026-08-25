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

"""Fused small-M MoE block-align (decode): single-kernel, sync-free, pure Gluon.

A single Gluon workgroup performs the full alignment. One kernel:

  * in-kernel sentinel/zero init of the output + ``gl.barrier`` (folds away the
    separate init-kernel launch),
  * EP localization from global expert IDs plus ``gl.histogram`` -> per-expert
    counts (remote routes are masked; EP = num_experts, no dump bin),
  * **single-block collapse**: top-k contains each expert at most once per
    token, so at M <= block_m, count[e] <= M <= block_m and every hit expert is
    exactly one block. So blocks_pe = (count>0), row_off = block_off*block_m, and
    ``sorted_expert_ids`` is a cheap O(E) scatter (no [NB, E] tile),
  * a stable [G, G] compare-tile rank for up to 128 routes, or per-expert
    atomic rank counters for larger route sets,
  * ``gl.gather`` of the per-expert block offset + scatter each slot to
    ``block_off[e]*block_m + rank``.

Sync-free: outputs use the compile-time bound
``EM_MAX = min(M*topk, num_experts)*block_m``; ``num_valid`` (real EM) is
on-device and the GEMM stages early-out on the padded tail. Decode-only: the
stable-rank path grows quadratically with ``M*topk``, while the atomic path
avoids that rank tile for larger route sets.
"""

from __future__ import annotations

import torch

from tokenspeed_kernel_amd._triton import gl, gluon


def _next_pow2(x: int) -> int:
    return 1 << max(0, (x - 1)).bit_length()


@gluon.jit
def _add(a, b):
    return a + b


@gluon.jit
def _fused_align_kernel(
    ids_ptr,  # [G] int32  flat topk_ids
    wts_ptr,  # [G] fp32   flat topk_weights
    sti_ptr,  # [EM_MAX] int32  out (packed slot<<24|token)
    sw_ptr,  # [EM_MAX] fp32   out (routed weight)
    sei_ptr,  # [NB_MAX] int32  out (expert per block, -1 pad)
    nv_ptr,  # [1] int32       out (EM)
    G,
    num_experts,
    block_m,
    sentinel,
    TOPK: gl.constexpr,
    GP: gl.constexpr,  # next_pow2(G)
    EP: gl.constexpr,  # next_pow2(num_experts)
    EXPERT_START: gl.constexpr,
    NB_MAX: gl.constexpr,  # min(G, num_experts) == sei length
    EM_MAX: gl.constexpr,  # == NB_MAX * block_m
    INIT_TILE: gl.constexpr,
):
    LG: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])  # [GP]
    LE: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])  # [EP]
    LR: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])  # init tile
    LT: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [4, 1], [1, 0])  # 2D rank

    # ---- in-kernel init of the full output (folds away a separate launch) ----
    for r0 in gl.static_range(0, EM_MAX, INIT_TILE):
        r = r0 + gl.arange(0, INIT_TILE, layout=LR)
        rm = r < EM_MAX
        gl.store(
            sti_ptr + r, gl.full([INIT_TILE], sentinel, gl.int32, layout=LR), mask=rm
        )
        gl.store(sw_ptr + r, gl.full([INIT_TILE], 0.0, gl.float32, layout=LR), mask=rm)
    jb = gl.arange(0, GP, layout=LG)
    gl.store(sei_ptr + jb, gl.full([GP], -1, gl.int32, layout=LG), mask=jb < NB_MAX)
    gl.barrier()  # order init stores before the (overlapping) scatter below

    g = gl.arange(0, GP, layout=LG)
    gmask = g < G
    global_idx = gl.load(ids_ptr + g, mask=gmask, other=EXPERT_START)
    idx = global_idx - EXPERT_START
    route_mask = gmask & (idx >= 0) & (idx < num_experts)
    safe_idx = gl.where(route_mask, idx, 0)
    vals = gl.load(wts_ptr + g, mask=route_mask, other=0.0)
    tok = g // TOPK
    slot = g % TOPK
    packed = ((slot << 24) | tok).to(gl.int32)

    # ---- per-expert counts (masked histogram -> masked lanes excluded) ----
    counts = gl.histogram(safe_idx, EP, mask=route_mask, layout=LE)
    e = gl.arange(0, EP, layout=LE)
    valid_e = e < num_experts
    # single-block collapse: 1 block per hit expert (count<=M<=block_m)
    hit = valid_e & (counts > 0)
    blocks_pe = hit.to(gl.int32)
    block_off = gl.associative_scan(blocks_pe, 0, _add) - blocks_pe  # exclusive
    num_blocks = gl.sum(blocks_pe, 0)
    gl.store(nv_ptr, num_blocks * block_m)  # EM

    # ---- sorted_expert_ids: scatter e -> sei[block_off[e]] (O(E)) ----
    gl.store(sei_ptr + block_off, e.to(gl.int32), mask=hit)

    # ---- stable per-expert rank via [G, G] compare tile ----
    idx_row = gl.expand_dims(
        gl.convert_layout(safe_idx, gl.SliceLayout(1, LT)),
        1,
    )
    idx_col = gl.expand_dims(
        gl.convert_layout(safe_idx, gl.SliceLayout(0, LT)),
        0,
    )
    valid_row = gl.expand_dims(
        gl.convert_layout(route_mask, gl.SliceLayout(1, LT)),
        1,
    )
    valid_col = gl.expand_dims(
        gl.convert_layout(route_mask, gl.SliceLayout(0, LT)),
        0,
    )
    g_row = gl.expand_dims(gl.arange(0, GP, layout=gl.SliceLayout(1, LT)), 1)
    g_col = gl.expand_dims(gl.arange(0, GP, layout=gl.SliceLayout(0, LT)), 0)
    match = ((idx_row == idx_col) & (g_col < g_row) & valid_row & valid_col).to(
        gl.int32
    )
    rank = gl.convert_layout(gl.sum(match, axis=1), LG)  # [GP]

    # ---- dest = block_off[expert]*block_m + rank, then scatter ----
    dest = gl.gather(block_off, safe_idx, axis=0) * block_m + rank
    gl.store(sti_ptr + dest, packed, mask=route_mask)
    gl.store(sw_ptr + dest, vals, mask=route_mask)


@gluon.jit
def _fused_atomic_align_kernel(
    ids_ptr,
    wts_ptr,
    sti_ptr,
    sw_ptr,
    sei_ptr,
    nv_ptr,
    fill_ctr_ptr,
    G,
    num_experts,
    block_m,
    sentinel,
    TOPK: gl.constexpr,
    GP: gl.constexpr,
    EP: gl.constexpr,
    EXPERT_START: gl.constexpr,
    NB_MAX: gl.constexpr,
    EM_MAX: gl.constexpr,
    INIT_TILE: gl.constexpr,
):
    """Align a larger decode route set in one CTA without a G-by-G rank tile."""
    LG: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    LE: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    LR: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])

    for r0 in gl.static_range(0, EM_MAX, INIT_TILE):
        r = r0 + gl.arange(0, INIT_TILE, layout=LR)
        mask = r < EM_MAX
        gl.store(
            sti_ptr + r,
            gl.full([INIT_TILE], sentinel, gl.int32, layout=LR),
            mask=mask,
        )
        gl.store(
            sw_ptr + r,
            gl.full([INIT_TILE], 0.0, gl.float32, layout=LR),
            mask=mask,
        )

    e = gl.arange(0, EP, layout=LE)
    valid_e = e < num_experts
    gl.store(fill_ctr_ptr + e, gl.zeros([EP], gl.float32, layout=LE), mask=valid_e)
    gl.store(
        sei_ptr + e,
        gl.full([EP], -1, gl.int32, layout=LE),
        mask=e < NB_MAX,
    )
    gl.barrier()

    g = gl.arange(0, GP, layout=LG)
    gmask = g < G
    global_idx = gl.load(ids_ptr + g, mask=gmask, other=EXPERT_START)
    idx = global_idx - EXPERT_START
    route_mask = gmask & (idx >= 0) & (idx < num_experts)
    safe_idx = gl.where(route_mask, idx, 0)
    weights = gl.load(wts_ptr + g, mask=route_mask, other=0.0)

    counts = gl.histogram(safe_idx, EP, mask=route_mask, layout=LE)
    hit = valid_e & (counts > 0)
    blocks_pe = hit.to(gl.int32)
    block_off = gl.associative_scan(blocks_pe, 0, _add) - blocks_pe
    num_blocks = gl.sum(blocks_pe, 0)
    gl.store(nv_ptr, num_blocks * block_m)
    gl.store(sei_ptr + block_off, e.to(gl.int32), mask=hit)

    rank = gl.amd.cdna4.buffer_atomic_add(
        fill_ctr_ptr,
        safe_idx,
        gl.full([GP], 1.0, gl.float32, layout=LG),
        mask=route_mask,
    ).to(gl.int32)
    route_block = gl.gather(block_off, safe_idx, axis=0)
    dest = route_block * block_m + rank
    token = g // TOPK
    slot = g % TOPK
    packed = ((slot << 24) | token).to(gl.int32)
    gl.store(sti_ptr + dest, packed, mask=route_mask)
    gl.store(sw_ptr + dest, weights, mask=route_mask)


def moe_align_block_size_fused(
    topk_ids: torch.Tensor,  # [M, topk] int
    topk_weights: torch.Tensor,  # [M, topk] float
    num_experts: int,
    block_m: int,
    *,
    expert_start: int = 0,
):
    """Single-kernel sync-free decode block-align (pure Gluon). Same return
    contract as ``moe_align_block_size``.

    ``topk_ids`` may use global expert IDs. ``expert_start`` identifies the
    first expert owned by this rank; routes outside the contiguous local range
    are ignored without a separate localization kernel.
    """
    assert topk_ids.shape == topk_weights.shape
    if expert_start < 0:
        raise ValueError("expert_start must be non-negative")
    device = topk_ids.device
    M, topk = topk_ids.shape
    G = M * topk
    sentinel = M
    assert M <= block_m, f"fused small-M align needs M ({M}) <= block_m ({block_m})"

    # At most one block is emitted per local expert.  Capping the static
    # output bound avoids making downstream grouped GEMMs traverse G padded
    # blocks when EP leaves this rank with fewer than G experts.
    NB_MAX = min(G, num_experts)
    EM_MAX = NB_MAX * block_m
    GP = _next_pow2(G)
    EP = _next_pow2(num_experts)

    ids = topk_ids.reshape(-1).to(torch.int32).contiguous()
    wts = topk_weights.reshape(-1).to(torch.float32).contiguous()
    sti = torch.empty(EM_MAX, dtype=torch.int32, device=device)
    sw = torch.empty(EM_MAX, dtype=torch.float32, device=device)
    sei = torch.empty(NB_MAX, dtype=torch.int32, device=device)
    nv = torch.empty(1, dtype=torch.int32, device=device)

    kernel = _fused_align_kernel
    extra_args = ()
    if G > 128:
        kernel = _fused_atomic_align_kernel
        extra_args = (torch.empty(EP, dtype=torch.float32, device=device),)
    kernel[(1,)](
        ids,
        wts,
        sti,
        sw,
        sei,
        nv,
        *extra_args,
        G,
        num_experts,
        block_m,
        sentinel,
        TOPK=topk,
        GP=GP,
        EP=EP,
        EXPERT_START=expert_start,
        NB_MAX=NB_MAX,
        EM_MAX=EM_MAX,
        INIT_TILE=_next_pow2(min(1024, EM_MAX)),
        num_warps=4,
    )
    return sti, sei, sw, nv
