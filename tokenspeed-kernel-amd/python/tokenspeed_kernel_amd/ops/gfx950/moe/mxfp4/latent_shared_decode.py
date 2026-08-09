# Copyright (c) 2026 LightSeek Foundation

"""Joint routed-expert and shared-expert decode for latent MoE."""

from __future__ import annotations

import os

import torch

from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.moe import (
    gluon_mxfp4_situ_moe_decode,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.situ_decode import (
    gluon_a16w4_situ_warp_decode_ep_gfx950,
)

_KIMI3_A4W4_DECODE = os.environ.get("TOKENSPEED_KIMI3_GLUON_A4W4") == "1"


def gluon_latent_expert_shared_decode_gfx950(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_input: torch.Tensor,
    shared_weight: torch.Tensor,
    *,
    situ_beta: float,
    situ_linear_beta: float | None,
    expert_start: int,
    w13_interleaved: bool,
    routed_out: torch.Tensor,
    shared_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run routed MXFP4 experts and the shared BF16 down projection jointly."""

    if _KIMI3_A4W4_DECODE and situ_linear_beta is not None:
        result = gluon_mxfp4_situ_moe_decode(
            hidden_states,
            w13_weight,
            w13_scale,
            w2_weight,
            w2_scale,
            topk_ids,
            topk_weights,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
            expert_start=expert_start,
            linear_weights=True,
            w13_interleaved=w13_interleaved,
            shared_input=shared_input,
            shared_weight=shared_weight,
            routed_out=routed_out,
            shared_out=shared_out,
        )
        if not isinstance(result, tuple):
            raise RuntimeError("joint A4W4 latent decode did not return both outputs")
        return result

    result = gluon_a16w4_situ_warp_decode_ep_gfx950(
        hidden_states,
        w13_weight,
        w13_scale,
        w2_weight,
        w2_scale,
        topk_weights,
        topk_ids,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        expert_start=expert_start,
        linear_weights=True,
        w13_interleaved=w13_interleaved,
        shared_input=shared_input,
        shared_weight=shared_weight,
        routed_out=routed_out,
        shared_out=shared_out,
    )
    if not isinstance(result, tuple):
        raise TypeError("joint latent decode did not return both outputs")
    return result


__all__ = ["gluon_latent_expert_shared_decode_gfx950"]
