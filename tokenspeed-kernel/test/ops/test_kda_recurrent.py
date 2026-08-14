"""KDA chunk-prefill and dispatch coverage."""

from __future__ import annotations

import pytest
import torch
from kimi3_reference import kda_gate
from kimi3_reference import kda_recurrent as reference_kda_recurrent
from tokenspeed_kernel.ops.attention import (
    _attention_format_signature,
    kda_paged_decode,
    kda_paged_prefill,
    try_kda_fused_paged_decode,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel


def test_k3_safe_gate_reference_matches_sigmoid_contract() -> None:
    """Distinguish K3's safe sigmoid gate from a clamped softplus gate."""
    raw_g = torch.tensor([[[-2.0, 0.0, 2.0]]])
    a_log = torch.tensor([0.25])
    dt_bias = torch.tensor([[0.5, -0.5, 1.0]])
    lower_bound = -5.0

    gate_input = raw_g + dt_bias
    expected = lower_bound * torch.sigmoid(torch.exp(a_log)[None, :, None] * gate_input)
    actual = kda_gate(raw_g, a_log, dt_bias, lower_bound=lower_bound)
    torch.testing.assert_close(actual, expected)

    legacy = torch.clamp_min(
        -torch.exp(a_log)[None, :, None] * torch.nn.functional.softplus(gate_input),
        lower_bound,
    )
    assert not torch.allclose(actual, legacy)


def test_kda_paged_prefill_uses_canonical_k_major_state() -> None:
    """Native prefill preserves the public [N,H,K,V] state layout."""
    platform = current_platform()
    if not (platform.is_cdna4 or platform.is_cdna5):
        pytest.skip("gfx950/gfx1250 KDA dispatch test")

    device = "cuda"
    torch.manual_seed(3)
    tokens, heads, key_dim, value_dim = 65, 2, 128, 128
    q = torch.randn(tokens, heads, key_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(
        tokens,
        heads,
        value_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    raw_g = torch.randn_like(q)
    beta = torch.randn(tokens, heads, device=device, dtype=torch.bfloat16)
    state = torch.randn(
        1,
        heads,
        key_dim,
        value_dim,
        device=device,
        dtype=torch.float32,
    )
    a_log = torch.randn(heads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(heads, key_dim, device=device, dtype=torch.float32)
    cu_seqlens = torch.tensor([0, tokens], device=device, dtype=torch.int32)

    expected_out, expected_state = reference_kda_recurrent(
        q,
        k,
        v,
        raw_g,
        beta,
        state[0].transpose(-1, -2).contiguous(),
        a_log,
        dt_bias,
    )
    result = kda_paged_prefill(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        raw_g.unsqueeze(0),
        beta.unsqueeze(0),
        a_log,
        dt_bias,
        initial_state=state,
        cu_seqlens=cu_seqlens,
    )

    torch.testing.assert_close(
        result.out[0].float(), expected_out.float(), atol=6e-2, rtol=6e-2
    )
    torch.testing.assert_close(
        result.final_state,
        expected_state.transpose(-1, -2).unsqueeze(0),
        atol=6e-2,
        rtol=6e-2,
    )


@pytest.mark.parametrize("lower_bound", [-5.0, None])
@pytest.mark.parametrize("strided_inputs", [False, True])
@pytest.mark.parametrize(
    ("heads", "key_dim", "value_dim"),
    [(2, 8, 8), (2, 8, 5), (12, 128, 128)],
)
def test_kda_paged_decode_defaults_to_specialized_kernel_on_amd(
    lower_bound: float | None,
    strided_inputs: bool,
    heads: int,
    key_dim: int,
    value_dim: int,
) -> None:
    """AMD single-token dispatch must select Gluon and preserve native math."""
    platform = current_platform()
    if not (platform.is_cdna4 or platform.is_cdna5):
        pytest.skip("gfx950/gfx1250 KDA dispatch test")

    device = "cuda"
    torch.manual_seed(13)
    tokens = 3
    if strided_inputs:
        qkv = torch.randn(
            1,
            tokens,
            heads * (2 * key_dim + value_dim) + 7,
            device=device,
            dtype=torch.bfloat16,
        )
        q_end = heads * key_dim
        k_end = 2 * q_end
        v_end = k_end + heads * value_dim
        q = qkv[..., :q_end].view(1, tokens, heads, key_dim)
        k = qkv[..., q_end:k_end].view(1, tokens, heads, key_dim)
        v = qkv[..., k_end:v_end].view(1, tokens, heads, value_dim)
        raw_g = torch.randn(
            1,
            tokens,
            heads * key_dim + 5,
            device=device,
            dtype=torch.bfloat16,
        )[..., :q_end].view(1, tokens, heads, key_dim)
        beta = torch.randn(
            1,
            tokens,
            heads + 3,
            device=device,
            dtype=torch.bfloat16,
        )[..., :heads]
    else:
        q = torch.randn(
            1, tokens, heads, key_dim, device=device, dtype=torch.bfloat16
        )
        k = torch.randn_like(q)
        v = torch.randn(
            1, tokens, heads, value_dim, device=device, dtype=torch.bfloat16
        )
        raw_g = torch.randn_like(q)
        beta = torch.randn(1, tokens, heads, device=device, dtype=torch.bfloat16)
    state_pool = torch.randn(
        6,
        heads,
        key_dim,
        value_dim,
        device=device,
        dtype=torch.float32,
    )
    initial_pool = state_pool.clone()
    expected_pool = state_pool.clone()
    a_log = torch.randn(heads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(heads, key_dim, device=device, dtype=torch.float32)
    cu_seqlens = torch.arange(tokens + 1, device=device, dtype=torch.int32)
    read_indices = torch.tensor([0, 1, 1], device=device, dtype=torch.int32)
    write_indices = torch.tensor([2, 3, 4], device=device, dtype=torch.int32)

    selected = select_kernel(
        "attention",
        "kda_paged_decode",
        _attention_format_signature(q=q, k=k, v=v),
        traits={"indexed_state": True, "single_token": True},
    )
    expected_kernel = (
        "gluon_kda_paged_decode_gfx1250"
        if platform.is_cdna5
        else "gluon_kda_paged_decode_gfx950"
    )
    assert selected.name == expected_kernel

    expected_out = []
    for row in range(tokens):
        out, final_state = reference_kda_recurrent(
            q[0, row : row + 1],
            k[0, row : row + 1],
            v[0, row : row + 1],
            raw_g[0, row : row + 1],
            beta[0, row : row + 1],
            initial_pool[read_indices[row].long()].transpose(-1, -2),
            a_log,
            dt_bias,
            lower_bound=lower_bound,
        )
        expected_out.append(out[0])
        expected_pool[write_indices[row].long()] = final_state.transpose(-1, -2)
    expected_out = torch.stack(expected_out).unsqueeze(0)
    actual_out = kda_paged_decode(
        q,
        k,
        v,
        raw_g,
        beta,
        a_log,
        dt_bias,
        state_pool=state_pool,
        read_indices=read_indices,
        write_indices=write_indices,
        cu_seqlens=cu_seqlens,
        lower_bound=lower_bound,
    )

    torch.testing.assert_close(
        actual_out.float(), expected_out.float(), atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(state_pool, expected_pool, atol=2e-5, rtol=2e-5)


def test_kda_paged_decode_graph_padding_and_page_stride() -> None:
    """Gluon decode supports padded graph batches and strided state pages."""
    if not (current_platform().is_cdna4 or current_platform().is_cdna5):
        pytest.skip("gfx950/gfx1250 KDA dispatch test")

    torch.manual_seed(23)
    batch, active, heads, key_dim, value_dim = 4, 2, 2, 8, 5
    state_elements = heads * key_dim * value_dim
    raw_pool = torch.randn(7, state_elements + 11, device="cuda", dtype=torch.float32)
    state_pool = raw_pool[:, :state_elements].view(7, heads, key_dim, value_dim)
    q = torch.randn(1, batch, heads, key_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, batch, heads, value_dim, device="cuda", dtype=torch.bfloat16)
    raw_g = torch.randn_like(q)
    beta = torch.randn(1, batch, heads, device="cuda", dtype=torch.bfloat16)
    a_log = torch.randn(heads, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(heads, key_dim, device="cuda", dtype=torch.float32)
    read_indices = torch.tensor([1, 2, -1, -1], device="cuda", dtype=torch.int32)
    write_indices = torch.tensor([3, 4, -1, -1], device="cuda", dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 1, 2, 2, 2], device="cuda", dtype=torch.int32)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = kda_paged_decode(
            q,
            k,
            v,
            raw_g,
            beta,
            a_log,
            dt_bias,
            state_pool=state_pool,
            read_indices=read_indices,
            write_indices=write_indices,
            cu_seqlens=cu_seqlens,
        )
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(
        captured[:, active:],
        torch.zeros_like(captured[:, active:]),
    )


def test_kda_paged_decode_does_not_select_nvidia_kernel_on_amd() -> None:
    """The NVIDIA portable adapter remains outside the AMD dispatch surface."""
    if not current_platform().is_amd:
        pytest.skip("AMD KDA dispatch test")

    q = torch.randn(1, 3, 2, 8, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    with pytest.raises(NoKernelFoundError):
        select_kernel(
            "attention",
            "kda_paged_decode",
            _attention_format_signature(q=q, k=k, v=v),
            traits={"indexed_state": True, "single_token": False},
        )


def test_k3_fused_paged_decode_matches_composed_path_on_gfx950() -> None:
    """The small-batch megafuse preserves K3 state and graph semantics."""
    if not current_platform().is_cdna4:
        pytest.skip("gfx950 KDA fusion test")

    from tokenspeed.runtime.layers.attention.linear.causal_conv1d import (
        causal_conv1d_update,
    )

    torch.manual_seed(29)
    batch, heads, head_dim = 4, 12, 128
    projection = heads * head_dim
    packed = (
        torch.randn(
            batch,
            3 * projection + head_dim + heads,
            device="cuda",
        )
        * 0.1
    ).to(torch.bfloat16)
    mixed_qkv = packed[:, : 3 * projection]
    f_a = packed[:, 3 * projection : 3 * projection + head_dim]
    beta = packed[:, 3 * projection + head_dim :]
    conv_weight = (
        torch.randn(3 * projection, 4, device="cuda") * 0.1
    ).to(torch.bfloat16)
    conv_seed = (
        torch.randn(2 * batch + 1, 3 * projection, 3, device="cuda") * 0.1
    ).to(torch.bfloat16)
    f_b_weight = (
        torch.randn(projection, head_dim, device="cuda") * 0.1
    ).to(torch.bfloat16)
    state_seed = (
        torch.randn(
            2 * batch + 1,
            heads,
            head_dim,
            head_dim,
            device="cuda",
        )
        * 0.01
    )
    a_log = torch.randn(heads, device="cuda") * 0.1
    dt_bias = torch.randn(projection, device="cuda") * 0.1
    read_indices = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
    write_indices = read_indices + batch
    cu_seqlens = torch.arange(batch + 1, device="cuda", dtype=torch.int32)

    expected_conv = conv_seed.clone()
    expected_state = state_seed.clone()
    mixed = causal_conv1d_update(
        mixed_qkv.clone(),
        expected_conv,
        conv_weight,
        None,
        "silu",
        conv_state_indices=read_indices,
        output_state_indices=write_indices.view(-1, 1),
    )
    q, k, v = mixed.split(projection, dim=-1)
    gate = torch.nn.functional.linear(f_a, f_b_weight)
    expected = kda_paged_decode(
        q.view(1, batch, heads, head_dim),
        k.view(1, batch, heads, head_dim),
        v.view(1, batch, heads, head_dim),
        gate.view(1, batch, heads, head_dim),
        beta.view(1, batch, heads),
        a_log,
        dt_bias,
        state_pool=expected_state,
        read_indices=read_indices,
        write_indices=write_indices,
        cu_seqlens=cu_seqlens,
        lower_bound=-5.0,
    )

    actual_conv = conv_seed.clone()
    actual_state = state_seed.clone()

    def fused() -> torch.Tensor:
        result = try_kda_fused_paged_decode(
            mixed_qkv,
            conv_weight,
            actual_conv,
            f_a,
            f_b_weight,
            beta,
            a_log,
            dt_bias,
            state_pool=actual_state,
            read_indices=read_indices,
            write_indices=write_indices,
            num_heads=heads,
            head_dim=head_dim,
            cu_seqlens=cu_seqlens,
            lower_bound=-5.0,
        )
        assert result is not None
        return result

    actual = fused()
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(actual_conv, expected_conv, atol=0, rtol=0)
    torch.testing.assert_close(actual_state, expected_state, atol=2e-2, rtol=2e-3)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused()
    graph.replay()
    torch.cuda.synchronize()

    assert (
        try_kda_fused_paged_decode(
            mixed_qkv.repeat(4, 1),
            conv_weight,
            actual_conv,
            f_a,
            f_b_weight,
            beta,
            a_log,
            dt_bias,
            state_pool=actual_state,
            read_indices=read_indices,
            write_indices=write_indices,
            num_heads=heads,
            head_dim=head_dim,
            cu_seqlens=cu_seqlens,
            lower_bound=-5.0,
        )
        is None
    )
