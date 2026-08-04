from __future__ import annotations

import pytest
import tokenspeed_kernel
import torch

if not torch.cuda.is_available():
    pytest.skip("requires a GPU", allow_module_level=True)

if torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0] != "gfx950":
    pytest.skip("requires gfx950", allow_module_level=True)


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("scale", [1.0, 2.5])
def test_kimi3_sigmoid_bias_topk_matches_and_captures(
    normalize: bool,
    scale: float,
) -> None:
    torch.manual_seed(7)
    logits = (torch.randn(1, 896, device="cuda") * 0.2).float()
    bias = (torch.randn(896, device="cuda") * 0.01).float()
    scores = logits.sigmoid()
    expected_ids = torch.topk(
        scores + bias.unsqueeze(0),
        16,
        dim=-1,
        sorted=False,
    ).indices
    expected_weights = scores.gather(1, expected_ids)
    if normalize:
        expected_weights /= expected_weights.sum(dim=-1, keepdim=True)
    expected_weights *= scale

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        weights, ids = tokenspeed_kernel.moe_sigmoid_bias_topk(
            logits,
            bias,
            16,
            routed_scaling_factor=scale,
            normalize_topk_weights=normalize,
        )
    graph.replay()
    torch.cuda.synchronize()

    assert set(ids[0].tolist()) == set(expected_ids[0].tolist())
    expected_by_id = {
        expert_id: weight
        for expert_id, weight in zip(
            expected_ids[0].tolist(),
            expected_weights[0].tolist(),
        )
    }
    actual_by_id = {
        expert_id: weight
        for expert_id, weight in zip(ids[0].tolist(), weights[0].tolist())
    }
    for expert_id, expected_weight in expected_by_id.items():
        torch.testing.assert_close(
            torch.tensor(actual_by_id[expert_id]),
            torch.tensor(expected_weight),
            rtol=2e-6,
            atol=2e-7,
        )


def test_kimi3_sigmoid_bias_topk_fuses_dispatch_map() -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260802)
    logits = torch.randn(4, 896, device="cuda", generator=generator)
    bias = torch.randn(896, device="cuda", generator=generator) * 0.01
    dispatch = torch.arange(895, -1, -1, device="cuda", dtype=torch.int32)

    _, logical_ids = tokenspeed_kernel.moe_sigmoid_bias_topk(logits, bias, 16)
    weights, physical_ids = tokenspeed_kernel.moe_sigmoid_bias_topk(
        logits,
        bias,
        16,
        logical_to_physical_map=dispatch,
    )

    torch.testing.assert_close(physical_ids, dispatch[logical_ids.long()])
    assert weights.shape == (4, 16)
