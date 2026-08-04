# Kimi K3 Gluon MXFP4 MoE

This experiment ports the applicable Kimi K2.5 ideas from Aiter
[#3470](https://github.com/ROCm/aiter/pull/3470),
[#3832](https://github.com/ROCm/aiter/pull/3832), and
[#3466](https://github.com/ROCm/aiter/pull/3466) to the Kimi K3 gfx950 path.
All new compute kernels are implemented in Gluon.

## Design

K3 differs from K2.5 in its expert shape and activation. With EP8, each rank
owns 112 of 896 routed experts, each token selects 16 experts, and W13 uses
SiTU rather than SwiGLU. The decode pipeline is:

```text
BF16 hidden
  -> Gluon BF16-to-MXFP4 quantization
  -> MXFP4 x MXFP4 W13 scaled MFMA
  -> SiTU in FP32 registers
  -> MXFP4 requantization in the W13 epilogue
  -> MXFP4 x MXFP4 W2 scaled MFMA
  -> route-weighted reduction
  -> BF16 output
```

The W13 result is never stored as a BF16 tensor. Scaled MFMA accumulates in
FP32, then the same kernel applies SiTU, rounds to the model's BF16 activation
precision in registers, and writes packed E2M1 values with E8M0 block scales.
W2 consumes those values directly. This avoids a BF16 global-memory round trip
and a standalone requantization launch.

The implementation also:

- supports EP-aware global-to-local expert IDs;
- handles linear and gdot128-preshuffled weights;
- handles concatenated and interleaved W13 gate/up layouts;
- writes the CDNA4 E8M0 scale layout expected by scaled MFMA;
- supports zero-token inputs and CUDA graph capture.

The experimental runtime path is enabled with:

```bash
TOKENSPEED_KIMI3_GLUON_A4W4=1
```

It retains the existing A16W4 warp-GEMV path for M <= 4, uses A4W4 for
M = 5..15, and uses grouped A16W4 at M >= 16. The A4W4 path
keeps an additional MFMA-native copy of W13 and W2, so its current memory cost
is too high for unconditional enablement.

## Techniques adopted

| Source | K3 adaptation |
| --- | --- |
| Aiter #3470 | Dynamic MXFP4 activation quantization, scaled FP4 MFMA, fused expert activation, and token-count-specific dispatch. |
| Aiter #3832 | Gluon/FlyDSL-style FP4 GEMMs, MFMA-native weight layout, fused W13 activation requantization, and decode-oriented launch tuning. |
| Aiter #3466 | Hardware `exp2` plus fast reciprocal for sigmoid routing, while retaining unbiased sigmoid scores locally for top-k selection. |

K3 routing uses a per-token Gluon CTA for its `[M, 896]`, top-k 16 shape and
fuses logical-to-physical expert mapping. NVIDIA continues to use its existing
packed-key Triton route.

## Results

The focused MoE benchmark uses the K3 EP8 local shape with two locally owned
routes per token. The preshuffled A4W4 kernel is faster than A16W4 throughout
the tested M = 5..16 range:

| M | A16W4 (us) | A4W4 (us) | Speedup |
| ---: | ---: | ---: | ---: |
| 5 | 134.65 | 69.18 | 1.95x |
| 8 | 192.02 | 73.78 | 2.60x |
| 12 | 317.83 | 111.62 | 2.85x |
| 16 | 397.08 | 135.32 | 2.93x |

The end-to-end benchmark uses K3 TP8/EP8, a 4096-token prompt, a 1024-token
completion, graph mode, and the median of three repetitions:

| M | Base tok/s | New tok/s | Change | Base TPOT | New TPOT |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 44.26 | 44.98 | +1.63% | 22.05 ms | 21.66 ms |
| 2 | 58.53 | 63.86 | +9.11% | 33.25 ms | 30.39 ms |
| 4 | 98.50 | 109.14 | +10.80% | 38.94 ms | 34.98 ms |

These M <= 4 rows retain the existing A16W4 expert kernels and measure the
new Gluon routing path. A4W4 is active in the following rows:

| M | A16W4 tok/s | A4W4 tok/s | Change | A16W4 TPOT | A4W4 TPOT |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 89.64 | 96.14 | +7.25% | 83.36 ms | 77.35 ms |
| 16 | 145.58 | 146.71 | +0.78% | 100.51 ms | 99.70 ms |

The M=16 A4W4 gain was too small to justify automatic selection, so M=16
remains on the stable grouped A16W4 path. A direct FP32-to-MXFP4 packing
experiment exposed a reproducible larger-grid graph-capture fault and is not
included. The current A4W4 dispatch range remains experimental.

## Validation

- 66 gfx950 MXFP4 kernel tests passed.
- 13 K3 prefill and routing tests passed.
- 79 kernel-selection tests passed; 33 were skipped.
- The M=8 coherence canary completed 8/8 requests for 256 output tokens with
  coherent responses.
- Exact tests cover quantization, linear versus preshuffled weights, fused
  SiTU requantization, EP dispatch, and runtime path selection.

Activation requantization can change output hashes relative to A16W4. Semantic
coherence and numerical kernel references, rather than bitwise model output
identity, are the relevant correctness checks.

The final M=16 dispatch returns to grouped A16W4. Its graph capture and
32-token warmup passed, but a subsequent long run faulted while compiling the
separate latent-projection GEMM, so no new M=16 TPS claim is made for the final
dispatch.
