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

"""Registrations for AMD Gluon GEMM kernels."""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

if current_platform().is_amd:
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.rmsnorm_linear_add import (
        gluon_rmsnorm_linear_add_gfx950 as _rmsnorm_linear_add_impl,
    )

    _BF16_SIGNATURE = frozenset(
        {
            format_signature(
                hidden_states=dense_tensor_format(torch.bfloat16),
                norm_weight=dense_tensor_format(torch.bfloat16),
                linear_weight=dense_tensor_format(torch.bfloat16),
                out=dense_tensor_format(torch.bfloat16),
            )
        }
    )
    _GFX950 = CapabilityRequirement(
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
        vendors=frozenset({"amd"}),
    )

    @register_kernel(
        "gemm",
        "rmsnorm_linear_add",
        name="gluon_rmsnorm_linear_add_gfx950",
        solution="gluon",
        capability=_GFX950,
        signatures=_BF16_SIGNATURE,
        priority=Priority.SPECIALIZED,
        traits={
            "tokens": frozenset({1}),
            "input_size": frozenset({3584}),
            "output_size": frozenset({7168}),
            "num_addends": frozenset({2}),
            "inputs_contiguous": frozenset({True}),
        },
    )
    def gluon_rmsnorm_linear_add_gfx950(**kwargs):
        addends = kwargs["addends"]
        return _rmsnorm_linear_add_impl(
            kwargs["hidden_states"],
            kwargs["norm_weight"],
            kwargs["linear_weight"],
            addends[0],
            addends[1],
            eps=kwargs["eps"],
            out=kwargs["out"],
        )
