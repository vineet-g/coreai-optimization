# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""KV-cache quantization export tests.

Converts a small attention block with quantized K/V caches to Core AI and checks, on the
converted IR, that each cache update is fed by ``coreai.quantize`` and consumed by
``coreai.dequantize`` so the cache itself stays in the quantized dtype. ``aten.slice_scatter``
stands in for the cache-update op since coreai-torch lowers it (to ``coreai.slice_update``).
"""

import coreai_torch
import pytest
import torch
from torch import nn

from coreai_opt import ExportBackend
from coreai_opt.quantization import (
    ModuleQuantizerConfig,
    Quantizer,
    QuantizerConfig,
)
from coreai_opt.quantization.config.quantization_config import (
    KVCacheQuantConfig,
    OpQuantizerConfig,
)
from coreai_opt.quantization.spec import (
    PerTensorGranularity,
    QuantizationSpec,
    default_weight_quantization_spec,
)


class _KVCacheAttentionBlock(nn.Module):
    """Multi-head attention block that writes the new K/V into the caches."""

    def __init__(self, dim: int = 32, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = t.shape
        return t.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self, x: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self._split_heads(self.q_proj(x))
        k = torch.ops.aten.slice_scatter(k_cache, self._split_heads(self.k_proj(x)), 2, 0, seq_len)
        v = torch.ops.aten.slice_scatter(v_cache, self._split_heads(self.v_proj(x)), 2, 0, seq_len)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return self.o_proj(attn.transpose(1, 2).reshape(batch, seq_len, -1))


@pytest.mark.parametrize("execution_mode", ["graph", "eager"])
def test_kv_cache_quant_dq_placement_in_coreai_ir(execution_mode):
    cache_dtype = torch.int8
    config = QuantizerConfig(
        execution_mode=execution_mode,
        global_config=ModuleQuantizerConfig(
            op_state_spec={"weight": default_weight_quantization_spec()},
            op_input_spec=None,
            op_output_spec=None,
        ),
        kv_cache_quant_configs={
            "slice_scatter": KVCacheQuantConfig(
                op_quantizer_config=OpQuantizerConfig(
                    # Input 1 of slice_scatter(cache, update, ...) is the new K/V.
                    op_input_spec={
                        1: QuantizationSpec(
                            dtype=cache_dtype,
                            qscheme="symmetric",
                            granularity=PerTensorGranularity(),
                            fake_quantize_cls="default",
                            qparam_calculator_cls="moving_average",
                            range_calculator_cls="minmax",
                        )
                    },
                    op_output_spec=None,
                    op_state_spec=None,
                ),
            ),
        },
    )
    x = torch.randn(1, 4, 32)
    k_cache = torch.zeros(1, 4, 16, 8)
    v_cache = torch.zeros(1, 4, 16, 8)
    quantizer = Quantizer(_KVCacheAttentionBlock().eval(), config)
    prepared = quantizer.prepare((x, k_cache, v_cache))
    with quantizer.calibration_mode():
        prepared(x, k_cache, v_cache)
    finalized = quantizer.finalize(backend=ExportBackend.CoreAI)

    # The finalized model takes the caches in the quantized dtype.
    exported = torch.export.export(
        finalized, (x, k_cache.to(cache_dtype), v_cache.to(cache_dtype)), strict=False
    ).run_decompositions()
    converter = coreai_torch.TorchConverter()
    converter.add_exported_program(exported)
    # Keep the program referenced: the graph op is a view into the program's module.
    program = converter.to_coreai()
    graph = program._get_graph("main")  # noqa: SLF001

    ops = [op.operation for op in graph.entry_block.operations]
    cache_updates = [op for op in ops if op.name == "coreai.slice_update"]
    assert len(cache_updates) == 2, f"expected one slice_update per cache, got {len(cache_updates)}"
    for cache_update in cache_updates:
        # The cache is a quantized function argument, updated in place without a dequantize.
        cache = cache_update.operands[0]
        assert isinstance(cache.owner, type(graph.entry_block)), (
            f"cache operand should be a function argument, got {cache.owner}"
        )
        assert "xsi8>" in str(cache.type), f"cache should be si8, got {cache.type}"

        # quantize at the input of slice_update.
        producers = [
            operand.owner.name
            for operand in cache_update.operands
            if not isinstance(operand.owner, type(graph.entry_block))
        ]
        assert producers.count("coreai.quantize") == 1, (
            f"expected the update to come from coreai.quantize; producers: {producers}"
        )

        # dequantize at the output of slice_update.
        users = [use.owner.name for use in cache_update.results[0].uses]
        assert users == ["coreai.dequantize"], f"expected a single dequantize user, got {users}"

    # Weights are compressed separately, so the only activation q/dq are the cache ones.
    op_names = [op.name for op in ops]
    assert op_names.count("coreai.quantize") == len(cache_updates)
    assert op_names.count("coreai.dequantize") == len(cache_updates)
