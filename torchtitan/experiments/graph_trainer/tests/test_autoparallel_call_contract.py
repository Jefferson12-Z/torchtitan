# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU regression tests for the AutoParallel GraphTrainer AOT call boundary."""

import copy
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch._functorch.aot_autograd import (
    aot_export_joint_with_descriptors,
    boxed_nop_preserve_node_meta,
)

from torchtitan.experiments.graph_trainer import autoparallel_api
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.models.common.attention import (
    VarlenAttentionMetadata,
    VarlenInnerAttention,
)


class _MetadataModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 3, bias=False)
        self.projection.register_buffer("offset", torch.tensor([1.0, 2.0, 3.0]))

    def forward(self, tokens, *, positions, padding_mask, attention_metadata):
        metadata = attention_metadata[VarlenInnerAttention]
        output = self.projection(tokens) + self.projection.offset
        output = output + positions[:, None].to(tokens.dtype)
        output = output + (metadata.cu_seq_q[1] + metadata.cu_seq_k[-1]).to(
            tokens.dtype
        )
        output = output + metadata.max_q + metadata.max_k + sum(metadata.window_size)
        return output.masked_fill(padding_mask[:, None], 0)


def _metadata_inputs():
    offsets = torch.tensor([0, 2, 4], dtype=torch.int32)
    metadata = VarlenAttentionMetadata(
        cu_seq_q=offsets, cu_seq_k=offsets, max_q=4, max_k=4
    )
    return {
        "positions": torch.tensor([0, 1, 0, 1], dtype=torch.int32),
        "padding_mask": torch.tensor([False, False, False, True]),
        "attention_metadata": {VarlenInnerAttention: metadata},
    }


def _build_aot_parallel_module(model, args, kwargs):
    autop = autoparallel_api.AutoParallelGraph.__new__(
        autoparallel_api.AutoParallelGraph
    )
    autop.model = model
    autop.compiler_fn = boxed_nop_preserve_node_meta
    with ExitStack() as stack:
        autop.joint_with_descriptors = aot_export_joint_with_descriptors(
            stack, model, args, kwargs
        )
        autop.gm = autop.joint_with_descriptors.graph_module
        with patch.object(
            autop,
            "_apply_placement_common",
            return_value=(dict(model.named_parameters()), dict(model.named_buffers())),
        ):
            parallel_model = autop.apply_placement_for_fx_module(
                compile_config=GraphTrainerCompileConfig()
            )
    return parallel_model


@pytest.mark.parametrize("reverse_kwargs", [False, True])
def test_aot_metadata_kwargs_preserve_outputs_and_gradients(reverse_kwargs):
    model = _MetadataModel()
    reference = copy.deepcopy(model)
    tokens = torch.arange(12, dtype=torch.float32).reshape(4, 3).requires_grad_()
    reference_tokens = tokens.detach().clone().requires_grad_()
    kwargs = _metadata_inputs()
    parallel_model = _build_aot_parallel_module(model, (tokens,), kwargs)
    kwargs = _metadata_inputs()
    kwargs["attention_metadata"][VarlenInnerAttention].cu_seq_q.copy_(
        torch.tensor([0, 1, 4], dtype=torch.int32)
    )
    kwargs["positions"] = torch.tensor([0, 0, 1, 2], dtype=torch.int32)
    kwargs["padding_mask"] = torch.zeros(4, dtype=torch.bool)
    if reverse_kwargs:
        kwargs = dict(reversed(list(kwargs.items())))

    output = parallel_model(tokens, **kwargs)
    expected = reference(reference_tokens, **kwargs)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    gradients = torch.autograd.grad(
        output.square().sum(), (tokens, parallel_model.projection.weight)
    )
    expected_gradients = torch.autograd.grad(
        expected.square().sum(), (reference_tokens, reference.projection.weight)
    )
    for actual, expected_gradient in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected_gradient, rtol=0, atol=0)


@pytest.mark.parametrize("with_bias", [False, True])
def test_aot_positional_only_call_preserves_outputs_and_gradients(with_bias):
    model = nn.Linear(3, 2, bias=with_bias)
    reference = copy.deepcopy(model)
    inputs = torch.arange(12, dtype=torch.float32).reshape(4, 3).requires_grad_()
    reference_inputs = inputs.detach().clone().requires_grad_()
    parallel_model = _build_aot_parallel_module(model, (inputs,), {})

    output = parallel_model(inputs)
    expected = reference(reference_inputs)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    gradients = torch.autograd.grad(
        output.sum(), (inputs, *parallel_model.parameters())
    )
    expected_gradients = torch.autograd.grad(
        expected.sum(), (reference_inputs, *reference.parameters())
    )
    for actual, expected_gradient in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected_gradient, rtol=0, atol=0)


def test_aot_call_forwards_named_inputs_and_current_module_state():
    model = _MetadataModel()
    autop = autoparallel_api.AutoParallelGraph.__new__(
        autoparallel_api.AutoParallelGraph
    )
    autop.model = model
    autop.compiler_fn = boxed_nop_preserve_node_meta
    autop.joint_with_descriptors = SimpleNamespace(
        params_spec=["projection.weight"], buffers_spec=["projection.offset"]
    )
    with (
        patch.object(
            autop,
            "_apply_placement_common",
            return_value=(dict(model.named_parameters()), dict(model.named_buffers())),
        ),
        patch.object(
            autoparallel_api, "aot_compile_joint_with_descriptors"
        ) as compile_joint,
    ):
        parallel_model = autop.apply_placement_for_fx_module(
            compile_config=GraphTrainerCompileConfig()
        )

    weight = nn.Parameter(torch.eye(3))
    offset = torch.tensor([3.0, 2.0, 1.0])
    parallel_model.projection.weight = weight
    parallel_model.projection.offset = offset
    tokens = torch.ones(4, 3, requires_grad=True)
    kwargs = dict(reversed(list(_metadata_inputs().items())))
    compiled = compile_joint.return_value
    compiled.return_value = tokens
    assert parallel_model(tokens, **kwargs) is tokens

    args, forwarded_kwargs = compiled.call_args
    assert len(args) == 3
    assert args[0] is weight
    assert args[1] is offset
    assert args[2] is tokens
    assert list(forwarded_kwargs) == list(kwargs)
    for name, value in kwargs.items():
        assert forwarded_kwargs[name] is value
    metadata = forwarded_kwargs["attention_metadata"][VarlenInnerAttention]
    assert metadata.cu_seq_q is metadata.cu_seq_k
    compile_joint.assert_called_once_with(
        autop.joint_with_descriptors,
        fw_compiler=boxed_nop_preserve_node_meta,
        bw_compiler=boxed_nop_preserve_node_meta,
    )
