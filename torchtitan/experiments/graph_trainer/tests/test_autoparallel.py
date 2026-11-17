# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from autoparallel import ForwardInputs, UNCONSTRAINED
from torch._functorch._aot_autograd.descriptors import PlainAOTInput
from torch.distributed.tensor.placement_types import Replicate

from torchtitan.config import PrecisionConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig


class _FakeMesh:
    def __init__(self, mesh_axis_names, size=2):
        self.mesh_dim_names = tuple(mesh_axis_names)
        self.ndim = len(self.mesh_dim_names)
        self._size = size

    def size(self):
        return self._size


class _FakeParallelismContext:
    dp_replicate_enabled = False
    cp_enabled = False
    pp_enabled = False
    tp_enabled = True
    dp_replicate = 1
    dp_shard = 2

    def __init__(self, *, sparse: bool = False):
        self.sparse = sparse
        self.tp_enabled = not sparse

    def get_optional_mesh(self, name):
        enabled = {"dp_shard", "tp"} if not self.sparse else {"edp_shard", "ep"}
        if name == "dp_shard" and self.dp_shard == 1:
            return None
        return _FakeMesh((name,)) if name in enabled else None

    def get_mesh(self, names):
        if isinstance(names, str):
            return _FakeMesh((names,))
        return _FakeMesh(tuple(names))


class _FakeAutoParallelGraph:
    instances = []

    def __init__(self, model, input_fn, mesh, **kwargs):
        self.model = model
        self.input_fn = input_fn
        self.kwargs = kwargs
        self.used_fx_path = False
        _FakeAutoParallelGraph.instances.append(self)

    def __enter__(self):
        inputs = self.input_fn()
        self.inputs = (
            inputs if isinstance(inputs, ForwardInputs) else ForwardInputs(args=inputs)
        )
        flat_inputs, _ = torch.utils._pytree.tree_flatten(
            (self.inputs.args, self.inputs.kwargs)
        )
        graph = torch.fx.Graph()
        seen = set()
        for index, value in enumerate(flat_inputs):
            if not isinstance(value, torch.Tensor) or id(value) in seen:
                continue
            seen.add(id(value))
            node = graph.placeholder(f"input_{index}")
            node.meta.update(desc=PlainAOTInput(index), val=value)
        graph.output(()).meta["desc"] = []
        self.gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def add_parameter_memory_constraint(self, *, low, high):
        pass

    def add_input_constraints(self, constraints):
        self.input_constraints = constraints

    def add_output_constraints(self, constraints):
        pass

    def optimize_placement(self, verbose=False):
        return object()

    def apply_placement_for_fx_module(self, *args, **kwargs):
        self.used_fx_path = True
        self.apply_kwargs = kwargs
        return torch.nn.Linear(1, 1)


def _training_config():
    return TrainingConfig(
        num_tokens_per_microbatch_per_dp_rank=2 * 8,
    )


def test_autoparallel_graph_pass_selection_uses_regular_memory_policy():
    from torchtitan.experiments.graph_trainer import passes

    traced_result = SimpleNamespace(
        gm=torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph()),
        state_fqns=[],
    )
    config = SimpleNamespace(
        compile=GraphTrainerCompileConfig(
            enable_autoparallel=True,
            enable_async_tensor_parallel=False,
        ),
        model=SimpleNamespace(layers=[SimpleNamespace(feed_forward=None)]),
        parallelism=SimpleNamespace(
            fsdp_reshard_after_forward="always",
            pipeline_parallel_degree=1,
        ),
        training=TrainingConfig(disable_accelerator_graphs=True),
    )

    graph_passes = passes.construct_default_graph_passes(traced_result, config)
    pass_fns = [getattr(pass_fn, "func", pass_fn) for pass_fn in graph_passes]

    assert passes.tag_with_memory_policy_pass in pass_fns
    assert passes.selective_activation_remat_pass in pass_fns
    assert passes.apply_cpu_offload_pass in pass_fns
    assert passes.joint_transformer_block_bucketing_reordering_pass in pass_fns


@pytest.mark.parametrize(
    (
        "model_name",
        "inductor_compilation",
        "fsdp_reshard_after_forward",
        "expected_reshard_after_forward",
    ),
    [
        ("llama", "regional", "always", True),
        ("deepseek", "full", "never", False),
    ],
)
def test_model_autoparallel_uses_fx_module_path_and_resolved_policy(
    model_name,
    inductor_compilation,
    fsdp_reshard_after_forward,
    expected_reshard_after_forward,
):
    class FakeDeepSeekV3Model(torch.nn.Module):
        def __init__(self, config, *, mesh, compute_dtype):
            super().__init__()
            assert config.layers[0].feed_forward is not None
            assert config.layers[0].moe is None
            assert config.layers[1].feed_forward is None
            assert config.layers[1].moe is not None
            self.model_args = SimpleNamespace(vocab_size=16)

    _FakeAutoParallelGraph.instances.clear()
    compile_config = GraphTrainerCompileConfig(
        enable_autoparallel=True,
        inductor_compilation=inductor_compilation,
    )
    parallelism = ParallelismConfig(
        fsdp_reshard_after_forward=fsdp_reshard_after_forward
    )

    if model_name == "llama":
        from torchtitan.experiments.graph_trainer.llama3 import parallelize_autoparallel

        model = SimpleNamespace(
            config=SimpleNamespace(vocab_size=16, max_context_length=8),
            _get_attention_metadata=lambda *args, **kwargs: {},
        )
        parallelism_context = _FakeParallelismContext()
        call_parallelize = parallelize_autoparallel.parallelize_autoparallel_llama
        extra_patches = ()
    else:
        from torchtitan.experiments.graph_trainer.deepseek_v3 import (
            parallelize_autoparallel,
        )

        from torchtitan.models.deepseek_v3 import build_model_config

        model = SimpleNamespace(
            config=build_model_config("debugmodel", parallelism=ParallelismConfig())
        )
        parallelism_context = _FakeParallelismContext(sparse=True)
        call_parallelize = parallelize_autoparallel.parallelize_autoparallel_deepseekv3
        extra_patches = (
            patch.object(
                parallelize_autoparallel,
                "_load_autoparallel_dsv3_dependency",
                return_value=(FakeDeepSeekV3Model, lambda model: None),
            ),
        )

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                parallelize_autoparallel, "AutoParallelGraph", _FakeAutoParallelGraph
            )
        )
        stack.enter_context(
            patch.object(
                parallelize_autoparallel, "apply_compile", lambda model, **_: model
            )
        )
        for extra_patch in extra_patches:
            stack.enter_context(extra_patch)

        stack.enter_context(
            patch.object(parallelize_autoparallel, "device_type", "cpu")
        )
        call_parallelize(
            model,
            parallelism_context=parallelism_context,
            precision=PrecisionConfig(param_dtype="bfloat16", reduce_dtype="float32"),
            training=_training_config(),
            parallelism=parallelism,
            compile_config=compile_config,
            activation_remat=object(),
            dump_folder="",
            max_num_documents=4,
        )

    autop = _FakeAutoParallelGraph.instances[0]
    mp_policy = autop.kwargs["mp_policy"]
    assert mp_policy.param_dtype is torch.bfloat16
    assert mp_policy.reduce_dtype is torch.float32
    assert autop.kwargs["reshard_after_forward"] is expected_reshard_after_forward
    assert autop.kwargs.get("dynamic", False) is (model_name == "deepseek")
    assert autop.apply_kwargs["compile_config"] is compile_config
    assert autop.used_fx_path


def _varlen_metadata_model():
    from torchtitan.models.common.attention import (
        create_varlen_metadata_for_document,
        VarlenInnerAttention,
    )

    def metadata(positions, *, padding_mask, max_num_documents):
        return {
            VarlenInnerAttention: create_varlen_metadata_for_document(
                positions,
                padding_mask=padding_mask,
                max_num_documents=max_num_documents,
                max_context_length=8,
            )
        }

    return SimpleNamespace(
        config=SimpleNamespace(vocab_size=16, max_context_length=8),
        _get_attention_metadata=metadata,
    )


def _call_llama_adapter(model, *, dp_shard=1, max_num_documents=4, training=None):
    from torchtitan.experiments.graph_trainer.llama3 import parallelize_autoparallel

    context = _FakeParallelismContext()
    context.dp_shard = dp_shard
    _FakeAutoParallelGraph.instances.clear()
    with (
        patch.object(
            parallelize_autoparallel, "AutoParallelGraph", _FakeAutoParallelGraph
        ),
        patch.object(
            parallelize_autoparallel, "apply_compile", lambda model, **_: model
        ),
        patch.object(parallelize_autoparallel, "device_type", "cpu"),
    ):
        parallelize_autoparallel.parallelize_autoparallel_llama(
            model,
            parallelism_context=context,
            precision=PrecisionConfig(param_dtype="bfloat16", reduce_dtype="float32"),
            training=training or _training_config(),
            parallelism=ParallelismConfig(),
            compile_config=GraphTrainerCompileConfig(enable_autoparallel=True),
            activation_remat=None,
            dump_folder="",
            max_num_documents=max_num_documents,
        )
    return _FakeAutoParallelGraph.instances[-1]


def test_llama_trace_preserves_named_metadata_capacity_and_aliases():
    from torchtitan.models.common.attention import VarlenInnerAttention

    autop = _call_llama_adapter(_varlen_metadata_model())
    inputs = autop.inputs
    assert len(inputs.args) == 1
    assert list(inputs.kwargs) == ["positions", "padding_mask", "attention_metadata"]
    assert inputs.args[0].shape == (16,)
    assert inputs.kwargs["padding_mask"].dtype == torch.bool
    assert not inputs.kwargs["padding_mask"].any()
    metadata = inputs.kwargs["attention_metadata"][VarlenInnerAttention]
    assert metadata.cu_seq_q.shape == (7,)
    assert metadata.cu_seq_q.dtype == torch.int32
    assert metadata.cu_seq_q is metadata.cu_seq_k
    assert metadata.max_q == metadata.max_k == 8
    assert metadata.window_size == (-1, 0)
    torch.testing.assert_close(
        metadata.cu_seq_q, torch.tensor([0, 8, 16, 16, 16, 16, 16], dtype=torch.int32)
    )
    assert autop.input_constraints[:4] == [(Replicate(),)] * 4
    assert autop.input_constraints[4] is UNCONSTRAINED
    assert len(autop.input_constraints) == 9


def test_llama_trace_uses_microbatch_not_accumulated_step_budget():
    training = TrainingConfig(
        num_tokens_per_microbatch_per_dp_rank=16,
        num_tokens_per_train_step=64,
    )
    autop = _call_llama_adapter(_varlen_metadata_model(), training=training)
    assert autop.inputs.args[0].shape == (16,)
    assert autop.inputs.kwargs["padding_mask"].shape == (16,)


def test_llama_trace_rejects_unrepresentable_data_parallel_offsets():
    with pytest.raises(ValueError, match="data parallel degree one"):
        _call_llama_adapter(_varlen_metadata_model(), dp_shard=2)


def test_llama_trace_requires_fixed_document_capacity():
    with pytest.raises(ValueError, match="fixed dataloader max_num_documents"):
        _call_llama_adapter(_varlen_metadata_model(), max_num_documents=None)


@pytest.mark.parametrize("enabled", [False, True])
def test_graph_engine_forwards_document_capacity_only_for_autoparallel(enabled):
    from torchtitan.experiments.graph_trainer.trainer import GraphTrainingEngine

    engine = GraphTrainingEngine.__new__(GraphTrainingEngine)
    engine.config = SimpleNamespace(
        compile=GraphTrainerCompileConfig(enable_autoparallel=enabled),
        training=_training_config(),
    )
    engine.parallelism_context = SimpleNamespace(pp_enabled=False)
    engine.max_num_documents = 32
    kwargs = engine._parallelize_kwargs()
    assert kwargs["training"] is engine.config.training
    assert kwargs["compile_config"] is engine.config.compile
    if enabled:
        assert kwargs["max_num_documents"] == 32
    else:
        assert "max_num_documents" not in kwargs


def test_graph_engine_rejects_autoparallel_pipeline_parallel():
    from torchtitan.experiments.graph_trainer.trainer import GraphTrainingEngine

    engine = GraphTrainingEngine.__new__(GraphTrainingEngine)
    engine.config = SimpleNamespace(
        compile=GraphTrainerCompileConfig(enable_autoparallel=True),
        training=_training_config(),
    )
    engine.parallelism_context = SimpleNamespace(pp_enabled=True)
    engine.max_num_documents = 32
    with pytest.raises(ValueError, match="does not support pipeline parallelism"):
        engine._parallelize_kwargs()
    engine.config.compile.enable_autoparallel = False
    assert engine._parallelize_kwargs() == {"compile_config": engine.config.compile}
