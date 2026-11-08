# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file provides the configs that apply activation remat (activation
# checkpointing) to the model. Technically, this is not a part of distributed,
# but distributed module is the best place to put it.

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

import torch
import torch.nn as nn
import torch_remat as remat

from torchtitan.models.base.module import Module


logger = logging.getLogger(__name__)

_T = TypeVar("_T")
_PackHook = Callable[[torch.Tensor], object]
_UnpackHook = Callable[[object], torch.Tensor]


_BLOCK_FQN = re.compile(r"([A-Za-z_]\w*\.)?layers\.\d+")


@dataclass(kw_only=True, slots=True)
class BlockRematOverride:
    """Remat policy fields for selected transformer blocks.

    A field left ``None`` inherits the base policy's value; a field that is
    set replaces it for these blocks (``recompute_regions`` still takes
    precedence over ``save_regions`` within the result).
    """

    blocks: list[str]
    """
    Exact FQNs of the blocks in the original model: ``layers.<id>`` for
    decoder blocks and ``<encoder>.layers.<id>`` for encoder blocks, e.g.
    ``vision_encoder.layers.3``. Pipeline splitting keeps original layer ids,
    so a name follows its block to whichever stage holds it. Names are not
    checked against the model, since each pipeline stage holds only some
    blocks.
    """

    save_regions: list[str] | None = None
    recompute_regions: list[str] | None = None
    offload_block_input: bool | None = None
    offload_regions: list[str] | None = None


@dataclass(frozen=True, slots=True)
class BlockRematPolicy:
    """The remat policy of one transformer block, after overrides."""

    save_regions: list[str]
    recompute_regions: list[str]
    offload_block_input: bool
    offload_regions: list[str]

    @property
    def offloads_activations(self) -> bool:
        return self.offload_block_input or bool(self.offload_regions)


@dataclass(kw_only=True, slots=True)
class ActivationRematConfig:
    """Retain selected model-declared regions and recompute everything else.

    Activation rematerialization, also known as activation checkpointing:
    ``apply`` checkpoints each transformer block with ``torch_remat``. Regions
    that model code always retains with ``recompute=False``, such as MoE
    routing decisions, are saved under every policy.
    """

    save_regions: list[str]
    """
    Qualified save-region glob patterns, relative to a transformer block.
    Region names are defined at the corresponding ``torch_remat.region``
    call sites in model code. Everything outside a retained region is
    recomputed.

    Save-region names are relative to a transformer block, and the policy
    applies to every block; ``block_overrides`` changes it for selected
    blocks.
    """

    recompute_regions: list[str] = field(default_factory=list)
    """
    Region glob patterns to recompute. ``recompute_regions`` takes precedence
    over ``save_regions``: a region that matches both is recomputed. For
    example, with ``save_regions=["*"]``, this lists the few regions to
    recompute, which suits starting from no remat and recomputing just enough
    to fit a memory budget.
    """

    offload_block_input: bool = False
    """
    Offload each transformer block's inputs that require grad (its input
    hidden state) to CPU instead of keeping them on GPU. Under full remat
    this is the whole stash. Inputs shared across blocks (positions, masks)
    do not require grad and stay on GPU, as does an input sharing storage
    with the model part's own input (e.g. a pipeline stage's input, which the
    stage retains). Requires ``TrainingEngine.Config.activation_offload``;
    see torchtitan/distributed/ACTIVATION_OFFLOAD.md.
    """

    offload_regions: list[str] = field(default_factory=list)
    """
    Glob patterns of save regions, relative to a transformer block, whose
    retained tensors are offloaded to CPU instead of kept on GPU: the tensors
    a region's ops save for backward and its output when recomputation reads
    it. Only retained regions have tensors to offload, so a pattern that
    matches only recomputed regions offloads nothing; ``["*"]`` offloads
    everything retained. A storage is offloaded once for all its saves: a
    save of an already-offloaded storage is offloaded whatever its region,
    and a save of a storage another save keeps on GPU stays on GPU, since
    offloading it would free nothing. Requires
    ``TrainingEngine.Config.activation_offload``.
    """

    block_overrides: list[BlockRematOverride] = field(default_factory=list)
    """
    Per-block changes to the fields above, e.g. recomputing an expensive
    region in only a few blocks, or retaining it in only a few. Each block
    may be named by at most one override.
    """

    preserve_rng_state: bool = False
    """
    Must remain false. torch_remat requires explicit RecomputeStateHooks for
    random state that can advance inside retained regions.
    """

    determinism_check: str = "default"
    """
    A string specifying the determinism function. See
    https://docs.pytorch.org/docs/stable/checkpoint.html for details.
    """

    debug: bool = False
    """Not supported by torch_remat; must remain false."""

    def __post_init__(self) -> None:
        if self.preserve_rng_state:
            raise ValueError(
                "torch_remat activation checkpointing does not support "
                "preserve_rng_state=True. Register a "
                "torch_remat RecomputeStateHook for random state used in retained "
                "regions."
            )
        if self.debug:
            raise ValueError(
                "torch_remat activation checkpointing does not support the debug "
                "option."
            )
        overridden: set[str] = set()
        for override in self.block_overrides:
            if not override.blocks:
                raise ValueError("BlockRematOverride.blocks must not be empty.")
            if (
                override.save_regions is None
                and override.recompute_regions is None
                and override.offload_block_input is None
                and override.offload_regions is None
            ):
                raise ValueError(
                    f"BlockRematOverride for {override.blocks} sets no field."
                )
            for block_fqn in override.blocks:
                if _BLOCK_FQN.fullmatch(block_fqn) is None:
                    raise ValueError(
                        "BlockRematOverride.blocks must name transformer blocks "
                        "as 'layers.<id>' or '<encoder>.layers.<id>'; got "
                        f"{block_fqn!r}."
                    )
                if block_fqn in overridden:
                    raise ValueError(
                        f"Block {block_fqn!r} is named by more than one "
                        "BlockRematOverride."
                    )
                overridden.add(block_fqn)

    def block_policy(self, block_fqn: str) -> BlockRematPolicy:
        """The policy of the block with original FQN ``block_fqn``."""
        policy = BlockRematPolicy(
            save_regions=self.save_regions,
            recompute_regions=self.recompute_regions,
            offload_block_input=self.offload_block_input,
            offload_regions=self.offload_regions,
        )
        for override in self.block_overrides:
            if block_fqn in override.blocks:
                return BlockRematPolicy(
                    save_regions=_override(override.save_regions, policy.save_regions),
                    recompute_regions=_override(
                        override.recompute_regions, policy.recompute_regions
                    ),
                    offload_block_input=_override(
                        override.offload_block_input, policy.offload_block_input
                    ),
                    offload_regions=_override(
                        override.offload_regions, policy.offload_regions
                    ),
                )
        return policy

    @property
    def offloads_activations(self) -> bool:
        """Whether this policy offloads any retained tensor to CPU, in any
        block."""
        return (
            self.offload_block_input
            or bool(self.offload_regions)
            or any(
                override.offload_block_input or override.offload_regions
                for override in self.block_overrides
            )
        )

    def _check_fixed_regions(
        self, save_regions: list[str], recompute_regions: list[str]
    ) -> None:
        if (
            self.save_regions != save_regions
            or self.recompute_regions != recompute_regions
            or any(
                override.save_regions is not None
                or override.recompute_regions is not None
                for override in self.block_overrides
            )
        ):
            raise ValueError(
                f"{type(self).__name__} is a fixed policy. Use "
                "ActivationRematConfig with explicit save_regions and "
                "recompute_regions to customize it (offload_block_input and "
                "offload_regions can be set on any policy, also per block)."
            )

    def apply(
        self,
        model: Module,
        *,
        module_fqn: str = "",
        saved_tensors_hooks: (
            Callable[[nn.Module, str], tuple[_PackHook, _UnpackHook] | None] | None
        ) = None,
    ) -> None:
        """Checkpoint every transformer block of ``model`` with ``torch_remat``.

        Args:
            model: Model whose ``layers`` children are the transformer blocks.
            module_fqn: FQN of ``model`` within the full model, such as
                ``"vision_encoder"``; empty for the root model. Prefixes the
                block FQNs that ``block_overrides`` are matched against.
            saved_tensors_hooks: Optional callable returning the saved-tensor
                hook pair for one block, given the block and its FQN
                (``layers.<id>``), or None to leave that block unhooked. The
                pair is passed to
                ``torch_remat.checkpoint(saved_tensors_hooks=...)``: ``pack``
                replaces each tensor the block retains for backward with an
                opaque payload and ``unpack`` rebuilds it, which enables CPU
                offloading, compression, or logging of retained activations.
                It sees the block's inputs and the tensors retained by its
                save regions; ``torch_remat.current_saved_tensor_info().kind``
                tells ``pack`` which one it is handling. Called once per
                block, so each block can own its hook state (e.g. an offload
                buffer). torch_remat rejects the pair under ``torch.compile``
                around the block. See torchtitan/distributed/REMAT.md and
                https://github.com/meta-pytorch/remat/blob/main/docs/offloading.md.
        """
        layers = model.get_submodule("layers")
        transformer_blocks = list(layers.named_children())
        if not transformer_blocks:
            logger.info(
                "%s found no transformer blocks in this model part",
                type(self).__name__,
            )
            return

        # TODO: Validate unmatched patterns once validation can account for save
        # regions across all pipeline stages instead of only this model part.
        overridden, not_offloaded = [], []
        for layer_id, transformer_block in transformer_blocks:
            assert isinstance(transformer_block, Module)
            base_fqn = f"layers.{layer_id}"
            block_fqn = f"{module_fqn}.{base_fqn}" if module_fqn else base_fqn
            policy = self.block_policy(block_fqn)
            if any(block_fqn in override.blocks for override in self.block_overrides):
                overridden.append(block_fqn)
            if module_fqn and policy.offloads_activations:
                not_offloaded.append(block_fqn)
            transformer_block.configure_remat_regions(
                policy.save_regions, policy.recompute_regions
            )
            transformer_block.forward = remat.checkpoint(
                region_name=base_fqn,
                determinism_check=self.determinism_check,
                preserve_rng_state=False,
                saved_tensors_hooks=(
                    saved_tensors_hooks(transformer_block, base_fqn)
                    if saved_tensors_hooks is not None
                    else None
                ),
            )(transformer_block.forward)
        logger.info(
            "Applied %s to %d transformer blocks. Save patterns: %s, "
            "recompute patterns: %s",
            type(self).__name__,
            len(transformer_blocks),
            self.save_regions or "none",
            self.recompute_regions or "none",
        )
        if overridden:
            logger.info("Block overrides apply to: %s", ", ".join(overridden))
        if not_offloaded:
            # The offloader hooks only the model part's own (decoder) blocks.
            logger.warning(
                "Activation offload does not apply to encoder blocks; these keep "
                "their retained tensors on GPU: %s",
                ", ".join(not_offloaded),
            )


@dataclass(kw_only=True, slots=True)
class FullActivationRematConfig(ActivationRematConfig):
    """A fixed policy that saves no policy-controlled region.

    Every operation in a transformer block is recomputed except the regions
    model code always retains with ``recompute=False``, such as MoE routing
    decisions, auxiliary-loss accumulation, and trailing residual adds.
    """

    save_regions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        super(FullActivationRematConfig, self).__post_init__()
        self._check_fixed_regions([], [])


_DEFAULT_SAVE_REGIONS = ["*"]
_DEFAULT_RECOMPUTE_REGIONS = ["*routed_experts.w13.*"]


@dataclass(kw_only=True, slots=True)
class DefaultActivationRematConfig(ActivationRematConfig):
    """The default remat policy and the trainer's default.

    Saves every model-declared region except the routed-expert ``w13`` grouped
    projection, whose saved activations scale with top-k and dominate MoE
    activation memory. It balances step time and memory for current MoE
    models and equals saving every region for dense models. Its regions may
    change as measurements improve; use ``ActivationRematConfig`` with
    explicit regions for a fixed policy. ``w2`` stays saved: its saved input is the activation
    output, which replay rebuilds anyway, so recomputing ``w2`` would cost time
    without freeing memory. Other regions under ``routed_experts`` (e.g. the EP
    token-dispatcher all-to-alls) are retained, so recomputation never replays
    EP communication. Code outside any model-declared region is always
    recomputed, so a model that declares no regions gets full recomputation.
    """

    save_regions: list[str] = field(default_factory=lambda: list(_DEFAULT_SAVE_REGIONS))
    recompute_regions: list[str] = field(
        default_factory=lambda: list(_DEFAULT_RECOMPUTE_REGIONS)
    )

    def __post_init__(self) -> None:
        super(DefaultActivationRematConfig, self).__post_init__()
        self._check_fixed_regions(_DEFAULT_SAVE_REGIONS, _DEFAULT_RECOMPUTE_REGIONS)


def _override(value: _T | None, default: _T) -> _T:
    return default if value is None else value
