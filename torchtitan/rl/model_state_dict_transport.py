# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Moves the trainer's model state dict into the generators' prefetch buffers with
one-sided reads over a ``torch.distributed._transport`` backend.

See ``MODEL_STATE_DICT_TRANSPORT.md``.
"""

from __future__ import annotations

import asyncio
import itertools
import math
import weakref
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

import torch

# TODO: torch.distributed._transport, torch.cuda._pin_memory_utils,
# torch.distributed.tensor._utils and placement_types._StridedShard are private
# upstream APIs; switch to public ones once they exist (no upstream issue yet).
import torch.cuda._pin_memory_utils as pin_memory_utils
from torch.distributed._transport import new_transport, wait_all
from torch.distributed.tensor import DTensor, Partial, Placement, Shard
from torch.distributed.tensor._utils import (
    _compute_local_shape_and_global_offset,
    compute_local_shape_and_global_offset,
)
from torch.distributed.tensor.placement_types import _StridedShard

from torchtitan.config import Configurable

# Byte alignment of each staged piece, so it can be viewed as any dtype.
_ALIGNMENT = 16


class TensorShard(NamedTuple):
    """A generator rank's box of one tensor and the dtype it receives it in."""

    offset: tuple[int, ...]
    """Global offset of the box."""
    shape: tuple[int, ...]
    """Shape of the box, i.e. of the rank's local tensor."""
    dtype: torch.dtype
    """Dtype of the generator rank's local tensor."""


class StagedPiece(NamedTuple):
    """The part of a generator rank's shard one trainer rank holds, and where that
    trainer rank stages it."""

    byte_offset: int
    """Offset in the trainer rank's staging buffer."""
    offset: tuple[int, ...]
    """Global offset of the piece."""
    shape: tuple[int, ...]
    """Shape of the intersection between the trainer and generator boxes."""


class TransferRange(NamedTuple):
    """One contiguous byte range transferred from a trainer buffer."""

    byte_offset: int
    """Start of the range in the registered trainer buffer."""
    nbytes: int
    """Number of bytes in the range."""


@dataclass(frozen=True, slots=True)
class ModelStateDictTransferMetadata:
    """Small control-plane metadata returned by one trainer rank's push."""

    changed_ranges: tuple[TransferRange, ...]
    """Ranges changed since the preceding delta push; empty for full sync."""
    changed_keys: frozenset[str] = frozenset()
    """Keys containing a changed range."""


@dataclass(frozen=True, slots=True)
class TrainerSource:
    """What a generator rank needs to read one trainer rank's staging buffers."""

    pieces: list[dict[str, StagedPiece]]
    """``pieces[rank][key]``: the staged part of generator rank ``rank``'s shard of
    ``key``, for each key this trainer rank holds part of."""
    baseline_buffer: Any
    """Primary staging buffer; the previous state when delta sync is used."""
    current_buffer: Any
    """Current-state buffer for delta sync; aliases ``baseline_buffer`` otherwise."""
    url: bytes
    """Bound transport endpoint for this trainer rank."""


class _Read(NamedTuple):
    source: int
    """Index of the trainer rank in the generator's sources."""
    byte_offset: int
    """Offset in the trainer rank's staging buffer."""
    nbytes: int
    """Number of bytes in this contiguous transfer."""
    key: str
    """Key of the generator tensor that receives the transfer."""
    local_byte_offset: int
    """Offset in the generator's local tensor of ``key``."""


def _global_offset_and_local_shape(
    tensor: torch.Tensor, placements: tuple[Placement, ...] | None = None
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Global offset and shape of this rank's local part of ``tensor``, if placed
    with ``placements`` (default: its own)."""
    if not isinstance(tensor, DTensor):
        return (0,) * tensor.ndim, tuple(tensor.shape)
    placements = tensor.placements if placements is None else placements
    for p in placements:
        if isinstance(p, Partial):
            raise ValueError(f"Partial placements are unsupported: {placements}")
        # An uneven strided shard is not one contiguous box.
        if isinstance(p, _StridedShard) and tensor.shape[p.dim] % p.split_factor:
            raise ValueError(
                f"Uneven {p} of a tensor of shape {tuple(tensor.shape)} is unsupported"
            )
    shape, offset = compute_local_shape_and_global_offset(
        tensor.shape, tensor.device_mesh, placements
    )
    return tuple(offset), tuple(shape)


def _staged_placements(
    tensor: DTensor, generator_shards: list[TensorShard]
) -> tuple[Placement, ...] | None:
    """The placements the trainer stages ``tensor`` in, or None for its own.

    A piece is read as one contiguous run per index of the dims before the last
    one it does not span. A trainer box that cuts a generator box on a dim after
    the first sharded dim multiplies the runs, so its shards of later dims are
    then moved to the first sharded dim. E.g. a row-parallel weight (FSDP
    ``Shard(0)``, TP ``Shard(1)``) read by generator ranks split on dim 0 is
    staged as ``(Shard(0), Shard(0))``, read at once. If every trainer box
    contains the generator boxes it overlaps on those dims (e.g. generator TP
    on dim 1 matching the trainer's), the restage would not reduce the runs.

    Decided from every trainer rank's box so all trainer ranks agree: a
    redistribute is a collective.
    """
    dims = [p.dim for p in tensor.placements if isinstance(p, Shard)]
    if not dims:
        return None
    first = min(dims)
    staged = tuple(
        Shard(first) if isinstance(p, (Shard, _StridedShard)) and p.dim > first else p
        for p in tensor.placements
    )
    if staged == tensor.placements:
        return None
    # Only the axes sharding later dims change a trainer box on those dims.
    axes = [
        axis
        for axis, p in enumerate(tensor.placements)
        if isinstance(p, (Shard, _StridedShard)) and p.dim > first
    ]
    mesh_shape = tensor.device_mesh.shape
    coordinate = [0] * len(mesh_shape)
    later = range(first + 1, tensor.ndim)
    for index in itertools.product(*(range(mesh_shape[axis]) for axis in axes)):
        for axis, i in zip(axes, index):
            coordinate[axis] = i
        shape, offset = _compute_local_shape_and_global_offset(
            tensor.shape, mesh_shape, coordinate, tensor.placements
        )
        for shard in generator_shards:
            spans = [
                (
                    offset[d],
                    offset[d] + shape[d],
                    shard.offset[d],
                    shard.offset[d] + shard.shape[d],
                )
                for d in later
            ]
            overlaps = all(lo < g_hi and g_lo < hi for lo, hi, g_lo, g_hi in spans)
            contains = all(lo <= g_lo and g_hi <= hi for lo, hi, g_lo, g_hi in spans)
            if overlaps and not contains:
                return staged
    return None


def _pinned_buffer(nbytes: int) -> torch.Tensor:
    """A pinned CPU byte buffer, pageable without an accelerator."""
    if not torch.cuda.is_available():
        return torch.empty(
            nbytes, dtype=torch.uint8, pin_memory=torch.accelerator.is_available()
        )
    # TODO: switch to a device-agnostic API once one exists. Unlike
    # pin_memory=True, this does not round the buffer up to a power of two.
    buffer = torch.empty(nbytes, dtype=torch.uint8)
    pin_memory_utils.pin_memory(buffer.data_ptr(), buffer.nbytes)
    weakref.finalize(buffer, pin_memory_utils.unpin_memory, buffer.data_ptr())
    return buffer


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def shard_layout(state_dict: dict[str, torch.Tensor]) -> dict[str, TensorShard]:
    """The layout of a generator rank's state dict, in its key order."""
    return {
        key: TensorShard(*_global_offset_and_local_shape(tensor), tensor.dtype)
        for key, tensor in state_dict.items()
    }


def shared_rank_layouts(
    generator_layouts: list[list[tuple[int, dict[str, TensorShard]]]],
) -> list[dict[str, TensorShard]]:
    """The ``shard_layout`` of each generator rank, in rank order.

    The trainer stages one copy that every generator reads, so every generator
    must have the same layout.

    Args:
        generator_layouts: Each generator's ``(rank, shard_layout)`` pairs, in any
            order.

    Raises:
        ValueError: if the generators' layouts differ.
    """
    layouts = []
    for pairs in generator_layouts:
        by_rank = dict(pairs)
        layouts.append([by_rank[rank] for rank in range(len(by_rank))])
    if any(layout != layouts[0] for layout in layouts[1:]):
        raise ValueError("the generators' model state dict layouts differ")
    return layouts[0]


def _staged_pieces(
    trainer_boxes: dict[str, tuple[tuple[int, ...], tuple[int, ...]]],
    generator_rank_layouts: list[dict[str, TensorShard]],
) -> tuple[list[dict[str, StagedPiece]], int]:
    """Lay out a trainer rank's staging buffer: for each generator rank in order, the
    intersection of each of its boxes with the trainer rank's, in its key order.
    Generator ranks that receive the same piece (e.g. a tensor replicated over TP)
    share one copy.

    Returns the pieces per generator rank and the buffer size in bytes.
    """
    nbytes = 0
    pieces = []
    byte_offsets: dict[tuple, int] = {}
    for layout in generator_rank_layouts:
        rank_pieces = {}
        for key, shard in layout.items():
            if key not in trainer_boxes:
                continue
            trainer_offset, trainer_shape = trainer_boxes[key]
            lo = [max(a, b) for a, b in zip(trainer_offset, shard.offset)]
            hi = [
                min(a + n, b + m)
                for a, n, b, m in zip(
                    trainer_offset, trainer_shape, shard.offset, shard.shape
                )
            ]
            if any(h <= low for low, h in zip(lo, hi)):
                continue
            shape = tuple(h - low for low, h in zip(lo, hi))
            byte_offset = byte_offsets.setdefault((key, tuple(lo), shape), nbytes)
            rank_pieces[key] = StagedPiece(byte_offset, tuple(lo), shape)
            if byte_offset == nbytes:
                size = math.prod(shape) * shard.dtype.itemsize
                nbytes += -(-size // _ALIGNMENT) * _ALIGNMENT
        pieces.append(rank_pieces)
    return pieces, nbytes


def _staged_transfer_ranges(
    pieces: list[dict[str, StagedPiece]],
    generator_rank_layouts: list[dict[str, TensorShard]],
) -> tuple[tuple[TransferRange, ...], dict[TransferRange, str]]:
    """Return the distinct direct tensor ranges staged by a trainer rank."""
    range_keys = {}
    for rank_pieces, layout in zip(pieces, generator_rank_layouts):
        for key, piece in rank_pieces.items():
            shard = layout[key]
            dim = max(
                (d for d, (n, m) in enumerate(zip(piece.shape, shard.shape)) if n != m),
                default=0,
            )
            run_nbytes = math.prod(piece.shape[dim:]) * shard.dtype.itemsize
            for i, _ in enumerate(itertools.product(*map(range, piece.shape[:dim]))):
                transfer_range = TransferRange(
                    piece.byte_offset + i * run_nbytes, run_nbytes
                )
                previous = range_keys.setdefault(transfer_range, key)
                assert previous == key

    sorted_ranges = sorted(range_keys)
    for transfer_range, next_range in itertools.pairwise(sorted_ranges):
        if next_range.byte_offset < transfer_range.byte_offset + transfer_range.nbytes:
            raise ValueError("staged transfer ranges overlap")
    return tuple(sorted_ranges), range_keys


def _check_disjoint(
    key: str, boxes: list[tuple[tuple[int, ...], tuple[int, ...]]]
) -> None:
    """Raise if any two ``(offset, shape)`` boxes overlap."""
    boxes = sorted(boxes)
    for i, (offset, shape) in enumerate(boxes):
        for other_offset, other_shape in boxes[i + 1 :]:
            # Sorted by offset: later boxes start past this one on dim 0.
            if other_offset[0] >= offset[0] + shape[0]:
                break
            if all(
                o < so + m and so < o + n
                for o, n, so, m in zip(offset, shape, other_offset, other_shape)
            ):
                raise ValueError(f"{key}: the trainer stages overlapping pieces")


def _plan_reads(
    sources: list[TrainerSource],
    layout: dict[str, TensorShard],
    *,
    rank: int,
    generator_index: int,
) -> list[_Read]:
    """Plan the reads that fill generator rank ``rank``'s shards from the trainer.

    When several trainer ranks stage the same piece (e.g. HSDP), generator ranks
    are spread across them round-robin, by generator then rank.

    Raises:
        ValueError: if the trainer does not cover a shard exactly once.
    """
    # Index of this rank across every generator's ranks.
    spread_index = generator_index * len(sources[0].pieces) + rank
    reads = []
    for key, shard in layout.items():
        # (offset, shape) -> the trainer ranks staging that piece.
        stagers: dict[tuple, list[tuple[int, StagedPiece]]] = {}
        for i, source in enumerate(sources):
            piece = source.pieces[rank].get(key)
            if piece is not None:
                stagers.setdefault((piece.offset, piece.shape), []).append((i, piece))

        _check_disjoint(key, list(stagers))
        numel = 0
        itemsize = shard.dtype.itemsize
        for piece_stagers in stagers.values():
            source, piece = piece_stagers[spread_index % len(piece_stagers)]
            if any(
                o < so or o + n > so + m
                for o, n, so, m in zip(
                    piece.offset, piece.shape, shard.offset, shard.shape
                )
            ):
                raise ValueError(
                    f"{key}: a staged piece is outside the generator shard"
                )
            numel += math.prod(piece.shape)
            # One contiguous run per index of the dims before the last one the
            # piece does not span (see _staged_placements).
            dim = max(
                (d for d, (n, m) in enumerate(zip(piece.shape, shard.shape)) if n != m),
                default=0,
            )
            strides = [math.prod(shard.shape[d + 1 :]) for d in range(len(shard.shape))]
            run = math.prod(piece.shape[dim:])
            for i, index in enumerate(
                itertools.product(*map(range, piece.shape[:dim]))
            ):
                start = sum(
                    (o - so + j) * stride
                    for o, so, j, stride in zip(
                        piece.offset, shard.offset, (*index, 0), strides
                    )
                )
                reads.append(
                    _Read(
                        source,
                        piece.byte_offset + i * run * itemsize,
                        run * itemsize,
                        key,
                        start * itemsize,
                    )
                )
        if numel != math.prod(shard.shape):
            raise ValueError(
                f"{key}: the trainer covers {numel} of the generator shard's "
                f"{math.prod(shard.shape)} elements"
            )
    return reads


class ModelStateDictTransport(Configurable):
    """Moves the model state dict from the trainer into the generators' prefetched
    model state dicts.

    The trainer and every generator rank each build one. The trainer stages its
    shards, cast to the generators' dtypes, in registered pinned CPU buffers laid
    out per generator rank. Full sync reads every range; delta sync reads the first
    transfer in full and only changed ranges afterward. Each generator rank reads
    its pieces from every trainer rank into its registered prefetch buffers.

    Trainer: ``register_trainer_source`` once, then ``stage`` per push.
    Generator: ``connect_generator`` once, then ``read`` per prefetch.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        backend: Literal["mooncake", "nixl"] = "mooncake"
        """The ``torch.distributed._transport`` backend."""

        sync_mode: Literal["full", "delta"] = "full"
        """Whether every pull reads the full state or only changed ranges."""

        read_timeout_s: float = 600.0
        """Seconds to wait for a generator rank's reads before raising. A pull
        moves tens of GB per generator rank at most, well under a minute at line
        rate; this fails a stuck transfer instead of hanging the controller."""

    def __init__(self, config: Config):
        if config.sync_mode not in ("full", "delta"):
            raise ValueError(f"Invalid sync mode: {config.sync_mode}")
        self._transport = new_transport(config.backend)
        self._sync_mode = config.sync_mode
        self._read_timeout_s = config.read_timeout_s
        # Trainer: the primary and optional delta staging buffers and the fixed
        # direct tensor ranges compared for delta sync.
        self._baseline: torch.Tensor | None = None
        self._current: torch.Tensor | None = None
        self._transfer_ranges: tuple[TransferRange, ...] = ()
        self._transfer_range_keys: dict[TransferRange, str] = {}
        self._baseline_initialized = False
        self._staging_plan: dict[
            str,
            tuple[
                torch.dtype,
                tuple[Placement, ...] | None,
                list[
                    tuple[
                        tuple[slice, ...],
                        torch.Tensor,
                        torch.Tensor,
                    ]
                ],
            ],
        ] = {}
        # Generator: the registered local, baseline, and current views for each
        # existing direct read, or None until connected.
        self._reads: list[tuple[_Read, Any, Any, Any]] | None = None
        self._gpu_reads: list[tuple[_Read, Any]] | None = None
        self._gpu_keys: frozenset[str] = frozenset()

    def register_trainer_source(
        self,
        model_state_dict: dict[str, torch.Tensor],
        *,
        generator_rank_layouts: list[dict[str, TensorShard]],
        transfer_dtype: torch.dtype | None,
        preserve_dtype_keys: frozenset[str],
    ) -> TrainerSource:
        """Allocate and register this trainer rank's staging buffers for the
        generator ranks' layouts, which every generator shares.

        Args:
            model_state_dict: This trainer rank's model state dict.
            generator_rank_layouts: ``shard_layout`` of each rank of a generator,
                in rank order.
            transfer_dtype: The dtype parameters are sent in, or None for their own.
            preserve_dtype_keys: Keys sent in their own dtype.

        Raises:
            ValueError: if a generator receives a key in a dtype other than the
                trainer sends it in.
        """
        dtypes = {}
        placements = {}
        boxes = {}
        for key, tensor in model_state_dict.items():
            if not any(key in layout for layout in generator_rank_layouts):
                continue
            dtypes[key] = (
                tensor.dtype
                if transfer_dtype is None or key in preserve_dtype_keys
                else transfer_dtype
            )
            for layout in generator_rank_layouts:
                if key in layout and layout[key].dtype != dtypes[key]:
                    raise ValueError(
                        f"{key}: the trainer sends {dtypes[key]}, the generator "
                        f"receives {layout[key].dtype}"
                    )
            boxes[key] = _global_offset_and_local_shape(tensor)
            if not isinstance(tensor, DTensor):
                continue
            staged = _staged_placements(
                tensor,
                [layout[key] for layout in generator_rank_layouts if key in layout],
            )
            if staged is not None:
                placements[key] = staged
                boxes[key] = _global_offset_and_local_shape(tensor, staged)

        pieces, nbytes = _staged_pieces(boxes, generator_rank_layouts)
        if self._sync_mode == "delta":
            transfer_ranges, transfer_range_keys = _staged_transfer_ranges(
                pieces, generator_rank_layouts
            )
        else:
            transfer_ranges, transfer_range_keys = (), {}
        baseline = _pinned_buffer(max(nbytes, 1))
        baseline_memory = self._transport.register_memory(baseline)
        if self._sync_mode == "delta":
            current = _pinned_buffer(max(nbytes, 1))
            current_memory = self._transport.register_memory(current)
        else:
            current = baseline
            current_memory = baseline_memory
        self._staging_plan = {
            key: (dtype, placements.get(key), []) for key, dtype in dtypes.items()
        }
        copied = set()
        for rank_pieces in pieces:
            for key, piece in rank_pieces.items():
                if piece.byte_offset in copied:
                    continue
                copied.add(piece.byte_offset)
                dtype = dtypes[key]
                start = piece.byte_offset
                end = start + math.prod(piece.shape) * dtype.itemsize
                index = tuple(
                    slice(o - to, o - to + n)
                    for o, to, n in zip(piece.offset, boxes[key][0], piece.shape)
                )
                baseline_view = baseline[start:end].view(dtype).view(piece.shape)
                current_view = current[start:end].view(dtype).view(piece.shape)
                self._staging_plan[key][2].append((index, baseline_view, current_view))
        # Keep restaged keys even without pieces on this rank: a redistribute is
        # a collective.
        self._staging_plan = {
            key: plan
            for key, plan in self._staging_plan.items()
            if plan[1] is not None or plan[2]
        }
        self._baseline = baseline
        self._current = current
        self._transfer_ranges = transfer_ranges
        self._transfer_range_keys = transfer_range_keys
        return TrainerSource(
            pieces,
            baseline_memory.to_remote_buffer(),
            current_memory.to_remote_buffer(),
            self._transport.bind(),
        )

    def stage(self, model_state_dict: dict[str, torch.Tensor]) -> None:
        """Cast and reshard this trainer rank's shards and copy them into the
        staging buffer.

        A collective over the mesh of each restaged DTensor: every trainer rank
        must call it, in the same order relative to its other collectives. The
        copies are asynchronous; synchronize the current stream before generators
        read.
        """
        with torch.no_grad():
            for key, (dtype, placements, copies) in self._staging_plan.items():
                # Cast on the device: a casting device-to-host copy_ goes through
                # a pageable temporary and casts on the CPU.
                tensor = model_state_dict[key].to(dtype)
                if placements is not None:
                    tensor = tensor.redistribute(placements=placements)
                local = _local(tensor)
                for index, baseline_view, current_view in copies:
                    view = (
                        current_view
                        if self._sync_mode == "delta" and self._baseline_initialized
                        else baseline_view
                    )
                    view.copy_(local[index], non_blocking=True)

    def finish_stage(self) -> ModelStateDictTransferMetadata:
        """Finish staging and report ranges changed for delta sync."""
        if self._sync_mode == "full":
            return ModelStateDictTransferMetadata(())
        if not self._baseline_initialized:
            self._baseline_initialized = True
            return ModelStateDictTransferMetadata(())

        assert self._baseline is not None
        assert self._current is not None
        changed_ranges = []
        changed_keys = set()
        with torch.no_grad():
            for transfer_range in self._transfer_ranges:
                start = transfer_range.byte_offset
                end = start + transfer_range.nbytes
                baseline = self._baseline[start:end]
                current = self._current[start:end]
                if torch.equal(baseline, current):
                    continue
                changed_ranges.append(transfer_range)
                changed_keys.add(self._transfer_range_keys[transfer_range])
                baseline.copy_(current)
        return ModelStateDictTransferMetadata(
            tuple(changed_ranges), frozenset(changed_keys)
        )

    def connect_generator(
        self,
        prefetched_model_state_dict: dict[str, torch.Tensor],
        *,
        sources: list[TrainerSource],
        rank: int,
        generator_index: int,
    ) -> None:
        """Register this generator rank's prefetch buffers, connect to every trainer
        rank and plan the reads.

        Args:
            prefetched_model_state_dict: The buffers ``read`` fills, whose
                ``shard_layout`` the trainer staged for.
            sources: Every trainer rank's ``register_trainer_source`` result.
            rank: This rank within its generator.
            generator_index: Index of this rank's generator.
        """
        reads = _plan_reads(
            sources,
            shard_layout(prefetched_model_state_dict),
            rank=rank,
            generator_index=generator_index,
        )
        for source in sources:
            self._transport.connect(source.url)
        # Only keys with reads: backends reject registering empty tensors.
        memories = {
            key: self._transport.register_memory(
                _local(prefetched_model_state_dict[key]).view(-1)
            )
            for key in {r.key for r in reads}
        }
        self._reads = []
        for read in reads:
            source = sources[read.source]
            local = memories[read.key].to_mutable_view(
                read.local_byte_offset, read.nbytes
            )
            self._reads.append(
                (
                    read,
                    local,
                    source.baseline_buffer.slice(read.byte_offset, read.nbytes),
                    source.current_buffer.slice(read.byte_offset, read.nbytes),
                )
            )

    def connect_generator_gpu(
        self,
        layout: dict[str, TensorShard],
        *,
        sources: list[TrainerSource],
        rank: int,
        generator_index: int,
    ) -> None:
        """Plan reads whose GPU destinations are supplied during each load."""
        reads = _plan_reads(sources, layout, rank=rank, generator_index=generator_index)
        for source in sources:
            self._transport.connect(source.url)
        self._gpu_keys = frozenset(layout)
        self._gpu_reads = [
            (
                read,
                sources[read.source].baseline_buffer.slice(
                    read.byte_offset, read.nbytes
                ),
            )
            for read in reads
        ]

    def changed_keys(
        self,
        metadata_by_trainer_rank: tuple[ModelStateDictTransferMetadata, ...],
        *,
        full: bool,
    ) -> frozenset[str]:
        """Return the keys to load directly into the GPU."""
        assert self._gpu_reads is not None, "connect_generator_gpu must be called first"
        if full or self._sync_mode == "full":
            return self._gpu_keys
        return self._gpu_keys.intersection(
            key
            for metadata in metadata_by_trainer_rank
            for key in metadata.changed_keys
        )

    def read_into(self, state_dict: dict[str, torch.Tensor]) -> Iterable[str]:
        """Read the latest trainer values directly into GPU tensors."""
        assert self._gpu_reads is not None, "connect_generator_gpu must be called first"
        tensors = {key: _local(tensor) for key, tensor in state_dict.items()}
        if any(tensor.device.type != "cpu" for tensor in tensors.values()):
            torch.accelerator.current_stream().synchronize()
        memories = {}
        registered = {}
        try:
            for key, tensor in tensors.items():
                destination = (tensor.data_ptr(), tensor.nbytes, tensor.device)
                if destination not in registered:
                    registered[destination] = self._transport.register_memory(
                        tensor.view(-1)
                    )
                memories[key] = registered[destination]
            for read, remote in self._gpu_reads:
                if read.key not in memories:
                    continue
                local = memories[read.key].to_mutable_view(
                    read.local_byte_offset, read.nbytes
                )
                self._transport.read(local, remote, timeout=self._read_timeout_s)
        finally:
            for memory in registered.values():
                self._transport.unregister_memory(memory)
        return state_dict.keys()

    async def read(
        self,
        metadata_by_trainer_rank: tuple[ModelStateDictTransferMetadata, ...],
        *,
        full: bool,
    ) -> None:
        """Fill the prefetch buffers from the trainer's staging buffers.

        ``metadata_by_trainer_rank`` is returned by the same trainer ranks whose
        ``TrainerSource`` values were passed to ``connect_generator``.
        ``full`` is set by a generator whose prefetch buffers do not yet contain
        a complete model state dict.

        A timeout leaves the transfers in flight: the prefetch buffers are
        undefined and the job must restart.
        """
        assert self._reads is not None, "connect_generator must be called first"
        if not self._reads:
            return

        if full or self._sync_mode == "full":
            reads = [(local, baseline) for _, local, baseline, _ in self._reads]
        else:
            changed_ranges_by_source = [
                set(metadata.changed_ranges) for metadata in metadata_by_trainer_rank
            ]
            reads = [
                (local, current)
                for read, local, _, current in self._reads
                if TransferRange(read.byte_offset, read.nbytes)
                in changed_ranges_by_source[read.source]
            ]
        if not reads:
            return
        works = await asyncio.to_thread(
            lambda: [
                self._transport.read(local, remote, async_op=True)
                for local, remote in reads
            ]
        )
        await wait_all(works, timeout=self._read_timeout_s)
