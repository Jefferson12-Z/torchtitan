# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the rollout workers: where they spawn, and how the Rollouter dispatches to them."""

import asyncio
import functools
import gc
import os
import pathlib
import subprocess
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace

import cloudpickle
import pytest
from monarch.actor import Actor, endpoint, this_host
from monarch.job import ProcessJob

from torchtitan.observability import structured_logger as sl
from torchtitan.rl.actors.monarch import (
    provisioning,
    rollout_worker as rollout_worker_module,
    system as monarch_system,
)
from torchtitan.rl.actors.monarch.rollout_worker import RolloutWorkerActor
from torchtitan.rl.actors.monarch.system import MonarchActorSystem
from torchtitan.rl.actors.system import RolloutWorkersHandle
from torchtitan.rl.observability import metrics as m

from torchtitan.rl.rollout import rollouter as rollouter_module, RolloutGroup
from torchtitan.rl.rollout.rollouter import MessageEnvRolloutWorker, Rollouter


class _RunGroupEndpoint:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.error: Exception | None = None

    def call_one(self, **kwargs) -> asyncio.Future[RolloutGroup]:
        self.calls.append(kwargs)
        self.started.set()
        return asyncio.create_task(self._execute(kwargs))

    async def _execute(self, kwargs) -> RolloutGroup:
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        if self.error is not None:
            raise self.error
        return RolloutGroup(group_id=kwargs["group_id"], rollouts=[])


def test_rollout_worker_actor_builds_its_worker_and_inits_its_logger(
    monkeypatch,
) -> None:
    async def run() -> None:
        worker = object()
        worker_config = SimpleNamespace(build=lambda: worker)
        init_calls = []
        monkeypatch.setattr(
            rollout_worker_module, "current_rank", lambda: SimpleNamespace(rank=2)
        )
        actor = object.__new__(RolloutWorkerActor)

        RolloutWorkerActor.__init__(
            actor,
            worker_config=worker_config,
            worker_kwargs={},
            num_threads=1,
            # Bound like the controller's; the worker overrides source and rank.
            structured_logger_init_fn=functools.partial(
                lambda **kwargs: init_calls.append(kwargs), source="rl", rank=0
            ),
            first_worker_index=3,
        )

        assert actor._worker is worker
        # Rank 2 in the proc mesh of a host whose first worker is worker 3.
        assert init_calls == [{"source": "rl_rollout_worker", "rank": 5}]

    asyncio.run(run())


class _LoopBlockingWorker:
    """Holds its event loop for 0.3 s in setup and for `sample` seconds in each group, as a slow
    tokenizer build or render would."""

    async def setup_async(self, **kwargs) -> None:
        time.sleep(0.3)

    async def run_group(
        self, *, sample: float, group_id: int, **kwargs
    ) -> RolloutGroup:
        time.sleep(sample)
        # Yield, so the lag task wakes up late and records the delay before the group returns.
        await asyncio.sleep(0.1)
        return RolloutGroup(group_id=group_id, rollouts=[])


def test_rollout_worker_reports_its_event_loop_lag() -> None:
    """A training group gets the worker's event-loop lag since its previous training group, and
    its peak memory:
        setup holds the loop for 0.3 s                not counted
        training group                                lag < 250 ms
        validation group holds the loop for 0.3 s     no metrics
        training group                                lag >= 250 ms
    """

    async def run() -> list[RolloutGroup]:
        actor = object.__new__(RolloutWorkerActor)
        RolloutWorkerActor.__init__(
            actor,
            worker_config=SimpleNamespace(build=_LoopBlockingWorker),
            worker_kwargs={},
            num_threads=1,
            structured_logger_init_fn=None,
            first_worker_index=0,
        )
        # Let the loop run between the spawn and the setup call, as in a real actor. Then a lag task
        # started in `__init__` would record setup's 0.3 s hold, so the "not counted" row can fail.
        await asyncio.sleep(0)
        # Call each endpoint's own body, which `concurrent_endpoint` keeps as `__wrapped__`.
        # The wrapper needs a Monarch response port, and this test has none.
        setup_async = RolloutWorkerActor.setup_async._method.__wrapped__
        run_group = RolloutWorkerActor.run_group._method.__wrapped__
        await setup_async(
            actor, tokenizer_config=None, renderer_config=None, hf_assets_path=""
        )
        kwargs = dict(
            generate_fn=None, release_session_fn=None, group_size=1, sampling=None
        )
        groups = [
            await run_group(actor, sample=0.0, group_id=0, **kwargs),
            await run_group(actor, sample=0.3, group_id=-1, **kwargs),
            await run_group(actor, sample=0.0, group_id=1, **kwargs),
        ]
        actor._event_loop_lag.close()
        return groups

    first_training, validation, second_training = asyncio.run(run())
    assert validation.metrics == []
    first_metrics = m.MetricsProcessor._aggregate_metrics(first_training.metrics)
    second_metrics = m.MetricsProcessor._aggregate_metrics(second_training.metrics)
    lag_mean = next(
        metric.value
        for metric in second_training.metrics
        if metric.key == "rollout_worker/event_loop_lag_ms_per_s"
    )
    # The lag task checks every 0.05 s, so the 0.3 s hold reads at least 0.25 s.
    assert first_metrics["rollout_worker/event_loop_lag_ms/max"] < 250.0
    assert second_metrics["rollout_worker/event_loop_lag_ms/max"] >= 250.0
    # `Mean.value` is the total ms late. The 0.3 s hold adds at least 250 ms.
    assert lag_mean.value >= 250.0
    assert second_metrics["rollout_worker/peak_memory_gib/max"] > 0.0


class _CallEndpoint:
    """Records each `.call(...)`."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def call(self, **kwargs) -> None:
        self.calls.append(kwargs)


class _Worker:
    """One worker's handle: `run_group.call_one` reaches this worker only."""

    def __init__(self) -> None:
        self.run_group = _RunGroupEndpoint()


class _WorkerActorMesh:
    """One host's workers, as `spawn` returns them: `flatten("rank").slice(rank=r)` is worker r."""

    def __init__(self, num_workers: int) -> None:
        self.setup_async = _CallEndpoint()
        self.sync_log_step = _CallEndpoint()
        self.workers = [_Worker() for _ in range(num_workers)]
        self.stopped = False

    def __len__(self) -> int:
        return len(self.workers)

    def flatten(self, name: str) -> "_WorkerActorMesh":
        return self

    def slice(self, *, rank: int) -> _Worker:
        return self.workers[rank]

    async def stop(self) -> None:
        self.stopped = True


class _WorkerMesh:
    def __init__(self, actor_mesh: _WorkerActorMesh) -> None:
        self.actor_mesh = actor_mesh
        self.spawn_args = None
        self.stopped = False

    def spawn(self, *args, **kwargs) -> _WorkerActorMesh:
        self.spawn_args = (args, kwargs)
        return self.actor_mesh

    async def stop(self) -> None:
        await self.actor_mesh.stop()
        self.stopped = True


class _ControllerHost:
    def __init__(self, worker_mesh: _WorkerMesh) -> None:
        self.worker_mesh = worker_mesh
        self.spawn_kwargs = None

    def spawn_procs(self, **kwargs) -> _WorkerMesh:
        self.spawn_kwargs = kwargs
        return self.worker_mesh


def _rollouter_without_datasets() -> Rollouter:
    class WorkerConfig:
        _owner = MessageEnvRolloutWorker

    worker_config = WorkerConfig()
    rollouter = object.__new__(Rollouter)
    rollouter._config = SimpleNamespace(
        worker=worker_config,
        num_threads_per_worker=2,
        worker_placement="all_hosts",
    )
    rollouter._workers = None
    rollouter._actor_system = None
    rollouter._inflight_run_group_calls = set()
    return rollouter


async def _setup(
    rollouter: Rollouter,
    actor_mesh: _WorkerActorMesh,
    monkeypatch,
    actor_system: MonarchActorSystem | None = None,
) -> _WorkerMesh:
    worker_mesh = _WorkerMesh(actor_mesh)
    host = _ControllerHost(worker_mesh)
    monkeypatch.setattr(provisioning, "this_host", lambda: host)
    await rollouter.setup_async(
        actor_system=actor_system or MonarchActorSystem(MonarchActorSystem.Config()),
        num_workers=len(actor_mesh),
        tokenizer_config="tokenizer_config",
        renderer_config="renderer_config",
        hf_assets_path="hf_assets_path",
    )
    return worker_mesh


def test_setup_raises_when_one_host_would_run_too_many_rollout_workers(
    monkeypatch,
) -> None:
    actor_system = MonarchActorSystem(
        MonarchActorSystem.Config(max_rollout_workers_per_host=1)
    )
    # No launcher, so all 3 workers would run on the controller's host.
    with pytest.raises(ValueError, match="max_rollout_workers_per_host"):
        asyncio.run(
            _setup(
                _rollouter_without_datasets(),
                _WorkerActorMesh(num_workers=3),
                monkeypatch,
                actor_system=actor_system,
            )
        )


def test_setup_spawns_worker_pool_on_controller_host(monkeypatch) -> None:
    async def run() -> None:
        worker_mesh = _WorkerMesh(_WorkerActorMesh(num_workers=3))
        host = _ControllerHost(worker_mesh)
        monkeypatch.setattr(provisioning, "this_host", lambda: host)
        structured_logger_init_fn = object()
        monkeypatch.setattr(
            sl,
            "get_structured_logger_subprocess_init_fn",
            lambda: structured_logger_init_fn,
        )
        rollouter = _rollouter_without_datasets()
        rollouter._config.worker_placement = "controller_host_only"
        # A launcher's hosts: with "controller_host_only", the spawn must not use them.
        launcher_hosts = provisioning.HostMeshes(
            trainer=object(), generators=[object()], gpus_per_node=4
        )

        await rollouter.setup_async(
            actor_system=MonarchActorSystem(
                MonarchActorSystem.Config(), host_meshes=launcher_hosts
            ),
            num_workers=3,
            tokenizer_config="tokenizer_config",
            renderer_config="renderer_config",
            hf_assets_path="hf_assets_path",
        )

        assert host.spawn_kwargs == {"per_host": {"cpus": 3}}
        args, kwargs = worker_mesh.spawn_args
        assert args[0] == "rollout_worker"
        assert args[1] is RolloutWorkerActor
        # The actor gets the worker's own config, not the whole Rollouter config.
        assert kwargs == {
            "worker_config": rollouter._config.worker,
            "worker_kwargs": {},
            "num_threads": 2,
            "structured_logger_init_fn": structured_logger_init_fn,
            "first_worker_index": 0,
        }
        assert rollouter._workers.by_host == [worker_mesh.actor_mesh]
        assert rollouter._workers.by_worker == worker_mesh.actor_mesh.workers
        assert worker_mesh.actor_mesh.setup_async.calls == [
            {
                "tokenizer_config": "tokenizer_config",
                "renderer_config": "renderer_config",
                "hf_assets_path": "hf_assets_path",
                "vision_processor_config": None,
            }
        ]
        await rollouter.close()
        assert worker_mesh.actor_mesh.stopped
        assert worker_mesh.stopped

    asyncio.run(run())


def test_setup_and_sync_log_step_reach_the_workers_on_every_host() -> None:
    host_meshes = [_WorkerActorMesh(num_workers=2), _WorkerActorMesh(num_workers=1)]

    class _ActorSystem:
        async def spawn_rollout_workers(self, config, **kwargs) -> RolloutWorkersHandle:
            self.spawn_kwargs = kwargs
            return RolloutWorkersHandle(
                by_host=host_meshes,
                by_worker=[worker for mesh in host_meshes for worker in mesh.workers],
            )

    async def run() -> None:
        actor_system = _ActorSystem()
        rollouter = _rollouter_without_datasets()
        await rollouter.setup_async(
            actor_system=actor_system,
            num_workers=3,
            tokenizer_config="tokenizer_config",
            renderer_config="renderer_config",
            hf_assets_path="hf_assets_path",
        )
        await rollouter.sync_log_step(step=7)

        assert actor_system.spawn_kwargs["placement"] == "all_hosts"
        assert [len(mesh.setup_async.calls) for mesh in host_meshes] == [1, 1]
        assert [mesh.sync_log_step.calls for mesh in host_meshes] == [
            [{"step": 7}],
            [{"step": 7}],
        ]

    asyncio.run(run())


class _FakeHostMesh:
    """A launcher's host mesh of `num_hosts` hosts: host i of a mesh named "g" is "g{i}"."""

    def __init__(self, *, num_hosts: int, name: str) -> None:
        self.num_hosts = num_hosts
        self.name = name

    def __len__(self) -> int:
        return self.num_hosts

    def flatten(self, name: str) -> "_FakeHostMesh":
        return self

    def slice(self, *, hosts: int) -> str:
        return f"{self.name}{hosts}"


def _host_meshes(trainer, generators) -> provisioning.HostMeshes | None:
    """Mesh "g*2" is 2 hosts, g0 and g1. Generators that name the same mesh share it."""
    if trainer is None:
        return None
    meshes = {
        spec: _FakeHostMesh(num_hosts=int(spec.split("*")[1]), name=spec.split("*")[0])
        for spec in [trainer, *generators]
    }
    return provisioning.HostMeshes(
        trainer=meshes[trainer],
        generators=[meshes[spec] for spec in generators],
        gpus_per_node=4,
    )


@pytest.mark.parametrize(
    "placement, trainer, generators, controller, expected",
    [
        # No launcher: the job is this host alone, which runs the controller.
        ("all_hosts", None, [], None, ["this_host"]),
        ("controller_host_only", None, [], None, ["this_host"]),
        # One host for every role, and it runs the controller.
        ("all_hosts", "t*1", ["t*1"], "t0", ["t0"]),
        ("controller_host_only", "t*1", ["t*1"], "t0", ["this_host"]),
        # 4 hosts: trainer t0, t1 and two generators that share mesh g.
        ("all_hosts", "t*2", ["g*2", "g*2"], "t0", ["t1", "g0", "g1", "t0"]),
        ("all_but_controller_host", "t*2", ["g*2", "g*2"], "t0", ["t1", "g0", "g1"]),
        ("controller_host_only", "t*2", ["g*2", "g*2"], "t0", ["this_host"]),
        # The controller on a generator host, or outside the job.
        ("all_hosts", "t*2", ["g*2", "g*2"], "g1", ["t0", "t1", "g0", "g1"]),
        (
            "all_but_controller_host",
            "t*2",
            ["g*2", "g*2"],
            None,
            ["t0", "t1", "g0", "g1"],
        ),
    ],
)
def test_rollout_worker_hosts(
    monkeypatch, placement, trainer, generators, controller, expected
):
    async def controller_host_indices(hosts: list[str]) -> list[int]:
        return [i for i, host in enumerate(hosts) if host == controller]

    monkeypatch.setattr(provisioning, "this_host", lambda: "this_host")
    hosts = asyncio.run(
        provisioning.rollout_worker_hosts(
            host_meshes=_host_meshes(trainer, generators),
            placement=placement,
            controller_host_indices=controller_host_indices,
        )
    )
    assert hosts == expected


@pytest.mark.parametrize("trainer, generators", [(None, []), ("t*1", ["t*1"])])
def test_all_but_controller_host_raises_when_the_job_has_one_host(
    monkeypatch, trainer, generators
):
    async def controller_host_indices(hosts: list[str]) -> list[int]:
        return list(range(len(hosts)))

    monkeypatch.setattr(provisioning, "this_host", lambda: "this_host")
    with pytest.raises(ValueError, match="leaves no host"):
        asyncio.run(
            provisioning.rollout_worker_hosts(
                host_meshes=_host_meshes(trainer, generators),
                placement="all_but_controller_host",
                controller_host_indices=controller_host_indices,
            )
        )


def test_host_identity_imports_without_torch() -> None:
    """The host probe's module loads no torch, so each probe process starts fast (see `host_identity`'s docstring)."""
    script = "import sys, torchtitan.rl.actors.monarch.host_identity; assert 'torch' not in sys.modules"
    result = subprocess.run(
        [sys.executable, "-c", script], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def _simulated_host() -> str:
    """The host this process runs on, as `ProcessJob` names it, e.g. "trainer_1"."""
    # TODO: read a public Monarch value once one names ProcessJob hosts. This env var is private,
    #   and the hosts can't be told apart by hostname: they all run on this machine.
    # e.g. "anon-0 @ trainer_1"
    return os.environ["HYPERACTOR_PROCESS_NAME"].split("@")[-1].strip()


def _record_log_rank(marker_dir: str, *, source: str, rank: int) -> None:
    """Stands in for the structured logger's init: one file per worker, naming its log rank and
    its host."""
    pathlib.Path(marker_dir, f"{rank}-{_simulated_host()}").touch()


class _EchoActor(Actor):
    """Stands in for the generator router: an actor on the controller's host."""

    @endpoint
    async def times_ten(self, value: int) -> int:
        return value * 10


async def _ask_echo(echo, *, group_id: int) -> int:
    return await echo.times_ten.call_one(group_id)


class _SpreadWorker:
    """Runs a group by calling `generate_fn`, and leaves a file naming the group and its host."""

    def __init__(self, marker_dir: str) -> None:
        self.marker_dir = marker_dir

    async def setup_async(self, **kwargs) -> None:
        pass

    async def run_group(self, *, generate_fn, group_id, **kwargs) -> RolloutGroup:
        pathlib.Path(self.marker_dir, f"ran-{group_id}-{_simulated_host()}").touch()
        return RolloutGroup(group_id=await generate_fn(group_id=group_id), rollouts=[])


@dataclass
class _SpreadWorkerConfig:
    marker_dir: str

    def build(self) -> _SpreadWorker:
        return _SpreadWorker(self.marker_dir)


def test_rollout_workers_spread_over_the_hosts_of_a_job(tmp_path, monkeypatch):
    """A 4-host job, simulated on this machine, so every host shares the controller's kernel
    and counts as the controller's host (`controller_host_indices` runs for real):
        "all_hosts"                 5 workers go 2, 1, 1, 1 to the hosts in job order, log ranks 0-4
        "all_but_controller_host"   leaves no host: raises
    """
    monkeypatch.setattr(
        monarch_system.sl,
        "get_structured_logger_subprocess_init_fn",
        lambda: functools.partial(_record_log_rank, str(tmp_path)),
    )

    async def run() -> list[int]:
        # ======== Start a 4-host job simulated on this machine ========
        # One host mesh per role, as `Layout.host_meshes` builds them.
        job = ProcessJob({"trainer": 2, "generator_0": 1, "generator_1": 1})
        state = job.state(cached_path=None)
        echo_procs = this_host().spawn_procs(per_host={"cpus": 1})
        system = MonarchActorSystem(
            MonarchActorSystem.Config(),
            host_meshes=provisioning.HostMeshes(
                trainer=state.trainer,
                generators=[state.generator_0, state.generator_1],
                gpus_per_node=4,
            ),
        )
        # The worker processes can't import this test module, so ship the fakes by value.
        cloudpickle.register_pickle_by_value(sys.modules[__name__])
        try:
            # ======== Spawn 5 workers and run one group on each ========
            echo = echo_procs.spawn("echo", _EchoActor)
            workers = await system.spawn_rollout_workers(
                _SpreadWorkerConfig(str(tmp_path)),
                num_workers=5,
                num_threads=1,
                placement="all_hosts",
            )
            # The Rollouter's calls: `.call()` on each host's workers, `call_one` on one worker.
            await asyncio.gather(
                *(
                    host_workers.setup_async.call(
                        tokenizer_config=None, renderer_config=None, hf_assets_path=""
                    )
                    for host_workers in workers.by_host
                )
            )
            await asyncio.gather(
                *(
                    host_workers.sync_log_step.call(step=1)
                    for host_workers in workers.by_host
                )
            )
            groups = await asyncio.gather(
                *(
                    worker.run_group.call_one(
                        generate_fn=functools.partial(_ask_echo, echo),
                        release_session_fn=None,
                        sample=None,
                        group_id=group_id,
                        group_size=1,
                        sampling=None,
                    )
                    for group_id, worker in enumerate(workers.by_worker)
                )
            )
            await system.stop_rollout_workers()

            # ======== A placement that leaves no host raises ========
            with pytest.raises(ValueError, match="leaves no host"):
                await system.spawn_rollout_workers(
                    _SpreadWorkerConfig(str(tmp_path)),
                    num_workers=5,
                    num_threads=1,
                    placement="all_but_controller_host",
                )
            return [group.group_id for group in groups]
        finally:
            cloudpickle.unregister_pickle_by_value(sys.modules[__name__])
            await echo_procs.stop()
            job.kill()

    # Each group called the actor on this host: a cross-mesh call, the path a cross-host call takes.
    assert asyncio.run(run()) == [0, 10, 20, 30, 40]
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "0-trainer_0",
        "1-trainer_0",
        "2-trainer_1",
        "3-generator_0_0",
        "4-generator_1_0",
        "ran-0-trainer_0",
        "ran-1-trainer_0",
        "ran-2-trainer_1",
        "ran-3-generator_0_0",
        "ran-4-generator_1_0",
    ]


def test_a_group_goes_to_one_worker(monkeypatch) -> None:
    async def run() -> None:
        actor_mesh = _WorkerActorMesh(num_workers=3)
        rollouter = _rollouter_without_datasets()
        await _setup(rollouter, actor_mesh, monkeypatch)

        dispatch = asyncio.create_task(
            rollouter.run_group_rollouts(
                generate_fn="generate_fn",
                release_session_fn="release_session_fn",
                sample="sample",
                group_id=1,
                group_size=2,
                sampling="sampling",
            )
        )
        # `run_group_rollouts` sends the call before its first wait.
        await asyncio.sleep(0)
        calls = [worker.run_group.calls for worker in actor_mesh.workers]
        assert sorted(calls, key=len) == [
            [],
            [],
            [
                {
                    "generate_fn": "generate_fn",
                    "release_session_fn": "release_session_fn",
                    "sample": "sample",
                    "group_id": 1,
                    "group_size": 2,
                    "sampling": "sampling",
                }
            ],
        ]

        for worker in actor_mesh.workers:
            worker.run_group.release.set()
        assert (await dispatch).group_id == 1
        await rollouter.close()
        assert actor_mesh.stopped

    asyncio.run(run())


async def _dispatch_then_cancel_caller(rollouter: Rollouter, worker: _Worker) -> None:
    """Start one `run_group` call, then cancel its caller as `Controller.run` does at the end."""
    dispatch = asyncio.create_task(
        rollouter.run_group_rollouts(
            generate_fn="generate_fn",
            release_session_fn="release_session_fn",
            sample="sample",
            group_id=1,
            group_size=2,
            sampling="sampling",
        )
    )
    await worker.run_group.started.wait()
    dispatch.cancel()
    await asyncio.gather(dispatch, return_exceptions=True)


def test_close_waits_for_run_group_of_cancelled_caller(monkeypatch) -> None:
    async def run() -> None:
        loop_errors: list[str] = []
        asyncio.get_running_loop().set_exception_handler(
            lambda loop, context: loop_errors.append(context["message"])
        )
        actor_mesh = _WorkerActorMesh(num_workers=1)
        (worker,) = actor_mesh.workers
        rollouter = _rollouter_without_datasets()
        worker_mesh = await _setup(rollouter, actor_mesh, monkeypatch)
        await _dispatch_then_cancel_caller(rollouter, worker)

        close = asyncio.create_task(rollouter.close())
        await asyncio.sleep(0)
        # The worker still runs the group, so the mesh stays up.
        assert not worker_mesh.stopped

        # Each rollout catches its own errors, but scoring can still make run_group raise.
        worker.run_group.error = RuntimeError("scoring failed")
        worker.run_group.release.set()
        await close
        assert worker_mesh.stopped
        assert not worker.run_group.cancelled.is_set()
        # The cancelled caller never reads the error; the done callback must, or asyncio logs it here.
        gc.collect()
        assert loop_errors == []

    asyncio.run(run())


def test_close_stops_waiting_after_drain_timeout(monkeypatch) -> None:
    async def run() -> None:
        monkeypatch.setattr(rollouter_module, "_RUN_GROUP_DRAIN_TIMEOUT_S", 0.0)
        actor_mesh = _WorkerActorMesh(num_workers=1)
        rollouter = _rollouter_without_datasets()
        worker_mesh = await _setup(rollouter, actor_mesh, monkeypatch)
        await _dispatch_then_cancel_caller(rollouter, actor_mesh.workers[0])

        await rollouter.close()

        assert worker_mesh.stopped
        assert not rollouter._inflight_run_group_calls

    asyncio.run(run())
