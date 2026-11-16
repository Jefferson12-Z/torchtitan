# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Which machine a process runs on.

`read_host_identities` starts one process per host only to run `HostIdentityActor`, so keep torch
out of this module's imports: with torch, each such process takes ~3.5 s to start instead of ~0.85 s.
"""

import asyncio
import socket
from dataclasses import dataclass

from monarch.actor import Actor, endpoint, HostMesh


@dataclass(frozen=True)
class HostIdentity:
    """Which machine a process runs on."""

    boot_id: str
    """A random id the kernel picks at each boot (`/proc/sys/kernel/random/boot_id`). It differs
    between machines. Every process on one kernel reads the same one, including containers that
    share it."""

    hostname: str
    """For logs. A container can have its own hostname."""


def this_host_identity() -> HostIdentity:
    with open("/proc/sys/kernel/random/boot_id") as file:
        return HostIdentity(boot_id=file.read().strip(), hostname=socket.gethostname())


class HostIdentityActor(Actor):
    @endpoint
    def host_identity(self) -> HostIdentity:
        return this_host_identity()


async def read_host_identities(hosts: list[HostMesh]) -> list[HostIdentity]:
    """Return each host's `HostIdentity`, in order.

    Monarch's `HostMesh` doesn't say which machine it is, so a short-lived process on each host
    reads it. The hosts are read in parallel.
    """
    procs = [host.spawn_procs() for host in hosts]
    actors = [proc.spawn("host_identity", HostIdentityActor) for proc in procs]
    identities = await asyncio.gather(
        *(actor.host_identity.call_one() for actor in actors)
    )
    await asyncio.gather(*(proc.stop() for proc in procs))
    return identities
