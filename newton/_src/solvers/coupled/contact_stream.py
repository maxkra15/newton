# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Public contact-buffer descriptors for coupled solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import warp as wp

from ...sim import Contacts


@dataclass(frozen=True)
class CoupledContactDiagnostics:
    """Host-readable counts and capacities for one coupled contact stream.

    Counts are raw device counter values and may exceed their stream capacities.
    Clamp a count to ``[0, capacity]`` separately when determining how many
    fixed-buffer entries are readable. Overflow reports the raw excess.
    """

    rigid_count: int
    rigid_capacity: int
    rigid_overflow: int
    soft_count: int
    soft_capacity: int
    soft_overflow: int


@dataclass(frozen=True)
class CoupledContactStream:
    """Zero-copy descriptor for a contact buffer used by a coupled solver.

    The descriptor does not own or copy ``contacts`` or either index map.
    These are live mutable buffers whose contents may change after enumeration.
    Coupled solvers conservatively report ``forces_available=False`` unless a
    standardized force representation is guaranteed for the complete stream.

    Attributes:
        name: Stable stream name within the coupled solver.
        kind: Origin of the contact buffer.
        contacts: Exact contact buffer used by the coupled solver.
        source: Source entry for a directional stream, if applicable.
        destination: Destination entry for a directional stream, if applicable.
        shape_local_to_parent: Optional map from stream-local shape ids to the
            parent model namespace.
        particle_local_to_parent: Optional map from stream-local particle ids
            to the parent model namespace.
        forces_available: Whether standardized solver-produced contact forces
            are guaranteed to be available in ``contacts``.
    """

    name: str
    kind: Literal["outer", "entry", "proxy", "admm"]
    contacts: Contacts
    source: str | None = None
    destination: str | None = None
    shape_local_to_parent: wp.array[int] | None = None
    particle_local_to_parent: wp.array[int] | None = None
    forces_available: bool = False

    def diagnostics(self) -> CoupledContactDiagnostics:
        """Read stream counts on the host and report fixed-buffer overflow.

        Calling this method synchronizes device-backed contact counters with
        the host. Merely enumerating contact streams performs no such read.
        """
        raw_counts = self.contacts.contact_counters.numpy()
        raw_rigid_count = int(raw_counts[0])
        raw_soft_count = int(raw_counts[1])
        rigid_capacity = int(self.contacts.rigid_contact_max)
        soft_capacity = int(self.contacts.soft_contact_max)
        return CoupledContactDiagnostics(
            rigid_count=raw_rigid_count,
            rigid_capacity=rigid_capacity,
            rigid_overflow=max(raw_rigid_count - rigid_capacity, 0),
            soft_count=raw_soft_count,
            soft_capacity=soft_capacity,
            soft_overflow=max(raw_soft_count - soft_capacity, 0),
        )


@dataclass(frozen=True)
class ProxyBodyFeedback:
    """Borrowed body-feedback buffers for one proxy direction.

    Tuple entries are aligned in proxy-configuration order. ID arrays use the
    parent-model namespace, and each dense ``forces`` array is indexed by its
    corresponding ``proxy_body_ids``. The arrays are live solver-owned buffers:
    consumers must treat them as read-only, and their contents may change after
    a solver step or reset. They remain valid only for the lifetime of the
    coupled solver that returned this descriptor.

    Attributes:
        source: Source coupled-solver entry.
        destination: Destination coupled-solver entry.
        source_body_ids: Parent-model source body IDs for each mapping.
        proxy_body_ids: Parent-model proxy body IDs for each mapping.
        forces: Dense parent-model spatial feedback arrays [N, N·m].
    """

    source: str
    destination: str
    source_body_ids: tuple[wp.array[int], ...]
    proxy_body_ids: tuple[wp.array[int], ...]
    forces: tuple[wp.array[wp.spatial_vector], ...]


@dataclass(frozen=True)
class ProxyParticleFeedback:
    """Borrowed particle-feedback buffers for one proxy direction.

    Tuple entries are aligned in proxy-configuration order. ID arrays use the
    parent-model namespace, and each dense ``forces`` array is indexed by its
    corresponding ``proxy_particle_ids``. The arrays are live solver-owned
    buffers: consumers must treat them as read-only, and their contents may
    change after a solver step or reset. They remain valid only for the lifetime
    of the coupled solver that returned this descriptor.

    Attributes:
        source: Source coupled-solver entry.
        destination: Destination coupled-solver entry.
        source_particle_ids: Parent-model source particle IDs for each mapping.
        proxy_particle_ids: Parent-model proxy particle IDs for each mapping.
        forces: Dense parent-model particle feedback arrays [N].
    """

    source: str
    destination: str
    source_particle_ids: tuple[wp.array[int], ...]
    proxy_particle_ids: tuple[wp.array[int], ...]
    forces: tuple[wp.array[wp.vec3], ...]


@dataclass(frozen=True)
class ProxyRelaxationDiagnostics:
    """Borrowed relaxation state for one proxy direction.

    Body and particle tuples are aligned with the corresponding feedback
    descriptor. A missing entity kind is represented by ``None``. Within a
    present kind, fixed-mode mappings have ``None`` entries for ``current``,
    ``has_previous``, and ``stats``; their configured scalar is authoritative.
    Aitken entries borrow the live solver-owned arrays and must be treated as
    read-only. Those arrays remain valid only for the lifetime of the coupled
    solver and may change after a step or reset.

    Aitken slot ``0`` represents global entities with world ID ``-1``. A
    regular world ID ``w`` uses slot ``w + 1``. Each stats row stores
    ``(dot(previous_residual, residual_delta), dot(residual_delta, residual_delta))``.
    Merely requesting this descriptor performs no device-to-host synchronization.

    Attributes:
        source: Source coupled-solver entry.
        destination: Destination coupled-solver entry.
        global_slot: Aitken slot reserved for global entities.
        world_slot_offset: Offset added to a world ID to obtain its slot.
        body_mode: Body relaxation modes, or ``None`` when absent.
        body_configured: Configured body relaxation values.
        body_min: Configured minimum body Aitken values.
        body_max: Configured maximum body Aitken values.
        body_world_ids: Per-mapping body world IDs.
        body_current: Current per-slot body Aitken values.
        body_has_previous: Per-slot body Aitken history-valid flags.
        body_stats: Per-slot body Aitken numerator/denominator statistics.
        particle_mode: Particle relaxation modes, or ``None`` when absent.
        particle_configured: Configured particle relaxation values.
        particle_min: Configured minimum particle Aitken values.
        particle_max: Configured maximum particle Aitken values.
        particle_world_ids: Per-mapping particle world IDs.
        particle_current: Current per-slot particle Aitken values.
        particle_has_previous: Per-slot particle Aitken history-valid flags.
        particle_stats: Per-slot particle Aitken numerator/denominator statistics.
    """

    source: str
    destination: str
    global_slot: int = 0
    world_slot_offset: int = 1
    body_mode: tuple[Literal["fixed", "aitken"], ...] | None = None
    body_configured: tuple[float, ...] | None = None
    body_min: tuple[float, ...] | None = None
    body_max: tuple[float, ...] | None = None
    body_world_ids: tuple[wp.array[int], ...] | None = None
    body_current: tuple[wp.array[float] | None, ...] | None = None
    body_has_previous: tuple[wp.array[int] | None, ...] | None = None
    body_stats: tuple[wp.array2d[float] | None, ...] | None = None
    particle_mode: tuple[Literal["fixed", "aitken"], ...] | None = None
    particle_configured: tuple[float, ...] | None = None
    particle_min: tuple[float, ...] | None = None
    particle_max: tuple[float, ...] | None = None
    particle_world_ids: tuple[wp.array[int], ...] | None = None
    particle_current: tuple[wp.array[float] | None, ...] | None = None
    particle_has_previous: tuple[wp.array[int] | None, ...] | None = None
    particle_stats: tuple[wp.array2d[float] | None, ...] | None = None


@dataclass(frozen=True)
class AdmmInterfaceDiagnostics:
    """Live convergence diagnostics for one ADMM owner pair.

    The arrays are stable one-element solver-owned buffers updated only by
    :meth:`SolverCoupledADMM.diagnostics`. Consumers must treat them as
    read-only. The primal residual is the consensus mismatch ``u - Jv``. The
    dual residual is the proximal/KKT fixed-point stationarity mismatch
    ``u - P_g(u - lambda / (rho W))`` for the row's local energy ``g``; it is
    not the temporal textbook ADMM residual based on consecutive iterates.

    Attributes:
        source: First owner in coupled-solver entry order.
        destination: Second owner in coupled-solver entry order.
        primal_residual_norm: L2 norm of all owner-pair primal residuals,
            shape ``[1]``.
        dual_residual_norm: L2 norm of all owner-pair proximal stationarity
            residuals, shape ``[1]``.
    """

    source: str
    destination: str
    primal_residual_norm: wp.array[float]
    dual_residual_norm: wp.array[float]


@dataclass(frozen=True)
class AdmmDiagnostics:
    """Live on-demand diagnostics for an ADMM coupled solver.

    Count arrays and interface residual arrays are stable one-element device
    buffers. Calling :meth:`SolverCoupledADMM.diagnostics` updates them without
    synchronizing with the host or modifying simulation state. Consumers must
    treat the arrays as read-only.

    Attributes:
        iterations: Configured ADMM iteration count.
        contact_count: Capacity-clamped active dynamic contact rows, shape
            ``[1]``.
        contact_count_max: Capacity-clamped dynamic contact high-water count,
            shape ``[1]``.
        contact_overflow: Sum of current raw active-count excess over group
            capacities, shape ``[1]``.
        interfaces: Diagnostics aggregated by canonical owner pair.
    """

    iterations: int
    contact_count: wp.array[int]
    contact_count_max: wp.array[int]
    contact_overflow: wp.array[int]
    interfaces: tuple[AdmmInterfaceDiagnostics, ...]
