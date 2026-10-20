# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Private, opt-in compliant particle contact experiment.

This is deliberately not exported. Mesh sweeps and rigid proxy coupling remain
those of the particle approximation. Only particle-particle contacts change.
"""

from __future__ import annotations

import math

import warp as wp

from ...geometry import ParticleFlags
from ..xpbd.kernels import particle_contact_eigenvalue_bound, particle_particle_contact_enabled
from .particle_solver import _ParticleSolver


@wp.kernel(enable_backward=False)
def _clear_contacts(
    contact_ids: wp.array2d[int],
    contact_count: wp.array[int],
    normal_lambda: wp.array2d[float],
    tangent_lambda: wp.array2d[wp.vec3],
):
    i = wp.tid()
    # Unused multipliers start at zero and have never been read or written.
    # Clearing only previously occupied slots makes capacity a storage cost.
    for slot in range(wp.min(contact_count[i], contact_ids.shape[1])):
        contact_ids[i, slot] = -1
        normal_lambda[i, slot] = 0.0
        tangent_lambda[i, slot] = wp.vec3(0.0)
    contact_count[i] = 0


@wp.kernel(enable_backward=False)
def _discover_contacts(
    grid: wp.uint64,
    positions: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    radius: wp.array[float],
    flags: wp.array[int],
    world: wp.array[int],
    max_radius: float,
    contact_ids: wp.array2d[int],
    contact_count: wp.array[int],
    overflow: wp.array[int],
    max_count: wp.array[int],
):
    i = wp.hash_grid_point_id(grid, wp.tid())
    if i < 0 or (flags[i] & ParticleFlags.ACTIVE) == 0:
        return
    count = contact_count[i]
    required = count
    query = wp.hash_grid_query(grid, positions[i], radius[i] + max_radius)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if j > i and particle_particle_contact_enabled(i, j, flags, inv_mass, world):
            # Proxy pairs and proxy/pinned pairs have no symmetric physical solve.
            if particle_particle_contact_enabled(j, i, flags, inv_mass, world):
                distance = wp.length(positions[i] - positions[j])
                if distance > 0.0 and distance <= radius[i] + radius[j] and inv_mass[i] + inv_mass[j] > 0.0:
                    found = bool(False)
                    for slot in range(wp.min(count, contact_ids.shape[1])):
                        if contact_ids[i, slot] == j:
                            found = True
                    if not found:
                        required += 1
                        if count < contact_ids.shape[1]:
                            contact_ids[i, count] = j
                            count += 1
                        else:
                            wp.atomic_add(overflow, 0, 1)
    contact_count[i] = count
    wp.atomic_max(max_count, 0, required)


@wp.kernel(enable_backward=False)
def _assemble_contact_weights(
    positions: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    contact_ids: wp.array2d[int],
    contact_count: wp.array[int],
    normal_matrix: wp.array[wp.mat33],
    tangent_degree: wp.array[float],
):
    i = wp.tid()
    for slot in range(wp.min(contact_count[i], contact_ids.shape[1])):
        j = contact_ids[i, slot]
        n = positions[i] - positions[j]
        distance = wp.length(n)
        if distance > 0.0:
            n /= distance
            denom = inv_mass[i] + inv_mass[j]
            matrix = wp.outer(n, n) / denom
            wp.atomic_add(normal_matrix, i, inv_mass[i] * matrix)
            wp.atomic_add(normal_matrix, j, inv_mass[j] * matrix)
            wp.atomic_add(tangent_degree, i, inv_mass[i] / denom)
            wp.atomic_add(tangent_degree, j, inv_mass[j] / denom)


@wp.kernel(enable_backward=False)
def _finalize_contact_weights(
    normal_matrix: wp.array[wp.mat33],
    tangent_degree: wp.array[float],
    weights: wp.array[wp.vec2],
):
    i = wp.tid()
    normal = wp.max(1.0, particle_contact_eigenvalue_bound(normal_matrix[i]))
    weights[i] = wp.vec2(normal, wp.max(normal, tangent_degree[i]))


@wp.kernel(enable_backward=False)
def _solve_contacts(
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    radius: wp.array[float],
    weights: wp.array[wp.vec2],
    contact_ids: wp.array2d[int],
    contact_count: wp.array[int],
    compliance: float,
    friction: float,
    relaxation: float,
    dt: float,
    normal_lambda: wp.array2d[float],
    tangent_lambda: wp.array2d[wp.vec3],
    deltas: wp.array[wp.vec3],
):
    i = wp.tid()
    alpha = compliance / (dt * dt)
    for slot in range(wp.min(contact_count[i], contact_ids.shape[1])):
        j = contact_ids[i, slot]
        n = positions[i] - positions[j]
        distance = wp.length(n)
        if distance > 0.0:
            n /= distance
            denom = inv_mass[i] + inv_mass[j]
            gamma_n = relaxation / wp.max(weights[i][0], weights[j][0])
            old_n = normal_lambda[i, slot]
            constraint = distance - radius[i] - radius[j]
            increment_n = -(constraint + alpha * old_n) / (denom + alpha)
            new_n = wp.max(0.0, old_n + gamma_n * increment_n)
            delta_n = new_n - old_n
            normal_lambda[i, slot] = new_n

            relative_velocity = velocities[i] - velocities[j]
            tangent_velocity = relative_velocity - wp.dot(n, relative_velocity) * n
            gamma_t = relaxation / wp.max(weights[i][1], weights[j][1])
            old_t = tangent_lambda[i, slot]
            candidate_t = old_t - gamma_t * dt * tangent_velocity / denom
            # Apply the complete vector difference, including tangent transport
            # and release when the accumulated normal Coulomb budget shrinks.
            new_t = candidate_t - wp.dot(n, candidate_t) * n
            budget = friction * new_n
            magnitude = wp.length(new_t)
            if magnitude > budget:
                new_t *= budget / magnitude
            tangent_lambda[i, slot] = new_t
            delta = n * delta_n + new_t - old_t
            wp.atomic_add(deltas, i, inv_mass[i] * delta)
            wp.atomic_sub(deltas, j, inv_mass[j] * delta)


@wp.kernel(enable_backward=False)
def _apply_contact_corrections(
    flags: wp.array[int],
    dt: float,
    max_velocity: float,
    delta: wp.array[wp.vec3],
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    velocity_cap_count: wp.array[int],
):
    i = wp.tid()
    if (flags[i] & ParticleFlags.ACTIVE) == 0:
        return
    velocity_before = velocities[i]
    velocity = velocity_before + delta[i] / dt
    speed = wp.length(velocity)
    if speed > max_velocity:
        # This preserves the existing safety cap, but invalidates an XPBD
        # comparison: the applied position delta no longer matches stored lambda.
        wp.atomic_add(velocity_cap_count, 0, 1)
        velocity *= max_velocity / speed
    positions[i] += (velocity - velocity_before) * dt
    velocities[i] = velocity


class _XPBDParticleSolver(_ParticleSolver):
    """Try accumulated compliant pair contacts without changing default behavior.

    Compliance is in m/N. A physical contact force is approximately lambda/dt².
    The normal compliance is zero for hard contacts; a positive value purposely
    softens contact. Tangential compliance is zero and total tangential lambda
    is bounded by friction times the accumulated compressive normal lambda.
    Multipliers reset at each native step, not at each solver iteration. Slots
    retain canonical particle IDs throughout that step, including separated
    contacts whose previous correction must be released. There is no cross-step
    warm start. Rotating normal gradients retain XPBD's quasi-Newton treatment.

    This prototype supports cohesion=0 only. Call check_contact_capacity after
    every captured policy step. Overflow or an active correction velocity cap
    invalidates the experiment and raises instead of silently accepting it.
    """

    def __init__(
        self,
        model,
        *,
        iterations: int,
        compliance: float,
        contact_capacity: int = 64,
        relaxation: float = 0.9,
        friction: float | None = None,
        cohesion: float | None = None,
    ):
        if not math.isfinite(compliance) or compliance < 0.0:
            raise ValueError("Contact compliance must be finite and nonnegative.")
        if not isinstance(contact_capacity, int) or contact_capacity < 1:
            raise ValueError("Contact capacity must be a positive integer.")
        if (model.particle_cohesion if cohesion is None else cohesion) != 0.0:
            raise ValueError("The private XPBD contact experiment supports cohesion=0 only.")
        super().__init__(model, iterations=iterations, relaxation=relaxation, friction=friction, cohesion=cohesion)
        self.compliance = compliance
        self.contact_capacity = contact_capacity
        shape = (model.particle_count, contact_capacity)
        self.contact_ids = wp.empty(shape, dtype=int, device=model.device)
        self.contact_ids.fill_(-1)
        self.contact_count = wp.zeros(model.particle_count, dtype=int, device=model.device)
        self.normal_lambda = wp.zeros(shape, dtype=float, device=model.device)
        self.tangent_lambda = wp.zeros(shape, dtype=wp.vec3, device=model.device)
        self.contact_overflow = wp.zeros(1, dtype=int, device=model.device)
        self.contact_max_count = wp.zeros(1, dtype=int, device=model.device)
        self.contact_velocity_cap_count = wp.zeros(1, dtype=int, device=model.device)
        self._normal_matrix = wp.empty(model.particle_count, dtype=wp.mat33, device=model.device)
        self._tangent_degree = wp.empty(model.particle_count, dtype=float, device=model.device)

    def _begin_particle_contacts(self) -> None:
        wp.launch(
            _clear_contacts,
            dim=self.model.particle_count,
            inputs=[self.contact_ids, self.contact_count, self.normal_lambda, self.tangent_lambda],
            device=self.model.device,
        )

    def check_contact_capacity(self) -> dict[str, int]:
        """Validate cumulative diagnostics outside a captured loop; never clear them."""
        diagnostics = {
            "overflow": int(self.contact_overflow.numpy()[0]),
            "max_count": int(self.contact_max_count.numpy()[0]),
            "velocity_cap_count": int(self.contact_velocity_cap_count.numpy()[0]),
        }
        if diagnostics["overflow"]:
            raise RuntimeError(f"XPBD contact capacity exceeded: {diagnostics}")
        if diagnostics["velocity_cap_count"]:
            raise RuntimeError(f"XPBD contact correction hit the velocity cap: {diagnostics}")
        return diagnostics

    def _compute_particle_contact_deltas(self, particle_flags, friction: float, cohesion: float, dt: float) -> None:
        if cohesion != 0.0:
            raise ValueError("The private XPBD contact experiment supports cohesion=0 only.")
        model = self.model
        model.particle_grid.build(self._positions, radius=2.0 * model.particle_max_radius)
        wp.launch(
            _discover_contacts,
            dim=model.particle_count,
            inputs=[
                model.particle_grid.id,
                self._positions,
                model.particle_inv_mass,
                model.particle_radius,
                particle_flags,
                model.particle_world,
                model.particle_max_radius,
                self.contact_ids,
                self.contact_count,
                self.contact_overflow,
                self.contact_max_count,
            ],
            device=model.device,
        )
        self._normal_matrix.zero_()
        self._tangent_degree.zero_()
        wp.launch(
            _assemble_contact_weights,
            dim=model.particle_count,
            inputs=[self._positions, model.particle_inv_mass, self.contact_ids, self.contact_count],
            outputs=[self._normal_matrix, self._tangent_degree],
            device=model.device,
        )
        wp.launch(
            _finalize_contact_weights,
            dim=model.particle_count,
            inputs=[self._normal_matrix, self._tangent_degree],
            outputs=[self._contact_weight],
            device=model.device,
        )
        self._deltas.zero_()
        wp.launch(
            _solve_contacts,
            dim=model.particle_count,
            inputs=[
                self._positions,
                self._velocities,
                model.particle_inv_mass,
                model.particle_radius,
                self._contact_weight,
                self.contact_ids,
                self.contact_count,
                self.compliance,
                friction,
                self.relaxation,
                dt,
                self.normal_lambda,
                self.tangent_lambda,
            ],
            outputs=[self._deltas],
            device=model.device,
        )

    def _apply_particle_contact_deltas(self, particle_flags, dt: float) -> None:
        wp.launch(
            _apply_contact_corrections,
            dim=self.model.particle_count,
            inputs=[
                particle_flags,
                dt,
                self.model.particle_max_velocity,
                self._deltas,
                self._positions,
                self._velocities,
                self.contact_velocity_cap_count,
            ],
            device=self.model.device,
        )
