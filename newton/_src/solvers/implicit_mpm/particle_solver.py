# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Particle-only approximation sharing MPM's rigid collider geometry."""

from __future__ import annotations

import math

import warp as wp

from ...geometry import ParticleFlags
from ...sim import BodyFlags, Model, State
from ..solver import integrate_particles
from ..xpbd.kernels import compute_particle_contact_weights, solve_particle_particle_contacts
from .contact_solver_kernels import solve_coulomb_isotropic
from .rasterized_collisions import _ALL_COLLIDER_WORLDS, Collider, _swept_collider_contact, collision_sdf


@wp.func
def _accumulate_body_impulse(
    collider: Collider,
    collider_id: int,
    body_q_reference: wp.array[wp.transform],
    body_com_reference: wp.array[wp.vec3],
    position: wp.vec3,
    impulse: wp.vec3,
    body_impulse: wp.array[wp.spatial_vector],
):
    body = collider.collider_body_index[collider_id]
    if body >= 0:
        center = wp.transform_point(body_q_reference[body], body_com_reference[body])
        wp.atomic_add(body_impulse, body, wp.spatial_vector(impulse, wp.cross(position - center, impulse)))


@wp.kernel(enable_backward=False)
def _apply_particle_corrections(
    particle_flags: wp.array[wp.int32],
    dt: float,
    max_velocity: float,
    delta: wp.array[wp.vec3],
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
):
    i = wp.tid()
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0:
        return
    velocity_before = velocities[i]
    velocity = velocity_before + delta[i] / dt
    speed = wp.length(velocity)
    if speed > max_velocity:
        velocity *= max_velocity / speed
    positions[i] += (velocity - velocity_before) * dt
    velocities[i] = velocity


@wp.kernel(enable_backward=False)
def _sweep_particles(
    positions_prev: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    particle_mass: wp.array[float],
    particle_radius: wp.array[float],
    particle_environment: wp.array[int],
    collider: Collider,
    body_q_prev: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    body_q_reference: wp.array[wp.transform],
    body_com_reference: wp.array[wp.vec3],
    dt: float,
    sweep_body_motion: bool,
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    body_impulse: wp.array[wp.spatial_vector],
):
    i = wp.tid()
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0 or particle_mass[i] == 0.0:
        return
    environment = int(_ALL_COLLIDER_WORLDS)
    if particle_environment:
        environment = particle_environment[i]
    start = positions_prev[i]
    position = positions[i]
    velocity = velocities[i]
    if not sweep_body_motion and wp.length_sq(position - start) < 1.0e-20:
        return
    for contact_pass in range(4):
        cid, material, contact, normal, surface_velocity = _swept_collider_contact(
            start, position, environment, collider, body_q_prev, body_q, dt, sweep_body_motion and contact_pass == 0
        )
        if cid < 0:
            break
        margin = (
            particle_radius[i]
            + collider.material_thickness[material]
            - collider.material_projection_threshold[material]
        )
        friction = collider.material_friction[material]
        start = contact + wp.max(1.0e-6, margin) * normal
        position = start + solve_coulomb_isotropic(friction, normal, position - contact)
        velocity_new = surface_velocity + solve_coulomb_isotropic(friction, normal, velocity - surface_velocity)
        impulse = particle_mass[i] * (velocity - velocity_new)
        _accumulate_body_impulse(collider, cid, body_q_reference, body_com_reference, contact, impulse, body_impulse)
        velocity = velocity_new
        if contact_pass == 3:
            position = start
    positions[i] = position
    velocities[i] = velocity


@wp.kernel(enable_backward=False)
def _solve_mesh_contacts(
    particle_flags: wp.array[wp.int32],
    particle_mass: wp.array[float],
    particle_radius: wp.array[float],
    particle_environment: wp.array[int],
    collider: Collider,
    body_q_prev: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_q_reference: wp.array[wp.transform],
    body_com_reference: wp.array[wp.vec3],
    dt: float,
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    body_impulse: wp.array[wp.spatial_vector],
):
    i = wp.tid()
    if (particle_flags[i] & ParticleFlags.ACTIVE) == 0 or particle_mass[i] == 0.0:
        return
    environment = int(_ALL_COLLIDER_WORLDS)
    if particle_environment:
        environment = particle_environment[i]
    position = positions[i]
    velocity = velocities[i]
    sdf, normal, surface_velocity, cid, material = collision_sdf(
        position, environment, collider, body_q, body_qd, body_q_prev, dt
    )
    distance = sdf + collider.material_projection_threshold[material] - particle_radius[i]
    if cid >= 0 and distance <= 1.0e-6:
        contact = position - (sdf + collider.material_thickness[material]) * normal
        position -= wp.min(distance, 0.0) * normal
        velocity_new = surface_velocity + solve_coulomb_isotropic(
            collider.material_friction[material], normal, velocity - surface_velocity
        )
        impulse = particle_mass[i] * (velocity - velocity_new)
        _accumulate_body_impulse(collider, cid, body_q_reference, body_com_reference, contact, impulse, body_impulse)
        velocity = velocity_new
    positions[i] = position
    velocities[i] = velocity


@wp.kernel(enable_backward=False)
def _reset_body_impulses(
    body_world: wp.array[int],
    world_mask: wp.array[wp.bool],
    body_impulse: wp.array[wp.spatial_vector],
):
    body = wp.tid()
    world = body_world[body]
    mask_index = wp.where(world < 0, world_mask.shape[0] - 1, world)
    if world_mask[mask_index]:
        body_impulse[body] = wp.spatial_vector()


@wp.kernel(enable_backward=False)
def _harvest_proxy_wrenches(
    mapping: wp.array[int],
    body_flags: wp.array[wp.int32],
    body_impulse: wp.array[wp.spatial_vector],
    dt: float,
    out_body_f: wp.array[wp.spatial_vector],
):
    body = wp.tid()
    proxy = mapping[body]
    if proxy >= 0 and proxy < out_body_f.shape[0] and (body_flags[body] & int(BodyFlags.PROXY)) != 0:
        wp.atomic_add(out_body_f, proxy, body_impulse[body] / dt)


class _ParticleSolver:
    """Advance granular particles without constructing or updating an MPM grid.

    Rigid collider motion is prescribed. Velocity corrections produce equal
    and opposite body impulses for lagged proxy coupling; they do not integrate
    rigid bodies. Mesh overlap repair alone contributes no impulse;
    particle-pair corrections update velocities to approximate pressure support.
    The sweep guards particle centers and is not finite-radius continuous
    collision detection. Finite-radius contacts are enforced at endpoints.
    """

    def __init__(
        self,
        model: Model,
        *,
        iterations: int,
        relaxation: float = 0.9,
        friction: float | None = None,
        cohesion: float | None = None,
    ):
        self.model = model
        self.iterations = iterations
        self.relaxation = relaxation
        self.friction = friction
        self.cohesion = cohesion
        self._previous_positions = wp.empty(model.particle_count, dtype=wp.vec3, device=model.device)
        self._positions = wp.empty_like(self._previous_positions)
        self._velocities = wp.empty_like(self._previous_positions)
        self._deltas = wp.empty_like(self._previous_positions)
        self._contact_weight = wp.empty(model.particle_count, dtype=wp.vec2, device=model.device)
        self.body_impulse = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=model.device)

    def reset(self, world_mask: wp.array[wp.bool] | None = None) -> None:
        """Clear the most recent reaction impulses for all or selected worlds."""
        if world_mask is None:
            self.body_impulse.zero_()
        elif self.model.body_count:
            wp.launch(
                _reset_body_impulses,
                dim=self.model.body_count,
                inputs=[self.model.body_world, world_mask, self.body_impulse],
                device=self.model.device,
            )

    def _begin_particle_contacts(self) -> None:
        """Begin a native step; subclasses may reset step-local contact state."""

    def _compute_particle_contact_deltas(self, particle_flags, friction: float, cohesion: float, dt: float) -> None:
        model = self.model
        # Corrections can move particles across hash cells. Rebuild
        # before weighting and solving the same current contact graph.
        model.particle_grid.build(self._positions, radius=2.0 * model.particle_max_radius + cohesion)
        wp.launch(
            compute_particle_contact_weights,
            dim=model.particle_count,
            inputs=[
                model.particle_grid.id,
                self._positions,
                self._velocities,
                model.particle_inv_mass,
                model.particle_radius,
                particle_flags,
                model.particle_world,
                friction,
                cohesion,
                model.particle_max_radius,
                dt,
            ],
            outputs=[self._contact_weight],
            device=model.device,
        )
        self._deltas.zero_()
        wp.launch(
            solve_particle_particle_contacts,
            dim=model.particle_count,
            inputs=[
                model.particle_grid.id,
                self._positions,
                self._velocities,
                model.particle_inv_mass,
                model.particle_radius,
                particle_flags,
                model.particle_world,
                friction,
                cohesion,
                model.particle_max_radius,
                dt,
                self.relaxation,
                self._contact_weight,
            ],
            outputs=[self._deltas],
            device=model.device,
        )

    def _apply_particle_contact_deltas(self, particle_flags, dt: float) -> None:
        wp.launch(
            _apply_particle_corrections,
            dim=self.model.particle_count,
            inputs=[
                particle_flags,
                dt,
                self.model.particle_max_velocity,
                self._deltas,
                self._positions,
                self._velocities,
            ],
            device=self.model.device,
        )

    def step(
        self,
        state_in: State,
        state_out: State,
        dt: float,
        *,
        collider: Collider | None = None,
        particle_flags: wp.array[wp.int32] | None = None,
        particle_environment: wp.array[int] | None = None,
        body_q_prev: wp.array[wp.transform] | None = None,
        body_q: wp.array[wp.transform] | None = None,
        body_qd: wp.array[wp.spatial_vector] | None = None,
        body_q_reference: wp.array[wp.transform] | None = None,
    ) -> None:
        """Advance particles and retain collider reaction impulses for this step."""
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("Particle stepping requires a finite positive dt.")
        model = self.model
        particle_flags = model.particle_flags if particle_flags is None else particle_flags
        friction = model.particle_mu if self.friction is None else self.friction
        cohesion = model.particle_cohesion if self.cohesion is None else self.cohesion
        body_q_reference = body_q if body_q_reference is None else body_q_reference
        wp.copy(self._previous_positions, state_in.particle_q)
        wp.copy(self._velocities, state_in.particle_qd)
        self.body_impulse.zero_()
        wp.launch(
            integrate_particles,
            dim=model.particle_count,
            inputs=[
                state_in.particle_q,
                state_in.particle_qd,
                state_in.particle_f,
                model.particle_inv_mass,
                particle_flags,
                model.particle_world,
                model.gravity,
                dt,
                model.particle_max_velocity,
            ],
            outputs=[self._positions, self._velocities],
            device=model.device,
        )
        has_colliders = collider is not None and collider.collider_mesh.shape[0] > 0
        if has_colliders and collider.query_max_dist < model.particle_max_radius:
            # A resized particle can contact the mesh outside the MPM grid's query band.
            particle_collider = Collider()
            particle_collider.assign(collider)
            particle_collider.query_max_dist = model.particle_max_radius
            collider = particle_collider
        if has_colliders:
            wp.launch(
                _sweep_particles,
                dim=model.particle_count,
                inputs=[
                    self._previous_positions,
                    particle_flags,
                    model.particle_mass,
                    model.particle_radius,
                    particle_environment,
                    collider,
                    body_q_prev,
                    body_q,
                    body_q_reference,
                    model.body_com,
                    dt,
                    True,
                    self._positions,
                    self._velocities,
                    self.body_impulse,
                ],
                device=model.device,
            )
        self._begin_particle_contacts()
        for _ in range(self.iterations):
            if model.particle_count > 1 and model.particle_grid is not None and model.particle_max_radius > 0.0:
                self._compute_particle_contact_deltas(particle_flags, friction, cohesion, dt)
                if has_colliders:
                    wp.copy(self._previous_positions, self._positions)
                self._apply_particle_contact_deltas(particle_flags, dt)
                if has_colliders:
                    # Constraint displacements can cross walls even after a safe ballistic sweep.
                    wp.launch(
                        _sweep_particles,
                        dim=model.particle_count,
                        inputs=[
                            self._previous_positions,
                            particle_flags,
                            model.particle_mass,
                            model.particle_radius,
                            particle_environment,
                            collider,
                            body_q_prev,
                            body_q,
                            body_q_reference,
                            model.body_com,
                            dt,
                            False,
                            self._positions,
                            self._velocities,
                            self.body_impulse,
                        ],
                        device=model.device,
                    )
            if has_colliders:
                wp.launch(
                    _solve_mesh_contacts,
                    dim=model.particle_count,
                    inputs=[
                        particle_flags,
                        model.particle_mass,
                        model.particle_radius,
                        particle_environment,
                        collider,
                        body_q_prev,
                        body_q,
                        body_qd,
                        body_q_reference,
                        model.body_com,
                        dt,
                        self._positions,
                        self._velocities,
                        self.body_impulse,
                    ],
                    device=model.device,
                )
        wp.copy(state_out.particle_q, self._positions)
        wp.copy(state_out.particle_qd, self._velocities)
        for name, source in vars(state_in.mpm).items():
            if isinstance(source, wp.array):
                destination = getattr(state_out.mpm, name)
                if destination.ptr != source.ptr:
                    wp.copy(destination, source)
        state_out.mpm.particle_qd_grad.zero_()

    def harvest_proxy_wrenches(
        self,
        body_local_to_proxy_global: wp.array[int],
        out_body_f: wp.array[wp.spatial_vector],
        dt: float,
    ) -> None:
        """Write the current step's reaction wrench in world coordinates at each COM."""
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("Particle proxy wrench harvesting requires a finite positive dt.")
        out_body_f.zero_()
        wp.launch(
            _harvest_proxy_wrenches,
            dim=body_local_to_proxy_global.shape[0],
            inputs=[body_local_to_proxy_global, self.model.body_flags, self.body_impulse, dt, out_body_f],
            device=self.model.device,
        )
