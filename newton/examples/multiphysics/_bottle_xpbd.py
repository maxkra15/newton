# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the experimental SolverXPBD fluid core with the PET boundary.

The private per-iteration adapter retains the example's continuous wall
projection and missing-half-space density support. It is an integration
experiment, not a proposed public constraint-hook API. Bulk density,
cohesion, velocity filtering and grid ownership use Eric Heiden's solver.
"""

import math

import warp as wp

import newton
from newton.examples.multiphysics._bottle import BottleFluid, _shell_contacts, _spiky_gradient, _update_velocities
from newton.solvers import SolverXPBD


@wp.kernel
def _update_mouth_plane(points: wp.array[wp.vec3], indices: wp.array[int], plane: wp.array[wp.vec4]):
    count = indices.shape[0]
    center = wp.vec3(0.0)
    for j in range(count):
        center += points[indices[j]] / float(count)
    normal = wp.vec3(0.0)
    for j in range(count):
        normal += wp.cross(points[indices[j]] - center, points[indices[(j + 1) % count]] - center)
    normal = wp.normalize(normal)
    plane[0] = wp.vec4(normal[0], normal[1], normal[2], -wp.dot(normal, center))


@wp.kernel
def _mark_spills(positions: wp.array[wp.vec3], plane: wp.array[wp.vec4], radius: float, inside_bottle: wp.array[int]):
    i = wp.tid()
    mouth = plane[0]
    if wp.dot(wp.vec3(mouth[0], mouth[1], mouth[2]), positions[i]) + mouth[3] > radius:
        inside_bottle[i] = 0


@wp.kernel
def _wall_pressure(
    grid: wp.uint64,
    positions: wp.array[wp.vec3],
    mesh: wp.uint64,
    h: float,
    volume: float,
    mass: float,
    rest_density: float,
    epsilon: float,
    bulk_density: wp.array[float],
    bulk_lambda: wp.array[float],
    lambda_difference: wp.array[float],
    wall_displacement: wp.array[wp.vec3],
    density: wp.array[float],
):
    i = wp.tid()
    p = positions[i]
    density[i] = bulk_density[i] / rest_density
    lambda_difference[i] = 0.0
    wall_displacement[i] = wp.vec3(0.0)
    query = wp.mesh_query_point_no_sign(mesh, p, h)
    if not query.result:
        return
    closest = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
    normal = wp.mesh_eval_face_normal(mesh, query.face)
    offset = p - closest
    distance = wp.length(offset)
    tangent = offset - wp.dot(offset, normal) * normal
    if distance <= 1.0e-6 or wp.length(tangent) >= 0.1 * h:
        return
    s = wp.clamp(distance / h, 0.0, 1.0)
    primitive = s - 4.0 / 3.0 * wp.pow(s, 3.0) + 6.0 / 5.0 * wp.pow(s, 5.0)
    primitive += -4.0 / 7.0 * wp.pow(s, 7.0) + wp.pow(s, 9.0) / 9.0
    missing_density = 0.5 - 315.0 / 256.0 * primitive
    wall_gradient = -315.0 / (256.0 * h) * wp.pow(1.0 - s * s, 4.0) * offset / distance
    gradient = wall_gradient
    gradient_squared = float(0.0)
    for j in wp.hash_grid_query(grid, p, h):
        g = volume * _spiky_gradient(p - positions[j], h)
        gradient += g
        gradient_squared += wp.dot(g, g)
    gradient_squared += wp.dot(gradient, gradient)
    total_density = bulk_density[i] / rest_density + missing_density
    # Uniform masses factor out. Correct only the difference from the core
    # solve, retaining its cohesion and short-range separation contributions.
    full_lambda = -wp.max(total_density - 1.0, 0.0) / (gradient_squared + mass * epsilon)
    lambda_difference[i] = full_lambda - bulk_lambda[i] / mass
    wall_displacement[i] = full_lambda * wall_gradient
    density[i] = total_density
    bulk_density[i] = total_density * rest_density


@wp.kernel
def _add_wall_pressure(
    grid: wp.uint64,
    positions: wp.array[wp.vec3],
    h: float,
    volume: float,
    relaxation: float,
    lambda_difference: wp.array[float],
    wall_displacement: wp.array[wp.vec3],
    deltas: wp.array[wp.vec3],
):
    i = wp.tid()
    p = positions[i]
    correction = wall_displacement[i]
    for j in wp.hash_grid_query(grid, p, h):
        correction += volume * (lambda_difference[i] + lambda_difference[j]) * _spiky_gradient(p - positions[j], h)
    length = wp.length(correction)
    if length > 0.2 * h:
        correction *= 0.2 * h / length
    deltas[i] += relaxation * correction


@wp.kernel
def _table_drag(positions: wp.array[wp.vec3], floor: float, radius: float, dt: float, velocities: wp.array[wp.vec3]):
    i = wp.tid()
    if positions[i][2] < floor + radius + 0.001:
        velocity = velocities[i]
        velocity[0] /= 1.0 + 4.0 * dt
        velocity[1] /= 1.0 + 4.0 * dt
        velocity[2] = wp.max(velocity[2], 0.0)
        velocities[i] = velocity


class _BottleSolver(SolverXPBD):
    def __init__(self, model, boundary, **kwargs):
        self.boundary = boundary
        super().__init__(model, **kwargs)

    def _apply_particle_deltas(self, model, state_in, state_out, particle_deltas, dt):
        boundary = self.boundary
        positions = state_out.particle_q if self._particle_delta_counter == 0 else state_in.particle_q
        if boundary.wall_density:
            wp.launch(
                _wall_pressure,
                model.particle_count,
                inputs=[
                    model.particle_grid.id,
                    positions,
                    boundary.shell_mesh.id,
                    boundary.h,
                    boundary.volume,
                    boundary.mass,
                    self._fluid_rest_density_eff,
                    self._fluid_eps,
                    self._fluid_density,
                    self._fluid_lambda,
                    boundary.multiplier_deltas,
                    boundary.boundary_gradients,
                    boundary.densities,
                ],
                device=model.device,
            )
            wp.launch(
                _add_wall_pressure,
                model.particle_count,
                inputs=[
                    model.particle_grid.id,
                    positions,
                    boundary.h,
                    boundary.volume,
                    self.fluid_relaxation,
                    boundary.multiplier_deltas,
                    boundary.boundary_gradients,
                    particle_deltas,
                ],
                device=model.device,
            )
        positions, velocities = super()._apply_particle_deltas(model, state_in, state_out, particle_deltas, dt)
        wp.launch(
            _mark_spills,
            model.particle_count,
            inputs=[positions, boundary.mouth_plane, boundary.radius, boundary.inside_bottle],
            device=model.device,
        )
        wp.launch(
            _shell_contacts,
            model.particle_count,
            inputs=[
                boundary.shell_mesh.id,
                boundary.shell_indices,
                boundary.shell_previous,
                self.particle_q_init,
                positions,
                boundary.corrections,
                boundary.inside_bottle,
                math.inf,  # The moving mouth plane owns spill classification.
                boundary.radius,
                2.0 * boundary.h,
                boundary.floor,
            ],
            device=model.device,
        )
        wp.launch(
            _update_velocities,
            model.particle_count,
            inputs=[self.particle_q_init, positions, dt, velocities],
            device=model.device,
        )
        return positions, velocities


class BottleFluidXPBD(BottleFluid):
    """Run the shared XPBD fluid solver with a one-way deforming PET boundary."""

    def __init__(self, *args, cohesion=0.1, wall_density=True, **kwargs):
        if kwargs.get("bulk_modulus", math.inf) != math.inf:
            raise ValueError("The experimental XPBD backend has no finite bulk-modulus option")
        super().__init__(*args, **kwargs)
        self.wall_density = wall_density
        self.corrections.zero_()
        # The procedural bottle's top ring is ordered counterclockwise.
        mouth_indices = [
            i
            for i, point in enumerate(self.shell_points.numpy())
            if math.isclose(float(point[2]), self.mouth_height, abs_tol=1.0e-4)
        ]
        if len(mouth_indices) < 3:
            raise ValueError("The bottle must have an open top ring at mouth_height")
        self.mouth_indices = wp.array(mouth_indices, dtype=int, device=self.device)
        self.mouth_plane = wp.empty(1, dtype=wp.vec4, device=self.device)
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -981.0))
        builder.add_particles(
            pos=self.positions.numpy(),
            vel=self.velocities.numpy(),
            mass=[self.mass] * len(self.positions),
            radius=[self.radius] * len(self.positions),
            flags=[int(newton.ParticleFlags.ACTIVE | newton.ParticleFlags.FLUID)] * len(self.positions),
        )
        self.model = builder.finalize(device=self.device)
        self.model.particle_max_velocity = 10_000.0
        self.solver = _BottleSolver(
            self.model,
            self,
            iterations=self.iterations,
            fluid_rest_distance=self.spacing,
            fluid_smoothing_length=self.h,
            fluid_rest_density=1000.0 / 100.0**3,
            fluid_cohesion=cohesion,
            fluid_viscosity=0.1,
            fluid_max_neighbors=0,
        )
        self.grid = self.model.particle_grid
        self.state_in, self.state_out = self.model.state(), self.model.state()
        self.state_in.particle_q = self.positions
        self.state_in.particle_qd = self.velocities

    def step(self, shell_positions, dt):
        wp.copy(self.shell_previous, self.shell_points)
        wp.copy(self.shell_points, shell_positions)
        self.shell_mesh.refit()
        wp.launch(
            _update_mouth_plane,
            1,
            inputs=[self.shell_points, self.mouth_indices, self.mouth_plane],
            device=self.device,
        )
        self.solver.step(self.state_in, self.state_out, None, None, dt)
        wp.copy(self.positions, self.state_out.particle_q)
        wp.copy(self.velocities, self.state_out.particle_qd)
        wp.launch(
            _table_drag,
            len(self.positions),
            inputs=[self.positions, self.floor, self.radius, dt, self.velocities],
            device=self.device,
        )
