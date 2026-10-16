# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run shared XPBD fluid pressure with the bottle's continuous PET contacts.

Boundary density is part of the shared pressure solve. The private projection
adapter supplies continuous contact with the moving thin wall; it remains an
integration experiment rather than a proposed public constraint-hook API.
"""

import math

import numpy as np
import warp as wp

import newton
from newton.examples.multiphysics._bottle import _project_shell_contact
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
def _project_boundary(
    mesh: wp.uint64,
    indices: wp.array[int],
    shell_previous: wp.array[wp.vec3],
    previous: wp.array[wp.vec3],
    mouth_plane: wp.array[wp.vec4],
    radius: float,
    search_radius: float,
    floor: float,
    dt: float,
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    inside_bottle: wp.array[int],
):
    i = wp.tid()
    mouth = mouth_plane[0]
    inside = inside_bottle[i]
    if wp.dot(wp.vec3(mouth[0], mouth[1], mouth[2]), positions[i]) + mouth[3] > radius:
        inside = 0
    p, inside = _project_shell_contact(
        mesh, indices, shell_previous, previous[i], positions[i], inside, wp.inf, radius, search_radius, floor
    )
    positions[i] = p
    velocities[i] = (p - previous[i]) / dt
    inside_bottle[i] = inside


@wp.kernel
def _density_ratios(density: wp.array[float], rest_density: float, ratios: wp.array[float]):
    i = wp.tid()
    ratios[i] = density[i] / rest_density


@wp.kernel
def _table_drag(positions: wp.array[wp.vec3], floor: float, radius: float, dt: float, velocities: wp.array[wp.vec3]):
    i = wp.tid()
    if positions[i][2] < floor + radius + 0.001:
        velocity = velocities[i]
        velocity[0] /= 1.0 + 4.0 * dt
        velocity[1] /= 1.0 + 4.0 * dt
        velocity[2] = wp.max(velocity[2], 0.0)
        velocities[i] = velocity


class BottleFluidXPBD:
    """Keep a shared XPBD fluid model coupled one-way to the VBD shell, in cm/kg/s."""

    class _BottleSolver(SolverXPBD):
        def __init__(self, model, boundary, **kwargs):
            self.boundary = boundary
            super().__init__(model, **kwargs)

        def _apply_particle_deltas(self, model, state_in, state_out, particle_deltas, dt):
            positions, velocities = super()._apply_particle_deltas(model, state_in, state_out, particle_deltas, dt)
            boundary = self.boundary
            wp.launch(
                _project_boundary,
                model.particle_count,
                inputs=[
                    boundary.shell_mesh.id,
                    boundary.shell_indices,
                    boundary.shell_previous,
                    self.particle_q_init,
                    boundary.mouth_plane,
                    boundary.radius,
                    2.0 * boundary.h,
                    boundary.floor,
                    dt,
                    positions,
                    velocities,
                    boundary.inside_bottle,
                ],
                device=model.device,
            )
            return positions, velocities

    def __init__(
        self, positions, spacing, shell_positions, indices, *, floor, mouth_height, iterations, device, cohesion=0.1
    ):
        self.device = device
        self.spacing = spacing
        self.h = 2.6 * spacing
        self.radius = 0.4 * spacing
        self.volume = spacing**3
        self.rest_density = 1000.0 / 100.0**3
        self.mass = self.rest_density * self.volume
        self.floor = floor
        self.shell_indices = wp.array(np.asarray(indices).flatten(), dtype=int, device=device)
        self.shell_points = wp.clone(shell_positions)
        self.shell_previous = wp.clone(shell_positions)
        self.shell_mesh = wp.Mesh(points=self.shell_points, indices=self.shell_indices)
        # The procedural bottle's top ring is ordered counterclockwise.
        mouth_indices = np.flatnonzero(np.isclose(self.shell_points.numpy()[:, 2], mouth_height, rtol=0.0, atol=1.0e-4))
        if len(mouth_indices) < 3:
            raise ValueError("The bottle must have an open top ring at mouth_height")
        self.mouth_indices = wp.array(mouth_indices, dtype=int, device=device)
        self.mouth_plane = wp.empty(1, dtype=wp.vec4, device=device)
        count = len(positions)
        self.inside_bottle = wp.ones(count, dtype=int, device=device)
        self.surface_radii = wp.full(
            count, (3.0 * self.volume / (4.0 * math.pi)) ** (1.0 / 3.0) * 0.01, dtype=float, device=device
        )
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -981.0))
        builder.add_particles(
            pos=positions,
            vel=np.zeros_like(positions),
            mass=[self.mass] * count,
            radius=[self.radius] * count,
            flags=[int(newton.ParticleFlags.ACTIVE | newton.ParticleFlags.FLUID)] * count,
        )
        self.model = builder.finalize(device=device)
        self.model.particle_max_velocity = 10_000.0
        self.solver = self._BottleSolver(
            self.model,
            self,
            iterations=iterations,
            fluid_rest_distance=spacing,
            fluid_smoothing_length=self.h,
            fluid_rest_density=self.rest_density,
            fluid_cohesion=cohesion,
            fluid_viscosity=0.1,
            # Damp the simultaneous bulk/wall Jacobi update to avoid overshoot.
            fluid_relaxation=0.8,
            fluid_max_neighbors=0,
            fluid_boundary_mesh=self.shell_mesh,
        )
        self.grid = self.model.particle_grid
        self.state_in, self.state_out = self.model.state(), self.model.state()
        self.positions = self.state_in.particle_q
        self.velocities = self.state_in.particle_qd
        self._density_ratio = wp.empty(count, dtype=float, device=device)

    @property
    def densities(self):
        """Compute normalized density on demand for diagnostics."""
        wp.launch(
            _density_ratios,
            len(self.positions),
            inputs=[self.solver._fluid_density, self.rest_density, self._density_ratio],
            device=self.device,
        )
        return self._density_ratio

    def step(self, shell_positions, dt):
        wp.copy(self.shell_previous, self.shell_points)
        wp.copy(self.shell_points, shell_positions)
        self.shell_mesh.refit()
        wp.launch(
            _update_mouth_plane, 1, inputs=[self.shell_points, self.mouth_indices, self.mouth_plane], device=self.device
        )
        self.solver.step(self.state_in, self.state_out, None, None, dt)
        # Stable public buffers keep the same ownership across captured substeps.
        wp.copy(self.positions, self.state_out.particle_q)
        wp.copy(self.velocities, self.state_out.particle_qd)
        wp.launch(
            _table_drag,
            len(self.positions),
            inputs=[self.positions, self.floor, self.radius, dt, self.velocities],
            device=self.device,
        )
