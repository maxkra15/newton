# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Example-local GPU XPBD water with contact against a moving shell.

Use PBF density kernels with the accumulated multipliers of XPBD (2016),
equation 18: https://mmacklin.com/xpbd.pdf. Kernel definitions follow
*Position Based Fluids* (2013): https://mmacklin.com/pbf_sig_preprint.pdf.
Lengths use the bottle example's centimeters; mass and time use kg and s.
"""

import math

import numpy as np
import warp as wp


@wp.func
def _poly6(delta: wp.vec3, h: float):
    r2 = wp.dot(delta, delta)
    value = float(0.0)
    if r2 < h * h:
        a = h * h - r2
        value = 315.0 / (64.0 * wp.pi * wp.pow(h, 9.0)) * a * a * a
    return value


@wp.func
def _spiky_gradient(delta: wp.vec3, h: float):
    r = wp.length(delta)
    value = wp.vec3(0.0)
    if r > 1.0e-6 and r < h:
        a = h - r
        value = -45.0 / (wp.pi * wp.pow(h, 6.0)) * a * a * delta / r
    return value


@wp.kernel
def _predict(
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    dt: float,
    predicted: wp.array[wp.vec3],
):
    i = wp.tid()
    predicted[i] = positions[i] + dt * velocities[i] + dt * dt * wp.vec3(0.0, 0.0, -981.0)


@wp.kernel
def _density_multipliers(
    grid: wp.uint64,
    positions: wp.array[wp.vec3],
    shell_mesh: wp.uint64,
    h: float,
    volume: float,
    compliance: float,
    multipliers: wp.array[float],
    multiplier_deltas: wp.array[float],
    densities: wp.array[float],
    boundary_gradients: wp.array[wp.vec3],
):
    i = wp.tid()
    p = positions[i]
    density = float(0.0)
    grad_sum = wp.vec3(0.0)
    grad_sq = float(0.0)
    for j in wp.hash_grid_query(grid, p, h):
        delta = p - positions[j]
        density += volume * _poly6(delta, h)
        grad = volume * _spiky_gradient(delta, h)
        grad_sum += grad
        grad_sq += wp.dot(grad, grad)
    boundary_gradient = wp.vec3(0.0)
    query = wp.mesh_query_point_no_sign(shell_mesh, p, h)
    if query.result:
        closest = wp.mesh_eval_position(shell_mesh, query.face, query.u, query.v)
        normal = wp.mesh_eval_face_normal(shell_mesh, query.face)
        offset = p - closest
        distance = wp.length(offset)
        tangent = offset - wp.dot(offset, normal) * normal
        if distance > 1.0e-6 and wp.length(tangent) < 0.1 * h:
            s = wp.clamp(distance / h, 0.0, 1.0)
            # Integrate the Poly6 kernel over the wall's missing half-space.
            # Fixed virtual water completes boundary density without forcing
            # fluid next to a wall to compress to twice its rest density.
            primitive = s - 4.0 / 3.0 * wp.pow(s, 3.0) + 6.0 / 5.0 * wp.pow(s, 5.0)
            primitive += -4.0 / 7.0 * wp.pow(s, 7.0) + wp.pow(s, 9.0) / 9.0
            density += 0.5 - 315.0 / 256.0 * primitive
            boundary_gradient = -315.0 / (256.0 * h) * wp.pow(1.0 - s * s, 4.0) * offset / distance
            grad_sum += boundary_gradient
    grad_sq += wp.dot(grad_sum, grad_sum)
    # Clamp the density constraint at the free surface, as in PBF. Releasing
    # a hard constraint on underfilled samples would pull their neighbors in.
    constraint = wp.max(density - 1.0, 0.0)
    previous = multipliers[i]
    multiplier_delta = -(constraint + compliance * previous) / (grad_sq + compliance + 0.01 / (h * h))
    current = wp.min(previous + multiplier_delta, 0.0)
    multipliers[i] = current
    multiplier_deltas[i] = current - previous
    densities[i] = density
    boundary_gradients[i] = boundary_gradient


@wp.kernel
def _density_corrections(
    grid: wp.uint64,
    positions: wp.array[wp.vec3],
    multiplier_deltas: wp.array[float],
    boundary_gradients: wp.array[wp.vec3],
    h: float,
    volume: float,
    corrections: wp.array[wp.vec3],
):
    i = wp.tid()
    p = positions[i]
    correction = multiplier_deltas[i] * boundary_gradients[i]
    for j in wp.hash_grid_query(grid, p, h):
        if i != j:
            delta = p - positions[j]
            correction += volume * (multiplier_deltas[i] + multiplier_deltas[j]) * _spiky_gradient(delta, h)
    # Limit each Jacobi update to avoid overshooting neighboring constraints.
    length = wp.length(correction)
    if length > 0.2 * h:
        correction *= 0.2 * h / length
    corrections[i] = correction


@wp.kernel
def _apply_corrections(positions: wp.array[wp.vec3], corrections: wp.array[wp.vec3]):
    i = wp.tid()
    positions[i] += corrections[i]


@wp.kernel
def _shell_contacts(
    mesh: wp.uint64,
    shell_indices: wp.array[int],
    shell_previous: wp.array[wp.vec3],
    previous: wp.array[wp.vec3],
    positions: wp.array[wp.vec3],
    inside_bottle: wp.array[int],
    mouth_height: float,
    radius: float,
    search_radius: float,
    floor: float,
):
    i = wp.tid()
    p = positions[i]
    motion = p - previous[i]
    travel = wp.length(motion)
    if travel > 1.0e-6:
        direction = motion / travel
        ray = wp.mesh_query_ray(mesh, previous[i], direction, travel)
        if ray.result:
            normal_ray = wp.mesh_eval_face_normal(mesh, ray.face)
            side_ray = wp.where(wp.dot(direction, normal_ray) > 0.0, -1.0, 1.0)
            target = previous[i] + ray.t * direction + radius * side_ray * normal_ray
            p = target
    if p[2] > mouth_height + radius:
        inside_bottle[i] = 0
    query = wp.mesh_query_point_no_sign(mesh, p, search_radius)
    if query.result:
        a = shell_indices[3 * query.face]
        b = shell_indices[3 * query.face + 1]
        c = shell_indices[3 * query.face + 2]
        u = query.u
        v = query.v
        w = 1.0 - u - v
        closest = wp.mesh_eval_position(mesh, query.face, u, v)
        normal = wp.mesh_eval_face_normal(mesh, query.face)
        closest_previous = u * shell_previous[a] + v * shell_previous[b] + w * shell_previous[c]
        old_delta = previous[i] - closest_previous
        normal_previous = wp.normalize(
            wp.cross(shell_previous[b] - shell_previous[a], shell_previous[c] - shell_previous[a])
        )
        side = wp.where(wp.dot(old_delta, normal_previous) >= 0.0, 1.0, -1.0)
        # A local closest-face sign is ambiguous at concave bottle ribs.
        # Interior particles retain their side until they cross the open mouth.
        if inside_bottle[i] != 0:
            side = -1.0
        delta = p - closest
        signed_distance = wp.dot(delta, normal)
        tangent = delta - signed_distance * normal
        # Only the finite triangle blocks a crossing; the open mouth stays open.
        if wp.length(tangent) < radius and side * signed_distance < radius:
            correction = (radius - side * signed_distance) * side * normal
            p += correction
    p[2] = wp.max(p[2], floor + radius)
    positions[i] = p


@wp.kernel
def _update_velocities(
    previous: wp.array[wp.vec3],
    positions: wp.array[wp.vec3],
    dt: float,
    velocities: wp.array[wp.vec3],
):
    i = wp.tid()
    velocities[i] = (positions[i] - previous[i]) / dt


@wp.kernel
def _xsph(
    grid: wp.uint64,
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    h: float,
    volume: float,
    dt: float,
    floor: float,
    radius: float,
    output: wp.array[wp.vec3],
):
    i = wp.tid()
    velocity = velocities[i]
    change = wp.vec3(0.0)
    for j in wp.hash_grid_query(grid, positions[i], h):
        change += volume * _poly6(positions[i] - positions[j], h) * (velocities[j] - velocity)
    # XSPH is a numerical velocity filter, not a calibrated water viscosity.
    velocity += wp.min(0.12, 48.0 * dt) * change
    if positions[i][2] < floor + radius + 0.001:
        # An empirical wet-table drag arrests escaped droplets and puddles.
        velocity[0] /= 1.0 + 4.0 * dt
        velocity[1] /= 1.0 + 4.0 * dt
        velocity[2] = wp.max(velocity[2], 0.0)
    output[i] = velocity


class BottleFluid:
    """Keep fluid state separate from the VBD shell model.

    The shell prescribes the fluid boundary (one-way coupling); pressure does
    not feed back onto the shell. Lengths use cm, masses kg and time seconds.
    The bulk modulus is supplied in Pa; infinity gives zero compliance.
    """

    def __init__(
        self,
        positions,
        spacing,
        shell_positions,
        indices,
        *,
        floor,
        mouth_height,
        iterations,
        device,
        bulk_modulus=math.inf,
    ):
        if math.isnan(bulk_modulus) or bulk_modulus <= 0.0:
            raise ValueError("bulk_modulus must be positive or infinity [Pa]")
        self.device = device
        self.iterations = iterations
        self.spacing = spacing
        self.h = 2.6 * spacing
        self.volume = spacing**3
        self.mass = 1000.0 / 100.0**3 * self.volume
        # Uniform masses factor out of XPBD. Convert K [Pa] to kg/(cm s^2).
        self.compliance = self.mass / (0.01 * bulk_modulus * self.volume)
        self.radius = 0.4 * spacing
        self.floor = floor
        self.mouth_height = mouth_height
        self.inside_bottle = wp.ones(len(positions), dtype=int, device=device)
        self.positions = wp.array(positions, dtype=wp.vec3, device=device)
        self.predicted = wp.empty_like(self.positions)
        self.velocities = wp.zeros_like(self.positions)
        self.velocity_temp = wp.empty_like(self.positions)
        self.corrections = wp.empty_like(self.positions)
        self.multipliers = wp.zeros(len(positions), dtype=float, device=device)
        self.multiplier_deltas = wp.empty_like(self.multipliers)
        self.densities = wp.empty_like(self.multipliers)
        self.boundary_gradients = wp.empty_like(self.positions)
        self.grid = wp.HashGrid(64, 64, 128, device=device)
        self.grid.reserve(len(positions))
        self.shell_indices = wp.array(np.asarray(indices).flatten(), dtype=int, device=device)
        self.shell_points = wp.clone(shell_positions)
        self.shell_previous = wp.clone(shell_positions)
        self.shell_mesh = wp.Mesh(points=self.shell_points, indices=self.shell_indices)
        # Sphere volume agrees with the fluid's physical particle volume.
        self.surface_radii = wp.full(
            len(positions),
            (3.0 * self.volume / (4.0 * math.pi)) ** (1.0 / 3.0) * 0.01,
            dtype=float,
            device=device,
        )

    def step(self, shell_positions, dt):
        wp.copy(self.shell_previous, self.shell_points)
        wp.copy(self.shell_points, shell_positions)
        self.shell_mesh.refit()
        count = len(self.positions)
        self.multipliers.zero_()
        wp.launch(_predict, count, inputs=[self.positions, self.velocities, dt, self.predicted], device=self.device)
        for _ in range(self.iterations):
            self.grid.build(self.predicted, self.h)
            wp.launch(
                _density_multipliers,
                count,
                inputs=[
                    self.grid.id,
                    self.predicted,
                    self.shell_mesh.id,
                    self.h,
                    self.volume,
                    self.compliance / (dt * dt),
                    self.multipliers,
                    self.multiplier_deltas,
                    self.densities,
                    self.boundary_gradients,
                ],
                device=self.device,
            )
            wp.launch(
                _density_corrections,
                count,
                inputs=[
                    self.grid.id,
                    self.predicted,
                    self.multiplier_deltas,
                    self.boundary_gradients,
                    self.h,
                    self.volume,
                    self.corrections,
                ],
                device=self.device,
            )
            wp.launch(_apply_corrections, count, inputs=[self.predicted, self.corrections], device=self.device)
            wp.launch(
                _shell_contacts,
                count,
                inputs=[
                    self.shell_mesh.id,
                    self.shell_indices,
                    self.shell_previous,
                    self.positions,
                    self.predicted,
                    self.inside_bottle,
                    self.mouth_height,
                    self.radius,
                    2.0 * self.h,
                    self.floor,
                ],
                device=self.device,
            )
        wp.launch(
            _update_velocities,
            count,
            inputs=[self.positions, self.predicted, dt, self.velocities],
            device=self.device,
        )
        self.grid.build(self.predicted, self.h)
        wp.launch(
            _xsph,
            count,
            inputs=[
                self.grid.id,
                self.predicted,
                self.velocities,
                self.h,
                self.volume,
                dt,
                self.floor,
                self.radius,
                self.velocity_temp,
            ],
            device=self.device,
        )
        wp.copy(self.velocities, self.velocity_temp)
        wp.copy(self.positions, self.predicted)
