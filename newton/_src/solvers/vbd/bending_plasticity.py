# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rate-independent, perfectly plastic dihedral bending for VBD shells."""

import warp as wp

from ...core.reset import reset_world_selected as _reset_world_selected


@wp.kernel
def prepare_bending_rest_angles(
    rest_angle: wp.array[float],
    plastic_angle: wp.array[float],
    effective_rest_angle: wp.array[float],
):
    edge = wp.tid()
    effective_rest_angle[edge] = rest_angle[edge] + plastic_angle[edge]


@wp.kernel
def update_bending_plasticity(
    positions: wp.array[wp.vec3],
    edges: wp.array2d[int],
    bending_properties: wp.array2d[float],
    yield_angle: wp.array[float],
    rest_angle: wp.array[float],
    plastic_angle_in: wp.array[float],
    plastic_angle_out: wp.array[float],
):
    edge = wp.tid()
    plastic_angle = plastic_angle_in[edge]
    plastic_angle_out[edge] = plastic_angle
    i, j = edges[edge, 0], edges[edge, 1]
    if i < 0 or j < 0 or bending_properties[edge, 0] <= 0.0:
        return

    k, l = edges[edge, 2], edges[edge, 3]
    x0, x1, x2, x3 = positions[i], positions[j], positions[k], positions[l]
    n0 = wp.cross(x2 - x0, x3 - x0)
    n1 = wp.cross(x3 - x1, x2 - x1)
    e = x3 - x2
    # Match the elastic bending kernel's degeneracy tolerance.
    if wp.length(n0) < 1.0e-6 or wp.length(n1) < 1.0e-6 or wp.length(e) < 1.0e-6:
        return
    n0, n1, e = wp.normalize(n0), wp.normalize(n1), wp.normalize(e)
    theta = wp.atan2(wp.dot(wp.cross(n0, n1), e), wp.dot(n0, n1))
    elastic_angle = theta - rest_angle[edge] - plastic_angle
    elastic_angle = wp.atan2(wp.sin(elastic_angle), wp.cos(elastic_angle))
    # Keep the yield-sized elastic curvature, which provides unloading springback.
    increment = elastic_angle - wp.clamp(elastic_angle, -yield_angle[edge], yield_angle[edge])
    if wp.abs(increment) > 1.0e-7:
        plastic_angle_out[edge] = plastic_angle + increment


@wp.kernel
def reset_bending_plasticity(
    world_mask: wp.array[wp.bool],
    reset_all: bool,
    world_count: int,
    particle_world: wp.array[int],
    edges: wp.array2d[int],
    plastic_angle: wp.array[float],
):
    edge = wp.tid()
    world = particle_world[edges[edge, 2]]
    if reset_all:
        plastic_angle[edge] = 0.0
    elif _reset_world_selected(world, world_mask, world_count):
        plastic_angle[edge] = 0.0
