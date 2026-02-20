# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Minimal implicit-MPM demo: reset one world and clear warmstart for that world."""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverImplicitMPM


def _build_two_world_model(device: str):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    builder.world_count = 2
    SolverImplicitMPM.register_custom_attributes(builder)

    voxel_size = 0.1
    ppc = 2
    spacing = voxel_size / ppc
    dim = 6

    for world_id, center_x in enumerate((-0.8, 0.8)):
        builder.current_world = world_id
        builder.add_particle_grid(
            pos=wp.vec3(center_x - 0.15, 0.05, -0.15),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=dim,
            dim_y=dim,
            dim_z=dim,
            cell_x=spacing,
            cell_y=spacing,
            cell_z=spacing,
            mass=1.0,
            jitter=0.0,
            custom_attributes={"mpm:friction": 0.7},
        )
        builder.add_ground_plane()

    model = builder.finalize(device=device)
    return model, voxel_size, dim**3


def _world_centroid(state: newton.State, particles_per_world: int, world_id: int) -> np.ndarray:
    q = state.particle_q.numpy()
    start = world_id * particles_per_world
    end = start + particles_per_world
    return q[start:end].mean(axis=0)


def main():
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    model, voxel_size, particles_per_world = _build_two_world_model(device=device)
    state_0 = model.state()
    state_1 = model.state()

    options = SolverImplicitMPM.Options()
    options.grid_type = "dense"
    options.voxel_size = voxel_size
    solver = SolverImplicitMPM(model, options)

    initial_q = state_0.particle_q.numpy().copy()

    for _ in range(6):
        solver.step(state_0, state_1, control=None, contacts=None, dt=0.02)
        state_0, state_1 = state_1, state_0

    ws_before = float(
        np.linalg.norm(solver._last_step_data.ws_impulse_field.dof_values.numpy())
        + np.linalg.norm(solver._last_step_data.ws_stress_field.dof_values.numpy())
    )

    world1_centroid_before = _world_centroid(state_0, particles_per_world, world_id=1).copy()

    q = state_0.particle_q.numpy()
    qd = state_0.particle_qd.numpy()
    q[:particles_per_world] = initial_q[:particles_per_world]
    qd[:particles_per_world] = 0.0
    state_0.particle_q.assign(q)
    state_0.particle_qd.assign(qd)

    bounds_lo = np.min(initial_q[:particles_per_world], axis=0, keepdims=True)
    bounds_hi = np.max(initial_q[:particles_per_world], axis=0, keepdims=True)
    cleared_cells = solver.clear_warmstart_for_bounds(bounds_lo=bounds_lo, bounds_hi=bounds_hi, padding_cells=1)

    ws_after = float(
        np.linalg.norm(solver._last_step_data.ws_impulse_field.dof_values.numpy())
        + np.linalg.norm(solver._last_step_data.ws_stress_field.dof_values.numpy())
    )

    solver.step(state_0, state_1, control=None, contacts=None, dt=0.02)
    world1_centroid_after = _world_centroid(state_1, particles_per_world, world_id=1)
    world1_delta = np.linalg.norm(world1_centroid_after - world1_centroid_before)

    print(f"device={device}")
    print(f"cleared_cells={cleared_cells}")
    print(f"warmstart_norm_before={ws_before:.6e}")
    print(f"warmstart_norm_after={ws_after:.6e}")
    print(f"world1_centroid_delta_after_world0_reset={world1_delta:.6e}")


if __name__ == "__main__":
    main()
