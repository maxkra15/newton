# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import warp as wp
from asv_runner.benchmarks.mark import SkipNotImplemented, skip_benchmark_if

wp.config.enable_backward = False
wp.config.log_level = wp.LOG_WARNING

import newton
import newton._src.solvers.particle_grid as particle_grid_module

PARTICLES_PER_WORLD = 64
PARTICLE_SPACING = 0.05
PARTICLE_RADIUS = 0.025
PARTICLE_MASS = 0.125


def _make_model(world_count):
    world = newton.ModelBuilder(gravity=0.0)
    world.add_particle_grid(
        pos=wp.vec3(),
        rot=wp.quat_identity(),
        vel=wp.vec3(),
        dim_x=4,
        dim_y=4,
        dim_z=4,
        cell_x=PARTICLE_SPACING,
        cell_y=PARTICLE_SPACING,
        cell_z=PARTICLE_SPACING,
        mass=PARTICLE_MASS,
        jitter=0.0,
        radius_mean=PARTICLE_RADIUS,
        radius_std=0.0,
        flags=newton.ParticleFlags.ACTIVE | newton.ParticleFlags.FLUID,
    )
    scene = newton.ModelBuilder(gravity=0.0)
    scene.replicate(world, world_count)
    return scene.finalize(device="cuda:0")


class TimeFluidMultiworld:
    params = (["xpbd", "sph"], [1, 8, 64, 256], ["filtered", "grouped"])
    param_names = ["solver", "world_count", "query_mode"]
    repeat = 5
    number = 20

    def setup(self, solver, world_count, query_mode):
        if wp.get_cuda_device_count() == 0 or not particle_grid_module._HASH_GRID_GROUPING_SUPPORTED:
            raise SkipNotImplemented

        self.model = _make_model(world_count)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        if solver == "xpbd":
            self.solver = newton.solvers.SolverXPBD(
                self.model,
                iterations=2,
                fluid_rest_distance=PARTICLE_SPACING,
                fluid_cohesion=0.0,
            )
        else:
            self.solver = newton.solvers.SolverSPH(
                self.model,
                smoothing_length=0.09,
                rest_density=1000.0,
                gas_constant=50.0,
            )

        expected_eligible = world_count > 1
        if self.solver._particle_grid_grouped != expected_eligible:
            raise AssertionError(
                f"{solver} grouped eligibility is {self.solver._particle_grid_grouped}, expected {expected_eligible}"
            )
        if query_mode == "filtered":
            self.solver._particle_grid_grouped = False
        expected_mode = query_mode == "grouped" and expected_eligible
        if self.solver._particle_grid_grouped != expected_mode:
            raise AssertionError(
                f"{solver} query mode is {self.solver._particle_grid_grouped}, expected {expected_mode}"
            )

        worlds = self.model.particle_world.numpy().reshape(world_count, PARTICLES_PER_WORLD)
        expected_worlds = np.broadcast_to(
            np.arange(world_count, dtype=np.int32)[:, None],
            worlds.shape,
        )
        np.testing.assert_array_equal(worlds, expected_worlds)

        self.solver.step(self.state_0, self.state_1, None, None, 1.0 / 240.0)
        self.state_0, self.state_1 = self.state_1, self.state_0
        wp.synchronize_device(self.model.device)
        with wp.ScopedCapture(device=self.model.device) as capture:
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, None, None, 1.0 / 240.0)
        self.graph = capture.graph

        wp.capture_launch(self.graph)
        positions = self.state_1.particle_q.numpy().reshape(world_count, PARTICLES_PER_WORLD, 3)
        velocities = self.state_1.particle_qd.numpy().reshape(world_count, PARTICLES_PER_WORLD, 3)
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
            raise AssertionError(f"{solver} produced non-finite particle state")
        np.testing.assert_allclose(
            positions,
            np.broadcast_to(positions[:1], positions.shape),
            atol=1.0e-5,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            velocities,
            np.broadcast_to(velocities[:1], velocities.shape),
            atol=1.0e-5,
            rtol=0.0,
        )

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_captured_step(self, solver, world_count, query_mode):
        wp.capture_launch(self.graph)
        wp.synchronize_device(self.model.device)


if __name__ == "__main__":
    from newton.utils import run_benchmark

    run_benchmark(TimeFluidMultiworld)
