# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Implicit MPM multi-world scaling benchmarks.

The ``multiworld`` layout advances all worlds with one isolated solver, model,
and state pair. The ``independent`` layout advances the same number of worlds
with one separately finalized solver, model, and state pair per world. Both
layouts preserve world isolation; this benchmark does not compare isolated
and shared-grid semantics.
"""

import time

import warp as wp

wp.config.enable_backward = False
wp.config.log_level = wp.LOG_WARNING

from asv_runner.benchmarks.mark import skip_benchmark_if

import newton
from newton.solvers import SolverImplicitMPM

_CUDA_UNAVAILABLE = wp.get_cuda_device_count() == 0
_PARTICLES_PER_WORLD = 8


def _make_particle_world() -> newton.ModelBuilder:
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_particle_grid(
        pos=wp.vec3(0.025, 0.025, 0.025),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=2,
        dim_y=2,
        dim_z=2,
        cell_x=0.05,
        cell_y=0.05,
        cell_z=0.05,
        mass=0.01,
        jitter=0.0,
        radius_mean=0.025,
        custom_attributes={"mpm:young_modulus": 1.0e4, "mpm:poisson_ratio": 0.2},
    )
    return builder


def _make_config() -> SolverImplicitMPM.Config:
    config = SolverImplicitMPM.Config()
    config.grid_type = "fixed"
    config.grid_padding = 1
    config.voxel_size = 0.1
    config.integration_scheme = "pic"
    config.transfer_scheme = "pic"
    config.separate_worlds = True
    config.solver = "jacobi"
    # This is a fixed work budget, not a claim of exactly four Jacobi sweeps:
    # the graph-based solver may execute iterations in larger chunks.
    config.max_iterations = 4
    config.tolerance = 0.0
    config.warmstart_mode = "grid"
    return config


def _make_multiworld_model(world_count: int, device: str) -> newton.Model:
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    SolverImplicitMPM.register_custom_attributes(builder)
    for _ in range(world_count):
        builder.add_world(_make_particle_world())
    return builder.finalize(device=device)


class _Runner:
    def __init__(self, model: newton.Model):
        self.model = model
        self.solver = SolverImplicitMPM(model, config=_make_config())
        self.state_0 = model.state()
        self.state_1 = model.state()

    def step(self, dt: float) -> None:
        self.solver.step(self.state_0, self.state_1, control=None, contacts=None, dt=dt)
        self.state_0, self.state_1 = self.state_1, self.state_0


class FastImplicitMPMMultiworld:
    """Compare correct isolated multi-world and independent-solver layouts.

    ``multiworld`` uses one isolated MPM solver for all local worlds, while
    ``independent`` uses one one-world solver per world as the correctness
    reference layout. Each world contains the same deterministic 2x2x2
    particle block.
    """

    params = ([1, 8, 32], ["multiworld", "independent"])
    param_names = ["world_count", "layout"]

    number = 1
    repeat = 3
    rounds = 2
    steps = 5
    dt = 0.01
    device = "cuda:0"

    def setup(self, world_count: int, layout: str) -> None:
        wp.init()

        if layout == "multiworld":
            models = [_make_multiworld_model(world_count, self.device)]
        elif layout == "independent":
            models = [_make_particle_world().finalize(device=self.device) for _ in range(world_count)]
        else:
            raise ValueError(f"Unknown benchmark layout: {layout}")

        self._runners = [_Runner(model) for model in models]
        expected_runner_count = 1 if layout == "multiworld" else world_count
        assert len(self._runners) == expected_runner_count
        assert sum(runner.model.particle_count for runner in self._runners) == world_count * _PARTICLES_PER_WORLD

        self._step_all()

    def _step_all(self) -> None:
        for _ in range(self.steps):
            for runner in self._runners:
                runner.step(self.dt)
        wp.synchronize_device(self.device)

    @skip_benchmark_if(_CUDA_UNAVAILABLE)
    def time_step(self, world_count: int, layout: str) -> None:
        self._step_all()

    @skip_benchmark_if(_CUDA_UNAVAILABLE)
    def track_milliseconds_per_world_step(self, world_count: int, layout: str) -> float:
        start_time = time.perf_counter()
        self._step_all()
        elapsed_seconds = time.perf_counter() - start_time
        return elapsed_seconds * 1000.0 / (world_count * self.steps)

    track_milliseconds_per_world_step.unit = "ms/world-step"


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    if _CUDA_UNAVAILABLE:
        print("Skipping FastImplicitMPMMultiworld: CUDA device cuda:0 is unavailable.")
        raise SystemExit(0)

    benchmark_list = {
        "FastImplicitMPMMultiworld": FastImplicitMPMMultiworld,
    }

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-b",
        "--bench",
        default=None,
        action="append",
        choices=benchmark_list.keys(),
        help="Run a specific benchmark; may be repeated to run multiple (e.g., --bench FastImplicitMPMMultiworld).",
    )
    args = parser.parse_known_args()[0]

    benchmarks = args.bench if args.bench is not None else benchmark_list.keys()
    for key in benchmarks:
        run_benchmark(benchmark_list[key])
