# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare equal-duration bowl rollouts at two particle counts and fidelities.

Run directly with ``uv run --extra dev asv/benchmarks/simulation/bench_mpm_bowl.py``.
Timings include integration, neighbor/grid rebuilds, swept collision projection,
and prescribed bowl motion, with no rendering or kernel compilation. Run on
an otherwise idle GPU so concurrent workloads do not dominate frame timings.
"""

import warp as wp
from asv_runner.benchmarks.mark import SkipNotImplemented

wp.config.enable_backward = False
wp.config.log_level = wp.LOG_WARNING

from newton.examples.mpm.example_mpm_bowl import Example
from newton.viewer import ViewerNull


class MPMBowl:
    params = [["particles", "mpm"], [False, True], [0.016, 0.008]]
    param_names = ["solver", "cuda_graph", "particle_radius"]
    number = 1
    repeat = 5
    rounds = 2

    def setup(self, solver, cuda_graph, particle_radius):
        if not wp.get_device().is_cuda:
            raise SkipNotImplemented
        args = Example.create_parser().parse_args(
            [
                "--solver",
                solver,
                "--particle-radius",
                str(particle_radius),
                "--use-cuda-graph" if cuda_graph else "--no-use-cuda-graph",
            ]
        )
        self.example = Example(ViewerNull(), args)
        for _ in range(10):
            self.example.step()
        wp.synchronize_device(self.example.model.device)

    def time_frame(self, solver, cuda_graph, particle_radius):
        self.example.step()
        wp.synchronize_device(self.example.model.device)

    def track_particle_count(self, solver, cuda_graph, particle_radius):
        return self.example.model.particle_count

    def teardown(self, solver, cuda_graph, particle_radius):
        self.example.mpm_solver.check_sparse_grid_rebuild_status()
        self.example.test_post_step()


if __name__ == "__main__":
    from statistics import median

    from newton.utils import run_benchmark

    runs = [run_benchmark(MPMBowl, number=30, print_results=False) for _ in range(3)]
    print("\nMedian of three runs, each timing 30 synchronized frames:")
    for key in runs[0]:
        method, params = key
        value = median(run[key] for run in runs)
        if method.startswith("time_"):
            print(f"{method} {params}: {value * 1000.0:.3f} ms")
        else:
            print(f"{method} {params}: {value:.0f}")
