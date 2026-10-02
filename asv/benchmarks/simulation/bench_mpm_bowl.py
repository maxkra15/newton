# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare equal-duration bowl rollouts at two particle counts and fidelities.

Run directly with ``uv run --extra dev asv/benchmarks/simulation/bench_mpm_bowl.py``.
Timings include integration, neighbor/grid rebuilds, swept collision projection,
and prescribed bowl motion, with no rendering or kernel compilation. Run on
an otherwise idle GPU so concurrent workloads do not dominate frame timings.
"""

import argparse
import json
import time
from pathlib import Path
from statistics import median

import numpy as np
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


def main():
    """Record timing and containment independently for equal-duration rollouts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--particle-iterations", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--particle-radii", type=float, nargs="+", default=[0.016, 0.008])
    parser.add_argument("--solvers", choices=("particles", "mpm"), nargs="+", default=["particles", "mpm"])
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--critical-fraction", type=float, default=0.0)
    parser.add_argument("--tolerance", type=float, default=1.0e-4)
    parser.add_argument("--output", type=Path, default=Path("/tmp/mpm-bowl-timing.json"))
    args = parser.parse_args()
    if args.frames <= 10 or args.repeats < 1 or any(count < 1 for count in args.particle_iterations):
        parser.error("Require more than ten frames, positive repeats, and positive particle iteration counts")
    wp.init()
    if not wp.get_device().is_cuda:
        parser.error("The sparse-grid bowl benchmark requires CUDA")
    if not args.no_cuda_graph and not wp.is_conditional_graph_supported():
        parser.error("Conditional CUDA graphs are unavailable; pass --no-cuda-graph")
    report = {
        "warp_version": wp.__version__,
        "device": wp.get_device().name,
        "frames": args.frames,
        "repeats": args.repeats,
        "warmup_frames": 10,
        "critical_fraction": args.critical_fraction,
        "tolerance": args.tolerance,
        "voxel_size_m": 0.08,
        "max_active_cell_count": 4096,
        "fps": 60,
        "substeps": 2,
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for trial in range(1, args.repeats + 1):
        for radius in args.particle_radii:
            for solver in args.solvers:
                iterations = args.particle_iterations if solver == "particles" else [2]
                for particle_iterations in iterations:
                    options = Example.create_parser().parse_args(
                        [
                            "--solver",
                            solver,
                            "--particle-radius",
                            str(radius),
                            "--particle-iterations",
                            str(particle_iterations),
                            "--critical-fraction",
                            str(args.critical_fraction),
                            "--tolerance",
                            str(args.tolerance),
                            "--no-use-cuda-graph" if args.no_cuda_graph else "--use-cuda-graph",
                        ]
                    )
                    example = Example(ViewerNull(), options)
                    times = []
                    for frame in range(args.frames):
                        start = time.perf_counter()
                        example.step()
                        wp.synchronize_device(example.model.device)
                        if frame >= 10:
                            times.append(1000.0 * (time.perf_counter() - start))
                    validation_error = None
                    try:
                        example.test_post_step()
                    except AssertionError as error:
                        validation_error = str(error)
                    q = example.state_0.particle_q.numpy()
                    velocity = example.state_0.particle_qd.numpy()
                    if not np.isfinite(q).all() or not np.isfinite(velocity).all():
                        raise RuntimeError("Cannot benchmark a nonfinite bowl rollout")
                    pose = example.state_0.body_q.numpy()[0]
                    rotation = np.asarray(wp.quat_to_matrix(wp.quat(*pose[3:]))).reshape(3, 3)
                    local = (q - pose[:3]) @ rotation
                    outside = (local[:, 2] < -0.04) & (np.linalg.norm(local, axis=1) >= example.bowl_radius + 0.01)
                    row = {
                        "trial": trial,
                        "solver": solver,
                        "particles": example.model.particle_count,
                        "particle_radius_m": radius,
                        "particle_iterations": particle_iterations if solver == "particles" else None,
                        "mpm_iterations": 100 if solver == "mpm" else None,
                        "cuda_graph": example.graph is not None,
                        "median_frame_ms": median(times),
                        "p95_frame_ms": float(np.quantile(times, 0.95)),
                        "validation_passed": validation_error is None,
                        "validation_error": validation_error,
                        "outside_below_rim_count": int(outside.sum()),
                        "speed_max_m_s": float(np.linalg.norm(velocity, axis=1).max()),
                    }
                    report["cases"].append(row)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    print(json.dumps(row), flush=True)
                    # Captured hash grids are measured serially, before another grid is built.
                    del example


if __name__ == "__main__":
    main()
