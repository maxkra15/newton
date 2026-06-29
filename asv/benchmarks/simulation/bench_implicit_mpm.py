# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""End-to-end implicit MPM multi-world benchmarks.

Each case advances the same deterministic particle block per environment. The
matrix compares legacy shared sparse allocation, isolated rebuildable sparse
stepping (eager and captured), and isolated captured fixed grids. Timings cover
complete, synchronized solver steps rather than selected kernels. Captured
cases record both directions of the two-state-buffer alternation in one graph.

CUDA memory is the Warp memory-pool high-water mark. ASV isolates parameter
cases in benchmark processes, and the standalone runner starts one child
process per parameter case so its memory results have the same scope.
"""

import itertools
import math
import statistics
import time

import warp as wp

wp.config.enable_backward = False
wp.config.log_level = wp.LOG_WARNING

from asv_runner.benchmarks.mark import SkipNotImplemented, skip_benchmark_if

import newton
from newton.solvers import SolverImplicitMPM

_CUDA_UNAVAILABLE = wp.get_cuda_device_count() == 0
_ISOLATED_MULTIWORLD_UNAVAILABLE = not hasattr(SolverImplicitMPM.Config, "separate_worlds")
_BENCHMARK_UNAVAILABLE = _CUDA_UNAVAILABLE or _ISOLATED_MULTIWORLD_UNAVAILABLE

_MODES = [
    "shared-sparse-allocating",
    "isolated-sparse-eager",
    "isolated-sparse-captured",
    "isolated-fixed-captured",
]
_CAPTURED_MODES = frozenset(("isolated-sparse-captured", "isolated-fixed-captured"))
_REBUILDABLE_SPARSE_MODES = frozenset(("isolated-sparse-eager", "isolated-sparse-captured"))
_ISOLATED_CAPACITY_MODES = _REBUILDABLE_SPARSE_MODES | {"isolated-fixed-captured"}

_PARTICLES_PER_ENVIRONMENT = 8
_CELLS_PER_ENVIRONMENT_ESTIMATE = 128
_STEPS_PER_CYCLE = 2
_WARMUP_CYCLES = 3
_SAMPLE_CYCLES = 15
_MEBIBYTE = 1024.0 * 1024.0


def _make_particle_world() -> newton.ModelBuilder:
    """Build the deterministic physical scene replicated by every case."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_particle_grid(
        pos=wp.vec3(0.025, 0.025, 0.025),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.25, 0.0, 0.0),
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


def _make_model(environment_count: int, device: wp.Device) -> newton.Model:
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    SolverImplicitMPM.register_custom_attributes(builder)
    for _ in range(environment_count):
        builder.add_world(_make_particle_world())
    return builder.finalize(device=device)


def _make_config(environment_count: int, mode: str) -> SolverImplicitMPM.Config:
    if mode not in _MODES:
        raise ValueError(f"Unknown benchmark mode: {mode}")

    config = SolverImplicitMPM.Config()
    config.grid_type = "fixed" if mode == "isolated-fixed-captured" else "sparse"
    config.grid_padding = 1 if config.grid_type == "fixed" else 0
    config.max_active_cell_count = (
        -1 if mode == "shared-sparse-allocating" else _CELLS_PER_ENVIRONMENT_ESTIMATE * environment_count
    )
    config.voxel_size = 0.1
    config.integration_scheme = "pic"
    config.transfer_scheme = "pic"
    config.separate_worlds = mode != "shared-sparse-allocating"
    config.solver = "jacobi"
    config.max_iterations = 4
    config.tolerance = 0.0
    config.warmstart_mode = "none"
    config.velocity_basis = "Q1"
    config.strain_basis = "P0"
    config.collider_basis = "Q1"
    return config


class FastImplicitMPMMultiworld:
    """Measure synchronized total-step scaling for the supported grid modes.

    The allocating mode has no reserved cell capacity, so its reported total
    capacity is zero. For every other mode the per-environment estimate is
    multiplied by the environment count to form one shared total reserve.
    """

    params = ([1, 8, 32], _MODES)
    param_names = ["environment_count", "mode"]

    number = 1
    repeat = 3
    rounds = 2
    dt = 0.01
    device = "cuda:0"

    def setup(self, environment_count: int, mode: str) -> None:
        wp.init()
        self._device = wp.get_device(self.device)
        if not self._device.is_mempool_supported:
            raise SkipNotImplemented("peak CUDA memory measurement requires Warp memory-pool support")

        self._environment_count = environment_count
        self._mode = mode
        self._graph = None
        self._steady_metrics = None
        self._peak_cuda_memory_bytes = wp.get_mempool_used_mem_high(self._device)

        setup_start = time.perf_counter()
        self._model = _make_model(environment_count, self._device)
        self._solver = SolverImplicitMPM(
            self._model,
            config=_make_config(environment_count, mode),
            enable_timers=False,
        )
        self._state_0 = self._model.state()
        self._state_1 = self._model.state()
        wp.synchronize_device(self._device)
        self._setup_milliseconds = (time.perf_counter() - setup_start) * 1000.0
        self._update_peak_cuda_memory()

        if self._model.particle_count != environment_count * _PARTICLES_PER_ENVIRONMENT:
            raise RuntimeError("benchmark scene replication produced an unexpected particle count")

        if mode in _REBUILDABLE_SPARSE_MODES and not self._solver._sparse_rebuildable:
            raise SkipNotImplemented(
                "isolated rebuildable sparse mode requires installed Warp rebuildable NanoVDB support "
                "for the selected Q1/P0 topology"
            )

        if (
            environment_count > 1
            and mode in _ISOLATED_CAPACITY_MODES
            and not self._has_capture_safe_environment_partitions()
        ):
            raise SkipNotImplemented(
                "isolated multi-world fixed-capacity stepping requires installed Warp capture-safe "
                "FEM environment partitions"
            )

        if mode in _CAPTURED_MODES and not self._solver.supports_cuda_graph_capture:
            try:
                self._solver.prepare_cuda_graph_capture()
            except RuntimeError as error:
                raise SkipNotImplemented(str(error)) from error
            raise SkipNotImplemented("solver did not advertise CUDA graph capture support")

        first_step_start = time.perf_counter()
        self._step_pair()
        self._synchronize_cycle()
        self._first_step_milliseconds = (time.perf_counter() - first_step_start) * 1000.0
        self._update_peak_cuda_memory()

        self._graph_recording_milliseconds = 0.0
        if mode in _CAPTURED_MODES:
            graph_start = time.perf_counter()
            self._solver.prepare_cuda_graph_capture()
            with wp.ScopedCapture(device=self._device, force_module_load=False) as capture:
                self._step_pair()
            self._graph = capture.graph
            wp.synchronize_device(self._device)
            self._graph_recording_milliseconds = (time.perf_counter() - graph_start) * 1000.0
            self._update_peak_cuda_memory()

        for _ in range(_WARMUP_CYCLES):
            self._execute_cycle()
        self._update_peak_cuda_memory()

    def _has_capture_safe_environment_partitions(self) -> bool:
        partition = self._solver._scratchpad._vel_space_restriction.space_partition
        return hasattr(type(partition), "_scatter_capped_partition_indices")

    def teardown(self, environment_count: int, mode: str) -> None:
        del environment_count, mode
        wp.synchronize_device(self._device)
        self._graph = None
        self._state_0 = None
        self._state_1 = None
        self._solver = None
        self._model = None

    def _step_pair(self) -> None:
        """Advance both directions of the fixed state-buffer alternation."""
        self._solver.step(self._state_0, self._state_1, control=None, contacts=None, dt=self.dt)
        self._solver.step(self._state_1, self._state_0, control=None, contacts=None, dt=self.dt)

    def _synchronize_cycle(self) -> None:
        if self._mode in _REBUILDABLE_SPARSE_MODES:
            # The public status check synchronizes before inspecting the sticky
            # device status, so no separate synchronization belongs here.
            self._solver.check_sparse_grid_rebuild_status()
        else:
            wp.synchronize_device(self._device)

    def _execute_cycle(self) -> None:
        if self._graph is None:
            self._step_pair()
        else:
            wp.capture_launch(self._graph)
        self._synchronize_cycle()

    def _update_peak_cuda_memory(self) -> None:
        self._peak_cuda_memory_bytes = max(
            self._peak_cuda_memory_bytes,
            wp.get_mempool_used_mem_high(self._device),
        )

    def _measure_steady_state(self) -> dict[str, float]:
        if self._steady_metrics is not None:
            return self._steady_metrics

        cycle_seconds = []
        for _ in range(_SAMPLE_CYCLES):
            start = time.perf_counter()
            self._execute_cycle()
            cycle_seconds.append(time.perf_counter() - start)
        self._update_peak_cuda_memory()

        step_seconds = [elapsed / _STEPS_PER_CYCLE for elapsed in cycle_seconds]
        ordered_step_seconds = sorted(step_seconds)
        p95_index = max(0, math.ceil(0.95 * len(ordered_step_seconds)) - 1)
        self._steady_metrics = {
            "median_total_step_milliseconds": statistics.median(step_seconds) * 1000.0,
            "p95_total_step_milliseconds": ordered_step_seconds[p95_index] * 1000.0,
            "steps_per_second": (_SAMPLE_CYCLES * _STEPS_PER_CYCLE) / sum(cycle_seconds),
        }
        return self._steady_metrics

    def _live_active_cells(self) -> int:
        self._measure_steady_state()
        grid = self._solver._scratchpad.grid
        if hasattr(grid, "cell_grid"):
            if hasattr(grid.cell_grid, "get_active_stats"):
                return int(grid.cell_grid.get_active_stats().voxel_count)
            return int(grid.cell_count())
        partition_cells = self._solver._scratchpad.domain.geometry_partition._cells
        return int((partition_cells.numpy() >= 0).sum())

    @skip_benchmark_if(_BENCHMARK_UNAVAILABLE)
    def time_total_step_cycle(self, environment_count: int, mode: str) -> None:
        """Time one synchronized cycle containing two complete solver steps."""
        del environment_count, mode
        self._execute_cycle()

    @skip_benchmark_if(_BENCHMARK_UNAVAILABLE)
    def track_setup_milliseconds(self, environment_count: int, mode: str) -> float:
        del environment_count, mode
        return self._setup_milliseconds

    track_setup_milliseconds.unit = "ms"

    @skip_benchmark_if(_BENCHMARK_UNAVAILABLE)
    def track_first_step_milliseconds(self, environment_count: int, mode: str) -> float:
        """Return the first synchronized step-pair time, including compilation."""
        del environment_count, mode
        return self._first_step_milliseconds

    track_first_step_milliseconds.unit = "ms"

    @skip_benchmark_if(_BENCHMARK_UNAVAILABLE)
    def track_graph_recording_milliseconds(self, environment_count: int, mode: str) -> float:
        del environment_count, mode
        return self._graph_recording_milliseconds

    track_graph_recording_milliseconds.unit = "ms"

    @skip_benchmark_if(_BENCHMARK_UNAVAILABLE)
    def track_median_total_step_milliseconds(self, environment_count: int, mode: str) -> float:
        del environment_count, mode
        return self._measure_steady_state()["median_total_step_milliseconds"]

    track_median_total_step_milliseconds.unit = "ms/step"

    @skip_benchmark_if(_BENCHMARK_UNAVAILABLE)
    def track_p95_total_step_milliseconds(self, environment_count: int, mode: str) -> float:
        del environment_count, mode
        return self._measure_steady_state()["p95_total_step_milliseconds"]

    track_p95_total_step_milliseconds.unit = "ms/step"

    @skip_benchmark_if(_BENCHMARK_UNAVAILABLE)
    def track_steps_per_second(self, environment_count: int, mode: str) -> float:
        del environment_count, mode
        return self._measure_steady_state()["steps_per_second"]

    track_steps_per_second.unit = "steps/s"

    @skip_benchmark_if(_BENCHMARK_UNAVAILABLE)
    def track_environment_count(self, environment_count: int, mode: str) -> int:
        del mode
        return environment_count

    track_environment_count.unit = "environments"

    @skip_benchmark_if(_BENCHMARK_UNAVAILABLE)
    def track_particles_per_environment(self, environment_count: int, mode: str) -> int:
        del environment_count, mode
        return _PARTICLES_PER_ENVIRONMENT

    track_particles_per_environment.unit = "particles/environment"

    @skip_benchmark_if(_BENCHMARK_UNAVAILABLE)
    def track_per_world_cell_estimate(self, environment_count: int, mode: str) -> int:
        del environment_count, mode
        return _CELLS_PER_ENVIRONMENT_ESTIMATE

    track_per_world_cell_estimate.unit = "cells/environment"

    @skip_benchmark_if(_BENCHMARK_UNAVAILABLE)
    def track_total_capacity(self, environment_count: int, mode: str) -> int:
        return 0 if mode == "shared-sparse-allocating" else _CELLS_PER_ENVIRONMENT_ESTIMATE * environment_count

    track_total_capacity.unit = "cells"

    @skip_benchmark_if(_BENCHMARK_UNAVAILABLE)
    def track_live_active_cells(self, environment_count: int, mode: str) -> int:
        del environment_count, mode
        return self._live_active_cells()

    track_live_active_cells.unit = "cells"

    @skip_benchmark_if(_BENCHMARK_UNAVAILABLE)
    def track_peak_cuda_memory_mebibytes(self, environment_count: int, mode: str) -> float:
        del environment_count, mode
        self._measure_steady_state()
        self._update_peak_cuda_memory()
        return self._peak_cuda_memory_bytes / _MEBIBYTE

    track_peak_cuda_memory_mebibytes.unit = "MiB"


def _run_standalone_case(benchmark_cls, params: tuple[int, str]) -> None:
    """Run one supported parameter case while preserving explicit skips."""
    instance = benchmark_cls()
    try:
        instance.setup(*params)
    except SkipNotImplemented as error:
        print(f"\n[Benchmark] Skipping {benchmark_cls.__name__}{params}: {error}")
        return

    try:
        print(f"\n[Benchmark] Running {benchmark_cls.__name__} with parameters {params}")
        cycle_start = time.perf_counter()
        instance.time_total_step_cycle(*params)
        cycle_seconds = time.perf_counter() - cycle_start
        print(f"  time_total_step_cycle: {cycle_seconds:.6f} s")
        for name in sorted(attr for attr in dir(instance) if attr.startswith("track_")):
            method = getattr(instance, name)
            value = method(*params)
            unit = getattr(method, "unit", "")
            print(f"  {name}: {value:.6f} {unit}".rstrip())
    finally:
        instance.teardown(*params)


if __name__ == "__main__":
    import argparse
    import subprocess
    import sys

    if _BENCHMARK_UNAVAILABLE:
        reason = (
            "CUDA device cuda:0 is unavailable"
            if _CUDA_UNAVAILABLE
            else "SolverImplicitMPM.Config.separate_worlds is unavailable"
        )
        print(f"Skipping FastImplicitMPMMultiworld: {reason}.")
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
        help="Run a specific benchmark; may be repeated (e.g., --bench FastImplicitMPMMultiworld).",
    )
    parser.add_argument(
        "--internal-environment-count", type=int, choices=FastImplicitMPMMultiworld.params[0], help=argparse.SUPPRESS
    )
    parser.add_argument("--internal-mode", choices=_MODES, help=argparse.SUPPRESS)
    args = parser.parse_known_args()[0]

    benchmarks = args.bench if args.bench is not None else benchmark_list.keys()
    internal_case = (args.internal_environment_count, args.internal_mode)
    if (internal_case[0] is None) != (internal_case[1] is None):
        parser.error("internal standalone case arguments must be supplied together")

    if internal_case[0] is not None:
        for key in benchmarks:
            _run_standalone_case(benchmark_list[key], internal_case)
    else:
        for key in benchmarks:
            for environment_count, mode in itertools.product(*benchmark_list[key].params):
                result = subprocess.run(
                    [
                        sys.executable,
                        __file__,
                        "--bench",
                        key,
                        "--internal-environment-count",
                        str(environment_count),
                        "--internal-mode",
                        mode,
                    ],
                    check=False,
                )
                if result.returncode != 0:
                    raise SystemExit(result.returncode)
