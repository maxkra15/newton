# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure low-iteration MPM approximations on identical sparse-grid bowl rollouts.

Run ``uv run --extra dev asv/benchmarks/simulation/bench_mpm_low_iterations.py
--output /tmp/mpm-low-iterations.json``. The XPBD stress seed and signed packing
bias below are experiments, not supported Newton solver features. They deliberately
live outside the production solver so their quality and cost can be compared first.
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np
import warp as wp
import warp.fem as fem

import newton
from newton._src.solvers.implicit_mpm.implicit_mpm_solver_kernels import integrate_particle_stress
from newton._src.solvers.implicit_mpm.rheology_solver_kernels import unilateral_offset_to_strain_rhs, vec6
from newton.examples.mpm.example_mpm_bowl import Example
from newton.solvers import SolverImplicitMPM, SolverXPBD
from newton.viewer import ViewerNull


@wp.kernel(enable_backward=False)
def _accumulate_pair_stress(
    grid: wp.uint64,
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    radius: wp.array[float],
    flags: wp.array[int],
    friction: float,
    cohesion: float,
    max_radius: float,
    dt: float,
    relaxation: float,
    stress: wp.array[wp.mat33],
):
    """Estimate compressive virial stress from one XPBD pair-contact iteration."""
    i = wp.hash_grid_point_id(grid, wp.tid())
    if i < 0 or (flags[i] & newton.ParticleFlags.ACTIVE) == 0 or inv_mass[i] <= 0.0:
        return
    x = positions[i]
    query = wp.hash_grid_query(grid, x, radius[i] + max_radius + cohesion)
    sigma = wp.mat33(0.0)
    for j in query:
        if i != j and (flags[j] & newton.ParticleFlags.ACTIVE) != 0:
            separation = x - positions[j]
            distance = wp.length(separation)
            gap = distance - radius[i] - radius[j]
            denominator = inv_mass[i] + inv_mass[j]
            if gap <= cohesion and denominator > 0.0 and distance > 0.0:
                normal = separation / distance
                relative_velocity = velocities[i] - velocities[j]
                tangent_velocity = relative_velocity - normal * wp.dot(normal, relative_velocity)
                friction_gap = wp.max(friction * gap, -wp.length(tangent_velocity) * dt)
                pair_force = (wp.normalize(tangent_velocity) * friction_gap - normal * gap) * (
                    relaxation / (denominator * dt * dt)
                )
                # Split each pair's virial equally between its two particles.
                virial = wp.outer(pair_force, separation)
                sigma += 0.25 * (virial + wp.transpose(virial))
    volume = 8.0 * radius[i] * radius[i] * radius[i]
    stress[i] += sigma / volume


@wp.kernel(enable_backward=False)
def _record_max_points_per_cell(offsets: wp.array[int], maximum: wp.array[int]):
    cell = wp.tid()
    wp.atomic_max(maximum, 0, offsets[cell + 1] - offsets[cell])


@wp.kernel(enable_backward=False)
def _apply_packing_bias(
    critical_fraction: float,
    beta: float,
    max_fraction_correction: float,
    particle_volume: wp.array[float],
    collider_volume: wp.array[float],
    node_volume: wp.array[float],
    offset: wp.array[float],
    strain_rhs: wp.array[vec6],
):
    """Add a bounded expansion target where particle volume exceeds free grid volume."""
    i = wp.tid()
    available_volume = wp.max(0.0, node_volume[i] - collider_volume[i]) * critical_fraction
    excess = wp.max(0.0, particle_volume[i] - available_volume)
    correction = -wp.min(beta * excess, max_fraction_correction * particle_volume[i])
    offset[i] += correction
    # Production preprocessing adds only the positive void offset. Inject the
    # negative part here; postprocessing already removes the complete offset
    # from the physical material strain. Do not disable cohesion for compression.
    strain_rhs[i] += unilateral_offset_to_strain_rhs(correction)


class _StressPredictor(SolverXPBD):
    """Collect pair stresses during a throwaway XPBD prediction."""

    def __init__(self, model):
        super().__init__(model, iterations=2)
        self.stress = wp.zeros(model.particle_count, dtype=wp.mat33, device=model.device)

    def _apply_particle_deltas(self, model, state_in, state_out, particle_deltas, dt):
        positions = state_out.particle_q if self._particle_delta_counter == 0 else state_in.particle_q
        velocities = state_out.particle_qd if self._particle_delta_counter == 0 else state_in.particle_qd
        wp.launch(
            _accumulate_pair_stress,
            dim=model.particle_count,
            inputs=[
                model.particle_grid.id,
                positions,
                velocities,
                model.particle_inv_mass,
                model.particle_radius,
                model.particle_flags,
                model.particle_mu,
                model.particle_cohesion,
                model.particle_max_radius,
                dt,
                self.soft_contact_relaxation,
            ],
            outputs=[self.stress],
            device=model.device,
        )
        return super()._apply_particle_deltas(model, state_in, state_out, particle_deltas, dt)


class _ExperimentalMPM(SolverImplicitMPM):
    """Keep experimental predictors and stabilization out of the production API."""

    def __init__(self, model, config, *, xpbd_seed=False, packing_beta=0.0, point_capacity=0):
        self.point_capacity = point_capacity
        self.max_points_per_cell = wp.zeros(1, dtype=int, device=model.device)
        super().__init__(model, config)
        self.packing_beta = packing_beta
        self.predictor = _StressPredictor(model) if xpbd_seed else None
        self.predictor_in = model.state() if xpbd_seed else None
        self.predictor_out = model.state() if xpbd_seed else None
        self.preprojection_q = wp.empty(model.particle_count, dtype=wp.vec3, device=model.device)
        self.packing_volumes = wp.zeros((3, config.max_active_cell_count), dtype=float, device=model.device)

    def _rebuild_scratchpad(self, pic):
        if self.point_capacity:
            # The public picN basis avoids a host maximum-PPC readback. Check
            # its bound on the device instead of silently dropping contacts.
            wp.launch(
                _record_max_points_per_cell,
                dim=pic.cell_particle_offsets.shape[0] - 1,
                inputs=[pic.cell_particle_offsets],
                outputs=[self.max_points_per_cell],
                device=self.model.device,
            )
        return super()._rebuild_scratchpad(pic)

    def check_sparse_grid_rebuild_status(self):
        super().check_sparse_grid_rebuild_status()
        if self.point_capacity and self.max_points_per_cell.numpy()[0] > self.point_capacity:
            raise RuntimeError("Experimental point-basis capacity exceeded; discard this rollout and increase capacity")

    def _build_plasticity_system(self, state_in, dt, pic, scratch, inv_cell_volume):
        super()._build_plasticity_system(state_in, dt, pic, scratch, inv_cell_volume)
        # Scratch arrays return to the temporary pool after the step. Record
        # measurements while they are valid, including on outer graph replay.
        self.packing_volumes.zero_()
        for row, volume in enumerate(
            (scratch.strain_node_particle_volume, scratch.strain_node_collider_volume, scratch.strain_node_volume)
        ):
            wp.copy(self.packing_volumes[row], volume, count=scratch.strain_node_count)
        if self.packing_beta > 0.0:
            wp.launch(
                _apply_packing_bias,
                dim=scratch.strain_node_count,
                inputs=[
                    self._mpm_model.critical_fraction,
                    self.packing_beta,
                    0.05,
                    scratch.strain_node_particle_volume,
                    scratch.strain_node_collider_volume,
                    scratch.strain_node_volume,
                ],
                outputs=[scratch.unilateral_strain_offset, scratch.elastic_strain_delta_field.dof_values],
                device=self.model.device,
            )

    def _load_warmstart(self, state_in, last_step_data, scratch, pic, inv_cell_volume):
        super()._load_warmstart(state_in, last_step_data, scratch, pic, inv_cell_volume)
        if self.predictor is not None:
            for name in ("particle_q", "particle_qd", "particle_f", "body_q", "body_qd"):
                wp.copy(getattr(self.predictor_in, name), getattr(state_in, name))
            self.predictor.stress.zero_()
            self.predictor.step(self.predictor_in, self.predictor_out, None, None, self._step_dt)
            fem.integrate(
                integrate_particle_stress,
                quadrature=pic,
                fields={"tau": scratch.sym_strain_test},
                values={
                    "particle_stress": self.predictor.stress,
                    "particle_flags": self._mpm_model.material_particle_flags,
                    "inv_cell_volume": inv_cell_volume,
                },
                output=scratch.stress_field.dof_values,
            )

    def step(self, state_in, state_out, control, contacts, dt):
        self._step_dt = dt
        return super().step(state_in, state_out, control, contacts, dt)

    def project_outside(self, state_in, state_out, dt, gap=None, *, state_prev=None):
        wp.copy(self.preprojection_q, state_in.particle_q)
        return super().project_outside(state_in, state_out, dt, gap, state_prev=state_prev)


@dataclass(frozen=True)
class _Case:
    name: str
    iterations: int
    solver: str | tuple[str, ...] = "gs"
    warmstart: str = "particles"
    xpbd_seed: bool = False
    packing_beta: float = 0.0
    collider_basis: str = "S2"
    point_capacity: int = 0


_CASES = (
    _Case("reference", 100),
    _Case("gs0", 0),
    _Case("gs2", 2),
    _Case("gs5", 5),
    _Case("gs10", 10),
    _Case("gs20", 20),
    _Case("cold2", 2, warmstart="none"),
    _Case("jacobi_gs2", 2, solver=("jacobi", "gs")),
    _Case("cr_gs2", 2, solver=("cr", "gs")),
    _Case("xpbd_gs2", 2, xpbd_seed=True),
    _Case("packing_gs2", 2, packing_beta=0.2),
    _Case("packing_gs20", 20, packing_beta=0.2),
    _Case("pic100", 100, collider_basis="pic"),
    _Case("pic2", 2, collider_basis="pic"),
    _Case("pic20", 20, collider_basis="pic"),
    _Case("pic_captured100", 100, collider_basis="pic", point_capacity=512),
    _Case("pic_captured2", 2, collider_basis="pic", point_capacity=512),
    _Case("pic_captured20", 20, collider_basis="pic", point_capacity=512),
)


def _make_example(case, *, voxel_size, particle_radius, cuda_graph, cell_capacity=4096):
    args = Example.create_parser().parse_args(
        [
            "--no-use-cuda-graph",
            "--voxel-size",
            str(voxel_size),
            "--particle-radius",
            str(particle_radius),
            "--max-active-cell-count",
            str(cell_capacity),
        ]
    )
    example = Example(ViewerNull(), args)
    config = SolverImplicitMPM.Config(
        voxel_size=voxel_size,
        grid_type="sparse",
        max_active_cell_count=cell_capacity,
        max_leaf_node_count=128,
        max_lower_node_count=64,
        max_upper_node_count=16,
        collider_basis=f"pic{case.point_capacity}" if case.point_capacity else case.collider_basis,
        warmstart_mode=case.warmstart,
        solver=case.solver,
        max_iterations=case.iterations,
        tolerance=0.0,
        critical_fraction=1.0,
    )
    example.mpm_solver = _ExperimentalMPM(
        example.model,
        config,
        xpbd_seed=case.xpbd_seed,
        packing_beta=case.packing_beta,
        point_capacity=case.point_capacity,
    )
    example.use_cuda_graph = cuda_graph
    example.capture()
    return example


def _metrics(example):
    solver = example.mpm_solver
    q = example.state_0.particle_q.numpy().astype(float)
    velocity = example.state_0.particle_qd.numpy().astype(float)
    masses = example.model.particle_mass.numpy().astype(float)
    volume, collider_volume, node_volume = solver.packing_volumes.numpy().astype(float)
    available = np.maximum(0.0, node_volume - collider_volume)
    available *= solver._mpm_model.critical_fraction
    projection = q - solver.preprojection_q.numpy()
    pose = example.state_0.body_q.numpy()[0]
    rotation = np.asarray(wp.quat_to_matrix(wp.quat(*pose[3:]))).reshape(3, 3)
    local = (q - pose[:3]) @ rotation
    return (
        {
            "packing_excess_fraction": float(np.maximum(volume - available, 0.0).sum() / volume.sum()),
            "projection_rms_m": float(np.sqrt(np.mean(np.sum(projection**2, axis=1)))),
            "projection_fraction": float(np.mean(np.linalg.norm(projection, axis=1) > 1.0e-6)),
            "speed_rms_m_s": float(np.sqrt(np.mean(np.sum(velocity**2, axis=1)))),
            "speed_max_m_s": float(np.linalg.norm(velocity, axis=1).max()),
            "kinetic_energy_j": float(0.5 * np.dot(masses, np.sum(velocity**2, axis=1))),
            "pile_height_p95_m": float(np.quantile(local[:, 2], 0.95) + example.bowl_radius),
            "outside_below_rim_count": int(
                np.count_nonzero((local[:, 2] < -0.04) & (np.linalg.norm(local, axis=1) >= example.bowl_radius + 0.01))
            ),
        },
        q,
        velocity,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/mpm-low-iterations.json"))
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--particle-radius", type=float, default=0.016)
    parser.add_argument("--cell-capacity", type=int, default=4096)
    parser.add_argument("--voxel-sizes", type=float, nargs="+", default=[0.08, 0.12])
    parser.add_argument("--cases", nargs="+", choices=[case.name for case in _CASES])
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--paired", action="store_true", help="Interleave captured MPM-only cases for paired timings.")
    args = parser.parse_args()
    if args.frames <= 10:
        parser.error("--frames must exceed the 10 warmup frames")

    wp.config.enable_backward = False
    wp.config.log_level = wp.LOG_WARNING
    wp.init()
    device = wp.get_device()
    if not device.is_cuda:
        parser.error("The sparse-grid bowl experiment requires CUDA")
    if not args.no_cuda_graph and not wp.is_conditional_graph_supported():
        parser.error("Conditional CUDA graphs are unavailable; pass --no-cuda-graph")

    report = {
        "warp_version": wp.__version__,
        "device": device.name,
        "particle_radius_m": args.particle_radius,
        "max_active_cell_count": args.cell_capacity,
        "frames": args.frames,
        "fps": 60,
        "substeps": 2,
        "cuda_graph": not args.no_cuda_graph,
        "paired": args.paired,
        "critical_fraction": 1.0,
        "tolerance": 0.0,
        "jacobi_seed_iterations": 5,
        "xpbd_seed_iterations": 2,
        "packing_beta": 0.2,
        "packing_correction_cap": 0.05,
        "cases": [],
    }
    requested = set(args.cases) if args.cases else {case.name for case in _CASES}
    required = {"reference"}
    if requested & {"pic2", "pic20"}:
        required.add("pic100")
    if requested & {"pic_captured2", "pic_captured20"}:
        required.add("pic_captured100")
    selected = [case for case in _CASES if case.name in requested | required]
    if args.paired and (
        args.no_cuda_graph
        or any(case.xpbd_seed or (case.collider_basis == "pic" and not case.point_capacity) for case in selected)
    ):
        parser.error("--paired requires captured MPM-only cases; exclude XPBD seeds and unbounded pic bases")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for voxel_size in args.voxel_sizes:
        reference_q = reference_v = None
        basis_references = {}
        paired_examples, paired_times = {}, {}
        if args.paired:
            for case in selected:
                paired_examples[case.name] = _make_example(
                    case,
                    voxel_size=voxel_size,
                    particle_radius=args.particle_radius,
                    cuda_graph=True,
                    cell_capacity=args.cell_capacity,
                )
                paired_times[case.name] = []
            order = np.random.default_rng(42)
            for frame in range(args.frames):
                for index in order.permutation(len(selected)):
                    name = selected[index].name
                    start = time.perf_counter()
                    paired_examples[name].step()
                    wp.synchronize_device(device)
                    if frame >= 10:
                        paired_times[name].append(1000.0 * (time.perf_counter() - start))
        for case in selected:
            # Warp 1.17's point basis queries max particles/cell on the host at
            # every rebuild. Keep this diagnostic path outside outer capture.
            cuda_graph = not args.no_cuda_graph and (case.collider_basis != "pic" or case.point_capacity > 0)
            if args.paired:
                example = paired_examples.pop(case.name)
                times = paired_times[case.name]
            else:
                example = _make_example(
                    case,
                    voxel_size=voxel_size,
                    particle_radius=args.particle_radius,
                    cuda_graph=cuda_graph,
                    cell_capacity=args.cell_capacity,
                )
                times = []
                for frame in range(args.frames):
                    start = time.perf_counter()
                    example.step()
                    wp.synchronize_device(device)
                    elapsed = 1000.0 * (time.perf_counter() - start)
                    if frame >= 10:
                        times.append(elapsed)
            validation_error = None
            try:
                example.test_post_step()
            except AssertionError as error:
                validation_error = str(error)
            metrics, q, velocity = _metrics(example)
            if not np.isfinite(q).all() or not np.isfinite(velocity).all():
                raise RuntimeError(f"{case.name} produced nonfinite state; cannot compare rollout metrics")
            if reference_q is None:
                reference_q, reference_v = q, velocity
            basis_key = case.collider_basis, bool(case.point_capacity)
            if case.iterations == 100:
                basis_references[basis_key] = q, velocity
            if basis_key in basis_references:
                basis_q, basis_v = basis_references[basis_key]
                metrics["position_rms_error_same_basis_m"] = float(np.sqrt(np.mean(np.sum((q - basis_q) ** 2, axis=1))))
                metrics["velocity_rms_error_same_basis_m_s"] = float(
                    np.sqrt(np.mean(np.sum((velocity - basis_v) ** 2, axis=1)))
                )
            metrics["position_rms_error_m"] = float(np.sqrt(np.mean(np.sum((q - reference_q) ** 2, axis=1))))
            metrics["velocity_rms_error_m_s"] = float(np.sqrt(np.mean(np.sum((velocity - reference_v) ** 2, axis=1))))
            result = {
                "case": case.name,
                "voxel_size_m": voxel_size,
                "particles": example.model.particle_count,
                "final_iterations": case.iterations,
                "solver": case.solver,
                "collider_basis": example.mpm_solver.collider_basis,
                "cuda_graph": cuda_graph,
                "point_capacity": case.point_capacity,
                "max_points_per_cell": int(example.mpm_solver.max_points_per_cell.numpy()[0])
                if case.point_capacity
                else None,
                "warmstart": case.warmstart,
                "validation_passed": validation_error is None,
                "validation_error": validation_error,
                "median_frame_ms": median(times),
                "p95_frame_ms": float(np.quantile(times, 0.95)),
                **metrics,
            }
            report["cases"].append(result)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(result), flush=True)
            # Finish each scene before building another hash grid: Warp 1.17's
            # captured hash-grid descriptors can be affected by another grid build.
            del example


if __name__ == "__main__":
    main()
