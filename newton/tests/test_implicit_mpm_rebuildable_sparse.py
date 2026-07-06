# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest import mock

import numpy as np
import warp as wp
import warp.fem as fem

import newton
from newton._src.solvers.implicit_mpm.implicit_mpm_solver_kernels import (
    supports_rebuildable_environment_nanogrid,
    supports_rebuildable_nanogrid,
    supports_rebuildable_volume,
)
from newton._src.solvers.implicit_mpm.solver_implicit_mpm import _sparse_grid_rebuild_error
from newton.solvers import SolverImplicitMPM
from newton.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices


def _require_rebuildable_sparse(test):
    if not supports_rebuildable_nanogrid():
        test.skipTest("Installed Warp does not expose rebuildable Nanogrids")


def _require_rebuildable_s2(test):
    _require_rebuildable_sparse(test)
    if not getattr(fem.Nanogrid, "REBUILDABLE_EDGE_TOPOLOGY", False):
        test.skipTest("Installed Warp does not expose rebuildable S2 edge topology")


def _make_particle_model(device, positions, inactive_indices=()):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    SolverImplicitMPM.register_custom_attributes(builder)
    for position in positions:
        builder.add_particle(wp.vec3(*position), wp.vec3(0.0), mass=1.0)

    model = builder.finalize(device=device)
    if inactive_indices:
        flags = model.particle_flags.numpy()
        for particle_index in inactive_indices:
            flags[particle_index] &= ~int(newton.ParticleFlags.ACTIVE)
        model.particle_flags.assign(flags)
    return model


def _make_sparse_solver(model, max_active_cell_count, collider_basis="Q1", voxel_size=0.1, **config_kwargs):
    config = SolverImplicitMPM.Config(
        grid_type="sparse",
        voxel_size=voxel_size,
        max_active_cell_count=max_active_cell_count,
        velocity_basis="Q1",
        strain_basis="P0",
        collider_basis=collider_basis,
        max_iterations=2,
        warmstart_mode="none",
        **config_kwargs,
    )
    return SolverImplicitMPM(model, config, verbose=False)


def test_rebuildable_sparse_capability_requires_node_capacity_api(test, device):
    del device

    def allocate_without_node_capacities(
        voxel_points,
        rebuildable=False,
        max_active_voxels=None,
        status=None,
        point_mask=None,
    ):
        del voxel_points, rebuildable, max_active_voxels, status, point_mask

    def rebuild_with_status(self, voxel_points, status=None, point_mask=None):
        del self, voxel_points, status, point_mask

    def environments_without_node_capacities(
        points,
        point_envs,
        env_count,
        *,
        rebuildable=False,
        max_active_voxels=None,
        status=None,
        point_mask=None,
    ):
        del points, point_envs, env_count, rebuildable, max_active_voxels, status, point_mask

    try:
        with (
            mock.patch.object(wp.Volume, "allocate_by_voxels", allocate_without_node_capacities),
            mock.patch.object(wp.Volume, "rebuild", rebuild_with_status, create=True),
        ):
            supports_rebuildable_volume.cache_clear()
            test.assertFalse(supports_rebuildable_volume())

        with (
            mock.patch(
                "newton._src.solvers.implicit_mpm.implicit_mpm_solver_kernels.supports_rebuildable_nanogrid",
                return_value=True,
            ),
            mock.patch.object(fem.Nanogrid, "from_environment_voxels", environments_without_node_capacities),
        ):
            supports_rebuildable_environment_nanogrid.cache_clear()
            test.assertFalse(supports_rebuildable_environment_nanogrid())
    finally:
        supports_rebuildable_volume.cache_clear()
        supports_rebuildable_nanogrid.cache_clear()
        supports_rebuildable_environment_nanogrid.cache_clear()


def test_rebuildable_sparse_s2_capability_gating(test, device):
    model = _make_particle_model(device, [(0.01, 0.01, 0.01)])
    solver = _make_sparse_solver(model, max_active_cell_count=4, collider_basis="S2")
    expected = supports_rebuildable_nanogrid() and getattr(fem.Nanogrid, "REBUILDABLE_EDGE_TOPOLOGY", False)
    test.assertEqual(solver._sparse_rebuildable, expected)


def test_rebuildable_sparse_grid_excludes_inactive_particles(test, device):
    _require_rebuildable_sparse(test)
    model = _make_particle_model(device, [(0.01, 0.01, 0.01), (1000.01, 1000.01, 1000.01)], (1,))
    solver = _make_sparse_solver(model, max_active_cell_count=2)

    test.assertTrue(solver._sparse_rebuildable)
    test.assertEqual(solver._scratchpad.grid.cell_grid.get_active_stats().voxel_count, 1)
    solver.check_sparse_grid_rebuild_status()


def test_rebuildable_sparse_grid_reserves_empty_capacity(test, device):
    _require_rebuildable_sparse(test)
    model = _make_particle_model(device, [(0.01, 0.01, 0.01), (1.01, 1.01, 1.01)], (0, 1))
    solver = _make_sparse_solver(model, max_active_cell_count=4)

    rebuild_info = solver._scratchpad.grid.cell_grid.get_rebuild_info()
    test.assertEqual(rebuild_info.max_voxel_count, 4)
    test.assertEqual(rebuild_info.max_leaf_node_count, 4)


def test_rebuildable_sparse_grid_reserves_explicit_node_capacities(test, device):
    _require_rebuildable_sparse(test)
    model = _make_particle_model(device, [(0.01, 0.01, 0.01), (0.11, 0.01, 0.01)])
    solver = _make_sparse_solver(
        model,
        max_active_cell_count=64,
        max_leaf_node_count=32,
        max_lower_node_count=16,
        max_upper_node_count=8,
        separate_worlds=False,
    )

    cell_capacity = solver._scratchpad.grid.cell_grid.get_rebuild_info()
    vertex_capacity = solver._scratchpad.grid.vertex_grid.get_rebuild_info()
    test.assertEqual(cell_capacity.max_voxel_count, 64)
    test.assertEqual(cell_capacity.max_leaf_node_count, 32)
    test.assertEqual(cell_capacity.max_lower_node_count, 16)
    test.assertEqual(cell_capacity.max_upper_node_count, 8)
    test.assertEqual(vertex_capacity.max_voxel_count, 64 * 8)
    test.assertEqual(vertex_capacity.max_leaf_node_count, 32 * 8)
    test.assertEqual(vertex_capacity.max_lower_node_count, 16 * 8)
    test.assertEqual(vertex_capacity.max_upper_node_count, 8 * 8)


def test_rebuildable_sparse_grid_upper_override_promotes_automatic_lower_capacity(test, device):
    _require_rebuildable_sparse(test)
    model = _make_particle_model(device, [(0.01, 0.01, 0.01)])
    solver = _make_sparse_solver(
        model,
        max_active_cell_count=64,
        voxel_size=0.03,
        max_upper_node_count=32,
    )

    capacity = solver._scratchpad.grid.cell_grid.get_rebuild_info()
    test.assertEqual(capacity.max_voxel_count, 64)
    test.assertEqual(capacity.max_leaf_node_count, 64)
    test.assertEqual(capacity.max_lower_node_count, 32)
    test.assertEqual(capacity.max_upper_node_count, 32)


def test_rebuildable_sparse_grid_explicit_upper_capacity_handles_spread(test, device):
    _require_rebuildable_sparse(test)
    voxel_size = 0.1
    lower_node_width = 8 * 16
    upper_node_width = lower_node_width * 32
    particle_count = 20
    initial_positions = [
        ((particle_index * lower_node_width + 1) * voxel_size, 0.01, 0.01) for particle_index in range(particle_count)
    ]
    model = _make_particle_model(device, initial_positions)
    automatic_solver = _make_sparse_solver(model, max_active_cell_count=32)
    explicit_solver = _make_sparse_solver(
        model,
        max_active_cell_count=32,
        max_leaf_node_count=32,
        max_lower_node_count=32,
        max_upper_node_count=32,
    )
    spread_positions = wp.array(
        [
            ((particle_index * upper_node_width + 1) * voxel_size, 0.01, 0.01)
            for particle_index in range(particle_count)
        ],
        dtype=wp.vec3,
        device=device,
    )

    for solver in (automatic_solver, explicit_solver):
        point_mask = solver._update_grid_point_mask(solver._mpm_model.particle_flags)
        solver._scratchpad.grid.rebuild(
            spread_positions,
            status=solver._grid_status,
            point_mask=point_mask,
        )

    with test.assertRaisesRegex(RuntimeError, "max_upper_node_count"):
        automatic_solver.check_sparse_grid_rebuild_status()
    automatic_status = int(automatic_solver._grid_status.numpy()[0])
    test.assertEqual(automatic_status, wp.Volume.REBUILD_UPPER_CAPACITY_EXCEEDED)

    explicit_solver.check_sparse_grid_rebuild_status()
    explicit_stats = explicit_solver._scratchpad.grid.cell_grid.get_active_stats()
    test.assertEqual(explicit_stats.voxel_count, particle_count)
    test.assertEqual(explicit_stats.upper_node_count, particle_count)


def test_rebuildable_sparse_grid_rejects_invalid_node_capacities(test, device):
    _require_rebuildable_sparse(test)
    model = _make_particle_model(device, [(0.01, 0.01, 0.01)])
    cases = (
        ("max_leaf_node_count", 0),
        ("max_lower_node_count", -2),
        ("max_upper_node_count", 1 << 32),
    )
    for field_name, value in cases:
        with test.subTest(field=field_name, value=value):
            with test.assertRaisesRegex(ValueError, field_name):
                _make_sparse_solver(model, max_active_cell_count=4, **{field_name: value})


def test_rebuildable_sparse_grid_accepts_independent_explicit_node_capacities(test, device):
    _require_rebuildable_sparse(test)
    model = _make_particle_model(device, [(0.01, 0.01, 0.01)])
    solver = _make_sparse_solver(
        model,
        max_active_cell_count=8,
        max_leaf_node_count=5,
        max_lower_node_count=6,
        max_upper_node_count=7,
    )

    capacity = solver._scratchpad.grid.cell_grid.get_rebuild_info()
    test.assertEqual(capacity.max_voxel_count, 8)
    test.assertEqual(capacity.max_leaf_node_count, 5)
    test.assertEqual(capacity.max_lower_node_count, 6)
    test.assertEqual(capacity.max_upper_node_count, 7)


def test_rebuildable_sparse_grid_reports_level_specific_capacity_guidance(test, device):
    del device
    status_flags = {
        "REBUILD_VOXEL_CAPACITY_EXCEEDED": 1,
        "REBUILD_LEAF_CAPACITY_EXCEEDED": 2,
        "REBUILD_LOWER_CAPACITY_EXCEEDED": 4,
        "REBUILD_UPPER_CAPACITY_EXCEEDED": 8,
    }
    with mock.patch.multiple(wp.Volume, create=True, **status_flags):
        cases = (
            (status_flags["REBUILD_VOXEL_CAPACITY_EXCEEDED"], "max_active_cell_count"),
            (status_flags["REBUILD_LEAF_CAPACITY_EXCEEDED"], "max_leaf_node_count"),
            (status_flags["REBUILD_LOWER_CAPACITY_EXCEEDED"], "max_lower_node_count"),
            (status_flags["REBUILD_UPPER_CAPACITY_EXCEEDED"], "max_upper_node_count"),
        )
        for status, field_name in cases:
            with test.subTest(status=status):
                test.assertIn(field_name, str(_sparse_grid_rebuild_error(status)))


def test_rebuildable_sparse_grid_reports_initial_overflow(test, device):
    _require_rebuildable_sparse(test)
    model = _make_particle_model(device, [(0.01, 0.01, 0.01), (1.01, 1.01, 1.01)])

    with test.assertRaisesRegex(RuntimeError, "sparse grid rebuild capacity"):
        _make_sparse_solver(model, max_active_cell_count=1)


def test_rebuildable_sparse_s2_cuda_graph(test, device):
    _require_rebuildable_s2(test)
    if not wp.is_mempool_enabled(device):
        test.skipTest("CUDA graph capture requires the Warp memory pool")

    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_particle_grid(
        pos=wp.vec3(0.05, 0.2, 0.05),
        rot=wp.quat_identity(),
        vel=wp.vec3(25.0, 0.0, 0.0),
        dim_x=2,
        dim_y=2,
        dim_z=2,
        cell_x=0.05,
        cell_y=0.05,
        cell_z=0.05,
        mass=1.0,
        jitter=0.0,
    )
    builder.add_ground_plane()
    model = builder.finalize(device=device)

    eager_state_0 = model.state()
    eager_state_1 = model.state()
    eager_solver = _make_sparse_solver(
        model,
        max_active_cell_count=64,
        collider_basis="S2",
        voxel_size=0.03,
        max_upper_node_count=32,
    )
    for _ in range(5):
        eager_solver.step(eager_state_0, eager_state_1, None, None, 0.005)
        eager_state_0, eager_state_1 = eager_state_1, eager_state_0
    eager_positions = eager_state_0.particle_q.numpy()
    eager_velocities = eager_state_0.particle_qd.numpy()

    state_0 = model.state()
    state_1 = model.state()
    solver = _make_sparse_solver(
        model,
        max_active_cell_count=64,
        collider_basis="S2",
        voxel_size=0.03,
        max_upper_node_count=32,
    )

    # Materialize the persistent cell and S2 edge topology before capture.
    solver.step(state_0, state_1, None, None, 0.005)
    state_0, state_1 = state_1, state_0
    grid = solver._scratchpad.grid
    test.assertIsNotNone(grid._edge_grid)
    test.assertEqual(grid.cell_grid.get_rebuild_info().max_upper_node_count, 32)
    cell_grid_id = grid.cell_grid.id
    edge_grid_id = grid.edge_grid.id
    initial_cell_count = grid.cell_grid.get_active_stats().voxel_count
    initial_edge_count = grid.edge_grid.get_active_stats().voxel_count
    initial_cells = {tuple(ijk) for ijk in grid.cell_grid.get_voxels().numpy()[:initial_cell_count]}
    initial_edges = {tuple(ijk) for ijk in grid.edge_grid.get_voxels().numpy()[:initial_edge_count]}

    with wp.ScopedCapture(device=device) as capture:
        solver.step(state_0, state_1, None, None, 0.005)
        solver.step(state_1, state_0, None, None, 0.005)

    for _ in range(2):
        wp.capture_launch(capture.graph)

    solver.check_sparse_grid_rebuild_status()
    test.assertEqual(solver._scratchpad.grid.cell_grid.id, cell_grid_id)
    test.assertEqual(solver._scratchpad.grid.edge_grid.id, edge_grid_id)
    final_cell_count = grid.cell_grid.get_active_stats().voxel_count
    final_edge_count = grid.edge_grid.get_active_stats().voxel_count
    final_cells = {tuple(ijk) for ijk in grid.cell_grid.get_voxels().numpy()[:final_cell_count]}
    final_edges = {tuple(ijk) for ijk in grid.edge_grid.get_voxels().numpy()[:final_edge_count]}
    test.assertNotEqual(final_cells, initial_cells)
    test.assertNotEqual(final_edges, initial_edges)
    test.assertTrue(np.isfinite(state_0.particle_q.numpy()).all())
    test.assertTrue(np.isfinite(state_0.particle_qd.numpy()).all())
    np.testing.assert_allclose(state_0.particle_q.numpy(), eager_positions, rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(state_0.particle_qd.numpy(), eager_velocities, rtol=1.0e-5, atol=1.0e-6)


def test_rebuildable_sparse_cuda_graph_reports_overflow(test, device):
    _require_rebuildable_sparse(test)
    if not wp.is_mempool_enabled(device):
        test.skipTest("CUDA graph capture requires the Warp memory pool")

    model = _make_particle_model(device, [(0.01, 0.01, 0.01), (0.02, 0.02, 0.02)])
    solver = _make_sparse_solver(model, max_active_cell_count=1)
    state_in = model.state()
    state_out = model.state()

    positions = state_in.particle_q.numpy()
    positions[1] = (1.01, 1.01, 1.01)
    state_in.particle_q.assign(positions)

    with wp.ScopedCapture(device=device) as capture:
        solver.step(state_in, state_out, None, None, 0.001)
    wp.capture_launch(capture.graph)

    with test.assertRaisesRegex(RuntimeError, "sparse grid rebuild capacity"):
        solver.check_sparse_grid_rebuild_status()
    status = int(solver._grid_accumulated_status.numpy()[0])
    test.assertTrue(status & wp.Volume.REBUILD_VOXEL_CAPACITY_EXCEEDED)


class TestImplicitMPMRebuildableSparse(unittest.TestCase):
    pass


cuda_devices = get_selected_cuda_test_devices(mode="basic")

add_function_test(
    TestImplicitMPMRebuildableSparse,
    "test_rebuildable_sparse_capability_requires_node_capacity_api",
    test_rebuildable_sparse_capability_requires_node_capacity_api,
    devices=cuda_devices,
    check_output=False,
)
add_function_test(
    TestImplicitMPMRebuildableSparse,
    "test_rebuildable_sparse_s2_capability_gating",
    test_rebuildable_sparse_s2_capability_gating,
    devices=cuda_devices,
    check_output=False,
)
add_function_test(
    TestImplicitMPMRebuildableSparse,
    "test_rebuildable_sparse_grid_excludes_inactive_particles",
    test_rebuildable_sparse_grid_excludes_inactive_particles,
    devices=cuda_devices,
    check_output=False,
)
add_function_test(
    TestImplicitMPMRebuildableSparse,
    "test_rebuildable_sparse_grid_reserves_empty_capacity",
    test_rebuildable_sparse_grid_reserves_empty_capacity,
    devices=cuda_devices,
    check_output=False,
)
add_function_test(
    TestImplicitMPMRebuildableSparse,
    "test_rebuildable_sparse_grid_reserves_explicit_node_capacities",
    test_rebuildable_sparse_grid_reserves_explicit_node_capacities,
    devices=cuda_devices,
    check_output=False,
)
add_function_test(
    TestImplicitMPMRebuildableSparse,
    "test_rebuildable_sparse_grid_upper_override_promotes_automatic_lower_capacity",
    test_rebuildable_sparse_grid_upper_override_promotes_automatic_lower_capacity,
    devices=cuda_devices,
    check_output=False,
)
add_function_test(
    TestImplicitMPMRebuildableSparse,
    "test_rebuildable_sparse_grid_explicit_upper_capacity_handles_spread",
    test_rebuildable_sparse_grid_explicit_upper_capacity_handles_spread,
    devices=cuda_devices,
    check_output=False,
)
add_function_test(
    TestImplicitMPMRebuildableSparse,
    "test_rebuildable_sparse_grid_rejects_invalid_node_capacities",
    test_rebuildable_sparse_grid_rejects_invalid_node_capacities,
    devices=cuda_devices,
    check_output=False,
)
add_function_test(
    TestImplicitMPMRebuildableSparse,
    "test_rebuildable_sparse_grid_accepts_independent_explicit_node_capacities",
    test_rebuildable_sparse_grid_accepts_independent_explicit_node_capacities,
    devices=cuda_devices,
    check_output=False,
)
add_function_test(
    TestImplicitMPMRebuildableSparse,
    "test_rebuildable_sparse_grid_reports_level_specific_capacity_guidance",
    test_rebuildable_sparse_grid_reports_level_specific_capacity_guidance,
    devices=cuda_devices,
    check_output=False,
)
add_function_test(
    TestImplicitMPMRebuildableSparse,
    "test_rebuildable_sparse_grid_reports_initial_overflow",
    test_rebuildable_sparse_grid_reports_initial_overflow,
    devices=cuda_devices,
    check_output=False,
)
add_function_test(
    TestImplicitMPMRebuildableSparse,
    "test_rebuildable_sparse_s2_cuda_graph",
    test_rebuildable_sparse_s2_cuda_graph,
    devices=cuda_devices,
    check_output=False,
)
add_function_test(
    TestImplicitMPMRebuildableSparse,
    "test_rebuildable_sparse_cuda_graph_reports_overflow",
    test_rebuildable_sparse_cuda_graph_reports_overflow,
    devices=cuda_devices,
    check_output=False,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
