# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp
import warp.fem as fem

import newton
from newton._src.solvers.implicit_mpm.implicit_mpm_solver_kernels import supports_rebuildable_nanogrid
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


def _make_sparse_solver(model, max_active_cell_count, collider_basis="Q1"):
    config = SolverImplicitMPM.Config(
        grid_type="sparse",
        voxel_size=0.1,
        max_active_cell_count=max_active_cell_count,
        velocity_basis="Q1",
        strain_basis="P0",
        collider_basis=collider_basis,
        max_iterations=2,
        warmstart_mode="none",
    )
    return SolverImplicitMPM(model, config, verbose=False)


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
    eager_solver = _make_sparse_solver(model, max_active_cell_count=64, collider_basis="S2")
    for _ in range(5):
        eager_solver.step(eager_state_0, eager_state_1, None, None, 0.005)
        eager_state_0, eager_state_1 = eager_state_1, eager_state_0
    eager_positions = eager_state_0.particle_q.numpy()
    eager_velocities = eager_state_0.particle_qd.numpy()

    state_0 = model.state()
    state_1 = model.state()
    solver = _make_sparse_solver(model, max_active_cell_count=64, collider_basis="S2")

    # Materialize the persistent cell and S2 edge topology before capture.
    solver.step(state_0, state_1, None, None, 0.005)
    state_0, state_1 = state_1, state_0
    grid = solver._scratchpad.grid
    test.assertIsNotNone(grid._edge_grid)
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
