# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for coupled multi-world implicit MPM."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverImplicitMPM, SolverXPBD
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledProxy
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


def _make_triangle_mesh(device) -> wp.Mesh:
    points = wp.array(
        ((-0.05, 0.0, -0.05), (0.05, 0.0, -0.05), (0.0, 0.0, 0.05)),
        dtype=wp.vec3,
        device=device,
    )
    indices = wp.array((0, 1, 2), dtype=wp.int32, device=device)
    return wp.Mesh(points=points, indices=indices, velocities=wp.zeros_like(points))


def _make_mpm_config() -> SolverImplicitMPM.Config:
    config = SolverImplicitMPM.Config()
    config.grid_type = "fixed"
    config.grid_padding = 1
    config.max_iterations = 1
    config.solver = "jacobi"
    config.transfer_scheme = "pic"
    config.warmstart_mode = "none"
    return config


def _make_two_world_particle_model(device) -> newton.Model:
    world_builder = newton.ModelBuilder(gravity=0.0)
    SolverImplicitMPM.register_custom_attributes(world_builder)
    for pos in ((-0.05, 0.0, -0.05), (0.05, 0.0, -0.05), (0.0, 0.0, 0.05)):
        world_builder.add_particle(pos=pos, vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.025)

    builder = newton.ModelBuilder(gravity=0.0)
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_world(world_builder)
    builder.add_world(world_builder)
    return builder.finalize(device=device)


def _assert_collider_unchanged(test, solver, expected_worlds, expected_particle_ids):
    collider = solver._mpm_model.collider
    np.testing.assert_array_equal(collider.collider_world.numpy(), expected_worlds)
    np.testing.assert_array_equal(collider.collider_particle_ids.numpy(), expected_particle_ids)


def test_mismatched_deformable_collider_particle_world_rejected(test, device):
    model = _make_two_world_particle_model(device)
    solver = SolverImplicitMPM(model, config=_make_mpm_config())
    collider = solver._mpm_model.collider
    initial_worlds = collider.collider_world.numpy().copy()
    initial_particle_ids = collider.collider_particle_ids.numpy().copy()
    world_starts = model.particle_world_start.numpy()
    world_0_particle_ids = list(range(world_starts[0], world_starts[1]))

    with test.assertRaisesRegex(
        ValueError,
        r"collider_particle_ids\[0\].*collider world 1.*particle world IDs \[0\]",
    ):
        solver.setup_collider(
            collider_meshes=[_make_triangle_mesh(device)],
            collider_particle_ids=[world_0_particle_ids],
            collider_world_ids=[1],
        )

    _assert_collider_unchanged(test, solver, initial_worlds, initial_particle_ids)


def test_global_deformable_collider_rejected(test, device):
    model = _make_two_world_particle_model(device)
    solver = SolverImplicitMPM(model, config=_make_mpm_config())
    collider = solver._mpm_model.collider
    initial_worlds = collider.collider_world.numpy().copy()
    initial_particle_ids = collider.collider_particle_ids.numpy().copy()
    world_starts = model.particle_world_start.numpy()
    world_0_particle_ids = list(range(world_starts[0], world_starts[1]))

    with test.assertRaisesRegex(
        ValueError,
        r"collider_particle_ids\[0\].*global deformable collider.*isolated worlds",
    ):
        solver.setup_collider(
            collider_meshes=[_make_triangle_mesh(device)],
            collider_particle_ids=[world_0_particle_ids],
            collider_world_ids=[-1],
        )

    _assert_collider_unchanged(test, solver, initial_worlds, initial_particle_ids)


def test_external_deformable_collider_particle_mapping_rejected(test, device):
    model = _make_two_world_particle_model(device)
    external_model = _make_two_world_particle_model(device)
    solver = SolverImplicitMPM(model, config=_make_mpm_config())
    collider = solver._mpm_model.collider
    initial_worlds = collider.collider_world.numpy().copy()
    initial_particle_ids = collider.collider_particle_ids.numpy().copy()
    world_starts = model.particle_world_start.numpy()
    world_0_particle_ids = list(range(world_starts[0], world_starts[1]))

    with test.assertRaisesRegex(ValueError, r"collider_particle_ids.*solver model"):
        solver.setup_collider(
            collider_meshes=[_make_triangle_mesh(device)],
            collider_particle_ids=[world_0_particle_ids],
            collider_world_ids=[0],
            model=external_model,
        )

    _assert_collider_unchanged(test, solver, initial_worlds, initial_particle_ids)


def test_coupled_multiworld_isolation(test, device):
    config = _make_mpm_config()
    test.assertTrue(config.separate_worlds)

    world_builder = newton.ModelBuilder(gravity=0.0)
    SolverImplicitMPM.register_custom_attributes(world_builder)

    # Three deformable-collider proxies, one transfer-only proxy, and one MPM
    # material particle. Replication keeps the two worlds spatially colocated.
    for pos in (
        (-0.05, 0.0, -0.05),
        (0.05, 0.0, -0.05),
        (0.0, 0.0, 0.05),
        (0.0, 0.1, 0.0),
        (0.0, 0.2, 0.0),
    ):
        world_builder.add_particle(pos=pos, vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.025)

    dynamic_body = world_builder.add_body(
        xform=wp.transform((0.0, -0.1, 0.0), wp.quat_identity()),
        inertia=wp.mat33(np.eye(3)),
        mass=1.0,
        lock_inertia=True,
    )
    world_builder.add_shape_box(dynamic_body, hx=0.2, hy=0.05, hz=0.2)

    builder = newton.ModelBuilder(gravity=0.0)
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_world(world_builder)
    builder.add_world(world_builder)
    model = builder.finalize(device=device)

    particle_starts = model.particle_world_start.numpy()
    body_starts = model.body_world_start.numpy()
    shape_starts = model.shape_world_start.numpy()
    collider_proxy_ids = [
        *range(particle_starts[0], particle_starts[0] + 3),
        *range(particle_starts[1], particle_starts[1] + 3),
    ]
    transfer_proxy_ids = [particle_starts[0] + 3, particle_starts[1] + 3]
    material_particle_ids = [particle_starts[0] + 4, particle_starts[1] + 4]
    proxy_particle_ids = collider_proxy_ids + transfer_proxy_ids
    collider_body_ids = [int(body_starts[0]), int(body_starts[1])]
    collider_shape_ids = [int(shape_starts[0]), int(shape_starts[1])]

    coupled = SolverCoupledProxy(
        model=model,
        entries=(
            SolverCoupled.Entry(name="xpbd", solver=SolverXPBD, particles=proxy_particle_ids),
            SolverCoupled.Entry(
                name="mpm",
                solver=lambda view: SolverImplicitMPM(view, config=config),
                bodies=collider_body_ids,
                particles=material_particle_ids,
                shapes=collider_shape_ids,
            ),
        ),
        coupling=SolverCoupledProxy.Config(
            proxies=(
                SolverCoupledProxy.Proxy(
                    source="xpbd",
                    destination="mpm",
                    particles=proxy_particle_ids,
                ),
            )
        ),
    )

    mpm_solver = coupled.solver("mpm")
    mpm_model = mpm_solver._mpm_model
    collider = mpm_model.collider
    expected_worlds = np.array([0, 1], dtype=np.int32)
    expected_body_ids = np.array(collider_body_ids, dtype=np.int32)

    test.assertEqual(mpm_solver._environment_count, 2)
    np.testing.assert_array_equal(collider.collider_world.numpy(), expected_worlds)
    np.testing.assert_array_equal(collider.collider_body_index.numpy(), expected_body_ids)
    test.assertTrue(np.all(mpm_model.collider_body_mass.numpy()[expected_body_ids] > 0.0))
    test.assertGreater(mpm_model.min_collider_mass, 0.0)

    triangle_meshes = [_make_triangle_mesh(device), _make_triangle_mesh(device)]
    deformable_ids_by_world = [collider_proxy_ids[:3], collider_proxy_ids[3:]]
    mpm_solver.setup_collider(
        collider_meshes=triangle_meshes,
        collider_particle_ids=deformable_ids_by_world,
        collider_world_ids=[0, 1],
        model=coupled.view("mpm"),
    )

    active = int(newton.ParticleFlags.ACTIVE)

    np.testing.assert_array_equal(collider.collider_world.numpy(), np.array([0, 1], dtype=np.int32))
    test.assertEqual(collider.query_world_offsets.shape[0], model.world_count + 2)
    np.testing.assert_array_equal(collider.collider_particle_offsets.numpy(), np.array([0, 3, 6], dtype=np.int32))
    np.testing.assert_array_equal(collider.collider_particle_ids.numpy(), np.array(collider_proxy_ids, dtype=np.int32))

    transfer_flags = mpm_model.particle_flags.numpy()
    material_flags = mpm_model.material_particle_flags.numpy()
    for particle_id in collider_proxy_ids:
        test.assertEqual(transfer_flags[particle_id] & active, 0)
        test.assertEqual(material_flags[particle_id] & active, 0)
    for particle_id in transfer_proxy_ids:
        test.assertNotEqual(transfer_flags[particle_id] & active, 0)
        test.assertEqual(material_flags[particle_id] & active, 0)
    for particle_id in material_particle_ids:
        test.assertNotEqual(transfer_flags[particle_id] & active, 0)
        test.assertNotEqual(material_flags[particle_id] & active, 0)


def _make_sparse_capture_config() -> SolverImplicitMPM.Config:
    config = SolverImplicitMPM.Config()
    config.grid_type = "sparse"
    config.voxel_size = 0.1
    config.grid_padding = 0
    config.max_active_cell_count = 128
    config.max_iterations = 5
    config.tolerance = 0.0
    config.solver = "jacobi"
    config.warmstart_mode = "none"
    config.transfer_scheme = "pic"
    config.integration_scheme = "pic"
    config.strain_basis = "P0"
    config.velocity_basis = "Q1"
    config.collider_basis = "S2"
    return config


def _make_sparse_capture_case(device):
    world_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    SolverImplicitMPM.register_custom_attributes(world_builder)
    world_builder.add_particle_grid(
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

    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_world(world_builder)
    builder.add_world(world_builder)
    model = builder.finalize(device=device)

    starts = model.particle_world_start.numpy()
    velocities = np.zeros((model.particle_count, 3), dtype=np.float32)
    velocities[starts[0] : starts[1], 0] = 3.0
    velocities[starts[1] : starts[2], 0] = -2.0
    model.particle_qd.assign(velocities)

    solver = SolverImplicitMPM(model, config=_make_sparse_capture_config(), enable_timers=False)
    return model, solver, model.state(), model.state()


def _require_sparse_capture_prerequisites(test, device):
    if not device.is_cuda:
        test.skipTest("Sparse implicit MPM outer capture requires CUDA.")
    if not device.is_mempool_supported or not wp.is_mempool_enabled(device):
        test.skipTest("Sparse implicit MPM outer capture requires a CUDA memory pool.")
    if not wp.is_conditional_graph_supported():
        test.skipTest("Sparse implicit MPM outer capture requires conditional CUDA graphs.")


def _warm_sparse_solver(model, solver, dt):
    warm_state_0 = model.state()
    warm_state_1 = model.state()
    solver.step(warm_state_0, warm_state_1, control=None, contacts=None, dt=dt)


def _sparse_grid_snapshot(grid):
    active_cell_count = grid.cell_grid.get_active_stats().voxel_count
    cell_ijks = wp.empty(grid.cell_count(), dtype=wp.vec3i, device=grid.cell_env.device)
    grid.cell_grid.get_voxels(out=cell_ijks)
    cell_env = grid.cell_env.numpy()[:active_cell_count]
    env_offsets = grid.env_offsets.numpy().copy()
    packed_cell_ijks = cell_ijks.numpy()[:active_cell_count]
    local_cell_ijks = packed_cell_ijks - env_offsets[cell_env]
    return {
        "cell_env": cell_env,
        "env_offsets": env_offsets,
        "packed_cell_ijks": packed_cell_ijks,
        "local_cell_ijks": local_cell_ijks,
    }


def _sparse_case_state_arrays(state):
    return {
        "particle_q": state.particle_q,
        "particle_qd": state.particle_qd,
        "particle_qd_grad": state.mpm.particle_qd_grad,
        "particle_elastic_strain": state.mpm.particle_elastic_strain,
        "particle_Jp": state.mpm.particle_Jp,
        "particle_stress": state.mpm.particle_stress,
        "particle_transform": state.mpm.particle_transform,
    }


def test_sparse_multiworld_constructs_environment_grid(test, device):
    model, solver, _state_0, _state_1 = _make_sparse_capture_case(device)
    grid = solver._scratchpad.grid

    test.assertEqual(model.world_count, 2)
    test.assertTrue(solver._separate_worlds)
    test.assertTrue(solver._sparse_rebuildable)
    test.assertEqual(grid.environment_count(), 2)
    test.assertTrue(solver.supports_cuda_graph_capture)
    test.assertEqual(solver.max_active_cell_count, 128)


def test_sparse_multiworld_capture_rebuilds_isolated_topology(test, device):
    _require_sparse_capture_prerequisites(test, device)
    model, solver, state_0, state_1 = _make_sparse_capture_case(device)
    dt = 0.05
    _warm_sparse_solver(model, solver, dt)

    grid = solver._scratchpad.grid
    initial = _sparse_grid_snapshot(grid)
    cell_grid_id = grid.cell_grid.id
    edge_grid = grid.edge_grid
    edge_grid_id = edge_grid.id
    test.assertEqual(solver._scratchpad._collision_space.topology._edge_grid, edge_grid_id)

    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        solver.step(state_0, state_1, control=None, contacts=None, dt=dt)
        solver.step(state_1, state_0, control=None, contacts=None, dt=dt)

    wp.capture_launch(capture.graph)
    solver.check_sparse_grid_rebuild_status()
    rebuilt = _sparse_grid_snapshot(grid)

    test.assertEqual(int(solver._grid_status.numpy()[0]), wp.Volume.REBUILD_SUCCESS)
    test.assertEqual(grid.cell_grid.id, cell_grid_id)
    test.assertIs(grid.edge_grid, edge_grid)
    test.assertEqual(grid.edge_grid.id, edge_grid_id)
    test.assertEqual(solver._scratchpad._collision_space.topology._edge_grid, edge_grid_id)
    test.assertEqual(grid.environment_count(), 2)
    test.assertEqual(set(rebuilt["cell_env"].tolist()), {0, 1})
    test.assertFalse(np.array_equal(rebuilt["env_offsets"], initial["env_offsets"]))

    packed_by_environment = []
    for environment in range(2):
        initial_local = initial["local_cell_ijks"][initial["cell_env"] == environment]
        rebuilt_local = rebuilt["local_cell_ijks"][rebuilt["cell_env"] == environment]
        test.assertGreater(initial_local.shape[0], 0)
        test.assertGreater(rebuilt_local.shape[0], 0)
        packed_by_environment.append(
            {tuple(cell) for cell in rebuilt["packed_cell_ijks"][rebuilt["cell_env"] == environment].tolist()}
        )

    test.assertGreater(
        np.mean(rebuilt["local_cell_ijks"][rebuilt["cell_env"] == 0, 0]),
        np.mean(initial["local_cell_ijks"][initial["cell_env"] == 0, 0]) + 0.5,
    )
    test.assertLess(
        np.mean(rebuilt["local_cell_ijks"][rebuilt["cell_env"] == 1, 0]),
        np.mean(initial["local_cell_ijks"][initial["cell_env"] == 1, 0]) - 0.5,
    )
    test.assertTrue(packed_by_environment[0].isdisjoint(packed_by_environment[1]))


def test_sparse_multiworld_outer_capture_matches_eager(test, device):
    _require_sparse_capture_prerequisites(test, device)
    eager_model, eager_solver, eager_state_0, eager_state_1 = _make_sparse_capture_case(device)
    captured_model, captured_solver, captured_state_0, captured_state_1 = _make_sparse_capture_case(device)
    dt = 0.02
    _warm_sparse_solver(eager_model, eager_solver, dt)
    _warm_sparse_solver(captured_model, captured_solver, dt)

    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        captured_solver.step(captured_state_0, captured_state_1, control=None, contacts=None, dt=dt)
        captured_solver.step(captured_state_1, captured_state_0, control=None, contacts=None, dt=dt)

    for cycle in range(3):
        eager_solver.step(eager_state_0, eager_state_1, control=None, contacts=None, dt=dt)
        eager_solver.step(eager_state_1, eager_state_0, control=None, contacts=None, dt=dt)
        wp.capture_launch(capture.graph)
        captured_solver.check_sparse_grid_rebuild_status()

        test.assertEqual(int(captured_solver._grid_status.numpy()[0]), wp.Volume.REBUILD_SUCCESS)
        eager_arrays = _sparse_case_state_arrays(eager_state_0)
        captured_arrays = _sparse_case_state_arrays(captured_state_0)
        for name, eager_array in eager_arrays.items():
            eager_values = eager_array.numpy()
            captured_values = captured_arrays[name].numpy()
            test.assertTrue(np.isfinite(eager_values).all(), f"{name} is non-finite after eager cycle {cycle}")
            test.assertTrue(np.isfinite(captured_values).all(), f"{name} is non-finite after capture cycle {cycle}")
            np.testing.assert_allclose(
                captured_values,
                eager_values,
                rtol=1.0e-5,
                atol=1.0e-6,
                equal_nan=False,
                err_msg=f"{name} differs after capture replay cycle {cycle}",
            )


class TestImplicitMPMMultiworldSparse(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
add_function_test(
    TestImplicitMPMMultiworldSparse,
    "test_mismatched_deformable_collider_particle_world_rejected",
    test_mismatched_deformable_collider_particle_world_rejected,
    devices=devices,
)
add_function_test(
    TestImplicitMPMMultiworldSparse,
    "test_global_deformable_collider_rejected",
    test_global_deformable_collider_rejected,
    devices=devices,
)
add_function_test(
    TestImplicitMPMMultiworldSparse,
    "test_external_deformable_collider_particle_mapping_rejected",
    test_external_deformable_collider_particle_mapping_rejected,
    devices=devices,
)
add_function_test(
    TestImplicitMPMMultiworldSparse,
    "test_coupled_multiworld_isolation",
    test_coupled_multiworld_isolation,
    devices=devices,
)
add_function_test(
    TestImplicitMPMMultiworldSparse,
    "test_sparse_multiworld_constructs_environment_grid",
    test_sparse_multiworld_constructs_environment_grid,
    devices=devices,
)
add_function_test(
    TestImplicitMPMMultiworldSparse,
    "test_sparse_multiworld_capture_rebuilds_isolated_topology",
    test_sparse_multiworld_capture_rebuilds_isolated_topology,
    devices=devices,
)
add_function_test(
    TestImplicitMPMMultiworldSparse,
    "test_sparse_multiworld_outer_capture_matches_eager",
    test_sparse_multiworld_outer_capture_matches_eager,
    devices=devices,
)
