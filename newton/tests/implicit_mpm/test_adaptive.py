# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test resolution changes and swept particle projection."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverImplicitMPM
from newton.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices, get_test_devices


def _make_solver(device, *, moving_wall=False, wall_positions=(0.0,), grid_type=None, integration_scheme="pic"):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_particle((0.5, 0.0, 0.0), (0.0, 0.0, 0.0), mass=0.01, radius=0.025)
    for x in wall_positions:
        body = builder.add_body(is_kinematic=True) if moving_wall else -1
        builder.add_shape_box(
            body,
            xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
            hx=0.01,
            hy=1.0,
            hz=1.0,
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.0, margin=0.0),
        )
    model = builder.finalize(device=device)
    config = SolverImplicitMPM.Config(
        voxel_size=0.1,
        grid_type=grid_type or ("sparse" if device.is_cuda else "dense"),
        grid_padding=0 if device.is_cuda and grid_type != "fixed" else 1,
        max_active_cell_count=64 if device.is_cuda else -1,
        max_iterations=1,
        warmstart_mode="particles",
        collider_basis="Q1",
        integration_scheme=integration_scheme,
    )
    return model, SolverImplicitMPM(model, config, verbose=False)


def test_swept_projection_thin_wall(test, device):
    """Stop a particle whose complete trajectory crosses a thin wall."""
    model, solver = _make_solver(device)
    previous, current = model.state(), model.state()
    previous.particle_q.fill_(wp.vec3(0.5, 0.0, 0.0))
    current.particle_q.fill_(wp.vec3(-0.5, 0.0, 0.0))
    current.particle_qd.fill_(wp.vec3(-10.0, 0.0, 0.0))

    # Endpoint-only projection cannot see the wall after the particle crosses it.
    solver.project_outside(current, current, 0.1)
    test.assertLess(current.particle_q.numpy()[0, 0], -0.4)

    solver.project_outside(current, current, 0.1, state_prev=previous)
    test.assertGreaterEqual(current.particle_q.numpy()[0, 0], 0.009)
    test.assertAlmostEqual(float(current.particle_qd.numpy()[0, 0]), 0.0, places=5)

    # Static meshes need swept queries even when no rigid-body pose arrays exist.
    solver.set_fidelity("particles", state=current)
    model.particle_max_velocity = 100.0
    for _ in range(2):
        current.particle_q.fill_(wp.vec3(0.5, 0.0, 0.0))
        current.particle_qd.fill_(wp.vec3(-10.0, 0.0, 0.0))
        solver.step(current, current, None, None, 0.1)
        test.assertGreaterEqual(float(current.particle_q.numpy()[0, 0]), 0.0339)
        test.assertAlmostEqual(float(current.particle_qd.numpy()[0, 0]), 0.0, places=5)


def test_swept_projection_first_wall(test, device):
    """Select the earliest collision independently of collider ordering."""
    model, solver = _make_solver(device, wall_positions=(0.0, 0.2))
    previous, current = model.state(), model.state()
    current.particle_q.fill_(wp.vec3(-0.5, 0.0, 0.0))
    current.particle_qd.fill_(wp.vec3(-10.0, 0.0, 0.0))
    solver.project_outside(current, current, 0.1, state_prev=previous)
    test.assertGreaterEqual(current.particle_q.numpy()[0, 0], 0.209)


def test_swept_projection_corner(test, device):
    """Stop the remaining slide from crossing a second thin wall at a corner."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_particle((0.5, 0.5, 0.0), (0.0, 0.0, 0.0), mass=0.01, radius=0.025)
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.0, margin=0.0)
    builder.add_shape_box(-1, hx=0.01, hy=1.0, hz=1.0, cfg=cfg)
    builder.add_shape_box(-1, hx=1.0, hy=0.01, hz=1.0, cfg=cfg)
    model = builder.finalize(device=device)
    solver = SolverImplicitMPM(
        model,
        SolverImplicitMPM.Config(
            voxel_size=0.1,
            grid_type="sparse" if device.is_cuda else "dense",
            grid_padding=0 if device.is_cuda else 1,
            max_active_cell_count=64 if device.is_cuda else -1,
            collider_basis="Q1",
            max_iterations=1,
        ),
    )
    previous, current = model.state(), model.state()
    current.particle_q.fill_(wp.vec3(-0.5, -0.8, 0.0))
    current.particle_qd.fill_(wp.vec3(-10.0, -13.0, 0.0))
    solver.project_outside(current, current, 0.1, state_prev=previous)
    position = current.particle_q.numpy()[0]
    test.assertGreaterEqual(position[0], 0.009)
    test.assertGreaterEqual(position[1], 0.009)


def test_swept_projection_moving_wall(test, device):
    """Catch a translating wall that passes completely through a stationary particle."""
    model, solver = _make_solver(device, moving_wall=True)
    previous, current = model.state(), model.state()
    previous.particle_q.zero_()
    current.particle_q.zero_()
    previous.body_q.fill_(wp.transform(wp.vec3(-0.25, 0.0, 0.0), wp.quat_identity()))
    current.body_q.fill_(wp.transform(wp.vec3(0.25, 0.0, 0.0), wp.quat_identity()))
    solver.project_outside(current, current, 0.1, state_prev=previous)
    test.assertGreaterEqual(current.particle_q.numpy()[0, 0], 0.259)
    test.assertAlmostEqual(float(current.particle_qd.numpy()[0, 0]), 5.0, places=4)


def test_swept_projection_inactive(test, device):
    """Preserve inactive and zero-mass particles during swept projection."""
    model, solver = _make_solver(device)
    previous, current = model.state(), model.state()
    current.particle_q.fill_(wp.vec3(-0.5, 0.0, 0.0))
    current.particle_qd.fill_(wp.vec3(-10.0, 0.0, 0.0))
    for inactive in (True, False):
        if inactive:
            model.particle_flags.zero_()
        else:
            model.particle_flags.fill_(int(newton.ParticleFlags.ACTIVE))
            model.particle_mass.zero_()
        solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
        solver.project_outside(current, current, 0.1, state_prev=previous)
        test.assertAlmostEqual(float(current.particle_q.numpy()[0, 0]), -0.5)
        test.assertAlmostEqual(float(current.particle_qd.numpy()[0, 0]), -10.0)


def test_resize_preserves_particle_state(test, device):
    """Rebuild PIC and GIMP grids without resetting particle history."""
    # GIMP also initializes overlap quadrature and its contact/history buffers.
    for integration_scheme in ("pic", "gimp"):
        with test.subTest(integration_scheme=integration_scheme):
            model, solver = _make_solver(device, integration_scheme=integration_scheme)
            current, next_state = model.state(), model.state()
            current.particle_q.fill_(wp.vec3(0.55, 0.02, 0.03))
            current.particle_qd.fill_(wp.vec3(0.1, 0.2, 0.3))
            current.mpm.particle_stress.fill_(wp.mat33(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1))
            current.mpm.particle_Jp.fill_(0.9)
            arrays = [current.particle_q, current.particle_qd]
            arrays.extend(value for value in vars(current.mpm).values() if isinstance(value, wp.array))
            snapshots = [array.numpy().copy() for array in arrays]
            old_grid = solver._scratchpad.grid

            solver.set_voxel_size(0.05, state=current)

            test.assertEqual(solver.voxel_size, 0.05)
            test.assertIsNot(solver._scratchpad.grid, old_grid)
            for array, snapshot in zip(arrays, snapshots, strict=True):
                np.testing.assert_array_equal(array.numpy(), snapshot)
            solver.step(current, next_state, None, None, 0.001)
            solver.check_sparse_grid_rebuild_status()
            test.assertTrue(np.isfinite(next_state.particle_q.numpy()).all())
            test.assertGreater(next_state.particle_q.numpy()[0, 0], 0.5)


def test_resize_validation(test, device):
    """Reject invalid resolution changes before changing solver resources."""
    model, solver = _make_solver(device)
    current = model.state()
    grid = solver._scratchpad.grid
    for value in (0.0, -0.1, np.inf, np.nan):
        with test.subTest(value=value), test.assertRaises(ValueError):
            solver.set_voxel_size(value, state=current)
        test.assertEqual(solver.voxel_size, 0.1)
        test.assertIs(solver._scratchpad.grid, grid)
    solver.set_voxel_size(0.1, state=current)
    test.assertIs(solver._scratchpad.grid, grid)


def test_resize_fixed_grid(test, device):
    """Reject fixed-grid resizing with a clear error."""
    model, solver = _make_solver(device, grid_type="fixed")
    with test.assertRaisesRegex(ValueError, "fixed"):
        solver.set_voxel_size(0.05, state=model.state())


def test_swept_projection_world_isolation(test, device):
    """Apply swept projection only to colliders belonging to the particle's world."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    for world in range(2):
        builder.begin_world()
        builder.add_particle((0.5, 0.0, 0.0), (0.0, 0.0, 0.0), mass=0.01, radius=0.025)
        if world == 0:
            builder.add_shape_box(-1, hx=0.01, hy=1.0, hz=1.0)
        builder.end_world()
    model = builder.finalize(device=device)
    solver = SolverImplicitMPM(
        model,
        SolverImplicitMPM.Config(
            voxel_size=0.1,
            grid_type="sparse" if device.is_cuda else "dense",
            grid_padding=0 if device.is_cuda else 1,
            max_active_cell_count=64 if device.is_cuda else -1,
            collider_basis="Q1",
            max_iterations=1,
            separate_worlds=True,
        ),
    )
    previous, current = model.state(), model.state()
    current.particle_q.fill_(wp.vec3(-0.5, 0.0, 0.0))
    solver.project_outside(current, current, 0.1, state_prev=previous)
    positions = current.particle_q.numpy()
    test.assertGreater(positions[0, 0], 0.0)
    test.assertAlmostEqual(float(positions[1, 0]), -0.5)


def test_resize_capacity_rollback(test, device):
    """Keep the original grid and particle state when a finer grid exceeds capacity."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    for i in range(8):
        builder.add_particle((0.01 + i * 0.1, 0.01, 0.01), (0.0, 0.0, 0.0), mass=0.01, radius=0.025)
    model = builder.finalize(device=device)
    solver = SolverImplicitMPM(
        model,
        SolverImplicitMPM.Config(voxel_size=1.0, max_active_cell_count=4, collider_basis="Q1", max_iterations=1),
    )
    state = model.state()
    positions = state.particle_q.numpy().copy()
    grid = solver._scratchpad.grid
    with test.assertRaisesRegex(RuntimeError, "capacity"):
        solver.set_voxel_size(0.01, state=state)
    test.assertEqual(solver.voxel_size, 1.0)
    test.assertIs(solver._scratchpad.grid, grid)
    solver.check_sparse_grid_rebuild_status()
    np.testing.assert_array_equal(state.particle_q.numpy(), positions)
    solver.step(state, state, None, None, 0.001)
    test.assertTrue(np.isfinite(state.particle_q.numpy()).all())


def test_resize_capture_rejection(test, device):
    """Reject resizing during capture before altering solver state."""
    model, solver = _make_solver(device)
    state = model.state()
    grid = solver._scratchpad.grid
    with wp.ScopedCapture(device=device):
        with test.assertRaisesRegex(RuntimeError, "capture"):
            solver.set_voxel_size(0.05, state=state)
    test.assertEqual(solver.voxel_size, 0.1)
    test.assertIs(solver._scratchpad.grid, grid)


def test_bowl_graph_and_runtime_changes(test, device):
    """Preserve live state across graph recapture, fidelity switches, and radius changes."""
    from newton.examples.mpm.example_mpm_bowl import Example  # noqa: PLC0415
    from newton.viewer import ViewerNull  # noqa: PLC0415

    with wp.ScopedDevice(device):
        args = Example.create_parser().parse_args(
            ["--solver", "particles", "--particle-radius", "0.03", "--iterations", "3"]
        )
        example = Example(ViewerNull(), args)
        for _ in range(3):
            example.step()
        before = example.state_0.particle_q.numpy().copy()
        velocities = example.state_0.particle_qd.numpy().copy()
        masses = example.model.particle_mass.numpy().copy()
        example.set_solver_type("mpm")
        np.testing.assert_array_equal(example.state_0.particle_q.numpy(), before)
        np.testing.assert_array_equal(example.state_0.particle_qd.numpy(), velocities)
        example.set_particle_radius(0.028)
        np.testing.assert_array_equal(example.state_0.particle_q.numpy(), before)
        np.testing.assert_array_equal(example.state_0.particle_qd.numpy(), velocities)
        np.testing.assert_array_equal(example.model.particle_mass.numpy(), masses)
        test.assertAlmostEqual(example.model.particle_max_radius, 0.028)
        test.assertAlmostEqual(float(example.sim_clock.numpy()[0]), example.sim_time, places=6)
        example.step()
        example.test_post_step()
        example.mpm_solver.check_sparse_grid_rebuild_status()

        args = Example.create_parser().parse_args(
            ["--solver", "particles", "--particle-radius", "0.03", "--no-use-cuda-graph"]
        )
        uncaptured = Example(ViewerNull(), args)
        args.use_cuda_graph = True
        captured = Example(ViewerNull(), args)
        # Warp 1.17 captures a shared host HashGrid descriptor: replay one grid
        # before building another so this checks a single-scene rollout.
        for _ in range(4):
            captured.step()
        for _ in range(4):
            uncaptured.step()
        np.testing.assert_allclose(captured.state_0.body_q.numpy(), uncaptured.state_0.body_q.numpy(), atol=1e-6)
        np.testing.assert_allclose(
            captured.state_0.particle_q.numpy(), uncaptured.state_0.particle_q.numpy(), atol=1e-5
        )


def test_fidelity_handoff(test, device):
    """Preserve particle motion across native fidelity changes and reset material history."""
    model, solver = _make_solver(device, moving_wall=True)
    state = model.state()
    state.particle_qd.fill_(wp.vec3(0.2, 0.0, 0.0))
    state.mpm.particle_stress.fill_(wp.mat33(3.0))
    position = state.particle_q.numpy().copy()
    velocity = state.particle_qd.numpy().copy()
    solver.set_fidelity("particles", state=state)
    test.assertEqual(solver.fidelity, SolverImplicitMPM.Fidelity.PARTICLES)
    np.testing.assert_array_equal(state.particle_q.numpy(), position)
    np.testing.assert_array_equal(state.particle_qd.numpy(), velocity)
    np.testing.assert_array_equal(state.mpm.particle_stress.numpy(), 0.0)
    solver.step(state, state, None, None, 0.01)
    np.testing.assert_allclose(state.particle_q.numpy(), position + 0.01 * velocity, atol=1.0e-6)

    # A rejected external collider must leave the original fast solver usable.
    external_builder = newton.ModelBuilder()
    external_builder.add_body(is_kinematic=True)
    external_body = external_builder.add_body(is_kinematic=True)
    external_builder.add_shape_box(external_body, hx=0.01, hy=1.0, hz=1.0)
    external = external_builder.finalize(device=device)
    collider_ids = solver.collider_body_index.numpy().copy()
    previous_poses = solver._last_step_data.body_q_prev.numpy().copy()
    with test.assertRaisesRegex(ValueError, "collider bodies to belong"):
        solver.setup_collider(model=external)
    np.testing.assert_array_equal(solver.collider_body_index.numpy(), collider_ids)
    np.testing.assert_array_equal(solver._last_step_data.body_q_prev.numpy(), previous_poses)

    # Static external geometry still needs the owning model's body COM layout
    # because forward pose prediction visits every body in that model.
    static_builder = newton.ModelBuilder()
    static_builder.add_shape_box(-1, hx=0.01, hy=1.0, hz=1.0)
    static_model = static_builder.finalize(device=device)
    with test.assertRaisesRegex(ValueError, "center of mass for every model body"):
        solver.setup_collider(model=static_model)
    np.testing.assert_array_equal(solver.collider_body_index.numpy(), collider_ids)
    np.testing.assert_array_equal(solver._last_step_data.body_q_prev.numpy(), previous_poses)
    solver.step(state, state, None, None, 0.01)
    test.assertTrue(np.isfinite(state.particle_q.numpy()).all())
    advanced = state.particle_q.numpy().copy()
    solver.set_fidelity("mpm", state=state)
    test.assertEqual(solver.fidelity, SolverImplicitMPM.Fidelity.MPM)
    np.testing.assert_array_equal(state.particle_q.numpy(), advanced)
    solver.step(state, state, None, None, 0.01)
    test.assertTrue(np.isfinite(state.particle_q.numpy()).all())


def test_particle_fidelity_lazy_grid(test, device):
    """Defer grid allocation until MPM is enabled and preserve mass during runtime size changes."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_particle((0.5, 0.0, 0.0), (0.2, 0.0, 0.0), mass=0.01, radius=0.025)
    model = builder.finalize(device=device)
    solver = SolverImplicitMPM(
        model,
        SolverImplicitMPM.Config(fidelity="particles", grid_type="dense", grid_padding=1, max_iterations=1),
        verbose=False,
    )
    test.assertIsNone(solver._scratchpad)
    state = model.state()
    masses = model.particle_mass.numpy().copy()
    solver.step(state, state, None, None, 0.01)
    test.assertIsNone(solver._scratchpad)
    solver.set_voxel_size(0.2, state=state)
    test.assertIsNone(solver._scratchpad)
    solver.set_particle_radius(0.02)
    np.testing.assert_array_equal(model.particle_mass.numpy(), masses)
    np.testing.assert_allclose(model.particle_radius.numpy(), 0.02)
    test.assertAlmostEqual(model.particle_max_radius, 0.02)
    position = state.particle_q.numpy().copy()
    solver.set_fidelity(SolverImplicitMPM.Fidelity.MPM, state=state)
    test.assertIsNotNone(solver._scratchpad)
    np.testing.assert_array_equal(state.particle_q.numpy(), position)
    test.assertAlmostEqual(solver.voxel_size, 0.2)
    solver.step(state, state, None, None, 0.01)
    test.assertTrue(np.isfinite(state.particle_q.numpy()).all())


class TestImplicitMPMAdaptive(unittest.TestCase):
    pass


for _device in get_test_devices():
    for _test in (
        test_swept_projection_thin_wall,
        test_swept_projection_first_wall,
        test_swept_projection_corner,
        test_swept_projection_moving_wall,
        test_swept_projection_inactive,
        test_resize_preserves_particle_state,
        test_resize_validation,
        test_resize_fixed_grid,
        test_swept_projection_world_isolation,
        test_fidelity_handoff,
        test_particle_fidelity_lazy_grid,
    ):
        add_function_test(TestImplicitMPMAdaptive, _test.__name__, _test, devices=[_device])

for _test in (test_resize_capacity_rollback, test_resize_capture_rejection, test_bowl_graph_and_runtime_changes):
    add_function_test(TestImplicitMPMAdaptive, _test.__name__, _test, devices=get_selected_cuda_test_devices())


if __name__ == "__main__":
    unittest.main()
