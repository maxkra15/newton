# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test particle-only integration and rigid-collider reaction impulses."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.implicit_mpm.particle_solver import _ParticleSolver
from newton.solvers import SolverImplicitMPM
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_particle_integration_state_ownership(test, device):
    """Preserve input and material history while supporting in-place particle stepping."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_particles(
        pos=[(0.0, 0.0, 1.0), (0.09, 0.0, 1.0), (1.0, 0.0, 1.0)],
        vel=[(0.0, 0.0, 0.0)] * 3,
        mass=[1.0] * 3,
        radius=[0.05] * 3,
    )
    builder.add_body(is_kinematic=True)
    model = builder.finalize(device=device)
    model.particle_flags.assign(wp.array([1, 1, 0], dtype=wp.int32, device=device))
    state_in, state_out, state_in_place = (model.state() for _ in range(3))
    stress = wp.mat33(2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0)
    for state in (state_in, state_in_place):
        state.mpm.particle_stress.fill_(stress)
        state.mpm.particle_Jp.fill_(0.8)
    initial_q = state_in.particle_q.numpy().copy()
    initial_qd = state_in.particle_qd.numpy().copy()
    state_out.body_q.fill_(wp.transform(wp.vec3(7.0, 8.0, 9.0), wp.quat_identity()))
    solver = _ParticleSolver(model, iterations=2)
    solver.step(state_in, state_out, 0.01)
    np.testing.assert_array_equal(state_in.particle_q.numpy(), initial_q)
    np.testing.assert_array_equal(state_in.particle_qd.numpy(), initial_qd)
    np.testing.assert_allclose(state_out.particle_q.numpy()[:2, 2], 1.0 - 9.81e-4)
    np.testing.assert_array_equal(state_out.particle_q.numpy()[2], initial_q[2])
    test.assertGreater(float(np.diff(state_out.particle_q.numpy()[:2, 0])[0]), 0.099)
    np.testing.assert_array_equal(state_out.mpm.particle_stress.numpy(), state_in.mpm.particle_stress.numpy())
    np.testing.assert_array_equal(state_out.mpm.particle_Jp.numpy(), state_in.mpm.particle_Jp.numpy())
    np.testing.assert_array_equal(state_out.body_q.numpy()[0, :3], (7.0, 8.0, 9.0))
    solver.step(state_in_place, state_in_place, 0.01)
    np.testing.assert_array_equal(state_in_place.particle_q.numpy(), state_out.particle_q.numpy())
    np.testing.assert_array_equal(state_in_place.particle_qd.numpy(), state_out.particle_qd.numpy())
    if device.is_cuda:
        wp.copy(state_in_place.particle_q, state_in.particle_q)
        wp.copy(state_in_place.particle_qd, state_in.particle_qd)
        with wp.ScopedCapture(device=device) as capture:
            solver.step(state_in_place, state_in_place, 0.01)
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(state_in_place.particle_q.numpy(), state_out.particle_q.numpy(), atol=1.0e-6)
        np.testing.assert_allclose(state_in_place.particle_qd.numpy(), state_out.particle_qd.numpy(), atol=1.0e-6)
        solver.step(state_out, state_out, 0.01)
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(state_in_place.particle_q.numpy(), state_out.particle_q.numpy(), atol=1.0e-6)
        np.testing.assert_allclose(state_in_place.particle_qd.numpy(), state_out.particle_qd.numpy(), atol=1.0e-6)


def test_particle_contact_relaxation(test, device):
    """Scale pair corrections without changing the historical default relaxation."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_particles(
        pos=[(0.0, 0.0, 0.0), (0.09, 0.0, 0.0)],
        vel=[(0.0, 0.0, 0.0)] * 2,
        mass=[1.0] * 2,
        radius=[0.05] * 2,
    )
    model = builder.finalize(device=device)
    for value in (0.0, -0.1, 1.01, float("nan"), float("inf"), -float("inf")):
        with test.subTest(invalid_relaxation=value), test.assertRaisesRegex(ValueError, "particle_relaxation"):
            SolverImplicitMPM(model, SolverImplicitMPM.Config(particle_relaxation=value), verbose=False)
    for value in (0.5, 0.9, 1.0):
        with test.subTest(relaxation=value):
            solver = SolverImplicitMPM(
                model,
                SolverImplicitMPM.Config(
                    fidelity="particles", particle_iterations=1, particle_relaxation=value, particle_friction=0.0
                ),
                verbose=False,
            )
            state = model.state()
            solver.step(state, state, None, None, 0.01)
            # Each equally weighted particle receives half of the 1 cm overlap.
            expected_delta = 0.005 * value
            np.testing.assert_allclose(
                state.particle_q.numpy()[:, 0], (-expected_delta, 0.09 + expected_delta), atol=1.0e-7
            )
            np.testing.assert_allclose(
                state.particle_qd.numpy()[:, 0], (-expected_delta / 0.01, expected_delta / 0.01), atol=1.0e-6
            )


def test_particle_confined_stack_relaxation(test, device):
    """Keep a dense stack bounded when under-relaxing summed local contacts."""
    radius = 0.00075
    spacing = 2.0 * radius
    margin = 0.0005
    width, layers = 4, 30
    inner_halfwidth = width * spacing / 2.0 + margin
    wall_thickness = 0.003
    wall_height = layers * spacing * 1.5
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    SolverImplicitMPM.register_custom_attributes(builder)
    shape_cfg = newton.ModelBuilder.ShapeConfig(mu=0.0, margin=margin)
    builder.add_shape_box(
        -1,
        xform=wp.transform(wp.vec3(0.0, 0.0, -wall_thickness / 2.0), wp.quat_identity()),
        hx=0.2,
        hy=0.2,
        hz=wall_thickness / 2.0,
        cfg=shape_cfg,
    )
    for axis in (0, 1):
        for sign in (-1, 1):
            center = [0.0, 0.0, wall_height / 2.0]
            center[axis] = sign * (inner_halfwidth + wall_thickness / 2.0)
            halfextent = [inner_halfwidth + wall_thickness] * 2 + [wall_height / 2.0]
            halfextent[axis] = wall_thickness / 2.0
            builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(*center), wp.quat_identity()),
                hx=halfextent[0],
                hy=halfextent[1],
                hz=halfextent[2],
                cfg=shape_cfg,
            )
    points = np.stack(
        np.meshgrid(np.arange(width), np.arange(width), np.arange(layers), indexing="ij"), axis=-1
    ).reshape(-1, 3)
    points = (points.astype(np.float32) + 0.5) * spacing
    points[:, :2] -= width * spacing / 2.0
    points[:, 2] += margin + 0.2 * spacing
    points += np.random.default_rng(7).uniform(-0.2, 0.2, points.shape).astype(np.float32) * spacing
    builder.add_particles(
        pos=points.tolist(),
        vel=np.zeros_like(points).tolist(),
        mass=[3.375e-6] * len(points),
        radius=[radius] * len(points),
    )
    model = builder.finalize(device=device)
    solver = SolverImplicitMPM(
        model,
        SolverImplicitMPM.Config(
            fidelity="particles",
            voxel_size=0.003,
            particle_iterations=2,
            particle_relaxation=0.5,
            particle_friction=0.0,
            particle_cohesion=0.0,
        ),
        verbose=False,
    )
    state = model.state()
    peak_speed = 0.0
    for step in range(120):
        solver.step(state, state, None, None, 0.00125)
        if step % 8 == 0 or step == 119:
            peak_speed = max(peak_speed, float(np.linalg.norm(state.particle_qd.numpy(), axis=1).max()))
    q = state.particle_q.numpy()
    test.assertTrue(np.isfinite(q).all())
    test.assertTrue(np.isfinite(state.particle_qd.numpy()).all())
    test.assertLess(peak_speed, 1.25)
    test.assertGreaterEqual(float(q[:, 2].min()), 0.0)
    test.assertLess(float(q[:, 2].max()), wall_height)
    test.assertLess(float(np.abs(q[:, :2]).max()), inner_halfwidth + wall_thickness + radius)


def test_particle_mesh_reactions(test, device):
    """Isolate mesh contacts by world and conserve momentum in proxy wrench harvesting."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.begin_world()
    body = builder.add_body(is_kinematic=True)
    builder.add_shape_box(
        body,
        hx=0.01,
        hy=1.0,
        hz=1.0,
        cfg=newton.ModelBuilder.ShapeConfig(mu=0.5, margin=0.0),
    )
    builder.add_particle((0.06, 0.2, 0.0), (-2.0, 1.0, 0.0), mass=1.0, radius=0.05)
    builder.end_world()
    builder.begin_world()
    builder.add_particle((0.06, 0.2, 0.0), (-2.0, 1.0, 0.0), mass=1.0, radius=0.05)
    builder.end_world()
    model = builder.finalize(device=device)
    model.body_flags.fill_(int(newton.BodyFlags.KINEMATIC | newton.BodyFlags.PROXY))
    owner = SolverImplicitMPM(
        model,
        SolverImplicitMPM.Config(
            grid_type="dense", grid_padding=1, voxel_size=0.1, max_iterations=1, separate_worlds=True
        ),
        verbose=False,
    )
    owner._mpm_model.collider.material_projection_threshold.zero_()
    owner._mpm_model.collider.query_max_dist = 0.001
    solver = _ParticleSolver(model, iterations=2)
    state = model.state()
    initial_velocity = state.particle_qd.numpy().copy()
    solver.step(
        state,
        state,
        0.01,
        collider=owner._mpm_model.collider,
        particle_environment=model.particle_world,
        body_q=state.body_q,
        body_q_prev=state.body_q,
    )
    test.assertGreaterEqual(float(state.particle_q.numpy()[0, 0]), 0.05999)
    np.testing.assert_allclose(state.particle_qd.numpy()[0], (0.0, 0.0, 0.0), atol=2.0e-6)
    np.testing.assert_allclose(state.particle_q.numpy()[1], (0.04, 0.21, 0.0), atol=2.0e-6)
    np.testing.assert_array_equal(state.particle_qd.numpy()[1], initial_velocity[1])
    test.assertEqual(owner._mpm_model.collider.query_max_dist, 0.001)
    impulse = solver.body_impulse.numpy()[0]
    np.testing.assert_allclose(impulse[:3], initial_velocity[0] - state.particle_qd.numpy()[0], atol=2.0e-6)
    np.testing.assert_allclose(impulse[:3], (-2.0, 1.0, 0.0), atol=2.0e-6)
    np.testing.assert_allclose(impulse[3:], (0.0, 0.0, 0.43), atol=2.0e-6)
    mapping = wp.array([0], dtype=wp.int32, device=device)
    forces = wp.empty(1, dtype=wp.spatial_vector, device=device)
    solver.harvest_proxy_wrenches(mapping, forces, 0.01)
    np.testing.assert_allclose(forces.numpy()[0], impulse / 0.01, atol=2.0e-5)

    # Zero new contact velocity means the next substep cannot replay old impulses.
    solver.step(
        state,
        state,
        0.01,
        collider=owner._mpm_model.collider,
        particle_environment=model.particle_world,
        body_q=state.body_q,
        body_q_prev=state.body_q,
    )
    np.testing.assert_allclose(solver.body_impulse.numpy(), 0.0, atol=2.0e-6)


def test_particle_swept_moving_mesh(test, device):
    """Catch a moving wall and report its equal and opposite momentum transfer."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    body = builder.add_body(is_kinematic=True)
    builder.add_shape_box(body, hx=0.01, hy=1.0, hz=1.0, cfg=newton.ModelBuilder.ShapeConfig(mu=0.5, margin=0.0))
    builder.add_particle((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), mass=1.0, radius=0.02)
    builder.add_particle((0.0, 0.2, 0.0), (0.0, 1.0, 0.0), mass=1.0, radius=0.02)
    model = builder.finalize(device=device)
    owner = SolverImplicitMPM(
        model,
        SolverImplicitMPM.Config(grid_type="dense", grid_padding=1, voxel_size=0.1, max_iterations=1),
        verbose=False,
    )
    owner._mpm_model.collider.material_projection_threshold.zero_()
    previous = wp.array([wp.transform(wp.vec3(-0.25, 0.0, 0.0), wp.quat_identity())], dtype=wp.transform, device=device)
    end = wp.array([wp.transform(wp.vec3(0.25, 0.0, 0.0), wp.quat_identity())], dtype=wp.transform, device=device)
    state = model.state()
    wp.copy(state.body_q, previous)
    solver = _ParticleSolver(model, iterations=2)
    solver.step(
        state,
        state,
        0.1,
        collider=owner._mpm_model.collider,
        body_q_prev=previous,
        body_q=end,
        body_q_reference=state.body_q,
    )
    test.assertGreaterEqual(float(state.particle_q.numpy()[0, 0]), 0.27999)
    np.testing.assert_allclose(state.particle_qd.numpy()[0], (5.0, 0.0, 0.0), atol=1.0e-5)
    np.testing.assert_allclose(state.particle_qd.numpy()[1], (5.0, 0.0, 0.0), atol=1.0e-5)
    # The tangential impulse acts at (0.26, 0.248, 0), about the caller's COM at (-0.25, 0, 0).
    np.testing.assert_allclose(solver.body_impulse.numpy()[0], (-10.0, 1.0, 0.0, 0.0, 0.0, 1.75), atol=1.0e-5)


def test_particle_contact_corrections_swept(test, device):
    """Prevent particle contact corrections from crossing a thin static wall."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_shape_box(-1, hx=0.01, hy=1.0, hz=1.0, cfg=newton.ModelBuilder.ShapeConfig(mu=0.0, margin=0.0))
    for x in (0.06, 0.061, 0.062, 0.063):
        builder.add_particle((x, 0.0, 0.0), (0.0, 0.0, 0.0), mass=1.0, radius=0.05)
    model = builder.finalize(device=device)
    model.particle_max_velocity = 1000.0
    solver = SolverImplicitMPM(
        model,
        SolverImplicitMPM.Config(fidelity="particles", grid_type="dense", particle_iterations=2),
        verbose=False,
    )
    state = model.state()
    solver.step(state, state, None, None, 0.01)
    test.assertTrue(np.isfinite(state.particle_qd.numpy()).all())
    test.assertGreaterEqual(float(state.particle_q.numpy()[:, 0].min()), 0.05899)


def test_particle_thin_wall_clearance(test, device):
    """Keep resting particles outside either face of a thin closed wall."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_shape_box(
        -1,
        hx=0.00025,
        hy=1.0,
        hz=1.0,
        cfg=newton.ModelBuilder.ShapeConfig(mu=0.0, margin=0.0005),
    )
    builder.add_particles(
        pos=[(-0.002, 0.01, 0.02), (0.002, 0.01, 0.02)],
        vel=[(0.0, 0.0, 0.0)] * 2,
        mass=[1.0] * 2,
        radius=[0.00075] * 2,
    )
    model = builder.finalize(device=device)
    solver = SolverImplicitMPM(
        model,
        SolverImplicitMPM.Config(fidelity="particles", voxel_size=0.003, particle_iterations=2),
        verbose=False,
    )
    state = model.state()
    initial_q = state.particle_q.numpy().copy()
    solver.step(state, state, None, None, 0.00125)
    np.testing.assert_allclose(state.particle_q.numpy(), initial_q, atol=1.0e-7, rtol=0.0)
    np.testing.assert_array_equal(state.particle_qd.numpy(), np.zeros((2, 3), dtype=np.float32))


def test_particle_surface_contact_normal(test, device):
    """Project a particle resting exactly on a plane along its surface normal."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    mesh = newton.Mesh(
        vertices=[(-0.1, -0.1, 0.0), (0.1, -0.1, 0.0), (0.1, 0.1, 0.0), (-0.1, 0.1, 0.0)],
        indices=[0, 1, 2, 0, 2, 3],
    )
    builder.add_shape_mesh(-1, mesh=mesh, cfg=newton.ModelBuilder.ShapeConfig(mu=0.0, margin=0.0))
    builder.add_particle((0.01, 0.02, 0.0), (0.0, 0.0, 0.0), mass=1.0, radius=0.00075)
    model = builder.finalize(device=device)
    solver = SolverImplicitMPM(
        model,
        SolverImplicitMPM.Config(fidelity="particles", voxel_size=0.003, particle_iterations=2),
        verbose=False,
    )
    solver._mpm_model.collider.material_projection_threshold.zero_()
    state = model.state()
    solver.step(state, state, None, None, 0.00125)
    np.testing.assert_allclose(state.particle_q.numpy()[0], (0.01, 0.02, 0.00075), atol=1.0e-7, rtol=0.0)
    np.testing.assert_array_equal(state.particle_qd.numpy(), np.zeros((1, 3), dtype=np.float32))


def test_particle_reaction_masked_reset(test, device):
    """Preserve unselected proxy reactions during masked and captured resets."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    for world in range(2):
        builder.begin_world()
        body = builder.add_body(is_kinematic=True)
        builder.add_shape_box(body, hx=0.01, hy=1.0, hz=1.0, cfg=newton.ModelBuilder.ShapeConfig(mu=0.5, margin=0.0))
        builder.add_particle((0.06, 0.2, 0.0), (-2.0, 1.0, 0.0), mass=1.0, radius=0.05)
        if world == 0:
            builder.add_particle((1.06, 0.2, 0.0), (-2.0, 1.0, 0.0), mass=1.0, radius=0.05)
        builder.end_world()
    global_body = builder.add_body(xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()), is_kinematic=True)
    builder.add_shape_box(global_body, hx=0.01, hy=1.0, hz=1.0, cfg=newton.ModelBuilder.ShapeConfig(mu=0.5, margin=0.0))
    model = builder.finalize(device=device)
    model.body_flags.fill_(int(newton.BodyFlags.KINEMATIC | newton.BodyFlags.PROXY))
    solver = SolverImplicitMPM(
        model,
        SolverImplicitMPM.Config(fidelity="particles", grid_type="dense", separate_worlds=True),
        verbose=False,
    )
    mapping = wp.array([0, 1, 2], dtype=int, device=device)
    forces = wp.empty(3, dtype=wp.spatial_vector, device=device)
    body_world = model.body_world.numpy()
    mask = wp.empty(3, dtype=wp.bool, device=device)
    for selected in ([True, False, False], [False, True, False], [False, False, True], [False, False, False], None):
        with test.subTest(selected=selected):
            state = model.state()
            solver.step(state, state, None, None, 0.01)
            before = solver._particle_solver.body_impulse.numpy()
            test.assertTrue(np.all(np.linalg.norm(before[:, :3], axis=1) > 1.0))
            expected = before.copy()
            if selected is None:
                expected.fill(0.0)
                world_mask = None
            else:
                mask.assign(np.asarray(selected, dtype=bool))
                expected[np.asarray(selected)[np.where(body_world < 0, 2, body_world)]] = 0.0
                world_mask = mask
            solver.reset(state, world_mask=world_mask, flags=newton.StateFlags.BODY_Q)
            np.testing.assert_array_equal(solver._particle_solver.body_impulse.numpy(), expected)
            solver._particle_solver.harvest_proxy_wrenches(mapping, forces, 0.01)
            np.testing.assert_allclose(forces.numpy(), expected / 0.01, atol=1.0e-5)

    if device.is_cuda:
        state = model.state()
        solver.step(state, state, None, None, 0.01)
        before = solver._particle_solver.body_impulse.numpy()
        mask.assign(np.asarray([True, False, False], dtype=bool))
        with wp.ScopedCapture(device=device) as capture:
            solver.reset(state, world_mask=mask, flags=newton.StateFlags.BODY_Q)
        wp.capture_launch(capture.graph)
        expected = before.copy()
        expected[0] = 0.0
        np.testing.assert_array_equal(solver._particle_solver.body_impulse.numpy(), expected)
        next_state = model.state()
        solver.step(next_state, next_state, None, None, 0.01)
        before = solver._particle_solver.body_impulse.numpy()
        mask.assign(np.asarray([False, False, True], dtype=bool))
        wp.capture_launch(capture.graph)
        expected = before.copy()
        expected[2] = 0.0
        np.testing.assert_array_equal(solver._particle_solver.body_impulse.numpy(), expected)


class TestParticleSolver(unittest.TestCase):
    """Verify particle-only solver kernels on available devices."""


for _test in (
    test_particle_integration_state_ownership,
    test_particle_contact_relaxation,
    test_particle_confined_stack_relaxation,
    test_particle_mesh_reactions,
    test_particle_swept_moving_mesh,
    test_particle_contact_corrections_swept,
    test_particle_thin_wall_clearance,
    test_particle_surface_contact_normal,
    test_particle_reaction_masked_reset,
):
    add_function_test(TestParticleSolver, _test.__name__, _test, devices=get_test_devices(), check_output=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
