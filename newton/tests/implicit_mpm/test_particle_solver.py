# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test particle-only integration and rigid-collider reaction impulses."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.implicit_mpm.particle_solver import _ParticleSolver
from newton._src.solvers.implicit_mpm.particle_solver_xpbd import _XPBDParticleSolver
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

    direction = np.asarray((3.0, 1.0, 2.0)) / np.sqrt(14.0)
    for masses in ((1.0, 4.0), (1.0, 0.0)):
        with test.subTest(oblique_masses=masses):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            SolverImplicitMPM.register_custom_attributes(builder)
            builder.add_particles(
                pos=[(0.0, 0.0, 0.0), tuple(0.09 * direction)],
                vel=[(0.0, 0.0, 0.0)] * 2,
                mass=masses,
                radius=[0.05] * 2,
            )
            model = builder.finalize(device=device)
            state = model.state()
            initial_q = state.particle_q.numpy().copy()
            config = SolverImplicitMPM.Config(fidelity="particles", particle_iterations=1, particle_friction=0.0)
            solver = SolverImplicitMPM(model, config, verbose=False)
            solver.step(state, state, None, None, 0.01)
            # A single contact must retain its inverse-mass split at any orientation,
            # including a pinned endpoint with a maximally unequal inverse-mass ratio.
            inverse_mass = np.asarray([1.0 / mass if mass else 0.0 for mass in masses])
            displacement = (
                config.particle_relaxation
                * 0.01
                * np.asarray((-inverse_mass[0], inverse_mass[1]))[:, None]
                * direction
                / inverse_mass.sum()
            )
            np.testing.assert_allclose(state.particle_q.numpy(), initial_q + displacement, atol=1.0e-7)
            np.testing.assert_allclose(state.particle_qd.numpy(), displacement / 0.01, atol=2.0e-6)
            if all(masses):
                momentum = np.sum(np.asarray(masses)[:, None] * state.particle_qd.numpy(), axis=0)
                np.testing.assert_allclose(momentum, 0.0, atol=2.0e-6)


def test_particle_redundant_contacts(test, device):
    """Repair redundant pinned contacts without ejecting the moving particle."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_particles(
        pos=[(0.0, 0.0, 0.0)] + [(1.8, 0.0, 0.0)] * 8,
        vel=[(0.0, 0.0, 0.0)] * 9,
        mass=[1.0] + [0.0] * 8,
        radius=[1.0] * 9,
    )
    model = builder.finalize(device=device)
    dt = 0.01
    for iterations in (2, 4, 8):
        with test.subTest(iterations=iterations):
            state = model.state()
            initial_q = state.particle_q.numpy().copy()
            solver = _ParticleSolver(model, iterations=iterations, friction=0.0, cohesion=0.0)
            solver.step(state, state, dt)
            q, qd = state.particle_q.numpy(), state.particle_qd.numpy()
            # These eight constraints describe the same 0.2 m overlap repair.
            test.assertGreaterEqual(float(q[0, 0]), -0.200001)
            test.assertLessEqual(float(q[0, 0]), -0.19)
            test.assertLessEqual(float(np.linalg.norm(qd[0])), 0.2 / dt + 1.0e-3)
            np.testing.assert_array_equal(q[1:], initial_q[1:])
            np.testing.assert_array_equal(qd[1:], 0.0)


def test_particle_isolated_contact_friction(test, device):
    """Preserve isolated oblique friction for free and pinned particle pairs."""
    normal = np.asarray((3.0, 1.0, 2.0)) / np.sqrt(14.0)
    tangent = np.asarray((-1.0, 3.0, 0.0)) / np.sqrt(10.0)
    dt = 0.01
    velocity = 0.1 * tangent
    for masses in ((1.0, 4.0), (1.0, 0.0)):
        with test.subTest(masses=masses):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            SolverImplicitMPM.register_custom_attributes(builder)
            builder.add_particles(
                pos=[tuple(-dt * velocity), tuple(0.09 * normal)],
                vel=[tuple(velocity), (0.0, 0.0, 0.0)],
                mass=masses,
                radius=[0.05] * 2,
            )
            model = builder.finalize(device=device)
            state = model.state()
            initial_momentum = np.sum(np.asarray(masses)[:, None] * state.particle_qd.numpy(), axis=0)
            config = SolverImplicitMPM.Config(fidelity="particles", particle_iterations=1, particle_friction=0.5)
            solver = SolverImplicitMPM(model, config, verbose=False)
            solver.step(state, state, None, None, dt)
            qd = state.particle_qd.numpy()
            # Prediction makes the 1 cm contact overlap normal to the relative
            # velocity; its friction budget is sufficient for sticking.
            expected_relative_velocity = (
                -config.particle_relaxation * normal + (1.0 - config.particle_relaxation) * velocity
            )
            np.testing.assert_allclose(qd[0] - qd[1], expected_relative_velocity, atol=3.0e-6)
            if all(masses):
                np.testing.assert_allclose(
                    np.sum(np.asarray(masses)[:, None] * qd, axis=0), initial_momentum, atol=2.0e-6
                )
            else:
                np.testing.assert_array_equal(qd[1], 0.0)


def test_particle_unequal_mass_contact_momentum(test, device):
    """Conserve free-particle momentum when contact degrees and masses differ."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_particles(
        pos=[(0.0, 0.0, 0.0), (1.2, 0.1, 0.0), (2.9, 0.3, 0.0)],
        vel=[(0.0, 0.0, 0.0)] * 3,
        mass=[1.0, 2.0, 4.0],
        radius=[1.0] * 3,
    )
    model = builder.finalize(device=device)
    state = model.state()
    mass = model.particle_mass.numpy().astype(np.float64)[:, None]
    initial_mass_position = np.sum(mass * state.particle_q.numpy(), axis=0)
    initial_momentum = np.sum(mass * state.particle_qd.numpy(), axis=0)
    # The middle particle contacts both ends, which do not contact each other.
    solver = _ParticleSolver(model, iterations=1, friction=0.0, cohesion=0.0)
    solver.step(state, state, 0.01)
    np.testing.assert_allclose(np.sum(mass * state.particle_q.numpy(), axis=0), initial_mass_position, atol=1.0e-6)
    np.testing.assert_allclose(np.sum(mass * state.particle_qd.numpy(), axis=0), initial_momentum, atol=2.0e-5)


def test_particle_contact_neighbor_updates(test, device):
    """Conserve momentum when contact corrections move particles between grid cells."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_particles(
        pos=[(x, 0.0, 0.0) for x in (0.0, 0.001, 0.002, 0.004)],
        vel=[(0.0, 0.0, 0.0)] * 4,
        mass=[0.001] * 4,
        radius=[0.005] * 4,
    )
    model = builder.finalize(device=device)
    mass = model.particle_mass.numpy().astype(np.float64)[:, None]
    for iterations in (2, 4, 8):
        with test.subTest(iterations=iterations):
            state = model.state()
            initial_com = np.sum(mass * state.particle_q.numpy(), axis=0) / mass.sum()
            solver = _ParticleSolver(model, iterations=iterations, friction=0.0, cohesion=0.0)
            # The initial overlaps deliberately require corrections spanning hash cells.
            solver.step(state, state, 0.002)
            com = np.sum(mass * state.particle_q.numpy(), axis=0) / mass.sum()
            momentum = np.sum(mass * state.particle_qd.numpy(), axis=0)
            np.testing.assert_allclose(com, initial_com, atol=1.0e-7)
            np.testing.assert_allclose(momentum, 0.0, atol=1.0e-8)


def test_particle_friction_dissipates_transverse_energy(test, device):
    """Dissipate tangential kinetic energy in a planar sticking-contact graph."""
    dt = 0.002
    points = np.asarray([(x * 0.009, y * 0.009, 0.0) for x in range(4) for y in range(4)], dtype=np.float32)
    velocities = np.zeros_like(points)
    velocities[:, 2] = [0.001 * (-1) ** (x + y) for x in range(4) for y in range(4)]
    # Ballistic prediction puts every contact normal in the xy plane, so normal
    # overlap repair cannot contribute to the measured transverse kinetic energy.
    points[:, 2] = -velocities[:, 2] * dt
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_particles(
        pos=points.tolist(),
        vel=velocities.tolist(),
        mass=[1.0] * len(points),
        radius=[0.005] * len(points),
    )
    model = builder.finalize(device=device)
    mass = model.particle_mass.numpy().astype(np.float64)
    for friction in (0.0, 0.5):
        with test.subTest(friction=friction):
            state = model.state()
            initial_velocity = state.particle_qd.numpy().copy()
            initial_energy = 0.5 * np.sum(mass * initial_velocity[:, 2] ** 2)
            initial_momentum = np.sum(mass[:, None] * initial_velocity, axis=0)
            solver = _ParticleSolver(model, iterations=1, friction=friction, cohesion=0.0)
            solver.step(state, state, dt)
            velocity = state.particle_qd.numpy()
            energy = 0.5 * np.sum(mass * velocity[:, 2] ** 2)
            test.assertLessEqual(float(energy), float(initial_energy) * (1.0 + 1.0e-5) + 1.0e-12)
            np.testing.assert_allclose(np.sum(mass[:, None] * velocity, axis=0), initial_momentum, atol=1.0e-6)
            if friction == 0.0:
                np.testing.assert_allclose(velocity[:, 2], initial_velocity[:, 2], atol=1.0e-8)


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


def test_particle_xpbd_compliance(test, device):
    """Converge a compliant pair to its analytical equilibrium across iteration counts."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_particles(
        pos=[(0.0, 0.0, 0.0), (0.09, 0.0, 0.0)],
        vel=[(0.0, 0.0, 0.0)] * 2,
        mass=[1.0, 4.0],
        radius=[0.05] * 2,
    )
    model = builder.finalize(device=device)
    dt, compliance = 0.01, 1.25e-4
    expected_lambda = 0.01 / (1.25 + compliance / dt**2)
    expected_q = np.asarray([[-expected_lambda, 0.0, 0.0], [0.09 + 0.25 * expected_lambda, 0.0, 0.0]])
    for iterations in (1, 2, 4, 8):
        with test.subTest(iterations=iterations):
            state = model.state()
            solver = _XPBDParticleSolver(
                model, iterations=iterations, compliance=compliance, relaxation=1.0, friction=0.0
            )
            solver.step(state, state, dt)
            np.testing.assert_allclose(state.particle_q.numpy(), expected_q, atol=2.0e-8)
            np.testing.assert_allclose(solver.normal_lambda.numpy()[0, 0], expected_lambda, atol=1.0e-8)
            q = state.particle_q.numpy()
            constraint = float(q[1, 0] - q[0, 0] - 0.1)
            test.assertAlmostEqual(constraint + compliance / dt**2 * expected_lambda, 0.0, delta=2.0e-8)
            test.assertEqual(solver.check_contact_capacity()["overflow"], 0)
    # Under-relaxation must also store exactly the correction that was applied.
    state = model.state()
    solver = _XPBDParticleSolver(model, iterations=1, compliance=compliance, relaxation=0.4, friction=0.0)
    solver.step(state, state, dt)
    np.testing.assert_allclose(
        state.particle_q.numpy(),
        np.asarray([[0.0, 0.0, 0.0], [0.09, 0.0, 0.0]])
        + 0.4 * (expected_q - np.asarray([[0.0, 0.0, 0.0], [0.09, 0.0, 0.0]])),
        atol=2.0e-8,
    )
    np.testing.assert_allclose(solver.normal_lambda.numpy()[0, 0], 0.4 * expected_lambda, atol=1.0e-8)


def test_particle_xpbd_friction_and_dense_contacts(test, device):
    """Bound accumulated friction and dense contact corrections while conserving momentum."""
    dt = 0.002
    points = np.asarray([(x * 0.009, y * 0.009, 0.0) for x in range(4) for y in range(4)], dtype=np.float32)
    velocities = np.zeros_like(points)
    velocities[:, 2] = [0.001 * (-1) ** (x + y) for x in range(4) for y in range(4)]
    points[:, 2] = -velocities[:, 2] * dt
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_particles(
        pos=points.tolist(), vel=velocities.tolist(), mass=[1.0] * len(points), radius=[0.005] * len(points)
    )
    model = builder.finalize(device=device)
    initial_energy = 0.5 * np.sum(velocities[:, 2] ** 2)
    for iterations in (2, 4, 8):
        with test.subTest(iterations=iterations):
            state = model.state()
            solver = _XPBDParticleSolver(model, iterations=iterations, compliance=2.0e-6, friction=0.5)
            solver.step(state, state, dt)
            q, qd = state.particle_q.numpy(), state.particle_qd.numpy()
            test.assertTrue(np.isfinite(q).all())
            test.assertLess(float(np.max(np.linalg.norm(q - points, axis=1))), 0.005)
            test.assertLessEqual(float(0.5 * np.sum(qd[:, 2] ** 2)), float(initial_energy) + 1.0e-11)
            np.testing.assert_allclose(np.sum(qd, axis=0), velocities.sum(axis=0), atol=2.0e-7)
            np.testing.assert_allclose(np.sum(q, axis=0), (points + dt * velocities).sum(axis=0), atol=2.0e-7)
            normal = solver.normal_lambda.numpy()
            tangent = solver.tangent_lambda.numpy()
            test.assertTrue(np.all(np.linalg.norm(tangent, axis=2) <= 0.5 * normal + 1.0e-10))
            solver.check_contact_capacity()

    # Mixed 3D modes are needed because the planar case separates normal and
    # tangent motion. Opposite velocity perturbations cancel the overlap-repair
    # energy to measure their average amplification about the resting case.
    points = np.asarray(
        [(x * 0.009, y * 0.009, z * 0.009) for x in range(3) for y in range(3) for z in range(3)], dtype=np.float32
    )
    velocities = np.asarray(
        [
            (0.001 * (-1) ** (x + y), 0.0007 * (-1) ** (y + z), 0.0005 * (-1) ** (x + z))
            for x in range(3)
            for y in range(3)
            for z in range(3)
        ],
        dtype=np.float32,
    )
    velocities -= velocities.mean(axis=0)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_particles(
        pos=points.tolist(), vel=velocities.tolist(), mass=[1.0] * len(points), radius=[0.005] * len(points)
    )
    model = builder.finalize(device=device)
    initial_energy = 0.5 * np.sum(velocities.astype(np.float64) ** 2)
    for iterations in (2, 4, 8):
        energies = []
        for sign in (0.0, 1.0, -1.0):
            with test.subTest(mixed_iterations=iterations, velocity_sign=sign):
                state = model.state()
                state.particle_q.assign(points - sign * dt * velocities)
                state.particle_qd.assign(sign * velocities)
                solver = _XPBDParticleSolver(model, iterations=iterations, compliance=2.0e-6, friction=0.5)
                solver.step(state, state, dt)
                q, qd = state.particle_q.numpy(), state.particle_qd.numpy().astype(np.float64)
                test.assertTrue(np.isfinite(q).all() and np.isfinite(qd).all())
                test.assertLess(float(np.max(np.linalg.norm(q - points, axis=1))), 0.005)
                np.testing.assert_allclose(qd.sum(axis=0), sign * velocities.sum(axis=0), atol=5.0e-7)
                np.testing.assert_allclose(q.sum(axis=0), points.sum(axis=0), atol=5.0e-7)
                energies.append(float(0.5 * np.sum(qd**2)))
                solver.check_contact_capacity()
        perturbation_energy = 0.5 * (energies[1] + energies[2]) - energies[0]
        test.assertLessEqual(perturbation_energy, 1.1 * initial_energy)


def test_particle_xpbd_contact_lifecycle(test, device):
    """Expose capacity overflow and reset pair multipliers once per native step."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_particles(
        pos=[(0.0, 0.0, 0.0), (0.09, 0.0, 0.0), (-0.09, 0.0, 0.0)],
        vel=[(0.0, 0.0, 0.0)] * 3,
        mass=[1.0, 0.0, 0.0],
        radius=[0.05] * 3,
    )
    model = builder.finalize(device=device)
    state = model.state()
    solver = _XPBDParticleSolver(model, iterations=2, compliance=1.0e-4, contact_capacity=1, friction=0.0)
    solver.step(state, state, 0.01)
    with test.assertRaisesRegex(RuntimeError, "capacity exceeded"):
        solver.check_contact_capacity()
    test.assertGreater(int(solver.contact_max_count.numpy()[0]), 1)
    solver = _XPBDParticleSolver(model, iterations=2, compliance=1.0e-4, friction=0.0)
    state = model.state()
    solver.step(state, state, 0.01)
    test.assertGreater(float(solver.normal_lambda.numpy().sum()), 0.0)
    np.testing.assert_array_equal(state.particle_q.numpy()[1:], model.particle_q.numpy()[1:])
    separated = np.asarray([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)], dtype=np.float32)
    state.particle_q.assign(separated)
    state.particle_qd.zero_()
    solver.step(state, state, 0.01)
    np.testing.assert_array_equal(solver.normal_lambda.numpy(), 0.0)
    np.testing.assert_array_equal(solver.tangent_lambda.numpy(), 0.0)
    np.testing.assert_array_equal(state.particle_q.numpy(), separated)
    solver.check_contact_capacity()
    model.particle_max_velocity = 0.01
    state.particle_q.assign(np.asarray([(0.0, 0.0, 0.0), (0.09, 0.0, 0.0), (-1.0, 0.0, 0.0)], dtype=np.float32))
    solver.step(state, state, 0.01)
    with test.assertRaisesRegex(RuntimeError, "velocity cap"):
        solver.check_contact_capacity()


def test_particle_xpbd_multiplier_release(test, device):
    """Apply full cached normal release and rotated tangent multiplier differences."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_particles(
        pos=[(0.0, 0.0, 0.0), (0.13, 0.0, 0.0)], vel=[(0.0, 0.0, 0.0)] * 2, mass=[1.0, 1.0], radius=[0.05, 0.05]
    )
    model = builder.finalize(device=device)
    solver = _XPBDParticleSolver(model, iterations=1, compliance=0.0, relaxation=1.0, friction=0.5)
    solver._begin_particle_contacts()
    solver.contact_ids.assign(np.asarray([[1] + [-1] * 63, [-1] * 64], dtype=np.int32))
    solver.contact_count.assign(np.asarray([1, 0], dtype=np.int32))
    normal = np.zeros((2, 64), dtype=np.float32)
    tangent = np.zeros((2, 64, 3), dtype=np.float32)
    normal[0, 0] = 0.01
    tangent[0, 0] = (0.0, 0.004, 0.0)
    solver.normal_lambda.assign(normal)
    solver.tangent_lambda.assign(tangent)
    solver._positions.assign(model.particle_q)
    solver._velocities.zero_()
    solver._compute_particle_contact_deltas(model.particle_flags, 0.5, 0.0, 0.01)
    # This separated pair is absent from discovery but its retained lambda must release.
    np.testing.assert_allclose(solver._deltas.numpy(), [(0.01, -0.004, 0.0), (-0.01, 0.004, 0.0)], atol=1.0e-8)
    np.testing.assert_array_equal(solver.normal_lambda.numpy(), 0.0)
    np.testing.assert_array_equal(solver.tangent_lambda.numpy(), 0.0)

    direction = np.asarray([1.0, 1.0, 0.0], dtype=np.float32) / np.sqrt(2.0)
    solver._positions.assign(np.asarray([[0.0, 0.0, 0.0], 0.09 * direction], dtype=np.float32))
    normal[0, 0] = 0.005
    tangent[0, 0] = (0.0, 0.001, 0.0)
    solver.normal_lambda.assign(normal)
    solver.tangent_lambda.assign(tangent)
    solver._compute_particle_contact_deltas(model.particle_flags, 0.5, 0.0, 0.01)
    projected = np.asarray([-0.0005, 0.0005, 0.0])
    expected = -0.005 * direction + projected - tangent[0, 0]
    np.testing.assert_allclose(solver._deltas.numpy(), [expected, -expected], atol=1.0e-8)
    np.testing.assert_allclose(solver.tangent_lambda.numpy()[0, 0], projected, atol=1.0e-9)


class TestParticleSolver(unittest.TestCase):
    """Verify particle-only solver kernels on available devices."""


for _test in (
    test_particle_integration_state_ownership,
    test_particle_contact_relaxation,
    test_particle_xpbd_compliance,
    test_particle_xpbd_friction_and_dense_contacts,
    test_particle_xpbd_contact_lifecycle,
    test_particle_xpbd_multiplier_release,
    test_particle_redundant_contacts,
    test_particle_isolated_contact_friction,
    test_particle_unequal_mass_contact_momentum,
    test_particle_contact_neighbor_updates,
    test_particle_friction_dissipates_transverse_energy,
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
