# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Velocity-level recovery of particle penetration into MPM colliders."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.implicit_mpm.rasterized_collisions import normalize_gradient
from newton.solvers import SolverImplicitMPM
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _make_contact_model(device, *, body_mass=0.0):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    body = -1
    if body_mass > 0.0:
        body = builder.add_body(mass=body_mass, inertia=wp.mat33(0.001, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.001))
    builder.add_shape_box(
        body,
        xform=wp.transform(wp.vec3(0.0, 0.0, -0.02), wp.quat_identity()),
        hx=0.12,
        hy=0.12,
        hz=0.02,
        cfg=builder.ShapeConfig(density=0.0, mu=0.0),
    )
    builder.add_particle(pos=(0.003, 0.004, -0.0005), vel=(0.0, 0.0, 0.0), mass=0.001, radius=0.005)
    return builder.finalize(device=device)


def _make_solver(model, recovery_time=0.0):
    config = SolverImplicitMPM.Config(
        voxel_size=0.025,
        grid_type="sparse",
        max_active_cell_count=16,
        collider_basis="pic8",
        strain_basis="P0",
        velocity_basis="Q1",
        solver="jacobi",
        max_iterations=24,
        warmstart_mode="none",
        air_drag=0.0,
    )
    config.contact_recovery_time = recovery_time
    return SolverImplicitMPM(model, config, verbose=False)


def _state_arrays(state):
    return [state.particle_q.numpy(), state.particle_qd.numpy()] + [
        getattr(state.mpm, name).numpy()
        for name in (
            "particle_qd_grad",
            "particle_elastic_strain",
            "particle_Jp",
            "particle_stress",
            "particle_transform",
        )
    ]


def test_contact_recovery_time_and_positive_gap(test, device):
    """Recover a prescribed fraction at either timestep without changing positive-gap contacts."""
    model = _make_contact_model(device)
    tau = -(1.0 / 480.0) / math.log(0.8)
    default_solver = _make_solver(model)
    solver = _make_solver(model, tau)
    depth = 0.0005
    for dt in (1.0 / 480.0, 1.0 / 960.0):
        with test.subTest(dt=dt):
            state = model.state()
            solver.reset(state)
            initial = state.particle_q.numpy().copy()
            solver.step(state, state, None, None, dt)
            beta = -math.expm1(-dt / tau)
            np.testing.assert_allclose(state.particle_qd.numpy()[0], (0.0, 0.0, beta * depth / dt), atol=2.0e-7)
            test.assertAlmostEqual(float(state.particle_q.numpy()[0, 2]), -(1.0 - beta) * depth, delta=2.0e-9)
            np.testing.assert_allclose(state.particle_q.numpy() - initial, dt * state.particle_qd.numpy(), atol=1.0e-10)

            # Closing a positive gap uses the existing speculative contact, including a nonzero impulse.
            states = [model.state(), model.state()]
            for current, current_solver in zip(states, (default_solver, solver), strict=True):
                current.particle_q.assign(wp.array([(0.003, 0.004, 0.002)], dtype=wp.vec3, device=device))
                current.particle_qd.assign(wp.array([(0.0, 0.0, -2.0)], dtype=wp.vec3, device=device))
                current_solver.reset(current)
                current_solver.step(current, current, None, None, dt)
            for previous, enabled in zip(_state_arrays(states[0]), _state_arrays(states[1]), strict=True):
                np.testing.assert_array_equal(previous, enabled)
            test.assertGreater(float(states[0].particle_qd.numpy()[0, 2]), -2.0)

    state = model.state()
    default_solver.reset(state)
    initial = state.particle_q.numpy().copy()
    default_solver.step(state, state, None, None, 1.0 / 480.0)
    np.testing.assert_array_equal(state.particle_q.numpy(), initial)
    np.testing.assert_array_equal(state.particle_qd.numpy(), np.zeros((1, 3)))

    # Normal smoothing must rotate the same bias while retaining the moving wall's velocity.
    dt = 1.0 / 480.0
    beta = -math.expm1(-dt / tau)
    gaps = np.array([-0.0005, 0.002], dtype=np.float32)
    effective_gaps = np.array([beta * gaps[0], gaps[1]])
    wall_velocity = np.array([0.1, -0.2, 0.3])
    velocity = wp.array(wall_velocity - (effective_gaps / dt)[:, None] * (0.0, 0.0, 1.0), dtype=wp.vec3, device=device)
    normal = wp.array([(0.0, 0.0, 1.0)] * 2, dtype=wp.vec3, device=device)
    gradient = wp.array([(0.0, 1.0, 0.0)] * 2, dtype=wp.vec3, device=device)
    wp.launch(normalize_gradient, dim=2, inputs=[gradient, normal, wp.array(gaps, device=device), velocity, dt, beta])
    np.testing.assert_allclose(
        velocity.numpy(), wall_velocity - (effective_gaps / dt)[:, None] * (0.0, 1.0, 0.0), atol=1e-7
    )


def test_contact_recovery_finite_body_reaction(test, device):
    """The recovery impulse carries the opposite force and moment to a finite-mass collider."""
    model = _make_contact_model(device, body_mass=0.01)
    solver = _make_solver(model, -(1.0 / 480.0) / math.log(0.8))
    state = model.state()
    dt = 1.0 / 480.0
    solver.step(state, state, None, None, dt)
    particle_impulse = model.particle_mass.numpy()[:, None] * state.particle_qd.numpy()
    impulses, positions, _ids = solver.collect_collider_impulses(state)
    reaction = impulses.numpy()
    points = positions.numpy()
    active = np.linalg.norm(reaction, axis=1) > 0.0
    test.assertTrue(np.any(active))
    test.assertGreater(float(particle_impulse[0, 2]), 0.0)
    np.testing.assert_allclose(reaction[active].sum(axis=0), -particle_impulse.sum(axis=0), atol=2.0e-10)
    np.testing.assert_allclose(
        np.cross(points[active], reaction[active]).sum(axis=0),
        -np.cross(state.particle_q.numpy(), particle_impulse).sum(axis=0),
        atol=2.0e-10,
    )


def test_contact_recovery_config_validation(test, device):
    """Reject nonphysical times and modes that cannot apply particle-contact recovery."""
    model = _make_contact_model(device)
    for value in (-1.0, math.inf, -math.inf, math.nan):
        with test.subTest(value=value), test.assertRaisesRegex(ValueError, "contact_recovery_time"):
            _make_solver(model, value)
    config = SolverImplicitMPM.Config()
    config.contact_recovery_time = 0.01
    with test.assertRaisesRegex(ValueError, "particle collider basis"):
        SolverImplicitMPM(model, config)


class TestImplicitMPMContactRecovery(unittest.TestCase):
    pass


for name, function in (
    ("test_contact_recovery_time_and_positive_gap", test_contact_recovery_time_and_positive_gap),
    ("test_contact_recovery_finite_body_reaction", test_contact_recovery_finite_body_reaction),
    ("test_contact_recovery_config_validation", test_contact_recovery_config_validation),
):
    add_function_test(TestImplicitMPMContactRecovery, name, function, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
