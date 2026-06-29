# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for particle interactions across Newton worlds."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverSPH, SolverXPBD
from newton.tests.unittest_utils import add_function_test, get_test_devices

SPACING = 0.05
RADIUS = 0.025
MASS = 0.125
FLUID_FLAGS = int(newton.ParticleFlags.ACTIVE | newton.ParticleFlags.FLUID)
ACTIVE_FLAGS = int(newton.ParticleFlags.ACTIVE)


def _make_particle_builder(positions, velocities, flags=FLUID_FLAGS):
    builder = newton.ModelBuilder(gravity=0.0)
    builder.default_particle_radius = RADIUS
    count = len(positions)
    builder.add_particles(
        pos=[wp.vec3(*p) for p in positions],
        vel=[wp.vec3(*v) for v in velocities],
        mass=[MASS] * count,
        radius=[RADIUS] * count,
        flags=[flags] * count if isinstance(flags, int) else flags,
    )
    return builder


def _make_colocated_model(device, *world_builders, global_builders=()):
    builder = newton.ModelBuilder(gravity=0.0)
    for global_builder in global_builders:
        builder.add_builder(global_builder)
    for world_builder in world_builders:
        builder.add_world(world_builder, xform=wp.transform_identity())
    return builder.finalize(device=device)


def _step_xpbd(model, **kwargs):
    state_in = model.state()
    state_out = model.state()
    solver = SolverXPBD(model, fluid_rest_distance=SPACING, **kwargs)
    solver.step(state_in, state_out, control=None, contacts=None, dt=1.0 / 120.0)
    return state_out, solver


def _step_sph(model, **kwargs):
    state_in = model.state()
    state_out = model.state()
    solver = SolverSPH(model, **kwargs)
    solver.step(state_in, state_out, control=None, contacts=None, dt=1.0 / 120.0)
    return state_out, solver


def test_xpbd_coincident_fluid_worlds_are_isolated(test, device):
    particle = _make_particle_builder([(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)])
    model = _make_colocated_model(device, particle, particle)
    initial_q = model.particle_q.numpy().copy()

    state, _solver = _step_xpbd(model, iterations=1, fluid_cohesion=0.0)

    np.testing.assert_allclose(state.particle_q.numpy(), initial_q, rtol=0.0, atol=1.0e-7)
    np.testing.assert_allclose(state.particle_qd.numpy(), 0.0, rtol=0.0, atol=1.0e-7)


def test_xpbd_mixed_fluid_solid_contacts_are_isolated(test, device):
    fluid = _make_particle_builder([(-0.01, 0.0, 0.0)], [(0.0, 0.0, 0.0)], FLUID_FLAGS)
    solid = _make_particle_builder([(0.01, 0.0, 0.0)], [(0.0, 0.0, 0.0)], ACTIVE_FLAGS)
    model = _make_colocated_model(device, fluid, solid)
    initial_q = model.particle_q.numpy().copy()

    state, _solver = _step_xpbd(model, iterations=1, fluid_cohesion=0.0)

    np.testing.assert_allclose(state.particle_q.numpy(), initial_q, rtol=0.0, atol=1.0e-7)
    np.testing.assert_allclose(state.particle_qd.numpy(), 0.0, rtol=0.0, atol=1.0e-7)


def test_xpbd_multiworld_matches_independent_fluid_baselines(test, device):
    positions = [(0.0, 0.0, 0.0), (0.035, 0.0, 0.0), (0.0, 0.04, 0.0), (0.03, 0.035, 0.01)]
    velocities = [
        [(0.3, 0.0, 0.0), (0.0, 0.2, 0.0), (-0.1, 0.0, 0.1), (0.0, -0.2, 0.0)],
        [(-0.2, 0.1, 0.0), (0.1, -0.3, 0.0), (0.0, 0.2, -0.1), (0.2, 0.0, 0.0)],
    ]
    templates = [_make_particle_builder(positions, world_velocities) for world_velocities in velocities]
    multi_model = _make_colocated_model(device, *templates)
    baseline_models = [template.finalize(device=device) for template in templates]
    kwargs = {
        "iterations": 2,
        "fluid_cohesion": 0.0,
        "fluid_viscosity": 0.35,
        "fluid_vorticity_confinement": 0.2,
    }

    multi_state, multi_solver = _step_xpbd(multi_model, **kwargs)
    baselines = [_step_xpbd(model, **kwargs) for model in baseline_models]
    worlds = multi_model.particle_world.numpy()

    for world_id, (baseline_state, baseline_solver) in enumerate(baselines):
        mask = worlds == world_id
        np.testing.assert_allclose(
            multi_state.particle_q.numpy()[mask], baseline_state.particle_q.numpy(), rtol=1.0e-5, atol=1.0e-6
        )
        np.testing.assert_allclose(
            multi_state.particle_qd.numpy()[mask], baseline_state.particle_qd.numpy(), rtol=1.0e-5, atol=1.0e-6
        )
        np.testing.assert_allclose(
            multi_solver._fluid_density.numpy()[mask],
            baseline_solver._fluid_density.numpy(),
            rtol=1.0e-5,
            atol=1.0e-5,
        )
        np.testing.assert_allclose(
            multi_solver._fluid_vorticity.numpy()[mask],
            baseline_solver._fluid_vorticity.numpy(),
            rtol=1.0e-5,
            atol=1.0e-5,
        )


def test_xpbd_multiworld_reorder_is_noop(test, device):
    # Keep each offset beyond one smoothing cell so the pre-fix Morton sort cannot preserve input order.
    positions = [(0.18, 0.0, 0.0), (0.0, 0.18, 0.0), (0.0, 0.0, 0.18), (0.0, 0.0, 0.0)]
    velocities = [(0.0, 0.0, 0.0)] * len(positions)
    template = _make_particle_builder(positions, velocities)
    model = _make_colocated_model(device, template, template)
    state = model.state()
    solver = SolverXPBD(model, iterations=1, fluid_rest_distance=SPACING)
    before_state = (state.particle_q.numpy().copy(), state.particle_qd.numpy().copy())
    before_model = tuple(
        array.numpy().copy()
        for array in (
            model.particle_q,
            model.particle_qd,
            model.particle_colors,
            model.particle_mass,
            model.particle_inv_mass,
            model.particle_radius,
            model.particle_flags,
            model.particle_world,
            model.particle_world_start,
        )
    )

    solver.reorder_particles(state)

    np.testing.assert_array_equal(state.particle_q.numpy(), before_state[0])
    np.testing.assert_array_equal(state.particle_qd.numpy(), before_state[1])
    for array, expected in zip(
        (
            model.particle_q,
            model.particle_qd,
            model.particle_colors,
            model.particle_mass,
            model.particle_inv_mass,
            model.particle_radius,
            model.particle_flags,
            model.particle_world,
            model.particle_world_start,
        ),
        before_model,
        strict=True,
    ):
        np.testing.assert_array_equal(array.numpy(), expected)


devices = get_test_devices(mode="basic")


class TestParticleWorldFiltering(unittest.TestCase):
    pass


for _name in (
    "test_xpbd_coincident_fluid_worlds_are_isolated",
    "test_xpbd_mixed_fluid_solid_contacts_are_isolated",
    "test_xpbd_multiworld_matches_independent_fluid_baselines",
    "test_xpbd_multiworld_reorder_is_noop",
):
    add_function_test(
        TestParticleWorldFiltering,
        _name,
        globals()[_name],
        devices=devices,
        check_output=False,
    )


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
