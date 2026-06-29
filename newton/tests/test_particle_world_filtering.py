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


devices = get_test_devices(mode="basic")


class TestParticleWorldFiltering(unittest.TestCase):
    pass


for _name in (
    "test_xpbd_coincident_fluid_worlds_are_isolated",
    "test_xpbd_mixed_fluid_solid_contacts_are_isolated",
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
