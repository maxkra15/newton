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


def test_sph_multiworld_matches_independent_baselines(test, device):
    positions = [(0.0, 0.0, 0.0), (0.04, 0.0, 0.0), (0.0, 0.05, 0.0), (0.035, 0.04, 0.015)]
    velocities = [
        [(0.4, 0.0, 0.0), (0.0, 0.3, 0.0), (-0.2, 0.0, 0.1), (0.0, -0.1, 0.0)],
        [(-0.3, 0.1, 0.0), (0.1, -0.4, 0.0), (0.0, 0.2, -0.1), (0.25, 0.0, 0.0)],
    ]
    templates = [_make_particle_builder(positions, world_velocities, ACTIVE_FLAGS) for world_velocities in velocities]
    multi_model = _make_colocated_model(device, *templates)
    baseline_models = [template.finalize(device=device) for template in templates]
    kwargs = {
        "smoothing_length": 0.12,
        "rest_density": 80.0,
        "gas_constant": 12.0,
        "viscosity": 0.2,
        "particle_friction": 0.15,
        "cohesion": 0.1,
        "surface_tension": 0.05,
        "vorticity_confinement": 0.1,
        "pbf_iterations": 2,
        "xsph_strength": 0.25,
        "shape_collision": False,
        "render_smoothing": 0.0,
        "render_anisotropy_scale": 0.0,
    }

    multi_state, multi_solver = _step_sph(multi_model, **kwargs)
    baselines = [_step_sph(model, **kwargs) for model in baseline_models]
    worlds = multi_model.particle_world.numpy()
    multi_q = multi_state.particle_q.numpy()
    multi_qd = multi_state.particle_qd.numpy()
    multi_density = multi_solver.particle_density.numpy()
    multi_pressure = multi_solver.particle_pressure.numpy()
    multi_vorticity = multi_solver.particle_vorticity.numpy()
    for values in (multi_q, multi_qd, multi_density, multi_pressure, multi_vorticity):
        test.assertTrue(np.isfinite(values).all())

    for world_id, (baseline_state, baseline_solver) in enumerate(baselines):
        mask = worlds == world_id
        baseline_q = baseline_state.particle_q.numpy()
        baseline_qd = baseline_state.particle_qd.numpy()
        baseline_density = baseline_solver.particle_density.numpy()
        baseline_pressure = baseline_solver.particle_pressure.numpy()
        baseline_vorticity = baseline_solver.particle_vorticity.numpy()
        for values in (baseline_q, baseline_qd, baseline_density, baseline_pressure, baseline_vorticity):
            test.assertTrue(np.isfinite(values).all())

        np.testing.assert_allclose(multi_q[mask], baseline_q, rtol=1.0e-5, atol=1.0e-6)
        np.testing.assert_allclose(multi_qd[mask], baseline_qd, rtol=1.0e-5, atol=1.0e-6)
        np.testing.assert_allclose(multi_density[mask], baseline_density, rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_allclose(multi_pressure[mask], baseline_pressure, rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_allclose(multi_vorticity[mask], baseline_vorticity, rtol=1.0e-5, atol=1.0e-5)


def test_particle_global_world_keeps_fallback_interactions(test, device):
    local = _make_particle_builder([(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)], ACTIVE_FLAGS)
    global_particle = _make_particle_builder([(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)], ACTIVE_FLAGS)
    model = _make_colocated_model(device, local, local, global_builders=(global_particle,))
    _state, solver = _step_sph(
        model,
        smoothing_length=0.1,
        rest_density=1.0,
        gas_constant=0.0,
        viscosity=0.0,
        shape_collision=False,
        render_smoothing=0.0,
        render_anisotropy_scale=0.0,
    )
    worlds = model.particle_world.numpy()
    density = solver.particle_density.numpy()
    local_density = density[worlds >= 0]
    global_density = density[worlds == -1]

    test.assertTrue(np.isfinite(density).all())
    test.assertEqual(local_density.shape, (2,))
    test.assertEqual(global_density.shape, (1,))
    np.testing.assert_allclose(local_density, global_density[0] * (2.0 / 3.0), rtol=1.0e-5, atol=1.0e-5)


def test_xpbd_render_neighbors_are_isolated(test, device):
    offsets = (-0.04, -0.02, 0.0, 0.02, 0.04)
    velocities = [(0.0, 0.0, 0.0)] * len(offsets)
    templates = [
        _make_particle_builder([(offset, 0.0, 0.0) for offset in offsets], velocities),
        _make_particle_builder([(0.0, offset, 0.0) for offset in offsets], velocities),
    ]
    multi_model = _make_colocated_model(device, *templates)
    baseline_models = [template.finalize(device=device) for template in templates]
    render_kwargs = {
        "smoothing": 1.0,
        "anisotropy_scale": 1.0,
        "anisotropy_min": 0.2,
        "anisotropy_max": 2.0,
    }

    def render(model):
        state = model.state()
        solver = SolverXPBD(model, iterations=1, fluid_rest_distance=SPACING)
        solver.update_render_particles(state, **render_kwargs)
        outputs = tuple(
            array.numpy()
            for array in (
                solver.render_positions,
                solver.render_anisotropy,
                solver.render_anisotropy_secondary,
                solver.render_anisotropy_tertiary,
            )
        )
        for values in outputs:
            test.assertTrue(np.isfinite(values).all())
        return outputs

    multi_outputs = render(multi_model)
    baseline_outputs = [render(model) for model in baseline_models]
    worlds = multi_model.particle_world.numpy()

    for world_id, expected_outputs in enumerate(baseline_outputs):
        mask = worlds == world_id
        test.assertTrue(np.all(expected_outputs[1][:, 3] > 1.0))
        for actual, expected in zip(multi_outputs, expected_outputs, strict=True):
            np.testing.assert_allclose(actual[mask], expected, rtol=1.0e-5, atol=1.0e-6)


def test_sph_diffuse_neighbors_are_isolated(test, device):
    dt = 1.0 / 120.0
    world_zero = _make_particle_builder([(0.0, 0.0, 0.0)], [(1.0, 0.0, 0.0)], ACTIVE_FLAGS)
    world_one = _make_particle_builder([(-2.0 * dt, 0.0, 0.0)], [(3.0, 0.0, 0.0)], ACTIVE_FLAGS)
    model = _make_colocated_model(device, world_zero, world_one)
    solver = SolverSPH(
        model,
        smoothing_length=0.1,
        rest_density=1.0,
        gas_constant=0.0,
        viscosity=0.0,
        max_diffuse_particles=2,
        diffuse_threshold=1.0e9,
        diffuse_spawn_probability=0.0,
        diffuse_drag=120.0,
        diffuse_buoyancy=1.0,
        diffuse_ballistic=1,
        shape_collision=False,
        render_smoothing=0.0,
        render_anisotropy_scale=0.0,
    )
    solver.diffuse_positions.assign(np.array([[dt, 0.0, 0.0, 1.0], [dt, 0.0, 0.0, 1.0]], dtype=np.float32))
    solver.diffuse_velocities.assign(np.zeros((2, 4), dtype=np.float32))
    solver.diffuse_worlds.assign(np.array([0, -1], dtype=np.int32))
    solver.diffuse_slot_states.assign(np.ones(2, dtype=np.int32))
    state_in, state_out = model.state(), model.state()

    solver.step(state_in, state_out, control=None, contacts=None, dt=dt)

    diffuse_v = solver.diffuse_velocities.numpy()[:, :3]
    test.assertTrue(np.isfinite(diffuse_v).all())
    np.testing.assert_allclose(diffuse_v[0], [1.0, 0.0, 0.0], rtol=1.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(diffuse_v[1], [2.0, 0.0, 0.0], rtol=1.0e-5, atol=1.0e-5)


def test_sph_diffuse_spawning_ignores_other_worlds(test, device):
    left = _make_particle_builder([(-0.01, 0.0, 0.0)], [(-1.0, 0.0, 0.0)], ACTIVE_FLAGS)
    right = _make_particle_builder([(0.01, 0.0, 0.0)], [(1.0, 0.0, 0.0)], ACTIVE_FLAGS)
    model = _make_colocated_model(device, left, right)
    _state, solver = _step_sph(
        model,
        smoothing_length=0.1,
        rest_density=1.0,
        gas_constant=0.0,
        viscosity=0.0,
        max_diffuse_particles=8,
        diffuse_threshold=0.01,
        diffuse_spawn_probability=1.0,
        diffuse_jitter=0.0,
        shape_collision=False,
        render_smoothing=0.0,
        render_anisotropy_scale=0.0,
    )

    diffuse_positions = solver.diffuse_positions.numpy()
    test.assertTrue(np.isfinite(diffuse_positions).all())
    test.assertEqual(int(solver.diffuse_spawn_counter.numpy()[0]), 0)
    test.assertEqual(int(np.count_nonzero(diffuse_positions[:, 3] > 0.0)), 0)


devices = get_test_devices(mode="basic")


class TestParticleWorldFiltering(unittest.TestCase):
    pass


for _name in (
    "test_xpbd_coincident_fluid_worlds_are_isolated",
    "test_xpbd_mixed_fluid_solid_contacts_are_isolated",
    "test_xpbd_multiworld_matches_independent_fluid_baselines",
    "test_xpbd_multiworld_reorder_is_noop",
    "test_sph_multiworld_matches_independent_baselines",
    "test_particle_global_world_keeps_fallback_interactions",
    "test_xpbd_render_neighbors_are_isolated",
    "test_sph_diffuse_neighbors_are_isolated",
    "test_sph_diffuse_spawning_ignores_other_worlds",
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
