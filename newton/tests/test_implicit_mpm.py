# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.implicit_mpm.solve_rheology import ArraySquaredNorm
from newton.solvers import SolverImplicitMPM
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


def _make_mpm_particle_builder(gravity=-9.81, velocity=(0.0, 0.0, 0.0), young_modulus=1.0e4):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=float(gravity))
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_particle_grid(
        pos=wp.vec3(0.025, 0.025, 0.025),
        rot=wp.quat_identity(),
        vel=wp.vec3(velocity),
        dim_x=2,
        dim_y=2,
        dim_z=2,
        cell_x=0.05,
        cell_y=0.05,
        cell_z=0.05,
        mass=0.01,
        jitter=0.0,
        radius_mean=0.025,
        custom_attributes={"mpm:young_modulus": young_modulus, "mpm:poisson_ratio": 0.2},
    )
    return builder


def _make_mpm_config(grid_type="dense", integration_scheme="pic", solver="jacobi"):
    config = SolverImplicitMPM.Config()
    config.grid_type = grid_type
    config.voxel_size = 0.1
    config.integration_scheme = integration_scheme
    config.solver = solver
    config.max_iterations = 4
    config.tolerance = 0.0
    config.warmstart_mode = "grid"
    return config


def _step_mpm(model, config, step_count=3, dt=0.01):
    solver = SolverImplicitMPM(model, config=config)
    state_0 = model.state()
    state_1 = model.state()
    for _ in range(step_count):
        solver.step(state_0, state_1, control=None, contacts=None, dt=dt)
        state_0, state_1 = state_1, state_0
    return solver, state_0


def _compressive_shear_velocity(positions, amplitude):
    centered = positions - np.mean(positions, axis=0)
    return amplitude * np.column_stack(
        (
            -1.0 * centered[:, 0] + 0.4 * centered[:, 1],
            -0.7 * centered[:, 1] + 0.3 * centered[:, 2],
            -0.5 * centered[:, 2] + 0.6 * centered[:, 0],
        )
    )


def test_array_squared_norm_batches(test, device):
    data = wp.array((1.0, 4.0, 9.0, 16.0, 25.0), dtype=float, device=device)
    offsets = wp.array((0, 2, 2, 5), dtype=int, device=device)
    norm = ArraySquaredNorm(max_length=5, batch_offsets=offsets, device=device)

    try:
        result = norm.compute_squared_norm(data)
        test.assertEqual(result.shape, (2, 3))
        np.testing.assert_array_equal(result.numpy()[0], np.array((5.0, 0.0, 50.0)))
        np.testing.assert_array_equal(result.numpy()[1], np.array((4.0, 0.0, 25.0)))
    finally:
        norm.release()


def test_multiworld_cr_matches_independent(test, device):
    young_moduli = (2.5e3, 4.0e4)
    velocity_amplitudes = (3.0, 11.0)
    reference_states = []
    reference_initial_q = []
    reference_initial_qd = []

    config = _make_mpm_config(grid_type="dense", integration_scheme="pic", solver="cr")
    config.max_iterations = 20
    config.tolerance = 1.0e-5
    config.warmstart_mode = "none"

    for young_modulus, velocity_amplitude in zip(young_moduli, velocity_amplitudes, strict=True):
        reference_model = _make_mpm_particle_builder(
            gravity=0.0,
            young_modulus=young_modulus,
        ).finalize(device=device)
        initial_q = reference_model.particle_q.numpy()
        initial_qd = _compressive_shear_velocity(initial_q, velocity_amplitude)
        reference_model.particle_qd.assign(initial_qd)
        _, reference_state = _step_mpm(reference_model, config, step_count=2)
        reference_states.append((reference_state.particle_q.numpy(), reference_state.particle_qd.numpy()))
        reference_initial_q.append(initial_q)
        reference_initial_qd.append(initial_qd)

    multiworld_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    SolverImplicitMPM.register_custom_attributes(multiworld_builder)
    for young_modulus in young_moduli:
        multiworld_builder.add_world(
            _make_mpm_particle_builder(
                gravity=0.0,
                young_modulus=young_modulus,
            )
        )
    multiworld_model = multiworld_builder.finalize(device=device)
    starts = multiworld_model.particle_world_start.numpy()
    multiworld_initial_q = multiworld_model.particle_q.numpy()
    multiworld_initial_qd = np.empty_like(multiworld_initial_q)
    for world, velocity_amplitude in enumerate(velocity_amplitudes):
        world_slice = slice(starts[world], starts[world + 1])
        multiworld_initial_qd[world_slice] = _compressive_shear_velocity(
            multiworld_initial_q[world_slice], velocity_amplitude
        )
    multiworld_model.particle_qd.assign(multiworld_initial_qd)

    _, multiworld_state = _step_mpm(multiworld_model, config, step_count=2)
    multiworld_q = multiworld_state.particle_q.numpy()
    multiworld_qd = multiworld_state.particle_qd.numpy()

    for world, (reference_q, reference_qd) in enumerate(reference_states):
        world_slice = slice(starts[world], starts[world + 1])
        world_q = multiworld_q[world_slice]
        world_qd = multiworld_qd[world_slice]
        np.testing.assert_allclose(world_q, reference_q, rtol=1.0e-5, atol=1.0e-6, equal_nan=False)
        np.testing.assert_allclose(world_qd, reference_qd, rtol=1.0e-5, atol=1.0e-6, equal_nan=False)
        test.assertTrue(np.isfinite(world_q).all())
        test.assertTrue(np.isfinite(world_qd).all())
        test.assertGreater(np.linalg.norm(reference_q - reference_initial_q[world]), 1.0e-4)
        test.assertGreater(np.linalg.norm(reference_qd - reference_initial_qd[world]), 1.0e-4)


def _run_multiworld_reference_case(device, grid_type="dense", integration_scheme="pic", solver="jacobi"):
    world_gravities = ((3.0, -2.0, 0.0), (-5.0, 1.0, 0.0))
    reference_states = []

    for world_gravity in world_gravities:
        reference_model = _make_mpm_particle_builder().finalize(device=device)
        reference_model.set_gravity(world_gravity)
        _, reference_state = _step_mpm(
            reference_model,
            _make_mpm_config(grid_type=grid_type, integration_scheme=integration_scheme, solver=solver),
        )
        reference_states.append((reference_state.particle_q.numpy(), reference_state.particle_qd.numpy()))

    local_builder = _make_mpm_particle_builder()
    multiworld_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    SolverImplicitMPM.register_custom_attributes(multiworld_builder)
    multiworld_builder.add_world(local_builder)
    multiworld_builder.add_world(local_builder)
    multiworld_model = multiworld_builder.finalize(device=device)
    for world, world_gravity in enumerate(world_gravities):
        multiworld_model.set_gravity(world_gravity, world=world)

    _, multiworld_state = _step_mpm(
        multiworld_model,
        _make_mpm_config(grid_type=grid_type, integration_scheme=integration_scheme, solver=solver),
    )
    starts = multiworld_model.particle_world_start.numpy()
    multiworld_q = multiworld_state.particle_q.numpy()
    multiworld_qd = multiworld_state.particle_qd.numpy()

    mean_velocities = []
    for world, (reference_q, reference_qd) in enumerate(reference_states):
        world_slice = slice(starts[world], starts[world + 1])
        world_q = multiworld_q[world_slice]
        world_qd = multiworld_qd[world_slice]
        np.testing.assert_allclose(world_q, reference_q, rtol=1.0e-5, atol=1.0e-6, equal_nan=False)
        np.testing.assert_allclose(world_qd, reference_qd, rtol=1.0e-5, atol=1.0e-6, equal_nan=False)
        mean_velocities.append(np.mean(world_qd, axis=0))

    mean_velocities = np.asarray(mean_velocities)
    np.testing.assert_array_equal(np.isfinite(mean_velocities), np.ones_like(mean_velocities, dtype=bool))
    np.testing.assert_array_less(np.full(2, 1.0e-3), np.abs(mean_velocities[:, 0]))
    np.testing.assert_array_equal(np.sign(mean_velocities[:, 0]), np.array((1.0, -1.0)))


def test_multiworld_dense_pic_matches_independent(test, device):
    _run_multiworld_reference_case(device, grid_type="dense", integration_scheme="pic")


def test_multiworld_dense_gimp_matches_independent(test, device):
    _run_multiworld_reference_case(device, grid_type="dense", integration_scheme="gimp")


def test_multiworld_fixed_pic_matches_independent(test, device):
    _run_multiworld_reference_case(device, grid_type="fixed", integration_scheme="pic")


def test_multiworld_sparse_pic_matches_independent(test, device):
    _run_multiworld_reference_case(device, grid_type="sparse", integration_scheme="pic")


def test_multiworld_sparse_gimp_matches_independent(test, device):
    _run_multiworld_reference_case(device, grid_type="sparse", integration_scheme="gimp")


def test_multiworld_sparse_empty_worlds_padding_matches_independent(test, device):
    populated_worlds = (1, 3)
    world_gravities = ((3.0, -2.0, 0.0), (-5.0, 1.0, 0.0))
    reference_states = []

    for world_gravity in world_gravities:
        reference_model = _make_mpm_particle_builder().finalize(device=device)
        reference_model.set_gravity(world_gravity)
        reference_config = _make_mpm_config(grid_type="sparse", integration_scheme="pic")
        reference_config.grid_padding = 1
        _, reference_state = _step_mpm(reference_model, reference_config)
        reference_states.append((reference_state.particle_q.numpy(), reference_state.particle_qd.numpy()))

    local_builder = _make_mpm_particle_builder()
    empty_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    empty_builder.add_body(is_kinematic=True, label="empty_world_marker")
    multiworld_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    SolverImplicitMPM.register_custom_attributes(multiworld_builder)
    for world_builder in (empty_builder, local_builder, empty_builder, local_builder, empty_builder):
        multiworld_builder.add_world(world_builder)

    multiworld_model = multiworld_builder.finalize(device=device)
    for world, world_gravity in zip(populated_worlds, world_gravities, strict=True):
        multiworld_model.set_gravity(world_gravity, world=world)

    starts = multiworld_model.particle_world_start.numpy()
    particles_per_world = reference_states[0][0].shape[0]
    for world in (0, 2, 4):
        test.assertEqual(starts[world], starts[world + 1])
    for world in populated_worlds:
        test.assertEqual(starts[world + 1] - starts[world], particles_per_world)

    multiworld_config = _make_mpm_config(grid_type="sparse", integration_scheme="pic")
    multiworld_config.grid_padding = 1
    _, multiworld_state = _step_mpm(multiworld_model, multiworld_config)
    multiworld_q = multiworld_state.particle_q.numpy()
    multiworld_qd = multiworld_state.particle_qd.numpy()

    mean_velocities = []
    for world, (reference_q, reference_qd) in zip(populated_worlds, reference_states, strict=True):
        world_slice = slice(starts[world], starts[world + 1])
        world_q = multiworld_q[world_slice]
        world_qd = multiworld_qd[world_slice]
        np.testing.assert_array_equal(np.isfinite(world_q), np.ones_like(world_q, dtype=bool))
        np.testing.assert_array_equal(np.isfinite(world_qd), np.ones_like(world_qd, dtype=bool))
        np.testing.assert_allclose(world_q, reference_q, rtol=1.0e-5, atol=1.0e-6, equal_nan=False)
        np.testing.assert_allclose(world_qd, reference_qd, rtol=1.0e-5, atol=1.0e-6, equal_nan=False)
        mean_velocities.append(np.mean(world_qd, axis=0))

    mean_velocities = np.asarray(mean_velocities)
    np.testing.assert_array_less(np.full(2, 1.0e-3), np.abs(mean_velocities[:, 0]))
    np.testing.assert_array_equal(np.sign(mean_velocities[:, 0]), np.array((1.0, -1.0)))


def test_multiworld_isolation_config(test, device):
    config = SolverImplicitMPM.Config()
    test.assertTrue(config.separate_worlds)


def test_empty_particle_model_rejected(test, device):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-9.81)
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_world(newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-9.81))
    builder.add_world(newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-9.81))
    model = builder.finalize(device=device)

    with test.assertRaisesRegex(ValueError, "at least one particle"):
        SolverImplicitMPM(model, _make_mpm_config())


def test_multiworld_global_particles_rejected(test, device):
    builder = _make_mpm_particle_builder()
    local = _make_mpm_particle_builder()
    builder.add_world(local)
    builder.add_world(local)
    model = builder.finalize(device=device)

    with test.assertRaisesRegex(ValueError, "global MPM particles"):
        SolverImplicitMPM(model, _make_mpm_config())


def test_single_world_global_particles_supported(test, device):
    model = _make_mpm_particle_builder().finalize(device=device)
    initial_q = model.particle_q.numpy()
    _solver, state = _step_mpm(model, _make_mpm_config(), step_count=1)
    particle_q = state.particle_q.numpy()
    particle_qd = state.particle_qd.numpy()
    test.assertTrue(np.isfinite(particle_q).all())
    test.assertFalse(np.array_equal(particle_q, initial_q))
    test.assertTrue(np.all(particle_qd[:, 1] < 0.0))


def test_multiworld_shared_grid_opt_out_accepts_global_particles(test, device):
    builder = _make_mpm_particle_builder()
    local = _make_mpm_particle_builder()
    builder.add_world(local)
    builder.add_world(local)
    model = builder.finalize(device=device)
    initial_q = model.particle_q.numpy()
    config = _make_mpm_config()
    config.separate_worlds = False
    _solver, state = _step_mpm(model, config, step_count=1)
    particle_q = state.particle_q.numpy()
    particle_qd = state.particle_qd.numpy()
    particle_world = model.particle_world.numpy()
    test.assertTrue(np.isfinite(particle_q).all())
    test.assertFalse(np.array_equal(particle_q, initial_q))
    for world in range(-1, model.world_count):
        world_qd = particle_qd[particle_world == world]
        test.assertGreater(world_qd.shape[0], 0)
        test.assertTrue(np.all(world_qd[:, 1] < 0.0))


def test_multiworld_invalid_particle_world_rejected(test, device):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-9.81)
    SolverImplicitMPM.register_custom_attributes(builder)
    local = _make_mpm_particle_builder()
    builder.add_world(local)
    builder.add_world(local)
    model = builder.finalize(device=device)
    particle_world = model.particle_world.numpy()

    for invalid_world in (-2, model.world_count):
        invalid_particle_world = particle_world.copy()
        invalid_particle_world[0] = invalid_world
        model.particle_world.assign(invalid_particle_world)
        with test.subTest(invalid_world=invalid_world):
            with test.assertRaisesRegex(ValueError, "invalid MPM particle world IDs"):
                SolverImplicitMPM(model, _make_mpm_config())


def test_multiworld_effective_isolation_mode(test, device):
    single_world_model = _make_mpm_particle_builder().finalize(device=device)
    single_world_solver = SolverImplicitMPM(single_world_model, _make_mpm_config())
    test.assertFalse(single_world_solver._separate_worlds)
    test.assertEqual(single_world_solver._environment_count, 1)
    test.assertIsNone(single_world_solver._particle_environment)

    local = _make_mpm_particle_builder()
    isolated_builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-9.81)
    SolverImplicitMPM.register_custom_attributes(isolated_builder)
    isolated_builder.add_world(local)
    isolated_builder.add_world(local)
    isolated_model = isolated_builder.finalize(device=device)
    isolated_solver = SolverImplicitMPM(isolated_model, _make_mpm_config())
    test.assertTrue(isolated_solver._separate_worlds)
    test.assertEqual(isolated_solver._environment_count, isolated_model.world_count)
    test.assertIs(isolated_solver._particle_environment, isolated_model.particle_world)

    shared_builder = _make_mpm_particle_builder()
    shared_builder.add_world(local)
    shared_builder.add_world(local)
    shared_model = shared_builder.finalize(device=device)
    shared_config = _make_mpm_config()
    shared_config.separate_worlds = False
    shared_solver = SolverImplicitMPM(shared_model, shared_config)
    test.assertFalse(shared_solver._separate_worlds)
    test.assertEqual(shared_solver._environment_count, 1)
    test.assertIsNone(shared_solver._particle_environment)


def test_sand_cube_on_plane(test, device):
    # Emits a cube of particles on the ground

    N = 4
    particles_per_cell = 3
    voxel_size = 0.5
    particle_spacing = voxel_size / particles_per_cell
    friction = 0.6
    dt = 0.04

    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)

    # Register MPM custom attributes before adding particles
    SolverImplicitMPM.register_custom_attributes(builder)

    builder.add_particle_grid(
        pos=wp.vec3(0.5 * particle_spacing),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=N * particles_per_cell,
        dim_y=N * particles_per_cell,
        dim_z=N * particles_per_cell,
        cell_x=particle_spacing,
        cell_y=particle_spacing,
        cell_z=particle_spacing,
        mass=1.0,
        jitter=0.0,
        custom_attributes={"mpm:friction": friction},
    )
    builder.add_ground_plane()

    model: newton.Model = builder.finalize(device=device)

    state_0: newton.State = model.state()
    state_1: newton.State = model.state()

    options = SolverImplicitMPM.Config()
    options.grid_type = "dense"  # use dense grid as sparse grid is GPU-only
    options.voxel_size = voxel_size

    solver = SolverImplicitMPM(model, config=options)

    init_pos = state_0.particle_q.numpy()

    # Run a few steps
    for _k in range(25):
        solver.step(state_0, state_1, control=None, contacts=None, dt=dt)
        state_0, state_1 = state_1, state_0

    # Checks the final bounding box corresponds to the expected collapse
    end_pos = state_0.particle_q.numpy()
    bb_min, bb_max = np.min(end_pos, axis=0), np.max(end_pos, axis=0)
    assert bb_min[model.up_axis] > -voxel_size
    assert voxel_size < bb_max[model.up_axis] < N * voxel_size

    assert np.all(bb_min > -N * voxel_size)
    assert np.all(bb_min < np.min(init_pos, axis=0))
    assert np.all(bb_max < 2 * N * voxel_size)

    # Checks that contact impulses are consistent
    impulses, impulse_positions, _collider_ids = solver.collect_collider_impulses(state_0)

    impulses = impulses.numpy()
    impulse_positions = impulse_positions.numpy()

    active_contacts = np.flatnonzero(np.linalg.norm(impulses, axis=1) > 0.01)
    contact_points = impulse_positions[active_contacts]
    contact_impulses = impulses[active_contacts]

    assert np.all(contact_points[:, model.up_axis] == 0.0)
    assert np.all(contact_impulses[:, model.up_axis] < 0.0)


def test_finite_difference_collider_velocity(test, device):
    """Test that finite-difference velocity mode correctly computes collider velocity.

    This test compares the two velocity modes with body_qd=0:
    - instantaneous mode: sees zero velocity (from body_qd), particles don't move with platform
    - finite_difference mode: computes velocity from position change, particles move with platform

    This directly validates that finite-difference mode correctly handles the case where
    body transforms are updated externally but body_qd doesn't reflect the actual motion.
    """
    voxel_size = 0.1
    particles_per_cell = 2
    particle_spacing = voxel_size / particles_per_cell
    dt = 0.02
    n_steps = 15

    # Platform moves in +X direction
    platform_vel_x = 0.5  # m/s

    def run_simulation(velocity_mode):
        """Run simulation with given velocity mode and return particle displacement."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)

        # Register MPM custom attributes before adding particles
        SolverImplicitMPM.register_custom_attributes(builder)

        # Add particles resting on the platform
        builder.add_particle_grid(
            pos=wp.vec3(-0.05, 0.12, -0.05),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=2 * particles_per_cell,
            dim_y=2 * particles_per_cell,
            dim_z=2 * particles_per_cell,
            cell_x=particle_spacing,
            cell_y=particle_spacing,
            cell_z=particle_spacing,
            mass=1.0,
            jitter=0.0,
            custom_attributes={"mpm:friction": 1.0},  # high friction
        )

        # Add a platform that particles rest on
        platform_body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
        platform_mesh = newton.Mesh.create_box(
            0.5,
            0.1,
            0.5,
            duplicate_vertices=False,
            compute_normals=False,
            compute_uvs=False,
            compute_inertia=False,
        )
        shape_cfg = newton.ModelBuilder.ShapeConfig(density=0.0)  # kinematic
        shape_cfg.margin = 0.02
        builder.add_shape_mesh(
            body=platform_body,
            mesh=platform_mesh,
            cfg=shape_cfg,
        )

        model = builder.finalize(device=device)

        state_0 = model.state()
        state_1 = model.state()

        options = SolverImplicitMPM.Config()
        options.voxel_size = voxel_size
        options.grid_type = "dense"
        options.collider_velocity_mode = velocity_mode

        solver = SolverImplicitMPM(model, config=options)

        init_mean_x = np.mean(state_0.particle_q.numpy()[:, 0])

        # Move platform with body_qd = 0
        for k in range(n_steps):
            t = (k + 1) * dt
            new_platform_x = platform_vel_x * t

            body_q_np = state_0.body_q.numpy()
            body_q_np[0] = (new_platform_x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
            state_0.body_q.assign(body_q_np)

            # KEY: body_qd is ZERO - doesn't reflect actual motion
            body_qd_np = state_0.body_qd.numpy()
            body_qd_np[0] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            state_0.body_qd.assign(body_qd_np)

            solver.step(state_0, state_1, control=None, contacts=None, dt=dt)
            state_0, state_1 = state_1, state_0

        end_mean_x = np.mean(state_0.particle_q.numpy()[:, 0])
        return end_mean_x - init_mean_x

    # 'forward' sees the current collider velocity; 'backward' derives it from
    # the previous-step collider position.
    displacement_instantaneous = run_simulation("forward")
    displacement_finite_diff = run_simulation("backward")

    # With instantaneous mode and body_qd=0, particles should barely move
    # (they see zero collider velocity, so no friction drag)
    test.assertLess(
        abs(displacement_instantaneous),
        0.02,
        f"instantaneous mode with body_qd=0 should show minimal particle movement, "
        f"but got {displacement_instantaneous:.3f}",
    )

    # With finite_difference mode, particles should move significantly
    # (velocity computed from position change)
    test.assertGreater(
        displacement_finite_diff,
        0.05,
        f"finite_difference mode should move particles with platform, "
        f"but displacement was only {displacement_finite_diff:.3f}",
    )

    # finite_difference should show significantly more movement than instantaneous
    test.assertGreater(
        displacement_finite_diff,
        displacement_instantaneous + 0.03,
        f"finite_difference ({displacement_finite_diff:.3f}) should show significantly more "
        f"movement than instantaneous ({displacement_instantaneous:.3f})",
    )


devices = get_test_devices()
basic_devices = get_test_devices(mode="basic")
basic_cuda_devices = get_cuda_test_devices(mode="basic")


class TestImplicitMPM(unittest.TestCase):
    pass


add_function_test(
    TestImplicitMPM,
    "test_array_squared_norm_batches",
    test_array_squared_norm_batches,
    devices=basic_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_multiworld_cr_matches_independent",
    test_multiworld_cr_matches_independent,
    devices=basic_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_multiworld_dense_pic_matches_independent",
    test_multiworld_dense_pic_matches_independent,
    devices=basic_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_multiworld_dense_gimp_matches_independent",
    test_multiworld_dense_gimp_matches_independent,
    devices=basic_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_multiworld_fixed_pic_matches_independent",
    test_multiworld_fixed_pic_matches_independent,
    devices=basic_cuda_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_multiworld_sparse_pic_matches_independent",
    test_multiworld_sparse_pic_matches_independent,
    devices=basic_cuda_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_multiworld_sparse_gimp_matches_independent",
    test_multiworld_sparse_gimp_matches_independent,
    devices=basic_cuda_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_multiworld_sparse_empty_worlds_padding_matches_independent",
    test_multiworld_sparse_empty_worlds_padding_matches_independent,
    devices=basic_cuda_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_multiworld_isolation_config",
    test_multiworld_isolation_config,
    devices=basic_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_empty_particle_model_rejected",
    test_empty_particle_model_rejected,
    devices=basic_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_multiworld_global_particles_rejected",
    test_multiworld_global_particles_rejected,
    devices=basic_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_single_world_global_particles_supported",
    test_single_world_global_particles_supported,
    devices=basic_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_multiworld_shared_grid_opt_out_accepts_global_particles",
    test_multiworld_shared_grid_opt_out_accepts_global_particles,
    devices=basic_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_multiworld_invalid_particle_world_rejected",
    test_multiworld_invalid_particle_world_rejected,
    devices=basic_devices,
)

add_function_test(
    TestImplicitMPM,
    "test_multiworld_effective_isolation_mode",
    test_multiworld_effective_isolation_mode,
    devices=basic_devices,
)

add_function_test(
    TestImplicitMPM, "test_sand_cube_on_plane", test_sand_cube_on_plane, devices=devices, check_output=False
)

add_function_test(
    TestImplicitMPM,
    "test_finite_difference_collider_velocity",
    test_finite_difference_collider_velocity,
    devices=devices,
    check_output=False,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
