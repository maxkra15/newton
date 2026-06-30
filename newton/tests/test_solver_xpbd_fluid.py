# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the position-based fluid (PBF) support in the XPBD solver."""

import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
import newton._src.solvers.particle_grid as particle_grid_module
from newton._src.solvers.xpbd.fluid_kernels import compute_fluid_lambdas
from newton._src.solvers.xpbd.kernels import apply_particle_deltas, clamp_body_velocities
from newton.tests.unittest_utils import add_function_test, get_test_devices

SPACING = 0.05
RADIUS = 0.025
REST_DENSITY = 1000.0
PARTICLE_MASS = REST_DENSITY * SPACING**3
FLUID_FLAGS = newton.ParticleFlags.ACTIVE | newton.ParticleFlags.FLUID


@wp.kernel
def _count_exact_particle_grid_neighbors(
    grid: wp.uint64,
    positions: wp.array[wp.vec3],
    worlds: wp.array[wp.int32],
    radius: float,
    counts: wp.array[wp.int32],
):
    tid = wp.tid()
    count = int(0)
    for _index in particle_grid_module.particle_grid_query(
        grid,
        positions[tid],
        radius,
        worlds[tid],
        True,
    ):
        count += 1
    counts[tid] = count


def _build_particle_grouping_model(
    device,
    world_particle_counts: tuple[int, ...],
    global_particle_count: int = 0,
):
    scene = newton.ModelBuilder(gravity=0.0)
    for particle_index in range(global_particle_count):
        scene.add_particle(
            pos=(particle_index * SPACING, 0.0, 0.0),
            vel=(0.0, 0.0, 0.0),
            mass=PARTICLE_MASS,
            radius=RADIUS,
            flags=FLUID_FLAGS,
        )

    for particle_count in world_particle_counts:
        world = newton.ModelBuilder(gravity=0.0)
        for particle_index in range(particle_count):
            world.add_particle(
                pos=(particle_index * SPACING, 0.0, 0.0),
                vel=(0.0, 0.0, 0.0),
                mass=PARTICLE_MASS,
                radius=RADIUS,
                flags=FLUID_FLAGS,
            )
        scene.add_world(world)

    return scene.finalize(device=device)


def _particle_grouping_solvers(device, world_particle_counts, global_particle_count=0):
    xpbd_model = _build_particle_grouping_model(device, world_particle_counts, global_particle_count)
    sph_model = _build_particle_grouping_model(device, world_particle_counts, global_particle_count)
    return (
        (
            "XPBD",
            xpbd_model,
            newton.solvers.SolverXPBD(xpbd_model, fluid_rest_distance=SPACING),
        ),
        (
            "SPH",
            sph_model,
            newton.solvers.SolverSPH(sph_model, smoothing_length=2.2 * RADIUS),
        ),
    )


def _assert_particle_grid_grouping(
    test,
    device,
    world_particle_counts: tuple[int, ...],
    expected: bool,
    global_particle_count: int = 0,
):
    for solver_name, model, solver in _particle_grouping_solvers(device, world_particle_counts, global_particle_count):
        test.assertEqual(
            solver._particle_grid_grouped,
            expected,
            f"unexpected grouped-grid eligibility for {solver_name}",
        )
        if expected:
            test.assertIsNotNone(model.particle_grid)
            test.assertEqual(model.particle_grid.device, model.device)


def test_particle_grid_grouping_model_flag(test, device):
    del device
    test.assertEqual(newton.ModelFlags.PARTICLE_PROPERTIES, 1 << 9)
    test.assertTrue(newton.ModelFlags.ALL & newton.ModelFlags.PARTICLE_PROPERTIES)


def test_particle_grid_grouping_requires_multiple_local_particles(test, device):
    _assert_particle_grid_grouping(
        test,
        device,
        (1, 1),
        expected=particle_grid_module._HASH_GRID_GROUPING_SUPPORTED,
    )
    _assert_particle_grid_grouping(test, device, (2,), expected=False)
    _assert_particle_grid_grouping(test, device, (0, 0), expected=False)


def test_particle_grid_grouping_rejects_global_particles(test, device):
    _assert_particle_grid_grouping(test, device, (0, 0), expected=False, global_particle_count=2)
    _assert_particle_grid_grouping(test, device, (1, 1), expected=False, global_particle_count=1)


def test_particle_grid_grouping_allows_empty_local_world(test, device):
    _assert_particle_grid_grouping(
        test,
        device,
        (2, 0),
        expected=particle_grid_module._HASH_GRID_GROUPING_SUPPORTED,
    )


def test_particle_grid_grouping_refreshes_world_ids(test, device):
    if not particle_grid_module._HASH_GRID_GROUPING_SUPPORTED:
        test.skipTest("Warp grouped HashGrid API is unavailable")

    for solver_name, model, solver in _particle_grouping_solvers(device, (1, 1)):
        test.assertTrue(solver._particle_grid_grouped, solver_name)

        model.particle_world.assign([-1, 1])
        solver.notify_model_changed(newton.ModelFlags.PARTICLE_PROPERTIES)
        test.assertFalse(solver._particle_grid_grouped, solver_name)

        model.particle_world.assign([0, 1])
        solver.notify_model_changed(newton.ModelFlags.PARTICLE_PROPERTIES)
        test.assertTrue(solver._particle_grid_grouped, solver_name)


def test_particle_grid_grouping_builds_exact_worlds(test, device):
    if not particle_grid_module._HASH_GRID_GROUPING_SUPPORTED:
        test.skipTest("Warp grouped HashGrid API is unavailable")

    for solver_name, model, solver in _particle_grouping_solvers(device, (1, 1)):
        solver._build_particle_grid(model.particle_q, radius=2.0 * RADIUS)
        counts = wp.zeros(model.particle_count, dtype=wp.int32, device=device)
        wp.launch(
            _count_exact_particle_grid_neighbors,
            dim=model.particle_count,
            inputs=[model.particle_grid.id, model.particle_q, model.particle_world, 2.0 * RADIUS],
            outputs=[counts],
            device=device,
        )
        test.assertTrue(np.array_equal(counts.numpy(), [1, 1]), solver_name)


def test_particle_grid_grouping_falls_back_without_warp_support(test, device):
    original_reserve = wp.HashGrid.reserve
    original_build = wp.HashGrid.build
    reserve_calls = []
    build_calls = []

    def legacy_reserve(grid, point_count):
        reserve_calls.append(point_count)
        return original_reserve(grid, point_count)

    def legacy_build(grid, points, radius):
        build_calls.append(len(points))
        return original_build(grid, points, radius)

    with (
        mock.patch.object(particle_grid_module, "_HASH_GRID_GROUPING_SUPPORTED", False),
        mock.patch.object(wp.HashGrid, "reserve", legacy_reserve),
        mock.patch.object(wp.HashGrid, "build", legacy_build),
    ):
        solvers = _particle_grouping_solvers(device, (1, 1))
        for solver_name, model, solver in solvers:
            test.assertFalse(solver._particle_grid_grouped, solver_name)
            solver._build_particle_grid(model.particle_q, radius=2.0 * RADIUS)

    test.assertEqual(reserve_calls, [2, 2])
    test.assertEqual(build_calls, [2, 2])


def test_xpbd_particle_notification_resets_all_fluid(test, device):
    model = _build_particle_grouping_model(device, (2,))
    solver = newton.solvers.SolverXPBD(model, fluid_rest_distance=SPACING)
    test.assertTrue(solver._all_fluid)

    model.particle_flags.assign([int(newton.ParticleFlags.ACTIVE)] * model.particle_count)
    solver.notify_model_changed(newton.ModelFlags.PARTICLE_PROPERTIES)

    test.assertFalse(solver._has_fluid)
    test.assertFalse(solver._all_fluid)


def test_sph_particle_grid_recreated_during_step(test, device):
    model = _build_particle_grouping_model(device, (1, 1))
    solver = newton.solvers.SolverSPH(model, smoothing_length=2.2 * RADIUS)
    model.particle_grid = None

    solver.step(model.state(), model.state(), control=None, contacts=None, dt=1.0 / 120.0)

    test.assertIsNotNone(model.particle_grid)
    test.assertEqual(solver._particle_grid_grouped, particle_grid_module._HASH_GRID_GROUPING_SUPPORTED)


def test_sph_missing_particle_grid_rejected_during_capture(test, device):
    model = _build_particle_grouping_model(device, (1, 1))
    solver = newton.solvers.SolverSPH(model, smoothing_length=2.2 * RADIUS)
    state_in = model.state()
    state_out = model.state()
    density = solver.particle_density
    model.particle_grid = None

    with (
        mock.patch.object(type(model.device), "is_capturing", new_callable=mock.PropertyMock, return_value=True),
        test.assertRaisesRegex(RuntimeError, "End graph capture.*notify_model_changed.*recapture"),
    ):
        solver.step(state_in, state_out, control=None, contacts=None, dt=1.0 / 120.0)

    test.assertIsNone(model.particle_grid)
    test.assertIs(solver.particle_density, density)


def test_sph_particle_storage_resize_rejected_during_capture(test, device):
    model = _build_particle_grouping_model(device, (1, 1))
    solver = newton.solvers.SolverSPH(model, smoothing_length=2.2 * RADIUS)
    state_in = model.state()
    state_out = model.state()
    density = solver.particle_density
    solver._capacity -= 1

    with (
        mock.patch.object(type(model.device), "is_capturing", new_callable=mock.PropertyMock, return_value=True),
        test.assertRaisesRegex(RuntimeError, "End graph capture.*notify_model_changed.*recapture"),
    ):
        solver.step(state_in, state_out, control=None, contacts=None, dt=1.0 / 120.0)

    test.assertIs(solver.particle_density, density)


def _build_fluid_grid(device, dims=(6, 6, 6), spacing=SPACING, gravity=0.0, ground=False, z0=0.0, fluid=True):
    builder = newton.ModelBuilder(up_axis="Z", gravity=gravity)
    builder.add_particle_grid(
        pos=wp.vec3(0.0, 0.0, z0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=dims[0],
        dim_y=dims[1],
        dim_z=dims[2],
        cell_x=spacing,
        cell_y=spacing,
        cell_z=spacing,
        mass=PARTICLE_MASS,
        jitter=0.0,
        radius_mean=RADIUS,
        flags=FLUID_FLAGS if fluid else newton.ParticleFlags.ACTIVE,
    )
    if ground:
        builder.add_ground_plane()
    return builder.finalize(device=device)


def _simulate(model, steps, ground=False, iterations=3, dt=1.0 / 240.0, **solver_kwargs):
    solver = newton.solvers.SolverXPBD(model, iterations=iterations, fluid_rest_distance=SPACING, **solver_kwargs)
    state_0 = model.state()
    state_1 = model.state()
    contacts = model.contacts() if ground else None
    for _ in range(steps):
        state_0.clear_forces()
        if ground:
            model.collide(state_0, contacts)
        solver.step(state_0, state_1, None, contacts, dt)
        state_0, state_1 = state_1, state_0
    return state_0, solver


def _densities(q, solver):
    """Poly6 SPH density of each particle, normalized by the solver's rest density."""
    h = solver._fluid_h
    d = np.linalg.norm(q[None, :, :] - q[:, None, :], axis=-1)
    w = np.where(d < h, (h * h - d**2) ** 3, 0.0) * 315.0 / (64.0 * np.pi * h**9)
    return PARTICLE_MASS * w.sum(axis=1) / solver._fluid_rest_density_eff


def test_fluid_rest_lattice_stays_at_rest(test, device):
    """A particle grid at rest spacing without gravity must stay bounded.

    The rest density is calibrated from the lattice, so the density constraint is
    inactive in the bulk (see :func:`test_fluid_rest_lattice_exact_without_cohesion`,
    which holds it to machine precision with cohesion off). With the default
    (maximum) cohesion the surface tension does reshape the small cube toward a
    rounder blob -- a few times the rest spacing of contraction -- but it must
    stay contractive and bounded, never gaining energy and blowing up.
    """
    model = _build_fluid_grid(device)
    q0 = model.particle_q.numpy().copy()
    state, _solver = _simulate(model, 240)

    q = state.particle_q.numpy()
    qd = state.particle_qd.numpy()
    test.assertTrue(np.isfinite(q).all())
    drift = np.linalg.norm(q - q0, axis=1).max()
    max_speed = np.linalg.norm(qd, axis=1).max()
    # cohesion reshapes the cube but must stay bounded (a blow-up runs away far
    # past this and also trips the speed check); the speed bound is the real
    # "no energy gain" guard
    test.assertLess(drift, 0.25, f"rest lattice drifted {drift:.4f} m")
    test.assertLess(max_speed, 1.0, f"rest lattice reached {max_speed:.3f} m/s")


def test_fluid_rest_lattice_exact_without_cohesion(test, device):
    """Without cohesion the calibrated rest lattice is an exact equilibrium."""
    model = _build_fluid_grid(device)
    q0 = model.particle_q.numpy().copy()
    state, _solver = _simulate(model, 120, fluid_cohesion=0.0)

    drift = np.linalg.norm(state.particle_q.numpy() - q0, axis=1).max()
    test.assertLess(drift, 1.0e-4, f"rest lattice drifted {drift:.6f} m without cohesion")


def test_fluid_compressed_block_decompresses(test, device):
    """A block compressed to 2x rest density must expand toward rest density."""
    model = _build_fluid_grid(device, spacing=0.8 * SPACING)
    state, solver = _simulate(model, 480, fluid_cohesion=0.5, fluid_viscosity=0.05)

    q = state.particle_q.numpy()
    test.assertTrue(np.isfinite(q).all())
    rho = _densities(q, solver)
    initial_ratio = (SPACING / (0.8 * SPACING)) ** 3  # ~1.95
    test.assertGreater(initial_ratio, 1.9)
    test.assertLess(
        float(np.percentile(rho, 90)),
        1.25,
        "compressed fluid did not decompress toward rest density",
    )


def test_fluid_drop_forms_cohesive_puddle(test, device):
    """A fluid cube dropped on the ground must settle into a compact puddle.

    With cohesion the puddle stays bound (surface tension); without it the
    particles disperse into a thin monolayer. Both must remain stable.
    """
    model = _build_fluid_grid(device, gravity=-9.81, ground=True, z0=0.3)
    state, _solver = _simulate(model, 480, ground=True, fluid_cohesion=1.0, fluid_viscosity=0.05)
    q = state.particle_q.numpy()
    qd = state.particle_qd.numpy()
    test.assertTrue(np.isfinite(q).all())
    test.assertTrue(np.isfinite(qd).all())
    test.assertGreater(q[:, 2].min(), -RADIUS, "particles fell through the ground")
    test.assertLess(np.linalg.norm(qd, axis=1).max(), 1.0, "puddle did not settle")
    spread_cohesive = np.linalg.norm(q[:, :2] - q[:, :2].mean(axis=0), axis=1).max()
    test.assertLess(spread_cohesive, 0.6, "cohesive puddle dispersed too far")

    model = _build_fluid_grid(device, gravity=-9.81, ground=True, z0=0.3)
    state, _solver = _simulate(model, 480, ground=True, fluid_cohesion=0.0, fluid_viscosity=0.05)
    q0 = state.particle_q.numpy()
    test.assertTrue(np.isfinite(q0).all())
    spread_loose = np.linalg.norm(q0[:, :2] - q0[:, :2].mean(axis=0), axis=1).max()
    test.assertGreater(
        spread_loose,
        spread_cohesive,
        "cohesion should reduce how far the splash disperses",
    )


def test_fluid_pair_coheres_without_oscillation(test, device):
    """Two separated fluid particles must pull together to a stable spacing.

    Near-isolated particles have a saturated density deficit; constraint-based
    attraction diverges for them, so this guards the bounded cohesion term.
    """
    builder = newton.ModelBuilder(up_axis="Z", gravity=0.0)
    builder.add_particles(
        pos=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(1.5 * SPACING, 0.0, 0.0)],
        vel=[wp.vec3(0.0)] * 2,
        mass=[PARTICLE_MASS] * 2,
        radius=[RADIUS] * 2,
        flags=[int(FLUID_FLAGS)] * 2,
    )
    model = builder.finalize(device=device)
    state, _solver = _simulate(model, 480)

    q = state.particle_q.numpy()
    qd = state.particle_qd.numpy()
    test.assertTrue(np.isfinite(q).all())
    dist = float(np.linalg.norm(q[1] - q[0]))
    test.assertLess(dist, 1.5 * SPACING, "pair did not cohere")
    test.assertGreater(dist, 0.1 * SPACING, "pair collapsed to a point")
    test.assertLess(np.linalg.norm(qd, axis=1).max(), 0.5, "pair oscillates")


def test_fluid_pairs_skip_contact_constraints(test, device):
    """Fluid-fluid pairs must not generate XPBD contact constraints.

    Fluid particles rest at less than two collision radii apart; if the contact
    kernel also acted on them it would fight the density constraint and push
    them to 2*radius spacing.
    """

    def run(fluid):
        builder = newton.ModelBuilder(up_axis="Z", gravity=0.0)
        flags = FLUID_FLAGS if fluid else newton.ParticleFlags.ACTIVE
        # closer than 2*radius: a contact constraint would push them apart
        builder.add_particles(
            pos=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(1.2 * RADIUS, 0.0, 0.0)],
            vel=[wp.vec3(0.0)] * 2,
            mass=[PARTICLE_MASS] * 2,
            radius=[RADIUS] * 2,
            flags=[int(flags)] * 2,
        )
        model = builder.finalize(device=device)
        state, _solver = _simulate(model, 60, fluid_cohesion=0.0)
        q = state.particle_q.numpy()
        return float(np.linalg.norm(q[1] - q[0]))

    dist_fluid = run(fluid=True)
    dist_solid = run(fluid=False)
    test.assertGreaterEqual(dist_solid, 2.0 * RADIUS - 1.0e-4, "solid contact should separate the pair")
    test.assertLess(dist_fluid, 2.0 * RADIUS - 1.0e-4, "fluid pair must not be separated by contact constraints")


def test_fluid_render_particles(test, device):
    """update_render_particles fills smoothed positions and ellipsoid axes."""
    model = _build_fluid_grid(device)
    state, solver = _simulate(model, 10)

    solver.update_render_particles(state, smoothing=0.5, anisotropy_scale=1.0)
    test.assertIsNotNone(solver.render_positions)
    render_q = solver.render_positions.numpy()
    test.assertEqual(render_q.shape, (model.particle_count, 3))
    test.assertTrue(np.isfinite(render_q).all())
    # smoothed positions must stay near the simulated positions
    err = np.linalg.norm(render_q - state.particle_q.numpy(), axis=1).max()
    test.assertLess(err, solver._fluid_h)
    for aniso in (
        solver.render_anisotropy,
        solver.render_anisotropy_secondary,
        solver.render_anisotropy_tertiary,
    ):
        a = aniso.numpy()
        test.assertEqual(a.shape, (model.particle_count, 4))
        test.assertTrue(np.isfinite(a).all())
        test.assertGreater(a[:, 3].min(), 0.0)


def test_fluid_diffuse_particles_spawn_and_expire(test, device):
    """A splashing drop must emit diffuse foam particles that age and expire."""
    model = _build_fluid_grid(device, gravity=-9.81, ground=True, z0=0.4)
    solver = newton.solvers.SolverXPBD(
        model,
        iterations=3,
        fluid_rest_distance=SPACING,
        fluid_cohesion=0.5,
        fluid_viscosity=0.05,
        max_diffuse_particles=2000,
        diffuse_threshold=1.0,
        diffuse_lifetime=0.5,
    )
    test.assertIsNotNone(solver.diffuse_positions)

    state_0 = model.state()
    state_1 = model.state()
    contacts = model.contacts()
    dt = 1.0 / 240.0

    def alive_count():
        return int(np.count_nonzero(solver.diffuse_positions.numpy()[:, 3] > 0.0))

    # drop and splash: foam must spawn around the impact
    for _ in range(120):
        state_0.clear_forces()
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, None, contacts, dt)
        state_0, state_1 = state_1, state_0
    spawned = int(solver.diffuse_spawn_counter.numpy()[0])
    test.assertGreater(spawned, 0, "splash did not emit diffuse particles")
    alive_after_splash = alive_count()
    test.assertGreater(alive_after_splash, 0, "no live diffuse particles after the splash")

    diffuse_q = solver.diffuse_positions.numpy()
    live = diffuse_q[:, 3] > 0.0
    test.assertTrue(np.isfinite(diffuse_q[live]).all())
    test.assertGreater(diffuse_q[live][:, 2].min(), -2.0 * RADIUS, "diffuse particles fell through the ground")

    # once the fluid settles, spawning stops and the foam expires
    for _ in range(360):
        state_0.clear_forces()
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, None, contacts, dt)
        state_0, state_1 = state_1, state_0
    test.assertLess(alive_count(), alive_after_splash, "diffuse particles did not expire")


def test_fluid_diffuse_disabled_by_default(test, device):
    """Without max_diffuse_particles the foam layer stays unallocated."""
    model = _build_fluid_grid(device)
    _state, solver = _simulate(model, 5)
    test.assertIsNone(solver.diffuse_positions)
    test.assertFalse(solver.diffuse_enabled)


def _watertight_box_mesh(hx, hy, hz):
    """A watertight (outward-wound) box mesh centered at the origin."""
    v = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float32,
    )
    f = np.array(
        [
            [0, 3, 2],
            [0, 2, 1],  # bottom (-z)
            [4, 5, 6],
            [4, 6, 7],  # top (+z)
            [0, 1, 5],
            [0, 5, 4],  # -y
            [2, 3, 7],
            [2, 7, 6],  # +y
            [1, 2, 6],
            [1, 6, 5],  # +x
            [3, 0, 4],
            [3, 4, 7],  # -x
        ],
        dtype=np.int32,
    ).flatten()
    return newton.Mesh(v, f)


def test_fluid_sdf_mesh_contains_particles(test, device):
    """A mesh with a texture SDF should contain fluid via the SDF soft-contact path.

    Exercises ``create_soft_contacts_sdf`` (CUDA-only): fluid dropped onto a
    static SDF box slab must rest on top instead of tunneling through, and the
    slab must be flagged as carrying an SDF.
    """
    if not wp.get_device(device).is_cuda:
        test.skipTest("texture SDFs require CUDA")

    slab_top = 0.1
    mesh = _watertight_box_mesh(0.5, 0.5, 0.5 * slab_top)
    mesh.build_sdf(max_resolution=64, narrow_band_range=(-0.1, 0.1), margin=0.05)

    builder = newton.ModelBuilder(up_axis="Z", gravity=-9.81)
    builder.default_particle_radius = RADIUS
    slab = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5 * slab_top), wp.quat_identity()))
    builder.add_shape_mesh(slab, mesh=mesh, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    builder.body_flags[slab] = int(newton.BodyFlags.KINEMATIC)
    builder.add_particle_grid(
        pos=wp.vec3(-0.1, -0.1, slab_top + RADIUS),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=5,
        dim_y=5,
        dim_z=5,
        cell_x=SPACING,
        cell_y=SPACING,
        cell_z=SPACING,
        mass=PARTICLE_MASS,
        jitter=0.0,
        radius_mean=RADIUS,
        flags=FLUID_FLAGS,
    )
    model = builder.finalize(device=device)
    test.assertGreaterEqual(int(model._shape_sdf_index.numpy()[0]), 0, "slab mesh should carry an SDF")

    solver = newton.solvers.SolverXPBD(model, iterations=3, fluid_rest_distance=SPACING)
    state_0, state_1 = model.state(), model.state()
    contacts = model.contacts()
    dt = 1.0 / 120.0
    for _ in range(120):
        state_0.clear_forces()
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, None, contacts, dt)
        state_0, state_1 = state_1, state_0

    q = state_0.particle_q.numpy()
    test.assertTrue(np.isfinite(q).all())
    # particles must rest on top of the slab, not tunnel through it
    test.assertGreater(q[:, 2].min(), slab_top - RADIUS, "fluid tunneled through the SDF slab")


def test_fluid_max_neighbors_truncates_density(test, device):
    """``fluid_max_neighbors`` caps the neighbor loop: a cap below the local
    count lowers the SPH density estimate, while a cap above it is a no-op.

    ``compute_fluid_lambdas`` writes one entry per thread with no atomics, so
    it is deterministic and can be compared bit-exactly across launches.
    """
    model = _build_fluid_grid(device, dims=(6, 6, 6))
    solver = newton.solvers.SolverXPBD(model, iterations=1, fluid_rest_distance=SPACING)
    state = model.state()
    h = solver._fluid_h
    model.particle_grid.build(state.particle_q, radius=h)
    n = model.particle_count

    def densities_with_cap(cap):
        density = wp.zeros(n, dtype=wp.float32, device=device)
        lam = wp.zeros(n, dtype=wp.float32, device=device)
        wp.launch(
            compute_fluid_lambdas,
            dim=n,
            inputs=[
                model.particle_grid.id,
                state.particle_q,
                model.particle_mass,
                model.particle_inv_mass,
                model.particle_flags,
                model.particle_world,
                solver._particle_grid_grouped,
                h,
                solver._fluid_rest_density_eff,
                solver._fluid_eps,
                cap,
                solver._fluid_rest_distance_eff,
            ],
            outputs=[density, lam],
            device=device,
        )
        return density.numpy()

    uncapped = densities_with_cap(0)
    truncated = densities_with_cap(2)
    above_bulk = densities_with_cap(100000)

    # truncation can only drop the density, and must drop it somewhere
    test.assertTrue(np.all(truncated <= uncapped + 1e-5))
    test.assertLess(float(truncated.max()), float(uncapped.max()))
    # a cap above every particle's neighbor count changes nothing
    test.assertEqual(float(np.abs(above_bulk - uncapped).max()), 0.0)


def test_fluid_coincident_particles_separate(test, device):
    """Near-coincident fluid particles must be driven apart, not fuse into a
    stuck "super particle".

    Two particles a micron apart are under-dense (so the compression-only density
    constraint is inactive) and have an undefined pair direction, so only the
    un-averaged minimum-separation repulsion can pull them apart.
    """
    builder = newton.ModelBuilder(up_axis="Z", gravity=0.0)
    builder.default_particle_radius = RADIUS
    builder.add_particle_grid(
        pos=wp.vec3(0.0, 0.0, 0.5),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=2,
        dim_y=1,
        dim_z=1,
        cell_x=1.0e-6,  # the two particles start essentially coincident
        cell_y=SPACING,
        cell_z=SPACING,
        mass=PARTICLE_MASS,
        jitter=0.0,
        radius_mean=RADIUS,
        flags=FLUID_FLAGS,
    )
    model = builder.finalize(device=device)
    state, _ = _simulate(model, steps=30, iterations=3)
    q = state.particle_q.numpy()
    sep = float(np.linalg.norm(q[0] - q[1]))
    test.assertTrue(np.all(np.isfinite(q)))
    test.assertGreater(sep, 0.25 * SPACING, "coincident fluid particles failed to separate")


def test_body_velocity_clamp_and_sanitize(test, device):
    """``clamp_body_velocities`` caps dynamic-body linear/angular speed, zeroes
    any non-finite component, and leaves static bodies (inv_mass 0) untouched.
    """
    # row layout: [linear xyz (spatial_top), angular xyz (spatial_bottom)]
    qd_np = np.array(
        [
            [1000.0, 0.0, 0.0, 0.0, 0.0, 1000.0],  # dynamic: huge linear + angular
            [np.nan, 0.0, 0.0, 0.0, 0.0, 0.0],  # dynamic: non-finite linear
            [50.0, 50.0, 50.0, 50.0, 50.0, 50.0],  # static: must be untouched
        ],
        dtype=np.float32,
    )
    qd = wp.array(qd_np, dtype=wp.spatial_vector, device=device)
    inv_mass = wp.array([1.0, 1.0, 0.0], dtype=float, device=device)

    wp.launch(clamp_body_velocities, dim=3, inputs=[inv_mass, 10.0, 20.0], outputs=[qd], device=device)
    out = qd.numpy()

    test.assertTrue(np.all(np.isfinite(out)))
    test.assertAlmostEqual(float(np.linalg.norm(out[0, :3])), 10.0, places=4)  # linear clamped
    test.assertAlmostEqual(float(np.linalg.norm(out[0, 3:])), 20.0, places=4)  # angular clamped
    test.assertEqual(float(np.abs(out[1, :3]).max()), 0.0)  # non-finite -> zero
    test.assertTrue(np.array_equal(out[2], qd_np[2]))  # static body untouched


def test_particle_delta_self_heals_nan(test, device):
    """A non-finite position correction must be reset to the pre-step position
    at rest rather than propagated by :func:`apply_particle_deltas`."""
    x0 = wp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
    xp = wp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
    flags = wp.array([int(FLUID_FLAGS), int(FLUID_FLAGS)], dtype=wp.int32, device=device)
    delta = wp.array([[0.01, 0.0, 0.0], [np.nan, 0.0, 0.0]], dtype=wp.vec3, device=device)
    x_out = wp.zeros(2, dtype=wp.vec3, device=device)
    v_out = wp.zeros(2, dtype=wp.vec3, device=device)

    wp.launch(
        apply_particle_deltas,
        dim=2,
        inputs=[x0, xp, flags, delta, 1.0 / 60.0, 100.0],
        outputs=[x_out, v_out],
        device=device,
    )
    xo = x_out.numpy()
    vo = v_out.numpy()
    test.assertTrue(np.all(np.isfinite(xo)) and np.all(np.isfinite(vo)))
    # the healed particle returns to its pre-step position at rest
    test.assertTrue(np.allclose(xo[1], [1.0, 0.0, 0.0]))
    test.assertEqual(float(np.abs(vo[1]).max()), 0.0)


def _particle_records(model, state):
    """Full per-particle record (q, qd, mass, radius, flags) for relabel checks."""
    return np.concatenate(
        [
            state.particle_q.numpy(),
            state.particle_qd.numpy(),
            model.particle_mass.numpy()[:, None],
            model.particle_radius.numpy()[:, None],
            model.particle_flags.numpy()[:, None].astype(np.float64),
        ],
        axis=1,
    )


def test_fluid_reorder_is_pure_relabel(test, device):
    """reorder_particles must permute particles without changing the data.

    The set of per-particle records must be bit-identical before and after, the
    order must actually change (a lattice's row-major order differs from Morton
    order), and the q/qd/mass coupling must stay intact.
    """
    model = _build_fluid_grid(device, dims=(5, 5, 5), gravity=-9.81, ground=True, z0=0.3)
    state, solver = _simulate(model, steps=15, ground=True)

    before = _particle_records(model, state)
    solver.reorder_particles(state)
    after = _particle_records(model, state)

    # the multiset of records is unchanged (sort each lexicographically, compare)
    sorted_before = before[np.lexsort(before.T[::-1])]
    sorted_after = after[np.lexsort(after.T[::-1])]
    test.assertEqual(float(np.abs(sorted_before - sorted_after).max()), 0.0)
    # the order genuinely changed (otherwise the test proves nothing)
    test.assertTrue(bool(np.any(np.abs(before - after).max(axis=1) > 0.0)))
    # a step after reorder must still integrate to a finite state
    state_1 = model.state()
    contacts = model.contacts()
    state.clear_forces()
    model.collide(state, contacts)
    solver.step(state, state_1, None, contacts, 1.0 / 240.0)
    test.assertTrue(np.isfinite(state_1.particle_q.numpy()).all())


def test_fluid_reorder_noop_when_not_all_fluid(test, device):
    """reorder_particles must leave non-fluid scenes untouched (it would
    otherwise scramble the index-based topology of cloth/soft bodies)."""
    model = _build_fluid_grid(device, dims=(4, 4, 4), fluid=False)
    solver = newton.solvers.SolverXPBD(model, iterations=2, fluid_rest_distance=SPACING)
    state = model.state()
    before = state.particle_q.numpy().copy()
    solver.reorder_particles(state)
    test.assertEqual(float(np.abs(before - state.particle_q.numpy()).max()), 0.0)


def test_xpbd_grouped_cuda_graph_capture(test, device):
    if not wp.get_device(device).is_cuda:
        test.skipTest("CUDA graph capture requires a CUDA device")
    if not particle_grid_module._HASH_GRID_GROUPING_SUPPORTED:
        test.skipTest("Warp grouped HashGrid API is unavailable")

    world = newton.ModelBuilder(gravity=-9.81)
    world.add_particle_grid(
        pos=wp.vec3(-0.075, -0.075, 0.1),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.2, 0.0, 0.0),
        dim_x=4,
        dim_y=4,
        dim_z=3,
        cell_x=SPACING,
        cell_y=SPACING,
        cell_z=SPACING,
        mass=PARTICLE_MASS,
        jitter=0.0,
        radius_mean=RADIUS,
        radius_std=0.0,
        flags=FLUID_FLAGS,
    )
    scene = newton.ModelBuilder(gravity=-9.81)
    scene.replicate(world, 2)
    model = scene.finalize(device=device)
    solver = newton.solvers.SolverXPBD(
        model,
        iterations=2,
        fluid_rest_distance=SPACING,
        fluid_cohesion=0.0,
    )

    test.assertEqual(model.world_count, 2)
    test.assertTrue(solver._particle_grid_grouped)
    particles_per_world = model.particle_count // model.world_count
    worlds = model.particle_world.numpy().reshape(model.world_count, particles_per_world)
    expected_worlds = np.broadcast_to(
        np.arange(model.world_count, dtype=np.int32)[:, None],
        worlds.shape,
    )
    np.testing.assert_array_equal(worlds, expected_worlds)

    state_0 = model.state()
    state_1 = model.state()
    initial_positions = state_0.particle_q.numpy().reshape(model.world_count, particles_per_world, 3)
    solver.step(state_0, state_1, control=None, contacts=None, dt=1.0 / 240.0)
    state_0, state_1 = state_1, state_0
    wp.synchronize_device(device)

    with wp.ScopedCapture(device=device) as capture:
        state_0.clear_forces()
        solver.step(state_0, state_1, control=None, contacts=None, dt=1.0 / 240.0)

    wp.capture_launch(capture.graph)
    positions = state_1.particle_q.numpy().reshape(model.world_count, particles_per_world, 3)
    velocities = state_1.particle_qd.numpy().reshape(model.world_count, particles_per_world, 3)
    test.assertTrue(np.isfinite(positions).all())
    test.assertTrue(np.isfinite(velocities).all())
    test.assertGreater(float(np.linalg.norm(positions - initial_positions)), 1.0e-6)
    np.testing.assert_allclose(positions[0], positions[1], atol=1.0e-6, rtol=0.0)
    np.testing.assert_allclose(velocities[0], velocities[1], atol=1.0e-6, rtol=0.0)


devices = get_test_devices()


class TestSolverXPBDFluid(unittest.TestCase):
    pass


for _name in (
    "test_particle_grid_grouping_model_flag",
    "test_particle_grid_grouping_requires_multiple_local_particles",
    "test_particle_grid_grouping_rejects_global_particles",
    "test_particle_grid_grouping_allows_empty_local_world",
    "test_particle_grid_grouping_refreshes_world_ids",
    "test_particle_grid_grouping_builds_exact_worlds",
    "test_particle_grid_grouping_falls_back_without_warp_support",
    "test_xpbd_particle_notification_resets_all_fluid",
    "test_sph_particle_grid_recreated_during_step",
    "test_sph_missing_particle_grid_rejected_during_capture",
    "test_sph_particle_storage_resize_rejected_during_capture",
    "test_fluid_rest_lattice_stays_at_rest",
    "test_fluid_rest_lattice_exact_without_cohesion",
    "test_fluid_compressed_block_decompresses",
    "test_fluid_drop_forms_cohesive_puddle",
    "test_fluid_pair_coheres_without_oscillation",
    "test_fluid_pairs_skip_contact_constraints",
    "test_fluid_render_particles",
    "test_fluid_diffuse_particles_spawn_and_expire",
    "test_fluid_diffuse_disabled_by_default",
    "test_fluid_sdf_mesh_contains_particles",
    "test_fluid_reorder_is_pure_relabel",
    "test_fluid_reorder_noop_when_not_all_fluid",
    "test_xpbd_grouped_cuda_graph_capture",
    "test_fluid_max_neighbors_truncates_density",
    "test_fluid_coincident_particles_separate",
    "test_body_velocity_clamp_and_sanitize",
    "test_particle_delta_self_heals_nan",
):
    add_function_test(
        TestSolverXPBDFluid,
        _name,
        globals()[_name],
        devices=devices,
        check_output=False,
    )


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
