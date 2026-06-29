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
