# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test live MPM resolution changes and swept particle projection."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.implicit_mpm.rasterized_collisions import (
    _ALL_COLLIDER_WORLDS,
    Collider,
    _swept_collider_contact,
)
from newton.solvers import SolverImplicitMPM
from newton.tests.unittest_utils import (
    NewtonTestCase,
    add_function_test,
    get_selected_cuda_test_devices,
    get_test_devices,
)


def _make_solver(
    device,
    *,
    moving_wall=False,
    wall_positions=(0.0,),
    particle_positions=((0.5, 0.0, 0.0),),
    grid_type=None,
    integration_scheme="pic",
):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    for position in particle_positions:
        builder.add_particle(position, (0.0, 0.0, 0.0), mass=0.01, radius=0.025)
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


def _particle_snapshot(model, state):
    arrays = {"particle_q": state.particle_q, "particle_qd": state.particle_qd, "particle_mass": model.particle_mass}
    if state.particle_f is not None:
        arrays["particle_f"] = state.particle_f
    arrays.update({f"mpm.{name}": value for name, value in vars(state.mpm).items() if isinstance(value, wp.array)})
    return {name: array.numpy().copy() for name, array in arrays.items()}


def _assert_particle_snapshot(model, state, expected):
    actual = _particle_snapshot(model, state)
    for name, values in expected.items():
        np.testing.assert_array_equal(actual[name], values, err_msg=name)


@wp.kernel
def _swept_material(collider: Collider, body_q: wp.array[wp.transform], material: wp.array[int]):
    _cid, mid, _point, _normal, _velocity = _swept_collider_contact(
        wp.vec3(0.5, 0.0, 0.0), wp.vec3(-0.5, 0.2, 0.0), _ALL_COLLIDER_WORLDS, collider, body_q, body_q, 0.1, True
    )
    material[0] = mid


def test_swept_projection_thin_wall(test, device):
    """Stop a tunneling particle at the earliest wall, independent of collider order."""
    # Separate stationary bodies keep the farther wall first in the collider list.
    model, solver = _make_solver(device, moving_wall=True, wall_positions=(0.0, 0.2))
    previous, current = model.state(), model.state()
    previous.particle_q.fill_(wp.vec3(0.5, 0.0, 0.0))
    current.particle_q.fill_(wp.vec3(-0.5, 0.0, 0.0))
    current.particle_qd.fill_(wp.vec3(-10.0, 0.0, 0.0))

    # Endpoint-only projection cannot see the wall after the particle crosses it.
    solver.project_outside(current, current, 0.1)
    test.assertLess(current.particle_q.numpy()[0, 0], -0.4)

    solver.project_outside(current, current, 0.1, state_prev=previous)
    np.testing.assert_allclose(current.particle_q.numpy()[0], (0.210001, 0.0, 0.0), atol=2.0e-6)
    np.testing.assert_allclose(current.particle_qd.numpy()[0], 0.0, atol=1.0e-6)

    # The nearer mesh's local material zero is not the first collider's material.
    solver.collider_materials(1)["friction"].fill_(1.0)
    current.particle_q.fill_(wp.vec3(-0.5, 0.2, 0.0))
    current.particle_qd.fill_(wp.vec3(-10.0, 2.0, 0.0))
    solver.project_outside(current, current, 0.1, state_prev=previous)
    np.testing.assert_allclose(current.particle_qd.numpy(), 0.0, atol=1.0e-6)

    material = wp.empty(1, dtype=int, device=device)
    first, _count = solver._mpm_model.collider_material_ranges[1]
    for local_id in (-9, 0, 999):
        solver.collider_face_materials(1).fill_(local_id)
        wp.launch(_swept_material, dim=1, inputs=[solver._mpm_model.collider, current.body_q, material], device=device)
        test.assertEqual(int(material.numpy()[0]), first)


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
    np.testing.assert_allclose(position, (0.010001, 0.010001, 0.0), atol=2.0e-6)
    np.testing.assert_allclose(current.particle_qd.numpy()[0], 0.0, atol=1.0e-6)


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
            faces = solver.collider_face_materials(0)
            materials = solver.collider_materials(0)
            material_pointers = {name: value.ptr for name, value in materials.items()}
            materials["friction"].fill_(0.3)
            current, next_state = model.state(), model.state()
            current.particle_q.fill_(wp.vec3(0.55, 0.02, 0.03))
            current.particle_qd.fill_(wp.vec3(0.1, 0.2, 0.3))
            current.particle_f.fill_(wp.vec3(0.01, 0.0, 0.0))
            current.mpm.particle_stress.fill_(wp.mat33(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1))
            current.mpm.particle_Jp.fill_(0.9)
            if integration_scheme == "pic":
                grid = solver._scratchpad.grid
                for value in (0.0, -0.1, np.inf, np.nan, 1.0e100, 1.0e-100, 1.0e20, 1.0e-20):
                    with test.subTest(value=value), test.assertRaises(ValueError):
                        solver.set_voxel_size(value, state=current)
                    test.assertEqual(solver.voxel_size, 0.1)
                    test.assertIs(solver._scratchpad.grid, grid)
                solver.set_voxel_size(0.1, state=current)
                test.assertIs(solver._scratchpad.grid, grid)
            for voxel_size in (0.05, 0.15, 0.1):
                with test.subTest(voxel_size=voxel_size):
                    before = _particle_snapshot(model, current)
                    old_grid = solver._scratchpad.grid
                    solver.set_voxel_size(voxel_size, state=current)
                    test.assertEqual(solver.voxel_size, voxel_size)
                    test.assertIsNot(solver._scratchpad.grid, old_grid)
                    _assert_particle_snapshot(model, current, before)
                    test.assertEqual(solver.collider_face_materials(0).ptr, faces.ptr)
                    test.assertEqual(
                        {name: value.ptr for name, value in solver.collider_materials(0).items()}, material_pointers
                    )
                    solver.step(current, next_state, None, None, 0.001)
                    solver.check_sparse_grid_rebuild_status()
                    test.assertTrue(np.isfinite(next_state.particle_q.numpy()).all())
                    test.assertGreater(next_state.particle_q.numpy()[0, 0], 0.5)
                    current, next_state = next_state, current

            solver.set_particle_radius(0.02)
            test.assertEqual(solver.collider_face_materials(0).ptr, faces.ptr)
            test.assertEqual(
                {name: value.ptr for name, value in solver.collider_materials(0).items()}, material_pointers
            )
            materials["friction"].fill_(0.4)
            np.testing.assert_allclose(solver.collider_materials(0)["friction"].numpy(), 0.4)
            solver.reset(current)
            current.particle_qd.zero_()
            current.particle_f.fill_(wp.vec3(0.01, 0.0, 0.0))
            solver.step(current, next_state, None, None, 0.001)
            test.assertGreater(float(next_state.particle_qd.numpy()[0, 0]), 0.0005)


def test_radius_preserves_particle_state(test, device):
    """Update every MPM support radius and density without resetting particle history."""
    model, solver = _make_solver(device, particle_positions=((0.5, 0.0, 0.0), (0.65, 0.0, 0.0)))
    current, next_state = model.state(), model.state()
    current.particle_qd.fill_(wp.vec3(0.1, 0.2, 0.3))
    current.mpm.particle_stress.fill_(wp.mat33(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1))
    current.mpm.particle_Jp.fill_(0.9)
    before = _particle_snapshot(model, current)
    grid = solver._scratchpad.grid
    original_radii = model.particle_radius.numpy().copy()
    for radius in (0.0, -0.01, np.inf, np.nan, 1.0e100, 1.0e-100, 1.0e20, 1.0e-20):
        with test.subTest(radius=radius), test.assertRaises(ValueError):
            solver.set_particle_radius(radius)
        np.testing.assert_array_equal(model.particle_radius.numpy(), original_radii)
        test.assertAlmostEqual(model.particle_max_radius, 0.025)
        test.assertIs(solver._scratchpad.grid, grid)
        _assert_particle_snapshot(model, current, before)

    for radius in (0.02, 0.03, 0.025):
        with test.subTest(radius=radius):
            before = _particle_snapshot(model, current)
            solver.set_particle_radius(radius)
            _assert_particle_snapshot(model, current, before)
            np.testing.assert_allclose(model.particle_radius.numpy(), radius)
            test.assertAlmostEqual(model.particle_max_radius, radius)
            np.testing.assert_allclose(solver._mpm_model.particle_volume.numpy(), 8.0 * radius**3)
            np.testing.assert_allclose(
                solver._mpm_model.particle_density.numpy(), before["particle_mass"] / (8.0 * radius**3)
            )
            solver.step(current, next_state, None, None, 0.001)
            solver.check_sparse_grid_rebuild_status()
            test.assertTrue(np.isfinite(next_state.particle_q.numpy()).all())
            test.assertTrue(np.isfinite(next_state.particle_qd.numpy()).all())
            current, next_state = next_state, current


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
    state.particle_qd.fill_(wp.vec3(0.1, 0.2, 0.3))
    state.mpm.particle_stress.fill_(wp.mat33(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1))
    state.mpm.particle_Jp.fill_(0.9)
    before = _particle_snapshot(model, state)
    grid = solver._scratchpad.grid
    with test.assertRaisesRegex(RuntimeError, "capacity"):
        solver.set_voxel_size(0.01, state=state)
    test.assertEqual(solver.voxel_size, 1.0)
    test.assertIs(solver._scratchpad.grid, grid)
    solver.check_sparse_grid_rebuild_status()
    _assert_particle_snapshot(model, state, before)
    solver.step(state, state, None, None, 0.001)
    test.assertTrue(np.isfinite(state.particle_q.numpy()).all())

    # A later successful coarsening must not hide a failure from a prior step.
    scattered = wp.array([(2.0 * i, 0.01, 0.01) for i in range(8)], dtype=wp.vec3, device=device)
    wp.copy(state.particle_q, scattered)
    test.expectOutputRegex(r"^Warning: Implicit MPM sparse grid rebuild failed with status 8\.\n", stream="stdout")
    solver.step(state, state, None, None, 0.001)
    with test.assertRaisesRegex(RuntimeError, "capacity"):
        solver.set_voxel_size(20.0, state=state)
    test.assertEqual(solver.voxel_size, 1.0)
    with test.assertRaisesRegex(RuntimeError, "capacity"):
        solver.check_sparse_grid_rebuild_status()
    solver.reset(state)
    solver.set_voxel_size(20.0, state=state)
    solver.check_sparse_grid_rebuild_status()


def test_resize_capture_rejection(test, device):
    """Reject voxel and radius changes during capture before altering live state."""
    model, solver = _make_solver(device)
    state = model.state()
    grid = solver._scratchpad.grid
    before = _particle_snapshot(model, state)
    radii = model.particle_radius.numpy().copy()
    with wp.ScopedCapture(device=device):
        with test.assertRaisesRegex(RuntimeError, "capture"):
            solver.set_voxel_size(0.05, state=state)
        with test.assertRaisesRegex(RuntimeError, "capture"):
            solver.set_particle_radius(0.02)
    test.assertEqual(solver.voxel_size, 0.1)
    test.assertIs(solver._scratchpad.grid, grid)
    test.assertAlmostEqual(model.particle_max_radius, 0.025)
    np.testing.assert_array_equal(model.particle_radius.numpy(), radii)
    _assert_particle_snapshot(model, state, before)


def test_bowl_graph_and_runtime_changes(test, device):
    """Preserve the MPM bowl state across voxel and radius changes with graph recapture."""
    from newton.examples.mpm.example_mpm_bowl import Example  # noqa: PLC0415
    from newton.viewer import ViewerNull  # noqa: PLC0415

    if not wp.is_mempool_enabled(device) or not wp.is_conditional_graph_supported():
        test.skipTest("MPM graph capture requires a memory pool and conditional graph support")

    with wp.ScopedDevice(device):
        args = Example.create_parser().parse_args(["--particle-radius", "0.04", "--voxel-size", "0.1"])
        example = Example(ViewerNull(), args)
        test.assertIsNotNone(example.graph)
        example.step()

        for update in (
            lambda: example.set_voxel_size(0.08),
            lambda: example.set_particle_radius(0.036),
            example.capture,
        ):
            before = _particle_snapshot(example.model, example.state_0)
            sim_time = example.sim_time
            old_graph = example.graph
            update()
            _assert_particle_snapshot(example.model, example.state_0, before)
            test.assertEqual(example.sim_time, sim_time)
            test.assertAlmostEqual(float(example.sim_clock.numpy()[0]), sim_time, places=6)
            test.assertIsNotNone(example.graph)
            test.assertIsNot(example.graph, old_graph)
            example.step()
            test.assertGreater(example.sim_time, sim_time)
            test.assertFalse(np.array_equal(example.state_0.particle_q.numpy(), before["particle_q"]))
            test.assertAlmostEqual(float(example.sim_clock.numpy()[0]), example.sim_time, places=6)
            example.test_post_step()
            example.mpm_solver.check_sparse_grid_rebuild_status()

        test.assertAlmostEqual(example.mpm_solver.voxel_size, 0.08)
        test.assertAlmostEqual(example.model.particle_max_radius, 0.036)
        np.testing.assert_allclose(example.model.particle_radius.numpy(), 0.036)


class TestImplicitMPMAdaptive(NewtonTestCase):
    def setUp(self):
        self.allowOutputRegex(r"^Module .* load on device .*\n?", stream="stdout")


for _device in get_test_devices():
    for _test in (
        test_swept_projection_thin_wall,
        test_swept_projection_corner,
        test_swept_projection_moving_wall,
        test_swept_projection_inactive,
        test_resize_preserves_particle_state,
        test_radius_preserves_particle_state,
        test_resize_fixed_grid,
        test_swept_projection_world_isolation,
    ):
        add_function_test(TestImplicitMPMAdaptive, _test.__name__, _test, devices=[_device])

for _test in (test_resize_capacity_rollback, test_resize_capture_rejection, test_bowl_graph_and_runtime_changes):
    add_function_test(TestImplicitMPMAdaptive, _test.__name__, _test, devices=get_selected_cuda_test_devices())


if __name__ == "__main__":
    unittest.main()
