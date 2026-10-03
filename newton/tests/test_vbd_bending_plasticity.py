# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check optional VBD bending plasticity through the public solver API."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd.particle_vbd_kernels import evaluate_dihedral_angle_based_bending_force_hessian
from newton.solvers import SolverVBD, SolverXPBD
from newton.solvers.experimental.coupled import SolverCoupledProxy
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _hinge_points(angle):
    return np.array(
        [[0.0, 1.0, 0.0], [0.0, -math.cos(angle), -math.sin(angle)], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float32,
    )


def _hinge_builder(*, yield_angle=0.2):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverVBD.register_custom_attributes(builder)
    for point in _hinge_points(0.0):
        builder.add_particle(wp.vec3(point), wp.vec3(0.0), 0.0)
    builder.add_triangle(0, 2, 3, tri_ke=0.0, tri_ka=0.0, tri_kd=0.0)
    builder.add_triangle(1, 3, 2, tri_ke=0.0, tri_ka=0.0, tri_kd=0.0)
    builder.add_edge(
        0, 1, 2, 3, rest=0.0, edge_ke=1.0, edge_kd=0.0, custom_attributes={"vbd:edge_bending_yield_angle": yield_angle}
    )
    return builder


def _make_hinge(device, *, enabled=True, yield_angle=0.2, tiled=False):
    builder = _hinge_builder(yield_angle=yield_angle)
    builder.color()
    model = builder.finalize(device=device)
    solver = SolverVBD(
        model, iterations=1, particle_enable_tile_solve=tiled, particle_enable_bending_plasticity=enabled
    )
    return model, solver, model.state(), model.state()


def _step_at(solver, state_in, state_out, angle):
    state_in.particle_q.assign(_hinge_points(angle))
    solver.step(state_in, state_out, None, None, 1.0 / 240.0)


def test_bending_plasticity_loading(test, device):
    """Yield, unload, and reverse while retaining elastic springback without creep."""
    model, solver, state_a, state_b = _make_hinge(device)
    for angle, expected in ((0.1, 0.0), (1.0, 0.8), (1.0, 0.8), (0.8, 0.8), (-1.0, -0.8), (-1.0, -0.8)):
        _step_at(solver, state_a, state_b, angle)
        np.testing.assert_allclose(state_b.vbd.edge_plastic_angle.numpy(), [expected], atol=1.0e-6)
        np.testing.assert_array_equal(model.edge_rest_angle.numpy(), [0.0])
        state_a, state_b = state_b, state_a


def test_bending_plasticity_state_isolation(test, device):
    """Keep material history in each State when states and solvers share one Model."""
    model, solver, state_a, state_b = _make_hinge(device)
    _step_at(solver, state_a, state_b, 1.0)
    np.testing.assert_array_equal(state_a.vbd.edge_plastic_angle.numpy(), [0.0])
    fresh_a, fresh_b = model.state(), model.state()
    _step_at(solver, fresh_a, fresh_b, 0.1)
    np.testing.assert_array_equal(fresh_b.vbd.edge_plastic_angle.numpy(), [0.0])
    other = SolverVBD(model, particle_enable_bending_plasticity=True, particle_enable_tile_solve=False)
    _step_at(other, fresh_a, fresh_b, -1.0)
    np.testing.assert_allclose(fresh_b.vbd.edge_plastic_angle.numpy(), [-0.8], atol=1.0e-6)
    np.testing.assert_allclose(state_b.vbd.edge_plastic_angle.numpy(), [0.8], atol=1.0e-6)
    solver.step(state_b, state_b, None, None, 1.0 / 240.0)
    np.testing.assert_allclose(state_b.vbd.edge_plastic_angle.numpy(), [0.8], atol=1.0e-6)


def test_bending_plasticity_disabled(test, device):
    """Preserve the default elastic solve even when yield parameters are authored."""
    builder = _hinge_builder()
    builder.particle_mass[1] = 1.0
    builder.color()
    model = builder.finalize(device=device)
    default = SolverVBD(model, particle_enable_tile_solve=False)
    disabled = SolverVBD(model, particle_enable_bending_plasticity=False, particle_enable_tile_solve=False)
    a, b, c = model.state(), model.state(), model.state()
    a.particle_q.assign(_hinge_points(1.0))
    default.step(a, b, None, None, 0.01)
    a.particle_q.assign(_hinge_points(1.0))
    disabled.step(a, c, None, None, 0.01)
    np.testing.assert_array_equal(b.particle_q.numpy(), c.particle_q.numpy())
    np.testing.assert_array_equal(b.particle_qd.numpy(), c.particle_qd.numpy())
    np.testing.assert_array_equal(b.vbd.edge_plastic_angle.numpy(), [0.0])
    np.testing.assert_array_equal(model.edge_rest_angle.numpy(), [0.0])


def test_bending_plasticity_limits(test, device):
    """Support zero yield, elastic infinite yield, and inert degenerate hinges."""
    for threshold, expected in ((0.0, 1.0), (math.inf, 0.0)):
        _, solver, a, b = _make_hinge(device, yield_angle=threshold)
        _step_at(solver, a, b, 1.0)
        np.testing.assert_allclose(b.vbd.edge_plastic_angle.numpy(), [expected], atol=1.0e-6)
    _, solver, a, b = _make_hinge(device)
    a.particle_q.zero_()
    solver.step(a, b, None, None, 0.01)
    np.testing.assert_array_equal(b.vbd.edge_plastic_angle.numpy(), [0.0])
    builder = _hinge_builder()
    for opposite, stiffness in ((-1, 1.0), (0, 0.0)):
        builder.add_edge(
            opposite,
            1,
            2,
            3,
            rest=0.0,
            edge_ke=stiffness,
            edge_kd=0.0,
            custom_attributes={"vbd:edge_bending_yield_angle": 0.2},
        )
    builder.color()
    model = builder.finalize(device=device)
    solver = SolverVBD(model, particle_enable_bending_plasticity=True, particle_enable_tile_solve=False)
    a, b = model.state(), model.state()
    a.vbd.edge_plastic_angle.assign(np.array([0.1, 0.3, 0.5], dtype=np.float32))
    _step_at(solver, a, b, 1.0)
    np.testing.assert_allclose(b.vbd.edge_plastic_angle.numpy(), [0.8, 0.3, 0.5], atol=1.0e-6)


def test_bending_plasticity_springback(test, device):
    """Recover the elastic part of a released crease with the actual bending forces."""
    builder = _hinge_builder()
    builder.particle_mass[1] = 1.0
    builder.edge_bending_properties[0] = (100.0, 4.0)
    builder.color()
    model = builder.finalize(device=device)
    model.particle_flags.assign(np.zeros(4, dtype=np.int32))
    solver = SolverVBD(
        model, iterations=10, particle_enable_bending_plasticity=True, particle_enable_tile_solve=device.is_cuda
    )
    a, b = model.state(), model.state()
    _step_at(solver, a, b, 1.0)
    a, b = b, a
    np.testing.assert_allclose(a.vbd.edge_plastic_angle.numpy(), [0.8], atol=1.0e-6)
    model.particle_flags.assign(np.full(4, int(newton.ParticleFlags.ACTIVE), dtype=np.int32))
    for _ in range(480):
        solver.step(a, b, None, None, 1.0 / 240.0)
        a, b = b, a
    p = a.particle_q.numpy()[1]
    theta = math.atan2(-p[2], -p[1])
    test.assertAlmostEqual(theta, 0.8, delta=0.01)
    np.testing.assert_allclose(a.vbd.edge_plastic_angle.numpy(), [0.8], atol=1.0e-5)
    np.testing.assert_array_equal(model.edge_rest_angle.numpy(), [0.0])


@wp.kernel
def _bending_force_probe(
    positions: wp.array[wp.vec3],
    edges: wp.array2d[int],
    rest: wp.array[float],
    length: wp.array[float],
    force: wp.array[wp.vec3],
    hessian: wp.array[wp.mat33],
):
    f, h = evaluate_dihedral_angle_based_bending_force_hessian(
        0, 1, positions, positions, edges, rest, length, 1.0, 0.0, 0.01, True
    )
    force[0] = f
    hessian[0] = h


def test_bending_plasticity_angle_branch(test, device):
    """Keep crease history and bending forces continuous across the dihedral branch cut."""
    model, solver, a, b = _make_hinge(device)
    a.vbd.edge_plastic_angle.fill_(math.pi - 0.1)
    _step_at(solver, a, b, -math.pi + 0.05)
    np.testing.assert_allclose(b.vbd.edge_plastic_angle.numpy(), [math.pi - 0.1], atol=1.0e-6)
    _step_at(solver, b, a, -math.pi + 0.3)
    np.testing.assert_allclose(a.vbd.edge_plastic_angle.numpy(), [math.pi + 0.1], atol=1.0e-6)
    forces = []
    for rest_angle in (math.pi - 0.01, -math.pi - 0.01):
        positions = wp.array(_hinge_points(-math.pi + 0.01), dtype=wp.vec3, device=device)
        rest = wp.array([rest_angle], dtype=float, device=device)
        force = wp.zeros(1, dtype=wp.vec3, device=device)
        hessian = wp.zeros(1, dtype=wp.mat33, device=device)
        wp.launch(
            _bending_force_probe,
            1,
            inputs=[positions, model.edge_indices, rest, model.edge_rest_length],
            outputs=[force, hessian],
            device=device,
        )
        forces.append(force.numpy())
        test.assertTrue(np.all(np.isfinite(hessian.numpy())))
    np.testing.assert_allclose(forces[0], forces[1], atol=1.0e-6)
    test.assertLess(np.linalg.norm(forces[0]), 0.03)


def test_bending_plasticity_missing_state(test, device):
    """Reject incomplete history before mutating a step or reset's particle positions."""
    _, solver, a, b = _make_hinge(device)
    a.particle_q.assign(_hinge_points(1.0))
    a.vbd.edge_plastic_angle = wp.zeros(0, dtype=float, device=device)
    before = a.particle_q.numpy()
    with test.assertRaisesRegex(ValueError, "edge_plastic_angle"):
        solver.step(a, b, None, None, 0.01)
    np.testing.assert_array_equal(a.particle_q.numpy(), before)
    with test.assertRaisesRegex(ValueError, "edge_plastic_angle"):
        solver.reset(a)
    np.testing.assert_array_equal(a.particle_q.numpy(), before)


def test_bending_plasticity_empty_model(test, device):
    """Accept an enabled solver with no bending edges without extra state registration."""
    model = newton.ModelBuilder().finalize(device=device)
    solver = SolverVBD(model, particle_enable_bending_plasticity=True)
    a, b = model.state(), model.state()
    solver.step(a, b, None, None, 0.01)
    solver.reset(a)


def test_bending_plasticity_autodiff(test, device):
    """Reject automatic differentiation explicitly for the nonsmooth plastic mode."""
    builder = _hinge_builder()
    builder.color()
    model = builder.finalize(device=device, requires_grad=True)
    with test.assertRaisesRegex(ValueError, "automatic differentiation"):
        SolverVBD(model, particle_enable_bending_plasticity=True)
    model, solver, a, _ = _make_hinge(device)
    with test.assertRaisesRegex(ValueError, "automatic differentiation"):
        solver.step(a, model.state(requires_grad=True), None, None, 0.01)


def test_bending_plasticity_coupled_reset(test, device):
    """Gather and reset plastic material history through the native proxy coupler."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.replicate(_hinge_builder(), world_count=2, spacing=(2.0, 0.0, 0.0))
    builder.color()
    model = builder.finalize(device=device)
    model.vbd.edge_bending_yield_angle.assign(np.array([0.2, 0.4], dtype=np.float32))
    particles = list(range(model.particle_count))
    solver = SolverCoupledProxy(
        model,
        coupling=SolverCoupledProxy.Config(
            proxies=[SolverCoupledProxy.Proxy(source="shell", destination="proxy", particles=particles)],
            iterations=3,
        ),
        entries=[
            SolverCoupledProxy.Entry(
                name="shell",
                particles=particles,
                solver=lambda view: SolverVBD(
                    view, particle_enable_bending_plasticity=True, particle_enable_tile_solve=False
                ),
            ),
            SolverCoupledProxy.Entry(name="proxy", solver=lambda view: SolverXPBD(view, iterations=0)),
        ],
    )
    a, b = model.state(), model.state()
    a.particle_q.assign(np.vstack((_hinge_points(1.0), _hinge_points(-0.9) + np.array([2.0, 0.0, 0.0]))))
    solver.step(a, b, None, None, 0.01)
    np.testing.assert_allclose(b.vbd.edge_plastic_angle.numpy(), [0.8, -0.5], atol=1.0e-6)
    if device.is_cuda:
        with wp.ScopedCapture(device=device) as capture:
            solver.step(a, b, None, None, 0.01)
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(b.vbd.edge_plastic_angle.numpy(), [0.8, -0.5], atol=1.0e-6)
    mask = wp.array([True, False, False], dtype=wp.bool, device=device)
    solver.reset(b, world_mask=mask, flags=newton.StateFlags.PARTICLE_Q)
    np.testing.assert_allclose(b.vbd.edge_plastic_angle.numpy(), [0.0, -0.5], atol=1.0e-6)
    if device.is_cuda:
        b.vbd.edge_plastic_angle.assign(np.array([0.8, -0.5], dtype=np.float32))
        with wp.ScopedCapture(device=device) as reset_capture:
            solver.reset(b, world_mask=mask, flags=newton.StateFlags.PARTICLE_Q)
        wp.capture_launch(reset_capture.graph)
        np.testing.assert_allclose(b.vbd.edge_plastic_angle.numpy(), [0.0, -0.5], atol=1.0e-6)


def test_bending_plasticity_reset_worlds(test, device):
    """Reset selected worlds' creases with particle positions and preserve other worlds."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.replicate(_hinge_builder(), world_count=2, spacing=(2.0, 0.0, 0.0))
    builder.add_builder(_hinge_builder(), xform=wp.transform((4.0, 0.0, 0.0), wp.quat_identity()))
    builder.color()
    model = builder.finalize(device=device)
    solver = SolverVBD(model, particle_enable_bending_plasticity=True, particle_enable_tile_solve=False)
    state = model.state()
    state.vbd.edge_plastic_angle.fill_(0.8)
    mask = wp.array([True, False, False], dtype=wp.bool, device=device)
    solver.reset(state, world_mask=mask, flags=newton.StateFlags.PARTICLE_QD)
    np.testing.assert_allclose(state.vbd.edge_plastic_angle.numpy(), [0.8, 0.8, 0.8])
    solver.reset(state, world_mask=mask, flags=newton.StateFlags.PARTICLE_Q)
    np.testing.assert_allclose(state.vbd.edge_plastic_angle.numpy(), [0.0, 0.8, 0.8])
    solver.reset(state, world_mask=wp.array([False, False, True], dtype=wp.bool, device=device))
    np.testing.assert_allclose(state.vbd.edge_plastic_angle.numpy(), [0.0, 0.8, 0.0])
    solver.reset(state)
    np.testing.assert_array_equal(state.vbd.edge_plastic_angle.numpy(), np.zeros(3))


def test_bending_plasticity_invalid_material(test, device):
    """Reject negative and NaN yield angles before allocating plastic solver state."""
    for threshold in (-0.1, math.nan, -math.inf):
        builder = _hinge_builder(yield_angle=threshold)
        builder.color()
        model = builder.finalize(device=device)
        with test.assertRaisesRegex(ValueError, "edge_bending_yield_angle"):
            SolverVBD(model, particle_enable_bending_plasticity=True)


def test_bending_plasticity_capture_and_tiles(test, device):
    """Replay plastic updates and masked resets without allocation in CUDA graphs."""
    if not device.is_cuda:
        test.skipTest("CUDA graph capture requires a CUDA device")
    model, solver, a, b = _make_hinge(device, tiled=True)
    _step_at(solver, a, b, 1.0)
    solver.reset(a)
    solver.reset(b)
    a.particle_q.assign(_hinge_points(1.0))
    with wp.ScopedCapture(device=device) as capture:
        solver.step(a, b, None, None, 1.0 / 240.0)
        solver.step(b, a, None, None, 1.0 / 240.0)
    wp.capture_launch(capture.graph)
    np.testing.assert_allclose(a.vbd.edge_plastic_angle.numpy(), [0.8], atol=1.0e-6)
    wp.capture_launch(capture.graph)
    np.testing.assert_allclose(a.vbd.edge_plastic_angle.numpy(), [0.8], atol=1.0e-6)
    mask = wp.array([True, True], dtype=wp.bool, device=device)
    with wp.ScopedCapture(device=device) as reset_capture:
        solver.reset(a, world_mask=mask)
    wp.capture_launch(reset_capture.graph)
    np.testing.assert_array_equal(a.vbd.edge_plastic_angle.numpy(), [0.0])
    np.testing.assert_array_equal(model.edge_rest_angle.numpy(), [0.0])


class TestVBDBendingPlasticity(unittest.TestCase):
    pass


for device in get_test_devices():
    for function in (
        test_bending_plasticity_loading,
        test_bending_plasticity_state_isolation,
        test_bending_plasticity_disabled,
        test_bending_plasticity_limits,
        test_bending_plasticity_springback,
        test_bending_plasticity_angle_branch,
        test_bending_plasticity_missing_state,
        test_bending_plasticity_empty_model,
        test_bending_plasticity_autodiff,
        test_bending_plasticity_coupled_reset,
        test_bending_plasticity_reset_worlds,
        test_bending_plasticity_invalid_material,
        test_bending_plasticity_capture_and_tiles,
    ):
        add_function_test(TestVBDBendingPlasticity, function.__name__, function, devices=[device])


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
