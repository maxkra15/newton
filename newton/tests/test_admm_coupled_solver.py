# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ADMM-coupled solvers.

These tests validate generic :class:`SolverCoupledADMM` ADMM plumbing against a
cloth-plus-rigid-body scene.
"""

from __future__ import annotations

import math
import unittest
from dataclasses import FrozenInstanceError

import numpy as np
import warp as wp

import newton
from newton._src.solvers.coupled import admm_utils
from newton._src.solvers.coupled.interface import CouplingInterface
from newton.solvers import (
    SolverBase,
    SolverMuJoCo,
    SolverSemiImplicit,
    SolverVBD,
    SolverXPBD,
)
from newton.solvers.experimental import coupled as coupled_api
from newton.solvers.experimental.coupled import (
    SolverCoupled,
    SolverCoupledADMM,
)


@wp.kernel(enable_backward=False)
def _set_admm_plane_angle_kernel(body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], angle: float):
    body_q[0] = wp.transform(
        wp.vec3(0.0, 0.0, 0.0),
        wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angle),
    )
    body_qd[0] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class _CustomAdmmParticleCopySolver(SolverBase, CouplingInterface):
    """Base test solver that copies particle state."""

    def step(self, state_in, state_out, control, contacts, dt):
        del control, contacts, dt
        if state_in.particle_q is not None and state_out.particle_q is not None:
            wp.copy(state_out.particle_q, state_in.particle_q)
            wp.copy(state_out.particle_qd, state_in.particle_qd)
        if state_in.body_q is not None and state_out.body_q is not None:
            wp.copy(state_out.body_q, state_in.body_q)
            wp.copy(state_out.body_qd, state_in.body_qd)


class _KinematicAdmmPlaneSolver(_CustomAdmmParticleCopySolver):
    """Test solver that prescribes a fixed kinematic plane angle."""

    def __init__(self, model, angle):
        super().__init__(model)
        self.angle = float(angle)

    def step(self, state_in, state_out, control, contacts, dt):
        super().step(state_in, state_out, control, contacts, dt)
        wp.launch(
            _set_admm_plane_angle_kernel,
            dim=1,
            inputs=[state_out.body_q, state_out.body_qd, self.angle],
            device=self.model.device,
        )


class _FailingAdmmCopySolver(_CustomAdmmParticleCopySolver):
    """ADMM test solver with an opt-in step failure."""

    def __init__(self, model):
        super().__init__(model)
        self.fail = False

    def step(self, state_in, state_out, control, contacts, dt):
        if self.fail:
            raise RuntimeError("intentional ADMM step failure")
        super().step(state_in, state_out, control, contacts, dt)


def _build_cloth_rigid_scene(
    rigid_pos: tuple[float, float, float] = (0.0, 0.0, 1.5),
    rigid_mass: float = 0.05,
    cloth_pos: tuple[float, float, float] = (-0.25, -0.25, 1.5),
    dim_xy: int = 5,
    fix_cloth_edges: bool = True,
) -> tuple[newton.Model, int, int, int]:
    """Build a pinned cloth + free rigid body scene for attachment tests."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()

    rigid_start = builder.body_count
    body = builder.add_body(
        xform=wp.transform(p=wp.vec3(*rigid_pos), q=wp.quat_identity()),
        mass=rigid_mass,
        inertia=wp.mat33(np.eye(3) * 0.001),
    )
    builder.add_shape_box(body, hx=0.03, hy=0.03, hz=0.03)
    rigid_end = builder.body_count

    particle_start = builder.particle_count
    builder.add_cloth_grid(
        pos=wp.vec3(*cloth_pos),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        fix_left=fix_cloth_edges,
        fix_right=fix_cloth_edges,
        dim_x=dim_xy,
        dim_y=dim_xy,
        cell_x=0.1,
        cell_y=0.1,
        mass=0.05,
        tri_ke=1.0e4,
        tri_ka=1.0e4,
        tri_kd=1e-2,
        edge_ke=0.01,
        edge_kd=1e-2,
        particle_radius=0.01,
    )
    center = dim_xy // 2
    particle_idx = particle_start + center * (dim_xy + 1) + center
    builder.color()
    model = builder.finalize()
    return model, rigid_start, rigid_end, particle_idx


def _make_solver(
    model: newton.Model,
    rigid_start: int,
    rigid_end: int,
    admm_iters: int = 5,
    rho: float = 50.0,
    gamma: float = 0.0,
    baumgarte: float = 0.1,
):
    """Standard MuJoCo/VBD ADMM configuration used across tests."""
    mjc_ids = wp.array(list(range(rigid_start, rigid_end)), dtype=int)
    vbd_ids = wp.array(
        [i for i in range(model.body_count) if i < rigid_start or i >= rigid_end],
        dtype=int,
    )
    return SolverCoupledADMM(
        model=model,
        entries=[
            SolverCoupled.Entry(
                name="mjc",
                solver=lambda v: SolverMuJoCo(model=v, use_mujoco_contacts=False, njmax=20),
                bodies=[int(i) for i in mjc_ids.numpy()],
                joints=list(range(model.joint_count)),
            ),
            SolverCoupled.Entry(
                name="vbd",
                solver=lambda v: SolverVBD(model=v, iterations=5),
                bodies=[int(i) for i in vbd_ids.numpy()],
                particles=list(range(model.particle_count)),
            ),
        ],
        coupling=SolverCoupledADMM.Config(
            iterations=admm_iters,
            rho=rho,
            gamma=gamma,
            baumgarte=baumgarte,
        ),
    )


def _run(solver, model: newton.Model, n_steps: int = 30, dt: float = 1.0 / 60.0):
    """Run ``n_steps`` of simulation and return (body_q, particle_q)."""
    state_0 = model.state()
    state_1 = model.state()
    contacts = model.contacts()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    for _ in range(n_steps):
        state_0.clear_forces()
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, dt)
        state_0, state_1 = state_1, state_0

    return state_0.body_q.numpy().copy(), state_0.particle_q.numpy().copy()


def _build_two_particle_scene() -> newton.Model:
    builder = newton.ModelBuilder(gravity=0.0)
    builder.add_particle(pos=(-0.5, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.0)
    builder.add_particle(pos=(0.5, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.0)
    builder.color()
    return builder.finalize(device="cpu")


def _build_two_particle_contact_scene(
    gap: float = -0.1,
    vel_a: tuple[float, float, float] = (0.0, 0.0, 0.0),
    vel_b: tuple[float, float, float] = (0.0, 0.0, 0.0),
    radius: float = 0.05,
) -> newton.Model:
    builder = newton.ModelBuilder(gravity=0.0)
    builder.add_particle(pos=(gap, 0.0, 0.0), vel=vel_a, mass=1.0, radius=radius)
    builder.add_particle(pos=(0.0, 0.0, 0.0), vel=vel_b, mass=1.0, radius=radius)
    builder.color()
    model = builder.finalize(device="cpu")
    model.particle_grid = None
    return model


def _run_particles(solver, model: newton.Model, n_steps: int = 5, dt: float = 1.0 / 60.0):
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    for _ in range(n_steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts=None, dt=dt)
        state_0, state_1 = state_1, state_0

    return state_0.particle_q.numpy().copy()


def _make_vbd_xpbd_particle_solver(model: newton.Model):
    return SolverCoupledADMM(
        model=model,
        entries=[
            SolverCoupled.Entry(
                name="vbd",
                solver=lambda v: SolverVBD(model=v, iterations=2),
                particles=[0],
            ),
            SolverCoupled.Entry(
                name="xpbd",
                solver=lambda v: SolverXPBD(model=v, iterations=2),
                particles=[1],
            ),
        ],
        coupling=SolverCoupledADMM.Config(
            iterations=8,
            rho=20.0,
            baumgarte=0.2,
        ),
    )


def _make_semi_particle_solver(model: newton.Model):
    return SolverCoupledADMM(
        model=model,
        entries=[
            SolverCoupled.Entry(
                name="a",
                solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                particles=[0],
            ),
            SolverCoupled.Entry(
                name="b",
                solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                particles=[1],
            ),
        ],
        coupling=SolverCoupledADMM.Config(
            iterations=10,
            rho=30.0,
            baumgarte=0.5,
        ),
    )


def _build_body_particle_contact_scene() -> newton.Model:
    builder = newton.ModelBuilder(gravity=0.0)
    builder.add_body(
        xform=wp.transform(p=wp.vec3(-0.1, 0.0, 0.0), q=wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(np.eye(3)),
    )
    builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.0)
    builder.color()
    model = builder.finalize(device="cpu")
    model.particle_grid = None
    return model


def _build_body_particle_attachment_scene(enabled: bool = True) -> newton.Model:
    builder = newton.ModelBuilder(gravity=0.0)
    body = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(np.eye(3)),
    )
    particle = builder.add_particle(pos=(0.3, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.0)
    SolverCoupledADMM.add_body_particle_attachment(
        builder,
        body,
        particle,
        stiffness=500.0,
        enabled=enabled,
    )
    builder.color()
    model = builder.finalize(device="cpu")
    model.particle_grid = None
    return model


def _build_two_body_contact_scene(gap: float = -0.1) -> newton.Model:
    builder = newton.ModelBuilder(gravity=0.0)
    builder.add_body(
        xform=wp.transform(p=wp.vec3(gap, 0.0, 0.0), q=wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(np.eye(3)),
    )
    builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(np.eye(3)),
    )
    builder.color()
    return builder.finalize(device="cpu")


def _build_collision_contact_scene() -> tuple[newton.Model, int, int, int]:
    builder = newton.ModelBuilder(gravity=0.0)
    tray_body = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
        mass=0.1,
        inertia=wp.mat33(np.eye(3) * 0.01),
    )

    tray_cfg = newton.ModelBuilder.ShapeConfig()
    tray_cfg.has_shape_collision = False
    tray_cfg.has_particle_collision = True
    tray_shape = builder.add_shape_box(
        tray_body,
        xform=wp.transform(p=wp.vec3(0.0, 0.0, -0.025), q=wp.quat_identity()),
        hx=0.1,
        hy=0.1,
        hz=0.025,
        cfg=tray_cfg,
    )
    particle = builder.add_particle(
        pos=(0.0, 0.0, 0.12),
        vel=(0.0, 0.0, -0.5),
        mass=0.025,
        radius=0.025,
    )
    builder.color()
    model = builder.finalize(device="cpu")
    model.particle_grid = None
    model.soft_contact_ke = 0.0
    model.soft_contact_kd = 0.0
    model.soft_contact_kf = 0.0
    model.soft_contact_mu = 0.0
    return model, particle, tray_body, tray_shape


def _run_body_particle(solver, model: newton.Model, n_steps: int = 4, dt: float = 1.0 / 60.0):
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    for _ in range(n_steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts=None, dt=dt)
        state_0, state_1 = state_1, state_0

    return state_0.body_q.numpy().copy(), state_0.particle_q.numpy().copy()


def _run_bodies(
    solver,
    model: newton.Model,
    n_steps: int = 4,
    dt: float = 1.0 / 60.0,
    body_qd: np.ndarray | None = None,
):
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    if body_qd is not None:
        state_0.body_qd = wp.array(body_qd, dtype=wp.spatial_vector, device=model.device)

    for _ in range(n_steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts=None, dt=dt)
        state_0, state_1 = state_1, state_0

    return state_0.body_q.numpy().copy(), state_0.body_qd.numpy().copy()


def _make_semi_body_particle_solver(model: newton.Model):
    return SolverCoupledADMM(
        model=model,
        entries=[
            SolverCoupled.Entry(
                name="body",
                solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                bodies=[0],
            ),
            SolverCoupled.Entry(
                name="particle",
                solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                particles=[0],
            ),
        ],
        coupling=SolverCoupledADMM.Config(
            iterations=10,
            rho=30.0,
            baumgarte=0.5,
        ),
    )


def _make_semi_body_body_solver(model: newton.Model):
    return SolverCoupledADMM(
        model=model,
        entries=[
            SolverCoupled.Entry(
                name="a",
                solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                bodies=[0],
            ),
            SolverCoupled.Entry(
                name="b",
                solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                bodies=[1],
            ),
        ],
        coupling=SolverCoupledADMM.Config(
            iterations=10,
            rho=30.0,
            baumgarte=0.5,
        ),
    )


def _build_inclined_plane_particle_box_scene(
    angle: float,
    *,
    particle_radius: float = 0.025,
    box_half_extent: float = 0.06,
    penetration: float = 0.002,
) -> tuple[newton.Model, int, int, list[int]]:
    builder = newton.ModelBuilder(gravity=-10.0)
    plane_q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angle)
    plane_body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), plane_q),
        mass=0.0,
        inertia=wp.mat33(),
        is_kinematic=True,
    )

    plane_cfg = newton.ModelBuilder.ShapeConfig()
    plane_cfg.has_shape_collision = False
    plane_cfg.has_particle_collision = True
    plane_shape = builder.add_shape_plane(
        body=plane_body,
        xform=wp.transform_identity(),
        width=2.0,
        length=2.0,
        cfg=plane_cfg,
    )

    n = np.array([math.sin(angle), 0.0, math.cos(angle)], dtype=np.float32)
    tangent = np.array([math.cos(angle), 0.0, -math.sin(angle)], dtype=np.float32)
    binormal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    center = (particle_radius - penetration) * n

    particle_ids = []
    for tangent_sign in (-1.0, 1.0):
        for binormal_sign in (-1.0, 1.0):
            pos = center + tangent_sign * box_half_extent * tangent + binormal_sign * box_half_extent * binormal
            particle_ids.append(
                builder.add_particle(
                    pos=tuple(float(x) for x in pos),
                    vel=(0.0, 0.0, 0.0),
                    mass=0.25,
                    radius=particle_radius,
                )
            )

    builder.color()
    model = builder.finalize(device="cpu")
    model.particle_grid = None
    model.soft_contact_ke = 0.0
    model.soft_contact_kd = 0.0
    model.soft_contact_kf = 0.0
    model.soft_contact_mu = 0.0
    return model, plane_body, plane_shape, particle_ids


def _make_admm_inclined_plane_particle_box_solver(
    model: newton.Model,
    plane_body: int,
    particle_ids: list[int],
    angle: float,
    friction: float,
) -> SolverCoupledADMM:
    return SolverCoupledADMM(
        model=model,
        entries=[
            SolverCoupled.Entry(
                name="plane",
                solver=lambda v: _KinematicAdmmPlaneSolver(model=v, angle=angle),
                bodies=[plane_body],
            ),
            SolverCoupled.Entry(
                name="box",
                solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                particles=particle_ids,
            ),
        ],
        coupling=SolverCoupledADMM.Config(
            iterations=18,
            rho=50.0,
            baumgarte=0.1,
            contact_pairs=[
                SolverCoupledADMM.ContactPair(
                    source="plane",
                    destination="box",
                )
            ],
        ),
    )


def _run_inclined_plane_particle_box(
    angle: float,
    friction: float,
    *,
    steps: int = 120,
    dt: float = 1.0 / 360.0,
) -> tuple[float, float, int]:
    model, plane_body, _, particle_ids = _build_inclined_plane_particle_box_scene(angle)
    # ADMM derives friction from material properties; set both sides so the
    # geometric-mean combine reduces to the requested coefficient.
    model.particle_mu = float(friction)
    model.shape_material_mu = wp.full(model.shape_count, float(friction), dtype=wp.float32, device=model.device)
    solver = _make_admm_inclined_plane_particle_box_solver(
        model,
        plane_body,
        particle_ids,
        angle,
        friction,
    )
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    initial_com = np.mean(state_0.particle_q.numpy()[particle_ids], axis=0)
    for _ in range(steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts=None, dt=dt)
        state_0, state_1 = state_1, state_0

    final_q = state_0.particle_q.numpy()[particle_ids]
    final_qd = state_0.particle_qd.numpy()[particle_ids]
    final_com = np.mean(final_q, axis=0)
    final_vel = np.mean(final_qd, axis=0)
    tangent = np.array([math.cos(angle), 0.0, -math.sin(angle)], dtype=np.float32)
    displacement = float(np.dot(final_com - initial_com, tangent))
    velocity = float(np.dot(final_vel, tangent))
    return displacement, velocity, solver.collision_contact_count_max


def _rotate_y_np(v: np.ndarray, angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]], dtype=np.float32)


def _build_inclined_plane_rigid_box_scene(
    angle: float,
    *,
    box_half_height: float = 0.08,
    penetration: float = 0.002,
) -> tuple[newton.Model, int, int]:
    builder = newton.ModelBuilder(gravity=-10.0)
    plane_q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angle)
    plane_body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), plane_q),
        mass=0.0,
        inertia=wp.mat33(),
        is_kinematic=True,
    )

    local_center = np.array([0.0, 0.0, box_half_height - penetration], dtype=np.float32)
    box_center = _rotate_y_np(local_center, angle)
    box_body = builder.add_body(
        xform=wp.transform(
            wp.vec3(float(box_center[0]), float(box_center[1]), float(box_center[2])),
            plane_q,
        ),
        mass=1.0,
        inertia=wp.mat33(np.eye(3) * 0.01),
    )
    builder.color()
    return builder.finalize(device="cpu"), plane_body, box_body


def _build_collision_inclined_plane_rigid_box_scene(
    angle: float,
    *,
    box_half_height: float = 0.08,
    penetration: float = 0.004,
) -> tuple[newton.Model, int, int]:
    builder = newton.ModelBuilder(gravity=-10.0)
    plane_q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angle)
    plane_body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), plane_q),
        mass=0.0,
        inertia=wp.mat33(),
        is_kinematic=True,
    )

    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.has_shape_collision = True
    cfg.has_particle_collision = False
    cfg.density = 0.0
    builder.add_shape_box(
        plane_body,
        xform=wp.transform(wp.vec3(1.0, 0.0, -0.025), wp.quat_identity()),
        hx=3.0,
        hy=0.4,
        hz=0.025,
        cfg=cfg,
    )

    local_center = np.array([0.0, 0.0, box_half_height - penetration], dtype=np.float32)
    box_center = _rotate_y_np(local_center, angle)
    box_body = builder.add_body(
        xform=wp.transform(
            wp.vec3(float(box_center[0]), float(box_center[1]), float(box_center[2])),
            plane_q,
        ),
        mass=1.0,
        inertia=wp.mat33(np.eye(3) * 0.01),
    )
    builder.add_shape_box(
        box_body,
        hx=0.08,
        hy=0.08,
        hz=box_half_height,
        cfg=cfg,
    )
    builder.color()
    return builder.finalize(device="cpu"), plane_body, box_body


def _make_collision_admm_inclined_plane_rigid_box_solver(
    model: newton.Model,
    plane_body: int,
    box_body: int,
    angle: float,
    friction: float,
    *,
    rigid_contact_matching: str = "disabled",
    contact_matching_pos_threshold: float | None = None,
    contact_matching_normal_dot_threshold: float | None = None,
    contact_matching_force_scale: float = 1.0,
) -> SolverCoupledADMM:
    del friction
    return SolverCoupledADMM(
        model=model,
        entries=[
            SolverCoupled.Entry(
                name="plane",
                solver=lambda v: _KinematicAdmmPlaneSolver(model=v, angle=angle),
                bodies=[plane_body],
            ),
            SolverCoupled.Entry(
                name="box",
                solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                bodies=[box_body],
            ),
        ],
        coupling=SolverCoupledADMM.Config(
            iterations=30,
            rho=5.0,
            gamma=0.2,
            baumgarte=0.03,
            rigid_contact_matching=rigid_contact_matching,
            contact_matching_pos_threshold=contact_matching_pos_threshold,
            contact_matching_normal_dot_threshold=contact_matching_normal_dot_threshold,
            contact_matching_force_scale=contact_matching_force_scale,
            contact_pairs=[
                SolverCoupledADMM.ContactPair(
                    source="plane",
                    destination="box",
                )
            ],
        ),
    )


def _run_collision_inclined_plane_rigid_box(
    angle: float,
    friction: float,
    *,
    steps: int = 120,
    dt: float = 1.0 / 360.0,
) -> tuple[float, float, float, int]:
    model, plane_body, box_body = _build_collision_inclined_plane_rigid_box_scene(angle)
    model.shape_material_mu = wp.full(model.shape_count, float(friction), dtype=wp.float32, device=model.device)
    solver = _make_collision_admm_inclined_plane_rigid_box_solver(model, plane_body, box_body, angle, friction)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    initial_pos = state_0.body_q.numpy()[box_body, :3].copy()
    min_gap = math.inf
    normal = _rotate_y_np(np.array([0.0, 0.0, 1.0], dtype=np.float32), angle)
    tangent = _rotate_y_np(np.array([1.0, 0.0, 0.0], dtype=np.float32), angle)
    for _ in range(steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts=None, dt=dt)
        state_0, state_1 = state_1, state_0
        center_gap = float(np.dot(normal, state_0.body_q.numpy()[box_body, :3]))
        min_gap = min(min_gap, center_gap - 0.08)

    final_pos = state_0.body_q.numpy()[box_body, :3]
    final_qd = state_0.body_qd.numpy()[box_body, :3]
    displacement = float(np.dot(final_pos - initial_pos, tangent))
    velocity = float(np.dot(final_qd, tangent))
    return displacement, velocity, min_gap, solver.collision_contact_count_max


class TestAdmmSmoke(unittest.TestCase):
    """End-to-end: construct, run, verify state advances without NaNs."""

    def test_rejects_invalid_numerical_config(self):
        model = _build_two_particle_scene()
        entries = [
            SolverCoupled.Entry(name="a", solver=SolverSemiImplicit, particles=[0]),
            SolverCoupled.Entry(name="b", solver=SolverSemiImplicit, particles=[1]),
        ]
        invalid_configs = (
            ({"iterations": 0}, "iterations"),
            ({"iterations": 1.5}, "iterations"),
            ({"rho": 0.0}, "rho"),
            ({"rho": float("nan")}, "rho"),
            ({"gamma": -1.0}, "gamma"),
            ({"baumgarte": float("inf")}, "baumgarte"),
            ({"joint_stiffness": -1.0}, "joint_stiffness"),
            ({"joint_proximal_mass_scale": 0.0}, "joint_proximal_mass_scale"),
            ({"contact_matching_normal_dot_threshold": 1.1}, "normal_dot_threshold"),
        )
        for kwargs, message in invalid_configs:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                SolverCoupledADMM(model=model, entries=entries, coupling=SolverCoupledADMM.Config(**kwargs))

    def test_construct_and_step_no_attachments(self):
        model, rs, re, _ = _build_cloth_rigid_scene()
        solver = _make_solver(model, rs, re, admm_iters=1)

        body_q, particle_q = _run(solver, model, n_steps=10)
        self.assertTrue(np.all(np.isfinite(body_q)))
        self.assertTrue(np.all(np.isfinite(particle_q)))

    def test_admm_iters_idempotent_with_no_coupling(self):
        """With gamma=0 and no attachments the iteration count should not
        change the result (no coupling = idempotent outer loop)."""
        model_a, rs, re, _ = _build_cloth_rigid_scene()
        solver_a = _make_solver(model_a, rs, re, admm_iters=1, gamma=0.0)
        body_a, part_a = _run(solver_a, model_a, n_steps=5)

        model_b, rs, re, _ = _build_cloth_rigid_scene()
        solver_b = _make_solver(model_b, rs, re, admm_iters=4, gamma=0.0)
        body_b, part_b = _run(solver_b, model_b, n_steps=5)

        np.testing.assert_allclose(body_a, body_b, atol=1e-6)
        np.testing.assert_allclose(part_a, part_b, atol=1e-6)


class TestAdmmProximal(unittest.TestCase):
    """Proximal terms affect constrained DOFs only."""

    def test_gamma_does_not_change_unconstrained_freefall(self):
        # Place the rigid body high so it stays in free-fall across the
        # window; with no ADMM constraints, gamma should not alter the result.
        model_ref, rs, re, _ = _build_cloth_rigid_scene(rigid_pos=(0.0, 0.0, 5.0))
        solver_ref = _make_solver(model_ref, rs, re, admm_iters=3, gamma=0.0)
        body_ref, part_ref = _run(solver_ref, model_ref, n_steps=5)

        model_g, rs, re, _ = _build_cloth_rigid_scene(rigid_pos=(0.0, 0.0, 5.0))
        solver_g = _make_solver(model_g, rs, re, admm_iters=3, gamma=5.0)
        body_g, part_g = _run(solver_g, model_g, n_steps=5)

        np.testing.assert_allclose(body_ref, body_g, atol=1.0e-6)
        np.testing.assert_allclose(part_ref, part_g, atol=1.0e-6)
        self.assertTrue(np.all(np.isfinite(body_g)))
        self.assertTrue(np.all(np.isfinite(part_g)))


class TestAdmmGraphCapture(unittest.TestCase):
    """CUDA graph-capture smoke tests for dynamic proximal refresh."""

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA graph capture requires CUDA")
    def test_xpbd_vbd_contact_proximal_refresh_is_graph_capturable(self):
        device = "cuda:0"
        builder = newton.ModelBuilder(gravity=0.0)
        builder.default_shape_cfg.density = 1000.0
        body_a = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
        builder.add_shape_box(body=body_a, hx=0.05, hy=0.05, hz=0.05)
        body_b = builder.add_body(xform=wp.transform(wp.vec3(0.08, 0.0, 0.0), wp.quat_identity()))
        builder.add_shape_box(body=body_b, hx=0.05, hy=0.05, hz=0.05)
        builder.color()
        model = builder.finalize(device=device)
        solver = SolverCoupledADMM(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="xpbd",
                    solver=lambda v: SolverXPBD(model=v, iterations=1),
                    bodies=[body_a],
                ),
                SolverCoupled.Entry(
                    name="vbd",
                    solver=lambda v: SolverVBD(model=v, iterations=1),
                    bodies=[body_b],
                ),
            ],
            coupling=SolverCoupledADMM.Config(
                iterations=1,
                gamma=1.0,
                contact_pairs=[SolverCoupledADMM.ContactPair(source="xpbd", destination="vbd")],
            ),
        )

        state_0 = model.state()
        state_1 = model.state()
        control = model.control()

        solver.step(state_0, state_1, control, contacts=None, dt=1.0 / 120.0)
        state_0, state_1 = state_1, state_0
        wp.synchronize_device(device)

        with wp.ScopedCapture(device=device) as capture:
            solver.step(state_0, state_1, control, contacts=None, dt=1.0 / 120.0)

        self.assertIsNotNone(capture.graph)
        wp.capture_launch(capture.graph)
        q = state_1.body_q.numpy()
        self.assertTrue(np.all(np.isfinite(q)))


class TestAdmmModelJointInterface(unittest.TestCase):
    """Cross-solver model joints are converted to ADMM attachments."""

    def _build_two_body_joint_scene(
        self,
        joint_type: str = "ball",
        *,
        friction: float = 0.0,
    ) -> tuple[newton.Model, int, int, int]:
        builder = newton.ModelBuilder(gravity=0.0)
        parent = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(np.eye(3) * 0.01),
        )
        child = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.3, 0.0, 0.0), q=wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(np.eye(3) * 0.01),
        )
        if joint_type == "ball":
            joint = builder.add_joint_ball(
                parent=parent,
                child=child,
                friction=friction,
                collision_filter_parent=False,
            )
        elif joint_type == "fixed":
            joint = builder.add_joint_fixed(parent=parent, child=child, collision_filter_parent=False)
        elif joint_type == "revolute":
            joint = builder.add_joint_revolute(
                parent=parent, child=child, friction=friction, collision_filter_parent=False
            )
        else:
            raise ValueError(joint_type)
        builder.color()
        return builder.finalize(device="cpu"), parent, child, joint

    def _make_two_body_joint_solver(self, model: newton.Model, parent: int, child: int, **coupling_kwargs):
        return SolverCoupledADMM(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="parent",
                    solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                    bodies=[parent],
                ),
                SolverCoupled.Entry(
                    name="child",
                    solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                    bodies=[child],
                ),
            ],
            coupling=SolverCoupledADMM.Config(
                iterations=12,
                rho=40.0,
                baumgarte=0.5,
                joint_stiffness=500.0,
                joint_angular_stiffness=50.0,
                **coupling_kwargs,
            ),
        )

    def test_ball_joint_attachment_closes_anchor_gap(self):
        model, parent, child, _ = self._build_two_body_joint_scene("ball")
        solver = self._make_two_body_joint_solver(model, parent, child)
        initial_gap = abs(model.state().body_q.numpy()[child, 0] - model.state().body_q.numpy()[parent, 0])

        body_q, _ = _run_bodies(solver, model, n_steps=8, dt=1.0 / 120.0)
        final_gap = abs(body_q[child, 0] - body_q[parent, 0])

        self.assertLess(final_gap, 0.5 * initial_gap)

    def test_rejects_cross_solver_joint_owned_by_subsolver(self):
        model, parent, child, joint = self._build_two_body_joint_scene("ball")
        with self.assertRaisesRegex(ValueError, "must not be owned"):
            SolverCoupledADMM(
                model=model,
                entries=[
                    SolverCoupled.Entry(
                        name="parent",
                        solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                        bodies=[parent],
                        joints=[joint],
                    ),
                    SolverCoupled.Entry(
                        name="child",
                        solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                        bodies=[child],
                    ),
                ],
                coupling=SolverCoupledADMM.Config(),
            )


class TestAdmmBodyParticleAttachment(unittest.TestCase):
    """Custom model attributes are converted to rigid-particle ADMM attachments."""

    def test_custom_attribute_attachment_closes_gap(self):
        model = _build_body_particle_attachment_scene()
        solver = _make_semi_body_particle_solver(model)
        initial_gap = np.linalg.norm(model.state().body_q.numpy()[0, :3] - model.state().particle_q.numpy()[0])

        body_q, particle_q = _run_body_particle(solver, model, n_steps=8, dt=1.0 / 120.0)
        final_gap = np.linalg.norm(body_q[0, :3] - particle_q[0])

        self.assertLess(final_gap, 0.5 * initial_gap)


class TestAdmmReset(unittest.TestCase):
    """World-masked reset behavior for static and dynamic ADMM state."""

    _STATIC_RESET_ATTRS = ("u", "lambda_", "Jv", "u_target")
    _DYNAMIC_RESET_ATTRS = ("u", "lambda_", "Jv", "u_min")

    @staticmethod
    def _build_solver() -> tuple[newton.Model, SolverCoupledADMM]:
        template = newton.ModelBuilder(gravity=0.0)

        def add_body(x: float) -> int:
            return template.add_body(
                xform=wp.transform(p=wp.vec3(x, 0.0, 0.0), q=wp.quat_identity()),
                mass=1.0,
                inertia=wp.mat33(np.eye(3) * 0.1),
            )

        fixed_parent = add_body(0.0)
        fixed_child = add_body(0.2)
        revolute_parent = add_body(0.4)
        revolute_child = add_body(0.6)
        ball_parent = add_body(0.8)
        ball_child = add_body(1.0)
        attachment_body = add_body(1.2)
        owned_body = add_body(1.6)
        template.add_shape_sphere(fixed_parent, radius=0.05)
        template.add_shape_sphere(fixed_child, radius=0.05)
        template.add_joint_fixed(parent=fixed_parent, child=fixed_child, collision_filter_parent=False)
        template.add_joint_revolute(
            parent=revolute_parent,
            child=revolute_child,
            axis=(1.0, 0.0, 0.0),
            friction=0.4,
            collision_filter_parent=False,
        )
        template.add_joint_ball(
            parent=ball_parent,
            child=ball_child,
            friction=0.3,
            collision_filter_parent=False,
        )
        owned_joint = template.add_joint_revolute(
            parent=-1,
            child=owned_body,
            axis=(0.0, 0.0, 1.0),
            collision_filter_parent=False,
        )
        template.add_articulation([owned_joint])
        particle = template.add_particle(pos=(1.4, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0)
        SolverCoupledADMM.add_body_particle_attachment(template, attachment_body, particle)
        template.color()

        builder = newton.ModelBuilder(gravity=0.0)
        builder.replicate(template, world_count=2)
        model = builder.finalize(device="cpu")
        model.particle_grid = None

        bodies_per_world = template.body_count
        entry_a_local = (fixed_parent, revolute_parent, ball_parent, attachment_body, owned_body)
        entry_b_local = (fixed_child, revolute_child, ball_child)
        entry_a_bodies = [world * bodies_per_world + body for world in range(2) for body in entry_a_local]
        entry_b_bodies = [world * bodies_per_world + body for world in range(2) for body in entry_b_local]
        entry_a_joints = [world * template.joint_count + owned_joint for world in range(2)]
        solver = SolverCoupledADMM(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="a",
                    solver=_CustomAdmmParticleCopySolver,
                    bodies=entry_a_bodies,
                    joints=entry_a_joints,
                ),
                SolverCoupled.Entry(
                    name="b",
                    solver=_CustomAdmmParticleCopySolver,
                    bodies=entry_b_bodies,
                    particles=list(range(model.particle_count)),
                ),
            ],
            coupling=SolverCoupledADMM.Config(
                iterations=1,
                contact_pairs=[SolverCoupledADMM.ContactPair(source="a", destination="b")],
            ),
        )
        return model, solver

    @staticmethod
    def _build_dynamic_contact_solver(
        contact_kind: str,
        *,
        rigid_contact_matching: str = "sticky",
        device: str = "cpu",
    ) -> tuple[newton.Model, SolverCoupledADMM]:
        template = newton.ModelBuilder(gravity=0.0)
        body_a = body_b = particle_a = particle_b = -1

        if contact_kind == "rr":
            cfg = newton.ModelBuilder.ShapeConfig()
            cfg.has_shape_collision = True
            cfg.has_particle_collision = False
            cfg.density = 0.0
            body_a = template.add_body(
                xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                mass=1.0,
                inertia=wp.mat33(np.eye(3) * 0.1),
            )
            body_b = template.add_body(
                xform=wp.transform(wp.vec3(0.09, 0.0, 0.0), wp.quat_identity()),
                mass=1.0,
                inertia=wp.mat33(np.eye(3) * 0.1),
            )
            template.add_shape_sphere(body_a, radius=0.05, cfg=cfg)
            template.add_shape_sphere(body_b, radius=0.05, cfg=cfg)
        elif contact_kind == "rp":
            cfg = newton.ModelBuilder.ShapeConfig()
            cfg.has_shape_collision = False
            cfg.has_particle_collision = True
            cfg.density = 0.0
            body_a = template.add_body(
                xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                mass=1.0,
                inertia=wp.mat33(np.eye(3) * 0.1),
            )
            template.add_shape_sphere(body_a, radius=0.08, cfg=cfg)
            particle_b = template.add_particle(
                pos=(0.1, 0.0, 0.0),
                vel=(0.0, 0.0, 0.0),
                mass=1.0,
                radius=0.05,
            )
        elif contact_kind == "pp":
            particle_a = template.add_particle(
                pos=(0.0, 0.0, 0.0),
                vel=(0.0, 0.0, 0.0),
                mass=1.0,
                radius=0.05,
            )
            particle_b = template.add_particle(
                pos=(0.09, 0.0, 0.0),
                vel=(0.0, 0.0, 0.0),
                mass=1.0,
                radius=0.05,
            )
        else:
            raise ValueError(f"Unknown dynamic contact kind {contact_kind!r}")

        template.color()
        builder = newton.ModelBuilder(gravity=0.0)
        builder.replicate(template, world_count=2)
        model = builder.finalize(device=device)
        model.particle_grid = None

        if contact_kind == "rr":
            entries = [
                SolverCoupled.Entry(
                    name="a",
                    solver=_CustomAdmmParticleCopySolver,
                    bodies=[world * template.body_count + body_a for world in range(2)],
                ),
                SolverCoupled.Entry(
                    name="b",
                    solver=_CustomAdmmParticleCopySolver,
                    bodies=[world * template.body_count + body_b for world in range(2)],
                ),
            ]
        elif contact_kind == "rp":
            entries = [
                SolverCoupled.Entry(
                    name="a",
                    solver=_CustomAdmmParticleCopySolver,
                    bodies=[world * template.body_count + body_a for world in range(2)],
                ),
                SolverCoupled.Entry(
                    name="b",
                    solver=_CustomAdmmParticleCopySolver,
                    particles=[world * template.particle_count + particle_b for world in range(2)],
                ),
            ]
        else:
            entries = [
                SolverCoupled.Entry(
                    name="a",
                    solver=_CustomAdmmParticleCopySolver,
                    particles=[world * template.particle_count + particle_a for world in range(2)],
                ),
                SolverCoupled.Entry(
                    name="b",
                    solver=_CustomAdmmParticleCopySolver,
                    particles=[world * template.particle_count + particle_b for world in range(2)],
                ),
            ]

        solver = SolverCoupledADMM(
            model=model,
            entries=entries,
            coupling=SolverCoupledADMM.Config(
                iterations=1,
                rigid_contact_matching=rigid_contact_matching,
                contact_matching_force_scale=1.0,
                contact_pairs=[SolverCoupledADMM.ContactPair(source="a", destination="b")],
            ),
        )
        return model, solver

    @staticmethod
    def _dynamic_group(solver: SolverCoupledADMM, contact_kind: str):
        groups = {
            "rr": solver._admm_dynamic_rr_contact_groups,
            "rp": solver._admm_dynamic_rp_contact_groups,
            "pp": solver._admm_dynamic_pp_contact_groups,
        }[contact_kind]
        if len(groups) != 1:
            raise AssertionError(f"Expected one {contact_kind} group, found {len(groups)}")
        return groups[0]

    @classmethod
    def _seed_dynamic_contact_state(cls, group) -> dict[str, np.ndarray]:
        worlds = group.world_ids.numpy()
        seeded = {}
        for attr_index, attr in enumerate(cls._DYNAMIC_RESET_ATTRS):
            array = getattr(group, attr)
            if array.dtype == wp.vec3:
                values = np.empty((group.count, 3), dtype=np.float32)
                for row, world in enumerate(worlds):
                    base = 10.0 * (attr_index + 1) + (float(world) if world >= 0 else 8.0)
                    values[row] = (base, base + 0.25, base + 0.5)
            else:
                values = np.asarray(
                    [10.0 * (attr_index + 1) + (float(world) if world >= 0 else 8.0) for world in worlds],
                    dtype=np.float32,
                )
            array.assign(values)
            seeded[attr] = values.copy()

        stream = getattr(group, "contact_stream", None)
        if stream is not None:
            stream_worlds = stream.world_ids.numpy()
            for attr_index, attr in enumerate(("normal_force", "normal_impulse"), start=1):
                values = np.asarray(
                    [100.0 * attr_index + (float(world) if world >= 0 else 8.0) for world in stream_worlds],
                    dtype=np.float32,
                )
                getattr(stream, attr).assign(values)
                seeded[f"stream.{attr}"] = values.copy()
        return seeded

    @staticmethod
    def _dynamic_contact_rows(group, contact_kind: str, world: int) -> list[tuple[tuple[int, ...], np.ndarray]]:
        active = group.active.numpy() != 0
        worlds = group.world_ids.numpy()
        lambda_values = group.lambda_.numpy()
        if contact_kind == "rr":
            columns = (group.shape_ids_a.numpy(), group.shape_ids_b.numpy(), group.point_ids.numpy())
        elif contact_kind == "rp":
            columns = (group.particle_ids.numpy(), group.shape_ids.numpy())
        else:
            columns = (group.particle_ids_a.numpy(), group.particle_ids_b.numpy())

        rows = []
        for row in np.flatnonzero(active & (worlds == world)):
            key = tuple(int(column[row]) for column in columns)
            rows.append((key, lambda_values[row].copy()))
        return sorted(rows, key=lambda item: item[0])

    @staticmethod
    def _static_groups(solver: SolverCoupledADMM):
        return (
            *solver._admm_rr_groups,
            *solver._admm_rr_angular_groups,
            *solver._admm_rr_revolute_angular_groups,
            *solver._admm_rr_angular_friction_groups,
            *solver._admm_rp_groups,
        )

    @staticmethod
    def _group_endpoint_worlds(solver: SolverCoupledADMM, group) -> np.ndarray:
        if hasattr(group, "body_entry_name_a"):
            entry_a = solver._entries[group.body_entry_name_a]
            global_a = entry_a.body_local_to_global.numpy()[group.body_ids_a.numpy()]
            worlds_a = solver.model.body_world.numpy()[global_a]
            entry_b = solver._entries[group.body_entry_name_b]
            global_b = entry_b.body_local_to_global.numpy()[group.body_ids_b.numpy()]
            worlds_b = solver.model.body_world.numpy()[global_b]
        else:
            entry_a = solver._entries[group.body_entry_name]
            global_a = entry_a.body_local_to_global.numpy()[group.body_ids.numpy()]
            worlds_a = solver.model.body_world.numpy()[global_a]
            entry_b = solver._entries[group.particle_entry_name]
            global_b = entry_b.particle_local_to_global.numpy()[group.particle_ids.numpy()]
            worlds_b = solver.model.particle_world.numpy()[global_b]
        np.testing.assert_array_equal(worlds_a, worlds_b)
        return worlds_a

    @staticmethod
    def _seed_array(array, offset: float) -> np.ndarray:
        count = array.shape[0]
        values = np.arange(count, dtype=np.float32) + offset
        if array.dtype == wp.transform:
            rows = np.zeros((count, 7), dtype=np.float32)
            rows[:, 0] = values
            rows[:, 6] = 1.0
        elif array.dtype == wp.spatial_vector:
            rows = np.repeat(values[:, None], 6, axis=1)
        elif array.dtype == wp.vec3:
            rows = np.repeat(values[:, None], 3, axis=1)
        else:
            rows = values
        array.assign(rows)
        return rows

    @classmethod
    def _seed_static_state(cls, solver: SolverCoupledADMM):
        group_before = []
        for group_index, group in enumerate(cls._static_groups(solver)):
            reset_before = {}
            for attr_index, attr in enumerate(cls._STATIC_RESET_ATTRS):
                array = getattr(group, attr, None)
                if array is not None:
                    reset_before[attr] = cls._seed_array(array, 10.0 * (group_index + 1) + attr_index)
            topology_before = {
                name: value.numpy().copy()
                for name, value in vars(group).items()
                if isinstance(value, wp.array) and name not in cls._STATIC_RESET_ATTRS
            }
            group_before.append((group, cls._group_endpoint_worlds(solver, group), reset_before, topology_before))

        buffer_before = []
        offset = 100.0
        for name, entry in solver._entries.items():
            buf = solver._admm_buffers[name]
            snapshot_rows = (
                ("body_q_n", entry.state_0.body_q, entry.view.body_world),
                ("body_qd_n", entry.state_0.body_qd, entry.view.body_world),
                ("body_qd_k", entry.state_0.body_qd, entry.view.body_world),
                ("particle_q_n", entry.state_0.particle_q, entry.view.particle_world),
                ("particle_qd_n", entry.state_0.particle_qd, entry.view.particle_world),
                ("particle_qd_k", entry.state_0.particle_qd, entry.view.particle_world),
                ("joint_q_n", entry.state_0.joint_q, entry.joint_coord_world),
                ("joint_qd_n", entry.state_0.joint_qd, entry.joint_dof_world),
                ("joint_qd_k", entry.state_0.joint_qd, entry.joint_dof_world),
            )
            snapshots = []
            for attr, source, worlds in snapshot_rows:
                array = getattr(buf, attr)
                if array is not None:
                    snapshots.append((attr, source, worlds, cls._seed_array(array, offset)))
                    offset += 10.0
            forces = []
            for attr, worlds in (("body_f", entry.view.body_world), ("particle_f", entry.view.particle_world)):
                array = getattr(buf, attr)
                if array is not None:
                    forces.append((attr, worlds, cls._seed_array(array, offset)))
                    offset += 10.0
            effective = {
                attr: value.numpy().copy()
                for attr, value in vars(buf).items()
                if "effective" in attr and isinstance(value, wp.array)
            }
            buffer_before.append((buf, snapshots, forces, effective))
        return group_before, buffer_before

    @staticmethod
    def _seed_parent_state(state: newton.State) -> None:
        offset = 500.0
        for array in (
            state.body_q,
            state.body_qd,
            state.particle_q,
            state.particle_qd,
            state.joint_q,
            state.joint_qd,
        ):
            if array is not None:
                TestAdmmReset._seed_array(array, offset)
                offset += 100.0

    def test_static_groups_have_row_aligned_world_ids(self):
        _model, solver = self._build_solver()

        group_lists = (
            solver._admm_rr_groups,
            solver._admm_rr_angular_groups,
            solver._admm_rr_revolute_angular_groups,
            solver._admm_rr_angular_friction_groups,
            solver._admm_rp_groups,
        )
        self.assertTrue(all(len(groups) == 1 for groups in group_lists))
        for group in self._static_groups(solver):
            self.assertTrue(hasattr(group, "world_ids"))
            np.testing.assert_array_equal(group.world_ids.numpy(), self._group_endpoint_worlds(solver, group))

    def test_partial_reset_partitions_dynamic_contact_warm_starts(self):
        for contact_kind in ("rr", "rp", "pp"):
            with self.subTest(contact_kind=contact_kind):
                model_reset, solver_reset = self._build_dynamic_contact_solver(contact_kind)
                model_control, solver_control = self._build_dynamic_contact_solver(contact_kind)
                state_reset = model_reset.state()
                state_control = model_control.state()

                solver_reset._refresh_collision_contact_groups(state_reset)
                solver_control._refresh_collision_contact_groups(state_control)

                group_reset = self._dynamic_group(solver_reset, contact_kind)
                group_control = self._dynamic_group(solver_control, contact_kind)
                self.assertEqual(group_reset.world_ids.shape[0], group_reset.count)
                active_before = group_reset.active.numpy().copy()
                worlds_before = group_reset.world_ids.numpy().copy()
                self.assertEqual(set(worlds_before[active_before != 0]), {0, 1})
                self.assertTrue(np.all(worlds_before[active_before == 0] == -1))

                seeded_reset = self._seed_dynamic_contact_state(group_reset)
                self._seed_dynamic_contact_state(group_control)
                group_reset.active_count_max.fill_(group_reset.count)
                group_control.active_count_max.fill_(group_control.count)
                active_count_before = group_reset.active_count.numpy().copy()
                active_count_max_before = group_reset.active_count_max.numpy().copy()
                self.assertGreater(int(active_count_before[0]), 0)
                self.assertLess(int(active_count_before[0]), group_reset.count)
                collision_high_water_before = solver_reset.collision_contact_count_max
                topology_before = {
                    name: value.numpy().copy()
                    for name, value in vars(group_reset).items()
                    if isinstance(value, wp.array) and name not in self._DYNAMIC_RESET_ATTRS and name != "active_count"
                }

                stream = getattr(group_reset, "contact_stream", None)
                if stream is not None:
                    self.assertEqual(stream.world_ids.shape[0], stream.capacity)
                    stream_count_before = stream.count.numpy().copy()
                    stream_count_max_before = stream.count_max.numpy().copy()
                    stream_worlds_before = stream.world_ids.numpy().copy()

                solver_reset.reset(
                    state_reset,
                    world_mask=wp.array([True, False], dtype=wp.bool, device=model_reset.device),
                )

                if solver_reset._admm_internal_contacts is not None:
                    self.assertEqual(int(solver_reset._admm_internal_contacts.rigid_contact_count.numpy()[0]), 0)
                    self.assertEqual(int(solver_reset._admm_internal_contacts.soft_contact_count.numpy()[0]), 0)
                np.testing.assert_array_equal(group_reset.active_count.numpy(), np.zeros_like(active_count_before))
                np.testing.assert_array_equal(group_reset.active_count_max.numpy(), active_count_max_before)
                self.assertEqual(solver_reset.collision_contact_count_max, collision_high_water_before)
                for name, before in topology_before.items():
                    np.testing.assert_array_equal(getattr(group_reset, name).numpy(), before)

                selected = worlds_before == 0
                preserved = worlds_before != 0
                for attr in self._DYNAMIC_RESET_ATTRS:
                    actual = getattr(group_reset, attr).numpy()
                    expected = seeded_reset[attr].copy()
                    expected[selected] = 0.0
                    np.testing.assert_array_equal(actual, expected)
                    np.testing.assert_array_equal(actual[preserved], seeded_reset[attr][preserved])

                if stream is not None:
                    np.testing.assert_array_equal(stream.count.numpy(), np.zeros_like(stream_count_before))
                    np.testing.assert_array_equal(stream.count_max.numpy(), stream_count_max_before)
                    stream_selected = stream_worlds_before == 0
                    for attr in ("normal_force", "normal_impulse"):
                        actual = getattr(stream, attr).numpy()
                        expected = seeded_reset[f"stream.{attr}"].copy()
                        expected[stream_selected] = 0.0
                        np.testing.assert_array_equal(actual, expected)

                solver_reset._refresh_collision_contact_groups(state_reset)
                solver_control._refresh_collision_contact_groups(state_control)

                reset_unselected = self._dynamic_contact_rows(group_reset, contact_kind, world=1)
                control_unselected = self._dynamic_contact_rows(group_control, contact_kind, world=1)
                self.assertEqual([key for key, _value in reset_unselected], [key for key, _value in control_unselected])
                np.testing.assert_array_equal(
                    np.asarray([value for _key, value in reset_unselected]),
                    np.asarray([value for _key, value in control_unselected]),
                )
                self.assertTrue(any(np.any(value != 0.0) for _key, value in reset_unselected))

                reset_selected = self._dynamic_contact_rows(group_reset, contact_kind, world=0)
                control_selected = self._dynamic_contact_rows(group_control, contact_kind, world=0)
                self.assertEqual([key for key, _value in reset_selected], [key for key, _value in control_selected])
                self.assertTrue(all(np.all(value == 0.0) for _key, value in reset_selected))
                self.assertTrue(any(np.any(value != 0.0) for _key, value in control_selected))
                np.testing.assert_array_equal(group_reset.active_count_max.numpy(), active_count_max_before)
                self.assertEqual(solver_reset.collision_contact_count_max, collision_high_water_before)

    def test_partial_reset_forces_selected_sticky_rigid_contacts_fresh(self):
        model, solver = self._build_dynamic_contact_solver("rr", rigid_contact_matching="sticky")
        state = model.state()
        solver._refresh_collision_contact_groups(state)
        group = self._dynamic_group(solver, "rr")
        old_selected_normal = group.normal.numpy()[(group.active.numpy() != 0) & (group.world_ids.numpy() == 0)].copy()
        self.assertGreater(len(old_selected_normal), 0)

        body_world = model.body_world.numpy()
        world_zero_bodies = np.flatnonzero(body_world == 0)
        self.assertEqual(len(world_zero_bodies), 2)
        body_q = state.body_q.numpy()
        body_q[world_zero_bodies[1], 1] += 0.0001
        state.body_q.assign(body_q)

        fresh_model, fresh_solver = self._build_dynamic_contact_solver("rr", rigid_contact_matching="sticky")
        fresh_state = fresh_model.state()
        fresh_body_q = fresh_state.body_q.numpy()
        fresh_body_q[np.flatnonzero(fresh_model.body_world.numpy() == 0)[1], 1] += 0.0001
        fresh_state.body_q.assign(fresh_body_q)
        fresh_solver._refresh_collision_contact_groups(fresh_state)
        fresh_group = self._dynamic_group(fresh_solver, "rr")
        fresh_selected_normal = fresh_group.normal.numpy()[
            (fresh_group.active.numpy() != 0) & (fresh_group.world_ids.numpy() == 0)
        ]
        self.assertFalse(np.array_equal(old_selected_normal, fresh_selected_normal))

        mask = wp.array([True, False], dtype=wp.bool, device=model.device)
        solver.reset(state, world_mask=mask)
        solver._refresh_collision_contact_groups(state)

        selected_normal = group.normal.numpy()[(group.active.numpy() != 0) & (group.world_ids.numpy() == 0)]
        np.testing.assert_array_equal(selected_normal, fresh_selected_normal)

        contacts = solver._admm_internal_contacts
        count = int(contacts.rigid_contact_count.numpy()[0])
        shape0 = contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = contacts.rigid_contact_shape1.numpy()[:count]
        shape_body = model.shape_body.numpy()
        worlds0 = model.body_world.numpy()[shape_body[shape0]]
        worlds1 = model.body_world.numpy()[shape_body[shape1]]
        contact_worlds = np.where(worlds0 >= 0, worlds0, worlds1)
        match_index = contacts.rigid_contact_match_index.numpy()[:count]
        self.assertTrue(np.all(match_index[contact_worlds == 0] == -1))
        self.assertTrue(np.all(match_index[contact_worlds == 1] >= 0))

    def test_dynamic_reset_kernel_preserves_global_rows_on_available_devices(self):
        kernel = getattr(admm_utils, "reset_dynamic_admm_rows_kernel", None)
        self.assertIsNotNone(kernel)
        devices = [wp.get_device("cpu")]
        if wp.is_cuda_available():
            devices.append(wp.get_device("cuda:0"))

        for device in devices:
            with self.subTest(device=str(device)):
                world_ids = wp.array([0, 1, -1], dtype=int, device=device)
                world_mask = wp.array([True, False], dtype=wp.bool, device=device)
                vector_arrays = [
                    wp.full(3, wp.vec3(float(index + 1)), dtype=wp.vec3, device=device) for index in range(3)
                ]
                u_min = wp.full(3, 4.0, dtype=float, device=device)

                wp.launch(kernel, dim=3, inputs=[world_ids, world_mask, *vector_arrays, u_min], device=device)

                for index, array in enumerate(vector_arrays):
                    expected = np.full((3, 3), float(index + 1), dtype=np.float32)
                    expected[0] = 0.0
                    np.testing.assert_array_equal(array.numpy(), expected)
                np.testing.assert_array_equal(u_min.numpy(), np.asarray([0.0, 4.0, 4.0], dtype=np.float32))

    def test_partial_reset_partitions_static_groups_and_entry_buffers(self):
        model, solver = self._build_solver()
        state = model.state()
        self._seed_parent_state(state)
        group_before, buffer_before = self._seed_static_state(solver)
        self.assertTrue(any(buf.joint_q_n is not None for buf, _snapshots, _forces, _effective in buffer_before))

        dynamic_high_water = []
        dynamic_groups = (*solver._admm_dynamic_rr_contact_groups, *solver._admm_dynamic_rp_contact_groups)
        self.assertGreater(len(dynamic_groups), 0)
        for index, group in enumerate(dynamic_groups, start=1):
            group.active_count_max.fill_(20 + index)
            dynamic_high_water.append((group.active_count_max, group.active_count_max.numpy().copy()))

        solver.reset(state, world_mask=wp.array([True, False], dtype=wp.bool, device=model.device))

        for group, row_worlds, reset_before, topology_before in group_before:
            selected = row_worlds == 0
            for attr, before in reset_before.items():
                expected = before.copy()
                expected[selected] = 0.0
                np.testing.assert_array_equal(getattr(group, attr).numpy(), expected)
            for attr, before in topology_before.items():
                np.testing.assert_array_equal(getattr(group, attr).numpy(), before)

        for buf, snapshots, forces, effective_before in buffer_before:
            for attr, source, worlds, before in snapshots:
                expected = before.copy()
                selected = worlds.numpy() == 0
                expected[selected] = source.numpy()[selected]
                np.testing.assert_array_equal(getattr(buf, attr).numpy(), expected)
            for attr, worlds, before in forces:
                expected = before.copy()
                expected[worlds.numpy() == 0] = 0.0
                np.testing.assert_array_equal(getattr(buf, attr).numpy(), expected)
            for attr, before in effective_before.items():
                np.testing.assert_array_equal(getattr(buf, attr).numpy(), before)

        for array, before in dynamic_high_water:
            np.testing.assert_array_equal(array.numpy(), before)

    def test_static_reset_kernel_preserves_global_and_unselected_rows(self):
        kernel = getattr(admm_utils, "reset_static_admm_rows_kernel", None)
        self.assertIsNotNone(kernel)
        world_ids = wp.array([0, 1, -1], dtype=int, device="cpu")
        world_mask = wp.array([True, False], dtype=wp.bool, device="cpu")
        arrays = [wp.full(3, wp.vec3(float(index + 1)), dtype=wp.vec3, device="cpu") for index in range(4)]

        wp.launch(kernel, dim=3, inputs=[world_ids, world_mask, *arrays, 1], device="cpu")

        for index, array in enumerate(arrays):
            expected = np.full((3, 3), float(index + 1), dtype=np.float32)
            expected[0] = 0.0
            np.testing.assert_array_equal(array.numpy(), expected)

    def test_body_particle_attachment_rejects_cross_world_endpoints(self):
        template = newton.ModelBuilder(gravity=0.0)
        template.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        template.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0)
        builder = newton.ModelBuilder(gravity=0.0)
        builder.replicate(template, world_count=2)
        SolverCoupledADMM.add_body_particle_attachment(builder, body=0, particle=1)
        model = builder.finalize(device="cpu")

        with self.assertRaisesRegex(ValueError, "same world"):
            SolverCoupledADMM(
                model=model,
                entries=[
                    SolverCoupled.Entry(name="body", solver=_CustomAdmmParticleCopySolver, bodies=[0]),
                    SolverCoupled.Entry(name="particle", solver=_CustomAdmmParticleCopySolver, particles=[1]),
                ],
                coupling=SolverCoupledADMM.Config(iterations=1),
            )

    def test_full_world_mask_matches_full_reset_for_static_state(self):
        model_full, solver_full = self._build_solver()
        model_masked, solver_masked = self._build_solver()
        state_full = model_full.state()
        state_masked = model_masked.state()
        self._seed_parent_state(state_full)
        self._seed_parent_state(state_masked)
        self._seed_static_state(solver_full)
        self._seed_static_state(solver_masked)

        solver_full.reset(state_full)
        solver_masked.reset(
            state_masked,
            world_mask=wp.array([True, True], dtype=wp.bool, device=model_masked.device),
        )

        for group_full, group_masked in zip(
            self._static_groups(solver_full), self._static_groups(solver_masked), strict=True
        ):
            for attr in self._STATIC_RESET_ATTRS:
                array_full = getattr(group_full, attr, None)
                array_masked = getattr(group_masked, attr, None)
                if array_full is not None:
                    np.testing.assert_array_equal(array_masked.numpy(), array_full.numpy())
        for name in solver_full._entries:
            buf_full = solver_full._admm_buffers[name]
            buf_masked = solver_masked._admm_buffers[name]
            for attr in (
                "body_q_n",
                "body_qd_n",
                "body_qd_k",
                "particle_q_n",
                "particle_qd_n",
                "particle_qd_k",
                "joint_q_n",
                "joint_qd_n",
                "joint_qd_k",
                "body_f",
                "particle_f",
            ):
                array_full = getattr(buf_full, attr)
                array_masked = getattr(buf_masked, attr)
                if array_full is not None:
                    np.testing.assert_array_equal(array_masked.numpy(), array_full.numpy())


class TestAdmmDiagnostics(unittest.TestCase):
    """Public, on-demand ADMM convergence and contact diagnostics."""

    @staticmethod
    def _dynamic_groups(solver: SolverCoupledADMM):
        return (
            *solver._admm_dynamic_rr_contact_groups,
            *solver._admm_dynamic_rp_contact_groups,
            *solver._admm_dynamic_pp_contact_groups,
        )

    @classmethod
    def _all_groups(cls, solver: SolverCoupledADMM):
        return (*TestAdmmReset._static_groups(solver), *cls._dynamic_groups(solver))

    @classmethod
    def _seed_diagnostic_state(cls, solver: SolverCoupledADMM):
        """Seed rows whose primal and proximal stationarity residuals are one."""
        expected_current = 0
        expected_max = 0
        expected_overflow = 0
        expected_residual_rows = 0

        for group in cls._all_groups(solver):
            if group.count == 0:
                continue

            group.u.fill_(wp.vec3(1.0, 0.0, 0.0))
            group.Jv.zero_()
            group.lambda_.fill_(wp.vec3(1.0, 0.0, 0.0))
            group.W.fill_(1.0)
            u_target = getattr(group, "u_target", None)
            if u_target is not None:
                u_target.zero_()
            kappa = getattr(group, "kappa", None)
            if kappa is not None:
                kappa.fill_(1.0)
            damping = getattr(group, "damping", None)
            if damping is not None:
                damping.zero_()
            friction = getattr(group, "friction", None)
            if friction is not None:
                friction.zero_()
            u_min = getattr(group, "u_min", None)
            if u_min is not None:
                u_min.fill_(-1.0e9)

            active_count = getattr(group, "active_count", None)
            if active_count is None:
                expected_residual_rows += group.count
                continue

            active_count.fill_(group.count)
            group.active_count_max.fill_(group.count)
            group.active.fill_(1)
            contact_stream = getattr(group, "contact_stream", None)
            if contact_stream is not None:
                contact_stream.count.fill_(contact_stream.capacity)
                contact_stream.count_max.fill_(contact_stream.capacity)
            expected_current += group.count
            expected_max += group.count
            expected_residual_rows += group.count

        snapshots = [
            (value, value.numpy().copy())
            for group in cls._all_groups(solver)
            for value in vars(group).values()
            if isinstance(value, wp.array)
        ]
        return {
            "contact_count": expected_current,
            "contact_count_max": expected_max,
            "contact_overflow": expected_overflow,
            "residual_norm": math.sqrt(float(expected_residual_rows)),
            "snapshots": snapshots,
        }

    def _assert_diagnostics(self, model, solver, expected):
        diagnostics = solver.diagnostics()
        self.assertIsInstance(diagnostics, coupled_api.AdmmDiagnostics)
        self.assertEqual(diagnostics.iterations, solver._coupling.iterations)
        self.assertEqual(len(diagnostics.interfaces), 1)

        count_arrays = (
            (diagnostics.contact_count, expected["contact_count"]),
            (diagnostics.contact_count_max, expected["contact_count_max"]),
            (diagnostics.contact_overflow, expected["contact_overflow"]),
        )
        for array, expected_value in count_arrays:
            self.assertEqual(array.shape, (1,))
            self.assertEqual(array.dtype, wp.int32)
            self.assertEqual(array.device, model.device)
            self.assertEqual(int(array.numpy()[0]), expected_value)

        interface = diagnostics.interfaces[0]
        self.assertIsInstance(interface, coupled_api.AdmmInterfaceDiagnostics)
        self.assertEqual((interface.source, interface.destination), ("a", "b"))
        for residual in (interface.primal_residual_norm, interface.dual_residual_norm):
            self.assertEqual(residual.shape, (1,))
            self.assertEqual(residual.dtype, wp.float32)
            self.assertEqual(residual.device, model.device)
            self.assertTrue(np.isfinite(residual.numpy()[0]))
            self.assertAlmostEqual(float(residual.numpy()[0]), expected["residual_norm"], places=5)

        return diagnostics

    def test_public_schema_aggregates_static_and_dynamic_interfaces_without_mutation(self):
        self.assertTrue(hasattr(coupled_api, "AdmmInterfaceDiagnostics"))
        self.assertTrue(hasattr(coupled_api, "AdmmDiagnostics"))
        model, solver = TestAdmmReset._build_solver()
        expected = self._seed_diagnostic_state(solver)

        diagnostics = self._assert_diagnostics(model, solver, expected)
        diagnostics_again = solver.diagnostics()

        self.assertIs(diagnostics_again, diagnostics)
        self.assertIs(diagnostics_again.contact_count, diagnostics.contact_count)
        self.assertIs(diagnostics_again.contact_count_max, diagnostics.contact_count_max)
        self.assertIs(diagnostics_again.contact_overflow, diagnostics.contact_overflow)
        self.assertIs(
            diagnostics_again.interfaces[0].primal_residual_norm,
            diagnostics.interfaces[0].primal_residual_norm,
        )
        self.assertIs(
            diagnostics_again.interfaces[0].dual_residual_norm,
            diagnostics.interfaces[0].dual_residual_norm,
        )
        for array, before in expected["snapshots"]:
            np.testing.assert_array_equal(array.numpy(), before)

        with self.assertRaises(FrozenInstanceError):
            diagnostics.iterations = 0
        with self.assertRaises(FrozenInstanceError):
            diagnostics.interfaces[0].source = "replacement"

    def test_masked_reset_clears_current_contact_diagnostics_but_preserves_high_water(self):
        """Diagnostics must agree with the cleared public ADMM contact stream after reset."""
        for contact_kind in ("rr", "rp", "pp"):
            with self.subTest(contact_kind=contact_kind):
                model, solver = TestAdmmReset._build_dynamic_contact_solver(contact_kind)
                state = model.state()
                solver._refresh_collision_contact_groups(state)

                internal_streams = [stream for stream in solver.contact_streams() if stream.name == "admm/internal"]
                if contact_kind != "pp":
                    self.assertEqual(len(internal_streams), 1)
                    stream_before = internal_streams[0].diagnostics()
                    raw_count_before = stream_before.rigid_count if contact_kind == "rr" else stream_before.soft_count
                    self.assertGreater(raw_count_before, 0)
                diagnostics_before = solver.diagnostics()
                current_before = int(diagnostics_before.contact_count.numpy()[0])
                high_water_before = int(diagnostics_before.contact_count_max.numpy()[0])
                self.assertGreater(current_before, 0)
                self.assertGreater(high_water_before, 0)

                solver.reset(
                    state,
                    world_mask=wp.array([True, False], dtype=wp.bool, device=model.device),
                )

                if contact_kind != "pp":
                    stream_after = next(
                        stream for stream in solver.contact_streams() if stream.name == "admm/internal"
                    ).diagnostics()
                    raw_count_after = stream_after.rigid_count if contact_kind == "rr" else stream_after.soft_count
                    self.assertEqual(raw_count_after, 0)
                self.assertEqual(solver.collision_contact_count, 0)
                self.assertEqual(solver.collision_contact_count_max, high_water_before)
                diagnostics_after = solver.diagnostics()
                self.assertEqual(int(diagnostics_after.contact_count.numpy()[0]), 0)
                self.assertEqual(int(diagnostics_after.contact_count_max.numpy()[0]), high_water_before)
                self.assertEqual(int(diagnostics_after.contact_overflow.numpy()[0]), 0)

    def test_particle_particle_rows_and_on_demand_updates(self):
        self.assertTrue(hasattr(SolverCoupledADMM, "diagnostics"))
        model, solver = TestAdmmReset._build_dynamic_contact_solver("pp")
        expected = self._seed_diagnostic_state(solver)
        group = solver._admm_dynamic_pp_contact_groups[0]
        group.u.fill_(wp.vec3(1.0, 0.0, 0.0))
        group.Jv.zero_()
        group.lambda_.fill_(wp.vec3(1.0, 0.0, 0.0))
        group.W.fill_(1.0)
        group.friction.zero_()
        group.u_min.fill_(-1.0e9)
        diagnostics = self._assert_diagnostics(model, solver, expected)

        diagnostics.contact_count.fill_(91)
        diagnostics.contact_count_max.fill_(92)
        diagnostics.contact_overflow.fill_(93)
        diagnostics.interfaces[0].primal_residual_norm.fill_(94.0)
        diagnostics.interfaces[0].dual_residual_norm.fill_(95.0)

        state_0 = model.state()
        state_1 = model.state()
        solver.step(state_0, state_1, control=None, contacts=None, dt=0.01)

        self.assertEqual(int(diagnostics.contact_count.numpy()[0]), 91)
        self.assertEqual(int(diagnostics.contact_count_max.numpy()[0]), 92)
        self.assertEqual(int(diagnostics.contact_overflow.numpy()[0]), 93)
        self.assertEqual(float(diagnostics.interfaces[0].primal_residual_norm.numpy()[0]), 94.0)
        self.assertEqual(float(diagnostics.interfaces[0].dual_residual_norm.numpy()[0]), 95.0)

        diagnostics_after_step = solver.diagnostics()
        self.assertIs(diagnostics_after_step, diagnostics)
        self.assertTrue(np.isfinite(diagnostics.interfaces[0].primal_residual_norm.numpy()[0]))
        self.assertTrue(np.isfinite(diagnostics.interfaces[0].dual_residual_norm.numpy()[0]))

    def test_particle_particle_detector_overflow_is_raw_and_capacity_bounded(self):
        model, solver = TestAdmmReset._build_dynamic_contact_solver("pp")
        group = solver._admm_dynamic_pp_contact_groups[0]
        contact_stream = group.contact_stream
        contact_stream.capacity = 1

        contact_stream.particle_a.fill_(-11)
        contact_stream.particle_b.fill_(-12)
        contact_stream.normal.fill_(wp.vec3(13.0, 14.0, 15.0))
        contact_stream.world_ids.fill_(-16)
        contact_stream.source_id.fill_(-17)
        tail_before = {
            name: getattr(contact_stream, name).numpy()[contact_stream.capacity :].copy()
            for name in ("particle_a", "particle_b", "normal", "world_ids", "source_id")
        }

        solver._refresh_collision_contact_groups(model.state())

        self.assertEqual(int(contact_stream.count.numpy()[0]), 2)
        self.assertEqual(int(contact_stream.count_max.numpy()[0]), contact_stream.capacity)
        self.assertEqual(int(group.active_count.numpy()[0]), contact_stream.capacity)
        self.assertEqual(int(group.active_count_max.numpy()[0]), contact_stream.capacity)
        self.assertEqual(solver.collision_contact_count, contact_stream.capacity)
        self.assertEqual(solver.collision_contact_count_max, contact_stream.capacity)
        for name, expected_tail in tail_before.items():
            np.testing.assert_array_equal(
                getattr(contact_stream, name).numpy()[contact_stream.capacity :],
                expected_tail,
                err_msg=name,
            )

        u = np.full((group.count, 3), np.nan, dtype=np.float32)
        u[0] = 0.0
        group.u.assign(u)
        group.Jv.zero_()
        group.lambda_.zero_()
        diagnostics = solver.diagnostics()

        self.assertEqual(int(diagnostics.contact_count.numpy()[0]), contact_stream.capacity)
        self.assertEqual(int(diagnostics.contact_count_max.numpy()[0]), contact_stream.capacity)
        self.assertEqual(int(diagnostics.contact_overflow.numpy()[0]), 1)
        self.assertEqual(float(diagnostics.interfaces[0].primal_residual_norm.numpy()[0]), 0.0)
        self.assertEqual(float(diagnostics.interfaces[0].dual_residual_norm.numpy()[0]), 0.0)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA graph capture requires CUDA")
    def test_diagnostics_reductions_are_graph_capturable(self):
        self.assertTrue(hasattr(SolverCoupledADMM, "diagnostics"))
        model, solver = TestAdmmReset._build_dynamic_contact_solver("pp", device="cuda:0")
        expected = self._seed_diagnostic_state(solver)
        diagnostics = self._assert_diagnostics(model, solver, expected)

        with wp.ScopedCapture(device=model.device) as capture:
            diagnostics_during_capture = solver.diagnostics()

        self.assertIs(diagnostics_during_capture, diagnostics)
        self.assertIsNotNone(capture.graph)
        wp.capture_launch(capture.graph)
        self.assertTrue(np.isfinite(diagnostics.interfaces[0].primal_residual_norm.numpy()[0]))
        self.assertTrue(np.isfinite(diagnostics.interfaces[0].dual_residual_norm.numpy()[0]))


class TestAdmmExternalForces(unittest.TestCase):
    """External forces set on ``state_in.body_f`` / ``particle_f`` by the
    caller (e.g. a viewer gizmo) must reach the sub-solvers."""

    def test_body_f_reaches_mujoco(self):
        """An upward ``body_f`` on the rigid sphere should slow its fall
        compared to the zero-force baseline."""
        # Baseline: no external force, body falls under gravity.
        model_a, rs, re, _ = _build_cloth_rigid_scene(rigid_pos=(0.0, 0.0, 5.0))
        solver_a = _make_solver(model_a, rs, re, admm_iters=1)
        state_0 = model_a.state()
        state_1 = model_a.state()
        contacts = model_a.contacts()
        control = model_a.control()
        newton.eval_fk(model_a, model_a.joint_q, model_a.joint_qd, state_0)
        for _ in range(5):
            state_0.clear_forces()
            model_a.collide(state_0, contacts)
            solver_a.step(state_0, state_1, control, contacts, 1.0 / 60.0)
            state_0, state_1 = state_1, state_0
        z_baseline = state_0.body_q.numpy()[0, 2]

        # With a strong upward body_f applied each step, the body should fall
        # less (or even rise).
        model_b, rs, re, _ = _build_cloth_rigid_scene(rigid_pos=(0.0, 0.0, 5.0))
        solver_b = _make_solver(model_b, rs, re, admm_iters=1)
        state_0 = model_b.state()
        state_1 = model_b.state()
        contacts = model_b.contacts()
        control = model_b.control()
        newton.eval_fk(model_b, model_b.joint_q, model_b.joint_qd, state_0)
        body_idx = rs  # only MuJoCo body
        body_mass = float(model_b.body_mass.numpy()[body_idx])
        upward_force = 5.0 * body_mass * 9.81  # 5 g upward wrench
        for _ in range(5):
            state_0.clear_forces()
            wrench = np.zeros((model_b.body_count, 6), dtype=np.float32)
            wrench[body_idx, 2] = upward_force  # linear z
            state_0.body_f = wp.array(wrench, dtype=wp.spatial_vector, device=model_b.device)
            model_b.collide(state_0, contacts)
            solver_b.step(state_0, state_1, control, contacts, 1.0 / 60.0)
            state_0, state_1 = state_1, state_0
        z_with_force = state_0.body_q.numpy()[0, 2]

        self.assertGreater(
            z_with_force,
            z_baseline + 0.02,
            f"external body_f didn't reach MuJoCo: baseline z={z_baseline:.4f}, "
            f"with 5g upward force z={z_with_force:.4f}",
        )


class TestAdmmCollisionDetection(unittest.TestCase):
    """Collision-detected ADMM contact constraints."""

    def test_internal_contacts_are_exposed_as_raw_public_stream(self):
        model, plane_body, _, particle_ids = _build_inclined_plane_particle_box_scene(math.radians(10.0))
        solver = _make_admm_inclined_plane_particle_box_solver(
            model,
            plane_body,
            particle_ids,
            math.radians(10.0),
            friction=0.0,
        )

        streams = solver.contact_streams()

        self.assertEqual([stream.name for stream in streams], ["admm/internal"])
        stream = streams[0]
        self.assertIsInstance(stream, coupled_api.CoupledContactStream)
        self.assertEqual(stream.kind, "admm")
        self.assertIs(stream.contacts, solver._admm_internal_contacts)
        self.assertIsNone(stream.source)
        self.assertIsNone(stream.destination)
        self.assertIsNone(stream.shape_local_to_parent)
        self.assertIsNone(stream.particle_local_to_parent)
        self.assertFalse(stream.forces_available)
        self.assertFalse(hasattr(coupled_api, "AdmmContactStream"))

    def test_failed_step_hides_internal_contact_stream_until_reset(self):
        model, plane_body, _, particle_ids = _build_inclined_plane_particle_box_scene(math.radians(10.0))
        failing_solvers = []
        solver = SolverCoupledADMM(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="plane",
                    solver=_CustomAdmmParticleCopySolver,
                    bodies=[plane_body],
                ),
                SolverCoupled.Entry(
                    name="box",
                    solver=lambda view: failing_solvers.append(_FailingAdmmCopySolver(view)) or failing_solvers[-1],
                    particles=particle_ids,
                ),
            ],
            coupling=SolverCoupledADMM.Config(
                iterations=1,
                contact_pairs=[SolverCoupledADMM.ContactPair(source="plane", destination="box")],
            ),
        )
        state = model.state()
        self.assertEqual([stream.name for stream in solver.contact_streams()], ["admm/internal"])

        failing_solvers[0].fail = True
        with self.assertRaisesRegex(RuntimeError, "intentional ADMM step failure"):
            solver.step(state, state, control=None, contacts=None, dt=0.01)

        self.assertEqual(solver.contact_streams(), ())
        self.assertEqual(solver.contact_streams(newton.Contacts(0, 0, device=model.device)), ())

        solver.reset(state)
        self.assertEqual([stream.name for stream in solver.contact_streams()], ["admm/internal"])

    def test_rigid_contact_detection_rejects_cross_world_pairs(self):
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.density = 1000.0

        builder.begin_world()
        body_a = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
        shape_a = builder.add_shape_box(body=body_a, hx=0.05, hy=0.05, hz=0.05)
        builder.end_world()

        builder.begin_world()
        body_b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
        shape_b = builder.add_shape_box(body=body_b, hx=0.05, hy=0.05, hz=0.05)
        builder.end_world()

        model = builder.finalize(device="cpu")
        model.shape_contact_pairs = wp.array(
            np.asarray([(shape_a, shape_b)], dtype=np.int32), dtype=wp.vec2i, device=model.device
        )
        model.shape_contact_pair_count = 1

        with self.assertRaisesRegex(ValueError, "same world"):
            SolverCoupledADMM(
                model=model,
                entries=[
                    SolverCoupled.Entry(
                        name="a",
                        solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                        bodies=[body_a],
                    ),
                    SolverCoupled.Entry(
                        name="b",
                        solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                        bodies=[body_b],
                    ),
                ],
                coupling=SolverCoupledADMM.Config(
                    contact_pairs=[SolverCoupledADMM.ContactPair(source="a", destination="b")],
                ),
            )

    def test_collision_particle_particle_contacts_are_refreshed_in_solver(self):
        model = _build_two_particle_contact_scene(gap=-0.08)
        solver = SolverCoupledADMM(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="a",
                    solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                    particles=[0],
                ),
                SolverCoupled.Entry(
                    name="b",
                    solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                    particles=[1],
                ),
            ],
            coupling=SolverCoupledADMM.Config(
                iterations=10,
                rho=30.0,
                baumgarte=0.5,
                contact_pairs=[
                    SolverCoupledADMM.ContactPair(source="a", destination="b"),
                ],
            ),
        )

        q_contact = _run_particles(solver, model, n_steps=4)

        self.assertGreater(solver.collision_contact_count_max, 0)
        self.assertGreater(q_contact[1, 0] - q_contact[0, 0], 0.08 + 1.0e-3)

    def test_collision_frictional_contact_matches_inclined_plane_box_motion(self):
        friction = 0.4
        angle = math.radians(35.0)
        dt = 1.0 / 360.0
        steps = 120
        displacement, velocity, contact_count = _run_inclined_plane_particle_box(
            angle,
            friction,
            steps=steps,
            dt=dt,
        )

        t = steps * dt
        acceleration = 10.0 * (math.sin(angle) - friction * math.cos(angle))
        expected_displacement = 0.5 * acceleration * t * t
        expected_velocity = acceleration * t

        self.assertGreater(contact_count, 0)
        self.assertGreater(acceleration, 0.0)
        self.assertAlmostEqual(displacement, expected_displacement, delta=0.45 * expected_displacement)
        self.assertAlmostEqual(velocity, expected_velocity, delta=0.45 * expected_velocity)

    def test_collision_frictional_contact_holds_subcritical_inclined_box(self):
        friction = 0.4
        angle = math.radians(15.0)
        displacement, velocity, contact_count = _run_inclined_plane_particle_box(
            angle,
            friction,
            steps=120,
            dt=1.0 / 360.0,
        )

        self.assertGreater(contact_count, 0)
        self.assertLess(math.tan(angle), friction)
        self.assertLess(abs(displacement), 0.01)
        self.assertLess(abs(velocity), 0.05)

    def test_collision_rigid_rigid_frictional_contact_matches_inclined_plane_box_motion(self):
        friction = 0.35
        angle = math.radians(24.0)
        steps = 120
        dt = 1.0 / 360.0
        displacement, velocity, min_gap, contact_count = _run_collision_inclined_plane_rigid_box(
            angle,
            friction,
            steps=steps,
            dt=dt,
        )

        t = steps * dt
        acceleration = 10.0 * (math.sin(angle) - friction * math.cos(angle))
        expected_displacement = 0.5 * acceleration * t * t
        expected_velocity = acceleration * t

        self.assertGreater(contact_count, 0)
        self.assertGreater(acceleration, 0.0)
        self.assertAlmostEqual(displacement, expected_displacement, delta=0.65 * expected_displacement)
        self.assertAlmostEqual(velocity, expected_velocity, delta=0.65 * expected_velocity)
        self.assertGreater(min_gap, -0.03)

    def test_collision_particle_shape_contacts_are_refreshed_in_solver(self):
        model, particle, tray_body, _ = _build_collision_contact_scene()
        solver = SolverCoupledADMM(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="drop",
                    solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                    particles=[particle],
                ),
                SolverCoupled.Entry(
                    name="tray",
                    solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                    bodies=[tray_body],
                ),
            ],
            coupling=SolverCoupledADMM.Config(
                iterations=12,
                rho=45.0,
                gamma=0.05,
                baumgarte=0.1,
                contact_pairs=[
                    SolverCoupledADMM.ContactPair(
                        source="drop",
                        destination="tray",
                    ),
                ],
            ),
        )

        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        initial_tray_z = float(state_0.body_q.numpy()[tray_body, 2])
        min_gap = float(state_0.particle_q.numpy()[particle, 2] - initial_tray_z)
        for _ in range(90):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts=None, dt=1.0 / 120.0)
            state_0, state_1 = state_1, state_0
            particle_z = float(state_0.particle_q.numpy()[particle, 2])
            tray_z = float(state_0.body_q.numpy()[tray_body, 2])
            min_gap = min(min_gap, particle_z - tray_z)

        final_particle_z = float(state_0.particle_q.numpy()[particle, 2])
        final_tray_z = float(state_0.body_q.numpy()[tray_body, 2])
        final_gap = final_particle_z - final_tray_z
        self.assertGreater(solver.collision_contact_count_max, 0)
        self.assertLessEqual(min_gap, 0.08)
        self.assertGreater(min_gap, -0.02)
        self.assertGreater(final_gap, 0.02)
        self.assertLess(final_tray_z, initial_tray_z - 1.0e-3)


if __name__ == "__main__":
    unittest.main()
