# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Waterhose Insertion (proxy coupling)
#
# An RBY1 robot grasps a VBD waterhose connector, inserts it into a
# refrigerator socket, releases it, and backs away. SolverMuJoCo simulates
# the robot, SolverVBD simulates the hose, and SolverCoupledProxy transfers
# gripper contact between them.
#
# Standalone: depends only on Newton and Warp (no IsaacLab).
#
# Command: python -m newton.examples waterhose_insert
###########################################################################

from __future__ import annotations

import math
from enum import IntEnum
from pathlib import Path

import numpy as np
import warp as wp
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledProxy
from pxr import Usd, UsdGeom

import newton
import newton.examples
import newton.ik as ik
from newton.math import quat_between_vectors_robust
from newton.solvers import SolverMuJoCo, SolverVBD

ASSET_DIR = Path(newton.examples.get_asset_directory()) / "waterhose"
ROBOT_USD = ASSET_DIR / "rby1df/rby1df_waterhose.usda"
FRIDGE_USD = ASSET_DIR / "fridge/fridge_waterhose.usda"
CABLE_USD = ASSET_DIR / "fridge/cable/cable001.usda"
PLUG_USD = ASSET_DIR / "fridge/cable/plug.usda"

SCENE_VERTICAL_OFFSET = 1.05
GROUND_HEIGHT = 0.0
FRIDGE_POSITION = wp.vec3(0.0, 0.0, 0.5 + SCENE_VERTICAL_OFFSET)
ROBOT_TRANSFORM = wp.transform(
    wp.vec3(0.0, 1.0, -1.0 + SCENE_VERTICAL_OFFSET),
    wp.quat(0.0, 0.0, -0.70710678, 0.70710678),
)
PLUG_TRANSFORM = wp.transform(
    wp.vec3(-0.38398558, 0.34585292, 0.13125312 + SCENE_VERTICAL_OFFSET),
    wp.quat(0.0, -0.57096256, 0.0, 0.8209761),
)

CABLE_RADIUS = 0.003
CONTACT_KE = 1.0e4
CONTACT_KD = 1.0e-1
CONTACT_MU = 0.5

SOCKET_POSITION = wp.vec3(-0.259345, 0.344709, 0.28698 + SCENE_VERTICAL_OFFSET)
SOCKET_ROTATION = wp.quat(0.173648, 0.0, 0.0, 0.984808)
RIGHT_EE_TRANSFORM = wp.transform(
    wp.vec3(0.0, 0.0, -0.125),
    wp.quat(0.70710677, 0.70710677, 0.0, 0.0),
)
PLUG_GRASP_OFFSET = wp.vec3(0.0, -CABLE_RADIUS + 0.002, 0.003)
CONNECTOR_TIP_LENGTH = 0.014106234

REST = 0
APPROACH = 1
ENGAGE = 2
GRASP = 3
HOLD_GRASP = 4
RETRACT = 5
SETTLE = 6
LIFT = 7
CARRY = 8
ALIGN = 9
INSERT = 10
HOLD_INSERTED = 11
RELEASE = 12
BACKOFF = 13
DONE = 14

_RIGHT_FINGER_BODY_SUFFIXES = ("right_gripper_leftfinger", "right_gripper_rightfinger")

_RBY1_INITIAL_Q = {
    "torso_joint_1": 0.0,
    "torso_joint_2": 0.872664213180542,
    "torso_joint_3": -1.5707811117172241,
    "torso_joint_4": 0.6981245279312134,
    "torso_joint_5": 3.796982127823867e-06,
    "torso_joint_6": 0.0,
    "right_arm_joint_1": 0.3021828234195709,
    "right_arm_joint_2": -0.013802030123770237,
    "right_arm_joint_3": -0.09509921818971634,
    "right_arm_joint_4": -2.2242417335510254,
    "right_arm_joint_5": -0.7117632627487183,
    "right_arm_joint_6": 0.14113007485866547,
    "right_arm_joint_7": 0.5137608647346497 + math.pi / 2.0,
    "left_arm_joint_1": -0.4555884897708893,
    "left_arm_joint_2": 0.2500312626361847,
    "left_arm_joint_3": -0.665743887424469,
    "left_arm_joint_4": -1.3314952850341797,
    "left_arm_joint_5": -0.19328542053699493,
    "left_arm_joint_6": -0.5307496786117554,
    "left_arm_joint_7": 0.6565361022949219 - math.pi / 2.0,
    "right_gripper_finger_joint_1": 0.09138019700534642,
    "right_gripper_left_finger_joint": -0.04569009850267321,
    "right_gripper_right_finger_joint": 0.04569009850267321,
    "left_gripper_finger_joint_1": 0.09098683297634125,
    "left_gripper_left_finger_joint": -0.045493416488170624,
    "left_gripper_right_finger_joint": 0.045493416488170624,
}


@wp.func
def _smoothstep(value: float) -> float:
    value = wp.clamp(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


@wp.func
def _rotation_error_angle(target: wp.quat, current: wp.quat) -> float:
    error = wp.normalize(target * wp.quat_inverse(current))
    return 2.0 * wp.acos(wp.clamp(wp.abs(error[3]), 0.0, 1.0))


@wp.kernel
def _update_state_machine(
    body_q: wp.array[wp.transform],
    phase: wp.array[int],
    elapsed: wp.array[float],
    durations: wp.array[float],
    phase_ee: wp.array[wp.transform],
    phase_connector: wp.array[wp.transform],
    frozen_tip_offset: wp.array[wp.vec3],
    phase_entry_tip: wp.array[wp.vec3],
    phase_entry_ee_position: wp.array[wp.vec3],
    phase_entry_connector_in_ee: wp.array[wp.vec3],
    phase_entry_connector_axis: wp.array[wp.vec3],
    phase_exit_time: wp.array[float],
    phase_exit_converged: wp.array[int],
    phase_exit_tip: wp.array[wp.vec3],
    phase_exit_ee_position: wp.array[wp.vec3],
    phase_exit_connector_in_ee: wp.array[wp.vec3],
    phase_exit_connector_axis: wp.array[wp.vec3],
    right_ee_body: int,
    cable_head_body: int,
    ee_local_xform: wp.transform,
    connector_local_xform: wp.transform,
    socket_position: wp.vec3,
    socket_rotation: wp.quat,
    plug_grasp_offset: wp.vec3,
    connector_tip_length: float,
    grasp_orientation_offset: wp.quat,
    static_tip_offset: wp.vec3,
    frame_dt: float,
    target_position: wp.array[wp.vec3],
    target_rotation: wp.array[wp.vec4],
    gripper_blend: wp.array[float],
):
    p = phase[0]
    time = elapsed[0]
    ee_tf = body_q[right_ee_body] * ee_local_xform
    connector_tf = body_q[cable_head_body] * connector_local_xform
    if time == 0.0:
        phase_ee[0] = ee_tf
        phase_connector[0] = connector_tf

    ee_position = wp.transform_get_translation(ee_tf)
    ee_rotation = wp.transform_get_rotation(ee_tf)
    connector_position = wp.transform_get_translation(connector_tf)
    connector_rotation = wp.transform_get_rotation(connector_tf)
    connector_axis = wp.normalize(wp.quat_rotate(connector_rotation, wp.vec3(0.0, 0.0, 1.0)))
    cable_tip_axis = -connector_axis
    connector_tip_position = connector_position + connector_tip_length * connector_axis
    if time == 0.0:
        phase_entry_tip[p] = connector_tip_position
        phase_entry_ee_position[p] = ee_position
        phase_entry_connector_in_ee[p] = wp.quat_rotate(wp.quat_inverse(ee_rotation), connector_position - ee_position)
        phase_entry_connector_axis[p] = connector_axis

    start_tf = phase_ee[0]
    start_position = wp.transform_get_translation(start_tf)
    start_rotation = wp.transform_get_rotation(start_tf)
    entry_connector_tf = phase_connector[0]
    entry_connector_position = wp.transform_get_translation(entry_connector_tf)
    entry_connector_rotation = wp.transform_get_rotation(entry_connector_tf)

    grasp_rotation = wp.normalize(entry_connector_rotation * grasp_orientation_offset)
    grasp_position = entry_connector_position + wp.quat_rotate(entry_connector_rotation, plug_grasp_offset)

    target_pos = start_position
    target_rot = start_rotation
    grip = 0.0

    insertion_axis = wp.normalize(wp.quat_rotate(socket_rotation, wp.vec3(0.0, 0.0, 1.0)))
    socket_grasp_rotation = wp.normalize(socket_rotation * grasp_orientation_offset)
    coax_delta = quat_between_vectors_robust(cable_tip_axis, -insertion_axis)
    coaxial_rotation = wp.normalize(coax_delta * ee_rotation)
    live_tip_offset = wp.quat_rotate(wp.quat_inverse(ee_rotation), connector_tip_position - ee_position)

    if p == INSERT and time == 0.0:
        frozen_tip_offset[0] = live_tip_offset

    tip_offset = static_tip_offset
    if p == LIFT or p == CARRY or p == ALIGN:
        tip_offset = live_tip_offset
    elif p == INSERT or p == HOLD_INSERTED:
        tip_offset = frozen_tip_offset[0]

    preinsert_tip = socket_position - 0.018 * insertion_axis
    inserted_tip = socket_position + 0.006 * insertion_axis
    preinsert_position = preinsert_tip - wp.quat_rotate(socket_grasp_rotation, tip_offset)
    lift_position = wp.vec3(start_position[0], start_position[1], preinsert_position[2])
    coaxial_preinsert_position = preinsert_tip - wp.quat_rotate(coaxial_rotation, tip_offset)
    coaxial_insert_position = inserted_tip - wp.quat_rotate(coaxial_rotation, tip_offset)

    if p == APPROACH:
        target_pos = grasp_position + wp.quat_rotate(entry_connector_rotation, wp.vec3(0.0, 0.08, 0.0))
        target_rot = grasp_rotation
    elif p == ENGAGE:
        target_pos = grasp_position + wp.vec3(0.01, 0.0, 0.0)
        target_rot = grasp_rotation
    elif p == GRASP:
        grip = _smoothstep(time / durations[GRASP])
    elif p == HOLD_GRASP:
        grip = 1.0
    elif p == RETRACT:
        target_pos = start_position + wp.quat_rotate(entry_connector_rotation, wp.vec3(0.0, 0.05, 0.0))
        grip = 1.0
    elif p == SETTLE:
        grip = 1.0
    elif p == LIFT:
        target_pos = lift_position
        target_rot = socket_grasp_rotation
        grip = 1.0
    elif p == CARRY:
        target_pos = preinsert_position
        target_rot = socket_grasp_rotation
        grip = 1.0
    elif p == ALIGN:
        target_pos = coaxial_preinsert_position
        target_rot = coaxial_rotation
        grip = 1.0
    elif p == INSERT:
        target_pos = coaxial_insert_position
        target_rot = coaxial_rotation
        grip = 1.0
    elif p == HOLD_INSERTED:
        target_pos = coaxial_insert_position
        target_rot = coaxial_rotation
        grip = 1.0
    elif p == RELEASE:
        grip = 1.0 - _smoothstep(time / durations[RELEASE])
    elif p == BACKOFF:
        withdraw_axis = wp.quat_rotate(socket_rotation, wp.vec3(0.0, 1.0, 0.0))
        target_pos = coaxial_insert_position + 0.10 * withdraw_axis

    blend = _smoothstep(time / durations[p])
    command_position = (1.0 - blend) * start_position + blend * target_pos
    command_rotation = wp.quat_slerp(start_rotation, target_rot, blend)
    target_position[0] = command_position
    target_rotation[0] = wp.vec4(command_rotation[0], command_rotation[1], command_rotation[2], command_rotation[3])
    gripper_blend[0] = grip

    position_error = target_pos - ee_position
    converged = (
        wp.abs(position_error[0]) < 0.01
        and wp.abs(position_error[1]) < 0.01
        and wp.abs(position_error[2]) < 0.01
        and _rotation_error_angle(target_rot, ee_rotation) < 0.2617994
    )
    if p == ALIGN and wp.dot(connector_axis, insertion_axis) <= 0.9995:
        converged = False

    next_time = time + frame_dt
    minimum_time_met = next_time >= durations[p]
    hard_timeout = next_time >= 2.0 * durations[p]
    if p < DONE and minimum_time_met and (converged or hard_timeout):
        phase_exit_time[p] = next_time
        phase_exit_converged[p] = wp.int32(converged)
        phase_exit_tip[p] = connector_tip_position
        phase_exit_ee_position[p] = ee_position
        phase_exit_connector_in_ee[p] = wp.quat_rotate(wp.quat_inverse(ee_rotation), connector_position - ee_position)
        phase_exit_connector_axis[p] = connector_axis
        phase[0] = p + 1
        elapsed[0] = 0.0
    else:
        elapsed[0] = next_time


@wp.kernel
def _write_robot_targets(
    ik_joint_q: wp.array2d[float],
    gripper_blend: wp.array[float],
    control_joint_target_q: wp.array[float],
    robot_coord_count: int,
    gripper_driver: int,
    gripper_left: int,
    gripper_right: int,
):
    coord = wp.tid()
    if coord >= robot_coord_count:
        return

    value = ik_joint_q[0, coord]
    grip = gripper_blend[0]
    if coord == gripper_driver:
        value = 0.09 + grip * (0.014 - 0.09)
    elif coord == gripper_left:
        value = -0.045 + grip * (-0.007 + 0.045)
    elif coord == gripper_right:
        value = 0.045 + grip * (0.007 - 0.045)
    ik_joint_q[0, coord] = value
    control_joint_target_q[coord] = value


@wp.kernel
def _record_grip_contacts(
    contact_count: wp.array[int],
    shape0: wp.array[int],
    shape1: wp.array[int],
    phase: wp.array[int],
    connector_shape: int,
    finger_shape0: int,
    finger_shape1: int,
    phase_contact_total: wp.array[int],
):
    contact = wp.tid()
    if contact >= contact_count[0]:
        return
    a = shape0[contact]
    b = shape1[contact]
    a_is_finger = a == finger_shape0 or a == finger_shape1
    b_is_finger = b == finger_shape0 or b == finger_shape1
    if (a == connector_shape and b_is_finger) or (b == connector_shape and a_is_finger):
        wp.atomic_add(phase_contact_total, phase[0], 1)


def _connector_metrics(
    tip_position: np.ndarray,
    connector_axis: np.ndarray,
    socket_position: np.ndarray,
    socket_axis: np.ndarray,
) -> tuple[float, float, float]:
    """Return connector axial depth, radial error, and socket-axis alignment."""

    socket_axis = np.asarray(socket_axis, dtype=np.float64)
    connector_axis = np.asarray(connector_axis, dtype=np.float64)
    socket_axis /= np.linalg.norm(socket_axis)
    connector_axis /= np.linalg.norm(connector_axis)
    offset = np.asarray(tip_position, dtype=np.float64) - np.asarray(socket_position, dtype=np.float64)
    depth = float(np.dot(offset, socket_axis))
    radial_error = float(np.linalg.norm(offset - depth * socket_axis))
    alignment = float(np.dot(connector_axis, socket_axis))
    return depth, radial_error, alignment


def _find_suffix(labels: list[str], suffix: str) -> int:
    matches = [index for index, label in enumerate(labels) if str(label).endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one label ending in {suffix!r}, found {len(matches)}")
    return matches[0]


class Example:
    class Phase(IntEnum):
        REST = 0
        APPROACH = 1
        ENGAGE = 2
        GRASP = 3
        HOLD_GRASP = 4
        RETRACT = 5
        SETTLE = 6
        LIFT = 7
        CARRY = 8
        ALIGN = 9
        INSERT = 10
        HOLD_INSERTED = 11
        RELEASE = 12
        BACKOFF = 13
        DONE = 14

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 100.0
        self.sim_substeps = max(1, int(args.substeps))
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.use_graph = bool(args.graph_capture)
        self._build_scene(args)

        self.control = self.model.control()
        self._build_solver(args)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="explicit",
            rigid_contact_max=65536,
            contact_matching="latest",
            contact_matching_pos_threshold=0.005,
            contact_matching_normal_dot_threshold=0.95,
        )
        self.contacts = self.collision_pipeline.contacts()
        self.solver.prepare_contacts(self.contacts)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)

        newton.examples.configure_coupled_view(self, args)
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.set_camera(wp.vec3(-2.55, -7.1, 2.3 + SCENE_VERTICAL_OFFSET), pitch=-12.0, yaw=-295.0)
            if hasattr(self.viewer.camera, "look_at"):
                self.viewer.camera.look_at(wp.vec3(0.55, -0.42, 0.9 + SCENE_VERTICAL_OFFSET))

        self._build_ik()
        self._build_state_machine(args)
        self.capture()

    def _build_scene(self, args):
        builder = newton.ModelBuilder(gravity=-9.81)
        builder.rigid_gap = 0.001
        SolverMuJoCo.register_custom_attributes(builder)
        SolverVBD.register_custom_attributes(builder, dahl_defaults_enabled=False)

        robot = self._add_robot(builder, load_visual_shapes=True)
        self.robot_bodies, self.robot_joints, self.robot_shapes, self.right_finger_bodies = robot
        self._add_cable(builder)
        self._add_fridge(builder)
        self.ground_shape = builder.add_ground_plane(
            height=GROUND_HEIGHT,
            cfg=newton.ModelBuilder.ShapeConfig(
                ke=CONTACT_KE,
                kd=CONTACT_KD,
                mu=CONTACT_MU,
                margin=0.0,
                gap=0.001,
            ),
            label="waterhose_ground",
        )

        builder.color()
        self.model = builder.finalize(device=args.device)
        self.device = self.model.device

    def _build_solver(self, args) -> None:
        def make_vbd(view):
            solver = SolverVBD(
                model=view,
                iterations=int(args.vbd_iterations),
                friction_epsilon=0.1,
                rigid_avbd_beta=1.0e2,
                rigid_avbd_gamma=0.999,
                rigid_contact_hard=True,
                rigid_contact_history=True,
                rigid_contact_k_start=1.0e3,
                rigid_body_contact_buffer_size=4096,
                rigid_joint_linear_ke=1.0e9,
                rigid_joint_angular_ke=1.0e9,
                rigid_joint_linear_k_start=1.0e4,
                rigid_joint_angular_k_start=1.0e1,
            )
            for joint in range(view.joint_count):
                solver.set_joint_constraint_mode(joint, hard=False)
            return solver

        self.solver = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupled.Entry(
                    name="mjc",
                    solver=lambda view: SolverMuJoCo(
                        model=view,
                        solver="newton",
                        integrator="implicitfast",
                        cone="elliptic",
                        iterations=int(args.mujoco_iterations),
                        ls_iterations=20,
                        use_mujoco_contacts=False,
                        njmax=1024,
                        nconmax=4096,
                    ),
                    bodies=self.robot_bodies,
                    joints=self.robot_joints,
                    configure_view=self._configure_mujoco_view,
                ),
                SolverCoupled.Entry(
                    name="vbd",
                    solver=make_vbd,
                    bodies=self.cable_bodies,
                    joints=self.cable_joints,
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="mjc",
                        destination="vbd",
                        bodies=self.right_finger_bodies,
                        mass_scale=1.0,
                        mode="staggered",
                        collision_pipeline=self._make_proxy_collision_pipeline,
                        collide_interval=1,
                    )
                ],
                iterations=1,
            ),
        )
        self.proxy_contacts = self.solver.get_proxy_contacts("mjc", "vbd")
        if self.proxy_contacts is None or len(self.proxy_grip_shapes) != 2:
            raise RuntimeError("Proxy grip contact diagnostics were not initialized")

    def _build_ik(self) -> None:
        ik_builder = newton.ModelBuilder(gravity=-9.81)
        SolverMuJoCo.register_custom_attributes(ik_builder)
        self._add_robot(ik_builder, load_visual_shapes=False)
        self.ik_model = ik_builder.finalize(device=self.device)
        self.robot_coord_count = self.ik_model.joint_coord_count
        if self.robot_coord_count > self.model.joint_coord_count:
            raise RuntimeError("The IK robot coordinates do not match the coupled model prefix")

        ik_state = self.ik_model.state()
        newton.eval_fk(self.ik_model, self.ik_model.joint_q, self.ik_model.joint_qd, ik_state)
        body_q = ik_state.body_q.numpy()
        right_body = _find_suffix(self.ik_model.body_label, "right_gripper_base")
        left_body = _find_suffix(self.ik_model.body_label, "left_gripper_base")
        torso_body = _find_suffix(self.ik_model.body_label, "torso_hip_yaw")

        right_tf = wp.transform(*body_q[right_body]) * RIGHT_EE_TRANSFORM
        left_tf = wp.transform(*body_q[left_body])
        torso_tf = wp.transform(*body_q[torso_body])
        self.ik_target_position = wp.array([wp.transform_get_translation(right_tf)], dtype=wp.vec3, device=self.device)
        right_rotation = wp.transform_get_rotation(right_tf)
        self.ik_target_rotation = wp.array([wp.vec4(*right_rotation)], dtype=wp.vec4, device=self.device)

        def position_target(transform):
            return wp.array([wp.transform_get_translation(transform)], dtype=wp.vec3, device=self.device)

        def rotation_target(transform):
            rotation = wp.transform_get_rotation(transform)
            return wp.array([wp.vec4(*rotation)], dtype=wp.vec4, device=self.device)

        objectives = [
            ik.IKObjectivePosition(
                link_index=right_body,
                link_offset=wp.transform_get_translation(RIGHT_EE_TRANSFORM),
                target_positions=self.ik_target_position,
            ),
            ik.IKObjectiveRotation(
                link_index=right_body,
                link_offset_rotation=wp.transform_get_rotation(RIGHT_EE_TRANSFORM),
                target_rotations=self.ik_target_rotation,
            ),
            ik.IKObjectivePosition(
                link_index=left_body,
                link_offset=wp.vec3(),
                target_positions=position_target(left_tf),
            ),
            ik.IKObjectiveRotation(
                link_index=left_body,
                link_offset_rotation=wp.quat_identity(),
                target_rotations=rotation_target(left_tf),
            ),
            ik.IKObjectivePosition(
                link_index=torso_body,
                link_offset=wp.vec3(),
                target_positions=position_target(torso_tf),
                weight=50.0,
            ),
            ik.IKObjectiveRotation(
                link_index=torso_body,
                link_offset_rotation=wp.quat_identity(),
                target_rotations=rotation_target(torso_tf),
                weight=50.0,
            ),
            ik.IKObjectiveJointLimit(
                joint_limit_lower=self.ik_model.joint_limit_lower,
                joint_limit_upper=self.ik_model.joint_limit_upper,
                weight=0.1,
            ),
        ]
        self.ik_solver = ik.IKSolver(
            model=self.ik_model,
            n_problems=1,
            objectives=objectives,
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self.ik_iterations = 24
        self.ik_joint_q = wp.clone(
            self.model.joint_q.reshape((1, self.model.joint_coord_count))[:, : self.robot_coord_count]
        )

        joint_q_start = self.ik_model.joint_q_start.numpy()
        self.gripper_coord_indices = tuple(
            int(joint_q_start[_find_suffix(self.ik_model.joint_label, name)])
            for name in (
                "right_gripper_finger_joint_1",
                "right_gripper_left_finger_joint",
                "right_gripper_right_finger_joint",
            )
        )
        self.right_ee_body = _find_suffix(self.model.body_label, "right_gripper_base")

    def _build_state_machine(self, args) -> None:
        durations = [
            max(float(args.settle_time), self.frame_dt),
            3.0,
            1.5,
            0.5,
            0.5,
            1.5,
            0.3,
            3.0,
            2.0,
            2.0,
            4.0,
            1.0,
            0.8,
            1.5,
            1.0e6,
        ]
        self.phase = wp.zeros(1, dtype=int, device=self.device)
        self.phase_elapsed = wp.zeros(1, dtype=float, device=self.device)
        self.phase_durations = wp.array(durations, dtype=float, device=self.device)
        self.phase_ee = wp.array([wp.transform()], dtype=wp.transform, device=self.device)
        self.phase_connector = wp.array([wp.transform()], dtype=wp.transform, device=self.device)
        self.frozen_tip_offset = wp.zeros(1, dtype=wp.vec3, device=self.device)
        self.gripper_blend = wp.zeros(1, dtype=float, device=self.device)
        self.phase_entry_tip = wp.zeros(DONE + 1, dtype=wp.vec3, device=self.device)
        self.phase_entry_ee_position = wp.zeros(DONE + 1, dtype=wp.vec3, device=self.device)
        self.phase_entry_connector_in_ee = wp.zeros(DONE + 1, dtype=wp.vec3, device=self.device)
        self.phase_entry_connector_axis = wp.zeros(DONE + 1, dtype=wp.vec3, device=self.device)
        self.phase_exit_time = wp.zeros(DONE + 1, dtype=float, device=self.device)
        self.phase_exit_converged = wp.zeros(DONE + 1, dtype=int, device=self.device)
        self.phase_exit_tip = wp.zeros(DONE + 1, dtype=wp.vec3, device=self.device)
        self.phase_exit_ee_position = wp.zeros(DONE + 1, dtype=wp.vec3, device=self.device)
        self.phase_exit_connector_in_ee = wp.zeros(DONE + 1, dtype=wp.vec3, device=self.device)
        self.phase_exit_connector_axis = wp.zeros(DONE + 1, dtype=wp.vec3, device=self.device)
        self.phase_grip_contact_total = wp.zeros(DONE + 1, dtype=int, device=self.device)

        q_rz = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -0.5 * math.pi)
        q_rx = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * math.pi)
        self.grasp_orientation_offset = wp.normalize(q_rx * q_rz)
        ideal_tip_from_grasp = wp.vec3(0.0, 0.0, CONNECTOR_TIP_LENGTH) - PLUG_GRASP_OFFSET
        self.static_tip_offset = wp.quat_rotate(wp.quat_inverse(self.grasp_orientation_offset), ideal_tip_from_grasp)

    def capture(self) -> None:
        self.graph = None
        if self.use_graph and self.device.is_cuda:
            if self.sim_substeps % 2 != 0:
                raise ValueError("CUDA graph capture requires an even --substeps value")
            with wp.ScopedDevice(self.device), wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def _configure_mujoco_view(self, view) -> None:
        flags = view.shape_flags.numpy().copy()
        collide_mask = int(newton.ShapeFlags.COLLIDE_SHAPES | newton.ShapeFlags.COLLIDE_PARTICLES)
        flags[self.socket_shape] &= ~collide_mask
        flags[self.ground_shape] &= ~collide_mask
        view.shape_flags = wp.array(flags, dtype=wp.int32, device=view.device)

    def _make_proxy_collision_pipeline(self, view):
        shape_body = view.shape_body.numpy()
        finger_bodies = {
            body
            for body, label in enumerate(view.body_label)
            if any(str(label).endswith(suffix) for suffix in _RIGHT_FINGER_BODY_SUFFIXES)
        }
        if len(finger_bodies) != 2:
            raise RuntimeError(f"Expected two proxy finger bodies, found {len(finger_bodies)}")

        finger_shapes = {shape for shape, body in enumerate(shape_body) if int(body) in finger_bodies}
        connector_shapes = {
            shape for shape, label in enumerate(view.shape_label) if str(label).endswith("waterhose_connector")
        }
        if not finger_shapes or connector_shapes != {self.connector_shape}:
            raise RuntimeError("Failed to resolve proxy finger and connector shapes")

        flags = view.shape_flags.numpy().copy()
        particle_bit = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        for shape in finger_shapes:
            flags[shape] &= ~particle_bit
        view.shape_flags = wp.array(flags, dtype=wp.int32, device=view.device)

        pairs = []
        grip_shapes = set()
        for raw_shape_a, raw_shape_b in view.shape_contact_pairs.numpy().reshape((-1, 2)):
            shape_a, shape_b = int(raw_shape_a), int(raw_shape_b)
            a_is_finger = shape_a in finger_shapes
            b_is_finger = shape_b in finger_shapes
            if a_is_finger or b_is_finger:
                other = shape_b if a_is_finger else shape_a
                if a_is_finger == b_is_finger or other not in connector_shapes:
                    continue
                grip_shapes.add(shape_a if a_is_finger else shape_b)
            pairs.append((shape_a, shape_b))

        self.proxy_grip_shapes = sorted(grip_shapes)

        return newton.CollisionPipeline(
            view,
            broad_phase="explicit",
            shape_pairs_filtered=wp.array(
                np.asarray(pairs, dtype=np.int32).reshape((-1, 2)),
                dtype=wp.vec2i,
                device=view.device,
            ),
            rigid_contact_max=30000,
            contact_matching="latest",
            contact_matching_pos_threshold=0.005,
            contact_matching_normal_dot_threshold=0.95,
        )

    def _add_robot(
        self, builder: newton.ModelBuilder, *, load_visual_shapes: bool
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        body_start = builder.body_count
        joint_start = builder.joint_count
        shape_start = builder.shape_count
        builder.add_usd(
            str(ROBOT_USD),
            xform=ROBOT_TRANSFORM,
            override_root_xform=True,
            floating=False,
            enable_self_collisions=False,
            load_visual_shapes=load_visual_shapes,
            hide_collision_shapes=True,
            parse_mujoco_options=False,
        )

        robot_bodies = list(range(body_start, builder.body_count))
        robot_joints = list(range(joint_start, builder.joint_count))
        robot_shapes = list(range(shape_start, builder.shape_count))
        right_finger_bodies = [
            body
            for body in robot_bodies
            if any(str(builder.body_label[body]).endswith(suffix) for suffix in _RIGHT_FINGER_BODY_SUFFIXES)
        ]
        if len(right_finger_bodies) != 2:
            raise RuntimeError(f"Expected two right-gripper finger bodies, found {len(right_finger_bodies)}")

        joint_by_name = {str(builder.joint_label[joint]).rsplit("/", 1)[-1]: joint for joint in robot_joints}
        for name, value in _RBY1_INITIAL_Q.items():
            joint = joint_by_name.get(name)
            if joint is None:
                raise RuntimeError(f"RBY1 joint {name!r} is missing")
            q_start = builder.joint_q_start[joint]
            builder.joint_q[q_start] = value
            builder.joint_target_q[q_start] = value

        for joint in robot_joints:
            label = str(builder.joint_label[joint]).rsplit("/", 1)[-1]
            dof_start = builder.joint_qd_start[joint]
            dof_end = builder.joint_qd_start[joint + 1] if joint + 1 < builder.joint_count else builder.joint_dof_count
            for dof in range(dof_start, dof_end):
                if "gripper_finger_joint_1" in label:
                    ke, kd, effort, armature = 1.0e4, 1.0e3, 150.0, 0.5
                elif "gripper_" in label and "finger_joint" in label:
                    ke, kd, effort, armature = 4.0e4, 2.0e3, 120.0, 0.5
                else:
                    ke, kd, effort, armature = 1.2e5, 1.2e4, 1.0e4, 0.2
                builder.joint_target_ke[dof] = ke
                builder.joint_target_kd[dof] = kd
                builder.joint_effort_limit[dof] = effort
                builder.joint_armature[dof] = armature

        for index, label in enumerate(builder.constraint_mimic_label):
            if "gripper_" in str(label):
                builder.constraint_mimic_enabled[index] = False

        gravcomp = builder.custom_attributes["mujoco:gravcomp"]
        gravcomp.values = {} if gravcomp.values is None else gravcomp.values
        for body in robot_bodies:
            gravcomp.values[body] = 1.0
        joint_gravcomp = builder.custom_attributes["mujoco:jnt_actgravcomp"]
        joint_gravcomp.values = {} if joint_gravcomp.values is None else joint_gravcomp.values
        for dof in range(builder.joint_qd_start[joint_start], builder.joint_dof_count):
            joint_gravcomp.values[dof] = True

        collision_bit = int(newton.ShapeFlags.COLLIDE_SHAPES)
        finger_set = set(right_finger_bodies)
        for shape in robot_shapes:
            if not builder.shape_flags[shape] & collision_bit:
                continue
            builder.shape_material_ke[shape] = CONTACT_KE
            builder.shape_material_kd[shape] = CONTACT_KD
            builder.shape_material_mu[shape] = 5.0 if builder.shape_body[shape] in finger_set else CONTACT_MU
            builder.shape_margin[shape] = 0.001

        return robot_bodies, robot_joints, robot_shapes, right_finger_bodies

    def _add_cable(self, builder: newton.ModelBuilder) -> None:
        stage = Usd.Stage.Open(str(CABLE_USD))
        if stage is None:
            raise RuntimeError(f"Failed to open cable asset: {CABLE_USD}")
        curve = UsdGeom.BasisCurves(stage.GetPrimAtPath("/cable001/curve_0"))
        points_attr = curve.GetPointsAttr().Get()
        widths = curve.GetWidthsAttr().Get()
        if points_attr is None or len(points_attr) < 3 or widths is None or len(widths) == 0:
            raise RuntimeError("Waterhose cable asset has no usable centerline or width")

        points = [wp.vec3(float(p[0]), float(p[1]), float(p[2])) + FRIDGE_POSITION for p in points_attr]
        radius = 0.5 * float(widths[0])
        if not math.isclose(radius, CABLE_RADIUS, rel_tol=0.0, abs_tol=1.0e-6):
            raise RuntimeError(f"Expected a {CABLE_RADIUS:g} m cable radius, found {radius:g} m")

        body_start = builder.body_count
        joint_start = builder.joint_count
        shape_start = builder.shape_count
        cable_cfg = newton.ModelBuilder.ShapeConfig(
            density=100.0,
            ke=CONTACT_KE,
            kd=CONTACT_KD,
            mu=CONTACT_MU,
            margin=0.0,
            # Match the restored IsaacLab cable setup's legacy broad contact window.
            gap=0.1,
            collision_group=-1,
        )
        self.cable_bodies, cable_joints = builder.add_rod(
            positions=points,
            quaternions=newton.utils.create_parallel_transport_cable_quaternions(points),
            radius=radius,
            cfg=cable_cfg,
            stretch_stiffness=1.0e6,
            stretch_damping=1.0e-2,
            bend_stiffness=3.0e-1,
            bend_damping=2.0e-2,
            label="waterhose",
            body_frame_origin="start",
            color=wp.vec3(0.08, 0.08, 0.08),
        )

        self.anchor_body = builder.add_body(
            xform=wp.transform(points[-2], wp.quat_identity()), mass=0.0, label="anchor"
        )
        anchor_joint = builder.add_joint_fixed(
            parent=self.anchor_body,
            child=self.cable_bodies[-1],
            parent_xform=wp.transform(),
            child_xform=wp.transform(),
            label="waterhose_anchor_joint",
        )
        self.cable_joints = [*cable_joints, anchor_joint]
        self.cable_bodies.append(self.anchor_body)

        plug_builder = newton.ModelBuilder()
        plug_result = plug_builder.add_usd(
            str(PLUG_USD),
            floating=False,
            load_visual_shapes=True,
            hide_collision_shapes=False,
            parse_mujoco_options=False,
        )
        plug_shapes = list(plug_result["path_shape_map"].values())
        if plug_builder.body_count != 1 or len(plug_shapes) != 1:
            raise RuntimeError("Expected the connector asset to contain one body and one shape")

        head_body = self.cable_bodies[0]
        self.connector_local_xform = wp.transform_multiply(
            wp.transform_inverse(builder.body_q[head_body]),
            PLUG_TRANSFORM,
        )
        plug_shape = int(plug_shapes[0])
        plug_mesh = plug_builder.shape_source[plug_shape]
        plug_density = plug_builder.body_mass[0] / plug_mesh.mass
        self.connector_shape = builder.add_shape_mesh(
            body=head_body,
            xform=self.connector_local_xform,
            mesh=plug_mesh,
            scale=plug_builder.shape_scale[plug_shape],
            cfg=newton.ModelBuilder.ShapeConfig(
                density=plug_density,
                ke=CONTACT_KE,
                kd=CONTACT_KD,
                mu=CONTACT_MU,
                margin=0.0,
                gap=0.01,
                collision_group=-1,
            ),
            color=plug_builder.shape_color[plug_shape],
            label="waterhose_connector",
        )
        self.cable_shapes = list(range(shape_start, builder.shape_count))
        if self.cable_bodies[0] != body_start or self.cable_joints[0] != joint_start:
            raise RuntimeError("Unexpected cable builder layout")

    def _add_fridge(self, builder: newton.ModelBuilder) -> None:
        result = builder.add_usd(
            str(FRIDGE_USD),
            xform=wp.transform(FRIDGE_POSITION, wp.quat_identity()),
            floating=False,
            root_path="/root",
            load_sites=False,
            load_visual_shapes=True,
            hide_collision_shapes=False,
            parse_mujoco_options=False,
            only_load_enabled_rigid_bodies=False,
        )
        shape_map = result["path_shape_map"]
        self.socket_shape = shape_map.get("/root/Cable008/SocketCollision/Cable008_SocketCollision")
        self.housing_shape = shape_map.get("/root/Cable008/BodyCollision/Cable008_BodyCollision")
        if self.socket_shape is None or self.housing_shape is None:
            raise RuntimeError("Failed to resolve the refrigerator housing and socket colliders")

    def simulate(self):
        wp.launch(
            _update_state_machine,
            dim=1,
            inputs=[
                self.state_0.body_q,
                self.phase,
                self.phase_elapsed,
                self.phase_durations,
                self.phase_ee,
                self.phase_connector,
                self.frozen_tip_offset,
                self.phase_entry_tip,
                self.phase_entry_ee_position,
                self.phase_entry_connector_in_ee,
                self.phase_entry_connector_axis,
                self.phase_exit_time,
                self.phase_exit_converged,
                self.phase_exit_tip,
                self.phase_exit_ee_position,
                self.phase_exit_connector_in_ee,
                self.phase_exit_connector_axis,
                self.right_ee_body,
                self.cable_bodies[0],
                RIGHT_EE_TRANSFORM,
                self.connector_local_xform,
                SOCKET_POSITION,
                SOCKET_ROTATION,
                PLUG_GRASP_OFFSET,
                CONNECTOR_TIP_LENGTH,
                self.grasp_orientation_offset,
                self.static_tip_offset,
                self.frame_dt,
                self.ik_target_position,
                self.ik_target_rotation,
                self.gripper_blend,
            ],
            device=self.device,
        )
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iterations)
        wp.launch(
            _write_robot_targets,
            dim=self.robot_coord_count,
            inputs=[
                self.ik_joint_q,
                self.gripper_blend,
                self.control.joint_target_q,
                self.robot_coord_count,
                *self.gripper_coord_indices,
            ],
            device=self.device,
        )
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            newton.examples.apply_coupled_viewer_forces(self, self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            newton.eval_ik(self.model, self.state_1, self.state_1.joint_q, self.state_1.joint_qd)
            self.state_0, self.state_1 = self.state_1, self.state_0
        wp.launch(
            _record_grip_contacts,
            dim=min(512, self.proxy_contacts.rigid_contact_max),
            inputs=[
                self.proxy_contacts.rigid_contact_count,
                self.proxy_contacts.rigid_contact_shape0,
                self.proxy_contacts.rigid_contact_shape1,
                self.phase,
                self.connector_shape,
                *self.proxy_grip_shapes,
                self.phase_grip_contact_total,
            ],
            device=self.device,
        )

    def step(self):
        if self.graph is None:
            self.simulate()
        else:
            wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        newton.examples.log_coupled_view(self, self.contacts)
        self.viewer.end_frame()

    def _phase_diagnostics(self) -> str:
        tips = self.phase_entry_tip.numpy()
        ee_positions = self.phase_entry_ee_position.numpy()
        connector_in_ee = self.phase_entry_connector_in_ee.numpy()
        entry_axes = self.phase_entry_connector_axis.numpy()
        exit_times = self.phase_exit_time.numpy()
        exit_converged = self.phase_exit_converged.numpy()
        exit_tips = self.phase_exit_tip.numpy()
        exit_ee_positions = self.phase_exit_ee_position.numpy()
        exit_connector_in_ee = self.phase_exit_connector_in_ee.numpy()
        exit_axes = self.phase_exit_connector_axis.numpy()
        grip_contacts = self.phase_grip_contact_total.numpy()
        current_phase = int(self.phase.numpy()[0])
        lines = []
        for phase in range(current_phase + 1):
            lines.append(
                f"{self.Phase(phase).name}: tip={np.round(tips[phase], 4).tolist()} "
                f"ee={np.round(ee_positions[phase], 4).tolist()} "
                f"connector_in_ee={np.round(connector_in_ee[phase], 4).tolist()} "
                f"axis={np.round(entry_axes[phase], 3).tolist()} "
                f"exit={exit_times[phase]:.2f}s converged={bool(exit_converged[phase])} "
                f"exit_tip={np.round(exit_tips[phase], 4).tolist()} "
                f"exit_ee={np.round(exit_ee_positions[phase], 4).tolist()} "
                f"exit_connector_in_ee={np.round(exit_connector_in_ee[phase], 4).tolist()} "
                f"exit_axis={np.round(exit_axes[phase], 3).tolist()} contacts={grip_contacts[phase]}"
            )
        return "\n".join(lines)

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        assert np.all(np.isfinite(body_q)), "Body poses contain NaN or inf values"
        assert np.all(np.isfinite(self.state_0.body_qd.numpy())), "Body velocities contain NaN or inf values"

        connector_tf = wp.transform(*body_q[self.cable_bodies[0]]) * self.connector_local_xform
        connector_position = np.asarray(wp.transform_get_translation(connector_tf), dtype=np.float64)
        connector_rotation = wp.transform_get_rotation(connector_tf)
        connector_axis = np.asarray(wp.quat_rotate(connector_rotation, wp.vec3(0.0, 0.0, 1.0)), dtype=np.float64)
        tip_position = connector_position + CONNECTOR_TIP_LENGTH * connector_axis
        socket_axis = np.asarray(wp.quat_rotate(SOCKET_ROTATION, wp.vec3(0.0, 0.0, 1.0)), dtype=np.float64)
        depth, radial_error, alignment = _connector_metrics(
            tip_position,
            connector_axis,
            np.asarray(SOCKET_POSITION, dtype=np.float64),
            socket_axis,
        )

        phase = int(self.phase.numpy()[0])
        diagnostics = self._phase_diagnostics()
        assert phase == DONE, f"State machine stopped in {self.Phase(phase).name}\n{diagnostics}"
        assert -0.001 <= depth <= 0.010, f"Connector depth {depth * 1000.0:.1f} mm is not seated\n{diagnostics}"
        assert radial_error < 0.012, (
            f"Connector radial error {radial_error * 1000.0:.1f} mm is too large\n{diagnostics}"
        )
        assert alignment > 0.75, f"Connector alignment cosine {alignment:.3f} is too low\n{diagnostics}"

        ee_tf = wp.transform(*body_q[self.right_ee_body]) * RIGHT_EE_TRANSFORM
        ee_position = np.asarray(wp.transform_get_translation(ee_tf), dtype=np.float64)
        socket_distance = float(np.linalg.norm(ee_position - np.asarray(SOCKET_POSITION, dtype=np.float64)))
        assert socket_distance > 0.08, (
            f"Released gripper backed off only {socket_distance * 1000.0:.1f} mm\n{diagnostics}"
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_coupled_view_args(parser)
        parser.add_argument("--substeps", type=int, default=10, help="Coupled substeps per rendered frame.")
        parser.add_argument("--vbd-iterations", type=int, default=20, help="VBD iterations per coupled substep.")
        parser.add_argument("--mujoco-iterations", type=int, default=100, help="MuJoCo solver iterations.")
        parser.add_argument("--settle-time", type=float, default=2.0, help="Initial hose settling time [s].")
        parser.add_argument(
            "--no-graph-capture",
            action="store_false",
            dest="graph_capture",
            default=True,
            help="Disable CUDA graph capture.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    parser.set_defaults(num_frames=4500)
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
