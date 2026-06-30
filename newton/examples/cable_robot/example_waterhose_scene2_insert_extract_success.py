# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example: Cable Robot Water Hose Insert + Pull (v2)
#
# Based on insert_pull. After inserting and snapping the cable, the
# robot releases, withdraws (backing away only), re-approaches,
# re-grasps, and pulls in the negative insertion direction to test
# the hold strength, then releases.
#
# Command:
#   uv run newton/examples/cable_robot/example_waterhose_scene2_insert_extract_success.py
#   uv run newton/examples/cable_robot/example_waterhose_scene2_insert_extract_success.py --primary-view mujoco
#
###########################################################################

import enum
import os
import sys

os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")

import mujoco_warp as _mjw
import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
from newton import GeoType
from newton.examples.cable_robot.usd_cable_curve_import import add_cable_from_usd_curve

# ---------------------------------------------------------------------------
# Warp kernels - coupling
# ---------------------------------------------------------------------------


@wp.func
def _quat_velocity(q_now: wp.quat, q_prev: wp.quat, dt: float) -> wp.vec3:
    """Angular velocity from successive quaternions (world frame)."""
    q1 = wp.normalize(q_now)
    q0 = wp.normalize(q_prev)
    if wp.dot(q1, q0) < 0.0:
        q0 = wp.quat(-q0[0], -q0[1], -q0[2], -q0[3])
    dq = wp.normalize(wp.mul(q1, wp.quat_inverse(q0)))
    axis, angle = wp.quat_to_axis_angle(dq)
    return axis * (angle / dt)


@wp.kernel(enable_backward=False)
def sync_proxy_states_kernel(
    mj_body_q: wp.array[wp.transform],
    mj_body_qd: wp.array[wp.spatial_vector],
    mj_to_vbd_map: wp.array[int],
    vbd_body_q: wp.array[wp.transform],
    vbd_body_qd: wp.array[wp.spatial_vector],
):
    """Copy MuJoCo body states to VBD proxy bodies."""
    mj_body_id = wp.tid()
    vbd_body_id = mj_to_vbd_map[mj_body_id]
    if vbd_body_id >= 0:
        vbd_body_q[vbd_body_id] = mj_body_q[mj_body_id]
        vbd_body_qd[vbd_body_id] = mj_body_qd[mj_body_id]


@wp.kernel(enable_backward=False)
def smooth_proxy_teleportation_kernel(
    dt: float,
    proxy_vbd_body_ids: wp.array[int],
    vbd_body_q: wp.array[wp.transform],
    vbd_body_qd: wp.array[wp.spatial_vector],
    vbd_solver_body_q_prev: wp.array[wp.transform],
):
    """Encode pose jump as velocity correction, reset body_q to body_q_prev."""
    i = wp.tid()
    if i >= proxy_vbd_body_ids.shape[0]:
        return
    b = proxy_vbd_body_ids[i]
    q_teleported = vbd_body_q[b]
    q_prev = vbd_solver_body_q_prev[b]
    p_teleported = wp.transform_get_translation(q_teleported)
    p_prev = wp.transform_get_translation(q_prev)
    dv = (p_teleported - p_prev) / dt
    r_teleported = wp.transform_get_rotation(q_teleported)
    r_prev = wp.transform_get_rotation(q_prev)
    dw = _quat_velocity(r_teleported, r_prev, dt)
    qd = vbd_body_qd[b]
    vbd_body_qd[b] = qd + wp.spatial_vector(dv, dw)
    vbd_body_q[b] = q_prev


@wp.kernel(enable_backward=False)
def subtract_proxy_forces_kernel(
    dt: float,
    gravity: wp.array[wp.vec3],
    body_world: wp.array[wp.int32],
    vbd_body_q: wp.array[wp.transform],
    proxy_forces: wp.array[wp.spatial_vector],
    proxy_mj_body_ids: wp.array[int],
    proxy_vbd_body_ids: wp.array[int],
    vbd_body_inv_mass: wp.array[float],
    vbd_body_inv_inertia: wp.array[wp.mat33],
    vbd_body_qd: wp.array[wp.spatial_vector],
):
    """Subtract previously applied coupling forces and gravity from VBD proxy velocities."""
    proxy_idx = wp.tid()
    if proxy_idx >= proxy_mj_body_ids.shape[0]:
        return
    mj_body_id = proxy_mj_body_ids[proxy_idx]
    vbd_body_id = proxy_vbd_body_ids[proxy_idx]
    f = proxy_forces[mj_body_id]
    inv_m = vbd_body_inv_mass[vbd_body_id]
    r = wp.transform_get_rotation(vbd_body_q[vbd_body_id])
    inv_I = vbd_body_inv_inertia[vbd_body_id]
    delta_v = dt * inv_m * wp.spatial_top(f)
    delta_w = dt * wp.quat_rotate(r, inv_I * wp.quat_rotate_inv(r, wp.spatial_bottom(f)))
    world_idx = body_world[vbd_body_id]
    g = gravity[wp.max(world_idx, 0)]
    delta_v_grav = dt * g
    vbd_body_qd[vbd_body_id] = vbd_body_qd[vbd_body_id] - wp.spatial_vector(delta_v + delta_v_grav, delta_w)


@wp.kernel(enable_backward=False)
def harvest_proxy_wrenches_kernel(
    rigid_contact_count: wp.array[int],
    contact_body0: wp.array[wp.int32],
    contact_body1: wp.array[wp.int32],
    contact_point0_world: wp.array[wp.vec3],
    contact_point1_world: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_force_on_body1: wp.array[wp.vec3],
    vbd_body_inv_mass: wp.array[float],
    vbd_to_mj_body_map: wp.array[int],
    mj_body_com: wp.array[wp.vec3],
    mj_body_q: wp.array[wp.transform],
    out_mj_body_f: wp.array[wp.spatial_vector],
):
    """Harvest coupling wrenches from VBD contacts onto MuJoCo bodies."""
    contact_id = wp.tid()
    if contact_id >= rigid_contact_count[0]:
        return
    body0 = int(contact_body0[contact_id])
    body1 = int(contact_body1[contact_id])
    if body0 < 0 or body1 < 0:
        return
    mj_body_id0 = vbd_to_mj_body_map[body0] if body0 < vbd_to_mj_body_map.shape[0] else -1
    mj_body_id1 = vbd_to_mj_body_map[body1] if body1 < vbd_to_mj_body_map.shape[0] else -1
    is_proxy0 = int(mj_body_id0 >= 0)
    is_proxy1 = int(mj_body_id1 >= 0)
    if (is_proxy0 + is_proxy1) != 1:
        return
    other_body_id = body1 if is_proxy0 == 1 else body0
    if vbd_body_inv_mass[other_body_id] <= 0.0:
        return
    force_on_body1_world = contact_force_on_body1[contact_id]
    if is_proxy1 == 1:
        mj_body_id = mj_body_id1
        contact_point_world = contact_point1_world[contact_id]
        force_on_proxy_world = force_on_body1_world
    else:
        mj_body_id = mj_body_id0
        contact_point_world = contact_point0_world[contact_id]
        force_on_proxy_world = -force_on_body1_world
    if mj_body_id < 0 or mj_body_id >= out_mj_body_f.shape[0]:
        return
    # Drop the tangential friction component. MuJoCo handles proxy friction for
    # these gripper geoms, so feeding VBD friction back double-counts it.
    n = contact_normal[contact_id]
    if wp.length(n) > 1.0e-8:
        force_on_proxy_world = wp.dot(force_on_proxy_world, n) * n
    com_world = wp.transform_point(mj_body_q[mj_body_id], mj_body_com[mj_body_id])
    torque_world = wp.cross(contact_point_world - com_world, force_on_proxy_world)
    wp.atomic_add(out_mj_body_f, mj_body_id, wp.spatial_vector(force_on_proxy_world, torque_world))


# ---------------------------------------------------------------------------
# Warp kernels - IK / state machine
# ---------------------------------------------------------------------------


@wp.kernel
def merge_ik_with_gripper_targets(
    ik_solution: wp.array[wp.float32],
    gripper_targets: wp.array[wp.float32],
    gripper_mask: wp.array[wp.int32],
    dof_count: int,
    output: wp.array[wp.float32],
):
    """Merge IK solution with gripper targets based on mask.

    For each DOF:
    - If gripper_mask[i] >= 0, use gripper_targets[gripper_mask[i]]
    - Otherwise, use ik_solution[i]
    """
    i = wp.tid()
    if i >= dof_count:
        return

    mask_val = gripper_mask[i]
    if mask_val >= 0:
        output[i] = gripper_targets[mask_val]
    else:
        output[i] = ik_solution[i]


@wp.kernel(enable_backward=False)
def set_target_pose_kernel(
    task_schedule: wp.array[wp.int32],
    task_time_soft_limits: wp.array[float],
    task_idx: wp.array[int],
    task_time_elapsed: wp.array[float],
    task_dt: float,
    # Geometry params
    plug_grasp_offset: wp.vec3,
    approach_offset: wp.vec3,
    engage_offset: wp.vec3,
    retract_vector: wp.vec3,
    insert_offset: wp.vec3,
    withdraw_offset: wp.vec3,
    socket_pos: wp.vec3,
    socket_rot: wp.quat,
    grasp_orientation_offset: wp.quat,
    gripper_open_value: float,
    gripper_closed_value: float,
    # Snapshots
    task_ee_init_body_q: wp.array[wp.transform],
    task_plug_body_q_prev: wp.array[wp.transform],
    # Outputs
    ee_pos_target: wp.array[wp.vec3],
    ee_pos_interp: wp.array[wp.vec3],
    ee_rot_target: wp.array[wp.vec4],
    ee_rot_interp: wp.array[wp.vec4],
    gripper_target: wp.array[wp.float32],
):
    """Compute EE target pose and gripper target for the right arm."""
    arm_idx = wp.tid()

    idx = task_idx[arm_idx]
    task = task_schedule[idx]
    time_limit = task_time_soft_limits[idx]

    task_time_elapsed[arm_idx] += task_dt

    t_lin = wp.min(1.0, task_time_elapsed[arm_idx] / time_limit)
    t = t_lin * t_lin * (3.0 - 2.0 * t_lin)

    ee_pos_prev = wp.transform_get_translation(task_ee_init_body_q[arm_idx])
    ee_quat_prev = wp.transform_get_rotation(task_ee_init_body_q[arm_idx])

    plug_quat_prev = wp.transform_get_rotation(task_plug_body_q_prev[arm_idx])
    plug_pos = wp.transform_get_translation(task_plug_body_q_prev[arm_idx])

    ee_quat_target = ee_quat_prev
    t_gripper = 0.0

    if task == TaskType.APPROACH.value:
        grasp_pos_world = wp.quat_rotate(plug_quat_prev, plug_grasp_offset)
        approach_world = wp.quat_rotate(plug_quat_prev, approach_offset)
        ee_pos_target[arm_idx] = plug_pos + grasp_pos_world + approach_world
        ee_quat_target = plug_quat_prev * grasp_orientation_offset
    elif task == TaskType.ENGAGE.value:
        grasp_pos_world = wp.quat_rotate(plug_quat_prev, plug_grasp_offset)
        ee_pos_target[arm_idx] = plug_pos + grasp_pos_world + engage_offset
        ee_quat_target = plug_quat_prev * grasp_orientation_offset
    elif task == TaskType.GRASP.value:
        ee_pos_target[arm_idx] = ee_pos_prev
        ee_quat_target = ee_quat_prev
        t_gripper = t
    elif task == TaskType.HOLD_GRASP.value:
        ee_pos_target[arm_idx] = ee_pos_prev
        ee_quat_target = ee_quat_prev
        t_gripper = 1.0
    elif task == TaskType.RETRACT.value:
        retract_world = wp.quat_rotate(plug_quat_prev, retract_vector)
        ee_pos_target[arm_idx] = ee_pos_prev + retract_world
        ee_quat_target = ee_quat_prev
        t_gripper = 1.0
    elif task == TaskType.APPROACH_TARGET.value:
        ee_pos_target[arm_idx] = socket_pos
        ee_quat_target = socket_rot * grasp_orientation_offset
        t_gripper = 1.0
    elif task == TaskType.ALIGN_AXES.value:
        ee_pos_target[arm_idx] = ee_pos_prev
        ee_quat_target = ee_quat_prev
        t_gripper = 1.0
    elif task == TaskType.INSERT.value:
        insert_world = wp.quat_rotate(socket_rot, insert_offset)
        ee_pos_target[arm_idx] = socket_pos + insert_world
        ee_quat_target = ee_quat_prev
        t_gripper = 1.0
    elif task == TaskType.RELEASE.value:
        ee_pos_target[arm_idx] = ee_pos_prev
        ee_quat_target = ee_quat_prev
        t_gripper = 1.0 - t
    elif task == TaskType.WITHDRAW.value:
        ee_pos_target[arm_idx] = ee_pos_prev + withdraw_offset
        ee_quat_target = ee_quat_prev
        t_gripper = 0.0
    elif task == TaskType.DONE.value:
        ee_pos_target[arm_idx] = ee_pos_prev
        ee_quat_target = ee_quat_prev
        t_gripper = 0.0
    else:
        ee_pos_target[arm_idx] = ee_pos_prev
        t_gripper = 1.0

    ee_pos_interp[arm_idx] = ee_pos_prev * (1.0 - t) + ee_pos_target[arm_idx] * t
    ee_quat_interpolated = wp.quat_slerp(ee_quat_prev, ee_quat_target, t)

    ee_rot_target[arm_idx] = ee_quat_target[:4]
    ee_rot_interp[arm_idx] = ee_quat_interpolated[:4]

    gripper_target[arm_idx] = gripper_open_value + (gripper_closed_value - gripper_open_value) * t_gripper


@wp.kernel(enable_backward=False)
def advance_task_kernel(
    task_time_soft_limits: wp.array[float],
    ee_pos_target: wp.array[wp.vec3],
    ee_rot_target: wp.array[wp.vec4],
    body_q: wp.array[wp.transform],
    ee_body_idx: int,
    pos_error_thresholds: wp.array[wp.vec3],
    rot_error_thresholds: wp.array[float],
    # Outputs
    task_idx: wp.array[int],
    task_time_elapsed: wp.array[float],
    task_ee_init_body_q: wp.array[wp.transform],
):
    """Check convergence and advance to the next task when ready."""
    arm_idx = wp.tid()

    idx = task_idx[arm_idx]
    time_limit = task_time_soft_limits[idx]

    ee_pos_current = wp.transform_get_translation(body_q[ee_body_idx])
    ee_rot_current = wp.transform_get_rotation(body_q[ee_body_idx])

    pos_err = wp.abs(ee_pos_target[arm_idx] - ee_pos_current)
    pos_tol = pos_error_thresholds[idx]

    rv = ee_rot_target[arm_idx]
    target_quat = wp.quaternion(rv[0], rv[1], rv[2], rv[3])

    quat_rel = ee_rot_current * wp.quat_inverse(target_quat)
    rot_err = wp.abs(2.0 * wp.atan2(wp.length(quat_rel[:3]), wp.abs(quat_rel[3])))

    if (
        task_time_elapsed[arm_idx] >= time_limit
        and task_idx[arm_idx] < wp.len(task_time_soft_limits) - 1
        and pos_err[0] < pos_tol[0]
        and pos_err[1] < pos_tol[1]
        and pos_err[2] < pos_tol[2]
        and rot_err < rot_error_thresholds[idx]
    ):
        task_idx[arm_idx] += 1
        task_time_elapsed[arm_idx] = 0.0
        task_ee_init_body_q[arm_idx] = body_q[ee_body_idx]


# ---------------------------------------------------------------------------
# Asset paths
# ---------------------------------------------------------------------------

_DIR = os.path.dirname(__file__)
_ASSETS_DIR = os.path.join(_DIR, "assets")
ROBOT_PATH = os.path.realpath(os.path.join(_ASSETS_DIR, "rby1df", "urdf")) + os.sep

# Per-shape contact stiffness/damping used by VBD. The solver averages these
# across each contact pair to form the per-contact normal penalty values.
VBD_KE = 1.0e3
VBD_KD = 0.0


def _default_cable_usd_path() -> str:
    return os.path.realpath(os.path.join(_ASSETS_DIR, "Cable008", "curve", "cable_SRA_curve03.usda"))


def _default_scene_usd_path() -> str:
    return os.path.realpath(os.path.join(_ASSETS_DIR, "Cable008", "Cable008_Body.usda"))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class CollisionMode(enum.Enum):
    MUJOCO = "mujoco"
    NEWTON_DEFAULT = "newton_default"
    NEWTON_SDF = "newton_sdf"
    NEWTON_HYDROELASTIC = "newton_hydroelastic"


class TaskType(enum.IntEnum):
    APPROACH = 0
    ENGAGE = 1
    GRASP = 2
    HOLD_GRASP = 3
    RETRACT = 4
    SETTLE = 5
    APPROACH_TARGET = 6
    ALIGN_AXES = 7
    VERIFY_ALIGN = 8
    INSERT = 9
    RELEASE = 10
    WITHDRAW = 11
    WAIT_AFTER_WITHDRAW = 12
    # --- Re-grasp and pull ---
    REAPPROACH = 13
    REENGAGE = 14
    REGRASP = 15
    PULL = 16
    FINAL_RELEASE = 17
    DONE = 18


def _np_quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def _np_quat_inverse(q):
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def _np_quat_rotate(q, v):
    xyz = np.array([q[0], q[1], q[2]], dtype=np.float64)
    t = 2.0 * np.cross(xyz, v)
    return v + q[3] * t + np.cross(xyz, t)


def _np_tf7(pos, quat):
    return np.concatenate([np.asarray(pos, dtype=np.float64), np.asarray(quat, dtype=np.float64)])


def _np_tf_multiply(t1, t2):
    p1, q1 = t1[:3], t1[3:]
    p2, q2 = t2[:3], t2[3:]
    return _np_tf7(p1 + _np_quat_rotate(q1, p2), _np_quat_multiply(q1, q2))


def _np_tf_inverse(t):
    p, q = t[:3], t[3:]
    qi = _np_quat_inverse(q)
    return _np_tf7(-_np_quat_rotate(qi, p), qi)


def _np_quat_slerp(q0, q1, t):
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        r = q0 + t * (q1 - q0)
        return r / np.linalg.norm(r)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    a = np.sin((1.0 - t) * theta) / sin_theta
    b = np.sin(t * theta) / sin_theta
    return a * q0 + b * q1


# ---------------------------------------------------------------------------
# Initial joint states (LeRobot episode 0, frame 0)
# ---------------------------------------------------------------------------

LEROBOT_INITIAL_STATE_22 = [
    0.0,
    0.872664213180542,
    -1.5707811117172241,
    0.6981245279312134,
    3.796982127823867e-06,
    0.0,
    0.3021828234195709,
    -0.013802030123770237,
    -0.09509921818971634,
    -2.2242417335510254,
    -0.7117632627487183,
    0.14113007485866547,
    0.5137608647346497,
    -0.4555884897708893,
    0.2500312626361847,
    -0.665743887424469,
    -1.3314952850341797,
    -0.19328542053699493,
    -0.5307496786117554,
    0.6565361022949219,
    0.0913801970053464174,
    0.09098683297634125,
]

_JOINT_7_OFFSET = np.pi / 2


def _lerobot_22_to_urdf_28(lr: list[float]) -> list[float]:
    q = [0.0] * 28
    q[0:6] = lr[0:6]
    q[6:13] = lr[6:13]
    q[12] += _JOINT_7_OFFSET
    q[13] = lr[20]
    q[14] = -lr[20] / 2.0
    q[15] = lr[20] / 2.0
    q[16:23] = lr[13:20]
    q[22] -= _JOINT_7_OFFSET
    q[23] = lr[21]
    q[24] = -lr[21] / 2.0
    q[25] = lr[21] / 2.0
    return q


def _get_initial_joint_q() -> list[float]:
    return _lerobot_22_to_urdf_28(LEROBOT_INITIAL_STATE_22)


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args

        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.frame_count = 0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer._paused = True
        self.rigid_contact_max = 100000

        self.ik_iters = 24
        self.auto_mode = True

        # MuJoCo collision - NEWTON_SDF (capture-safe)
        self.collision_mode = CollisionMode.NEWTON_SDF
        self.default_shape_cfg = self._create_shape_config(self.collision_mode)
        self.mesh_sdf_params = None
        self.mujoco_collide_substeps = 100
        self.vbd_collide_substeps = 5

        self.gripper_joint_dofs = [13, 23]
        self.gripper_finger_dofs = [14, 15, 24, 25]

        # Cable capsule segment radius (from USD: widths=0.006, radius=0.003 m)
        self.capsule_radius = 0.003

        # VBD mesh SDF
        self.vbd_mesh_use_sdf = True
        self.vbd_mesh_sdf_max_resolution = 64

        # Two-way coupling parameters
        self.enable_two_way_coupling = not getattr(args, "no_twoway", False)
        self.proxy_mass_source = "effective"
        self.proxy_mass_scale = 1.0
        self.vbd_proxy_mu = 1.0e6
        self.vbd_proxy_margin = 0.001
        self.verbose = False

        # Fridge xform - keeps robot-fridge relative pose
        self.fridge_xform = self._compute_fridge_xform()

        # Socket approach xform (SM target for APPROACH_TARGET / INSERT)
        self.socket_approach_xform = self._compute_socket_approach_xform(self.fridge_xform)
        self.sm_socket_pos = wp.transform_get_translation(self.socket_approach_xform)
        self.sm_socket_rot = wp.transform_get_rotation(self.socket_approach_xform)

        # Insertion depth parameters:
        # - final_depth: snapped/locked connector-tip depth along insertion_dir
        # - snap_margin: allow snapping slightly before the final depth
        self.insert_final_depth = 0.035
        self.insert_snap_margin = 0.001

        # Fixed joint penalty gains activated at snap. Strong linear + weak angular
        # so the robot can pull the cable. Fixed-k: snap_k_*_max == snap_k_* so
        # AVBD has no headroom to ramp.
        self.snap_k_lin = 1.0e7
        self.snap_k_ang = 1.0e1
        self.snap_k_lin_max = self.snap_k_lin
        self.snap_k_ang_max = self.snap_k_ang
        self.snap_kd_lin = 1.0e-1
        self.snap_kd_ang = 1.0e-1

        self.regrasp_z_offset = 0.003  # [m] Z offset for re-grasp target

        # Pull motion parameters: move in (-) insertion direction after re-grasp
        self.pull_distance = 0.06  # [m] how far to pull along -insertion_dir
        self.pull_duration = 4.0  # [s] smoothstep duration for pull

        # ----- MuJoCo world (robot only) -----
        self._setup_mujoco_world(args)

        # ----- VBD world (fridge scene + cable) -----
        self._setup_vbd_world(args)

        # ----- IK, grippers, state machine -----
        self.setup_end_effectors()
        self.setup_ik()
        self.setup_gripper_targets()
        self.setup_state_machine()

        self.joint_target_pos = wp.zeros_like(self.control.joint_target_pos)
        wp.copy(self.joint_target_pos, self.control.joint_target_pos)

        # Viewer primary model: "mujoco" shows robot + VBD overlay; "vbd" shows VBD only (incl. proxies)
        self.viewer_primary_model = str(getattr(args, "primary_view", "vbd")) if args is not None else "vbd"
        if self.viewer_primary_model == "mujoco":
            self.viewer.set_model(self.mujoco_model)
        else:
            self.viewer.set_model(self.vbd_model)
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.set_camera(wp.vec3(-2.55, -7.1, 2.3), pitch=-12.0, yaw=-295.0)
            self.viewer.camera.fov = 15.0

        self.capture()

        if self.auto_mode:
            self._start_auto_mode()

    # ------------------------------------------------------------------
    # MuJoCo world (robot only)
    # ------------------------------------------------------------------

    def _setup_mujoco_world(self, args):
        robot = self.setup_robot_builder()
        scene = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(scene)
        scene.default_shape_cfg = self.default_shape_cfg
        scene.add_builder(robot)

        self.single_robot_model = robot.finalize()
        self.mujoco_model = scene.finalize()

        newton.eval_fk(self.mujoco_model, self.mujoco_model.joint_q, self.mujoco_model.joint_qd, self.mujoco_model)

        self.mujoco_collision_pipeline = newton.CollisionPipeline(
            self.mujoco_model,
            reduce_contacts=True,
            broad_phase="explicit",
        )

        if not hasattr(_mjw, "set_length_range"):
            _mjw.set_length_range = lambda m, d: None

        self.mujoco_solver = newton.solvers.SolverMuJoCo(
            self.mujoco_model,
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            njmax=self.rigid_contact_max,
            nconmax=self.rigid_contact_max,
            ls_parallel=True,
            iterations=20,
            ls_iterations=10,
            use_mujoco_contacts=False,
            impratio=1000.0,
        )

        self.state_0 = self.mujoco_model.state()
        self.state_1 = self.mujoco_model.state()
        self.control = self.mujoco_model.control()

        newton.eval_fk(self.mujoco_model, self.mujoco_model.joint_q, self.mujoco_model.joint_qd, self.state_0)
        self.mujoco_contacts = self.mujoco_collision_pipeline.contacts()
        self.mujoco_collision_pipeline.collide(self.state_0, self.mujoco_contacts)

    # ------------------------------------------------------------------
    # VBD world (fridge scene + cable)
    # ------------------------------------------------------------------

    def _setup_vbd_world(self, args):
        builder = newton.ModelBuilder()
        builder.rigid_contact_margin = 0.0
        builder.rigid_gap = 0.001

        builder.default_shape_cfg.density = 1000.0
        builder.default_shape_cfg.ke = VBD_KE
        builder.default_shape_cfg.kd = VBD_KD
        builder.default_shape_cfg.mu = 0.2

        # Load fridge scene (static, zero mass).
        scene_usd_path = _default_scene_usd_path()
        self.scene_body_ids: list[int] = []
        if os.path.isfile(scene_usd_path):
            scene_result = builder.add_usd(
                scene_usd_path,
                xform=self.fridge_xform,
                root_path="/root",
                load_sites=False,
                load_visual_shapes=True,
                hide_collision_shapes=False,
                parse_mujoco_options=False,
                only_load_enabled_joints=True,
                only_load_enabled_rigid_bodies=False,
            )
            self.scene_body_ids = sorted({int(v) for v in scene_result["path_body_map"].values()})
            for body_id in self.scene_body_ids:
                builder.body_mass[body_id] = 0.0
                builder.body_inv_mass[body_id] = 0.0
                builder.body_inertia[body_id] = wp.mat33()
                builder.body_inv_inertia[body_id] = wp.mat33()

            scene_shape_ids = sorted(int(v) for v in scene_result["path_shape_map"].values())

            # Filter self-collisions among all scene shapes (static body).
            for i in range(len(scene_shape_ids)):
                for j in range(i + 1, len(scene_shape_ids)):
                    builder.add_shape_collision_filter_pair(scene_shape_ids[i], scene_shape_ids[j])

            self._scene_shape_ids = scene_shape_ids

        # Load cables from USD (two curves)
        cable_usd_path = _default_cable_usd_path()
        curve_prim_paths = ["/World/cable001/curve_0", "/World/cable002/curve_0"]

        self.cable_body_ids: list[int] = []
        self.cable_joint_ids: list[int] = []
        self.cable_fixed_body_ids: list[int] = []
        self.cable_head_body_ids: list[int] = []
        cable_results = []

        light_head_cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, ke=VBD_KE, kd=VBD_KD)
        cable_shape_cfg = builder.default_shape_cfg.copy()
        cable_shape_cfg.ke = VBD_KE
        cable_shape_cfg.kd = VBD_KD

        # Per-joint cable target_ke values, tuned for waterhose-scale segments
        # (mean segment length is about 1 cm).
        stretch_stiffness_per_joint = 1.0e6
        bend_stiffness_per_joint = 2.0e1

        for i, curve_prim_path in enumerate(curve_prim_paths):
            result = add_cable_from_usd_curve(
                builder=builder,
                source_usd_path=cable_usd_path,
                curve_prim_path=curve_prim_path,
                cable_label=f"water_hose_cable_{i}",
                cable_cfg=cable_shape_cfg,
                stretch_stiffness=stretch_stiffness_per_joint,
                stretch_damping=1.0e-5,
                bend_stiffness=bend_stiffness_per_joint,
                bend_damping=1.0e0,
                wrap_in_articulation=False,
                head_shape_mode="mesh",
                head_cfg=light_head_cfg,
                head_mass=0.0,
            )
            cable_results.append(result)
            self.cable_body_ids.extend(result.cable_body_ids)
            self.cable_body_ids.extend(result.head_body_ids)
            self.cable_joint_ids.extend(result.cable_joint_ids)
            self.cable_joint_ids.extend(result.head_fixed_joint_ids)
            self.cable_fixed_body_ids.extend(int(v) for v in result.fixed_body_ids)
            self.cable_head_body_ids.extend(result.head_body_ids)

            if i == 0 and result.head_fixed_joint_ids:
                self._capsule_plug_joint_idx = result.head_fixed_joint_ids[0]

            # Filter collision: head mesh <-> capsule adjacent to its parent.
            # Fixed joint already filters mesh <-> parent capsule.
            if result.head_body_ids and len(result.cable_body_ids) >= 2:
                neighbor_idx = 1 if i == 0 else -2
                neighbor_body = result.cable_body_ids[neighbor_idx]
                for hb in result.head_body_ids:
                    for hs in builder.body_shapes.get(hb, []):
                        for ns in builder.body_shapes.get(neighbor_body, []):
                            builder.add_shape_collision_filter_pair(hs, ns)

        # Filter intra-cable self-collisions: each cable's bodies only
        # collide with the other cable (and scene/proxies), not themselves.
        self._debug_cable_results = cable_results
        for result in cable_results:
            all_bodies = list(result.cable_body_ids) + list(result.head_body_ids)
            shape_ids: list[int] = []
            for bid in all_bodies:
                shape_ids.extend(builder.body_shapes.get(bid, []))
            for si in range(len(shape_ids)):
                for sj in range(si + 1, len(shape_ids)):
                    builder.add_shape_collision_filter_pair(shape_ids[si], shape_ids[sj])

        # Zero mass on fixed bodies (cable pinned at one end)
        fixed_body_seen: set[int] = set()
        for body_id in self.cable_fixed_body_ids:
            if body_id in fixed_body_seen:
                continue
            fixed_body_seen.add(body_id)
            builder.body_mass[body_id] = 0.0
            builder.body_inv_mass[body_id] = 0.0
            builder.body_inertia[body_id] = wp.mat33()
            builder.body_inv_inertia[body_id] = wp.mat33()
        self.cable_fixed_body_ids = list(fixed_body_seen)

        if self.cable_joint_ids:
            builder.add_articulation(self.cable_joint_ids, label="water_hose_cable_articulation")

        # Add proxy gripper bodies from MuJoCo robot into VBD world.
        if self.enable_two_way_coupling:
            self._create_proxy_bodies(builder)

        # Pre-create a dormant fixed joint between the tip capsule and an anchor
        # at the insert target pose. Stiffness is set to zero initially; activated
        # at runtime when INSERT completes.
        # Use numpy math (same path as _tip_insert_rot_np / pin snap).
        spos = np.array([float(self.sm_socket_pos[i]) for i in range(3)], dtype=np.float64)
        socket_rot = np.array([float(self.sm_socket_rot[i]) for i in range(4)], dtype=np.float64)
        ins_dir = _np_quat_rotate(socket_rot, np.array([0.0, 0.0, 1.0]))
        flip_x = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        anchor_rot = _np_quat_multiply(socket_rot, flip_x)
        anchor_pos = spos + self.insert_final_depth * ins_dir
        anchor_xform = wp.transform(
            wp.vec3(float(anchor_pos[0]), float(anchor_pos[1]), float(anchor_pos[2])),
            wp.quat(float(anchor_rot[0]), float(anchor_rot[1]), float(anchor_rot[2]), float(anchor_rot[3])),
        )
        self._insert_anchor_body = builder.add_body(xform=anchor_xform, mass=0.0)
        builder.body_inv_mass[self._insert_anchor_body] = 0.0
        builder.body_inertia[self._insert_anchor_body] = wp.mat33()
        builder.body_inv_inertia[self._insert_anchor_body] = wp.mat33()

        tip_body = cable_results[0].cable_body_ids[0] if cable_results else 0
        # Body origin is at the connector tip.  Pin the tip (body origin)
        # by using identity child_xform and offsetting the parent anchor.
        tip_shape = builder.body_shapes[tip_body][0]
        hh = float(builder.shape_scale[tip_shape][1])
        self._insert_fixed_joint = builder.add_joint_fixed(
            parent=self._insert_anchor_body,
            child=tip_body,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, -hh), wp.quat_identity()),
            child_xform=wp.transform(),
        )

        # Dormant fixed joint for the plug mesh body (world-anchored).
        head_body = cable_results[0].head_body_ids[0] if (cable_results and cable_results[0].head_body_ids) else None
        if head_body is not None:
            self._plug_anchor_body = builder.add_body(xform=anchor_xform, mass=0.0)
            builder.body_inv_mass[self._plug_anchor_body] = 0.0
            builder.body_inertia[self._plug_anchor_body] = wp.mat33()
            builder.body_inv_inertia[self._plug_anchor_body] = wp.mat33()
            self._plug_fixed_joint = builder.add_joint_fixed(
                parent=self._plug_anchor_body,
                child=head_body,
                parent_xform=wp.transform(),
                child_xform=wp.transform(),
            )
        else:
            self._plug_anchor_body = None
            self._plug_fixed_joint = None

        builder.color()
        sim_device = wp.get_device(args.device)
        self.vbd_model = builder.finalize(device=sim_device)
        self.vbd_model.set_gravity((0.0, 0.0, -9.81))

        if self.vbd_mesh_use_sdf:
            shape_type_np = self.vbd_model.shape_type.numpy()
            scene_shape_set = set(getattr(self, "_scene_shape_ids", []))
            for s in range(self.vbd_model.shape_count):
                if int(shape_type_np[s]) == GeoType.MESH and s in scene_shape_set:
                    mesh = self.vbd_model.shape_source[s]
                    if mesh is not None:
                        mesh.build_sdf(max_resolution=self.vbd_mesh_sdf_max_resolution)

        self.vbd_solver = newton.solvers.SolverVBD(
            self.vbd_model,
            iterations=15,
            friction_epsilon=0.1,
            rigid_body_contact_buffer_size=1024,
            rigid_body_particle_contact_buffer_size=1,
            rigid_contact_hard=False,
            rigid_joint_linear_ke=1.0e6,
            rigid_joint_angular_ke=1.0e6,
        )

        # Preallocated snapshot of body_q_prev for collect_rigid_contact_forces;
        # filled via wp.copy each substep instead of wp.clone (avoids per-substep
        # GPU allocations).
        self._vbd_body_q_prev_snapshot = wp.zeros_like(self.vbd_solver.body_q_prev)

        # Soften every VBD joint to penalty-only mode (no augmented-Lagrangian
        # dual variable). Avoids lambda accumulation against cable bend torques.
        for j in range(self.vbd_model.joint_count):
            self.vbd_solver.set_joint_constraint_mode(j, hard=False)

        # Zero out dormant joint constraint caps until snap activation.
        jc_start = self.vbd_solver.joint_constraint_start.numpy()
        dev = self.vbd_solver.joint_penalty_k_max.device

        c0 = int(jc_start[self._insert_fixed_joint])
        self._insert_joint_constraint_start = c0

        dormant_slots = [(c0, c0 + 1)]
        if self._plug_fixed_joint is not None:
            cp = int(jc_start[self._plug_fixed_joint])
            self._plug_joint_constraint_start = cp
            dormant_slots.append((cp, cp + 1))
        else:
            self._plug_joint_constraint_start = None

        for arr_name in ("joint_penalty_k_max", "joint_penalty_k_min", "joint_penalty_k", "joint_penalty_kd"):
            arr = getattr(self.vbd_solver, arr_name)
            tmp = arr.numpy()
            for s0, s1 in dormant_slots:
                tmp[s0] = 0.0
                tmp[s1] = 0.0
            wp.copy(arr, wp.array(tmp, dtype=float, device=dev))

        self.vbd_state_0 = self.vbd_model.state()
        self.vbd_state_1 = self.vbd_model.state()
        self.vbd_control = self.vbd_model.control()

        # Initialize coupling buffers (GPU arrays for kernel launches).
        if self.enable_two_way_coupling:
            self._init_coupling_buffers()

        # Explicit collision pipeline + pre-allocated contacts (capture-safe).
        self.vbd_collision_pipeline = newton.examples.create_collision_pipeline(
            self.vbd_model,
            args,
            rigid_contact_max=30000,
        )
        self.vbd_contacts = self.vbd_collision_pipeline.contacts()

        # Warm up VBD collision pipeline (compile kernels before CUDA graph capture).
        self.vbd_collision_pipeline.collide(self.vbd_state_0, self.vbd_contacts)

        # Transform cable body positions by fridge_xform (cable was loaded at
        # USD-native position; scene was loaded with xform= so it's already
        # transformed).  VBD operates on body_q directly, so this persists.
        self._transform_cable_bodies(self.fridge_xform)
        if getattr(args, "print_cable_poses", False):
            self._print_cable_pose_summary("reference_after_fridge_xform", cable_results)

        # Cable head body index in VBD model (grasp target for SM)
        if self.cable_head_body_ids:
            self.cable_head_body_idx = self.cable_head_body_ids[0]
        else:
            self.cable_head_body_idx = 0

        # Tip capsule body = long capsule (edge 0 for cable001).
        # Body origin = connector tip (start node), +Z toward cable.
        if cable_results:
            self.tip_capsule_body_idx = cable_results[0].cable_body_ids[0]
            shape_idx = builder.body_shapes[self.tip_capsule_body_idx][0]
            self.tip_half_height = float(builder.shape_scale[shape_idx][1])
        else:
            self.tip_capsule_body_idx = 0
            self.tip_half_height = 0.022

        # Pre-compute rendering geometry for VBD overlay.
        self._setup_vbd_render_data(cable_results)

    # ------------------------------------------------------------------
    # VBD render data
    # ------------------------------------------------------------------

    def _setup_vbd_render_data(self, cable_results):
        """Pre-compute VBD rendering geometry (cable capsule specs, scene mesh batches)."""
        shape_body_np = self.vbd_model.shape_body.numpy()
        shape_type_np = self.vbd_model.shape_type.numpy()
        shape_transform_np = self.vbd_model.shape_transform.numpy()
        body_q_np = self.vbd_state_0.body_q.numpy()

        # Per-cable capsule segment body IDs (list of lists).
        self.vbd_cable_all_body_ids = [list(r.cable_body_ids) for r in cable_results]

        # Per-body capsule shape index: body_id -> shape_index.
        # Stores the shape index so _render_vbd_objects can read both
        # shape_scale (geo_scale for log_shapes) and shape_transform
        # (for the exact same world-xform computation that log_state uses).
        all_cable_body_set: set[int] = set()
        for cable_body_ids in self.vbd_cable_all_body_ids:
            all_cable_body_set.update(cable_body_ids)
        self._body_capsule_shape: dict[int, int] = {}
        for s in range(self.vbd_model.shape_count):
            bid = int(shape_body_np[s])
            if bid in all_cable_body_set and int(shape_type_np[s]) == GeoType.CAPSULE:
                self._body_capsule_shape[bid] = s

        # Head body mesh shapes (dynamic - read body_q each frame).
        self.vbd_head_mesh_shapes = []  # list of (shape_idx, mesh)
        head_body_set = set(self.cable_head_body_ids)
        for s in range(self.vbd_model.shape_count):
            body_idx = int(shape_body_np[s])
            if body_idx not in head_body_set:
                continue
            if int(shape_type_np[s]) != GeoType.MESH:
                continue
            mesh = self.vbd_model.shape_source[s]
            if mesh is not None:
                self.vbd_head_mesh_shapes.append((s, mesh))

        self.tip_mesh_friction = 1.0e1
        self.plug_xy_scale = 0.95
        if self.vbd_head_mesh_shapes and self.vbd_model.shape_material_mu is not None:
            mu_np = self.vbd_model.shape_material_mu.numpy()
            for s, _ in self.vbd_head_mesh_shapes:
                mu_np[s] = self.tip_mesh_friction
            self.vbd_model.shape_material_mu = wp.array(mu_np, dtype=float)

        if self.vbd_head_mesh_shapes:
            scale_np = self.vbd_model.shape_scale.numpy()
            for s, _ in self.vbd_head_mesh_shapes:
                scale_np[s][0] *= self.plug_xy_scale
                scale_np[s][1] *= self.plug_xy_scale
            self.vbd_model.shape_scale = wp.array(scale_np, dtype=wp.vec3)

        # Scene mesh xforms (static - pre-computed once, zero-mass bodies).
        # Group by mesh source for batched rendering.
        scene_body_set = set(self.scene_body_ids)
        mesh_group_map: dict[int, tuple[object, list[wp.transform]]] = {}
        for s in range(self.vbd_model.shape_count):
            body_idx = int(shape_body_np[s])
            if body_idx not in scene_body_set:
                continue
            if int(shape_type_np[s]) != GeoType.MESH:
                continue
            mesh = self.vbd_model.shape_source[s]
            if mesh is None:
                continue
            bq = body_q_np[body_idx]
            body_xform = wp.transform(
                wp.vec3(bq[0], bq[1], bq[2]),
                wp.quat(bq[3], bq[4], bq[5], bq[6]),
            )
            st = shape_transform_np[s]
            shape_local = wp.transform(
                wp.vec3(st[0], st[1], st[2]),
                wp.quat(st[3], st[4], st[5], st[6]),
            )
            world_xform = wp.transform_multiply(body_xform, shape_local)
            mesh_id = id(mesh)
            if mesh_id not in mesh_group_map:
                mesh_group_map[mesh_id] = (mesh, [])
            mesh_group_map[mesh_id][1].append(world_xform)

        self.vbd_scene_mesh_batches = []  # list of (mesh, wp.array of xforms)
        for _mesh_id, (mesh, xforms) in mesh_group_map.items():
            self.vbd_scene_mesh_batches.append((mesh, wp.array(xforms, dtype=wp.transform)))

    # ------------------------------------------------------------------
    # Proxy coupling infrastructure
    # ------------------------------------------------------------------

    def _build_newton_to_mjc_body_map(self):
        """Invert ``mjc_body_to_newton`` for world 0."""
        mjc_to_newton_np = self.mujoco_solver.mjc_body_to_newton.numpy()
        newton_to_mjc: dict[int, int] = {}
        mjc_nbody = mjc_to_newton_np.shape[1]
        for mjc_body in range(mjc_nbody):
            newton_body = int(mjc_to_newton_np[0, mjc_body])
            if newton_body >= 0:
                newton_to_mjc[newton_body] = mjc_body
        return newton_to_mjc

    def _compute_proxy_mass(self, newton_body_id, local_mass, newton_to_mjc):
        """Compute proxy body mass (effective or local)."""
        if self.proxy_mass_source == "local":
            return local_mass
        mjc_body = newton_to_mjc.get(newton_body_id)
        if mjc_body is None:
            return local_mass
        invweight0_np = self.mujoco_solver.mjw_model.body_invweight0.numpy()
        inv_w_trans = float(invweight0_np[0, mjc_body, 0])
        if inv_w_trans <= 0.0:
            return local_mass
        effective_mass = 1.0 / inv_w_trans
        if self.verbose:
            print(
                f"    Effective mass for body {newton_body_id} "
                f"(mjc {mjc_body}): {effective_mass:.4f} kg  "
                f"(local: {local_mass:.4f} kg, ratio: {effective_mass / max(local_mass, 1e-30):.1f}x)"
            )
        return effective_mass

    def _compute_proxy_inertia(self, newton_body_id, local_mass, effective_mass, newton_to_mjc):
        """Compute proxy inertia tensor scaled by effective mass ratio."""
        if local_mass <= 0.0 or effective_mass <= 0.0:
            return None
        body_inertia_np = self.mujoco_model.body_inertia.numpy()
        local_inertia = body_inertia_np[newton_body_id]
        local_diag_mean = float(np.mean(np.diag(local_inertia)))
        if local_diag_mean < 1e-30:
            return None
        if self.proxy_mass_source == "effective":
            mjc_body = newton_to_mjc.get(newton_body_id)
            if mjc_body is not None:
                invweight0_np = self.mujoco_solver.mjw_model.body_invweight0.numpy()
                inv_w_rot = float(invweight0_np[0, mjc_body, 1])
                if inv_w_rot > 0.0:
                    effective_rot_inertia = 1.0 / inv_w_rot
                    rot_ratio = effective_rot_inertia / local_diag_mean
                    scaled = local_inertia * rot_ratio
                    return wp.mat33(
                        scaled[0, 0],
                        scaled[0, 1],
                        scaled[0, 2],
                        scaled[1, 0],
                        scaled[1, 1],
                        scaled[1, 2],
                        scaled[2, 0],
                        scaled[2, 1],
                        scaled[2, 2],
                    )
        mass_ratio = effective_mass / local_mass
        scaled = local_inertia * mass_ratio
        return wp.mat33(
            scaled[0, 0],
            scaled[0, 1],
            scaled[0, 2],
            scaled[1, 0],
            scaled[1, 1],
            scaled[1, 2],
            scaled[2, 0],
            scaled[2, 1],
            scaled[2, 2],
        )

    def _create_proxy_bodies(self, vbd_builder):
        """Create proxy gripper bodies in VBD that mirror MuJoCo robot fingers."""
        if self.verbose:
            print("Creating proxy bodies for two-way coupling...")
            print(f"  Proxy mass source: {self.proxy_mass_source}")

        self.mj_to_vbd_body_map = {}
        self.proxy_body_ids = []
        self.proxy_mj_body_ids = []

        newton_to_mjc = self._build_newton_to_mjc_body_map()

        body_inv_mass_np = self.mujoco_model.body_inv_mass.numpy()
        body_q_np = self.state_0.body_q.numpy()
        shape_body_np = self.mujoco_model.shape_body.numpy()
        shape_type_np = self.mujoco_model.shape_type.numpy()
        shape_scale_np = self.mujoco_model.shape_scale.numpy()
        shape_transform_np = self.mujoco_model.shape_transform.numpy()

        gripper_finger_keys = {
            "right_gripper_leftfinger",
            "right_gripper_rightfinger",
            "left_gripper_leftfinger",
            "left_gripper_rightfinger",
            # Uncomment to include gripper bases as proxy bodies:
            # "right_gripper_base",
            # "left_gripper_base",
        }

        proxy_finger_shape_ids: dict[str, list[int]] = {}

        proxy_shape_cfg = vbd_builder.default_shape_cfg.copy()
        proxy_shape_cfg.ke = VBD_KE
        proxy_shape_cfg.kd = VBD_KD
        proxy_shape_cfg.mu = self.vbd_proxy_mu
        proxy_shape_cfg.margin = self.vbd_proxy_margin

        for mj_body_id in range(self.mujoco_model.body_count):
            if mj_body_id == 0:
                self.mj_to_vbd_body_map[mj_body_id] = -1
                continue

            body_lbl = (
                self.mujoco_model.body_label[mj_body_id] if mj_body_id < len(self.mujoco_model.body_label) else ""
            )
            body_short = body_lbl.rsplit("/", 1)[-1] if "/" in body_lbl else body_lbl
            if body_short not in gripper_finger_keys:
                self.mj_to_vbd_body_map[mj_body_id] = -1
                continue

            body_inv_mass = body_inv_mass_np[mj_body_id]
            local_mass = float(1.0 / body_inv_mass) if body_inv_mass > 0 else 0.0
            if local_mass <= 0.0 or local_mass > 1e9:
                self.mj_to_vbd_body_map[mj_body_id] = -1
                continue

            effective_mass = self._compute_proxy_mass(mj_body_id, local_mass, newton_to_mjc)
            scaled_mass = effective_mass * self.proxy_mass_scale
            proxy_inertia = self._compute_proxy_inertia(mj_body_id, local_mass, effective_mass, newton_to_mjc)

            body_q = body_q_np[mj_body_id]
            initial_xform = wp.transform(
                wp.vec3(body_q[0], body_q[1], body_q[2]),
                wp.quat(body_q[3], body_q[4], body_q[5], body_q[6]),
            )

            proxy_body_id = vbd_builder.add_body(
                xform=initial_xform,
                mass=scaled_mass,
                inertia=proxy_inertia,
                lock_inertia=proxy_inertia is not None,
                label=f"proxy_{body_short}",
            )

            shape_ids: list[int] = []
            for shape_idx in range(len(shape_body_np)):
                if shape_body_np[shape_idx] != mj_body_id:
                    continue
                shape_type = shape_type_np[shape_idx]
                shape_scale = shape_scale_np[shape_idx]
                shape_xform_data = shape_transform_np[shape_idx]
                pos = wp.vec3(shape_xform_data[0], shape_xform_data[1], shape_xform_data[2])
                rot = wp.quat(shape_xform_data[3], shape_xform_data[4], shape_xform_data[5], shape_xform_data[6])

                if shape_type == GeoType.SPHERE:
                    sid = vbd_builder.add_shape_sphere(
                        body=proxy_body_id,
                        radius=float(shape_scale[0]),
                        pos=pos,
                        rot=rot,
                        cfg=proxy_shape_cfg,
                    )
                    shape_ids.append(int(sid))
                elif shape_type == GeoType.BOX:
                    sid = vbd_builder.add_shape_box(
                        body=proxy_body_id,
                        hx=float(shape_scale[0]),
                        hy=float(shape_scale[1]),
                        hz=float(shape_scale[2]),
                        pos=pos,
                        rot=rot,
                        cfg=proxy_shape_cfg,
                    )
                    shape_ids.append(int(sid))
                elif shape_type == GeoType.CAPSULE:
                    sid = vbd_builder.add_shape_capsule(
                        body=proxy_body_id,
                        radius=float(shape_scale[0]),
                        half_height=float(shape_scale[1]),
                        pos=pos,
                        rot=rot,
                        cfg=proxy_shape_cfg,
                    )
                    shape_ids.append(int(sid))
                elif shape_type == GeoType.MESH:
                    shape_source = self.mujoco_model.shape_source[shape_idx]
                    shape_xform = wp.transform(p=pos, q=rot)
                    sid = vbd_builder.add_shape_mesh(
                        body=proxy_body_id,
                        mesh=shape_source,
                        xform=shape_xform,
                        cfg=proxy_shape_cfg,
                    )
                    shape_ids.append(int(sid))

            if not shape_ids:
                sid = vbd_builder.add_shape_box(
                    body=proxy_body_id,
                    hx=0.02,
                    hy=0.01,
                    hz=0.04,
                    cfg=proxy_shape_cfg,
                )
                shape_ids.append(int(sid))

            proxy_finger_shape_ids[body_short] = shape_ids
            self.mj_to_vbd_body_map[mj_body_id] = proxy_body_id
            self.proxy_body_ids.append(proxy_body_id)
            self.proxy_mj_body_ids.append(mj_body_id)

        all_proxy_shape_ids: list[int] = []
        for sids in proxy_finger_shape_ids.values():
            all_proxy_shape_ids.extend(sids)
        self._proxy_shape_ids = all_proxy_shape_ids

        # Filter self-collisions between proxy shapes of the same gripper.
        for parts in [
            ["right_gripper_base", "right_gripper_leftfinger", "right_gripper_rightfinger"],
            ["left_gripper_base", "left_gripper_leftfinger", "left_gripper_rightfinger"],
        ]:
            for i in range(len(parts)):
                for j in range(i + 1, len(parts)):
                    sa = proxy_finger_shape_ids.get(parts[i], [])
                    sb = proxy_finger_shape_ids.get(parts[j], [])
                    for s1 in sa:
                        for s2 in sb:
                            vbd_builder.add_shape_collision_filter_pair(int(s1), int(s2))

        # Filter proxy shapes against scene shapes - proxies only need to
        # collide with cables. Proxy-vs-scene would double-count forces
        # already handled by MuJoCo coupling and generates massive
        # mesh-mesh triangle pairs (~1.8M with 251 fridge meshes).
        if hasattr(self, "_scene_shape_ids"):
            for proxy_sid in all_proxy_shape_ids:
                for scene_sid in self._scene_shape_ids:
                    vbd_builder.add_shape_collision_filter_pair(proxy_sid, scene_sid)

    def _init_coupling_buffers(self):
        """Initialize GPU arrays for proxy state sync and force exchange."""
        device = wp.get_device(self.vbd_model.device)

        mj_to_vbd_map_list = [-1] * int(self.mujoco_model.body_count)
        for mj_id, vbd_id in self.mj_to_vbd_body_map.items():
            mj_to_vbd_map_list[mj_id] = vbd_id
        self.mj_to_vbd_body_map_array = wp.array(mj_to_vbd_map_list, dtype=int, device=device)

        self.proxy_vbd_body_ids_array = wp.array(self.proxy_body_ids, dtype=int, device=device)
        self.proxy_mj_body_ids_array = wp.array(self.proxy_mj_body_ids, dtype=int, device=device)

        vbd_to_mj = [-1] * self.vbd_model.body_count
        for vbd_id, mj_id in zip(self.proxy_body_ids, self.proxy_mj_body_ids, strict=True):
            vbd_to_mj[vbd_id] = mj_id
        self.vbd_to_mj_body_map_array = wp.array(vbd_to_mj, dtype=int, device=device)

        body_force_template = self.state_0.body_f
        self.proxy_forces = wp.zeros_like(body_force_template)
        self.coupling_forces_cache = wp.zeros_like(body_force_template)

    # ------------------------------------------------------------------
    # Robot setup
    # ------------------------------------------------------------------

    def setup_robot_builder(self):
        robot = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(robot)
        robot.default_shape_cfg = self.default_shape_cfg

        robot_file = ROBOT_PATH + "robot_edited.urdf"

        robot.add_urdf(
            robot_file,
            floating=False,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
            ignore_inertial_definitions=True,
        )

        for i, label in enumerate(robot.body_label):
            if label.endswith("gripper_dummy") and robot.body_mass[i] == 0.0:
                robot.body_mass[i] = 1e-6
                robot.body_inv_mass[i] = 1.0 / 1e-6
                robot.body_inertia[i] = wp.mat33(np.eye(3, dtype=np.float32) * 1e-10)
                robot.body_inv_inertia[i] = wp.inverse(robot.body_inertia[i])

        for dof in range(robot.joint_dof_count):
            if dof in self.gripper_joint_dofs or dof in self.gripper_finger_dofs:
                continue
            robot.joint_target_ke[dof] = 120000.0
            robot.joint_target_kd[dof] = 12000.0
            robot.joint_effort_limit[dof] = 10000.0
            robot.joint_armature[dof] = 0.2

        for dof in self.gripper_joint_dofs:
            robot.joint_target_ke[dof] = 10000.0
            robot.joint_target_kd[dof] = 1000.0
            robot.joint_effort_limit[dof] = 100000.0
            robot.joint_armature[dof] = 0.5

        for dof in self.gripper_finger_dofs:
            robot.joint_target_ke[dof] = 500000.0
            robot.joint_target_kd[dof] = 10000.0
            robot.joint_effort_limit[dof] = 500000.0
            robot.joint_armature[dof] = 0.5

        robot.joint_q = _get_initial_joint_q()

        gravcomp_body = robot.custom_attributes["mujoco:gravcomp"]
        if gravcomp_body.values is None:
            gravcomp_body.values = {}
        for body_idx in range(1, robot.body_count):
            gravcomp_body.values[body_idx] = 1.0

        gravcomp_jnt = robot.custom_attributes["mujoco:jnt_actgravcomp"]
        if gravcomp_jnt.values is None:
            gravcomp_jnt.values = {}
        for dof_idx in range(robot.joint_dof_count):
            if dof_idx not in self.gripper_joint_dofs and dof_idx not in self.gripper_finger_dofs:
                gravcomp_jnt.values[dof_idx] = True

        return robot

    # ------------------------------------------------------------------
    # End effectors
    # ------------------------------------------------------------------

    def setup_end_effectors(self):
        ee_body_keys = [
            "rby1_dfactorybot/right_gripper_end_effector",
            "rby1_dfactorybot/left_gripper_end_effector",
            "rby1_dfactorybot/torso_hip_yaw",
        ]
        self.ee_weights = [1.0, 1.0, 50.0]

        self.ee_configs = []
        for key in ee_body_keys:
            idx = self.mujoco_model.body_label.index(key)
            self.ee_configs.append((key, idx))

        for key, idx in self.ee_configs:
            single_idx = self.single_robot_model.body_label.index(key)
            assert idx == single_idx, f"Body index mismatch for {key}: model={idx}, single={single_idx}"

    # ------------------------------------------------------------------
    # IK solver
    # ------------------------------------------------------------------

    def setup_ik(self):
        def _q2v4(q):
            return wp.vec4(q[0], q[1], q[2], q[3])

        body_q_np = self.state_0.body_q.numpy()

        self.ee_tfs = []
        self.pos_objs = []
        self.rot_objs = []

        for i, (_name, link_idx) in enumerate(self.ee_configs):
            w = self.ee_weights[i]
            tf = wp.transform(*body_q_np[link_idx])
            self.ee_tfs.append(tf)

            self.pos_objs.append(
                ik.IKObjectivePosition(
                    link_index=link_idx,
                    link_offset=wp.vec3(0.0, 0.0, 0.0),
                    target_positions=wp.array([wp.transform_get_translation(tf)], dtype=wp.vec3),
                    weight=w,
                )
            )

            self.rot_objs.append(
                ik.IKObjectiveRotation(
                    link_index=link_idx,
                    link_offset_rotation=wp.quat_identity(),
                    target_rotations=wp.array([_q2v4(wp.transform_get_rotation(tf))], dtype=wp.vec4),
                    weight=w,
                )
            )

        self.obj_joint_limits = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.single_robot_model.joint_limit_lower,
            joint_limit_upper=self.single_robot_model.joint_limit_upper,
            weight=10.0,
        )

        self.ik_joint_q = wp.array(
            self.single_robot_model.joint_q, shape=(1, self.single_robot_model.joint_coord_count)
        )

        objectives = [*self.pos_objs, *self.rot_objs, self.obj_joint_limits]
        self.ik_solver = ik.IKSolver(
            model=self.single_robot_model,
            n_problems=1,
            objectives=objectives,
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    # ------------------------------------------------------------------
    # Gripper targets
    # ------------------------------------------------------------------

    def setup_gripper_targets(self):
        self.gripper_limits_lower = self.single_robot_model.joint_limit_lower.numpy()[self.gripper_joint_dofs]
        self.gripper_limits_upper = self.single_robot_model.joint_limit_upper.numpy()[self.gripper_joint_dofs]

        init_q = _get_initial_joint_q()
        right_driver = init_q[self.gripper_joint_dofs[0]]
        left_driver = init_q[self.gripper_joint_dofs[1]]
        self.gripper_targets_list = [
            right_driver,
            left_driver,
            -right_driver,
            right_driver,
            -left_driver,
            left_driver,
        ]
        self.gripper_targets = wp.array(self.gripper_targets_list, dtype=wp.float32)

        gripper_mask_np = [-1] * self.single_robot_model.joint_dof_count
        for gripper_idx, dof_idx in enumerate(self.gripper_joint_dofs):
            gripper_mask_np[dof_idx] = gripper_idx
        follower_map = {14: 2, 15: 3, 24: 4, 25: 5}
        for dof_idx, target_idx in follower_map.items():
            gripper_mask_np[dof_idx] = target_idx
        self.gripper_mask = wp.array(gripper_mask_np, dtype=wp.int32)

    def _sync_gripper_followers(self):
        gripper_np = self.gripper_targets.numpy()
        right_driver = gripper_np[0]
        left_driver = gripper_np[1]
        gripper_np[2] = -right_driver
        gripper_np[3] = right_driver
        gripper_np[4] = -left_driver
        gripper_np[5] = left_driver
        wp.copy(self.gripper_targets, wp.array(gripper_np, dtype=wp.float32))

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def setup_state_machine(self):
        task_schedule_list = [
            TaskType.APPROACH,
            TaskType.ENGAGE,
            TaskType.GRASP,
            TaskType.HOLD_GRASP,
            TaskType.RETRACT,
            TaskType.SETTLE,
            TaskType.APPROACH_TARGET,
            TaskType.ALIGN_AXES,
            TaskType.VERIFY_ALIGN,
            TaskType.INSERT,
            TaskType.RELEASE,
            TaskType.WITHDRAW,
            TaskType.WAIT_AFTER_WITHDRAW,
            # --- Re-grasp and pull ---
            TaskType.REAPPROACH,
            TaskType.REENGAGE,
            TaskType.REGRASP,
            TaskType.PULL,
            TaskType.FINAL_RELEASE,
            TaskType.DONE,
        ]
        self.num_sm_tasks = len(task_schedule_list)
        self.sm_task_schedule = wp.array(task_schedule_list, dtype=wp.int32)

        # Time limits per schedule slot
        task_time_limits_list = [
            3.0,  # APPROACH
            1.5,  # ENGAGE
            0.5,  # GRASP
            0.5,  # HOLD_GRASP
            1.5,  # RETRACT
            0.3,  # SETTLE
            5.0,  # APPROACH_TARGET
            5.0,  # ALIGN_AXES
            2.0,  # VERIFY_ALIGN
            5.0,  # INSERT
            1.0,  # RELEASE
            2.0,  # WITHDRAW
            1.0,  # WAIT_AFTER_WITHDRAW
            # --- Re-grasp and pull ---
            2.0,  # REAPPROACH
            0.1,  # REENGAGE
            3.0,  # REGRASP
            4.0,  # PULL
            2.0,  # FINAL_RELEASE
            999.0,  # DONE
        ]
        self.sm_task_time_limits = wp.array(task_time_limits_list, dtype=float)

        self.sm_task_idx = wp.zeros(1, dtype=int)
        self.sm_task_time_elapsed = wp.zeros(1, dtype=float)

        # Snapshot EE transform
        body_q_np = self.state_0.body_q.numpy()
        _, ee_link_idx = self.ee_configs[0]
        init_tf = wp.transform(*body_q_np[ee_link_idx])
        self.sm_task_init_body_q = wp.array([init_tf], dtype=wp.transform)

        # Snapshot cable head transform from VBD
        vbd_body_q_np = self.vbd_state_0.body_q.numpy()
        cable_head_tf = wp.transform(*vbd_body_q_np[self.cable_head_body_idx])
        self.sm_task_plug_body_q_prev = wp.array([cable_head_tf], dtype=wp.transform)

        # Geometry parameters - shift grasp point toward the cable body
        # (away from the head mesh tip) so both fingers wrap the cable symmetrically.
        vbd_bq = self.vbd_state_0.body_q.numpy()
        head_pos = vbd_bq[self.cable_head_body_idx][:3]
        head_quat = vbd_bq[self.cable_head_body_idx][3:]  # (qx, qy, qz, qw)
        last_capsule_id = self.vbd_cable_all_body_ids[0][-1]
        capsule_pos = vbd_bq[last_capsule_id][:3]
        toward_cable_world = capsule_pos - head_pos
        toward_cable_world /= max(np.linalg.norm(toward_cable_world), 1e-8)
        toward_cable_local = _np_quat_rotate(_np_quat_inverse(head_quat), toward_cable_world)
        self._grasp_shift = 0.01  # 1 cm toward the cable body
        grasp_shift = self._grasp_shift
        self.sm_plug_grasp_offset = wp.vec3(
            float(toward_cable_local[0]) * grasp_shift,
            -self.capsule_radius + 0.002 + float(toward_cable_local[1]) * grasp_shift,
            float(toward_cable_local[2]) * grasp_shift,
        )
        self.sm_approach_offset = wp.vec3(0.0, 0.08, 0.0)
        self.sm_engage_offset = wp.vec3(0.01, 0.0, 0.0)
        self.sm_retract_vector = wp.vec3(0.0, 0.05, 0.0)
        self.sm_insert_offset = wp.vec3(0.0, 0.0, 0.02)
        self.sm_withdraw_offset = wp.vec3(-0.10, 0.0, 0.0)

        # Grasp orientation offset
        q_rz = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -wp.pi / 2.0)
        q_rx = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi / 2.0)
        self.sm_grasp_orientation_offset = q_rx * q_rz

        self.sm_gripper_open_value = float(self.gripper_limits_upper[0]) * 0.5
        self.sm_gripper_closed_value = 2.0 * 0.0036

        # Insertion geometry (used by APPROACH_TARGET / VERIFY_ALIGN)
        self._insertion_start_depth = 0.005
        insertion_dir = wp.quat_rotate(self.sm_socket_rot, wp.vec3(0.0, 0.0, 1.0))
        self._insertion_dir_np = np.array(
            [float(insertion_dir[i]) for i in range(3)],
            dtype=np.float64,
        )
        self._socket_pos_np = np.array(
            [float(self.sm_socket_pos[i]) for i in range(3)],
            dtype=np.float64,
        )
        self._socket_rot_np = np.array(
            [float(self.sm_socket_rot[i]) for i in range(4)],
            dtype=np.float64,
        )
        # Tip capsule -Z (connector end) should align with insertion_dir.
        flip_x = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._tip_insert_rot_np = _np_quat_multiply(self._socket_rot_np, flip_x)

        self._verify_align_retries = 0
        self._max_verify_retries = 2
        self._approach_ee_target_pos = np.zeros(3, dtype=np.float64)
        self._approach_ee_target_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

        # ALIGN_AXES state: per-axis coordinate-descent alignment
        self._align_axis_idx = 0
        self._align_phase = "probe_plus"
        self._align_best_cos = 0.0
        self._align_best_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self._align_ee_pos = np.zeros(3, dtype=np.float64)
        self._align_total_angle = 0.0
        self._align_delta_angle = 1.0 * np.pi / 180.0
        self._align_max_angle = 15.0 * np.pi / 180.0
        ins = self._insertion_dir_np
        arbitrary = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(ins, arbitrary)) > 0.9:
            arbitrary = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        perp1 = np.cross(ins, arbitrary)
        perp1 /= np.linalg.norm(perp1)
        perp2 = np.cross(ins, perp1)
        perp2 /= np.linalg.norm(perp2)
        self._align_axes = np.array([perp1, perp2], dtype=np.float64)
        self._align_target_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self._align_settle_frames = 0
        self._align_settle_wait = 5

        # VERIFY_ALIGN state: lateral correction targets
        self._verify_ee_target_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self._verify_lateral_gain = 1.0

        # Convergence thresholds - Python-controlled tasks use 999 to
        # prevent the kernel from auto-advancing them.
        pos_thresholds = [wp.vec3(0.001, 0.001, 0.002) for _ in range(self.num_sm_tasks)]
        pos_thresholds[TaskType.DONE] = wp.vec3(999.0, 999.0, 999.0)
        pos_thresholds[TaskType.ENGAGE] = wp.vec3(0.005, 0.005, 0.005)
        pos_thresholds[TaskType.GRASP] = wp.vec3(0.005, 0.005, 0.005)
        pos_thresholds[TaskType.HOLD_GRASP] = wp.vec3(999.0, 999.0, 999.0)
        pos_thresholds[TaskType.RETRACT] = wp.vec3(0.01, 0.01, 0.01)
        pos_thresholds[TaskType.SETTLE] = wp.vec3(999.0, 999.0, 999.0)
        pos_thresholds[TaskType.APPROACH_TARGET] = wp.vec3(999.0, 999.0, 999.0)
        pos_thresholds[TaskType.ALIGN_AXES] = wp.vec3(999.0, 999.0, 999.0)
        pos_thresholds[TaskType.VERIFY_ALIGN] = wp.vec3(999.0, 999.0, 999.0)
        pos_thresholds[TaskType.INSERT] = wp.vec3(999.0, 999.0, 999.0)
        pos_thresholds[TaskType.RELEASE] = wp.vec3(999.0, 999.0, 999.0)
        pos_thresholds[TaskType.WITHDRAW] = wp.vec3(0.01, 0.01, 0.01)
        for t in (
            TaskType.WAIT_AFTER_WITHDRAW,
            TaskType.REAPPROACH,
            TaskType.REENGAGE,
            TaskType.REGRASP,
            TaskType.PULL,
            TaskType.FINAL_RELEASE,
        ):
            pos_thresholds[t] = wp.vec3(999.0, 999.0, 999.0)
        rot_thresholds = [5.0 * wp.pi / 180.0] * self.num_sm_tasks
        rot_thresholds[TaskType.ENGAGE] = 10.0 * wp.pi / 180.0
        rot_thresholds[TaskType.HOLD_GRASP] = 999.0
        rot_thresholds[TaskType.RETRACT] = 10.0 * wp.pi / 180.0
        rot_thresholds[TaskType.SETTLE] = 999.0
        rot_thresholds[TaskType.APPROACH_TARGET] = 999.0
        rot_thresholds[TaskType.ALIGN_AXES] = 999.0
        rot_thresholds[TaskType.VERIFY_ALIGN] = 999.0
        rot_thresholds[TaskType.INSERT] = 999.0
        rot_thresholds[TaskType.RELEASE] = 999.0
        rot_thresholds[TaskType.WITHDRAW] = 10.0 * wp.pi / 180.0
        for t in (
            TaskType.WAIT_AFTER_WITHDRAW,
            TaskType.REAPPROACH,
            TaskType.REENGAGE,
            TaskType.REGRASP,
            TaskType.PULL,
            TaskType.FINAL_RELEASE,
        ):
            rot_thresholds[t] = 999.0

        self.sm_pos_error_threshold = wp.array(pos_thresholds, dtype=wp.vec3)
        self.sm_rot_error_threshold = wp.array(rot_thresholds, dtype=float)

        self.sm_ee_body_idx = self.ee_configs[0][1]

        self.sm_ee_pos_target = wp.zeros(1, dtype=wp.vec3)
        self.sm_ee_pos_interp = wp.zeros(1, dtype=wp.vec3)
        self.sm_ee_rot_target = wp.zeros(1, dtype=wp.vec4)
        self.sm_ee_rot_interp = wp.zeros(1, dtype=wp.vec4)
        self.sm_gripper_target = wp.zeros(1, dtype=wp.float32)

        self._sm_prev_task_idx = -1

    # ------------------------------------------------------------------
    # Tip tracking helpers
    # ------------------------------------------------------------------

    def _get_tip_pose(self):
        """Tip capsule world pose from VBD: (pos[3], quat[4]) as float64."""
        bq = self.vbd_state_0.body_q.numpy()
        tf = bq[self.tip_capsule_body_idx]
        return tf[:3].astype(np.float64), tf[3:].astype(np.float64)

    def _get_ee_pose(self):
        """MuJoCo EE world pose: (pos[3], quat[4]) as float64."""
        bq = self.state_0.body_q.numpy()
        tf = bq[self.sm_ee_body_idx]
        return tf[:3].astype(np.float64), tf[3:].astype(np.float64)

    def _write_sm_targets(self, pos, quat, t=1.0):
        """Write EE targets to GPU arrays, lerping from task-init pose by *t*."""
        pos32 = pos.astype(np.float32)
        q32 = quat.astype(np.float32)

        wp.copy(self.sm_ee_pos_target, wp.array([wp.vec3(*pos32)], dtype=wp.vec3))
        wp.copy(self.sm_ee_rot_target, wp.array([wp.vec4(q32[0], q32[1], q32[2], q32[3])], dtype=wp.vec4))

        if t >= 1.0:
            interp_pos = pos32
            interp_q = q32
        else:
            init_bq = self.sm_task_init_body_q.numpy()[0]
            init_pos = init_bq[:3].astype(np.float64)
            init_quat = init_bq[3:].astype(np.float64)
            interp_pos = (init_pos * (1.0 - t) + pos * t).astype(np.float32)
            interp_q = _np_quat_slerp(init_quat, quat, t).astype(np.float32)

        wp.copy(self.sm_ee_pos_interp, wp.array([wp.vec3(*interp_pos)], dtype=wp.vec3))
        interp_rot = wp.array([wp.vec4(interp_q[0], interp_q[1], interp_q[2], interp_q[3])], dtype=wp.vec4)
        wp.copy(self.sm_ee_rot_interp, interp_rot)
        wp.copy(self.sm_gripper_target, wp.array([self.sm_gripper_closed_value], dtype=wp.float32))

    def _advance_sm_to(self, new_schedule_idx):
        """Advance the SM task index and snapshot EE init pose."""
        wp.copy(self.sm_task_idx, wp.array([new_schedule_idx], dtype=int))
        self.sm_task_time_elapsed.zero_()
        body_q_np = self.state_0.body_q.numpy()
        init_tf = wp.transform(*body_q_np[self.sm_ee_body_idx])
        wp.copy(self.sm_task_init_body_q, wp.array([init_tf], dtype=wp.transform))

    def _init_verify_align(self):
        """Initialize VERIFY_ALIGN: begin iterative lateral correction."""
        _, ee_quat = self._get_ee_pose()
        self._verify_ee_target_quat = ee_quat.copy()

    def _compute_verify_align(self):
        """Each frame: re-measure lateral error and update EE target to compensate."""
        t_elapsed = float(self.sm_task_time_elapsed.numpy()[0]) + self.frame_dt
        wp.copy(self.sm_task_time_elapsed, wp.array([t_elapsed], dtype=float))

        ee_pos, _ = self._get_ee_pose()
        tip_pos, _ = self._get_tip_pose()
        ins = self._insertion_dir_np
        tip_target = self._socket_pos_np + self._insertion_start_depth * ins

        delta = tip_pos - tip_target
        lateral = delta - np.dot(delta, ins) * ins

        corrected_ee_pos = ee_pos - self._verify_lateral_gain * lateral
        self._write_sm_targets(corrected_ee_pos, self._verify_ee_target_quat, t=1.0)

    def _advance_verify_align(self, schedule_idx, t_elapsed):
        """Advance when laterally centered and aligned, or timeout."""
        tip_pos, tip_quat = self._get_tip_pose()
        tip_z = _np_quat_rotate(tip_quat, np.array([0.0, 0.0, 1.0]))
        cosine = np.dot(tip_z, self._insertion_dir_np)
        tip_target = self._socket_pos_np + self._insertion_start_depth * self._insertion_dir_np
        delta = tip_pos - tip_target
        lateral = delta - np.dot(delta, self._insertion_dir_np) * self._insertion_dir_np
        lateral_err = np.linalg.norm(lateral)

        if t_elapsed >= 0.5 and lateral_err < 0.010 and cosine < -0.90:
            self._verify_align_retries = 0
            self._advance_sm_to(schedule_idx + 1)
        elif t_elapsed >= 2.0:
            self._verify_align_retries += 1
            if self._verify_align_retries < self._max_verify_retries:
                self._advance_sm_to(schedule_idx - 1)
            else:
                self._advance_sm_to(schedule_idx + 1)

    # ------------------------------------------------------------------
    # INSERT: Python-controlled insertion with capsule tracking
    # ------------------------------------------------------------------

    def _init_insert(self):
        """Initialize INSERT: lock rotation, measure actual depth, set up insertion ramp."""
        ee_pos, ee_quat = self._get_ee_pose()
        self._insert_ee_quat = ee_quat.copy()
        self._insert_ee_start_pos = ee_pos.copy()
        self._insert_final_depth = self.insert_final_depth
        self._insert_snap_depth = max(0.0, self.insert_final_depth - self.insert_snap_margin)
        self._insert_duration = 4.0
        self._insert_lateral_gain = 0.5
        self._insert_lateral_integral_gain = 5.0
        self._insert_lateral_integral = np.zeros(3, dtype=np.float64)

        self._insert_orient_gain = 0.2
        self._insert_cos_pause_threshold = -0.95
        self._insert_cos_resume_threshold = -0.97
        self._insert_depth_paused = False

        self._insert_start_depth = self._insertion_start_depth
        self._insert_command_depth = self._insert_start_depth
        self._insert_t_paused = 0.0

    def _compute_insert(self):
        """Per-frame: depth ramp + lateral correction + orientation correction.

        If alignment degrades past cos_pause_threshold, depth ramp freezes until
        the orientation corrector brings it back past cos_resume_threshold.
        """
        t_elapsed = float(self.sm_task_time_elapsed.numpy()[0]) + self.frame_dt
        wp.copy(self.sm_task_time_elapsed, wp.array([t_elapsed], dtype=float))

        ins = self._insertion_dir_np

        # --- Orientation correction: nudge EE rotation so capsule Z -> -ins ---
        tip_pos, tip_quat = self._get_tip_pose()
        tip_z = _np_quat_rotate(tip_quat, np.array([0.0, 0.0, 1.0]))
        cos_val = np.dot(tip_z, ins)

        desired_z = -ins
        rot_axis = np.cross(tip_z, desired_z)
        sin_err = np.linalg.norm(rot_axis)
        if sin_err > 1e-6:
            rot_axis /= sin_err
            angle = np.arcsin(np.clip(sin_err, -1.0, 1.0)) * self._insert_orient_gain
            half = 0.5 * angle
            s, c = np.sin(half), np.cos(half)
            dq = np.array([rot_axis[0] * s, rot_axis[1] * s, rot_axis[2] * s, c], dtype=np.float64)
            self._insert_ee_quat = _np_quat_multiply(dq, self._insert_ee_quat)

        # --- Pause/resume depth ramp based on alignment ---
        if cos_val > self._insert_cos_pause_threshold:
            if not self._insert_depth_paused:
                self._insert_depth_paused = True
        elif cos_val < self._insert_cos_resume_threshold and self._insert_depth_paused:
            self._insert_depth_paused = False

        # --- Depth ramp (frozen while paused) ---
        if self._insert_depth_paused:
            self._insert_t_paused += self.frame_dt
        effective_t = t_elapsed - self._insert_t_paused
        t_lin = min(1.0, max(0.0, effective_t / self._insert_duration))
        t = t_lin * t_lin * (3.0 - 2.0 * t_lin)

        depth_travel = t * (self._insert_final_depth - self._insert_start_depth)
        command_depth = self._insert_start_depth + depth_travel
        self._insert_command_depth = command_depth

        ee_target = self._insert_ee_start_pos + depth_travel * ins

        # --- Lateral correction: keep capsule on insertion axis ---
        tip_target = self._socket_pos_np + command_depth * ins
        delta = tip_pos - tip_target
        lateral = delta - np.dot(delta, ins) * ins
        self._insert_lateral_integral += lateral * self.frame_dt
        corrected_ee_pos = (
            ee_target
            - self._insert_lateral_gain * lateral
            - self._insert_lateral_integral_gain * self._insert_lateral_integral
        )

        self._write_sm_targets(corrected_ee_pos, self._insert_ee_quat, t=1.0)

    def _pin_capsule_at_target(self):
        """Activate dormant fixed joints to lock the capsule at the current pose.

        Force-only snap: anchor bodies are placed at the target (centerline,
        corrected orientation) and the fixed joint penalty forces gradually
        pull the capsule into place.  No teleport - avoids discontinuities
        in bend joint constraints.
        """
        anchor_idx = self._insert_anchor_body
        plug_anchor_idx = self._plug_anchor_body

        bq_before = self.vbd_state_0.body_q.numpy()

        # Pin the head body at the final insertion depth on the socket centerline,
        # then derive the capsule body pose from the cable<->head fixed joint.
        target_quat = self._tip_insert_rot_np
        ins = self._insertion_dir_np

        j = self._capsule_plug_joint_idx
        X_p = self.vbd_model.joint_X_p.numpy()[j].astype(np.float64)
        X_c = self.vbd_model.joint_X_c.numpy()[j].astype(np.float64)

        joint_rot = _np_quat_multiply(X_p[3:], _np_quat_inverse(X_c[3:]))
        head_quat_target = _np_quat_multiply(target_quat, joint_rot)

        head_target_pos = self._socket_pos_np + self.insert_final_depth * ins
        current_head_z = float(bq_before[self.cable_head_body_idx][2])
        if head_target_pos[2] < current_head_z and abs(ins[2]) > 1e-6:
            clamped_depth = (current_head_z - self._socket_pos_np[2]) / ins[2]
            head_target_pos = self._socket_pos_np + clamped_depth * ins

        head_tf64 = _np_tf7(head_target_pos, head_quat_target)
        head_tf = head_tf64.astype(np.float32)

        # Derive capsule pose from head pose: capsule = head * X_c * X_p^-1.
        capsule_tf64 = _np_tf_multiply(_np_tf_multiply(head_tf64, X_c), _np_tf_inverse(X_p))

        capsule_origin_target = capsule_tf64[:3]
        body_z_world = _np_quat_rotate(target_quat, np.array([0.0, 0.0, 1.0]))
        com_target = capsule_origin_target + self.tip_half_height * body_z_world

        # Place anchor bodies at target pose in all state arrays.
        anchor_tf = np.array([*com_target.astype(np.float32), *target_quat.astype(np.float32)], dtype=np.float32)
        for arr in (
            self.vbd_model.body_q,
            self.vbd_state_0.body_q,
            self.vbd_state_1.body_q,
            self.vbd_solver.body_q_prev,
        ):
            arr_np = arr.numpy()
            arr_np[anchor_idx] = anchor_tf
            if plug_anchor_idx is not None:
                arr_np[plug_anchor_idx] = head_tf
            wp.copy(arr, wp.array(arr_np, dtype=wp.transform, device=arr.device))

        # Activate dormant fixed joints with snap stiffness.
        solver = self.vbd_solver
        dev = solver.joint_penalty_k_max.device

        fixed_slots = [self._insert_joint_constraint_start]
        if self._plug_joint_constraint_start is not None:
            fixed_slots.append(self._plug_joint_constraint_start)

        for arr_name in ("joint_penalty_k", "joint_penalty_k_min"):
            arr = getattr(solver, arr_name)
            tmp = arr.numpy()
            for cs in fixed_slots:
                tmp[cs] = self.snap_k_lin
                tmp[cs + 1] = self.snap_k_ang
            wp.copy(arr, wp.array(tmp, dtype=float, device=dev))

        tmp = solver.joint_penalty_k_max.numpy()
        for cs in fixed_slots:
            tmp[cs] = self.snap_k_lin_max
            tmp[cs + 1] = self.snap_k_ang_max
        wp.copy(solver.joint_penalty_k_max, wp.array(tmp, dtype=float, device=dev))

        kd_np = solver.joint_penalty_kd.numpy()
        for cs in fixed_slots:
            kd_np[cs] = self.snap_kd_lin
            kd_np[cs + 1] = self.snap_kd_ang
        wp.copy(solver.joint_penalty_kd, wp.array(kd_np, dtype=float, device=dev))

        self._pinned_depth = float(np.dot(capsule_origin_target - self._socket_pos_np, ins))
        head_depth = float(np.dot(head_target_pos - self._socket_pos_np, ins))
        if self.verbose:
            print(
                f"  [PIN] head_depth={head_depth * 1000:.2f}mm "
                f"tip_depth={self._pinned_depth * 1000:.2f}mm  anchor={com_target}"
                f"  k_lin={self.snap_k_lin}  k_ang={self.snap_k_ang}"
            )

    def _unpin_capsule(self):
        """Deactivate snap fixed joints by zeroing stiffness (reverse of _pin_capsule_at_target)."""
        solver = self.vbd_solver
        dev = solver.joint_penalty_k_max.device

        fixed_slots = [self._insert_joint_constraint_start]
        if self._plug_joint_constraint_start is not None:
            fixed_slots.append(self._plug_joint_constraint_start)

        for arr_name in ("joint_penalty_k_max", "joint_penalty_k_min", "joint_penalty_k", "joint_penalty_kd"):
            arr = getattr(solver, arr_name)
            tmp = arr.numpy()
            for cs in fixed_slots:
                tmp[cs] = 0.0
                tmp[cs + 1] = 0.0
            wp.copy(arr, wp.array(tmp, dtype=float, device=dev))

        if self.verbose:
            tip_pos, _ = self._get_tip_pose()
            depth_now = np.dot(tip_pos - self._socket_pos_np, self._insertion_dir_np)
            print(f"  [UNPIN] fixed joints deactivated  depth={depth_now * 1000:.2f}mm")

    def _advance_insert(self, schedule_idx):
        """Advance after insertion ramp completes."""
        t_elapsed = float(self.sm_task_time_elapsed.numpy()[0])

        tip_pos, _ = self._get_tip_pose()
        ins = self._insertion_dir_np
        axial_depth = np.dot(tip_pos - self._socket_pos_np, ins)
        command_depth = self._insert_command_depth

        reached_depth = axial_depth >= self._insert_snap_depth
        reached_command = command_depth >= self._insert_snap_depth
        timed_out = t_elapsed >= self._insert_duration

        if reached_depth and reached_command:
            self._pin_capsule_at_target()
            self._advance_sm_to(schedule_idx + 1)
        elif timed_out:
            self._advance_sm_to(schedule_idx + 1)

    # ------------------------------------------------------------------
    # Re-grasp and pull helpers
    # ------------------------------------------------------------------

    def _init_reapproach(self):
        """Reverse the withdraw motion to re-grasp the cable.

        Moves back by -withdraw_offset and applies minimal rotation corrections
        so that fingers are level (local X aligns with world Z, local Y stays
        horizontal).
        """
        ee_pos, ee_quat = self._get_ee_pose()
        withdraw = np.array([float(self.sm_withdraw_offset[i]) for i in range(3)], dtype=np.float64)

        self._reapproach_target_pos = ee_pos - withdraw
        self._reapproach_target_pos[2] += self.regrasp_z_offset
        # Minimal rotation so local X aligns with world Z (up or down,
        # whichever is closer). This makes both local Y and local Z
        # fully horizontal, so finger tips and bases are at the same Z.
        x_w = _np_quat_rotate(ee_quat, np.array([1.0, 0.0, 0.0]))
        target_x = np.array([0.0, 0.0, 1.0]) if x_w[2] >= 0 else np.array([0.0, 0.0, -1.0])
        axis = np.cross(x_w, target_x)
        sin_a = np.linalg.norm(axis)
        cos_a = np.dot(x_w, target_x)
        if sin_a > 1e-8:
            axis /= sin_a
            angle = np.arctan2(sin_a, cos_a)
            ha = 0.5 * angle
            s = np.sin(ha)
            dq = np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(ha)], dtype=np.float64)
            regrasp_quat = _np_quat_multiply(dq, ee_quat)
        else:
            regrasp_quat = ee_quat.copy()
        regrasp_quat /= np.linalg.norm(regrasp_quat)
        # Second rotation around world Z to snap local Y to world Y.
        y_w = _np_quat_rotate(regrasp_quat, np.array([0.0, 1.0, 0.0]))
        yaw_correction = -np.arctan2(y_w[0], y_w[1])
        hy = 0.5 * yaw_correction
        dq_yaw = np.array([0.0, 0.0, np.sin(hy), np.cos(hy)], dtype=np.float64)
        regrasp_quat = _np_quat_multiply(dq_yaw, regrasp_quat)
        regrasp_quat /= np.linalg.norm(regrasp_quat)
        # Third rotation around world X to zero out any residual Z in local Y.
        y_w2 = _np_quat_rotate(regrasp_quat, np.array([0.0, 1.0, 0.0]))
        pitch_correction = -np.arctan2(y_w2[2], y_w2[1])
        hp = 0.5 * pitch_correction
        dq_pitch = np.array([np.sin(hp), 0.0, 0.0, np.cos(hp)], dtype=np.float64)
        regrasp_quat = _np_quat_multiply(dq_pitch, regrasp_quat)
        regrasp_quat /= np.linalg.norm(regrasp_quat)
        self._reapproach_target_quat = regrasp_quat.copy()
        self._reengage_target_pos = self._reapproach_target_pos.copy()
        self._reengage_target_quat = regrasp_quat.copy()

        if self.verbose:
            print(
                f"  [REAPPROACH] from [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}] "
                f"-> to [{self._reapproach_target_pos[0]:.3f}, "
                f"{self._reapproach_target_pos[1]:.3f}, {self._reapproach_target_pos[2]:.3f}] m"
            )
            print(
                f"    orientation corrections: yaw={np.degrees(yaw_correction):.1f}deg, "
                f"pitch={np.degrees(pitch_correction):.1f}deg"
            )

    def _compute_reapproach(self):
        """Smoothstep EE toward re-approach target with gripper open."""
        t_elapsed = float(self.sm_task_time_elapsed.numpy()[0]) + self.frame_dt
        wp.copy(self.sm_task_time_elapsed, wp.array([t_elapsed], dtype=float))
        t_lin = min(1.0, t_elapsed / 2.0)
        t = t_lin * t_lin * (3.0 - 2.0 * t_lin)
        self._write_sm_targets(self._reapproach_target_pos, self._reapproach_target_quat, t=t)
        wp.copy(self.sm_gripper_target, wp.array([self.sm_gripper_open_value], dtype=wp.float32))

    def _compute_reengage(self):
        """Smoothstep EE toward engage position (close to cable head)."""
        t_elapsed = float(self.sm_task_time_elapsed.numpy()[0]) + self.frame_dt
        wp.copy(self.sm_task_time_elapsed, wp.array([t_elapsed], dtype=float))
        t_lin = min(1.0, t_elapsed / 1.5)
        t = t_lin * t_lin * (3.0 - 2.0 * t_lin)
        self._write_sm_targets(self._reengage_target_pos, self._reengage_target_quat, t=t)
        wp.copy(self.sm_gripper_target, wp.array([self.sm_gripper_open_value], dtype=wp.float32))

    def _compute_regrasp(self):
        """Hold position, close gripper via smoothstep."""
        t_elapsed = float(self.sm_task_time_elapsed.numpy()[0]) + self.frame_dt
        wp.copy(self.sm_task_time_elapsed, wp.array([t_elapsed], dtype=float))
        t_lin = min(1.0, t_elapsed / 1.0)
        t = t_lin * t_lin * (3.0 - 2.0 * t_lin)
        ee_pos, ee_quat = self._get_ee_pose()
        self._write_sm_targets(ee_pos, ee_quat, t=1.0)
        gripper_val = self.sm_gripper_open_value + (self.sm_gripper_closed_value - self.sm_gripper_open_value) * t
        wp.copy(self.sm_gripper_target, wp.array([gripper_val], dtype=wp.float32))

    def _init_pull(self):
        """2/3: -insertion_dir. 1/3: blend from -insertion_dir toward -X."""
        ee_pos, ee_quat = self._get_ee_pose()
        self._pull_start = ee_pos.copy()
        self._pull_ee_quat = ee_quat.copy()
        pull_dir = -self._insertion_dir_np
        self._pull_mid = ee_pos + (2.0 / 3.0) * self.pull_distance * pull_dir
        blend_dir = 0.5 * (pull_dir + np.array([-1.0, 0.0, 0.0]))
        norm = np.linalg.norm(blend_dir)
        if norm > 1e-8:
            blend_dir /= norm
        self._pull_end = self._pull_mid + (1.0 / 3.0) * self.pull_distance * blend_dir
        self._pull_unpinned = False
        if self.verbose:
            print(f"  [PULL] phase1 (2/3): {ee_pos} -> {self._pull_mid} (-ins_dir)")
            print(f"  [PULL] phase2 (1/3): {self._pull_mid} -> {self._pull_end} (blend -ins_dir & -X)")
            print(f"  [PULL] distance={self.pull_distance * 1000:.1f}mm  duration={self.pull_duration:.1f}s")

    def _compute_pull(self):
        """2/3: -insertion_dir. 1/3: blend toward -X."""
        t_elapsed = float(self.sm_task_time_elapsed.numpy()[0]) + self.frame_dt
        wp.copy(self.sm_task_time_elapsed, wp.array([t_elapsed], dtype=float))
        t_lin = min(1.0, t_elapsed / self.pull_duration)
        t = t_lin * t_lin * (3.0 - 2.0 * t_lin)
        if t <= 2.0 / 3.0:
            u = t / (2.0 / 3.0)
            target_pos = self._pull_start + u * (self._pull_mid - self._pull_start)
        else:
            u = (t - 2.0 / 3.0) / (1.0 / 3.0)
            target_pos = self._pull_mid + u * (self._pull_end - self._pull_mid)
        self._write_sm_targets(target_pos, self._pull_ee_quat, t=1.0)
        wp.copy(self.sm_gripper_target, wp.array([self.sm_gripper_closed_value], dtype=wp.float32))

    def _advance_pull(self, schedule_idx):
        """Mirror of _advance_insert: unpin when depth drops below snap_depth."""
        t_elapsed = float(self.sm_task_time_elapsed.numpy()[0])
        tip_pos, _ = self._get_tip_pose()
        ins = self._insertion_dir_np
        axial_depth = np.dot(tip_pos - self._socket_pos_np, ins)
        unpin_threshold = getattr(self, "_pinned_depth", self._insert_snap_depth)
        if not self._pull_unpinned and axial_depth < unpin_threshold:
            self._unpin_capsule()
            self._pull_unpinned = True
        time_limit = float(self.sm_task_time_limits.numpy()[schedule_idx])
        if t_elapsed >= time_limit and schedule_idx < self.num_sm_tasks - 1:
            self._advance_sm_to(schedule_idx + 1)

    def _init_final_release(self):
        """Snapshot EE pose for final release + withdraw."""
        ee_pos, ee_quat = self._get_ee_pose()
        self._final_release_start = ee_pos.copy()
        self._final_release_quat = ee_quat.copy()
        withdraw = np.array([float(self.sm_withdraw_offset[i]) for i in range(3)], dtype=np.float64)
        self._final_release_target = ee_pos + withdraw
        if self.verbose:
            print(f"  [FINAL_RELEASE] {ee_pos} -> {self._final_release_target} (open + withdraw)")

    def _compute_final_release(self):
        """Open gripper while moving in -X."""
        t_elapsed = float(self.sm_task_time_elapsed.numpy()[0]) + self.frame_dt
        wp.copy(self.sm_task_time_elapsed, wp.array([t_elapsed], dtype=float))
        t_lin = min(1.0, t_elapsed / 2.0)
        t = t_lin * t_lin * (3.0 - 2.0 * t_lin)
        target_pos = self._final_release_start + t * (self._final_release_target - self._final_release_start)
        self._write_sm_targets(target_pos, self._final_release_quat, t=1.0)
        gripper_val = self.sm_gripper_closed_value + (self.sm_gripper_open_value - self.sm_gripper_closed_value) * t
        wp.copy(self.sm_gripper_target, wp.array([gripper_val], dtype=wp.float32))

    def _advance_timed(self, schedule_idx):
        """Advance to next task when time limit is reached."""
        t_elapsed = float(self.sm_task_time_elapsed.numpy()[0])
        time_limit = float(self.sm_task_time_limits.numpy()[schedule_idx])
        if t_elapsed >= time_limit and schedule_idx < self.num_sm_tasks - 1:
            self._advance_sm_to(schedule_idx + 1)

    # ------------------------------------------------------------------
    # Main SM driver
    # ------------------------------------------------------------------

    def _compute_approach_ee_target(self):
        """Set EE target: socket_pos + socket_rot * grasp_offset."""
        socket_pos = self._socket_pos_np
        socket_rot = self._socket_rot_np
        grasp_rot = np.array(
            [float(self.sm_grasp_orientation_offset[i]) for i in range(4)],
            dtype=np.float64,
        )
        target_quat = _np_quat_multiply(socket_rot, grasp_rot)

        self._approach_ee_target_pos = socket_pos + self._insertion_start_depth * self._insertion_dir_np
        self._approach_ee_target_quat = target_quat

    def _compute_approach_target(self):
        """Smoothstep-lerp EE toward the precomputed target (same as v3 kernel)."""
        t_elapsed = float(self.sm_task_time_elapsed.numpy()[0]) + self.frame_dt
        wp.copy(self.sm_task_time_elapsed, wp.array([t_elapsed], dtype=float))

        t_lin = min(1.0, t_elapsed / 5.0)
        t = t_lin * t_lin * (3.0 - 2.0 * t_lin)

        self._write_sm_targets(
            self._approach_ee_target_pos,
            self._approach_ee_target_quat,
            t=t,
        )

    def _init_align_axes(self):
        """Initialize ALIGN_AXES state: snapshot current EE pose, measure baseline alignment."""
        ee_pos, ee_quat = self._get_ee_pose()
        self._align_ee_pos = ee_pos.copy()
        self._align_axis_idx = 0
        self._align_phase = "probe_plus"
        self._align_total_angle = 0.0
        self._align_settle_frames = 0

        self._align_target_quat = ee_quat.copy()
        self._align_best_quat = ee_quat.copy()

        _, tip_quat = self._get_tip_pose()
        tip_z = _np_quat_rotate(tip_quat, np.array([0.0, 0.0, 1.0]))
        self._align_best_cos = np.dot(tip_z, self._insertion_dir_np)

    def _align_axes_step(self):
        """Per-frame coordinate-descent: rotate EE around axes perpendicular to insertion_dir."""
        t_elapsed = float(self.sm_task_time_elapsed.numpy()[0]) + self.frame_dt
        wp.copy(self.sm_task_time_elapsed, wp.array([t_elapsed], dtype=float))

        if self._align_phase == "done":
            self._write_sm_targets(self._align_ee_pos, self._align_best_quat, t=1.0)
            return

        if self._align_settle_frames > 0:
            self._align_settle_frames -= 1
            self._write_sm_targets(self._align_ee_pos, self._align_target_quat, t=1.0)
            return

        _, tip_quat = self._get_tip_pose()
        tip_z = _np_quat_rotate(tip_quat, np.array([0.0, 0.0, 1.0]))
        cos_now = np.dot(tip_z, self._insertion_dir_np)

        n_axes = len(self._align_axes)

        if self._align_phase == "probe_plus":
            if cos_now < self._align_best_cos:
                self._align_best_cos = cos_now
                self._align_best_quat = self._align_target_quat.copy()

            if cos_now > self._align_best_cos + 1e-4 or self._align_total_angle >= self._align_max_angle:
                self._align_phase = "probe_minus"
                self._align_total_angle = 0.0
                self._align_target_quat = self._align_best_quat.copy()
                self._write_sm_targets(self._align_ee_pos, self._align_best_quat, t=1.0)
                self._align_settle_frames = self._align_settle_wait
                return

            axis_vec = self._align_axes[self._align_axis_idx]
            half = 0.5 * self._align_delta_angle
            s, c = np.sin(half), np.cos(half)
            dq = np.array([axis_vec[0] * s, axis_vec[1] * s, axis_vec[2] * s, c], dtype=np.float64)
            self._align_target_quat = _np_quat_multiply(dq, self._align_target_quat)
            self._align_total_angle += self._align_delta_angle
            self._write_sm_targets(self._align_ee_pos, self._align_target_quat, t=1.0)
            self._align_settle_frames = self._align_settle_wait

        elif self._align_phase == "probe_minus":
            if cos_now < self._align_best_cos:
                self._align_best_cos = cos_now
                self._align_best_quat = self._align_target_quat.copy()

            if cos_now > self._align_best_cos + 1e-4 or self._align_total_angle >= self._align_max_angle:
                self._align_target_quat = self._align_best_quat.copy()
                self._write_sm_targets(self._align_ee_pos, self._align_best_quat, t=1.0)
                self._align_settle_frames = self._align_settle_wait

                self._align_axis_idx += 1
                if self._align_axis_idx >= n_axes:
                    self._align_phase = "done"
                else:
                    self._align_phase = "probe_plus"
                    self._align_total_angle = 0.0
                return

            axis_vec = self._align_axes[self._align_axis_idx]
            half = 0.5 * (-self._align_delta_angle)
            s, c = np.sin(half), np.cos(half)
            dq = np.array([axis_vec[0] * s, axis_vec[1] * s, axis_vec[2] * s, c], dtype=np.float64)
            self._align_target_quat = _np_quat_multiply(dq, self._align_target_quat)
            self._align_total_angle += self._align_delta_angle
            self._write_sm_targets(self._align_ee_pos, self._align_target_quat, t=1.0)
            self._align_settle_frames = self._align_settle_wait

    def _advance_align_axes(self, schedule_idx):
        """Advance to next state once all perpendicular axes are swept (or timeout)."""
        t_elapsed = float(self.sm_task_time_elapsed.numpy()[0])
        if self._align_phase == "done" or t_elapsed >= 5.0:
            tip_pos, tip_quat = self._get_tip_pose()
            tip_z = _np_quat_rotate(tip_quat, np.array([0.0, 0.0, 1.0]))
            ins = self._insertion_dir_np
            cos_val = np.dot(tip_z, ins)

            tip_target = self._socket_pos_np + self._insertion_start_depth * ins
            delta = tip_pos - tip_target
            lateral = delta - np.dot(delta, ins) * ins
            lateral_err = np.linalg.norm(lateral)

            if self.verbose:
                print("  [ALIGN_AXES] final diagnostics:")
                print(f"    capsule_z = [{tip_z[0]:.4f}, {tip_z[1]:.4f}, {tip_z[2]:.4f}]")
                print(f"    insert_dir= [{ins[0]:.4f}, {ins[1]:.4f}, {ins[2]:.4f}]")
                print(f"    cos(capsule_z, insert_dir) = {cos_val:.4f}")
                print(f"    lateral_err = {lateral_err * 1000:.1f}mm")

            self._advance_sm_to(schedule_idx + 1)

    def _advance_approach_target(self, schedule_idx):
        """Advance after smoothstep completes (5s)."""
        t_elapsed = float(self.sm_task_time_elapsed.numpy()[0])
        if t_elapsed >= 5.0:
            self._advance_sm_to(schedule_idx + 1)

    def set_sm_targets(self):
        idx = int(self.sm_task_idx.numpy()[0])
        schedule = self.sm_task_schedule.numpy()
        task = int(schedule[idx])

        # --- 1. Compute EE targets ---
        if task == TaskType.APPROACH_TARGET:
            self._compute_approach_target()
        elif task == TaskType.ALIGN_AXES:
            self._align_axes_step()
        elif task == TaskType.VERIFY_ALIGN:
            self._compute_verify_align()
        elif task == TaskType.INSERT:
            self._compute_insert()
        elif task == TaskType.WAIT_AFTER_WITHDRAW:
            t_elapsed = float(self.sm_task_time_elapsed.numpy()[0]) + self.frame_dt
            wp.copy(self.sm_task_time_elapsed, wp.array([t_elapsed], dtype=float))
            ee_pos, ee_quat = self._get_ee_pose()
            self._write_sm_targets(ee_pos, ee_quat, t=1.0)
            wp.copy(self.sm_gripper_target, wp.array([self.sm_gripper_open_value], dtype=wp.float32))
        elif task == TaskType.REAPPROACH:
            self._compute_reapproach()
        elif task == TaskType.REENGAGE:
            self._compute_reengage()
        elif task == TaskType.REGRASP:
            self._compute_regrasp()
        elif task == TaskType.PULL:
            self._compute_pull()
        elif task == TaskType.FINAL_RELEASE:
            self._compute_final_release()
        else:
            wp.launch(
                set_target_pose_kernel,
                dim=1,
                inputs=[
                    self.sm_task_schedule,
                    self.sm_task_time_limits,
                    self.sm_task_idx,
                    self.sm_task_time_elapsed,
                    self.frame_dt,
                    self.sm_plug_grasp_offset,
                    self.sm_approach_offset,
                    self.sm_engage_offset,
                    self.sm_retract_vector,
                    self.sm_insert_offset,
                    self.sm_withdraw_offset,
                    self.sm_socket_pos,
                    self.sm_socket_rot,
                    self.sm_grasp_orientation_offset,
                    self.sm_gripper_open_value,
                    self.sm_gripper_closed_value,
                    self.sm_task_init_body_q,
                    self.sm_task_plug_body_q_prev,
                ],
                outputs=[
                    self.sm_ee_pos_target,
                    self.sm_ee_pos_interp,
                    self.sm_ee_rot_target,
                    self.sm_ee_rot_interp,
                    self.sm_gripper_target,
                ],
            )

        # --- 2. IK ---
        self.pos_objs[0].set_target_positions(self.sm_ee_pos_interp[0:1])
        self.rot_objs[0].set_target_rotations(self.sm_ee_rot_interp[0:1])

        for i in (1, 2):
            tf = self.ee_tfs[i]
            self.pos_objs[i].set_target_position(0, wp.transform_get_translation(tf))
            q = wp.transform_get_rotation(tf)
            self.rot_objs[i].set_target_rotation(0, wp.vec4(q[0], q[1], q[2], q[3]))

        if self.graph_ik is not None:
            wp.capture_launch(self.graph_ik)
        else:
            self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iters)

        # --- 3. Gripper ---
        gripper_np = self.gripper_targets.numpy()
        gripper_np[0] = self.sm_gripper_target.numpy()[0]
        wp.copy(self.gripper_targets, wp.array(gripper_np, dtype=wp.float32))
        self._sync_gripper_followers()

        # --- 4. Merge IK + grippers ---
        wp.launch(
            merge_ik_with_gripper_targets,
            dim=self.single_robot_model.joint_dof_count,
            inputs=[
                self.ik_joint_q.flatten(),
                self.gripper_targets,
                self.gripper_mask,
                self.single_robot_model.joint_dof_count,
            ],
            outputs=[self.joint_target_pos],
        )
        wp.copy(self.control.joint_target_pos, self.joint_target_pos)

        # --- 5. Advance ---
        if task == TaskType.APPROACH_TARGET:
            self._advance_approach_target(idx)
        elif task == TaskType.ALIGN_AXES:
            self._advance_align_axes(idx)
        elif task == TaskType.VERIFY_ALIGN:
            t_elapsed = float(self.sm_task_time_elapsed.numpy()[0])
            self._advance_verify_align(idx, t_elapsed)
        elif task == TaskType.INSERT:
            self._advance_insert(idx)
        elif task == TaskType.PULL:
            self._advance_pull(idx)
        elif task in (
            TaskType.WAIT_AFTER_WITHDRAW,
            TaskType.REAPPROACH,
            TaskType.REENGAGE,
            TaskType.REGRASP,
            TaskType.FINAL_RELEASE,
        ):
            self._advance_timed(idx)
        else:
            wp.launch(
                advance_task_kernel,
                dim=1,
                inputs=[
                    self.sm_task_time_limits,
                    self.sm_ee_pos_target,
                    self.sm_ee_rot_target,
                    self.state_0.body_q,
                    self.sm_ee_body_idx,
                    self.sm_pos_error_threshold,
                    self.sm_rot_error_threshold,
                ],
                outputs=[
                    self.sm_task_idx,
                    self.sm_task_time_elapsed,
                    self.sm_task_init_body_q,
                ],
            )

        # --- 6. Handle transitions ---
        new_idx = int(self.sm_task_idx.numpy()[0])
        new_task = int(schedule[new_idx]) if new_idx < len(schedule) else TaskType.DONE

        if new_idx != self._sm_prev_task_idx:
            prev_task = int(schedule[self._sm_prev_task_idx]) if self._sm_prev_task_idx >= 0 else -1
            prev_name = TaskType(prev_task).name if prev_task >= 0 else "INIT"
            self._sm_prev_task_idx = new_idx
            if self.verbose:
                print(f"[SM] {prev_name} -> {TaskType(new_task).name}")

            if new_task == TaskType.APPROACH_TARGET:
                self._compute_approach_ee_target()

            if new_task == TaskType.ALIGN_AXES:
                self._init_align_axes()

            if new_task == TaskType.VERIFY_ALIGN:
                self._init_verify_align()

            if new_task == TaskType.INSERT:
                self._init_insert()

            if new_task == TaskType.ENGAGE:
                vbd_body_q_np = self.vbd_state_0.body_q.numpy()
                cable_head_tf = wp.transform(*vbd_body_q_np[self.cable_head_body_idx])
                wp.copy(self.sm_task_plug_body_q_prev, wp.array([cable_head_tf], dtype=wp.transform))

                head_pos = vbd_body_q_np[self.cable_head_body_idx][:3]
                head_quat = vbd_body_q_np[self.cable_head_body_idx][3:]
                last_cap_id = self.vbd_cable_all_body_ids[0][-1]
                cap_pos = vbd_body_q_np[last_cap_id][:3]
                tcw = cap_pos - head_pos
                tcw /= max(np.linalg.norm(tcw), 1e-8)
                tcl = _np_quat_rotate(_np_quat_inverse(head_quat), tcw)
                gs = self._grasp_shift
                self.sm_plug_grasp_offset = wp.vec3(
                    float(tcl[0]) * gs,
                    -self.capsule_radius + 0.002 + float(tcl[1]) * gs,
                    float(tcl[2]) * gs,
                )

            if new_task == TaskType.REAPPROACH:
                self._init_reapproach()

            if new_task == TaskType.PULL:
                self._init_pull()

            if new_task == TaskType.FINAL_RELEASE:
                self._init_final_release()

    def _start_auto_mode(self):
        self.sm_task_idx.zero_()
        self.sm_task_time_elapsed.zero_()
        self._verify_align_retries = 0
        self._sm_prev_task_idx = -1

        # Snapshot EE
        body_q_np = self.state_0.body_q.numpy()
        _, ee_link_idx = self.ee_configs[0]
        init_tf = wp.transform(*body_q_np[ee_link_idx])
        wp.copy(self.sm_task_init_body_q, wp.array([init_tf], dtype=wp.transform))

        # Snapshot cable head from VBD
        vbd_body_q_np = self.vbd_state_0.body_q.numpy()
        cable_head_tf = wp.transform(*vbd_body_q_np[self.cable_head_body_idx])
        wp.copy(self.sm_task_plug_body_q_prev, wp.array([cable_head_tf], dtype=wp.transform))

        # Reset both grippers to open
        gripper_np = self.gripper_targets.numpy()
        gripper_np[0] = self.sm_gripper_open_value
        gripper_np[1] = self.sm_gripper_open_value
        wp.copy(self.gripper_targets, wp.array(gripper_np, dtype=wp.float32))
        self.gripper_targets_list[0] = self.sm_gripper_open_value
        self.gripper_targets_list[1] = self.sm_gripper_open_value

        # Re-seed IK from current joint state
        joint_q_np = self.state_0.joint_q.numpy()
        robot_dofs = self.single_robot_model.joint_coord_count
        ik_np = joint_q_np[:robot_dofs].reshape(1, robot_dofs)
        ik_np[0, self.gripper_joint_dofs[0]] = self.sm_gripper_open_value
        ik_np[0, self.gripper_joint_dofs[1]] = self.sm_gripper_open_value
        wp.copy(self.ik_joint_q, wp.array(ik_np, dtype=wp.float32))

        target_np = self.control.joint_target_pos.numpy()
        target_np[self.gripper_joint_dofs[0]] = self.sm_gripper_open_value
        target_np[self.gripper_joint_dofs[1]] = self.sm_gripper_open_value
        wp.copy(self.control.joint_target_pos, wp.array(target_np, dtype=wp.float32))

    def _stop_auto_mode(self):
        body_q_np = self.state_0.body_q.numpy()
        for i, (_name, link_idx) in enumerate(self.ee_configs):
            self.ee_tfs[i] = wp.transform(*body_q_np[link_idx])

        joint_q_np = self.state_0.joint_q.numpy()
        robot_dofs = self.single_robot_model.joint_coord_count
        ik_np = joint_q_np[:robot_dofs].reshape(1, robot_dofs)
        wp.copy(self.ik_joint_q, wp.array(ik_np, dtype=wp.float32))

        gripper_np = self.gripper_targets.numpy()
        gripper_np[0] = self.sm_gripper_open_value
        gripper_np[1] = self.sm_gripper_open_value
        wp.copy(self.gripper_targets, wp.array(gripper_np, dtype=wp.float32))

    def _reset_state_machine(self):
        if self.auto_mode:
            self._start_auto_mode()

    # ------------------------------------------------------------------
    # Manual IK control
    # ------------------------------------------------------------------

    def set_joint_targets(self):
        for i, tf in enumerate(self.ee_tfs):
            self.pos_objs[i].set_target_position(0, wp.transform_get_translation(tf))
            q = wp.transform_get_rotation(tf)
            self.rot_objs[i].set_target_rotation(0, wp.vec4(q[0], q[1], q[2], q[3]))

        if self.graph_ik is not None:
            wp.capture_launch(self.graph_ik)
        else:
            self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iters)

        wp.launch(
            merge_ik_with_gripper_targets,
            dim=self.single_robot_model.joint_dof_count,
            inputs=[
                self.ik_joint_q.flatten(),
                self.gripper_targets,
                self.gripper_mask,
                self.single_robot_model.joint_dof_count,
            ],
            outputs=[self.joint_target_pos],
        )
        wp.copy(self.control.joint_target_pos, self.joint_target_pos)

    # ------------------------------------------------------------------
    # CUDA graph capture
    # ------------------------------------------------------------------

    def capture(self):
        """Record CUDA graphs for simulation and IK, restoring state afterward."""
        self.graph_sim = None
        self.graph_ik = None
        if not wp.get_device().is_cuda:
            return

        # Back up states before warm-up.
        mj_state_0_backup = self.mujoco_model.state()
        mj_state_1_backup = self.mujoco_model.state()
        mj_state_0_backup.assign(self.state_0)
        mj_state_1_backup.assign(self.state_1)
        vbd_state_0_backup = self.vbd_model.state()
        vbd_state_1_backup = self.vbd_model.state()
        vbd_state_0_backup.assign(self.vbd_state_0)
        vbd_state_1_backup.assign(self.vbd_state_1)
        if self.enable_two_way_coupling and hasattr(self, "proxy_forces"):
            proxy_forces_backup = wp.clone(self.proxy_forces)
            coupling_cache_backup = wp.clone(self.coupling_forces_cache)

        # Warm-up: run simulate() once outside capture to trigger all lazy
        # initialization (e.g. MuJoCo inverse shape mapping, VBD first-step
        # allocations).  This ensures no host-side work during graph capture.
        self.simulate()

        def _restore():
            self.state_0.assign(mj_state_0_backup)
            self.state_1.assign(mj_state_1_backup)
            self.vbd_state_0.assign(vbd_state_0_backup)
            self.vbd_state_1.assign(vbd_state_1_backup)
            if self.enable_two_way_coupling and hasattr(self, "proxy_forces"):
                wp.copy(self.proxy_forces, proxy_forces_backup)
                wp.copy(self.coupling_forces_cache, coupling_cache_backup)

        _restore()

        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph_sim = capture.graph

        _restore()

        with wp.ScopedCapture() as capture:
            self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iters)
        self.graph_ik = capture.graph

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(self):
        """Staggered coupling loop (per substep):

        1. Apply lagged VBD->MuJoCo wrench (from previous substep)
        2. Step MuJoCo
        3. Sync proxy states (MuJoCo -> VBD)
        4. Subtract previously applied forces + gravity from VBD proxies
        5. Step VBD
        6. Harvest new forces from VBD contacts
        """
        mujoco_collision_step_counter = 0
        vbd_collision_step_counter = 0
        substep_dt = self.sim_dt
        two_way = self.enable_two_way_coupling and hasattr(self, "proxy_forces")

        mujoco_state_0 = self.state_0
        mujoco_state_1 = self.state_1
        vbd_state_0 = self.vbd_state_0
        vbd_state_1 = self.vbd_state_1

        applied_coupling_wrenches = None
        for _substep_idx in range(self.sim_substeps):
            # --- Step 1: Apply lagged VBD->MuJoCo wrench ---
            mujoco_state_0.clear_forces()
            self.viewer.apply_forces(mujoco_state_0)
            if two_way:
                self.coupling_forces_cache.assign(self.proxy_forces)
                mujoco_state_0.body_f.assign(mujoco_state_0.body_f + self.coupling_forces_cache)
                applied_coupling_wrenches = self.coupling_forces_cache

            # --- Step 2: Step MuJoCo ---
            if mujoco_collision_step_counter % self.mujoco_collide_substeps == 0:
                self.mujoco_collision_pipeline.collide(mujoco_state_0, self.mujoco_contacts)
            self.mujoco_solver.step(mujoco_state_0, mujoco_state_1, self.control, self.mujoco_contacts, substep_dt)
            mujoco_collision_step_counter += 1
            mujoco_state_0, mujoco_state_1 = mujoco_state_1, mujoco_state_0
            self.state_0, self.state_1 = mujoco_state_0, mujoco_state_1

            # --- Step 3: Sync proxy states (MuJoCo -> VBD) ---
            if hasattr(self, "mj_to_vbd_body_map_array"):
                wp.launch(
                    sync_proxy_states_kernel,
                    dim=self.mujoco_model.body_count,
                    inputs=[
                        mujoco_state_1.body_q,
                        mujoco_state_0.body_qd,
                        self.mj_to_vbd_body_map_array,
                        vbd_state_0.body_q,
                        vbd_state_0.body_qd,
                    ],
                )
                wp.launch(
                    smooth_proxy_teleportation_kernel,
                    dim=len(self.proxy_body_ids),
                    inputs=[
                        substep_dt,
                        self.proxy_vbd_body_ids_array,
                        vbd_state_0.body_q,
                        vbd_state_0.body_qd,
                        self.vbd_solver.body_q_prev,
                    ],
                )

            # --- Step 4: Subtract previously applied wrench + gravity ---
            if two_way and applied_coupling_wrenches is not None and hasattr(self, "proxy_mj_body_ids_array"):
                wp.launch(
                    subtract_proxy_forces_kernel,
                    dim=len(self.proxy_body_ids),
                    inputs=[
                        substep_dt,
                        self.vbd_model.gravity,
                        self.vbd_model.body_world,
                        vbd_state_0.body_q,
                        applied_coupling_wrenches,
                        self.proxy_mj_body_ids_array,
                        self.proxy_vbd_body_ids_array,
                        self.vbd_model.body_inv_mass,
                        self.vbd_model.body_inv_inertia,
                        vbd_state_0.body_qd,
                    ],
                )

            # --- Step 5: Step VBD ---
            update_vbd_history = (vbd_collision_step_counter % self.vbd_collide_substeps == 0) or (
                self.vbd_contacts is None
            )
            if update_vbd_history:
                self.vbd_collision_pipeline.collide(vbd_state_0, self.vbd_contacts)
            self.vbd_solver.set_rigid_history_update(bool(update_vbd_history))
            # Snapshot before step(); collect_rigid_contact_forces needs the
            # start-of-step pose, while step() advances solver.body_q_prev.
            wp.copy(self._vbd_body_q_prev_snapshot, self.vbd_solver.body_q_prev)
            self.vbd_solver.step(vbd_state_0, vbd_state_1, self.vbd_control, self.vbd_contacts, substep_dt)
            vbd_collision_step_counter += 1

            # --- Step 6: Harvest new VBD->MuJoCo wrench ---
            if two_way and hasattr(self, "proxy_vbd_body_ids_array"):
                self.proxy_forces.zero_()
                if self.vbd_contacts is not None and hasattr(self.vbd_solver, "collect_rigid_contact_forces"):
                    c_b0, c_b1, c_p0w, c_p1w, c_f_b1, c_count = self.vbd_solver.collect_rigid_contact_forces(
                        vbd_state_1.body_q, self._vbd_body_q_prev_snapshot, self.vbd_contacts, substep_dt
                    )
                    wp.launch(
                        harvest_proxy_wrenches_kernel,
                        dim=c_b0.shape[0],
                        inputs=[
                            c_count,
                            c_b0,
                            c_b1,
                            c_p0w,
                            c_p1w,
                            self.vbd_contacts.rigid_contact_normal,
                            c_f_b1,
                            self.vbd_model.body_inv_mass,
                            self.vbd_to_mj_body_map_array,
                            self.mujoco_model.body_com,
                            mujoco_state_0.body_q,
                            self.proxy_forces,
                        ],
                    )

            vbd_state_0, vbd_state_1 = vbd_state_1, vbd_state_0
            self.vbd_state_0, self.vbd_state_1 = vbd_state_0, vbd_state_1

    # ------------------------------------------------------------------
    # Example API
    # ------------------------------------------------------------------

    def step(self):
        if self.auto_mode:
            self.set_sm_targets()
        else:
            self.set_joint_targets()

        if self.graph_sim:
            wp.capture_launch(self.graph_sim)
        else:
            self.simulate()

        self.sim_time += self.frame_dt
        self.frame_count += 1

    def render(self):
        self.viewer.begin_frame(self.sim_time)

        if self.viewer_primary_model == "mujoco":
            self.viewer.log_state(self.state_0)
            self.viewer.log_contacts(self.mujoco_contacts, self.state_0)
            self._render_vbd_objects()
        else:
            self.viewer.log_state(self.vbd_state_0)
            self.viewer.log_contacts(self.vbd_contacts, self.vbd_state_0)

        self.viewer.end_frame()

    def _render_vbd_objects(self):
        """Render VBD cable segments, head meshes, and scene meshes when MuJoCo is the primary viewer."""
        vbd_body_q_np = self.vbd_state_0.body_q.numpy()
        shape_transform_np = self.vbd_model.shape_transform.numpy()
        shape_scale_np = self.vbd_model.shape_scale.numpy()
        shape_body_np = self.vbd_model.shape_body.numpy()

        # --- Cable capsule segments - render via the same path log_state uses ---
        # log_state for CAPSULE shapes calls viewer.log_capsules with:
        #   xforms = body_q[parent] * shape_transform[s]
        #   scales = (radius, radius, half_height)
        # The GL backend renders a cylinder body plus sphere caps for each capsule,
        # matching the VBD-primary view.
        all_xforms: list[wp.transform] = []
        all_scales: list[wp.vec3] = []
        for cable_bodies in self.vbd_cable_all_body_ids:
            for body_idx in cable_bodies:
                sidx = self._body_capsule_shape.get(body_idx)
                if sidx is None:
                    continue
                bq = vbd_body_q_np[body_idx]
                body_xform = wp.transform(
                    wp.vec3(bq[0], bq[1], bq[2]),
                    wp.quat(bq[3], bq[4], bq[5], bq[6]),
                )
                st = shape_transform_np[sidx]
                shape_local = wp.transform(
                    wp.vec3(st[0], st[1], st[2]),
                    wp.quat(st[3], st[4], st[5], st[6]),
                )
                all_xforms.append(wp.transform_multiply(body_xform, shape_local))
                radius = float(shape_scale_np[sidx][0])
                seg_hh = float(shape_scale_np[sidx][1])
                all_scales.append(wp.vec3(radius, radius, seg_hh))
        if all_xforms:
            xforms_arr = wp.array(all_xforms, dtype=wp.transform, device=self.viewer.device)
            scales_arr = wp.array(all_scales, dtype=wp.vec3, device=self.viewer.device)
            colors_arr = wp.array([wp.vec3(0.2, 0.6, 0.9)] * len(all_xforms), dtype=wp.vec3, device=self.viewer.device)
            self.viewer.log_capsules(
                "/vbd_cable_capsules",
                "",
                xforms_arr,
                scales_arr,
                colors_arr,
                None,
            )

        # --- Cable head meshes (dynamic) ---
        for shape_idx, mesh in self.vbd_head_mesh_shapes:
            body_idx = int(shape_body_np[shape_idx])
            bq = vbd_body_q_np[body_idx]
            body_xform = wp.transform(
                wp.vec3(bq[0], bq[1], bq[2]),
                wp.quat(bq[3], bq[4], bq[5], bq[6]),
            )
            st = shape_transform_np[shape_idx]
            shape_local = wp.transform(
                wp.vec3(st[0], st[1], st[2]),
                wp.quat(st[3], st[4], st[5], st[6]),
            )
            world_xform = wp.transform_multiply(body_xform, shape_local)
            self.viewer.log_shapes(
                f"/vbd_head_mesh_{shape_idx}",
                GeoType.MESH,
                1.0,
                wp.array([world_xform], dtype=wp.transform),
                colors=wp.array([wp.vec3(0.2, 0.6, 0.9)], dtype=wp.vec3),
                geo_src=mesh,
            )

        # --- Scene meshes (batched, static - pre-computed xforms) ---
        for batch_idx, (mesh, xform_arr) in enumerate(self.vbd_scene_mesh_batches):
            self.viewer.log_shapes(
                f"/vbd_scene_mesh_{batch_idx}",
                GeoType.MESH,
                1.0,
                xform_arr,
                colors=wp.array([wp.vec3(0.6, 0.6, 0.6)], dtype=wp.vec3),
                geo_src=mesh,
            )

    # ------------------------------------------------------------------
    # GUI
    # ------------------------------------------------------------------

    def gui(self, ui):
        self._gui_auto_grasp(ui)
        if not self.auto_mode:
            self._gui_gripper_controls(ui)

    def _gui_auto_grasp(self, ui):
        if not ui.collapsing_header("Auto Grasp", flags=0):
            return

        changed, value = ui.checkbox("Enable Auto Grasp", self.auto_mode)
        if changed:
            self.auto_mode = value
            if value:
                self._start_auto_mode()
            else:
                self._stop_auto_mode()

        if self.auto_mode and ui.button("Reset State Machine"):
            self._reset_state_machine()

        ui.separator()

        if self.auto_mode:
            task_idx_np = self.sm_task_idx.numpy()
            schedule_np = self.sm_task_schedule.numpy()
            task_time_np = self.sm_task_time_elapsed.numpy()
            time_limits_np = self.sm_task_time_limits.numpy()

            idx = int(task_idx_np[0])
            task_type = TaskType(int(schedule_np[idx]))
            elapsed = float(task_time_np[0])
            time_limit = float(time_limits_np[idx])

            ui.text(f"Right Arm: {task_type.name}")
            if task_type != TaskType.DONE:
                ui.text(f"  Time: {elapsed:.2f} / {time_limit:.1f} s")
        else:
            ui.text("Right Arm: IDLE (SM disabled)")

    def _gui_gripper_controls(self, ui):
        if not ui.collapsing_header("Gripper Controls", flags=0):
            return

        changed, value = ui.slider_float(
            "Right Gripper",
            self.gripper_targets_list[0],
            self.gripper_limits_lower[0],
            self.gripper_limits_upper[0],
        )
        if changed:
            self.gripper_targets_list[0] = value
            gripper_np = self.gripper_targets.numpy()
            gripper_np[0] = value
            wp.copy(self.gripper_targets, wp.array(gripper_np, dtype=wp.float32))
            self._sync_gripper_followers()

        changed, value = ui.slider_float(
            "Left Gripper",
            self.gripper_targets_list[1],
            self.gripper_limits_lower[1],
            self.gripper_limits_upper[1],
        )
        if changed:
            self.gripper_targets_list[1] = value
            gripper_np = self.gripper_targets.numpy()
            gripper_np[1] = value
            wp.copy(self.gripper_targets, wp.array(gripper_np, dtype=wp.float32))
            self._sync_gripper_followers()

    # ------------------------------------------------------------------
    # Fridge xform
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_fridge_xform():
        """Compute the fridge world transform for the robot-fridge layout.

        The Cable008 asset bbox Z is [-0.902, +0.902].  The asset sits on a
        virtual table surface; we place it at the correct absolute height so
        the robot-fridge relative pose is preserved.
        """
        # Table geometry (not physically present here)
        table_half_z = 0.5 * (0.6 - 0.215)
        table_z = table_half_z  # table_pos[2]
        table_top_z = table_z + table_half_z
        fridge_z_offset = 0.902 + table_top_z
        fridge_y_offset = (0.293 - 0.395) / 2
        quat = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), wp.pi / 2)
        return wp.transform(wp.vec3(0.95, fridge_y_offset, fridge_z_offset), quat)

    def _transform_cable_bodies(self, xform):
        """Apply a rigid transform to all cable body positions in VBD state.

        Updates state_0, state_1, model.body_q (rest config), and the VBD
        solver's body_q_prev so the solver sees consistent initial conditions
        with zero spurious velocity.
        """
        rot = wp.transform_get_rotation(xform)
        pos = wp.transform_get_translation(xform)

        all_cable_ids = set(self.cable_body_ids)

        def _apply(body_q_arr):
            body_q_np = body_q_arr.numpy()
            for bid in all_cable_ids:
                bq = body_q_np[bid]
                old_pos = wp.vec3(bq[0], bq[1], bq[2])
                old_rot = wp.quat(bq[3], bq[4], bq[5], bq[6])
                new_pos = wp.quat_rotate(rot, old_pos) + pos
                new_rot = rot * old_rot
                body_q_np[bid] = [new_pos[0], new_pos[1], new_pos[2], new_rot[0], new_rot[1], new_rot[2], new_rot[3]]
            wp.copy(body_q_arr, wp.array(body_q_np, dtype=wp.transform, device=body_q_arr.device))

        for state in (self.vbd_state_0, self.vbd_state_1):
            _apply(state.body_q)

        _apply(self.vbd_model.body_q)

        # Keep solver's body_q_prev in sync so the first step sees zero velocity.
        _apply(self.vbd_solver.body_q_prev)

    def _print_cable_pose_summary(self, tag: str, cable_results) -> None:
        body_q = self.vbd_state_0.body_q.numpy()

        def fmt_transform(body_id: int) -> str:
            q = body_q[body_id]
            return (
                f"id={body_id} pos=({float(q[0]): .6f}, {float(q[1]): .6f}, {float(q[2]): .6f}) "
                f"quat_xyzw=({float(q[3]): .6f}, {float(q[4]): .6f}, {float(q[5]): .6f}, {float(q[6]): .6f})"
            )

        print(f"[CABLE_POSE:{tag}] cable_count={len(cable_results)}", flush=True)
        for cable_index, result in enumerate(cable_results):
            print(f"[CABLE_POSE:{tag}] cable={cable_index} first {fmt_transform(result.cable_body_ids[0])}", flush=True)
            print(
                f"[CABLE_POSE:{tag}] cable={cable_index} last  {fmt_transform(result.cable_body_ids[-1])}", flush=True
            )
            for head_index, head_body in enumerate(result.head_body_ids):
                print(
                    f"[CABLE_POSE:{tag}] cable={cable_index} head{head_index} {fmt_transform(head_body)}",
                    flush=True,
                )
            fixed_ids = ",".join(str(int(body_id)) for body_id in result.fixed_body_ids)
            print(f"[CABLE_POSE:{tag}] cable={cable_index} fixed_body_ids=[{fixed_ids}]", flush=True)

    def print_robot_pose_summary(self, tag: str) -> None:
        body_q = self.state_0.body_q.numpy()
        body_names = [
            "torso_hip_yaw",
            "right_gripper_base",
            "right_gripper_leftfinger",
            "right_gripper_rightfinger",
            "right_gripper_end_effector",
            "left_gripper_base",
            "left_gripper_leftfinger",
            "left_gripper_rightfinger",
            "left_gripper_end_effector",
        ]

        def find_body(name: str) -> int:
            suffix = "/" + name
            for body_id, label in enumerate(self.mujoco_model.body_label):
                if label == name or label.endswith(suffix):
                    return body_id
            raise ValueError(f"Body {name!r} not found.")

        print(f"[ROBOT_POSE:{tag}] body_count={self.mujoco_model.body_count}", flush=True)
        for body_name in body_names:
            body_id = find_body(body_name)
            q = body_q[body_id]
            print(
                f"[ROBOT_POSE:{tag}] {body_name} id={body_id} "
                f"pos=({float(q[0]): .6f}, {float(q[1]): .6f}, {float(q[2]): .6f}) "
                f"quat_xyzw=({float(q[3]): .6f}, {float(q[4]): .6f}, {float(q[5]): .6f}, {float(q[6]): .6f})",
                flush=True,
            )

    @staticmethod
    def _compute_socket_approach_xform(fridge_xform):
        socket_offset = wp.vec3(-0.259404, 0.362961, -0.262711)
        pos = wp.transform_point(fridge_xform, socket_offset)
        fridge_rot = wp.transform_get_rotation(fridge_xform)
        socket_local_quat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 20.0 * wp.pi / 180.0)
        rot = fridge_rot * socket_local_quat
        return wp.transform(pos, rot)

    # ------------------------------------------------------------------
    # Collision helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _create_shape_config(collision_mode: CollisionMode) -> newton.ModelBuilder.ShapeConfig:
        base_cfg = newton.ModelBuilder.ShapeConfig(
            margin=0.0,
            gap=0.002,
            ke=5.0e4,
            kd=5.0e2,
            mu=2.0,
            mu_torsional=0.01,
            mu_rolling=0.00,
        )
        shape_cfg = base_cfg.copy()
        shape_cfg.is_hydroelastic = False
        if collision_mode == CollisionMode.NEWTON_HYDROELASTIC:
            shape_cfg.kh = 1e10
        return shape_cfg

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test_final(self):
        q = self.state_0.body_q.numpy()
        qd = self.state_0.body_qd.numpy()
        assert np.isfinite(q).all(), "Non-finite values in MuJoCo body_q"
        assert np.isfinite(qd).all(), "Non-finite values in MuJoCo body_qd"

        vq = self.vbd_state_0.body_q.numpy()
        vqd = self.vbd_state_0.body_qd.numpy()
        assert np.isfinite(vq).all(), "Non-finite values in VBD body_q"
        assert np.isfinite(vqd).all(), "Non-finite values in VBD body_qd"

        assert len(self.cable_body_ids) > 0, "No cable bodies imported."

        # Robot torso stayed upright
        torso_idx = self.mujoco_model.body_label.index("rby1_dfactorybot/torso_hip_yaw")
        torso_z = float(q[torso_idx, 2])
        assert torso_z > 0.5, f"Robot torso Z={torso_z:.3f} fell below 0.5 m."


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--primary-view", type=str, default="vbd", choices=["mujoco", "vbd"])
    parser.add_argument(
        "--no-twoway",
        action="store_true",
        help="Disable two-way coupling (MuJoCo and VBD run independently).",
    )
    parser.add_argument(
        "--print-cable-poses",
        action="store_true",
        help="Print imported cable body/head world poses for debugging asset alignment.",
    )
    parser.add_argument(
        "--cable-pose-settle-seconds",
        type=float,
        default=None,
        help="When printing cable poses, also print them after this many simulated seconds.",
    )
    parser.add_argument(
        "--print-robot-poses",
        action="store_true",
        help="Print selected robot body world poses for debugging robot alignment.",
    )

    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    if args.print_cable_poses and args.cable_pose_settle_seconds is not None:
        settle_frames = max(0, int(round(args.cable_pose_settle_seconds * example.fps)))
        for _ in range(settle_frames):
            example.step()
        example._print_cable_pose_summary(
            f"reference_after_{args.cable_pose_settle_seconds:g}s", example._debug_cable_results
        )
    if args.print_robot_poses:
        example.print_robot_pose_summary("reference_after_fk")
    if args.print_cable_poses or args.print_robot_poses:
        sys.exit(0)
    newton.examples.run(example, args)
