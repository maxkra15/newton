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
#
###########################################################################
# Example MuJoCo-VBD Two-Way Coupling
#
# Two dual-arm robots extract and re-insert a flexible cable (rod)
# through static hose connectors.  MuJoCo drives the articulated
# robots while VBD simulates the cable and cable-robot contacts.
#
# Architecture:
# - Universe A (MuJoCo): Robot with articulation, motors, sensors
# - Universe B (VBD): Cable + Robot Proxies (approximation of dynamics)
# - Staggered coupling loop with lagged impulses
#
# Key Features:
# 1. Proxy Bodies: MuJoCo robot bodies duplicated in VBD for collision
# 2. State Sync: MuJoCo poses copied to VBD proxies each substep
# 3. Force Harvesting: Contact forces from VBD applied to MuJoCo
# 4. Force Subtraction: Previously applied forces removed before VBD step
# 5. Lagged Impulses: Forces from substep k applied at substep k+1
# 6. Effective Mass: Proxy mass from body_invweight0
#
# Command:
#   uv run newton/examples/cable_robot/example_waterhose_v1.py
#   uv run newton/examples/cable_robot/example_waterhose_v1.py --primary-view mujoco
#   uv run -m newton.examples.cable_robot.example_waterhose_v1
#   uv run -m newton.examples.cable_robot.example_waterhose_v1 --primary-view mujoco
#
###########################################################################

import struct
import time
from enum import Enum, IntEnum
from numbers import Integral
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
from newton import Contacts, GeoType, JointTargetMode


class CollisionMode(Enum):
    """Collision pipeline modes for the MuJoCo solver.

    MUJOCO: Uses MuJoCo's native contact solver (use_mujoco_contacts=True).
    NEWTON_DEFAULT: Newton's standard collision pipeline with GJK/MPR.
    NEWTON_SDF: Newton collision with SDF (Signed Distance Field) for mesh shapes.
    NEWTON_HYDROELASTIC: Newton collision with hydroelastic contact model.
    """

    MUJOCO = "mujoco"
    NEWTON_DEFAULT = "newton_default"
    NEWTON_SDF = "newton_sdf"
    NEWTON_HYDROELASTIC = "newton_hydroelastic"


class TaskType(IntEnum):
    """State machine states for automated cable grasping and extraction."""

    IDLE = 0
    APPROACH = 1
    ENGAGE = 2
    GRASP = 3
    HOLD_GRASP = 4
    EXTRACT = 5
    HOLD_EXTRACT = 6
    INJECT = 7
    RELEASE = 8
    DONE = 9
    SIDE_SHIFT = 10
    SIDE_SHIFT_BACK = 11


NUM_ARMS = 2


_EXAMPLE_DIR = Path(__file__).resolve().parent
_ASSETS_DIR = _EXAMPLE_DIR / "assets"

ROBOT_PATH = _ASSETS_DIR / "rby1df" / "urdf"
HOSE_CONNECTOR_PATH = _ASSETS_DIR / "connector.stl"


def _load_stl_as_tri_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load an STL file into (vertices, indices) arrays.

    Args:
        path: Path to an ASCII or binary STL.

    Returns:
        vertices: float32 array of shape (N, 3).
        indices: int32 array of shape (M,) -- triangle indices, 3 per triangle.
    """
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"STL file too small: {path}")

    # Heuristic: if size matches binary STL layout, treat as binary.
    tri_count = struct.unpack_from("<I", data, 80)[0]
    expected_size = 84 + 50 * tri_count
    is_binary = expected_size == len(data)

    if is_binary:
        vertices = np.empty((tri_count * 3, 3), dtype=np.float32)
        indices = np.arange(tri_count * 3, dtype=np.int32)

        offset = 84
        for t in range(tri_count):
            # normal (3 floats) then 3 vertices (9 floats) then 2-byte attribute
            offset += 12
            v = struct.unpack_from("<fffffffff", data, offset)
            offset += 36
            offset += 2  # attribute byte count

            base = 3 * t
            vertices[base + 0] = (v[0], v[1], v[2])
            vertices[base + 1] = (v[3], v[4], v[5])
            vertices[base + 2] = (v[6], v[7], v[8])

        return vertices, indices

    # ASCII STL
    text = data.decode("utf-8", errors="ignore")
    verts: list[list[float]] = []
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith("vertex"):
            continue
        _tag, xs, ys, zs = s.split(maxsplit=3)
        verts.append([float(xs), float(ys), float(zs)])

    if len(verts) == 0 or (len(verts) % 3) != 0:
        raise ValueError(f"Failed to parse ASCII STL (no vertices): {path}")

    vertices = np.asarray(verts, dtype=np.float32)
    indices = np.arange(vertices.shape[0], dtype=np.int32)
    return vertices, indices


def _find_label_index(labels: list[str], short_name: str) -> int:
    """Find the index of *short_name* in *labels*, accepting URDF-namespaced entries.

    URDF-imported labels may be stored as ``"namespace/short_name"``; this
    helper checks for both the exact short name and the namespaced suffix so
    callers don't need to know the prefix.

    Raises:
        ValueError: If no matching label is found.
    """
    suffix = "/" + short_name
    for i, lbl in enumerate(labels):
        if lbl == short_name or lbl.endswith(suffix):
            return i
    raise ValueError(f"Label '{short_name}' not found in {labels}")


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
    """Encode the teleportation correction as velocity and undo the position jump.

    After sync_proxy_states_kernel teleports proxy body_q to MuJoCo's
    begin-of-step pose, this kernel computes the residual between the
    teleported pose and VBD's previous end-of-step pose (body_q_prev),
    folds it into body_qd as a smooth velocity correction, and resets
    body_q back to body_q_prev.  This avoids a position discontinuity
    that would contaminate VBD's finite-difference velocity estimates
    used for contact damping and friction.
    """
    i = wp.tid()
    if i >= proxy_vbd_body_ids.shape[0]:
        return

    b = proxy_vbd_body_ids[i]
    q_teleported = vbd_body_q[b]
    q_prev = vbd_solver_body_q_prev[b]

    # Translational correction velocity
    p_teleported = wp.transform_get_translation(q_teleported)
    p_prev = wp.transform_get_translation(q_prev)
    dv = (p_teleported - p_prev) / dt

    # Rotational correction velocity
    r_teleported = wp.transform_get_rotation(q_teleported)
    r_prev = wp.transform_get_rotation(q_prev)
    dw = _quat_velocity(r_teleported, r_prev, dt)

    # Add correction to the synced velocity
    qd = vbd_body_qd[b]
    vbd_body_qd[b] = qd + wp.spatial_vector(dv, dw)

    # Reset position to body_q_prev (no discontinuity)
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
    """Subtract previously applied coupling forces and gravity from VBD proxy velocities.

    Coupling forces were already applied to MuJoCo; undoing them here prevents
    double-counting.  Gravity is also subtracted because MuJoCo already accounts
    for it, and VBD must not apply it a second time to proxy bodies.
    """
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

    # Subtract gravity: look up the per-world gravity vector for this body.
    world_idx = body_world[vbd_body_id]
    g = gravity[wp.max(world_idx, 0)]
    delta_v_grav = dt * g

    vbd_body_qd[vbd_body_id] = vbd_body_qd[vbd_body_id] - wp.spatial_vector(delta_v + delta_v_grav, delta_w)


@wp.kernel(enable_backward=False)
def harvest_proxy_wrenches_from_contact_forces_kernel(
    # Contact buffers (per-contact, world frame)
    rigid_contact_count: wp.array[int],
    contact_body0: wp.array[wp.int32],
    contact_body1: wp.array[wp.int32],
    contact_point0_world: wp.array[wp.vec3],
    contact_point1_world: wp.array[wp.vec3],
    contact_force_on_body1: wp.array[wp.vec3],
    # VBD body inv mass
    vbd_body_inv_mass: wp.array[float],
    # Reverse lookup: vbd_body_id -> MuJoCo body id (-1 if not a proxy)
    vbd_to_mj_body_map: wp.array[int],
    # MuJoCo model data (to compute torque about COM)
    mj_body_com: wp.array[wp.vec3],
    mj_body_q: wp.array[wp.transform],
    # Output: wrench as spatial_vector(force, torque) [N, N*m]
    out_mj_body_f: wp.array[wp.spatial_vector],
):
    """Harvest coupling wrenches from VBD per-contact forces onto MuJoCo bodies.

    Includes only contacts where exactly one body is a proxy and the other is
    dynamic (inv_mass > 0).  Excludes proxy-proxy contacts and temporarily
    excludes proxy-static contacts.

    Force convention:
      - ``contact_force_on_body1`` is force applied to body1 at ``contact_point1_world``
      - Force on body0 is ``-contact_force_on_body1`` at ``contact_point0_world``
    """
    contact_id = wp.tid()
    if contact_id >= rigid_contact_count[0]:
        return

    body0 = int(contact_body0[contact_id])
    body1 = int(contact_body1[contact_id])
    if body0 < 0 or body1 < 0:
        return

    # O(1) proxy lookup via reverse map
    mj_body_id0 = vbd_to_mj_body_map[body0] if body0 < vbd_to_mj_body_map.shape[0] else -1
    mj_body_id1 = vbd_to_mj_body_map[body1] if body1 < vbd_to_mj_body_map.shape[0] else -1
    is_proxy0 = int(mj_body_id0 >= 0)
    is_proxy1 = int(mj_body_id1 >= 0)

    # Exactly one body must be a proxy (excludes proxy-proxy and env-env)
    if (is_proxy0 + is_proxy1) != 1:
        return

    # Currently excludes proxy vs static environment
    other_body_id = body1 if is_proxy0 == 1 else body0
    if vbd_body_inv_mass[other_body_id] <= 0.0:
        return

    # Determine the force on the proxy and its MuJoCo body id
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

    # Compute wrench (force + torque about MuJoCo body COM)
    com_world = wp.transform_point(mj_body_q[mj_body_id], mj_body_com[mj_body_id])
    torque_world = wp.cross(contact_point_world - com_world, force_on_proxy_world)
    wp.atomic_add(out_mj_body_f, mj_body_id, wp.spatial_vector(force_on_proxy_world, torque_world))


@wp.func
def top_side_unit_from_capsule_quat(cq: wp.quat):
    """Unit vector pointing 'upward' in the plane perpendicular to the capsule axis.

    Gram-Schmidt projects world-up onto the plane orthogonal to the capsule's
    local Z-axis.  When the capsule is nearly vertical the projection degenerates;
    fall back to an arbitrary perpendicular direction.
    """
    cap_axis = wp.quat_rotate(cq, wp.vec3(0.0, 0.0, 1.0))

    side = wp.vec3(0.0, 0.0, 1.0) - wp.dot(wp.vec3(0.0, 0.0, 1.0), cap_axis) * cap_axis
    side_len = wp.length(side)
    if side_len > 1.0e-8:
        return side / side_len

    # Degenerate: capsule axis ~ world-up.  Pick any perpendicular direction.
    fallback = wp.cross(cap_axis, wp.vec3(1.0, 0.0, 0.0))
    fb_len = wp.length(fallback)
    if fb_len > 1.0e-8:
        return fallback / fb_len
    return wp.vec3(0.0, 1.0, 0.0)


@wp.kernel(enable_backward=False)
def set_target_pose_kernel(
    task_schedule: wp.array[wp.int32],
    task_time_soft_limits: wp.array[float],
    task_idx: wp.array[int],
    task_time_elapsed: wp.array[float],
    task_dt: float,
    approach_offsets: wp.array[wp.vec3],
    capsule_grasp_offset_from_com: wp.array[wp.vec3],
    grasp_top_bias: float,
    extract_distance: float,
    inject_distance: float,
    inject_forward_offset_x: float,
    spread_distance_y: float,
    spread_direction_sign: float,
    capsule_body_indices: wp.array[int],
    grasp_orientation_offset: wp.array[wp.vec4],
    gripper_open_values: wp.array[wp.float32],
    gripper_closed_values: wp.array[wp.float32],
    home_ee_body_q: wp.array[wp.transform],
    task_ee_init_body_q: wp.array[wp.transform],
    task_capsule_body_q_prev: wp.array[wp.transform],
    capsule_body_q: wp.array[wp.transform],
    # outputs
    ee_pos_target: wp.array[wp.vec3],
    ee_pos_target_interpolated: wp.array[wp.vec3],
    ee_rot_target: wp.array[wp.vec4],
    ee_rot_target_interpolated: wp.array[wp.vec4],
    gripper_target: wp.array2d[wp.float32],
):
    """Compute per-arm EE position/orientation targets and gripper open/close targets.

    Each thread handles one arm.  The task phase (from the state machine) determines
    how the target is computed:

      APPROACH       -- track live capsule with approach offset, align EE to capsule axis
      ENGAGE         -- move to grasp pose using live capsule position
      GRASP          -- close gripper, track capsule position
      HOLD_GRASP     -- same as GRASP (dwell before extraction)
      EXTRACT        -- linear pull along capsule axis at extraction start
      HOLD_EXTRACT   -- spread grippers sideways along world Y
      SIDE_SHIFT     -- same relative motion as HOLD_EXTRACT from a new start pose
      SIDE_SHIFT_BACK -- reverse the sideways spread
      INJECT         -- push back opposite extraction direction from shifted pose
      RELEASE        -- hold position, open gripper
      DONE           -- return to home pose, open gripper
      else (IDLE)    -- hold position, keep gripper closed

    Outputs are the raw target and a time-interpolated (lerp/slerp) version that
    the downstream IK solver consumes.
    """
    arm_idx = wp.tid()

    idx = task_idx[arm_idx]
    task = task_schedule[idx]
    time_limit = task_time_soft_limits[idx]

    task_time_elapsed[arm_idx] += task_dt

    t = wp.min(1.0, task_time_elapsed[arm_idx] / time_limit)

    # EE pose snapshot at the start of this task
    ee_pos_prev = wp.transform_get_translation(task_ee_init_body_q[arm_idx])
    ee_quat_prev = wp.transform_get_rotation(task_ee_init_body_q[arm_idx])

    # Capsule orientation snapshot at the start of this task
    capsule_quat_prev = wp.transform_get_rotation(task_capsule_body_q_prev[arm_idx])

    # Live capsule state from VBD
    capsule_pos = wp.transform_get_translation(capsule_body_q[capsule_body_indices[arm_idx]])
    capsule_quat = wp.transform_get_rotation(capsule_body_q[capsule_body_indices[arm_idx]])

    # Per-arm grasp orientation (stored as vec4 xyzw -> quaternion)
    gv = grasp_orientation_offset[arm_idx]
    grasp_quat_offset = wp.quaternion(gv[:3], gv[3])

    ee_quat_target = ee_quat_prev
    t_gripper = 0.0

    if task == TaskType.APPROACH.value:
        grasp_pos_offset = wp.quat_rotate(capsule_quat, capsule_grasp_offset_from_com[arm_idx])
        if grasp_top_bias != 0.0:
            grasp_pos_offset = grasp_pos_offset + grasp_top_bias * top_side_unit_from_capsule_quat(capsule_quat)
        ee_pos_target[arm_idx] = capsule_pos + grasp_pos_offset + approach_offsets[arm_idx]

        capsule_axis = wp.quat_rotate(capsule_quat, wp.vec3(0.0, 0.0, 1.0))
        ee_quat_target = wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), capsule_axis) * grasp_quat_offset
    elif task == TaskType.ENGAGE.value:
        grasp_pos_offset = wp.quat_rotate(capsule_quat_prev, capsule_grasp_offset_from_com[arm_idx])
        if grasp_top_bias != 0.0:
            grasp_pos_offset = grasp_pos_offset + grasp_top_bias * top_side_unit_from_capsule_quat(capsule_quat_prev)
        ee_pos_target[arm_idx] = capsule_pos + grasp_pos_offset
        ee_quat_target = ee_quat_prev
    elif task == TaskType.GRASP.value or task == TaskType.HOLD_GRASP.value:
        grasp_pos_offset = wp.quat_rotate(capsule_quat_prev, capsule_grasp_offset_from_com[arm_idx])
        if grasp_top_bias != 0.0:
            grasp_pos_offset = grasp_pos_offset + grasp_top_bias * top_side_unit_from_capsule_quat(capsule_quat_prev)
        ee_pos_target[arm_idx] = capsule_pos + grasp_pos_offset
        ee_quat_target = ee_quat_prev
        t_gripper = 1.0
    elif task == TaskType.EXTRACT.value:
        extract_axis = wp.quat_rotate(capsule_quat_prev, wp.vec3(0.0, 0.0, 1.0))
        ee_pos_target[arm_idx] = ee_pos_prev + extract_axis * extract_distance
        ee_quat_target = ee_quat_prev
        t_gripper = 1.0
    elif task == TaskType.HOLD_EXTRACT.value:
        spread_sign = (-1.0 if arm_idx == 0 else 1.0) * spread_direction_sign
        ee_pos_target[arm_idx] = ee_pos_prev + wp.vec3(0.0, spread_sign * spread_distance_y, 0.0)
        ee_quat_target = ee_quat_prev
        t_gripper = 1.0
    elif task == TaskType.SIDE_SHIFT.value:
        spread_sign = (-1.0 if arm_idx == 0 else 1.0) * spread_direction_sign
        ee_pos_target[arm_idx] = ee_pos_prev + wp.vec3(0.0, spread_sign * spread_distance_y, 0.0)
        ee_quat_target = ee_quat_prev
        t_gripper = 1.0
    elif task == TaskType.SIDE_SHIFT_BACK.value:
        spread_sign = -((-1.0 if arm_idx == 0 else 1.0) * spread_direction_sign)
        ee_pos_target[arm_idx] = ee_pos_prev + wp.vec3(0.0, spread_sign * spread_distance_y, 0.0)
        ee_quat_target = ee_quat_prev
        t_gripper = 1.0
    elif task == TaskType.INJECT.value:
        extract_axis = wp.quat_rotate(capsule_quat_prev, wp.vec3(0.0, 0.0, 1.0))
        ee_pos_target[arm_idx] = (
            ee_pos_prev - extract_axis * inject_distance + wp.vec3(inject_forward_offset_x, 0.0, 0.0)
        )
        ee_quat_target = ee_quat_prev
        t_gripper = 1.0
    elif task == TaskType.RELEASE.value:
        ee_pos_target[arm_idx] = ee_pos_prev
        ee_quat_target = ee_quat_prev
        t_gripper = 0.0
    elif task == TaskType.DONE.value:
        ee_pos_target[arm_idx] = wp.transform_get_translation(home_ee_body_q[arm_idx]) + approach_offsets[arm_idx]
        ee_quat_target = wp.transform_get_rotation(home_ee_body_q[arm_idx])
        t_gripper = 0.0
    else:
        ee_pos_target[arm_idx] = ee_pos_prev
        t_gripper = 1.0

    # Interpolate: lerp position, slerp orientation
    ee_pos_target_interpolated[arm_idx] = ee_pos_prev * (1.0 - t) + ee_pos_target[arm_idx] * t
    ee_quat_interpolated = wp.quat_slerp(ee_quat_prev, ee_quat_target, t)

    ee_rot_target[arm_idx] = ee_quat_target[:4]
    ee_rot_target_interpolated[arm_idx] = ee_quat_interpolated[:4]

    # Gripper: lerp between open and closed values based on t_gripper (0=open, 1=closed)
    base = arm_idx * 2
    gripper_target[arm_idx, 0] = gripper_open_values[base] * (1.0 - t_gripper) + gripper_closed_values[base] * t_gripper
    gripper_target[arm_idx, 1] = (
        gripper_open_values[base + 1] * (1.0 - t_gripper) + gripper_closed_values[base + 1] * t_gripper
    )


@wp.kernel(enable_backward=False)
def apply_gripper_centering_correction_kernel(
    task_schedule: wp.array[wp.int32],
    task_idx: wp.array[int],
    vbd_body_q: wp.array[wp.transform],
    capsule_body_indices: wp.array[int],
    finger_proxy_body_indices: wp.array2d[int],
    k_center: float,
    k_axis_center: float,
    max_step: float,
    # in/out
    ee_pos_target_interpolated: wp.array[wp.vec3],
):
    """Shift the EE target to keep the capsule centered between finger proxies.

    Two proportional corrections are applied:
      1. Closing-axis (``k_center``): if the capsule is off-center between the two
         fingers, nudge the EE in the opposite direction.
      2. Axial (``k_axis_center``): if the finger midpoint is off the capsule's
         medial axis (local +Z through COM), nudge the EE radially toward it.

    Active during ENGAGE, GRASP, and HOLD_GRASP phases.
    """
    arm_idx = wp.tid()
    task = task_schedule[task_idx[arm_idx]]

    if not (task == TaskType.ENGAGE.value or task == TaskType.GRASP.value or task == TaskType.HOLD_GRASP.value):
        return

    cap = capsule_body_indices[arm_idx]
    f0 = finger_proxy_body_indices[arm_idx, 0]
    f1 = finger_proxy_body_indices[arm_idx, 1]
    if cap < 0 or f0 < 0 or f1 < 0:
        return

    pcap = wp.transform_get_translation(vbd_body_q[cap])
    cap_q = wp.transform_get_rotation(vbd_body_q[cap])
    pf0 = wp.transform_get_translation(vbd_body_q[f0])
    pf1 = wp.transform_get_translation(vbd_body_q[f1])

    # Closing-axis unit vector (finger0 -> finger1)
    u = pf1 - pf0
    ulen = wp.length(u)
    if ulen < 1.0e-8:
        return
    u = u / ulen

    # 1) Re-center capsule between fingers along the closing axis.
    mid = 0.5 * (pf0 + pf1)
    delta_u = -k_center * wp.dot(pcap - mid, u) * u

    # 2) Align finger midpoint onto capsule medial axis (local +Z through COM).
    cap_axis = wp.quat_rotate(cap_q, wp.vec3(0.0, 0.0, 1.0))
    d_mid = mid - pcap
    delta_axis = -k_axis_center * (d_mid - wp.dot(d_mid, cap_axis) * cap_axis)

    delta = delta_u + delta_axis
    dlen = wp.length(delta)
    if dlen > max_step:
        delta = delta * (max_step / dlen)

    ee_pos_target_interpolated[arm_idx] = ee_pos_target_interpolated[arm_idx] + delta


@wp.kernel(enable_backward=False)
def advance_task_kernel(
    task_time_soft_limits: wp.array[float],
    ee_pos_target: wp.array[wp.vec3],
    ee_rot_target: wp.array[wp.vec4],
    robot_body_q: wp.array[wp.transform],
    capsule_body_q: wp.array[wp.transform],
    ee_body_indices: wp.array[int],
    capsule_body_indices: wp.array[int],
    pos_error_thresholds: wp.array[float],
    rot_error_thresholds: wp.array[float],
    # outputs
    task_idx: wp.array[int],
    task_time_elapsed: wp.array[float],
    task_ee_init_body_q: wp.array[wp.transform],
    task_capsule_body_q_prev: wp.array[wp.transform],
):
    """Advance the per-arm task state machine when convergence criteria are met.

    Transitions to the next task when all of the following hold:
      - Elapsed time >= soft time limit for the current task.
      - Position error (EE vs target) < threshold [m].
      - Rotation error (geodesic angle, EE vs target) < threshold [rad].
      - Current task is not the last in the schedule.

    On transition, snapshots the current EE and capsule transforms as the
    starting reference for the new task.
    """
    arm_idx = wp.tid()

    idx = task_idx[arm_idx]
    time_limit = task_time_soft_limits[idx]

    ee_body_id = ee_body_indices[arm_idx]
    ee_pos_current = wp.transform_get_translation(robot_body_q[ee_body_id])
    ee_rot_current = wp.transform_get_rotation(robot_body_q[ee_body_id])

    pos_err = wp.length(ee_pos_target[arm_idx] - ee_pos_current)

    # Geodesic rotation error: angle of the relative quaternion (shortest arc).
    rv = ee_rot_target[arm_idx]
    target_quat = wp.quaternion(rv[:3], rv[3])
    quat_rel = ee_rot_current * wp.quat_inverse(target_quat)
    rot_err = 2.0 * wp.atan2(wp.length(quat_rel[:3]), wp.abs(quat_rel[3]))

    if (
        task_time_elapsed[arm_idx] >= time_limit
        and task_idx[arm_idx] < task_time_soft_limits.shape[0] - 1
        and pos_err < pos_error_thresholds[idx]
        and rot_err < rot_error_thresholds[idx]
    ):
        task_idx[arm_idx] += 1
        task_time_elapsed[arm_idx] = 0.0
        task_ee_init_body_q[arm_idx] = robot_body_q[ee_body_id]
        task_capsule_body_q_prev[arm_idx] = capsule_body_q[capsule_body_indices[arm_idx]]


@wp.kernel(enable_backward=False)
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


class Example:
    def __init__(self, viewer, num_worlds=1, args=None):
        # ----------------------------------------------------------------
        # Common (both MuJoCo and VBD)
        # ----------------------------------------------------------------
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        self.substeps = int(getattr(args, "substeps", 10)) if args is not None else 10
        self.substep_dt = self.frame_dt / self.substeps

        self.num_worlds = num_worlds
        self.viewer = viewer

        # Keep the example quiet by default (avoid spamming stdout).
        self.verbose = False
        self.frame_count = 0
        self.use_graph = wp.get_device().is_cuda
        self.enable_auto_grasp = True

        # ----------------------------------------------------------------
        # Coupling
        # ----------------------------------------------------------------
        coupling_mode = str(getattr(args, "coupling_mode", "two-way")) if args is not None else "two-way"
        if coupling_mode not in ("two-way", "none"):
            raise ValueError(f"coupling_mode must be 'two-way' or 'none', got: {coupling_mode}")
        self.coupling_mode = coupling_mode
        self.enable_two_way_coupling = self.coupling_mode == "two-way"

        # Shared table geometry used by both MuJoCo and VBD setup.
        self.table_half_size = [0.25, 0.5, 0.02]
        self.table_pos = [0.5, 0, 0.75]

        # ----------------------------------------------------------------
        # MuJoCo
        # ----------------------------------------------------------------
        self.mujoco_collision_mode = CollisionMode.NEWTON_SDF
        self.mujoco_collide_substeps = 100  # run MuJoCo collision every X MuJoCo substeps
        self.mujoco_iterations = 20
        self.mujoco_ls_iterations = 10
        self.enable_robot = True
        self.rigid_contact_max = 100000

        # IK settings.
        self.ik_iters = 24

        # Gripper joint drive tuning: stiff drives resist cable reaction forces
        # during extraction.
        self.gripper_joint_target_ke = 10000.0
        self.gripper_joint_target_kd = 1000.0
        self.gripper_joint_effort_limit = 100000.0

        # MuJoCo scene defaults derived from collision mode.
        self.mujoco_default_shape_cfg = self._create_shape_config(self.mujoco_collision_mode)

        # Keep ground settings separate from default shape settings.
        self.mujoco_default_ground_cfg = self.mujoco_default_shape_cfg.copy()
        self.mujoco_default_ground_cfg.ke = 1e4
        self.mujoco_default_ground_cfg.kd = 1e0
        self.mujoco_default_ground_cfg.mu = 0.3

        # ----------------------------------------------------------------
        # State machine tuning
        # ----------------------------------------------------------------
        # Approach offset [m]: lateral (Y) stand-off from capsule COM.
        self.sm_approach_offset_y = 0.15

        # Grasp point along capsule local axis.
        self.sm_grasp_axis_fraction = 1.1
        # World-up nudge [m] orthogonal to capsule axis for grasp alignment.
        self.sm_grasp_top_bias = 0.002

        # Task durations [s].
        self.sm_time_approach = 1.0
        self.sm_time_engage = 0.5
        self.sm_time_grasp = 1.0
        self.sm_time_extract = 1.5
        self.sm_time_side_shift = 1.0
        self.sm_time_inject = 4.0
        self.sm_time_release = 0.5
        self.sm_time_done = 2.5
        self.sm_spread_distance_y = 0.02
        self.sm_spread_direction_sign = 1.0

        self._setup_mujoco_world(args)

        # ----------------------------------------------------------------
        # VBD
        # ----------------------------------------------------------------
        self.vbd_iterations = 10

        self.vbd_collide_substeps = 5  # run VBD collision every X VBD substeps
        self.vbd_mesh_use_sdf = True
        self.vbd_mesh_sdf_max_resolution = 64

        self.vbd_default_contact_ke = 1.0e5
        self.vbd_default_contact_kd = 1.0e-1
        self.vbd_default_contact_margin = 0.001

        self.vbd_solver_friction_epsilon = 0.1
        self.vbd_rigid_contact_buffer_size = 128

        self.proxy_mass_scale = 1.0

        self.proxy_mass_source = (
            str(getattr(args, "proxy_mass_source", "effective")) if args is not None else "effective"
        )
        if self.proxy_mass_source not in ("effective", "local"):
            raise ValueError(f"proxy_mass_source must be 'effective' or 'local', got: {self.proxy_mass_source}")

        self.vbd_proxy_mu = 1000000.0
        self.vbd_proxy_margin = 0.001

        self.vbd_cable_mu = 1.0
        self.vbd_cable_margin = 0.0
        self.vbd_cable_gap = 0.001

        self.vbd_static_margin = 1.0e-4
        self.vbd_static_gap = 0.001
        self.vbd_near_tip_mu = 1.0e1
        self.vbd_far_tip_mu = 1.0e5
        self.vbd_ground_mu = 1.0e5

        # Cable capsule segment geometry
        self.capsule_radius = 0.003
        self.capsule_cylinder_height = 4.0 / 60.0
        self.capsule_tilt_angle_deg = 30.0
        self.capsule_length_offset = 0.01
        self.capsule_spawn_x_bias = 0.005
        self.capsule_spawn_axis_offset = 0.009
        self.hose_y_offset = 0.15

        # Cable parameters
        self.cable_num_segments = max(4, int(getattr(args, "cable_segments", 100)) if args is not None else 100)
        self.cable_num_straight_ends = max(1, int(getattr(args, "cable_straight_ends", 5)) if args is not None else 5)
        self.cable_span = 2.0 * float(self.hose_y_offset)
        self.cable_stretch_stiffness = 1.0e12  # EA [N]
        self.cable_bend_stiffness = 3.0e0  # EI [N*m^2]
        self.cable_stretch_damping = 1.0e-3
        self.cable_bend_damping = 1.0e0
        self.cable_density = 10000.0  # [kg/m^3]
        self.cable_tip_near_y_offset = 0.017

        # VBD rigid-body solver penalty parameters
        self.vbd_rigid_avbd_beta = 1.0e6
        self.vbd_rigid_contact_k_start = 1.0e2
        self.vbd_rigid_joint_linear_k_start = 1.0e4
        self.vbd_rigid_joint_angular_k_start = 1.0e1

        self._setup_vbd_world_and_coupling(args)

        # ----------------------------------------------------------------
        # MuJoCo planning/control setup (depends on VBD capsule setup)
        # ----------------------------------------------------------------

        if self.enable_robot:
            self.setup_end_effectors()
            self.setup_ik()
            self.setup_gripper_targets()
            self.setup_state_machine()

            # Optional auto grasp/state-machine startup.
            self.auto_mode = bool(self.enable_auto_grasp)
            if self.auto_mode:
                self._start_auto_mode()

            # Store joint target positions for merging.
            self.joint_target_pos = wp.zeros_like(self.control.joint_target_pos)
            wp.copy(self.joint_target_pos, self.control.joint_target_pos)

        # ----------------------------------------------------------------
        # Viewer
        # ----------------------------------------------------------------
        # Start paused in interactive GL viewer; leave other viewer types untouched.
        if hasattr(self.viewer, "_paused"):
            self.viewer._paused = isinstance(self.viewer, newton.viewer.ViewerGL)
        self.show_isosurface = False
        self.viewer_primary_model = str(getattr(args, "primary_view", "vbd")) if args is not None else "vbd"
        if self.viewer_primary_model not in ("mujoco", "vbd"):
            raise ValueError(f"primary_view must be 'mujoco' or 'vbd', got: {self.viewer_primary_model}")
        self.viewer_camera_mode = str(getattr(args, "camera_view", "side")) if args is not None else "side"
        if self.viewer_camera_mode not in ("front", "side"):
            raise ValueError(f"camera_view must be 'front' or 'side', got: {self.viewer_camera_mode}")

        if self.enable_robot and self.viewer_primary_model == "mujoco":
            self.viewer.set_model(self.mujoco_model)
        else:
            self.viewer.set_model(self.vbd_model)

        if isinstance(self.viewer, newton.viewer.ViewerGL):
            if self.viewer_camera_mode == "front":
                self.viewer.set_camera(wp.vec3(6.5, 0.0, 1.6), pitch=-5.0, yaw=-180.0)
            else:
                self.viewer.set_camera(wp.vec3(0.5, 4.0, 1.25), pitch=-5.0, yaw=-90.0)
            self.viewer.camera.fov = 15.0

        # Profiling (enabled with --profile-interval N, prints every N frames).
        # Disables CUDA graphs so simulate() runs directly with per-section timing.
        profile_interval = int(getattr(args, "profile_interval", 0)) if args is not None else 0
        if profile_interval > 0:
            self._profile_timers = {
                "mj_collide": 0.0,
                "mj_step": 0.0,
                "sync": 0.0,
                "vbd_collide": 0.0,
                "vbd_step": 0.0,
                "harvest": 0.0,
            }
            self._profile_frame_count = 0
            self._profile_interval = profile_interval
            self.use_graph = False
        else:
            self._profile_timers = None
            self._profile_frame_count = 0
            self._profile_interval = 0

        self.capture()

    # ------------------------------------------------------------------
    # World building: MuJoCo
    # ------------------------------------------------------------------

    def _create_shape_config(self, collision_mode: CollisionMode) -> newton.ModelBuilder.ShapeConfig:
        """Create shape configuration based on collision mode.

        Args:
            collision_mode: The collision pipeline mode to configure for.

        Returns:
            ShapeConfig configured for the specified collision mode.
        """
        # Base configuration common to all modes
        base_cfg = newton.ModelBuilder.ShapeConfig(
            margin=0.0,
            gap=0.005,
            ke=5.0e4,
            kd=5.0e2,
            mu=2.0,
        )

        if collision_mode == CollisionMode.NEWTON_SDF:
            shape_cfg = base_cfg.copy()
            shape_cfg.is_hydroelastic = False
            return shape_cfg
        elif collision_mode == CollisionMode.NEWTON_HYDROELASTIC:
            shape_cfg = base_cfg.copy()
            shape_cfg.is_hydroelastic = True
            shape_cfg.k_hydro = 1e11
            return shape_cfg
        else:
            shape_cfg = base_cfg.copy()
            shape_cfg.is_hydroelastic = False
            return shape_cfg

    def _create_collision_pipeline(self, collision_mode: CollisionMode, args=None):
        """Create collision pipeline based on collision mode.

        Args:
            collision_mode: The collision pipeline mode to use.
            args: Optional command-line arguments.

        Returns:
            CollisionPipeline instance or None for MuJoCo native contacts.
        """
        if collision_mode == CollisionMode.MUJOCO:
            # MuJoCo handles collision detection and contact solving internally.
            return None
        elif collision_mode == CollisionMode.NEWTON_DEFAULT:
            # Standard Newton collision pipeline
            return newton.examples.create_collision_pipeline(self.mujoco_model, args)
        elif collision_mode == CollisionMode.NEWTON_SDF:
            return newton.CollisionPipeline(self.mujoco_model, reduce_contacts=True, broad_phase="explicit")
        elif collision_mode == CollisionMode.NEWTON_HYDROELASTIC:
            from newton.geometry import HydroelasticSDF  # noqa: PLC0415

            return newton.CollisionPipeline(
                self.mujoco_model,
                reduce_contacts=True,
                broad_phase="explicit",
                sdf_hydroelastic_config=HydroelasticSDF.Config(output_contact_surface=True),
            )
        else:
            raise ValueError(f"Unknown collision mode: {collision_mode}")

    def setup_robot_builder(self):
        """Build a ModelBuilder for the dual-arm robot from URDF, configure joints and gains."""
        robot = newton.ModelBuilder()
        robot.default_shape_cfg = self.mujoco_default_shape_cfg
        robot.bound_mass = 1.0e-4
        robot.bound_inertia = 1.0e-6

        robot_file = ROBOT_PATH / "robot_edited.urdf"
        if not robot_file.is_file():
            raise FileNotFoundError(f"Robot URDF not found: {robot_file}. Expected assets at: {_ASSETS_DIR}")

        robot.add_urdf(
            str(robot_file),
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0)),
            floating=False,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
            ignore_inertial_definitions=True,
        )
        # Keep pre-mimic-support behavior for this example: gripper mimic constraints
        # were previously ignored by SolverMuJoCo and alter the baseline trajectories.
        mimic_count = len(robot.constraint_mimic_enabled)
        if mimic_count > 0:
            robot.constraint_mimic_enabled[-mimic_count:] = [False] * mimic_count

        # Discover gripper DOFs by joint label (robust to URDF changes).
        gripper_joint_names = [
            "right_gripper_left_finger_joint",
            "right_gripper_right_finger_joint",
            "left_gripper_left_finger_joint",
            "left_gripper_right_finger_joint",
        ]
        dofs: list[int] = []
        for name in gripper_joint_names:
            try:
                j = _find_label_index(robot.joint_label, name)
            except ValueError:
                dofs.append(-1)
                continue
            dof_start = int(robot.joint_qd_start[j])
            dofs.append(dof_start)
        self.gripper_joint_dofs = [d for d in dofs if d >= 0]

        # Robot-wide PD gains and effort limits.
        robot.joint_target_ke[: robot.joint_dof_count] = [45000.0] * robot.joint_dof_count
        robot.joint_target_kd[: robot.joint_dof_count] = [4500.0] * robot.joint_dof_count
        robot.joint_effort_limit[: robot.joint_dof_count] = [1000.0] * robot.joint_dof_count
        robot.joint_armature[: robot.joint_dof_count] = [0.2] * robot.joint_dof_count

        # Override gripper joint drives with tuned values from __init__.
        for dof in self.gripper_joint_dofs:
            robot.joint_target_ke[dof] = self.gripper_joint_target_ke
            robot.joint_target_kd[dof] = self.gripper_joint_target_kd
            robot.joint_effort_limit[dof] = self.gripper_joint_effort_limit
            robot.joint_armature[dof] = 0.5

        # Initial joint positions: EE above table, arms in a ready pose.
        robot.joint_q = [
            4.8646115e-02,
            -1.1358134e-01,
            2.8509942e-01,
            3.0236751e-01,
            -4.3634601e-02,
            9.6731670e-03,
            -8.5306484e-01,
            -1.0891527e00,
            6.6765565e-01,
            -2.0121396e00,
            -1.0203781e00,
            1.5501461e00,
            5.6562239e-01,
            1.9687047e-07,
            -3.9999921e-02,
            4.0000085e-02,
            -7.0531148e-01,
            1.0506693e00,
            -4.4851208e-01,
            -1.9159117e00,
            1.0035634e00,
            1.5637023e00,
            -8.4481186e-01,
            -3.3100471e-07,
            -4.0000360e-02,
            3.9999638e-02,
            -5.4279553e-06,
            1.4788106e-04,
        ]

        # Explicitly set joint_target_mode to POSITION for all DOFs (URDF
        # parsing infers NONE when gains are zero at parse time, but we set
        # non-zero gains above).
        for i in range(robot.joint_dof_count):
            robot.joint_target_mode[i] = int(JointTargetMode.POSITION)

        # Initialize joint target positions to the initial pose so the robot
        # holds its configured posture rather than collapsing to zero.
        robot.joint_target_pos = list(robot.joint_q)

        finger_body_indices = {
            _find_label_index(robot.body_label, "right_gripper_base"),
            _find_label_index(robot.body_label, "right_gripper_camera_bracket"),
            _find_label_index(robot.body_label, "right_gripper_leftfinger"),
            _find_label_index(robot.body_label, "right_gripper_rightfinger"),
            _find_label_index(robot.body_label, "left_gripper_base"),
            _find_label_index(robot.body_label, "left_gripper_camera_bracket"),
            _find_label_index(robot.body_label, "left_gripper_leftfinger"),
            _find_label_index(robot.body_label, "left_gripper_rightfinger"),
        }
        for shape_idx, body_idx in enumerate(robot.shape_body):
            if body_idx not in finger_body_indices:
                robot.shape_flags[shape_idx] &= ~newton.ShapeFlags.HYDROELASTIC

        return robot

    def setup_scene_builder(self, robot):
        """Wrap the robot builder into a scene-level ModelBuilder."""
        scene = newton.ModelBuilder()
        scene.default_shape_cfg = self.mujoco_default_shape_cfg
        scene.bound_mass = robot.bound_mass
        scene.bound_inertia = robot.bound_inertia
        scene.add_builder(robot)

        return scene

    def _setup_mujoco_world(self, args=None):
        """Build and initialize the MuJoCo-side robot world and solver state."""
        # Gripper DOF indices are discovered from joint names after URDF import
        # (see setup_robot_builder()).
        self.gripper_joint_dofs = []

        if not self.enable_robot:
            self.single_robot_model = None
            self.mujoco_model = None
            self.collision_pipeline = None
            self._use_mujoco_contacts = False
            self.mujoco_solver = None
            self.mujoco_state_0 = None
            self.mujoco_state_1 = None
            self.control = None
            self.mujoco_contacts = None
            return

        robot = self.setup_robot_builder()
        scene = self.setup_scene_builder(robot)

        self.single_robot_model = robot.finalize()
        self.mujoco_model = scene.finalize()

        self.collision_pipeline = self._create_collision_pipeline(self.mujoco_collision_mode, args)

        # Evaluate FK into the model so body transforms are valid for collision pipeline setup.
        newton.eval_fk(self.mujoco_model, self.mujoco_model.joint_q, self.mujoco_model.joint_qd, self.mujoco_model)

        self._use_mujoco_contacts = self.mujoco_collision_mode == CollisionMode.MUJOCO
        num_per_world = self.rigid_contact_max // self.num_worlds
        self.mujoco_solver = newton.solvers.SolverMuJoCo(
            self.mujoco_model,
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            njmax=num_per_world,
            nconmax=num_per_world,
            ls_parallel=True,
            iterations=self.mujoco_iterations,
            ls_iterations=self.mujoco_ls_iterations,
            use_mujoco_contacts=self._use_mujoco_contacts,
            impratio=1000.0,
        )

        if self.verbose:
            print(f"Collision mode: {self.mujoco_collision_mode.value}")
            print(f"  use_mujoco_contacts: {self._use_mujoco_contacts}")

        self.mujoco_state_0 = self.mujoco_model.state()
        self.mujoco_state_1 = self.mujoco_model.state()
        self.control = self.mujoco_model.control()

        # Evaluate FK into state_0 so it starts with correct body transforms.
        newton.eval_fk(self.mujoco_model, self.mujoco_model.joint_q, self.mujoco_model.joint_qd, self.mujoco_state_0)

        if self._use_mujoco_contacts:
            self.mujoco_contacts = Contacts(0, 0)
        else:
            self.mujoco_contacts = self.collision_pipeline.contacts()
            self.collision_pipeline.collide(self.mujoco_state_0, self.mujoco_contacts)

    # ------------------------------------------------------------------
    # World building: VBD
    # ------------------------------------------------------------------

    def _compute_hose_layout(self):
        """Return shared table-top center and symmetric hose lane y positions."""
        table_top_center = [self.table_pos[0], self.table_pos[1], self.table_pos[2] + self.table_half_size[2]]
        right_hose_y = -self.hose_y_offset
        left_hose_y = -right_hose_y
        return table_top_center, right_hose_y, left_hose_y

    def _compute_capsule_specs(self):
        """Compute capsule transforms and parameters without changing geometry."""
        table_top_center, right_hose_y, left_hose_y = self._compute_hose_layout()

        capsule_radius = self.capsule_radius
        capsule_height = self.capsule_cylinder_height
        tilt_angle_rad = self.capsule_tilt_angle_deg * wp.pi / 180.0

        capsule_total_length = capsule_height + 2.0 * capsule_radius + self.capsule_length_offset
        base_x = self.table_pos[0] - 0.5 * capsule_total_length * wp.sin(tilt_angle_rad) + self.capsule_spawn_x_bias
        base_z = table_top_center[2] + 0.5 * capsule_total_length * wp.cos(tilt_angle_rad)

        capsule_quat = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -tilt_angle_rad)
        capsule_axis = wp.quat_rotate(capsule_quat, wp.vec3(0.0, 0.0, 1.0))

        # Slightly "unseat" capsules from the static STL cradle at t=0.
        spawn_offset = capsule_axis * self.capsule_spawn_axis_offset

        pos_a = wp.vec3(base_x, right_hose_y, base_z) + spawn_offset
        xform_a = wp.transform(pos_a, capsule_quat)

        pos_b = wp.vec3(base_x, left_hose_y, base_z) + spawn_offset
        xform_b = wp.transform(pos_b, capsule_quat)

        return {
            "capsule_radius": capsule_radius,
            "capsule_half_height": 0.5 * capsule_height,
            "capsules": [("test_capsule_a", xform_a), ("test_capsule_b", xform_b)],
            "tilt_angles": [-tilt_angle_rad, -tilt_angle_rad],
            "capsule_axis": capsule_axis,
        }

    @staticmethod
    def _create_cable_points(
        pos_a: wp.vec3,
        pos_b: wp.vec3,
        axis: wp.vec3,
        num_elements: int,
        num_straight_ends: int = 1,
    ) -> list[wp.vec3]:
        """Generate cable polyline from *pos_a* to *pos_b* with a symmetric arch.

        Symmetry is enforced by mirroring about the perpendicular bisector of
        the A-to-B segment.  When *pos_a* and *pos_b* share the same Z
        coordinate the cable start and end will also share the same Z.

        Construction:

        - First *num_straight_ends* segments go straight along *axis*.
        - Last *num_straight_ends* segments are the mirror of the first about
          the perpendicular bisector of A-to-B.
        - Middle: C1-smooth cubic Bezier connecting the two stub tips, with
          control points extending along *axis*.
        - Resampled to approximately uniform segment length.
        - Exact symmetry enforced by mirroring the first half onto the second.

        Returns:
            List of ``num_elements + 1`` ``wp.vec3`` points.
        """
        n_end = int(num_straight_ends)
        if 2 * n_end >= num_elements:
            raise ValueError("num_straight_ends too large for num_elements (need 2*n_end < num_elements)")

        span_vec = pos_b - pos_a
        span_length = float(wp.length(span_vec))
        e_span = wp.normalize(span_vec)
        seg_len = span_length / num_elements

        e_np = np.array([e_span[0], e_span[1], e_span[2]])
        a_np = np.array([pos_a[0], pos_a[1], pos_a[2]])

        def _mirror(p_np: np.ndarray) -> np.ndarray:
            """Mirror *p_np* about the perpendicular bisector of A-to-B."""
            s = np.dot(p_np - a_np, e_np)
            return p_np + (span_length - 2.0 * s) * e_np

        d0 = axis

        num_points = num_elements + 1
        points_np = np.zeros((num_points, 3))

        # First end: straight stubs along d0 from pos_a.
        d0_np = np.array([d0[0], d0[1], d0[2]])
        for i in range(n_end + 1):
            points_np[i] = a_np + i * seg_len * d0_np

        # Last end: mirror of first end about the perpendicular bisector.
        for i in range(n_end + 1):
            points_np[num_elements - i] = _mirror(points_np[i])

        # Bezier middle connecting the two stub tips.
        p_a = points_np[n_end]
        p_b = points_np[num_elements - n_end]
        chord_len = np.linalg.norm(p_b - p_a)
        ctrl = 1.5 * chord_len
        c1 = p_a + ctrl * d0_np
        c2 = _mirror(c1)

        mid_segments = num_elements - 2 * n_end
        for j in range(1, mid_segments):
            u = j / mid_segments
            omu = 1.0 - u
            b0 = omu * omu * omu
            b1 = 3.0 * omu * omu * u
            b2 = 3.0 * omu * u * u
            b3 = u * u * u
            points_np[n_end + j] = b0 * p_a + b1 * c1 + b2 * c2 + b3 * p_b

        # Resample for approximately uniform segment length.
        ds = np.linalg.norm(points_np[1:] - points_np[:-1], axis=1)
        arc = np.concatenate([[0.0], np.cumsum(ds)])
        total = arc[-1]
        if total > 1.0e-12:
            target = np.linspace(0.0, total, num_elements + 1, dtype=float)
            resampled = np.empty((num_elements + 1, 3), dtype=float)
            seg = 0
            for i in range(num_elements + 1):
                ti = target[i]
                while seg + 1 < arc.size and arc[seg + 1] < ti:
                    seg += 1
                if seg + 1 >= arc.size:
                    resampled[i] = points_np[-1]
                else:
                    t0 = arc[seg]
                    t1 = arc[seg + 1]
                    w = 0.0 if t1 <= t0 else (ti - t0) / (t1 - t0)
                    resampled[i] = (1.0 - w) * points_np[seg] + w * points_np[seg + 1]

            # Enforce exact symmetry by mirroring the first half.
            mid = num_elements // 2
            for i in range(mid + 1):
                resampled[num_elements - i] = _mirror(resampled[i])

            points_np = resampled

        return [wp.vec3(p[0], p[1], p[2]) for p in points_np]

    def _create_cable_objects(self, vbd_builder, capsule_specs):
        """Create one cable per capsule, each forming a symmetric arch.

        Each cable replaces its corresponding capsule:

        - The cable starts at the capsule *bottom* (not center), so that the
          first ``cable_num_straight_ends`` segments align with the capsule.
        - The cable end is at the same Y and Z as the start, offset by
          ``cable_span`` in +X, so start and end share the same height.
        - The remaining segments follow a smooth Bezier curve that arches
          symmetrically about the midpoint, mirrored across the perpendicular
          bisector of the span.
        """
        n_seg = self.cable_num_segments
        radius = capsule_specs["capsule_radius"]
        axis = capsule_specs["capsule_axis"]
        half_height = capsule_specs["capsule_half_height"]
        cable_span = self.cable_span

        cable_cfg = vbd_builder.default_shape_cfg.copy()
        cable_cfg.mu = self.vbd_cable_mu
        cable_cfg.margin = self.vbd_cable_margin
        cable_cfg.gap = self.vbd_cable_gap

        self.vbd_capsule_body_ids = []
        self.vbd_cable_all_body_ids = []
        self.cable_grasp_segment_indices = []
        self._cable_grasp_offsets: list[float] = []
        arc_length = 0.0

        capsule_full_length = 2.0 * (half_height + radius)

        for cable_idx, (_key, xform) in enumerate(capsule_specs["capsules"]):
            pos = wp.transform_get_translation(xform)

            # Start from the axis-aligned cable endpoints, then rotate only the
            # near tip about the far tip in the XY plane so the projected near-tip
            # Y offset matches cable_tip_near_y_offset while preserving span length.
            base_start_y = pos[1] - half_height * axis[1]
            base_start_x = pos[0] - half_height * axis[0]
            base_start_z = pos[2] - half_height * axis[2]
            far_tip = wp.vec3(
                base_start_x + cable_span,
                base_start_y,
                base_start_z,
            )

            gripper_side_sign = (-1.0 if cable_idx == 0 else 1.0) * self.sm_spread_direction_sign
            near_tip_side_sign = -gripper_side_sign

            near_tip_y_offset = near_tip_side_sign * self.cable_tip_near_y_offset
            max_dy = 0.999 * cable_span
            dy = np.clip(near_tip_y_offset, -max_dy, max_dy)
            dx = np.sqrt(max(cable_span * cable_span - dy * dy, 0.0))

            cable_start = wp.vec3(
                far_tip[0] - dx,
                far_tip[1] + dy,
                far_tip[2],
            )
            cable_end = far_tip

            cable_points = self._create_cable_points(
                pos_a=cable_start,
                pos_b=cable_end,
                axis=axis,
                num_elements=n_seg,
                num_straight_ends=self.cable_num_straight_ends,
            )
            cable_edge_q = newton.utils.create_parallel_transport_cable_quaternions(cable_points)

            rod_bodies, _rod_joints = vbd_builder.add_rod(
                positions=cable_points,
                quaternions=cable_edge_q,
                radius=radius,
                cfg=cable_cfg,
                stretch_stiffness=self.cable_stretch_stiffness,
                stretch_damping=self.cable_stretch_damping,
                bend_stiffness=self.cable_bend_stiffness,
                bend_damping=self.cable_bend_damping,
                label=f"cable_{cable_idx}",
            )

            arc_length = sum(
                float(wp.length(cable_points[i + 1] - cable_points[i])) for i in range(len(cable_points) - 1)
            )

            # Find the cable segment closest to where the robot would grasp
            # the single capsule.  The capsule grasp point is at fraction
            # sm_grasp_axis_fraction along the capsule's local +Z axis.
            grasp_offset_along_axis = (self.sm_grasp_axis_fraction - 0.5) * capsule_full_length
            capsule_grasp_pt = wp.vec3(
                pos[0] + grasp_offset_along_axis * axis[0],
                pos[1] + grasp_offset_along_axis * axis[1],
                pos[2] + grasp_offset_along_axis * axis[2],
            )

            best_seg = 0
            best_dist = float("inf")
            best_local_offset = 0.0
            for seg_i in range(n_seg):
                seg_start = cable_points[seg_i]
                seg_end = cable_points[seg_i + 1]
                seg_center = wp.vec3(
                    0.5 * (seg_start[0] + seg_end[0]),
                    0.5 * (seg_start[1] + seg_end[1]),
                    0.5 * (seg_start[2] + seg_end[2]),
                )
                d = float(wp.length(seg_center - capsule_grasp_pt))
                if d < best_dist:
                    best_dist = d
                    best_seg = seg_i
                    # Project grasp point onto the segment axis (local +Z).
                    seg_dir = wp.normalize(seg_end - seg_start)
                    seg_half = 0.5 * float(wp.length(seg_end - seg_start))
                    # Offset from the segment body origin (= seg_start) to the
                    # grasp point, along the segment's local Z.
                    proj = float(wp.dot(capsule_grasp_pt - seg_start, seg_dir))
                    best_local_offset = max(0.0, min(proj, 2.0 * seg_half))

            self.vbd_capsule_body_ids.append(rod_bodies[best_seg])
            self.vbd_cable_all_body_ids.append(rod_bodies)
            self.cable_grasp_segment_indices.append(best_seg)
            self._cable_grasp_offsets.append(best_local_offset)

        self.cable_total_length = arc_length
        self.vbd_cable_segment_half_height = arc_length / n_seg / 2.0

        if self.verbose:
            print(
                f"  Created {len(capsule_specs['capsules'])} cables: "
                f"{n_seg} segments each, arc_length={arc_length:.4f} m, "
                f"span={cable_span:.4f} m, straight_ends={self.cable_num_straight_ends}, "
                f"grasp segments={self.cable_grasp_segment_indices}"
            )

    # ------------------------------------------------------------------
    # Effective-mass helpers
    # ------------------------------------------------------------------

    def _build_newton_to_mjc_body_map(self):
        """Build reverse mapping from Newton body ID to MuJoCo body ID.

        Uses ``mjc_body_to_newton`` from the MuJoCo solver (shape ``[nworld, mjc_nbody]``)
        and inverts it for world 0.

        Returns:
            Dict mapping Newton body ID (int) to MuJoCo body ID (int).
        """
        mjc_to_newton_np = self.mujoco_solver.mjc_body_to_newton.numpy()
        newton_to_mjc: dict[int, int] = {}
        mjc_nbody = mjc_to_newton_np.shape[1]
        for mjc_body in range(mjc_nbody):
            newton_body = int(mjc_to_newton_np[0, mjc_body])
            if newton_body >= 0:
                newton_to_mjc[newton_body] = mjc_body
        return newton_to_mjc

    def _compute_proxy_mass(self, newton_body_id, local_mass, newton_to_mjc):
        """Compute proxy body mass based on ``proxy_mass_source``.

        Args:
            newton_body_id: Newton body index.
            local_mass: The body's own mass [kg] from ``body_inv_mass``.
            newton_to_mjc: Reverse mapping from :meth:`_build_newton_to_mjc_body_map`.

        Returns:
            Proxy mass [kg].
        """
        if self.proxy_mass_source == "local":
            return local_mass

        # "effective": use MuJoCo's body_invweight0 (CRB-derived).
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
        """Compute proxy body inertia tensor scaled by effective mass ratio.

        Uses ``invweight0[1]`` (rotational) independently from the translational
        component so that translational and rotational effective inertia are each
        properly represented in the 6x6 spatial inertia M-hat.

        Args:
            newton_body_id: Newton body index.
            local_mass: The body's own mass [kg].
            effective_mass: Proxy mass from :meth:`_compute_proxy_mass` [kg].
            newton_to_mjc: Reverse mapping from :meth:`_build_newton_to_mjc_body_map`.

        Returns:
            ``wp.mat33`` scaled inertia tensor, or ``None`` if unavailable.
        """
        if local_mass <= 0.0 or effective_mass <= 0.0:
            return None

        body_inertia_np = self.mujoco_model.body_inertia.numpy()
        local_inertia = body_inertia_np[newton_body_id]
        local_diag_mean = float(np.mean(np.diag(local_inertia)))
        if local_diag_mean < 1e-30:
            return None

        # Use rotational invweight0 independently when in "effective" mode.
        if self.proxy_mass_source == "effective":
            mjc_body = newton_to_mjc.get(newton_body_id)
            if mjc_body is not None:
                invweight0_np = self.mujoco_solver.mjw_model.body_invweight0.numpy()
                inv_w_rot = float(invweight0_np[0, mjc_body, 1])
                if inv_w_rot > 0.0:
                    effective_rot_inertia = 1.0 / inv_w_rot
                    rot_ratio = effective_rot_inertia / local_diag_mean
                    scaled = local_inertia * rot_ratio
                    if self.verbose:
                        print(
                            f"    Rotational scaling for body {newton_body_id}: "
                            f"eff_rot_inertia={effective_rot_inertia:.6f}, "
                            f"local_diag_mean={local_diag_mean:.6f}, "
                            f"rot_ratio={rot_ratio:.1f}x"
                        )
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

        # Fallback: scale uniformly by translational mass ratio.
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
        """Create proxy bodies in VBD that mirror MuJoCo robot gripper finger bodies.

        Uses the effective mass from MuJoCo's ``body_invweight0``
        (when ``proxy_mass_source == "effective"``) or the body's own mass
        (when ``proxy_mass_source == "local"``), optionally scaled by
        ``proxy_mass_scale``.
        """
        if self.verbose:
            print(f"Creating proxy bodies for {self.coupling_mode} coupling...")
            print(f"  Proxy mass source: {self.proxy_mass_source}")

        # Mapping from MuJoCo body ID to VBD proxy body ID
        # -1 means no proxy (e.g., ground body)
        self.mj_to_vbd_body_map = {}
        self.proxy_body_ids = []  # List of VBD proxy body IDs
        self.proxy_mj_body_ids = []  # List of corresponding MuJoCo body IDs

        # Build Newton -> MuJoCo reverse mapping for effective mass lookup.
        newton_to_mjc = self._build_newton_to_mjc_body_map()

        # Convert arrays to numpy for CPU access
        body_inv_mass_np = self.mujoco_model.body_inv_mass.numpy()
        body_q_np = self.mujoco_state_0.body_q.numpy()
        shape_body_np = self.mujoco_model.shape_body.numpy()
        shape_type_np = self.mujoco_model.shape_type.numpy()
        shape_scale_np = self.mujoco_model.shape_scale.numpy()
        shape_transform_np = self.mujoco_model.shape_transform.numpy()

        # Create proxies for gripper contact-critical bodies (fingers + bases).
        gripper_finger_keys = {
            "right_gripper_base",
            "right_gripper_leftfinger",
            "right_gripper_rightfinger",
            "left_gripper_base",
            "left_gripper_leftfinger",
            "left_gripper_rightfinger",
        }

        # Track per-finger VBD shape ids so we can filter left/right finger self-collisions.
        # This is important when `vbd_proxy_margin` is non-zero: closed fingers can otherwise
        # interpenetrate and generate large proxy-proxy forces unrelated to cable grasping.
        proxy_finger_shape_ids: dict[str, list[int]] = {}

        # VBD-tuned contact parameters for all proxy shapes (ke/kd/mu from the builder's
        # defaults, with overrides for mu and margin specific to proxies).
        proxy_shape_cfg = vbd_builder.default_shape_cfg.copy()
        proxy_shape_cfg.mu = self.vbd_proxy_mu
        proxy_shape_cfg.margin = self.vbd_proxy_margin

        if self.verbose:
            print(f"  Looking for gripper proxy bodies in {self.mujoco_model.body_count} bodies...")
        for mj_body_id in range(self.mujoco_model.body_count):
            # Skip ground body (body 0)
            if mj_body_id == 0:
                self.mj_to_vbd_body_map[mj_body_id] = -1
                continue

            # Only create proxies for gripper contact-critical bodies.
            body_lbl = (
                self.mujoco_model.body_label[mj_body_id] if mj_body_id < len(self.mujoco_model.body_label) else ""
            )
            # URDF labels may be namespaced (e.g. "ns/right_gripper_leftfinger");
            # check the suffix after the last '/'.
            body_short = body_lbl.rsplit("/", 1)[-1] if "/" in body_lbl else body_lbl
            if body_short not in gripper_finger_keys:
                self.mj_to_vbd_body_map[mj_body_id] = -1
                continue

            if self.verbose:
                print(f"  Found gripper proxy body: {body_short} (body {mj_body_id})")

            body_inv_mass = body_inv_mass_np[mj_body_id]
            local_mass = float(1.0 / body_inv_mass) if body_inv_mass > 0 else 0.0

            if local_mass <= 0.0 or local_mass > 1e9:
                self.mj_to_vbd_body_map[mj_body_id] = -1
                continue

            # Compute proxy mass: "effective" uses CRB-derived invweight0, "local" uses body's own.
            effective_mass = self._compute_proxy_mass(mj_body_id, local_mass, newton_to_mjc)
            scaled_mass = effective_mass * self.proxy_mass_scale

            # Compute scaled inertia tensor matching the effective mass.
            proxy_inertia = self._compute_proxy_inertia(mj_body_id, local_mass, effective_mass, newton_to_mjc)

            body_q = body_q_np[mj_body_id]
            initial_xform = wp.transform(
                wp.vec3(body_q[0], body_q[1], body_q[2]), wp.quat(body_q[3], body_q[4], body_q[5], body_q[6])
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
                    radius = float(shape_scale[0])
                    sid = vbd_builder.add_shape_sphere(
                        body=proxy_body_id, radius=radius, pos=pos, rot=rot, cfg=proxy_shape_cfg
                    )
                    shape_ids.append(int(sid))
                    if self.verbose:
                        print(f"    Added SPHERE (r={radius:.3f})")

                elif shape_type == GeoType.BOX:
                    hx, hy, hz = float(shape_scale[0]), float(shape_scale[1]), float(shape_scale[2])
                    sid = vbd_builder.add_shape_box(
                        body=proxy_body_id, hx=hx, hy=hy, hz=hz, pos=pos, rot=rot, cfg=proxy_shape_cfg
                    )
                    shape_ids.append(int(sid))
                    if self.verbose:
                        print(f"    Added BOX ({hx:.3f}, {hy:.3f}, {hz:.3f})")

                elif shape_type == GeoType.CAPSULE:
                    radius = float(shape_scale[0])
                    half_height = float(shape_scale[1])
                    sid = vbd_builder.add_shape_capsule(
                        body=proxy_body_id,
                        radius=radius,
                        half_height=half_height,
                        pos=pos,
                        rot=rot,
                        cfg=proxy_shape_cfg,
                    )
                    shape_ids.append(int(sid))
                    if self.verbose:
                        print(f"    Added CAPSULE (r={radius:.3f}, h={half_height:.3f})")

                elif shape_type == GeoType.MESH:
                    shape_source = self.mujoco_model.shape_source[shape_idx]
                    shape_xform = wp.transform(p=pos, q=rot)
                    sid = vbd_builder.add_shape_mesh(
                        body=proxy_body_id, mesh=shape_source, xform=shape_xform, cfg=proxy_shape_cfg
                    )
                    shape_ids.append(int(sid))
                    if self.verbose:
                        print(f"    Added MESH (source={shape_source})")

                else:
                    if self.verbose:
                        print(f"    WARNING: Unsupported shape type {shape_type}, skipping")

            if not shape_ids:
                if self.verbose:
                    print(f"    WARNING: No shapes found for body {mj_body_id}, adding fallback box")
                sid = vbd_builder.add_shape_box(body=proxy_body_id, hx=0.02, hy=0.01, hz=0.04, cfg=proxy_shape_cfg)
                shape_ids.append(int(sid))

            proxy_finger_shape_ids[body_short] = shape_ids
            self.mj_to_vbd_body_map[mj_body_id] = proxy_body_id
            self.proxy_body_ids.append(proxy_body_id)
            self.proxy_mj_body_ids.append(mj_body_id)

        # Filter self-collisions between all proxy shapes of the same gripper.
        right_gripper_parts = [
            "right_gripper_base",
            "right_gripper_leftfinger",
            "right_gripper_rightfinger",
        ]
        left_gripper_parts = [
            "left_gripper_base",
            "left_gripper_leftfinger",
            "left_gripper_rightfinger",
        ]
        for parts in [right_gripper_parts, left_gripper_parts]:
            for i in range(len(parts)):
                for j in range(i + 1, len(parts)):
                    sa = proxy_finger_shape_ids.get(parts[i], [])
                    sb = proxy_finger_shape_ids.get(parts[j], [])
                    for s1 in sa:
                        for s2 in sb:
                            vbd_builder.add_shape_collision_filter_pair(int(s1), int(s2))

        if self.verbose:
            print(f"  Created {len(self.proxy_body_ids)} proxy bodies")

    def _init_coupling_buffers(self):
        """Initialize proxy buffers and optional two-way force exchange buffers."""

        mujoco_device = wp.get_device(self.mujoco_model.device)
        vbd_device = wp.get_device(self.vbd_model.device)
        if str(mujoco_device) != str(vbd_device):
            raise RuntimeError(
                "Proxy coupling requires MuJoCo and VBD models on the same device. "
                f"Got mujoco_device={mujoco_device}, vbd_device={vbd_device}"
            )
        device = mujoco_device

        # Convert MuJoCo->VBD body map to a dense array for GPU kernels.
        mj_to_vbd_map_list = [-1] * int(self.mujoco_model.body_count)
        for mj_id, vbd_id in self.mj_to_vbd_body_map.items():
            mj_to_vbd_map_list[mj_id] = vbd_id
        self.mj_to_vbd_body_map_array = wp.array(mj_to_vbd_map_list, dtype=int, device=device)

        # Convert proxy ID lists to GPU arrays for kernels.
        self.proxy_vbd_body_ids_array = wp.array(self.proxy_body_ids, dtype=int, device=device)
        self.proxy_mj_body_ids_array = wp.array(self.proxy_mj_body_ids, dtype=int, device=device)

        # Reverse lookup: vbd_body_id -> MuJoCo body id (-1 if not a proxy).
        vbd_to_mj = [-1] * self.vbd_model.body_count
        for vbd_id, mj_id in zip(self.proxy_body_ids, self.proxy_mj_body_ids, strict=True):
            vbd_to_mj[vbd_id] = mj_id
        self.vbd_to_mj_body_map_array = wp.array(vbd_to_mj, dtype=int, device=device)

        if self.enable_two_way_coupling:
            body_force_template = self.mujoco_state_0.body_f
            self.proxy_forces = wp.zeros_like(body_force_template)
            self.coupling_forces_cache = wp.zeros_like(body_force_template)

        if self.verbose:
            print("  Initialized coupling buffers")

    def _setup_vbd_world_and_coupling(self, args):
        """Create VBD world state and optional proxy plumbing for two-way coupling."""

        vbd_builder = newton.ModelBuilder()
        newton.solvers.SolverVBD.register_custom_attributes(vbd_builder)

        vbd_builder.default_shape_cfg.ke = self.vbd_default_contact_ke
        vbd_builder.default_shape_cfg.kd = self.vbd_default_contact_kd
        vbd_builder.default_shape_cfg.gap = self.vbd_default_contact_margin
        vbd_builder.default_shape_cfg.mu = self.vbd_proxy_mu

        capsule_specs = self._compute_capsule_specs()
        self.capsule_body_keys = [name for name, _xform in capsule_specs["capsules"]]
        self.capsule_tilt_angles = capsule_specs["tilt_angles"]

        self.vbd_capsule_radius = capsule_specs["capsule_radius"]
        self.vbd_capsule_half_height = capsule_specs["capsule_half_height"]

        self._create_cable_objects(vbd_builder, capsule_specs)

        # Add static scene geometry to VBD (table + STL connectors + ground plane).
        if not HOSE_CONNECTOR_PATH.exists():
            raise FileNotFoundError(f"Missing STL asset: {HOSE_CONNECTOR_PATH}")

        mesh_vertices, mesh_indices = _load_stl_as_tri_mesh(HOSE_CONNECTOR_PATH)

        scale_factor = 0.001
        mesh_vertices_centered = mesh_vertices * scale_factor
        mesh = newton.Mesh(mesh_vertices_centered, mesh_indices, compute_inertia=True, is_solid=True)
        if self.vbd_mesh_use_sdf:
            mesh.build_sdf(max_resolution=int(self.vbd_mesh_sdf_max_resolution))
        self.vbd_mesh = mesh

        table_top_center, right_hose_y, left_hose_y = self._compute_hose_layout()

        min_z = np.min(mesh_vertices_centered[:, 2]) if mesh_vertices_centered.size else 0.0
        mesh_z = -min_z
        table_offset = wp.vec3(*table_top_center)

        mesh_pos_a = wp.vec3(0.0, right_hose_y, mesh_z) + table_offset
        mesh_pos_b = wp.vec3(0.0, left_hose_y, mesh_z) + table_offset

        mesh_quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -wp.half_pi)
        self.vbd_mesh_xforms = [
            wp.transform(p=mesh_pos_a, q=mesh_quat),
            wp.transform(p=mesh_pos_b, q=mesh_quat),
        ]

        # Add mirrored connectors at the free end of each arch.
        rot_pi_z = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi)
        axis = capsule_specs["capsule_axis"]
        half_ht = capsule_specs["capsule_half_height"]
        orig_xforms = list(self.vbd_mesh_xforms)
        cable_mid_x = 0.0
        for orig_xform, (_key, cap_xform) in zip(orig_xforms, capsule_specs["capsules"], strict=True):
            cap_pos = wp.transform_get_translation(cap_xform)
            cable_start_x = cap_pos[0] - half_ht * axis[0]
            cable_start_y = cap_pos[1] - half_ht * axis[1]
            cable_mid_x = cable_start_x + self.cable_span / 2.0
            cable_mid_y = cable_start_y

            orig_pos = wp.transform_get_translation(orig_xform)
            orig_quat = wp.transform_get_rotation(orig_xform)
            dx = orig_pos[0] - cable_mid_x
            dy = orig_pos[1] - cable_mid_y
            mirror_pos = wp.vec3(
                cable_mid_x - dx,
                cable_mid_y - dy,
                orig_pos[2],
            )
            mirror_quat = rot_pi_z * orig_quat
            self.vbd_mesh_xforms.append(wp.transform(p=mirror_pos, q=mirror_quat))

        self._cable_layout_mid_x = cable_mid_x

        vbd_static_base_cfg = vbd_builder.default_shape_cfg.copy()
        vbd_static_base_cfg.kd = self.vbd_default_contact_kd
        vbd_static_base_cfg.margin = self.vbd_static_margin
        vbd_static_base_cfg.gap = self.vbd_static_gap

        vbd_near_tip_cfg = vbd_static_base_cfg.copy()
        vbd_near_tip_cfg.mu = self.vbd_near_tip_mu

        vbd_far_tip_cfg = vbd_static_base_cfg.copy()
        vbd_far_tip_cfg.mu = self.vbd_far_tip_mu

        vbd_ground_cfg = vbd_static_base_cfg.copy()
        vbd_ground_cfg.mu = self.vbd_ground_mu

        vbd_builder.add_shape_mesh(
            body=-1,
            mesh=mesh,
            xform=self.vbd_mesh_xforms[0],
            cfg=vbd_near_tip_cfg,
            label="rby1_hose_connectorv3_a",
        )
        vbd_builder.add_shape_mesh(
            body=-1,
            mesh=mesh,
            xform=self.vbd_mesh_xforms[1],
            cfg=vbd_near_tip_cfg,
            label="rby1_hose_connectorv3_b",
        )
        vbd_builder.add_shape_mesh(
            body=-1,
            mesh=mesh,
            xform=self.vbd_mesh_xforms[2],
            cfg=vbd_far_tip_cfg,
            label="rby1_hose_connectorv3_a_mirror",
        )
        vbd_builder.add_shape_mesh(
            body=-1,
            mesh=mesh,
            xform=self.vbd_mesh_xforms[3],
            cfg=vbd_far_tip_cfg,
            label="rby1_hose_connectorv3_b_mirror",
        )

        table_box_pos = list(self.table_pos)
        if self._cable_layout_mid_x is not None:
            table_box_pos[0] = self._cable_layout_mid_x
        self.vbd_table_xform = wp.transform(wp.vec3(table_box_pos))
        vbd_builder.add_shape_box(
            body=-1,
            xform=self.vbd_table_xform,
            hx=self.table_half_size[0],
            hy=self.table_half_size[1],
            hz=self.table_half_size[2],
            cfg=vbd_ground_cfg,
        )

        vbd_builder.add_ground_plane(cfg=vbd_ground_cfg)

        if self.enable_robot and self.enable_two_way_coupling:
            self._create_proxy_bodies(vbd_builder)

        vbd_builder.color()

        self.vbd_model = vbd_builder.finalize()
        self.vbd_solver = newton.solvers.SolverVBD(
            self.vbd_model,
            iterations=self.vbd_iterations,
            friction_epsilon=self.vbd_solver_friction_epsilon,
            rigid_avbd_beta=self.vbd_rigid_avbd_beta,
            rigid_contact_k_start=self.vbd_rigid_contact_k_start,
            rigid_body_contact_buffer_size=self.vbd_rigid_contact_buffer_size,
            rigid_body_particle_contact_buffer_size=1,
            rigid_joint_linear_k_start=self.vbd_rigid_joint_linear_k_start,
            rigid_joint_angular_k_start=self.vbd_rigid_joint_angular_k_start,
        )

        self.vbd_state_0 = self.vbd_model.state()
        self.vbd_state_1 = self.vbd_model.state()
        self.vbd_control = self.vbd_model.control()

        self.vbd_collision_pipeline = newton.examples.create_collision_pipeline(
            self.vbd_model,
            args,
        )
        self.vbd_contacts = self.vbd_collision_pipeline.contacts()
        self.vbd_collision_pipeline.collide(self.vbd_state_0, self.vbd_contacts)

        # Initialize proxy/coupling buffers when two-way coupling is active.
        if self.enable_robot and self.enable_two_way_coupling:
            self._init_coupling_buffers()

        if self.verbose:
            total_cable_bodies = sum(len(b) for b in self.vbd_cable_all_body_ids)
            print(
                f"Created VBD world with {self.vbd_model.body_count} bodies, "
                f"{len(self.vbd_cable_all_body_ids)} cables, {total_cable_bodies} segments"
            )
            if self.enable_robot and self.enable_two_way_coupling:
                print(f"  Proxy bodies: {len(self.proxy_body_ids)}")

    # ------------------------------------------------------------------
    # IK and control setup
    # ------------------------------------------------------------------

    def setup_end_effectors(self):
        """Discover end-effector body indices from MuJoCo model labels."""
        ee_body_keys = [
            "right_gripper_end_effector",
            "left_gripper_end_effector",
            "torso_hip_yaw",  # This target helps to keep the robot upright.
        ]

        self.ee_configs = []
        for name in ee_body_keys:
            try:
                idx = _find_label_index(self.mujoco_model.body_label, name)
                self.ee_configs.append((name, idx))
                if self.verbose:
                    print(f"End effector: {name} (body index {idx})")
            except ValueError:
                if self.verbose:
                    print(f"WARNING: End effector label not found: {name}")
                    print(f"  Available labels: {self.mujoco_model.body_label}")

    def setup_ik(self):
        """Set up IK solver with position and rotation objectives for each end effector."""

        def _q2v4(q):
            return wp.vec4(q[0], q[1], q[2], q[3])

        body_q_np = self.mujoco_state_0.body_q.numpy()

        self.ee_tfs = []
        self.pos_objs = []
        self.rot_objs = []

        for _name, link_idx in self.ee_configs:
            tf = wp.transform(*body_q_np[link_idx])
            self.ee_tfs.append(tf)

            self.pos_objs.append(
                ik.IKObjectivePosition(
                    link_index=link_idx,
                    link_offset=wp.vec3(0.0, 0.0, 0.0),
                    target_positions=wp.array([wp.transform_get_translation(tf)], dtype=wp.vec3),
                )
            )

            self.rot_objs.append(
                ik.IKObjectiveRotation(
                    link_index=link_idx,
                    link_offset_rotation=wp.quat_identity(),
                    target_rotations=wp.array([_q2v4(wp.transform_get_rotation(tf))], dtype=wp.vec4),
                )
            )

        # Joint limit arrays must match single_robot_model, not the full scene model.
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

        if self.verbose:
            print(f"IK solver initialized with {len(self.ee_configs)} end effector(s)")

    def setup_gripper_targets(self):
        """Build gripper open/closed target arrays and the IK-merge mask."""
        self.gripper_limits_lower = self.mujoco_model.joint_limit_lower.numpy()[self.gripper_joint_dofs]
        self.gripper_limits_upper = self.mujoco_model.joint_limit_upper.numpy()[self.gripper_joint_dofs]

        # Open values [rad] (used at APPROACH/ENGAGE).
        self.gripper_targets_list = [-0.04, 0.04, -0.04, 0.04]

        # Closed values [rad]: near inner joint limits, slightly inside to avoid chatter.
        eps = 1.0e-4
        gl = self.gripper_limits_lower.astype(np.float64, copy=False)
        gu = self.gripper_limits_upper.astype(np.float64, copy=False)
        self.gripper_closed_values_list = [
            float(gu[0] - eps),  # right_gripper_left_finger_joint (upper)
            float(gl[1] + eps),  # right_gripper_right_finger_joint (lower)
            float(gu[2] - eps),  # left_gripper_left_finger_joint (upper)
            float(gl[3] + eps),  # left_gripper_right_finger_joint (lower)
        ]

        self.gripper_targets = wp.array(self.gripper_targets_list, dtype=wp.float32)

        # Merge mask: gripper_mask[dof] = gripper_index if dof is a gripper joint, else -1.
        gripper_mask_np = [-1] * self.single_robot_model.joint_dof_count
        for gripper_idx, dof_idx in enumerate(self.gripper_joint_dofs):
            gripper_mask_np[dof_idx] = gripper_idx
        self.gripper_mask = wp.array(gripper_mask_np, dtype=wp.int32)

    def setup_state_machine(self):
        """Initialize the state machine for automated capsule grasping.

        Creates GPU arrays so that target-pose computation and task
        advancement run entirely inside Warp kernels.
        """
        self.auto_mode = False
        self.num_arms = NUM_ARMS

        # Both arms follow the same state sequence.
        task_schedule_list = [
            TaskType.APPROACH,
            TaskType.ENGAGE,
            TaskType.GRASP,
            TaskType.EXTRACT,
            TaskType.SIDE_SHIFT,
            TaskType.INJECT,
            TaskType.RELEASE,
            TaskType.DONE,
        ]
        self.num_tasks = len(task_schedule_list)
        self.sm_task_schedule = wp.array(task_schedule_list, dtype=wp.int32)

        # Time limits [s] per task entry.
        task_time_limits_list = [
            self.sm_time_approach,
            self.sm_time_engage,
            self.sm_time_grasp,
            self.sm_time_extract,
            self.sm_time_side_shift,
            self.sm_time_inject,
            self.sm_time_release,
            self.sm_time_done,
        ]
        self.sm_task_time_soft_limits = wp.array(task_time_limits_list, dtype=float)

        # Per-arm mutable state
        self.sm_task_idx = wp.zeros(self.num_arms, dtype=int)
        self.sm_task_time_elapsed = wp.zeros(self.num_arms, dtype=float)

        # Snapshot of each arm's EE transform at the start of the current task.
        body_q_np = self.mujoco_state_0.body_q.numpy()
        init_tfs = []
        for arm_idx in range(self.num_arms):
            _, ee_link_idx = self.ee_configs[arm_idx]
            init_tfs.append(wp.transform(*body_q_np[ee_link_idx]))
        self.sm_task_init_body_q = wp.array(init_tfs, dtype=wp.transform)
        self.sm_home_ee_body_q = wp.array(init_tfs, dtype=wp.transform)

        # Grasp offset [m] specified in the tracked body's local frame, rotated into world by the
        # kernel via `wp.quat_rotate(capsule_quat(_prev), capsule_grasp_offset_from_com[arm_idx])`.
        offsets = [wp.vec3(0.0, 0.0, self._cable_grasp_offsets[arm_idx]) for arm_idx in range(self.num_arms)]
        self.capsule_grasp_offset_from_com = wp.array(offsets, dtype=wp.vec3)

        # Approach offsets [m]: +/-Y lateral stand-off from capsule COM, per arm.
        oy = self.sm_approach_offset_y
        self.approach_offsets = wp.array(
            [
                wp.vec3(0.0, -oy, 0.0),  # Right arm -> Capsule A
                wp.vec3(0.0, oy, 0.0),  # Left arm  -> Capsule B
            ],
            dtype=wp.vec3,
        )

        self.extract_distance = 0.05  # Total pull distance along capsule axis [m]
        self.inject_distance = 1.2 * self.extract_distance  # Inject distance [m]
        self.inject_forward_offset_x = -0.01  # +X nudge during INJECT [m]

        # Per-task convergence thresholds.  GRASP uses larger tolerances so the
        # state machine advances reliably despite contact-force noise.
        pos_thresh = [0.003] * self.num_tasks  # default 3 mm
        rot_thresh = [wp.pi / 180.0] * self.num_tasks  # default 1 deg
        for i, task in enumerate(task_schedule_list):
            if task == TaskType.GRASP:
                pos_thresh[i] = 0.02
                rot_thresh[i] = 5.0 * wp.pi / 180.0
            elif task in (TaskType.EXTRACT, TaskType.SIDE_SHIFT, TaskType.SIDE_SHIFT_BACK):
                pos_thresh[i] = 0.01
                rot_thresh[i] = 3.0 * wp.pi / 180.0
        self.pos_error_threshold = wp.array(pos_thresh, dtype=float)
        self.rot_error_threshold = wp.array(rot_thresh, dtype=float)

        # Capsule body indices (right arm -> capsule A, left arm -> capsule B)
        capsule_a_idx = self.vbd_capsule_body_ids[0]
        capsule_b_idx = self.vbd_capsule_body_ids[1]
        self.sm_capsule_body_indices = wp.array([capsule_a_idx, capsule_b_idx], dtype=int)
        vbd_body_q_np = self.vbd_state_0.body_q.numpy()
        if self.verbose:
            print(
                f"Capsule A  position: {vbd_body_q_np[capsule_a_idx, :3]}, quaternion: {vbd_body_q_np[capsule_a_idx, 3:]}"
            )
            print(
                f"Capsule B  position: {vbd_body_q_np[capsule_b_idx, :3]}, quaternion: {vbd_body_q_np[capsule_b_idx, 3:]}"
            )

        # Snapshot of the capsule transform at the start of the current task
        capsule_tf_a = wp.transform(*vbd_body_q_np[capsule_a_idx])
        capsule_tf_b = wp.transform(*vbd_body_q_np[capsule_b_idx])
        self.sm_task_capsule_body_q_prev = wp.array([capsule_tf_a, capsule_tf_b], dtype=wp.transform)

        # Grasp orientations: 180 deg Y-flip, +/-90 deg Z-yaw, then -36 deg Y-tilt.
        tilt = -wp.pi / 5.0  # -36 deg [rad]
        quat_a_offset = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), wp.pi)
        quat_a_offset = quat_a_offset * wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -wp.half_pi)
        quat_a_offset = quat_a_offset * wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), tilt)

        quat_b_offset = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), wp.pi)
        quat_b_offset = quat_b_offset * wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.half_pi)
        quat_b_offset = quat_b_offset * wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), tilt)

        self.sm_grasp_orientation_offset = wp.array([quat_a_offset, quat_b_offset], dtype=wp.vec4)

        self.sm_gripper_open_values = wp.array(self.gripper_targets_list, dtype=wp.float32)
        self.sm_gripper_closed_values = wp.array(self.gripper_closed_values_list, dtype=wp.float32)
        gripper_dof_pairs = [(-1, -1), (-1, -1)]
        if len(self.gripper_joint_dofs) >= 4:
            gripper_dof_pairs = [
                (int(self.gripper_joint_dofs[0]), int(self.gripper_joint_dofs[1])),
                (int(self.gripper_joint_dofs[2]), int(self.gripper_joint_dofs[3])),
            ]
        self.sm_gripper_dof_indices = wp.array(gripper_dof_pairs, dtype=int)

        self.sm_ee_body_indices = wp.array(
            [self.ee_configs[0][1], self.ee_configs[1][1]],
            dtype=int,
        )

        # Map MuJoCo finger body ids to VBD proxy body ids for centering correction.
        body_label_to_id: dict[str, int] = {}
        if hasattr(self.mujoco_model, "body_label"):
            for i, lbl in enumerate(self.mujoco_model.body_label):
                short = lbl.rsplit("/", 1)[-1] if "/" in lbl else lbl
                body_label_to_id[short] = i
        arm_finger_keys = [
            ("right_gripper_leftfinger", "right_gripper_rightfinger"),
            ("left_gripper_leftfinger", "left_gripper_rightfinger"),
        ]
        mujoco_to_vbd = {}
        if hasattr(self, "proxy_mj_body_ids") and hasattr(self, "proxy_body_ids"):
            for mujoco_body_id, vbd_body_id in zip(self.proxy_mj_body_ids, self.proxy_body_ids, strict=True):
                mujoco_to_vbd[int(mujoco_body_id)] = int(vbd_body_id)
        finger_proxy_ids = []
        for arm_idx in range(self.num_arms):
            fk0, fk1 = arm_finger_keys[arm_idx]
            mujoco_finger0_body_id = int(body_label_to_id.get(fk0, -1))
            mujoco_finger1_body_id = int(body_label_to_id.get(fk1, -1))
            vbd_f0 = int(mujoco_to_vbd.get(mujoco_finger0_body_id, -1))
            vbd_f1 = int(mujoco_to_vbd.get(mujoco_finger1_body_id, -1))
            finger_proxy_ids.append((vbd_f0, vbd_f1))
        self.sm_finger_proxy_body_indices = wp.array(finger_proxy_ids, dtype=int)

        # Centering correction gains: keep the capsule centered between fingers.
        self.gripper_centering_enable = True
        self.gripper_centering_k = 0.4  # closing-axis gain
        self.gripper_axis_centering_k = 0.8  # medial-axis gain
        self.gripper_centering_max_step = 0.003  # max correction per frame [m]

        self.sm_ee_pos_target = wp.zeros(self.num_arms, dtype=wp.vec3)
        self.sm_ee_pos_interp = wp.zeros(self.num_arms, dtype=wp.vec3)
        self.sm_ee_rot_target = wp.zeros(self.num_arms, dtype=wp.vec4)
        self.sm_ee_rot_interp = wp.zeros(self.num_arms, dtype=wp.vec4)
        self.sm_gripper_target = wp.zeros(shape=(self.num_arms, 2), dtype=wp.float32)

        if self.verbose:
            print("State machine initialized (kernel-based)")
            print(f"  Capsule A body index: {capsule_a_idx}")
            print(f"  Capsule B body index: {capsule_b_idx}")

    # ------------------------------------------------------------------
    # State machine lifecycle
    # ------------------------------------------------------------------

    def _start_auto_mode(self):
        """Begin the automated grasping sequence for both arms."""
        self.sm_task_idx.zero_()
        self.sm_task_time_elapsed.zero_()

        # Snapshot current EE and capsule transforms as interpolation start points.
        body_q_np = self.mujoco_state_0.body_q.numpy()
        init_tfs = []
        for arm_idx in range(self.num_arms):
            _, ee_link_idx = self.ee_configs[arm_idx]
            init_tfs.append(wp.transform(*body_q_np[ee_link_idx]))
        wp.copy(self.sm_task_init_body_q, wp.array(init_tfs, dtype=wp.transform))

        vbd_body_q_np = self.vbd_state_0.body_q.numpy()
        capsule_indices = self.sm_capsule_body_indices.numpy()
        capsule_tfs = [wp.transform(*vbd_body_q_np[idx]) for idx in capsule_indices]
        wp.copy(self.sm_task_capsule_body_q_prev, wp.array(capsule_tfs, dtype=wp.transform))

        wp.copy(self.gripper_targets, self.sm_gripper_open_values)

        if self.verbose:
            print("Auto-grasp mode STARTED")

    def _stop_auto_mode(self):
        """Cancel the automated sequence and return to manual GUI control."""
        wp.copy(self.gripper_targets, self.sm_gripper_open_values)
        if self.verbose:
            print("Auto-grasp mode STOPPED")

    def _reset_state_machine(self):
        """Restart the automated sequence from the APPROACH state."""
        if self.auto_mode:
            self._start_auto_mode()

    # ------------------------------------------------------------------
    # Simulation core
    # ------------------------------------------------------------------

    def capture(self):
        self.capture_sim()
        if self.enable_robot:
            self.capture_ik()

    def capture_sim(self):
        """Record a CUDA graph of :meth:`simulate`, restoring state afterward."""
        self.graph_sim = None
        if not self.use_graph:
            return
        if self.enable_robot:
            state_0_backup = self.mujoco_model.state()
            state_1_backup = self.mujoco_model.state()
            state_0_backup.assign(self.mujoco_state_0)
            state_1_backup.assign(self.mujoco_state_1)
        else:
            state_0_backup = None
            state_1_backup = None

        vbd_state_0_backup = self.vbd_model.state()
        vbd_state_1_backup = self.vbd_model.state()
        vbd_state_0_backup.assign(self.vbd_state_0)
        vbd_state_1_backup.assign(self.vbd_state_1)

        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph_sim = capture.graph

        if self.enable_robot:
            self.mujoco_state_0.assign(state_0_backup)
            self.mujoco_state_1.assign(state_1_backup)
        self.vbd_state_0.assign(vbd_state_0_backup)
        self.vbd_state_1.assign(vbd_state_1_backup)

    def capture_ik(self):
        """Record a CUDA graph of the IK solver step."""
        self.graph_ik = None
        if self.use_graph:
            with wp.ScopedCapture() as capture:
                self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iters)
            self.graph_ik = capture.graph

    def simulate(self):
        """Coupling simulation loop (shared substep cadence).

        Implements staggered coupling at every substep:
        1. Apply VBD forces to MuJoCo (from previous substep)
        2. Step MuJoCo
        3. Sync proxy states (MuJoCo -> VBD)
        4. Subtract previously applied forces + gravity from VBD proxies
        5. Step VBD
        6. Harvest new forces from VBD contacts
        """
        substep_count = max(int(self.substeps), 1)
        substep_dt = self.frame_dt / substep_count
        vbd_collide_substeps = max(int(self.vbd_collide_substeps), 1)
        profile = self._profile_timers

        mujoco_state_0 = self.mujoco_state_0
        mujoco_state_1 = self.mujoco_state_1
        vbd_state_0 = self.vbd_state_0
        vbd_state_1 = self.vbd_state_1
        mujoco_collision_step_counter = 0
        vbd_collision_step_counter = 0

        applied_coupling_wrenches = None
        for _substep_idx in range(substep_count):
            # --- Step 1: Apply lagged VBD->MuJoCo wrench ---
            if self.enable_robot:
                mujoco_state_0.clear_forces()
                self.viewer.apply_forces(mujoco_state_0)

                if self.enable_two_way_coupling and hasattr(self, "proxy_forces"):
                    self.coupling_forces_cache.assign(self.proxy_forces)
                    mujoco_state_0.body_f.assign(mujoco_state_0.body_f + self.coupling_forces_cache)
                    applied_coupling_wrenches = self.coupling_forces_cache
                else:
                    applied_coupling_wrenches = None

                # --- Step 2: Step MuJoCo ---
                if profile is not None:
                    wp.synchronize()
                    t0 = time.perf_counter()

                if not self._use_mujoco_contacts and (
                    mujoco_collision_step_counter % self.mujoco_collide_substeps == 0
                ):
                    self.collision_pipeline.collide(mujoco_state_0, self.mujoco_contacts)

                if profile is not None:
                    wp.synchronize()
                    profile["mj_collide"] += time.perf_counter() - t0
                    t0 = time.perf_counter()

                self.mujoco_solver.step(mujoco_state_0, mujoco_state_1, self.control, self.mujoco_contacts, substep_dt)

                if profile is not None:
                    wp.synchronize()
                    profile["mj_step"] += time.perf_counter() - t0

                mujoco_collision_step_counter += 1
                mujoco_state_0, mujoco_state_1 = mujoco_state_1, mujoco_state_0
                self.mujoco_state_0, self.mujoco_state_1 = mujoco_state_0, mujoco_state_1

            # --- Step 3: Sync proxy states (MuJoCo -> VBD) ---
            if profile is not None:
                wp.synchronize()
                t0 = time.perf_counter()

            if self.enable_robot and hasattr(self, "mj_to_vbd_body_map_array"):
                # Sync MuJoCo poses/velocities to VBD proxy bodies.
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

                # Encode the teleportation residual (teleported body_q vs
                # solver body_q_prev) as a velocity correction and reset
                # body_q to body_q_prev, avoiding a position discontinuity
                # that would contaminate VBD's finite-difference damping
                # and friction estimates.
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
            if (
                self.enable_two_way_coupling
                and self.enable_robot
                and applied_coupling_wrenches is not None
                and hasattr(self, "proxy_mj_body_ids_array")
            ):
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

            if profile is not None:
                wp.synchronize()
                profile["sync"] += time.perf_counter() - t0

            # --- Step 5: Step VBD ---
            if profile is not None:
                wp.synchronize()
                t0 = time.perf_counter()

            update_vbd_history = (vbd_collision_step_counter % vbd_collide_substeps == 0) or (self.vbd_contacts is None)
            if update_vbd_history:
                self.vbd_collision_pipeline.collide(vbd_state_0, self.vbd_contacts)

            if profile is not None:
                wp.synchronize()
                profile["vbd_collide"] += time.perf_counter() - t0
                t0 = time.perf_counter()

            self.vbd_solver.set_rigid_history_update(bool(update_vbd_history))
            self.vbd_solver.step(vbd_state_0, vbd_state_1, self.vbd_control, self.vbd_contacts, substep_dt)
            vbd_collision_step_counter += 1

            if profile is not None:
                wp.synchronize()
                profile["vbd_step"] += time.perf_counter() - t0

            # --- Step 6: Harvest new VBD->MuJoCo wrench ---
            if profile is not None:
                wp.synchronize()
                t0 = time.perf_counter()

            if self.enable_two_way_coupling and self.enable_robot and hasattr(self, "proxy_vbd_body_ids_array"):
                self.proxy_forces.zero_()

                if (
                    hasattr(self, "vbd_contacts")
                    and self.vbd_contacts is not None
                    and hasattr(self, "vbd_solver")
                    and hasattr(self.vbd_solver, "collect_rigid_contact_forces")
                ):
                    c_b0, c_b1, c_p0w, c_p1w, c_f_b1, c_count = self.vbd_solver.collect_rigid_contact_forces(
                        vbd_state_1, self.vbd_contacts, substep_dt
                    )
                    wp.launch(
                        harvest_proxy_wrenches_from_contact_forces_kernel,
                        dim=c_b0.shape[0],
                        inputs=[
                            c_count,
                            c_b0,
                            c_b1,
                            c_p0w,
                            c_p1w,
                            c_f_b1,
                            self.vbd_model.body_inv_mass,
                            self.vbd_to_mj_body_map_array,
                            self.mujoco_model.body_com,
                            mujoco_state_0.body_q,
                            self.proxy_forces,
                        ],
                    )

            if profile is not None:
                wp.synchronize()
                profile["harvest"] += time.perf_counter() - t0

            vbd_state_0, vbd_state_1 = vbd_state_1, vbd_state_0
            self.vbd_state_0, self.vbd_state_1 = vbd_state_0, vbd_state_1

    def set_joint_targets(self):
        """Update IK targets, run IK solver, and set joint target positions.

        When ``auto_mode`` is active the two Warp kernels drive the arm EE
        targets and gripper values.  Otherwise the GUI-driven ``ee_tfs`` list
        is used (manual control).
        """
        if self.auto_mode:
            # --- kernel path: compute targets on GPU ---
            wp.launch(
                set_target_pose_kernel,
                dim=self.num_arms,
                inputs=[
                    self.sm_task_schedule,
                    self.sm_task_time_soft_limits,
                    self.sm_task_idx,
                    self.sm_task_time_elapsed,
                    self.frame_dt,
                    self.approach_offsets,
                    self.capsule_grasp_offset_from_com,
                    self.sm_grasp_top_bias,
                    self.extract_distance,
                    self.inject_distance,
                    self.inject_forward_offset_x,
                    self.sm_spread_distance_y,
                    self.sm_spread_direction_sign,
                    self.sm_capsule_body_indices,
                    self.sm_grasp_orientation_offset,
                    self.sm_gripper_open_values,
                    self.sm_gripper_closed_values,
                    self.sm_home_ee_body_q,
                    self.sm_task_init_body_q,
                    self.sm_task_capsule_body_q_prev,
                    self.vbd_state_0.body_q,
                ],
                outputs=[
                    self.sm_ee_pos_target,
                    self.sm_ee_pos_interp,
                    self.sm_ee_rot_target,
                    self.sm_ee_rot_interp,
                    self.sm_gripper_target,
                ],
            )

            # Centering correction: keep capsule centered between finger proxies.
            if self.gripper_centering_enable:
                wp.launch(
                    apply_gripper_centering_correction_kernel,
                    dim=self.num_arms,
                    inputs=[
                        self.sm_task_schedule,
                        self.sm_task_idx,
                        self.vbd_state_0.body_q,
                        self.sm_capsule_body_indices,
                        self.sm_finger_proxy_body_indices,
                        self.gripper_centering_k,
                        self.gripper_axis_centering_k,
                        self.gripper_centering_max_step,
                    ],
                    outputs=[self.sm_ee_pos_interp],
                )

            # Push kernel outputs into IK objectives (arms only, no CPU sync)
            for arm_idx in range(self.num_arms):
                self.pos_objs[arm_idx].set_target_positions(self.sm_ee_pos_interp[arm_idx : arm_idx + 1])
                self.rot_objs[arm_idx].set_target_rotations(self.sm_ee_rot_interp[arm_idx : arm_idx + 1])

            # Torso objective (index 2) is still driven by the GUI
            tf = self.ee_tfs[2]
            self.pos_objs[2].set_target_position(0, wp.transform_get_translation(tf))
            q = wp.transform_get_rotation(tf)
            self.rot_objs[2].set_target_rotation(0, wp.vec4(q[0], q[1], q[2], q[3]))

            wp.copy(self.gripper_targets, self.sm_gripper_target.flatten())

        else:
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

        # Advance state machine after IK so the next sim step uses updated targets.
        if self.auto_mode:
            wp.launch(
                advance_task_kernel,
                dim=self.num_arms,
                inputs=[
                    self.sm_task_time_soft_limits,
                    self.sm_ee_pos_interp,
                    self.sm_ee_rot_interp,
                    self.mujoco_state_0.body_q,
                    self.vbd_state_0.body_q,
                    self.sm_ee_body_indices,
                    self.sm_capsule_body_indices,
                    self.pos_error_threshold,
                    self.rot_error_threshold,
                ],
                outputs=[
                    self.sm_task_idx,
                    self.sm_task_time_elapsed,
                    self.sm_task_init_body_q,
                    self.sm_task_capsule_body_q_prev,
                ],
            )

    def step(self):
        """Run one frame: IK targeting, simulation substeps, and profiling reports."""
        if self.enable_robot:
            self.set_joint_targets()

        if self.graph_sim:
            wp.capture_launch(self.graph_sim)
        else:
            self.simulate()

        if self.graph_sim:
            # With odd substep counts the state double-buffer ends on state_1;
            # copy back so state_0 is always the "current" state for rendering.
            if int(self.substeps) % 2 == 1:
                if self.enable_robot:
                    self.mujoco_state_0.assign(self.mujoco_state_1)
                self.vbd_state_0.assign(self.vbd_state_1)

        self.sim_time += self.frame_dt
        self.frame_count += 1

        if self._profile_timers is not None:
            self._profile_frame_count += 1
            if self._profile_frame_count % self._profile_interval == 0:
                n = self._profile_frame_count
                total = sum(self._profile_timers.values())
                print(f"\n--- Profile ({n} frames, total {total * 1000:.1f} ms) ---")
                for k, v in self._profile_timers.items():
                    pct = 100.0 * v / total if total > 0 else 0.0
                    print(f"  {k:15s}: {v * 1000:8.1f} ms  ({pct:5.1f}%)")
                print(f"  {'avg/frame':15s}: {total / n * 1000:8.1f} ms")
                print()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self):
        """Submit simulation state to the viewer for the current frame."""
        self.viewer.begin_frame(self.sim_time)

        if self.enable_robot:
            if self.viewer_primary_model == "mujoco":
                self.viewer.log_state(self.mujoco_state_0)
                self.viewer.log_contacts(self.mujoco_contacts, self.mujoco_state_0)
                if self.mujoco_collision_mode == CollisionMode.NEWTON_HYDROELASTIC:
                    self.viewer.log_hydro_contact_surface(
                        self.collision_pipeline.get_hydro_contact_surface(), penetrating_only=True
                    )
                self._render_vbd_objects()
            else:
                self.viewer.log_state(self.vbd_state_0)
                self.viewer.log_contacts(self.vbd_contacts, self.vbd_state_0)
                self._render_proxy_bodies()
        else:
            self.viewer.log_state(self.vbd_state_0)
            self.viewer.log_contacts(self.vbd_contacts, self.vbd_state_0)

        self.viewer.end_frame()

    def _render_vbd_objects(self):
        """Render VBD cable segments, table, and mesh connectors when MuJoCo is the primary viewer model."""
        vbd_body_q_np = self.vbd_state_0.body_q.numpy()

        seg_hh = self.vbd_cable_segment_half_height
        cable_xforms = []
        for cable_bodies in self.vbd_cable_all_body_ids:
            for body_idx in cable_bodies:
                bq = vbd_body_q_np[body_idx]
                pos = wp.vec3(bq[0], bq[1], bq[2])
                rot = wp.quat(bq[3], bq[4], bq[5], bq[6])
                center = pos + wp.quat_rotate(rot, wp.vec3(0.0, 0.0, seg_hh))
                cable_xforms.append(wp.transform(center, rot))
        if cable_xforms:
            self.viewer.log_shapes(
                "/vbd_capsules",
                GeoType.CAPSULE,
                (self.vbd_capsule_radius, seg_hh),
                wp.array(cable_xforms, dtype=wp.transform),
                colors=wp.array([wp.vec3(0.2, 0.6, 0.9)], dtype=wp.vec3),
            )

        self.viewer.log_shapes(
            "/vbd_table",
            GeoType.BOX,
            (self.table_half_size[0], self.table_half_size[1], self.table_half_size[2]),
            wp.array([self.vbd_table_xform], dtype=wp.transform),
            colors=wp.array([wp.vec3(0.5, 0.5, 0.5)], dtype=wp.vec3),
        )

        self.viewer.log_shapes(
            "/vbd_mesh_connectors",
            GeoType.MESH,
            1.0,
            wp.array(self.vbd_mesh_xforms, dtype=wp.transform),
            colors=wp.array([wp.vec3(0.6, 0.6, 0.6)], dtype=wp.vec3),
            geo_src=self.vbd_mesh,
        )

    def _render_proxy_bodies(self):
        """Render VBD proxy bodies with distinct color to visualize two-way coupling."""
        if not hasattr(self, "proxy_body_ids") or len(self.proxy_body_ids) == 0:
            return

        vbd_body_q_np = self.vbd_state_0.body_q.numpy()
        shape_body_np = self.vbd_model.shape_body.numpy()
        shape_type_np = self.vbd_model.shape_type.numpy()
        shape_scale_np = self.vbd_model.shape_scale.numpy()

        proxy_color = wp.array([wp.vec3(1.0, 0.5, 0.0)], dtype=wp.vec3)
        scale_factor = 1.05  # slightly enlarge proxies for visibility

        for proxy_vbd_id in self.proxy_body_ids:
            proxy_shapes = [s for s in range(self.vbd_model.shape_count) if shape_body_np[s] == proxy_vbd_id]
            if not proxy_shapes:
                continue

            bq = vbd_body_q_np[proxy_vbd_id]
            proxy_xform = wp.transform(
                wp.vec3(bq[0], bq[1], bq[2]),
                wp.quat(bq[3], bq[4], bq[5], bq[6]),
            )
            xform_arr = wp.array([proxy_xform], dtype=wp.transform)

            for shape_id in proxy_shapes:
                geo_type = shape_type_np[shape_id]
                name = f"/proxy_body_{proxy_vbd_id}_shape_{shape_id}"

                sc = shape_scale_np[shape_id]

                if geo_type == GeoType.SPHERE:
                    self.viewer.log_shapes(
                        name,
                        GeoType.SPHERE,
                        float(sc[0]) * scale_factor,
                        xform_arr,
                        colors=proxy_color,
                    )
                elif geo_type == GeoType.BOX:
                    self.viewer.log_shapes(
                        name,
                        GeoType.BOX,
                        (float(sc[0]) * scale_factor, float(sc[1]) * scale_factor, float(sc[2]) * scale_factor),
                        xform_arr,
                        colors=proxy_color,
                    )
                elif geo_type == GeoType.CAPSULE:
                    self.viewer.log_shapes(
                        name,
                        GeoType.CAPSULE,
                        (float(sc[0]) * scale_factor, float(sc[1]) * scale_factor),
                        xform_arr,
                        colors=proxy_color,
                    )
                elif geo_type == GeoType.MESH:
                    geo_src = self.vbd_model.shape_source[shape_id]
                    mesh = None
                    if isinstance(geo_src, newton.Mesh):
                        mesh = geo_src
                    elif isinstance(geo_src, Integral):
                        mesh_id = int(geo_src)
                        if 0 <= mesh_id < len(self.vbd_model.geo_meshes):
                            mesh = self.vbd_model.geo_meshes[mesh_id]
                    if mesh is not None:
                        self.viewer.log_shapes(
                            name,
                            GeoType.MESH,
                            scale_factor,
                            xform_arr,
                            colors=proxy_color,
                            geo_src=mesh,
                        )

    # ------------------------------------------------------------------
    # GUI
    # ------------------------------------------------------------------

    def gui(self, ui):
        changed, self.show_isosurface = ui.checkbox("Show Isosurface", self.show_isosurface)
        if changed:
            self.viewer.show_hydro_contact_surface = self.show_isosurface

        self.gui_auto_grasp(ui)
        self.gui_gripper_controls(ui)
        self.gui_ee_target_controls(ui)
        self.gui_ik_settings(ui)

    def gui_auto_grasp(self, ui):
        """GUI section for the automated capsule grasping state machine."""
        if not ui.collapsing_header("Auto Grasp", flags=0):
            return

        changed, value = ui.checkbox("Enable Auto Grasp", self.auto_mode)
        if changed:
            self.auto_mode = value
            if self.auto_mode:
                self._start_auto_mode()
            else:
                self._stop_auto_mode()

        if self.auto_mode:
            if ui.button("Reset State Machine"):
                self._reset_state_machine()

        ui.separator()

        # Read GPU arrays once per frame to avoid repeated CPU-GPU syncs.
        arm_labels = ["Right Arm", "Left Arm"]
        capsule_labels = ["Capsule A", "Capsule B"]
        if self.auto_mode:
            task_idx_np = self.sm_task_idx.numpy()
            task_time_np = self.sm_task_time_elapsed.numpy()
            schedule_np = self.sm_task_schedule.numpy()
            time_limits_np = self.sm_task_time_soft_limits.numpy()
            ee_pos_target_np = self.sm_ee_pos_interp.numpy()
            ee_rot_target_np = self.sm_ee_rot_interp.numpy()
            body_q_np = self.mujoco_state_0.body_q.numpy()
            ee_body_indices_np = self.sm_ee_body_indices.numpy()
            for arm_idx in range(self.num_arms):
                idx = int(task_idx_np[arm_idx])
                task_type = TaskType(int(schedule_np[idx]))
                elapsed = float(task_time_np[arm_idx])
                time_limit = float(time_limits_np[idx])
                ee_body_id = int(ee_body_indices_np[arm_idx])
                ee_pos = body_q_np[ee_body_id][:3]
                ee_rot = body_q_np[ee_body_id][3:]  # quaternion xyzw
                pos_err = ee_pos - ee_pos_target_np[arm_idx]
                pos_err_norm = np.linalg.norm(pos_err)
                target_rot = ee_rot_target_np[arm_idx]
                quat_rel = wp.quat(*ee_rot) * wp.quat_inverse(wp.quat(*target_rot))
                rot_err = 2.0 * wp.atan2(wp.length(quat_rel[:3]), wp.abs(quat_rel[3]))
                rot_err_deg = np.degrees(rot_err)
                ui.text(f"{arm_labels[arm_idx]} ({capsule_labels[arm_idx]}): {task_type.name}")
                if task_type != TaskType.DONE:
                    ui.text(f"  Time: {elapsed:.2f} / {time_limit:.1f} s")
                    ui.text(f"  Pos error: {pos_err_norm:.4f} m")
                    ui.text(f"  Pos error XYZ: {pos_err[0]:.4f} m, {pos_err[1]:.4f} m, {pos_err[2]:.4f} m")
                    ui.text(f"  Rot error: {rot_err:.4f} rad ({rot_err_deg:.2f} deg)")
        else:
            for arm_idx in range(self.num_arms):
                ui.text(f"{arm_labels[arm_idx]} ({capsule_labels[arm_idx]}): IDLE")

        ui.separator()

    def gui_ee_target_controls(self, ui):
        """GUI controls for end effector target positions and rotations."""
        if not ui.collapsing_header("End Effector Targets", flags=0):
            return

        min_z_pos = self.table_pos[2] + self.table_half_size[2] + 0.001
        pos_limit_lower = [-1.0, -1.0, min_z_pos]
        pos_limit_upper = [1.0, 1.0, 0.9]

        rot_limit_lower = -np.pi
        rot_limit_upper = np.pi

        def update_ee_position(ee_idx, axis, value):
            """Update a single axis of an end effector's target position."""
            tf = self.ee_tfs[ee_idx]
            pos = list(wp.transform_get_translation(tf))
            pos[axis] = value
            rot = wp.transform_get_rotation(tf)
            self.ee_tfs[ee_idx] = wp.transform(wp.vec3(*pos), rot)

        def update_ee_rotation(ee_idx, axis, value):
            """Update a single axis of an end effector's target rotation (Euler angles)."""
            tf = self.ee_tfs[ee_idx]
            pos = wp.transform_get_translation(tf)
            euler = self._quat_to_euler(wp.transform_get_rotation(tf))
            euler[axis] = value
            self.ee_tfs[ee_idx] = wp.transform(pos, self._euler_to_quat(euler))

        axis_names = ["X", "Y", "Z"]

        for ee_idx, (ee_name, _link_idx) in enumerate(self.ee_configs):
            short_name = ee_name.replace("_end_effector", "").replace("_", " ").title()

            ui.text(f"{short_name}:")
            ui.separator()

            tf = self.ee_tfs[ee_idx]
            pos = wp.transform_get_translation(tf)
            rot = wp.transform_get_rotation(tf)
            euler = self._quat_to_euler(rot)

            for axis in range(3):
                ui.text(f"{short_name} {axis_names[axis]}:")
                changed, value = ui.slider_float(
                    f"{axis_names[axis]}##pos_slider_{ee_idx}_{axis}",
                    pos[axis],
                    pos_limit_lower[axis],
                    pos_limit_upper[axis],
                    format="%.3f",
                )
                if changed:
                    update_ee_position(ee_idx, axis, value)

                changed, value = ui.input_float(
                    f"{axis_names[axis]}##pos_input_{ee_idx}_{axis}",
                    pos[axis],
                    format="%.4f",
                )
                if changed:
                    value = min(max(value, pos_limit_lower[axis]), pos_limit_upper[axis])
                    update_ee_position(ee_idx, axis, value)

            rot_axis_names = ["Roll", "Pitch", "Yaw"]
            for axis in range(3):
                ui.text(f"{short_name} {rot_axis_names[axis]}:")
                changed, value = ui.slider_float(
                    f"{rot_axis_names[axis]}##rot_slider_{ee_idx}_{axis}",
                    euler[axis],
                    rot_limit_lower,
                    rot_limit_upper,
                    format="%.3f",
                )
                if changed:
                    update_ee_rotation(ee_idx, axis, value)

                changed, value = ui.input_float(
                    f"{rot_axis_names[axis]}##rot_input_{ee_idx}_{axis}",
                    euler[axis],
                    format="%.4f",
                )
                if changed:
                    value = min(max(value, rot_limit_lower), rot_limit_upper)
                    update_ee_rotation(ee_idx, axis, value)

            ui.separator()

    def gui_gripper_controls(self, ui):
        """GUI sliders for coupled gripper finger control."""
        if not ui.collapsing_header("Gripper Controls", flags=0):
            return

        def update_gripper_target(joint_idx, value):
            self.gripper_targets_list[joint_idx] = value
            gripper_np = self.gripper_targets.numpy()
            gripper_np[joint_idx] = value
            wp.copy(self.gripper_targets, wp.array(gripper_np, dtype=wp.float32))

        # Both fingers of each gripper move together (coupled).
        ui.text("Coupled Controls:")
        ui.separator()

        # Right gripper (indices 0, 1)
        if len(self.gripper_targets_list) >= 2:
            changed, value = ui.slider_float(
                "Right Gripper",
                self.gripper_targets_list[1],
                self.gripper_limits_lower[1],
                self.gripper_limits_upper[1],
            )
            if changed:
                update_gripper_target(0, -value)
                update_gripper_target(1, value)

        # Left gripper (indices 2, 3)
        if len(self.gripper_targets_list) >= 4:
            changed, value = ui.slider_float(
                "Left Gripper",
                self.gripper_targets_list[3],
                self.gripper_limits_lower[3],
                self.gripper_limits_upper[3],
            )
            if changed:
                update_gripper_target(2, -value)
                update_gripper_target(3, value)

        ui.separator()

    def gui_ik_settings(self, ui):
        if not ui.collapsing_header("IK Settings", flags=0):
            return

        ui.text("End Effectors:")
        for i, (name, idx) in enumerate(self.ee_configs):
            tf = self.ee_tfs[i]
            pos = wp.transform_get_translation(tf)
            ui.text(f"  {name} (body {idx})")
            ui.text(f"    pos: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _quat_to_euler(q):
        """Convert quaternion (x, y, z, w) to intrinsic XYZ Euler angles [rad]."""
        x, y, z, w = q[0], q[1], q[2], q[3]

        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return [roll, pitch, yaw]

    @staticmethod
    def _euler_to_quat(euler):
        """Convert intrinsic XYZ Euler angles [rad] (roll, pitch, yaw) to quaternion (x, y, z, w)."""
        roll, pitch, yaw = euler

        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return wp.quat(x, y, z, w)

    # ------------------------------------------------------------------
    # Testing
    # ------------------------------------------------------------------

    def test_final(self):
        pass


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--num-worlds", type=int, default=1, help="Total number of simulated worlds.")
    parser.add_argument(
        "--coupling-mode",
        type=str,
        choices=("two-way", "none"),
        default="two-way",
        help="Coupling mode. 'two-way' exchanges forces between MuJoCo and VBD; 'none' disables coupling entirely.",
    )
    parser.add_argument(
        "--primary-view",
        type=str,
        choices=("mujoco", "vbd"),
        default="vbd",
        help="Primary model to visualize in the viewer.",
    )
    parser.add_argument(
        "--camera-view",
        type=str,
        choices=("front", "side"),
        default="front",
        help="Preset camera view for ViewerGL.",
    )
    parser.add_argument(
        "--profile-interval",
        type=int,
        default=0,
        help="Print profiling breakdown every N frames. 0 disables profiling.",
    )
    parser.add_argument(
        "--cable-segments",
        type=int,
        default=100,
        help="Number of cable segments per cable. Must be >= 4.",
    )
    parser.add_argument(
        "--cable-straight-ends",
        type=int,
        default=5,
        help="Number of straight cable segments at each end that follow the capsule axis direction.",
    )
    parser.add_argument(
        "--substeps",
        type=int,
        default=10,
        help="Number of coupled substeps per frame (shared by MuJoCo and VBD).",
    )
    parser.add_argument(
        "--proxy-mass-source",
        type=str,
        choices=("effective", "local"),
        default="effective",
        help="Source for proxy body mass. "
        "'effective' uses MuJoCo's body_invweight0 (CRB-derived, accounts for full kinematic chain); "
        "'local' uses the body's own mass (original behavior).",
    )
    viewer, args = newton.examples.init(parser)

    example = Example(viewer, num_worlds=args.num_worlds, args=args)

    newton.examples.run(example, args)
