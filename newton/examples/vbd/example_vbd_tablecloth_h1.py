# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example VBD Tablecloth H1 (unified AVBD/VBD)
#
# A fixed-base Unitree H1 approaches a hanging tablecloth edge from the front,
# curls the support fingers during approach, guides the cloth upward with its
# index fingers, closes each thumb, and follows a smooth task-space pull path
# down and away. Newton IK converts the hand path into joint targets.
# Every H1 collider interacts with the cloth; the twelve thumb/index meshes use
# finer texture-backed SDFs for the pinch.
#
# One SolverVBD instance advances the H1 articulation with AVBD and advances
# the cloth with VBD, including two-way robot-cloth and cloth-tableware
# reactions. No cloth particle positions are scripted or pinned.
#
# Command: python -m newton.examples vbd_tablecloth_h1
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
import newton.utils

PARAMS = {
    # simulation
    "fps": 60,
    "sim_substeps": 16,
    "solver_iterations": 8,
    "gravity": -9.81,
    "num_frames": 300,
    # task-space trajectory
    "settle_time": 0.50,
    "approach_time": 0.80,
    "descend_time": 0.60,
    "insert_time": 0.50,
    "prelift_time": 0.60,
    "close_time": 0.40,
    "lift_time": 0.50,
    "pinch_dwell_time": 0.40,
    "pull_ramp_time": 0.25,
    "pull_speed": 1.50,
    "pull_distance": 0.40,
    "pull_drop_start": 0.08,
    "pull_drop_distance": 0.175,
    # presentation
    "cloth_color": (0.10, 0.34, 0.82),
    "camera_position": (-1.65, -1.80, 1.58),
    "camera_pitch": -9.0,
    "camera_yaw": 45.0,
    "camera_fov": 46.0,
    # compact, H1-reachable table setting
    "table_half_width": 0.20,
    "table_half_depth": 0.36,
    "table_top_z": 1.09,
    "tabletop_half_height": 0.04,
    "cloth_width": 0.46,
    "cloth_grasp_edge_x": -0.24,
    "cloth_depth": 0.70,
    "cloth_resolution": 24,
    "cloth_areal_density": 0.24,
    "cloth_particle_radius": 0.001,
    "cloth_tri_ke": 5.0e4,
    "cloth_tri_ka": 5.0e4,
    "cloth_tri_kd": 5.0e1,
    "cloth_edge_ke": 0.10,
    "cloth_edge_kd": 1.0e-3,
    # rigid-soft contact
    "soft_contact_ke": 1.0e3,
    "soft_contact_kd": 1.0e-2,
    "soft_contact_mu": 0.25,
    "soft_contact_margin": 0.008,
    "enable_water_tight_rigid_soft_contact": True,
    "shape_ke": 1.0e3,
    "shape_kd": 1.0e-4,
    "shape_mu": 0.25,
    "tableware_mu": 0.04,
    # Avoid excessive speculative rigid contacts in this clearance-calibrated grasp.
    "rigid_contact_gap": 0.001,
    # H1 and actual thumb/index mesh contacts
    "robot_base_x": -0.75,
    "grasp_y": 0.24,
    "robot_contact_ke": 1.0e3,
    "robot_contact_kd": 1.0e-2,
    "robot_contact_mu": 0.50,
    "robot_contact_margin": 0.002,
    "finger_contact_margin": 0.002,
    "robot_sdf_padding": 0.012,
    "robot_sdf_max_resolution": 64,
    "finger_sdf_padding": 0.012,
    "finger_contact_ke": 8.0e3,
    "finger_contact_kd": 2.0e1,
    # The fast pull needs a deliberately high effective grasp mu=sqrt(200*0.25)=7.07.
    "finger_contact_mu": 200.0,
    "finger_sdf_max_resolution": 128,
    # Start from the recorded h1_tablecloth_2.usd scoop, then transition to a
    # mesh-calibrated fingertip pinch before closing the thumbs.
    "index_insert_fraction": 0.75,
    "left_index_pinch_fraction": 0.737080,
    "right_index_pinch_fraction": 0.713855,
    "left_thumb_close_fraction": 1.0,
    "right_thumb_close_fraction": 1.0,
    "other_finger_fraction": 0.80,
    # task-space IK targets, ordered [left hand, right hand]
    "rest_x": -0.48,
    "rest_z": 1.24,
    "hover_x": -0.30,
    "hover_z": 1.16,
    "grasp_x": -0.225,
    "left_pinch_x": -0.195,
    "right_pinch_x": -0.195,
    "left_preinsert_z": 1.050,
    "right_preinsert_z": 1.052,
    "pinch_z": 1.110,
    "lift_z": 1.115,
    # AVBD joint drives; NewtonIK only generates their targets
    "joint_drive_ke": 5.0e4,
    "joint_drive_kd": 5.0e2,
    "torso_drive_ke": 2.0e5,
    "torso_drive_kd": 2.0e3,
    "finger_drive_ke": 4.0e4,
    "finger_drive_kd": 1.0e2,
    "joint_target_velocity_limit": 40.0,
    "torso_ik_position_weight": 50.0,
    "torso_ik_rotation_weight": 50.0,
}

HAND_OFFSETS = (
    (0.146273, -0.068447, 0.028077),
    (0.148808, 0.068652, 0.026675),
)

HAND_ROTATIONS = (
    (-0.09, 0.46, 0.03, 0.88),
    (0.09023, 0.46115, -0.03008, 0.88221),
)

# Side-specific angles bring the terminal thumb and index mesh patches within
# 0.15 mm without intersecting them or contacting the sides of the fingers.
THUMB_CLOSED_VALUES = (
    (1.273907, 0.160957, 0.369535, 0.892908),
    (1.192278, 0.195421, 0.400690, 0.679765),
)

_FINGER_GROUP_LEFT_THUMB = wp.constant(0)
_FINGER_GROUP_RIGHT_THUMB = wp.constant(1)
_FINGER_GROUP_LEFT_INDEX = wp.constant(2)
_FINGER_GROUP_RIGHT_INDEX = wp.constant(3)
_FINGER_GROUP_OTHER = wp.constant(4)


@wp.kernel
def set_finger_targets(
    joint_q: wp.array[float],
    finger_indices: wp.array[wp.int32],
    closed_values: wp.array[float],
    finger_groups: wp.array[wp.int32],
    left_thumb_fraction: float,
    right_thumb_fraction: float,
    left_index_fraction: float,
    right_index_fraction: float,
    other_fraction: float,
):
    i = wp.tid()
    group = finger_groups[i]
    fraction = other_fraction
    if group == _FINGER_GROUP_LEFT_THUMB:
        fraction = left_thumb_fraction
    elif group == _FINGER_GROUP_RIGHT_THUMB:
        fraction = right_thumb_fraction
    elif group == _FINGER_GROUP_LEFT_INDEX:
        fraction = left_index_fraction
    elif group == _FINGER_GROUP_RIGHT_INDEX:
        fraction = right_index_fraction
    joint_q[finger_indices[i]] = fraction * closed_values[i]


@wp.kernel
def update_control_targets(
    desired_q: wp.array[float],
    previous_q: wp.array[float],
    inv_dt: float,
    velocity_limit: float,
    target_q: wp.array[float],
    target_qd: wp.array[float],
):
    i = wp.tid()
    q_prev = previous_q[i]
    max_delta = velocity_limit / inv_dt
    delta = wp.clamp(desired_q[i] - q_prev, -max_delta, max_delta)
    q = q_prev + delta
    qd = delta * inv_dt
    target_q[i] = q
    target_qd[i] = qd
    previous_q[i] = q


def _find_suffix(labels: list[str], suffix: str) -> int:
    matches = [i for i, label in enumerate(labels) if label.endswith(f"/{suffix}")]
    if len(matches) != 1:
        raise ValueError(f"Expected one label ending in '/{suffix}', found {len(matches)}")
    return matches[0]


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _normalized_quat(values: tuple[float, float, float, float]) -> wp.quat:
    q = np.asarray(values, dtype=np.float32)
    q /= np.linalg.norm(q)
    return wp.quat(*q)


def _add_h1(
    builder: newton.ModelBuilder,
    params: dict,
) -> tuple[dict[str, int], list[int]]:
    robot_body_start = builder.body_count
    robot_joint_start = builder.joint_count
    robot_dof_start = builder.joint_dof_count
    robot_shape_start = builder.shape_count
    builder.add_mjcf(
        newton.utils.download_asset("unitree_h1") / "mjcf/h1_with_hand.xml",
        xform=wp.transform(wp.vec3(params["robot_base_x"], 0.0, 0.0), wp.quat_identity()),
        floating=False,
        enable_self_collisions=False,
        ctrl_direct=False,
        parse_visuals=True,
        parse_sites=True,
        collider_classes=("collision",),
        no_class_as_colliders=True,
    )
    robot_body_end = builder.body_count
    robot_joint_end = builder.joint_count
    robot_dof_end = builder.joint_dof_count
    robot_shape_end = builder.shape_count

    for dof in range(robot_dof_start, robot_dof_end):
        builder.joint_target_ke[dof] = params["joint_drive_ke"]
        builder.joint_target_kd[dof] = params["joint_drive_kd"]
    torso_joint = _find_suffix(builder.joint_label, "torso_joint")
    torso_dof = builder.joint_qd_start[torso_joint]
    builder.joint_target_ke[torso_dof] = params["torso_drive_ke"]
    builder.joint_target_kd[torso_dof] = params["torso_drive_kd"]
    finger_tokens = tuple(
        f"/{side}_{digit}_" for side in ("L", "R") for digit in ("thumb", "index", "middle", "ring", "pinky")
    )
    for joint in range(robot_joint_start, robot_joint_end):
        if any(token in builder.joint_label[joint] for token in finger_tokens):
            dof = builder.joint_qd_start[joint]
            builder.joint_target_ke[dof] = params["finger_drive_ke"]
            builder.joint_target_kd[dof] = params["finger_drive_kd"]

    body_names = {
        "torso": "torso_link",
        "left_hand": "left_hand_link",
        "right_hand": "right_hand_link",
    }
    body_indices = {name: _find_suffix(builder.body_label, suffix) for name, suffix in body_names.items()}

    # Keep every imported rigid collider active for both AVBD rigid contact and
    # cloth contact. The complete thumb/index chains use a finer texture SDF.
    shape_collision_flag = int(newton.ShapeFlags.COLLIDE_SHAPES)
    particle_collision_flag = int(newton.ShapeFlags.COLLIDE_PARTICLES)
    collision_mask = shape_collision_flag | particle_collision_flag
    grasp_finger_tokens = ("/L_thumb_", "/L_index_", "/R_thumb_", "/R_index_")
    finger_bodies = {
        body
        for body in range(robot_body_start, robot_body_end)
        if any(token in builder.body_label[body] for token in grasp_finger_tokens)
    }
    finger_contact_shape_count = 0
    robot_rigid_shapes = []
    for shape in range(robot_shape_start, robot_shape_end):
        original_flags = int(builder.shape_flags[shape])
        is_rigid_collider = bool(original_flags & shape_collision_flag)
        is_grasp_collider = is_rigid_collider and builder.shape_body[shape] in finger_bodies
        builder.shape_flags[shape] &= ~collision_mask
        if is_rigid_collider:
            builder.shape_flags[shape] |= shape_collision_flag | particle_collision_flag
            builder.shape_gap[shape] = params["rigid_contact_gap"]
            builder.shape_material_ke[shape] = params["robot_contact_ke"]
            builder.shape_material_kd[shape] = params["robot_contact_kd"]
            builder.shape_material_mu[shape] = params["robot_contact_mu"]
            builder.shape_margin[shape] = params["robot_contact_margin"]
            builder.shape_sdf_padding[shape] = params["robot_sdf_padding"]
            builder.shape_sdf_max_resolution[shape] = params["robot_sdf_max_resolution"]
            builder.shape_sdf_target_voxel_size[shape] = None
            robot_rigid_shapes.append(shape)
        if is_grasp_collider:
            builder.shape_material_ke[shape] = params["finger_contact_ke"]
            builder.shape_material_kd[shape] = params["finger_contact_kd"]
            builder.shape_material_mu[shape] = params["finger_contact_mu"]
            builder.shape_margin[shape] = params["finger_contact_margin"]
            builder.shape_sdf_padding[shape] = params["finger_sdf_padding"]
            builder.shape_sdf_max_resolution[shape] = params["finger_sdf_max_resolution"]
            builder.shape_sdf_target_voxel_size[shape] = None
            finger_contact_shape_count += 1

    if finger_contact_shape_count != 12:
        raise RuntimeError(f"Expected 12 H1 thumb/index colliders, found {finger_contact_shape_count}")
    return body_indices, robot_rigid_shapes


def _add_table(builder: newton.ModelBuilder, params: dict):
    table_cfg = newton.ModelBuilder.ShapeConfig(
        ke=params["shape_ke"],
        kd=params["shape_kd"],
        mu=params["shape_mu"],
        gap=params["rigid_contact_gap"],
        has_particle_collision=True,
    )
    top_z = params["table_top_z"]
    half_height = params["tabletop_half_height"]
    wood = wp.vec3(0.46, 0.24, 0.10)
    builder.add_shape_box(
        -1,
        xform=wp.transform(wp.vec3(0.0, 0.0, top_z - half_height), wp.quat_identity()),
        hx=params["table_half_width"],
        hy=params["table_half_depth"],
        hz=half_height,
        cfg=table_cfg,
        color=wood,
    )

    leg_half_width = 0.03
    leg_half_height = 0.5 * (top_z - 2.0 * half_height)
    for x_sign, y_sign in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        builder.add_shape_box(
            -1,
            xform=wp.transform(
                wp.vec3(
                    x_sign * (params["table_half_width"] - 0.05),
                    y_sign * (params["table_half_depth"] - 0.06),
                    leg_half_height,
                ),
                wp.quat_identity(),
            ),
            hx=leg_half_width,
            hy=leg_half_width,
            hz=leg_half_height,
            cfg=table_cfg,
            color=wood,
        )


def _add_tableware(builder: newton.ModelBuilder, z_base: float, params: dict) -> list[int]:
    object_cfg = newton.ModelBuilder.ShapeConfig(
        density=2400.0,
        ke=params["shape_ke"],
        kd=params["shape_kd"],
        mu=params["tableware_mu"],
        gap=params["rigid_contact_gap"],
        has_particle_collision=True,
        margin=0.0,
    )

    plate_half_height = 0.008
    plate = builder.add_body(xform=wp.transform(wp.vec3(0.045, -0.08, z_base + plate_half_height), wp.quat_identity()))
    builder.add_shape_cylinder(
        plate,
        radius=0.065,
        half_height=plate_half_height,
        cfg=object_cfg,
        color=wp.vec3(0.92, 0.91, 0.82),
    )

    glass_half_height = 0.045
    glass = builder.add_body(xform=wp.transform(wp.vec3(0.04, 0.11, z_base + glass_half_height), wp.quat_identity()))
    glass_cfg = object_cfg.copy()
    glass_cfg.density = 2500.0
    builder.add_shape_cylinder(
        glass,
        radius=0.032,
        half_height=glass_half_height,
        cfg=glass_cfg,
        color=wp.vec3(0.52, 0.78, 0.90),
    )

    fork_half_height = 0.004
    fork = builder.add_body(xform=wp.transform(wp.vec3(0.08, -0.20, z_base + fork_half_height), wp.quat_identity()))
    fork_cfg = object_cfg.copy()
    fork_cfg.density = 8000.0
    builder.add_shape_box(
        fork,
        hx=0.060,
        hy=0.009,
        hz=fork_half_height,
        cfg=fork_cfg,
        color=wp.vec3(0.72, 0.74, 0.76),
    )
    return [plate, glass, fork]


def _add_cloth(builder: newton.ModelBuilder, params: dict) -> dict:
    dim_x = params["cloth_resolution"]
    cell_x = params["cloth_width"] / dim_x
    dim_y = round(params["cloth_depth"] / cell_x)
    cell_y = params["cloth_depth"] / dim_y
    particle_mass = (
        params["cloth_areal_density"] * params["cloth_width"] * params["cloth_depth"] / ((dim_x + 1) * (dim_y + 1))
    )
    cloth_z = params["table_top_z"] + params["cloth_particle_radius"] + 0.002
    particle_start = len(builder.particle_q)
    builder.add_cloth_grid(
        pos=wp.vec3(params["cloth_grasp_edge_x"], -0.5 * params["cloth_depth"], cloth_z),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=dim_x,
        dim_y=dim_y,
        cell_x=cell_x,
        cell_y=cell_y,
        mass=particle_mass,
        tri_ke=params["cloth_tri_ke"],
        tri_ka=params["cloth_tri_ka"],
        tri_kd=params["cloth_tri_kd"],
        edge_ke=params["cloth_edge_ke"],
        edge_kd=params["cloth_edge_kd"],
        particle_radius=params["cloth_particle_radius"],
    )
    grasp_edge = np.asarray([particle_start + y * (dim_x + 1) for y in range(dim_y + 1)], dtype=np.int32)
    return {
        "cloth_z": cloth_z,
        "grasp_edge": grasp_edge,
    }


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.params = PARAMS
        self.fps = self.params["fps"]
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = self.params["sim_substeps"]
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.pull_speed = float(args.pull_speed)
        self.pull_ramp_time = float(args.pull_ramp_time)
        if not np.isfinite(self.pull_speed) or not np.isfinite(self.pull_ramp_time):
            raise ValueError("Pull speed and ramp time must be finite")
        if self.pull_speed <= 0.0 or self.pull_ramp_time <= 0.0:
            raise ValueError("Pull speed and ramp time must be positive")
        self.pull_distance = 0.0
        self.phase = "settle"
        self.pull_start_pinch_distances = None
        self.pull_completion_pinch_distances = None

        builder = newton.ModelBuilder(gravity=self.params["gravity"])
        self.robot_bodies, robot_rigid_shapes = _add_h1(builder, self.params)
        self.robot_coord_count = builder.joint_coord_count
        _add_table(builder, self.params)
        self.cloth_info = _add_cloth(builder, self.params)
        self.tableware_bodies = _add_tableware(builder, self.cloth_info["cloth_z"], self.params)
        ground_cfg = newton.ModelBuilder.ShapeConfig(
            ke=self.params["shape_ke"],
            kd=self.params["shape_kd"],
            mu=self.params["shape_mu"],
            gap=self.params["rigid_contact_gap"],
        )
        ground_shape = builder.add_ground_plane(cfg=ground_cfg)
        for robot_shape in robot_rigid_shapes:
            builder.add_shape_collision_filter_pair(robot_shape, ground_shape)
        builder.color(include_bending=True)

        if self.params["enable_water_tight_rigid_soft_contact"]:
            builder.enable_rigid_mesh_sdfs()
        self.model = builder.finalize()
        self.model.soft_contact_ke = self.params["soft_contact_ke"]
        self.model.soft_contact_kd = self.params["soft_contact_kd"]
        self.model.soft_contact_mu = self.params["soft_contact_mu"]

        self.hand_bodies = [self.robot_bodies["left_hand"], self.robot_bodies["right_hand"]]
        self.hand_offsets = [wp.vec3(*values) for values in HAND_OFFSETS]
        self.hand_rotations = [_normalized_quat(values) for values in HAND_ROTATIONS]
        grasp_y = self.params["grasp_y"]
        self.rest_positions = np.asarray(
            [
                (self.params["rest_x"], grasp_y, self.params["rest_z"]),
                (self.params["rest_x"], -grasp_y, self.params["rest_z"]),
            ],
            dtype=np.float32,
        )
        self.hover_positions = np.asarray(
            [
                (self.params["hover_x"], grasp_y, self.params["hover_z"]),
                (self.params["hover_x"], -grasp_y, self.params["hover_z"]),
            ],
            dtype=np.float32,
        )
        self.preinsert_positions = np.asarray(
            [
                (self.params["hover_x"], grasp_y, self.params["left_preinsert_z"]),
                (self.params["hover_x"], -grasp_y, self.params["right_preinsert_z"]),
            ],
            dtype=np.float32,
        )
        grasp_x = self.params["grasp_x"]
        self.insert_positions = np.asarray(
            [
                (grasp_x, grasp_y, self.params["left_preinsert_z"]),
                (grasp_x, -grasp_y, self.params["right_preinsert_z"]),
            ],
            dtype=np.float32,
        )
        self.pinch_positions = np.asarray(
            [
                (self.params["left_pinch_x"], grasp_y, self.params["pinch_z"]),
                (self.params["right_pinch_x"], -grasp_y, self.params["pinch_z"]),
            ],
            dtype=np.float32,
        )
        self.lift_positions = np.asarray(
            [
                (self.params["left_pinch_x"], grasp_y, self.params["lift_z"]),
                (self.params["right_pinch_x"], -grasp_y, self.params["lift_z"]),
            ],
            dtype=np.float32,
        )
        self.target_hand_positions = self.rest_positions.copy()

        self._setup_ik()
        self._solve_ik(
            self.rest_positions,
            left_thumb_fraction=0.0,
            right_thumb_fraction=0.0,
            left_index_fraction=0.0,
            right_index_fraction=0.0,
            other_fraction=0.0,
            iterations=48,
        )
        self.model.joint_q.assign(self.ik_joint_q_flat)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
        self.torso_initial_transform = self.model.body_q.numpy()[self.torso_body].copy()

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="sap",
            soft_contact_margin=self.params["soft_contact_margin"],
            enable_water_tight_rigid_soft_contact=self.params["enable_water_tight_rigid_soft_contact"],
            contact_matching="latest",
        )
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.params["solver_iterations"],
            # AVBD advances the H1 and tableware in the same solve as the VBD cloth.
            integrate_with_external_rigid_solver=False,
            particle_enable_self_contact=False,
            particle_vertex_contact_buffer_size=32,
            particle_edge_contact_buffer_size=64,
            rigid_avbd_contact_alpha=0.0,
            rigid_contact_history=True,
            rigid_body_contact_buffer_size=512,
            rigid_body_particle_contact_buffer_size=2048,
            rigid_joint_linear_ke=1.0e6,
            rigid_joint_angular_ke=1.0e6,
            rigid_joint_linear_kd=1.0e2,
            rigid_joint_angular_kd=1.0e2,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.collision_pipeline.contacts()
        wp.copy(self.state_1.body_q, self.state_0.body_q)
        wp.copy(self.state_1.body_qd, self.state_0.body_qd)
        wp.copy(self.control.joint_target_q, self.model.joint_q, count=self.robot_coord_count)
        self.control.joint_target_qd.zero_()
        self.previous_joint_targets = wp.clone(self.model.joint_q[: self.robot_coord_count])

        particle_q = self.model.particle_q.numpy()
        grasp_edge_indices = self.cloth_info["grasp_edge"]
        self.grasp_edge_rest = particle_q[grasp_edge_indices].copy()
        self.grasp_particle_groups = [
            grasp_edge_indices[np.argsort(np.abs(self.grasp_edge_rest[:, 1] - target_y))[:3]]
            for target_y in (grasp_y, -grasp_y)
        ]
        self.tableware_initial_positions = self.model.body_q.numpy()[self.tableware_bodies, :3].copy()

        self.pull_start_time = sum(
            self.params[name]
            for name in (
                "settle_time",
                "approach_time",
                "descend_time",
                "insert_time",
                "prelift_time",
                "close_time",
                "lift_time",
                "pinch_dwell_time",
            )
        )

        self.viewer.set_model(self.model)
        self.viewer.log_mesh(
            "/model/triangles",
            self.model.particle_q,
            self.model.tri_indices.flatten(),
            hidden=False,
            backface_culling=False,
            color=self.params["cloth_color"],
            roughness=0.9,
        )
        self.viewer.set_camera(
            wp.vec3(*self.params["camera_position"]),
            self.params["camera_pitch"],
            self.params["camera_yaw"],
        )
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = self.params["camera_fov"]

    def _setup_ik(self):
        initial_state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, initial_state)
        body_q = initial_state.body_q.numpy()

        self.torso_body = self.robot_bodies["torso"]
        torso_transform = wp.transform(*body_q[self.torso_body])
        torso_position = wp.transform_get_translation(torso_transform)
        torso_rotation = wp.transform_get_rotation(torso_transform)
        self.torso_position_objective = ik.IKObjectivePosition(
            link_index=self.torso_body,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([torso_position], dtype=wp.vec3),
            weight=self.params["torso_ik_position_weight"],
        )
        self.torso_rotation_objective = ik.IKObjectiveRotation(
            link_index=self.torso_body,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([wp.vec4(*torso_rotation)], dtype=wp.vec4),
            weight=self.params["torso_ik_rotation_weight"],
        )

        self.position_objectives = []
        self.rotation_objectives = []
        for body, offset, rotation in zip(
            self.hand_bodies,
            self.hand_offsets,
            self.hand_rotations,
            strict=True,
        ):
            initial_position = wp.transform_point(wp.transform(*body_q[body]), offset)
            self.position_objectives.append(
                ik.IKObjectivePosition(
                    link_index=body,
                    link_offset=offset,
                    target_positions=wp.array([initial_position], dtype=wp.vec3),
                    weight=5.0,
                )
            )
            self.rotation_objectives.append(
                ik.IKObjectiveRotation(
                    link_index=body,
                    link_offset_rotation=wp.quat_identity(),
                    target_rotations=wp.array([wp.vec4(*rotation)], dtype=wp.vec4),
                    weight=0.2,
                )
            )

        joint_limits = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            weight=1.0,
        )
        self.ik_joint_q = wp.clone(self.model.joint_q).reshape((1, self.model.joint_coord_count))
        self.ik_joint_q_flat = self.ik_joint_q.reshape((-1,))
        self.ik_solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=[
                *self.position_objectives,
                *self.rotation_objectives,
                self.torso_position_objective,
                self.torso_rotation_objective,
                joint_limits,
            ],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

        q_starts = self.model.joint_q_start.numpy()
        finger_indices = []
        closed_values = []
        finger_groups = []
        for side_index, side in enumerate(("L", "R")):
            thumb_yaw, thumb_pitch, thumb_intermediate, thumb_distal = THUMB_CLOSED_VALUES[side_index]
            finger_names_and_values = (
                ("thumb_proximal_yaw_joint", thumb_yaw),
                ("thumb_proximal_pitch_joint", thumb_pitch),
                ("thumb_intermediate_joint", thumb_intermediate),
                ("thumb_distal_joint", thumb_distal),
                ("index_proximal_joint", 1.2),
                ("index_intermediate_joint", 1.2),
                ("middle_proximal_joint", 1.0),
                ("middle_intermediate_joint", 1.0),
                ("ring_proximal_joint", 1.0),
                ("ring_intermediate_joint", 1.0),
                ("pinky_proximal_joint", 1.0),
                ("pinky_intermediate_joint", 1.0),
            )
            for suffix, value in finger_names_and_values:
                joint = _find_suffix(self.model.joint_label, f"{side}_{suffix}")
                finger_indices.append(int(q_starts[joint]))
                closed_values.append(value)
                if suffix.startswith("thumb_"):
                    finger_groups.append(_FINGER_GROUP_LEFT_THUMB if side == "L" else _FINGER_GROUP_RIGHT_THUMB)
                elif suffix.startswith("index_"):
                    finger_groups.append(_FINGER_GROUP_LEFT_INDEX if side == "L" else _FINGER_GROUP_RIGHT_INDEX)
                else:
                    finger_groups.append(_FINGER_GROUP_OTHER)
        self.finger_indices = wp.array(finger_indices, dtype=wp.int32, device=self.model.device)
        self.closed_finger_values = wp.array(closed_values, dtype=float, device=self.model.device)
        self.finger_groups = wp.array(finger_groups, dtype=wp.int32, device=self.model.device)

    def _set_ik_positions(self, positions: np.ndarray):
        for objective, position in zip(self.position_objectives, positions, strict=True):
            objective.set_target_position(0, wp.vec3(*position))

    def _solve_ik(
        self,
        positions: np.ndarray,
        left_thumb_fraction: float,
        right_thumb_fraction: float,
        left_index_fraction: float,
        right_index_fraction: float,
        other_fraction: float,
        iterations: int = 24,
    ):
        self._set_ik_positions(positions)
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=iterations)
        wp.launch(
            set_finger_targets,
            dim=self.finger_indices.shape[0],
            inputs=[
                self.ik_joint_q_flat,
                self.finger_indices,
                self.closed_finger_values,
                self.finger_groups,
                left_thumb_fraction,
                right_thumb_fraction,
                left_index_fraction,
                right_index_fraction,
                other_fraction,
            ],
        )

    def _update_trajectory(self):
        t = self.sim_time
        settle_end = self.params["settle_time"]
        approach_end = settle_end + self.params["approach_time"]
        descend_end = approach_end + self.params["descend_time"]
        insert_end = descend_end + self.params["insert_time"]
        prelift_end = insert_end + self.params["prelift_time"]
        close_end = prelift_end + self.params["close_time"]
        lift_end = close_end + self.params["lift_time"]

        left_thumb_fraction = 0.0
        right_thumb_fraction = 0.0
        left_index_fraction = 0.0
        right_index_fraction = 0.0
        other_fraction = 0.0
        if t < settle_end:
            self.phase = "settle"
            positions = self.rest_positions
        elif t < approach_end:
            self.phase = "approach"
            u = _smoothstep((t - settle_end) / self.params["approach_time"])
            positions = self.rest_positions * (1.0 - u) + self.hover_positions * u
            other_fraction = self.params["other_finger_fraction"] * u
        elif t < descend_end:
            self.phase = "descend"
            u = _smoothstep((t - approach_end) / self.params["descend_time"])
            positions = self.hover_positions * (1.0 - u) + self.preinsert_positions * u
            left_index_fraction = self.params["index_insert_fraction"] * u
            right_index_fraction = self.params["index_insert_fraction"] * u
            other_fraction = self.params["other_finger_fraction"]
        elif t < insert_end:
            self.phase = "insert"
            u = _smoothstep((t - descend_end) / self.params["insert_time"])
            positions = self.preinsert_positions * (1.0 - u) + self.insert_positions * u
            left_index_fraction = self.params["index_insert_fraction"]
            right_index_fraction = self.params["index_insert_fraction"]
            other_fraction = self.params["other_finger_fraction"]
        elif t < prelift_end:
            self.phase = "prelift"
            u = _smoothstep((t - insert_end) / self.params["prelift_time"])
            positions = self.insert_positions * (1.0 - u) + self.pinch_positions * u
            left_index_fraction = (
                self.params["index_insert_fraction"] * (1.0 - u) + self.params["left_index_pinch_fraction"] * u
            )
            right_index_fraction = (
                self.params["index_insert_fraction"] * (1.0 - u) + self.params["right_index_pinch_fraction"] * u
            )
            other_fraction = self.params["other_finger_fraction"]
        elif t < close_end:
            self.phase = "close"
            positions = self.pinch_positions
            u = _smoothstep((t - prelift_end) / self.params["close_time"])
            left_thumb_fraction = self.params["left_thumb_close_fraction"] * u
            right_thumb_fraction = self.params["right_thumb_close_fraction"] * u
            left_index_fraction = self.params["left_index_pinch_fraction"]
            right_index_fraction = self.params["right_index_pinch_fraction"]
            other_fraction = self.params["other_finger_fraction"]
        elif t < lift_end:
            self.phase = "lift"
            u = _smoothstep((t - close_end) / self.params["lift_time"])
            positions = self.pinch_positions * (1.0 - u) + self.lift_positions * u
            left_thumb_fraction = self.params["left_thumb_close_fraction"]
            right_thumb_fraction = self.params["right_thumb_close_fraction"]
            left_index_fraction = self.params["left_index_pinch_fraction"]
            right_index_fraction = self.params["right_index_pinch_fraction"]
            other_fraction = self.params["other_finger_fraction"]
        elif t < self.pull_start_time:
            self.phase = "pinch"
            positions = self.lift_positions
            left_thumb_fraction = self.params["left_thumb_close_fraction"]
            right_thumb_fraction = self.params["right_thumb_close_fraction"]
            left_index_fraction = self.params["left_index_pinch_fraction"]
            right_index_fraction = self.params["right_index_pinch_fraction"]
            other_fraction = self.params["other_finger_fraction"]
        else:
            left_thumb_fraction = self.params["left_thumb_close_fraction"]
            right_thumb_fraction = self.params["right_thumb_close_fraction"]
            left_index_fraction = self.params["left_index_pinch_fraction"]
            right_index_fraction = self.params["right_index_pinch_fraction"]
            other_fraction = self.params["other_finger_fraction"]
            speed_ramp = _smoothstep((t - self.pull_start_time) / self.pull_ramp_time)
            self.pull_distance = min(
                self.pull_distance + speed_ramp * self.pull_speed * self.frame_dt,
                self.params["pull_distance"],
            )
            drop_fraction = 0.0
            if self.pull_distance > self.params["pull_drop_start"]:
                drop_fraction = _smoothstep(
                    (self.pull_distance - self.params["pull_drop_start"])
                    / (self.params["pull_distance"] - self.params["pull_drop_start"])
                )
            offset = np.asarray(
                [
                    -self.pull_distance,
                    0.0,
                    -self.params["pull_drop_distance"] * drop_fraction,
                ],
                dtype=np.float32,
            )
            positions = self.lift_positions + offset
            self.phase = "pull" if self.pull_distance < self.params["pull_distance"] else "hold"

        self.target_hand_positions = np.asarray(positions, dtype=np.float32)
        self._solve_ik(
            self.target_hand_positions,
            left_thumb_fraction,
            right_thumb_fraction,
            left_index_fraction,
            right_index_fraction,
            other_fraction,
        )
        wp.launch(
            update_control_targets,
            dim=self.robot_coord_count,
            inputs=[
                self.ik_joint_q_flat,
                self.previous_joint_targets,
                1.0 / self.frame_dt,
                self.params["joint_target_velocity_limit"],
            ],
            outputs=[self.control.joint_target_q, self.control.joint_target_qd],
        )

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _tracked_pinch_distances(self) -> np.ndarray:
        particle_q = self.state_0.particle_q.numpy()
        body_q = self.state_0.body_q.numpy()
        distances = []
        for body, offset, group in zip(
            self.hand_bodies,
            self.hand_offsets,
            self.grasp_particle_groups,
            strict=True,
        ):
            pinch = np.asarray(wp.transform_point(wp.transform(*body_q[body]), offset))
            distances.append(float(np.max(np.linalg.norm(particle_q[group] - pinch, axis=1))))
        return np.asarray(distances)

    def step(self):
        previous_phase = self.phase
        self._update_trajectory()
        self.simulate()
        if self.phase == "pull" and previous_phase not in ("pull", "hold"):
            self.pull_start_pinch_distances = self._tracked_pinch_distances()
        elif self.phase == "hold" and previous_phase != "hold":
            self.pull_completion_pinch_distances = self._tracked_pinch_distances()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def gui(self, ui):
        ui.text(f"Phase: {self.phase}")
        ui.text(f"Pull distance: {self.pull_distance:.3f} m")
        changed, speed = ui.slider_float("Pull speed [m/s]", self.pull_speed, 0.10, 5.00)
        if changed:
            self.pull_speed = speed
        changed, ramp_time = ui.slider_float("Pull ramp [s]", self.pull_ramp_time, 0.01, 1.00)
        if changed:
            self.pull_ramp_time = ramp_time

    def test_final(self):
        particle_q = self.state_0.particle_q.numpy()
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        assert np.all(np.isfinite(particle_q)), "Cloth state contains non-finite values"
        assert np.all(np.isfinite(body_q)), "Rigid state contains non-finite values"
        assert np.all(np.isfinite(body_qd)), "Rigid velocity contains non-finite values"

        grasp_edge = particle_q[self.cloth_info["grasp_edge"]]
        edge_displacement = float(np.median(self.grasp_edge_rest[:, 0] - grasp_edge[:, 0]))
        assert edge_displacement > 0.10, f"H1 did not pull the cloth edge far enough: {edge_displacement:.3f} m"

        hand_errors = []
        for body, offset, target in zip(self.hand_bodies, self.hand_offsets, self.target_hand_positions, strict=True):
            pinch = np.asarray(wp.transform_point(wp.transform(*body_q[body]), offset))
            hand_errors.append(float(np.linalg.norm(pinch - target)))
        assert max(hand_errors) < 0.12, f"H1 hand tracking error is too large: {max(hand_errors):.3f} m"

        assert self.pull_start_pinch_distances is not None, "H1 never started its pull"
        assert float(np.max(self.pull_start_pinch_distances)) < 0.06, (
            "H1 did not acquire both cloth pinches before pulling: "
            f"maximum tracked edge distance is {float(np.max(self.pull_start_pinch_distances)):.3f} m"
        )
        assert self.pull_completion_pinch_distances is not None, "H1 never completed its pull"
        assert float(np.max(self.pull_completion_pinch_distances)) < 0.06, (
            "H1 released a cloth pinch before completing the pull: "
            f"maximum tracked edge distance is {float(np.max(self.pull_completion_pinch_distances)):.3f} m"
        )
        final_pinch_distances = self._tracked_pinch_distances()
        assert float(np.min(final_pinch_distances)) < 0.06, (
            "H1 released both cloth pinches during the final hold: "
            f"closest tracked edge distance is {float(np.min(final_pinch_distances)):.3f} m"
        )

        torso_q = body_q[self.torso_body]
        torso_position_error = float(np.linalg.norm(torso_q[:3] - self.torso_initial_transform[:3]))
        torso_rotation_dot = float(abs(np.dot(torso_q[3:], self.torso_initial_transform[3:])))
        assert torso_position_error < 0.01, f"H1 torso translated too far: {torso_position_error:.3f} m"
        assert torso_rotation_dot > 0.999, f"H1 torso rotated away from its fixed IK target: {torso_rotation_dot:.5f}"

        tableware_q = body_q[self.tableware_bodies, :3]
        tableware_xy_drift = np.linalg.norm(tableware_q[:, :2] - self.tableware_initial_positions[:, :2], axis=1)
        assert float(np.max(tableware_xy_drift)) < 0.12, (
            f"Tableware moved too far during the pull: {float(np.max(tableware_xy_drift)):.3f} m"
        )
        assert np.all(np.abs(tableware_q[:, 0]) < self.params["table_half_width"] - 0.01), (
            "Tableware left the tabletop in X"
        )
        assert np.all(np.abs(tableware_q[:, 1]) < self.params["table_half_depth"] - 0.01), (
            "Tableware left the tabletop in Y"
        )
        assert np.all(np.abs(tableware_q[:, 2] - self.tableware_initial_positions[:, 2]) < 0.02), (
            "Tableware lifted off or fell below the tabletop"
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--pull-speed",
            type=float,
            default=PARAMS["pull_speed"],
            help="Task-space hand pull speed in meters per second.",
        )
        parser.add_argument(
            "--pull-ramp-time",
            type=float,
            default=PARAMS["pull_ramp_time"],
            help="Seconds spent accelerating to pull speed; smaller gives a sharper launch.",
        )
        parser.set_defaults(num_frames=PARAMS["num_frames"])
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
