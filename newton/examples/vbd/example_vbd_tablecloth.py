# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example VBD Tablecloth Trick
#
# Five identical table settings compare full-edge tablecloth pulls at
# increasing speeds. The pulled edge clears the tabletop before peeling down
# and away. Water-tight rigid-soft contacts prevent the cloth from tunneling
# through the tableware. Use the viewer sliders to adjust each pull speed
# while the simulation runs.
#
# Command: python -m newton.examples vbd_tablecloth
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples

PARAMS = {
    "pull_speeds": (0.5, 1, 2, 4, 6),
    "pull_distance": 1.25,
    "pull_drop_start": 0.10,
    "pull_drop_distance": 0.30,
    "settle_time": 0.5,
    "fps": 60,
    "sim_substeps": 25,
    "solver_iterations": 10,
    "table_half_width": 0.50,
    "table_half_depth": 0.36,
    "table_top_z": 0.75,
    "tabletop_half_height": 0.04,
    "lane_spacing": 0.95,
    "cloth_width": 1.08,
    "cloth_depth": 0.70,
    "cloth_resolution": 24,
    "cloth_areal_density": 0.24,
    "cloth_particle_radius": 0.0005,
    "cloth_tri_ke": 5.0e4,
    "cloth_tri_ka": 5.0e4,
    "cloth_tri_kd": 5.0e1,
    "cloth_edge_ke": 0.10,
    "cloth_edge_kd": 1.0e-3,
    "soft_contact_ke": 1.0e3,
    "soft_contact_kd": 1.0e-3,
    "soft_contact_mu": 0.25,
    "soft_contact_margin": 0.005,
    "enable_water_tight": True,
    "shape_ke": 1.0e3,
    "shape_kd": 1.0e-4,
    "shape_mu": 0.70,
    "plate_density": 2400.0,
    "glass_density": 2500.0,
    "fork_density": 8000.0,
}


@wp.kernel
def advance_pull_distances(
    speeds: wp.array[float],
    dt: float,
    max_distance: float,
    distances: wp.array[float],
):
    lane = wp.tid()
    distances[lane] = wp.min(distances[lane] + speeds[lane] * dt, max_distance)


@wp.kernel
def move_pulled_edges(
    particle_indices: wp.array[wp.int32],
    lane_indices: wp.array[wp.int32],
    rest_positions: wp.array[wp.vec3],
    speeds: wp.array[float],
    distances: wp.array[float],
    max_distance: float,
    drop_start: float,
    max_drop: float,
    active: int,
    particle_q_0: wp.array[wp.vec3],
    particle_q_1: wp.array[wp.vec3],
    particle_qd_0: wp.array[wp.vec3],
    particle_qd_1: wp.array[wp.vec3],
):
    edge_index = wp.tid()
    particle_index = particle_indices[edge_index]
    lane = lane_indices[edge_index]
    distance = distances[lane]
    velocity = 0.0
    drop = 0.0
    drop_velocity = 0.0
    if distance > drop_start:
        drop_fraction = wp.clamp((distance - drop_start) / (max_distance - drop_start), 0.0, 1.0)
        drop = max_drop * drop_fraction
    if active != 0 and distance < max_distance:
        velocity = speeds[lane]
        if distance > drop_start:
            drop_velocity = speeds[lane] * max_drop / (max_distance - drop_start)

    position = rest_positions[edge_index] + wp.vec3(distance, 0.0, -drop)
    linear_velocity = wp.vec3(velocity, 0.0, -drop_velocity)
    particle_q_0[particle_index] = position
    particle_q_1[particle_index] = position
    particle_qd_0[particle_index] = linear_velocity
    particle_qd_1[particle_index] = linear_velocity


def _add_table(builder: newton.ModelBuilder, lane_y: float, params: dict):
    table_cfg = newton.ModelBuilder.ShapeConfig(
        ke=params["shape_ke"],
        kd=params["shape_kd"],
        mu=params["shape_mu"],
        has_particle_collision=True,
    )
    wood = wp.vec3(0.46, 0.24, 0.10)
    top_half_height = params["tabletop_half_height"]
    top_z = params["table_top_z"]

    builder.add_shape_box(
        -1,
        xform=wp.transform(wp.vec3(0.0, lane_y, top_z - top_half_height), wp.quat_identity()),
        hx=params["table_half_width"],
        hy=params["table_half_depth"],
        hz=top_half_height,
        cfg=table_cfg,
        color=wood,
    )

    leg_half_width = 0.035
    leg_half_height = (top_z - 2.0 * top_half_height) * 0.5
    for x_sign, y_sign in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        builder.add_shape_box(
            -1,
            xform=wp.transform(
                wp.vec3(
                    x_sign * (params["table_half_width"] - 0.07),
                    lane_y + y_sign * (params["table_half_depth"] - 0.07),
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


def _add_tableware(builder: newton.ModelBuilder, lane_y: float, z_base: float, params: dict):
    object_cfg = newton.ModelBuilder.ShapeConfig(
        density=params["plate_density"],
        ke=params["shape_ke"],
        kd=params["shape_kd"],
        mu=params["shape_mu"],
        has_particle_collision=True,
        margin=0.002,
    )

    plate_half_height = 0.012
    plate = builder.add_body(
        xform=wp.transform(wp.vec3(-0.22, lane_y - 0.08, z_base + plate_half_height), wp.quat_identity())
    )
    builder.add_shape_cylinder(
        plate,
        radius=0.105,
        half_height=plate_half_height,
        cfg=object_cfg,
        color=wp.vec3(0.92, 0.91, 0.82),
    )

    glass_half_height = 0.050
    glass = builder.add_body(
        xform=wp.transform(wp.vec3(0.02, lane_y + 0.12, z_base + glass_half_height), wp.quat_identity())
    )
    glass_cfg = object_cfg.copy()
    glass_cfg.density = params["glass_density"]
    builder.add_shape_cylinder(
        glass,
        radius=0.050,
        half_height=glass_half_height,
        cfg=glass_cfg,
        color=wp.vec3(0.52, 0.78, 0.90),
    )

    fork_half_height = 0.006
    fork = builder.add_body(
        xform=wp.transform(wp.vec3(0.22, lane_y - 0.11, z_base + fork_half_height), wp.quat_identity())
    )
    fork_cfg = object_cfg.copy()
    fork_cfg.density = params["fork_density"]
    builder.add_shape_box(
        fork,
        hx=0.110,
        hy=0.016,
        hz=fork_half_height,
        cfg=fork_cfg,
        color=wp.vec3(0.72, 0.74, 0.76),
    )

    return [plate, glass, fork]


def build_scene(builder: newton.ModelBuilder, params: dict):
    dim_x = params["cloth_resolution"]
    cell_x = params["cloth_width"] / dim_x
    dim_y = round(params["cloth_depth"] / cell_x)
    cell_y = params["cloth_depth"] / dim_y
    particle_mass = (
        params["cloth_areal_density"] * params["cloth_width"] * params["cloth_depth"] / ((dim_x + 1) * (dim_y + 1))
    )
    particle_radius = params["cloth_particle_radius"]
    cloth_z = params["table_top_z"] + particle_radius + 0.002

    pulled_indices = []
    pulled_lane_indices = []
    lane_particle_ranges = []
    lane_body_indices = []
    lane_centers = []

    lane_count = len(params["pull_speeds"])
    for lane_index in range(lane_count):
        lane_y = (lane_index - 0.5 * (lane_count - 1)) * params["lane_spacing"]
        lane_centers.append(lane_y)
        _add_table(builder, lane_y, params)

        particle_start = len(builder.particle_q)
        builder.add_cloth_grid(
            pos=wp.vec3(-0.5 * params["cloth_width"], lane_y - 0.5 * params["cloth_depth"], cloth_z),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=dim_x,
            dim_y=dim_y,
            cell_x=cell_x,
            cell_y=cell_y,
            mass=particle_mass,
            fix_right=True,
            tri_ke=params["cloth_tri_ke"],
            tri_ka=params["cloth_tri_ka"],
            tri_kd=params["cloth_tri_kd"],
            edge_ke=params["cloth_edge_ke"],
            edge_kd=params["cloth_edge_kd"],
            particle_radius=particle_radius,
        )
        particle_end = len(builder.particle_q)
        lane_particle_ranges.append((particle_start, particle_end))

        for y in range(dim_y + 1):
            pulled_indices.append(particle_start + y * (dim_x + 1) + dim_x)
            pulled_lane_indices.append(lane_index)

        tableware_z = cloth_z + particle_radius
        lane_body_indices.append(_add_tableware(builder, lane_y, tableware_z, params))

    ground_cfg = newton.ModelBuilder.ShapeConfig(
        ke=params["shape_ke"],
        kd=params["shape_kd"],
        mu=params["shape_mu"],
    )
    builder.add_ground_plane(cfg=ground_cfg)
    builder.color(include_bending=True)

    return {
        "lane_centers": lane_centers,
        "lane_particle_ranges": lane_particle_ranges,
        "lane_body_indices": lane_body_indices,
        "pulled_indices": np.asarray(pulled_indices, dtype=np.int32),
        "pulled_lane_indices": np.asarray(pulled_lane_indices, dtype=np.int32),
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

        builder = newton.ModelBuilder()
        self.info = build_scene(builder, self.params)
        self.model = builder.finalize(enable_water_tight_rigid_soft_contact=self.params["enable_water_tight"])
        self.model.soft_contact_ke = self.params["soft_contact_ke"]
        self.model.soft_contact_kd = self.params["soft_contact_kd"]
        self.model.soft_contact_mu = self.params["soft_contact_mu"]

        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.params["solver_iterations"],
            particle_enable_self_contact=False,
            rigid_avbd_contact_alpha=0.0,
            rigid_contact_history=True,
            rigid_body_contact_buffer_size=256,
            rigid_body_particle_contact_buffer_size=2048,
        )
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="nxn",
            soft_contact_margin=self.params["soft_contact_margin"],
            enable_water_tight_rigid_soft_contact=self.params["enable_water_tight"],
            contact_matching="latest",
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.collision_pipeline.contacts()

        pulled_indices = self.info["pulled_indices"]
        initial_particle_q = self.model.particle_q.numpy()
        self.pulled_indices = wp.array(pulled_indices, dtype=wp.int32)
        self.pulled_lane_indices = wp.array(self.info["pulled_lane_indices"], dtype=wp.int32)
        self.pulled_rest_positions = wp.array(initial_particle_q[pulled_indices], dtype=wp.vec3)
        self.pull_speeds = list(self.params["pull_speeds"])
        self.pull_speeds_device = wp.array(self.pull_speeds, dtype=float)
        self.pull_distances = wp.zeros(len(self.pull_speeds), dtype=float)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(4.3, -5.7, 3.4), pitch=-20.0, yaw=128.0)
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = 62.0

    def simulate(self):
        for substep in range(self.sim_substeps):
            pull_active = self.sim_time + (substep + 1) * self.sim_dt >= self.params["settle_time"]
            if pull_active:
                wp.launch(
                    advance_pull_distances,
                    dim=len(self.pull_speeds),
                    inputs=[self.pull_speeds_device, self.sim_dt, self.params["pull_distance"]],
                    outputs=[self.pull_distances],
                )
            wp.launch(
                move_pulled_edges,
                dim=self.pulled_indices.shape[0],
                inputs=[
                    self.pulled_indices,
                    self.pulled_lane_indices,
                    self.pulled_rest_positions,
                    self.pull_speeds_device,
                    self.pull_distances,
                    self.params["pull_distance"],
                    self.params["pull_drop_start"],
                    self.params["pull_drop_distance"],
                    int(pull_active),
                ],
                outputs=[
                    self.state_0.particle_q,
                    self.state_1.particle_q,
                    self.state_0.particle_qd,
                    self.state_1.particle_qd,
                ],
            )

            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def gui(self, ui):
        ui.text("Tablecloth pull speeds")
        changed_any = False
        for lane, speed in enumerate(self.pull_speeds):
            changed, value = ui.slider_float(f"Table {lane + 1}", speed, 0.0, 6.0)
            if changed:
                self.pull_speeds[lane] = value
                changed_any = True
        if changed_any:
            self.pull_speeds_device.assign(np.asarray(self.pull_speeds, dtype=np.float32))

    def test_final(self):
        particle_q = self.state_0.particle_q.numpy()
        body_q = self.state_0.body_q.numpy()
        pulled_q = particle_q[self.info["pulled_indices"]]
        pulled_rest = self.pulled_rest_positions.numpy()
        pulled_distance = pulled_q[:, 0] - pulled_rest[:, 0]
        assert np.min(pulled_distance) > 0.99 * self.params["pull_distance"], (
            f"Pulled edge did not complete its stroke: min distance={np.min(pulled_distance):.3f}"
        )

        on_table_counts = []
        for lane, body_indices in enumerate(self.info["lane_body_indices"]):
            positions = body_q[body_indices, :3]
            on_table = np.logical_and.reduce(
                (
                    np.abs(positions[:, 0]) < self.params["table_half_width"],
                    np.abs(positions[:, 1] - self.info["lane_centers"][lane]) < self.params["table_half_depth"],
                    positions[:, 2] > self.params["table_top_z"] - 0.02,
                )
            )
            on_table_counts.append(int(np.count_nonzero(on_table)))

        assert on_table_counts[0] == 0, f"Slowest pull left {on_table_counts[0]}/3 objects on the table"
        fastest_lane = len(self.pull_speeds) - 1
        assert on_table_counts[fastest_lane] == 3, (
            f"Fastest pull retained only {on_table_counts[fastest_lane]}/3 objects"
        )

        particle_start, particle_end = self.info["lane_particle_ranges"][fastest_lane]
        fastest_cloth = particle_q[particle_start:particle_end]
        off_table_fraction = np.mean(fastest_cloth[:, 0] > self.params["table_half_width"])
        assert off_table_fraction > 0.65, (
            f"Fastest cloth did not clear the tabletop: off-table fraction={off_table_fraction:.1%}"
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=240)
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
