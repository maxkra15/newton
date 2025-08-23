# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Newton MPM Shovel Pushing Sand Example

This example demonstrates Material Point Method (MPM) simulation with a shovel-like plate
pushing sand particles. The simulation uses a simplified setup similar to the AnyMal
walking on sand example, with sand particles spawned on the ground and a shovel blade
that pushes through them.

Features:
- Realistic shovel geometry with flat pushing surface
- Sand particles spawned on the ground using the same method as AnyMal example
- Realistic sand density and physics parameters
- Simple, clean simulation setup focused on MPM particle interaction
- Ground plane for particle support

Example usage:
uv run --extra cu12 newton/examples/example_shovel_pushing_sand.py
"""

import sys
import argparse
import numpy as np
import warp as wp

wp.config.enable_backward = False

import newton
from newton.solvers import SolverImplicitMPM


@wp.kernel
def _move_shovel_mesh(
    rest_points: wp.array(dtype=wp.vec3),
    mesh_id: wp.uint64,
    center0: wp.vec3,
    dir_axis: wp.vec3,
    amplitude: float,
    period: float,
    t: float,
    dt: float,
):
    """Move shovel mesh with smooth sinusoidal motion."""
    v = wp.tid()
    mesh = wp.mesh_get(mesh_id)

    # Smooth sinusoidal motion along dir_axis
    s = wp.sin(2.0 * 3.14159 * t / period)

    cur_p = mesh.points[v] + dt * mesh.velocities[v]
    tgt_p = center0 + rest_points[v] + dir_axis * (amplitude * s)
    vel = (tgt_p - cur_p) / dt

    mesh.velocities[v] = vel
    mesh.points[v] = cur_p


@wp.kernel
def _update_body_transform(body_q: wp.array(dtype=wp.transform), body_id: int, new_transform: wp.transform):
    """Update body transform for a specific body ID."""
    if wp.tid() == body_id:
        body_q[body_id] = new_transform


def _spawn_particles(builder: newton.ModelBuilder, res, bounds_lo, bounds_hi, packing_fraction):
    """Spawn particles in a grid pattern with jitter, similar to AnyMal example."""
    Nx = res[0]
    Ny = res[1] 
    Nz = res[2]

    px = np.linspace(bounds_lo[0], bounds_hi[0], Nx + 1)
    py = np.linspace(bounds_lo[1], bounds_hi[1], Ny + 1)
    pz = np.linspace(bounds_lo[2], bounds_hi[2], Nz + 1)

    points = np.stack(np.meshgrid(px, py, pz)).reshape(3, -1).T

    cell_size = (bounds_hi - bounds_lo) / res
    cell_volume = np.prod(cell_size)

    radius = np.max(cell_size) * 0.5
    volume = np.prod(cell_volume) * packing_fraction

    rng = np.random.default_rng()
    points += 2.0 * radius * (rng.random(points.shape) - 0.5)
    vel = np.zeros_like(points)

    builder.particle_q = points
    builder.particle_qd = vel
    builder.particle_mass = np.full(points.shape[0], volume)
    builder.particle_radius = np.full(points.shape[0], radius)
    builder.particle_flags = np.zeros(points.shape[0], dtype=int)

    print("Particle count: ", points.shape[0])


def _make_shovel_mesh(width: float, height: float, thickness: float, center_xyz: np.ndarray) -> wp.Mesh:
    """Create a simple shovel-like plate mesh."""
    cx, cy, cz = center_xyz

    # Create vertices for a simple rectangular shovel blade
    half_width = width / 2
    half_height = height / 2
    half_thickness = thickness / 2

    # Simple box vertices
    vertices = np.array([
        # Front face (pushing surface)
        [-half_width, -half_height, -half_thickness],  # 0: bottom-left-front
        [half_width, -half_height, -half_thickness],   # 1: bottom-right-front
        [half_width, half_height, -half_thickness],    # 2: top-right-front
        [-half_width, half_height, -half_thickness],   # 3: top-left-front
        # Back face
        [-half_width, -half_height, half_thickness],   # 4: bottom-left-back
        [half_width, -half_height, half_thickness],    # 5: bottom-right-back
        [half_width, half_height, half_thickness],     # 6: top-right-back
        [-half_width, half_height, half_thickness],    # 7: top-left-back
    ], dtype=np.float32)

    # Translate to center position
    vertices = vertices + np.array([cx, cy, cz], dtype=np.float32)

    # Define faces for a simple box
    indices = np.array([
        # Front face
        0, 1, 2, 0, 2, 3,
        # Back face
        4, 7, 6, 4, 6, 5,
        # Left face
        0, 3, 7, 0, 7, 4,
        # Right face
        1, 5, 6, 1, 6, 2,
        # Top face
        3, 2, 6, 3, 6, 7,
        # Bottom face
        0, 4, 5, 0, 5, 1
    ], dtype=np.int32)

    # Create warp mesh with proper velocities array
    n_vertices = len(vertices)
    velocities = np.zeros((n_vertices, 3), dtype=np.float32)

    return wp.Mesh(
        points=wp.array(vertices, dtype=wp.vec3),
        velocities=wp.array(velocities, dtype=wp.vec3),
        indices=wp.array(indices.flatten(), dtype=int),
    )


class Example:
    def __init__(
        self,
        stage_path="example_shovel_pushing_sand.usd",
        voxel_size=0.05,
        particles_per_cell=3,
        tolerance=1.0e-5,
        headless=False,
        sand_friction=0.48,
    ):
        self.device = wp.get_device()
        
        # Build the model
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        builder.add_ground_plane()
        builder.gravity = wp.vec3(0.0, -9.81, 0.0)

        # Timing parameters
        self.sim_time = 0.0
        self.sim_step = 0
        fps = 60
        self.frame_dt = 1.0 / fps
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps
        
        # Shovel parameters - larger and more visible
        self.shovel_width = 1.2   # Wider blade for better sand interaction
        self.shovel_height = 0.8  # Taller blade
        self.shovel_thickness = 0.08  # Slightly thicker for visibility
        self.shovel_center = np.array([-1.5, 0.4, 0.0])  # Start position (further back, higher up)
        
        # Create shovel collision mesh FIRST
        self.shovel_mesh = _make_shovel_mesh(
            self.shovel_width,
            self.shovel_height,
            self.shovel_thickness,
            self.shovel_center
        )
        self.shovel_rest_points = wp.array(
            self.shovel_mesh.points.numpy() - self.shovel_center, dtype=wp.vec3
        )

        # Add shovel as kinematic body with EXACT same dimensions as collision mesh
        self.shovel_body_id = builder.add_body(xform=wp.transform(self.shovel_center, wp.quat_identity()))
        builder.add_shape_box(
            self.shovel_body_id,
            hx=self.shovel_thickness * 0.5,  # Exact same dimensions
            hy=self.shovel_height * 0.5,     # Exact same dimensions
            hz=self.shovel_width * 0.5,      # Exact same dimensions
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0),  # kinematic
        )

        # Add sand particles - create a larger, more interesting sand pile
        max_fraction = 1.0
        particle_lo = np.array([-0.8, 0.0, -0.8])  # Larger area
        particle_hi = np.array([3.0, 0.25, 0.8])   # Higher pile, wider spread
        particle_res = np.array(
            np.ceil(particles_per_cell * (particle_hi - particle_lo) / voxel_size),
            dtype=int,
        )

        _spawn_particles(builder, particle_res, particle_lo, particle_hi, max_fraction)

        # Finalize model
        self.model = builder.finalize()
        self.model.particle_mu = sand_friction

        # Setup MPM solver first
        options = SolverImplicitMPM.Options()
        options.voxel_size = voxel_size
        options.max_fraction = max_fraction
        options.tolerance = tolerance
        options.unilateral = False
        options.max_iterations = 50
        options.dynamic_grid = True

        self.mpm_solver = SolverImplicitMPM(self.model, options)
        self.mpm_solver.setup_collider(self.model, [self.shovel_mesh])

        # Create states and enrich them with MPM fields
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # Enrich states with MPM-specific fields
        self.mpm_solver.enrich_state(self.state_0)
        self.mpm_solver.enrich_state(self.state_1)

        # Setup renderer
        self.renderer = None if headless else newton.viewer.RendererOpenGL(self.model, stage_path)

        print(f"Created {self.model.particle_count} sand particles")

    def step(self):
        """Advance simulation by one time step."""
        self.state_0.clear_forces()
        self._update_shovel(self.sim_time, self.sim_dt)
        
        self.mpm_solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += self.sim_dt

    def _update_shovel(self, t, dt):
        """Update shovel position with smooth motion - perfectly synchronized collision and visual."""
        # Motion parameters for dramatic shovel operation
        amplitude = 2.5  # push distance (meters) - longer stroke
        period = 8.0     # period in seconds (slower for better visualization)

        # Calculate current position ONCE using the same formula as the kernel
        s = np.sin(2.0 * np.pi * t / period)
        current_offset = amplitude * s

        # Calculate the exact new position
        new_pos = np.array([
            self.shovel_center[0] + current_offset,
            self.shovel_center[1],
            self.shovel_center[2]
        ])

        # Update collision mesh using the kernel (this will calculate the same position)
        center0 = wp.vec3(self.shovel_center[0], self.shovel_center[1], self.shovel_center[2])
        dir_axis = wp.vec3(1.0, 0.0, 0.0)  # move along X (forward pushing motion)

        wp.launch(
            _move_shovel_mesh,
            dim=self.shovel_rest_points.shape[0],
            inputs=[
                self.shovel_rest_points,
                self.shovel_mesh.id,
                center0,
                dir_axis,
                float(amplitude),
                float(period),
                float(t),
                float(dt),
            ],
        )
        self.shovel_mesh.refit()

        # Update visual body to EXACTLY match collision mesh position
        new_transform = wp.transform(new_pos, wp.quat_identity())

        # Update the body transform in the state using kernel
        if self.shovel_body_id < self.model.body_count:
            wp.launch(
                _update_body_transform,
                dim=self.model.body_count,
                inputs=[self.state_0.body_q, self.shovel_body_id, new_transform],
            )

    def render(self):
        """Render current frame."""
        if self.renderer is not None:
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        default="example_shovel_pushing_sand.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num-frames", type=int, default=10000, help="Total number of frames.")
    parser.add_argument("--voxel-size", "-dx", type=float, default=0.04)  # Slightly larger for better performance
    parser.add_argument("--particles-per-cell", "-ppc", type=float, default=4.0)  # More particles for better visualization
    parser.add_argument("--sand-friction", "-mu", type=float, default=0.6)  # Higher friction for more realistic sand behavior
    parser.add_argument("--tolerance", "-tol", type=float, default=1.0e-5)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction)

    args = parser.parse_known_args()[0]

    if wp.get_device(args.device).is_cpu:
        print("Error: This example requires a GPU device.")
        sys.exit(1)

    with wp.ScopedDevice(args.device):
        example = Example(
            stage_path=args.stage_path,
            voxel_size=args.voxel_size,
            particles_per_cell=args.particles_per_cell,
            tolerance=args.tolerance,
            headless=args.headless,
            sand_friction=args.sand_friction,
        )

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()
