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
- Modern ViewerGL implementation for better performance

Example usage:
uv run --extra cu12 newton/examples/example_mpm_pushing_soil.py
"""

import numpy as np
import warp as wp

import newton
import newton.examples
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
    def __init__(self, viewer, options):
        # setup simulation parameters first
        self.fps = options.fps
        self.frame_dt = 1.0 / self.fps

        # group related attributes by prefix
        self.sim_time = 0.0
        self.sim_substeps = options.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        # save a reference to the viewer
        self.viewer = viewer

        # Build the model
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.gravity = wp.vec3(options.gravity)

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

        # Add sand particles using the same pattern as granular example
        Example.emit_particles(builder, options)

        # Finalize model
        self.model = builder.finalize()
        self.model.particle_mu = options.friction_coeff

        # Create states
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # Setup MPM solver with optimized options
        options.grid_padding = 0 if options.dynamic_grid else 5
        options.yield_stresses = wp.vec3(
            options.yield_stress,
            -options.stretching_yield_stress,
            options.compression_yield_stress,
        )

        self.solver = SolverImplicitMPM(self.model, options)
        self.solver.setup_collider(self.model, colliders=[self.shovel_mesh])

        # Enrich states with MPM-specific fields
        self.solver.enrich_state(self.state_0)
        self.solver.enrich_state(self.state_1)

        # Setup viewer
        self.viewer.set_model(self.model)
        self.viewer.show_particles = True

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self._update_shovel(self.sim_time, self.sim_dt)
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test(self):
        pass

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

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

    @staticmethod
    def emit_particles(builder: newton.ModelBuilder, args):
        max_fraction = args.max_fraction
        voxel_size = args.voxel_size

        particles_per_cell = 3
        particle_lo = np.array(args.emit_lo)
        particle_hi = np.array(args.emit_hi)
        particle_res = np.array(
            np.ceil(particles_per_cell * (particle_hi - particle_lo) / voxel_size),
            dtype=int,
        )

        Example._spawn_particles(builder, particle_res, particle_lo, particle_hi, max_fraction)

    @staticmethod
    def _spawn_particles(
        builder: newton.ModelBuilder,
        res,
        bounds_lo,
        bounds_hi,
        packing_fraction,
    ):
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


def _create_collider_mesh(collider: str):
    """Create a collider mesh."""
    if collider == "cube":
        cube_points, cube_indices = newton.utils.create_box_mesh(extents=(0.5, 2.0, 1.0))
        return wp.Mesh(
            wp.array(cube_points[:, 0:3] + [0, 0, 0.5], dtype=wp.vec3),
            wp.array(cube_indices, dtype=int),
        )
    else:
        return None


if __name__ == "__main__":
    import argparse

    # Create parser that inherits common arguments and adds example-specific ones
    parser = newton.examples.create_parser()

    # Add MPM-specific arguments
    parser.add_argument("--emit-lo", type=float, nargs=3, default=[-0.8, 0.0, -0.8])
    parser.add_argument("--emit-hi", type=float, nargs=3, default=[3.0, 0.25, 0.8])
    parser.add_argument("--gravity", type=float, nargs=3, default=[0, -9.81, 0])
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--substeps", type=int, default=1)

    parser.add_argument("--max-fraction", type=float, default=1.0)

    parser.add_argument("--compliance", type=float, default=0.0)
    parser.add_argument("--poisson-ratio", "-nu", type=float, default=0.3)
    parser.add_argument("--friction-coeff", "-mu", type=float, default=0.6)
    parser.add_argument("--yield-stress", "-ys", type=float, default=0.0)
    parser.add_argument("--compression-yield-stress", "-cys", type=float, default=1.0e8)
    parser.add_argument("--stretching-yield-stress", "-sys", type=float, default=1.0e8)
    parser.add_argument("--unilateral", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dynamic-grid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gauss-seidel", "-gs", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--max-iterations", "-it", type=int, default=250)
    parser.add_argument("--tolerance", "-tol", type=float, default=1.0e-5)
    parser.add_argument("--voxel-size", "-dx", type=float, default=0.05)

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Create example and run
    example = Example(viewer, args)

    newton.examples.run(example)
