# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Newton MPM Sand Plow Example

This example demonstrates Material Point Method (MPM) simulation with a plow
pushing through sand particles. Features a fixed plate geometry with configurable
orientation and placement, realistic sand physics, and smooth plow motion.

Features:
- Realistic plow geometry with configurable pitch angle
- Sand particles spawned in a bed configuration
- Smooth plow motion along X-axis
- Modern ViewerGL implementation for better performance
- Optimized MPM solver configuration

Example usage:
python newton/newton/examples/example_sand_plow.py --viewer gl --plow-speed 0.5
"""

import math
import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverImplicitMPM


# ------- kernels -------
@wp.kernel
def update_body_transform(
    body_q: wp.array(dtype=wp.transform),
    body_id: int,
    new_transform: wp.transform,
):
    tid = wp.tid()
    if tid == body_id:
        body_q[tid] = new_transform





# ------- helpers -------
def mat3_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    a = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = a
    c, s, C = math.cos(angle_rad), math.sin(angle_rad), 1.0 - math.cos(angle_rad)
    return np.array([
        [x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, z*z*C + c  ],
    ], dtype=np.float32)


# ------- helpers -------
# (Using standard Newton box shapes instead of custom mesh creation)





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
        self.renderer = None  # Will be set up later

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        builder.add_ground_plane()
        builder.gravity = wp.vec3(options.gravity)

        # --- sand bed: fine sand configuration ---
        # Set emit bounds for sand bed (finer, more realistic sand depth)
        soil_w = 4.0
        ground_clearance = options.voxel_size * 0.05  # Minimal clearance for fine sand
        sand_depth = 0.20  # Higher sand pile for better plow interaction
        options.emit_lo = [-soil_w/2, ground_clearance, -0.30]
        options.emit_hi = [ soil_w/2, sand_depth + ground_clearance,  2.00]

        # Use exact same particle emission as granular example
        Example.emit_particles(builder, options)

        # motion bounds (left->right along X) - for visual body init
        self.x0 = -soil_w/2.0  # start before the longer pile
        self.x1 = +soil_w/2.0  # end after the longer pile
        self.plow_y = 0.05  # Raised plow slightly higher from ground
        self.plow_z = 1.0  # back to original Z position
        self.plow_speed = float(options.plow_speed)
        self.plow_finished = False  # track if plow has completed its pass

        # --- plow geometry: clean L-shape with 90-degree intersection ---
        plow_width = 1.2        # span across Z
        plow_thickness = 0.08   # thickness for both components

        # Bottom plate: horizontal pushing surface
        bottom_length = 0.50    # length along X (travel direction)
        bottom_center = np.array([0.0, 0.16, 0.0], dtype=np.float32)
        bottom_size = np.array([bottom_length, plow_thickness, plow_width], dtype=np.float32)

        # Top plate: vertical wall - positioned to create clean L-shape
        top_height = 0.60       # height along Y
        # Position top plate so it intersects cleanly at the back edge of bottom plate
        top_center = np.array([
            -bottom_length/2 + plow_thickness/2,  # Back edge of bottom plate
            bottom_center[1] + plow_thickness/2 + top_height/2,  # Above bottom plate
            0.0
        ], dtype=np.float32)
        top_size = np.array([plow_thickness, top_height, plow_width], dtype=np.float32)

        # ==== Build plow from individual box colliders ====
        pitch = math.radians(options.plow_pitch_deg)  # positive means "downward" now
        Rz = mat3_from_axis_angle(np.array([0,0,1], dtype=np.float32), pitch)

        # Store the rotation matrix and plow parameters for motion updates
        self.plow_rotation = Rz
        self.bottom_center = bottom_center
        self.top_center = top_center
        self.bottom_size = bottom_size
        self.top_size = top_size

        # Helper function to convert numpy rotation matrix to warp
        def mat33_from_np_rowmajor(M: np.ndarray) -> wp.mat33:
            return wp.mat33(
                float(M[0,0]), float(M[0,1]), float(M[0,2]),
                float(M[1,0]), float(M[1,1]), float(M[1,2]),
                float(M[2,0]), float(M[2,1]), float(M[2,2]),
            )

        # Create kinematic bodies for interactive picking (like quadruped example)
        # For interactive picking to work, bodies need proper mass properties
        shape_cfg = newton.ModelBuilder.ShapeConfig()
        shape_cfg.density = 1000.0  # Give them mass for picking to work
        shape_cfg.ke = 1e6  # High stiffness
        shape_cfg.kd = 1e3  # Damping
        shape_cfg.kf = 1e3  # Friction stiffness
        shape_cfg.mu = 0.5  # Friction coefficient

        # Bottom plate: horizontal pushing surface
        self.bottom_body_id = builder.add_body(
            xform=wp.transform(
                wp.vec3(*bottom_center),
                wp.quat_from_matrix(mat33_from_np_rowmajor(Rz))
            )
        )
        builder.add_shape_box(
            self.bottom_body_id,
            hx=bottom_size[0]*0.5,
            hy=bottom_size[1]*0.5,
            hz=bottom_size[2]*0.5,
            cfg=shape_cfg
        )

        # Top plate: vertical wall - creates clean L-shape
        self.top_body_id = builder.add_body(
            xform=wp.transform(
                wp.vec3(*top_center),
                wp.quat_from_matrix(mat33_from_np_rowmajor(Rz))
            )
        )
        builder.add_shape_box(
            self.top_body_id,
            hx=top_size[0]*0.5,
            hy=top_size[1]*0.5,
            hz=top_size[2]*0.5,
            cfg=shape_cfg
        )

        print(f"Interactive L-shaped plow built from 2 kinematic box bodies:")
        print(f"  Bottom plate: {bottom_size} at {bottom_center} (body {self.bottom_body_id})")
        print(f"  Top plate: {top_size} at {top_center} (body {self.top_body_id})")
        print(f"  Perfect 90-degree intersection at back edge")
        print(f"  Right-click and drag to move plow components!")

        # finalize model & MPM
        self.model = builder.finalize()
        self.model.particle_mu = options.friction_coeff

        # Setup MPM solver with same options as granular example
        options.grid_padding = 0 if options.dynamic_grid else 5
        options.yield_stresses = wp.vec3(
            options.yield_stress,
            -options.stretching_yield_stress,
            options.compression_yield_stress,
        )

        self.solver = SolverImplicitMPM(self.model, options)
        self.solver.setup_collider(self.model, colliders=[])

        # improved: use dual state approach like granular example for stability
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.solver.enrich_state(self.state_0)
        self.solver.enrich_state(self.state_1)

        # Setup renderer (proper pattern from working examples)
        self._setup_renderer()

        # Note: CUDA graphs not compatible with MPM solver operations
        self.use_cuda_graph = False
        self.graph = None

        # Note: Plow bodies are now interactive and can be moved with right-click
        print("Interactive plow ready - right-click and drag to move components!")

    def _setup_renderer(self):
        """Setup renderer for interactive simulation."""
        if self.viewer is not None:
            # Set up the viewer/renderer properly
            self.viewer.set_model(self.model)
            self.viewer.show_particles = True

            # Store reference to renderer for picking
            self.renderer = self.viewer

    def _update_plow_position(self, x_pos: float, z_pos: float, dt: float):
        # Note: Plow is now static - no position updates needed
        pass

    def simulate(self):
        """MPM simulation loop with interactive picking support."""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # Apply picking forces for interactive body movement (like quadruped example)
            # Use renderer instead of viewer for picking forces
            if self.renderer and hasattr(self.renderer, "apply_picking_force"):
                self.renderer.apply_picking_force(self.state_0)

            # Run MPM simulation with plow interaction
            self.solver.step(self.state_0, self.state_1, contacts=None, control=None, dt=self.sim_dt)
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

    @staticmethod
    def emit_particles(builder: newton.ModelBuilder, args):
        max_fraction = args.max_fraction
        voxel_size = args.voxel_size

        particles_per_cell = 4  # More particles per cell for finer sand
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

        # Smaller radius for finer sand particles
        radius = np.max(cell_size) * 0.75  # Reduced from 0.5 for finer appearance
        volume = np.prod(cell_volume) * packing_fraction

        rng = np.random.default_rng()
        # Increased jitter for more natural sand distribution
        points += 2.5 * radius * (rng.random(points.shape) - 0.5)
        vel = np.zeros_like(points)

        builder.particle_q = points
        builder.particle_qd = vel
        builder.particle_mass = np.full(points.shape[0], volume)
        builder.particle_radius = np.full(points.shape[0], radius)
        builder.particle_flags = np.zeros(points.shape[0], dtype=int)


if __name__ == "__main__":
    import argparse

    # Create parser that inherits common arguments and adds example-specific ones
    parser = newton.examples.create_parser()

    # Add MPM-specific arguments (fine sand configuration)
    parser.add_argument("--emit-lo", type=float, nargs=3, default=[-2.0, 0.002, -0.30])
    parser.add_argument("--emit-hi", type=float, nargs=3, default=[2.0, 0.20, 2.00])  # Higher sand pile
    parser.add_argument("--gravity", type=float, nargs=3, default=[0, -9.81, 0])
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--substeps", type=int, default=1)

    parser.add_argument("--max-fraction", type=float, default=1.0)

    parser.add_argument("--compliance", type=float, default=0.0)
    parser.add_argument("--poisson-ratio", "-nu", type=float, default=0.3)
    parser.add_argument("--friction-coeff", "-mu", type=float, default=0.75)  # Higher friction for sand
    parser.add_argument("--yield-stress", "-ys", type=float, default=0.0)
    parser.add_argument("--compression-yield-stress", "-cys", type=float, default=1.0e8)
    parser.add_argument("--stretching-yield-stress", "-sys", type=float, default=1.0e8)
    parser.add_argument("--unilateral", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dynamic-grid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gauss-seidel", "-gs", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--max-iterations", "-it", type=int, default=250)
    parser.add_argument("--tolerance", "-tol", type=float, default=1.0e-5)
    parser.add_argument("--voxel-size", "-dx", type=float, default=0.08)  # Smaller voxels for finer sand

    # Add plow-specific arguments
    parser.add_argument("--plow-pitch-deg", type=float, default=-18.0)
    parser.add_argument("--plow-speed", type=float, default=1)

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Create example and run
    example = Example(viewer, args)

    newton.examples.run(example)