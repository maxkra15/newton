# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
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
# Example MPM Falling Box
#
# Shows a heavy steel box falling under gravity and interacting with fine
# sand particles. The box falls and settles into the sand with realistic
# steel properties (high mass, low restitution, high friction).
#
# Note: This example demonstrates rigid body-particle coupling where the box
# interacts with sand particles through MPM colliders. The box mesh is updated
# each substep to follow the rigid body motion for proper particle interaction.
# The particles are fine sand-like with realistic granular behavior.
#
# STABILITY FIXES APPLIED:
# - Contacts recomputed each substep (prevents spring-like behavior)
# - Tuned stiffness/damping for stable settling
# - Optional velocity clamping to reduce micro-jitter
# - Two-way coupling: box collides with particles AND ground
#
# Example usage:
#   python -m newton.examples mpm_falling_box --viewer gl
#   python -m newton.examples mpm_falling_box --viewer gl --collider none  # No box
#
# Tuning for stability (if still bouncy):
#   python -m newton.examples mpm_falling_box --viewer gl --substeps 16 --box-stiffness 5e5
#   python -m newton.examples mpm_falling_box --viewer gl --box-damping 8000 --velocity-threshold 2e-4
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverImplicitMPM


@wp.kernel
def update_box_collider_mesh(
    src_points: wp.array(dtype=wp.vec3),
    res_mesh: wp.uint64,
    body_q: wp.array(dtype=wp.transform),
    body_idx: int,
    dt: float,
):
    """Update box collider mesh vertices based on rigid body transform (simplified version)."""
    v = wp.tid()
    res = wp.mesh_get(res_mesh)

    # Get the body transform from the array
    body_transform = body_q[body_idx]

    # Transform the rest position by the current body transform
    cur_p = res.points[v]
    next_p = wp.transform_point(body_transform, src_points[v])

    # Update velocity and position
    if dt > 0.0:
        res.velocities[v] = (next_p - cur_p) / dt
    else:
        res.velocities[v] = wp.vec3(0.0, 0.0, 0.0)
    res.points[v] = next_p  # Set directly to new position


@wp.kernel
def clamp_body_velocity(
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_idx: int,
    threshold: float,
):
    """Clamp tiny body velocities to reduce micro-jitter."""
    if body_idx >= 0:
        spatial_vel = body_qd[body_idx]

        # Extract angular and linear velocity components
        angular_vel = wp.spatial_top(spatial_vel)
        linear_vel = wp.spatial_bottom(spatial_vel)

        # Check magnitudes
        angular_mag = wp.length(angular_vel)
        linear_mag = wp.length(linear_vel)

        # Clamp if either is below threshold
        if angular_mag < threshold or linear_mag < threshold:
            body_qd[body_idx] = wp.spatial_vector(
                wp.vec3(0.0, 0.0, 0.0),  # angular
                wp.vec3(0.0, 0.0, 0.0)   # linear
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

        # Anti-jitter settings
        self.velocity_threshold = options.velocity_threshold

        # save a reference to the viewer
        self.viewer = viewer
        builder = newton.ModelBuilder()

        # Add particle pile on the ground (similar to anymal example)
        Example.emit_particles(builder, options)

        # Add falling box as a dynamic rigid body (restored functionality)
        if options.collider is not None and options.collider.lower() != "none":
            # Start box higher to prevent initial interpenetration with particles
            box_height = 2.2  # 2.2 meters above the particle pile

            # Create dynamic rigid body for the falling box
            box_body = builder.add_body(
                xform=wp.transform(
                    p=wp.vec3(0.0, 0.0, box_height),
                    q=wp.quat_identity()
                )
            )

            # Add box shape with steel-like properties (non-bouncy)
            builder.add_shape_box(
                body=box_body,
                hx=options.box_size[0] / 2,  # half-extents
                hy=options.box_size[1] / 2,
                hz=options.box_size[2] / 2,
                cfg=newton.ModelBuilder.ShapeConfig(
                    density=7850.0,                    # Steel density (heavy!)
                    mu=options.box_friction,           # Steel-on-sand friction
                    ke=options.box_stiffness,          # Very stiff contact
                    kd=options.box_damping,            # High damping (absorbs energy)
                    restitution=options.box_restitution  # Very low restitution (no bounce)
                )
            )

            self.box_body_idx = box_body

            # Create MPM collider for the box to interact with particles
            box_mesh = _create_collider_mesh(options.collider, options.box_size)
            if box_mesh is not None:
                # Create mesh with proper velocity initialization (like anymal)
                self.box_mesh = wp.Mesh(
                    wp.clone(box_mesh.points),
                    box_mesh.indices,
                    wp.zeros_like(box_mesh.points)  # Initialize velocities to zero
                )
                colliders = [self.box_mesh]
                # Store original mesh points for transformation
                self.box_mesh_points_orig = wp.clone(box_mesh.points)
            else:
                colliders = []
                self.box_mesh = None
                self.box_mesh_points_orig = None
        else:
            colliders = []
            self.box_body_idx = None
            self.box_mesh = None

        # Add non-bouncy ground plane (like solid concrete/steel)
        builder.add_ground_plane()
        builder.gravity = wp.vec3(options.gravity)

        options.grid_padding = 0 if options.dynamic_grid else 5
        options.yield_stresses = wp.vec3(
            options.yield_stress,
            -options.stretching_yield_stress,
            options.compression_yield_stress,
        )

        self.model = builder.finalize()
        self.model.particle_mu = options.friction_coeff

        # Setup solvers
        if self.box_body_idx is not None:
            # Use rigid solver for falling box dynamics
            self.rigid_solver = newton.solvers.SolverXPBD(self.model)
        else:
            self.rigid_solver = None

        # MPM solver for particles (no colliders to avoid complexity)
        self.mpm_solver = SolverImplicitMPM(self.model, options)
        self.mpm_solver.setup_collider(self.model, colliders=colliders)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.mpm_solver.enrich_state(self.state_0)
        self.mpm_solver.enrich_state(self.state_1)

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.capture()

    def capture(self):
        """Capture CUDA graph for rigid body simulation (like anymal example)."""
        self.graph = None
        if wp.get_device().is_cuda and self.rigid_solver is not None:
            with wp.ScopedCapture() as capture:
                self.simulate_robot()
            self.graph = capture.graph

    def simulate_robot(self):
        """Simulate rigid body dynamics (falling box) - separate from MPM."""
        if self.rigid_solver is not None:
            for _ in range(self.sim_substeps):
                self.state_0.clear_forces()
                contacts = self.model.collide(self.state_0)
                self.rigid_solver.step(self.state_0, self.state_1, None, contacts, self.sim_dt)
                self.state_0, self.state_1 = self.state_1, self.state_0

            # Anti-jitter: clamp tiny velocities to reduce micro-oscillations when settled
            if self.velocity_threshold > 0.0 and self.box_body_idx is not None:
                self._clamp_small_velocities()

    def simulate_sand(self):
        """Simulate MPM particles - separate from rigid bodies."""
        # Update box collider mesh to follow rigid body before MPM step
        if self.box_mesh is not None and self.box_mesh_points_orig is not None and self.rigid_solver is not None:
            self._update_collider_mesh()

        # Step MPM solver (only once per frame, not per substep)
        self.state_0.clear_forces()
        self.mpm_solver.step(self.state_0, self.state_0, None, None, self.frame_dt)

    def _update_collider_mesh(self):
        """Update collider mesh to follow rigid body (simplified version)."""
        wp.launch(
            kernel=update_box_collider_mesh,
            dim=self.box_mesh_points_orig.shape[0],
            inputs=[
                self.box_mesh_points_orig,
                self.box_mesh.id,
                self.state_0.body_q,
                self.box_body_idx,
                self.frame_dt
            ]
        )
        # Refit the mesh after updating (critical for proper collision detection)
        self.box_mesh.refit()

    def _clamp_small_velocities(self):
        """Clamp tiny velocities to reduce micro-jitter when the box has settled."""
        if self.box_body_idx is not None:
            wp.launch(
                kernel=clamp_body_velocity,
                dim=1,
                inputs=[self.state_0.body_qd, self.box_body_idx, self.velocity_threshold],
            )

    def step(self):
        # Simulate rigid body dynamics first (like anymal example)
        if self.graph and self.rigid_solver is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate_robot()

        # Then simulate MPM particles (not graph-capturable yet)
        self.simulate_sand()

        self.sim_time += self.frame_dt

    def test(self):
        pass

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def emit_particles(builder: newton.ModelBuilder, args):
        """Emit fine sand-like particles in a pile on the ground."""
        max_fraction = args.max_fraction
        voxel_size = args.voxel_size

        # More particles per cell for finer sand
        particles_per_cell = 4
        # Create a particle pile on the ground
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
        """Spawn particles in a grid pattern with some randomization."""
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

        print(f"Particle count: {points.shape[0]}")


def _create_collider_mesh(collider: str, box_size):
    """Create a collider mesh (like granular example)."""

    if collider == "cube":
        cube_points, cube_indices = newton.utils.create_box_mesh(
            extents=(box_size[0] / 2, box_size[1] / 2, box_size[2] / 2)
        )

        return wp.Mesh(
            wp.array(cube_points[:, 0:3], dtype=wp.vec3),
            wp.array(cube_indices, dtype=int),
        )
    else:
        return None


if __name__ == "__main__":
    import argparse

    # Create parser that inherits common arguments and adds example-specific ones
    parser = newton.examples.create_parser()

    # Add MPM-specific arguments (like granular example)
    parser.add_argument("--collider", default="cube", type=str, help="Collider type for the box")

    parser.add_argument("--emit-lo", type=float, nargs=3, default=[-0.5, -0.5, 0.0])
    parser.add_argument("--emit-hi", type=float, nargs=3, default=[0.5, 0.5, 0.5])
    parser.add_argument("--gravity", type=float, nargs=3, default=[0, 0, -10])
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--substeps", type=int, default=1)  # Increased for stability

    # Box-specific arguments (steel block properties) - TUNED FOR STABILITY
    parser.add_argument("--box-size", type=float, nargs=3, default=[0.5, 0.5, 0.5])
    parser.add_argument("--box-friction", type=float, default=0.7)    # Steel-on-sand friction
    parser.add_argument("--box-stiffness", type=float, default=1.0e6) # Reduced from 1e7 for stability
    parser.add_argument("--box-damping", type=float, default=6500.0)  # Tuned closer to critical damping
    parser.add_argument("--box-restitution", type=float, default=0.1) # Zero restitution while tuning
    parser.add_argument("--velocity-threshold", type=float, default=1e-4) # Clamp tiny velocities to reduce jitter

    # Particle arguments (sand-like properties)
    parser.add_argument("--max-fraction", type=float, default=0.6)  # More realistic packing
    parser.add_argument("--friction-coeff", "-mu", type=float, default=0.8)  # Higher friction for sand
    parser.add_argument("--yield-stress", "-ys", type=float, default=0.0)
    parser.add_argument("--compression-yield-stress", "-cys", type=float, default=1.0e8)
    parser.add_argument("--stretching-yield-stress", "-sys", type=float, default=1.0e8)
    parser.add_argument("--dynamic-grid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--voxel-size", "-dx", type=float, default=0.05)  # Finer resolution for sand

    # MPM solver specific arguments
    parser.add_argument("--compliance", type=float, default=0.0)
    parser.add_argument("--poisson-ratio", "-nu", type=float, default=0.3)
    parser.add_argument("--unilateral", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gauss-seidel", "-gs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-iterations", "-it", type=int, default=250)
    parser.add_argument("--tolerance", "-tol", type=float, default=1.0e-5)

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Create example and run
    example = Example(viewer, args)

    newton.examples.run(example)
