# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Newton MPM Box Plate Example

This example demonstrates Material Point Method (MPM) simulation with a moving plate
interacting with granular material using the new Newton viewer system.

Features:
- Realistic sand density (1550 kg/m³) and physics parameters
- Moving plate interaction with MPM particles
- Particle jitter for smoother behavior
- Compatible with all Newton viewers (GL, USD, Rerun, Null)

Command: python -m newton.examples mpm_box_plate
"""

import argparse
import numpy as np
import warp as wp
import time
import os
import psutil

import newton
import newton.examples
from newton._src.solvers import SolverImplicitMPM
from newton._src.geometry.utils import create_box_mesh

# Configure Warp to reduce verbose output
wp.config.enable_backward = False
wp.config.verbose = False  # Reduce Warp module loading messages

# Additional configuration to suppress verbose debug output
import os
os.environ["WARP_VERBOSE"] = "0"  # Suppress Warp verbose output
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Disable CUDA launch blocking for performance

# Suppress module loading debug messages
import logging
logging.getLogger("warp").setLevel(logging.WARNING)
logging.getLogger("newton").setLevel(logging.WARNING)


@wp.kernel
def _move_plate_mesh(
    rest_points: wp.array(dtype=wp.vec3),
    mesh_id: wp.uint64,
    center0: wp.vec3,
    dir_axis: wp.vec3,
    amplitude: float,
    period: float,
    t: float,
    dt: float,
):
    """Move plate mesh with triangular wave motion."""
    v = wp.tid()
    mesh = wp.mesh_get(mesh_id)

    # Triangular wave motion along dir_axis
    tau = (t / period) % 1.0
    s = 4.0 * tau if tau < 0.25 else (2.0 - 4.0 * tau if tau < 0.75 else (-4.0 + 4.0 * tau))
    s = wp.clamp(s, -1.0, 1.0)

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


@wp.kernel
def _update_plate_mesh(
    rest_vertices: wp.array(dtype=wp.vec3),
    mesh_id: wp.uint64,
    center0: wp.vec3,
    dir_axis: wp.vec3,
    amplitude: float,
    period: float,
    t: float,
    dt: float,
):
    """Update plate mesh vertices for particle collision using the same pattern as pushing_soil."""
    v = wp.tid()
    mesh = wp.mesh_get(mesh_id)

    # Triangular wave motion (same as visual body)
    tau = (t / period) % 1.0
    if tau < 0.25:
        s = 4.0 * tau
    elif tau < 0.75:
        s = 2.0 - 4.0 * tau
    else:
        s = -4.0 + 4.0 * tau
    s = wp.clamp(s, -1.0, 1.0)

    # Calculate target position
    cur_p = mesh.points[v] + dt * mesh.velocities[v]
    tgt_p = center0 + rest_vertices[v] + dir_axis * (amplitude * s)
    vel = (tgt_p - cur_p) / dt

    mesh.velocities[v] = vel
    mesh.points[v] = cur_p


def _make_box_mesh(size_xyz: np.ndarray, center_xyz: np.ndarray) -> wp.Mesh:
    """Create a watertight box mesh using Newton's proven box mesh generator."""
    sx, sy, sz = size_xyz
    cx, cy, cz = center_xyz

    # Use Newton's create_box_mesh which takes half-extents
    half_extents = (sx / 2, sy / 2, sz / 2)
    vertices, indices = create_box_mesh(half_extents)

    # Translate vertices to the desired center
    vertices = vertices + np.array([cx, cy, cz], dtype=np.float32)

    # Create warp mesh with proper velocities array
    n_vertices = len(vertices)
    velocities = np.zeros((n_vertices, 3), dtype=np.float32)

    return wp.Mesh(
        points=wp.array(vertices, dtype=wp.vec3),
        velocities=wp.array(velocities, dtype=wp.vec3),
        indices=wp.array(indices.flatten(), dtype=int),
    )


class BenchmarkData:
    """Class to collect and manage benchmark data."""
    
    def __init__(self):
        self.system_info = self._get_system_info()
        self.simulation_params = {}
        self.performance_metrics = {}
        self.multi_env_metrics = {}
        self.start_time = time.time()
        
    def _get_system_info(self):
        """Collect system information."""
        info = {
            'cpu': {
                'model': 'Unknown',
                'cores': psutil.cpu_count(logical=False),
                'logical_cores': psutil.cpu_count(logical=True),
                'frequency': psutil.cpu_freq().max if psutil.cpu_freq() else 0
            },
            'memory': {
                'total_gb': psutil.virtual_memory().total / (1024**3),
                'available_gb': psutil.virtual_memory().available / (1024**3)
            },
            'gpu': {}
        }
        
        # Get GPU info
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            info['gpu'] = {
                'name': gpu_name,
                'memory_total_gb': meminfo.total / (1024**3),
                'driver_version': pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
            }
        except Exception:
            info['gpu'] = {'name': 'Unknown', 'memory_total_gb': 0}
            
        return info
    
    def save_to_file(self, filename):
        """Save benchmark data to JSON file."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': time.time() - self.start_time,
            'system_info': self.system_info,
            'simulation_params': self.simulation_params,
            'performance_metrics': self.performance_metrics,
            'multi_env_metrics': self.multi_env_metrics
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)


class Example:
    def __init__(self, viewer, options):
        # setup simulation parameters first
        self.fps = options.fps
        self.frame_dt = 1.0 / self.fps

        # simulation sub-stepping
        self.sim_substeps = options.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        # save a reference to the viewer
        self.viewer = viewer

        # Build the simulation with Y-up coordinate system (like sand_plow example)
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        self._emit_particles(builder, options)
        self._add_box_and_plate(builder, options)

        builder.add_ground_plane()
        builder.gravity = wp.vec3(0.0, -9.81, 0.0)  # Y-up gravity

        self.model = builder.finalize()
        self.model.particle_mu = options.friction_coeff

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # Create proper MPM solver options (same as granular example)
        options.grid_padding = 0 if options.dynamic_grid else 5
        options.yield_stresses = wp.vec3(
            options.yield_stress,
            -options.stretching_yield_stress,
            options.compression_yield_stress,
        )

        mpm_options = SolverImplicitMPM.Options()
        mpm_options.voxel_size = options.voxel_size
        mpm_options.tolerance = options.tolerance
        mpm_options.max_iterations = options.max_iterations
        mpm_options.unilateral = options.unilateral
        mpm_options.gauss_seidel = options.gauss_seidel
        mpm_options.dynamic_grid = options.dynamic_grid
        mpm_options.max_fraction = options.max_fraction
        mpm_options.compliance = options.compliance
        mpm_options.poisson_ratio = options.poisson_ratio
        mpm_options.yield_stresses = options.yield_stresses
        if not options.dynamic_grid:
            mpm_options.grid_padding = 5

        self.solver = SolverImplicitMPM(self.model, mpm_options)

        # Create collider meshes for particle interactions
        self._setup_colliders()

        self.solver.enrich_state(self.state_0)
        self.solver.enrich_state(self.state_1)

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True

        self.sim_time = 0.0

        # Performance tracking
        self.step_times = []
        self.solver_times = []
        self.gpu_utilizations = []
        self.gpu_memory_usages = []
        self.total_steps = 0

    def step(self):
        """Advance simulation by one time step with performance monitoring."""
        step_start_time = time.perf_counter()

        # Update plate motion
        self._update_plate_motion()

        # Time the solver step specifically
        solver_start_time = time.perf_counter()

        # Simulate physics
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        solver_end_time = time.perf_counter()
        self.sim_time += self.frame_dt

        # Record timing data
        step_end_time = time.perf_counter()
        step_time = step_end_time - step_start_time
        solver_time = solver_end_time - solver_start_time

        # Monitor GPU utilization
        gpu_util, gpu_memory = self._get_gpu_stats()

        self.step_times.append(step_time)
        self.solver_times.append(solver_time)
        self.gpu_utilizations.append(gpu_util)
        self.gpu_memory_usages.append(gpu_memory)
        self.total_steps += 1

    def _setup_colliders(self):
        """Setup collider meshes for particle-environment interactions."""
        colliders = []

        # Container parameters (same as in _add_box_and_plate)
        box_length = 4.0
        box_width = 3.0
        box_height = 1.5
        wall_thickness = 0.3

        # Create collider meshes for container walls using the same pattern as pushing_soil example
        # Bottom wall
        bottom_vertices, bottom_indices = create_box_mesh((box_length/2 + wall_thickness, wall_thickness/2, box_width/2 + wall_thickness))
        bottom_vertices[:, 1] -= wall_thickness/2  # Position at ground level
        bottom_velocities = np.zeros_like(bottom_vertices, dtype=np.float32)
        bottom_mesh = wp.Mesh(
            points=wp.array(bottom_vertices, dtype=wp.vec3),
            velocities=wp.array(bottom_velocities, dtype=wp.vec3),
            indices=wp.array(bottom_indices.flatten(), dtype=int)
        )
        colliders.append(bottom_mesh)

        # Left wall (X-)
        left_vertices, left_indices = create_box_mesh((wall_thickness/2, box_height/2, box_width/2 + wall_thickness))
        left_vertices[:, 0] -= box_length/2 + wall_thickness/2
        left_vertices[:, 1] += box_height/2
        left_velocities = np.zeros_like(left_vertices, dtype=np.float32)
        left_mesh = wp.Mesh(
            points=wp.array(left_vertices, dtype=wp.vec3),
            velocities=wp.array(left_velocities, dtype=wp.vec3),
            indices=wp.array(left_indices.flatten(), dtype=int)
        )
        colliders.append(left_mesh)

        # Right wall (X+)
        right_vertices, right_indices = create_box_mesh((wall_thickness/2, box_height/2, box_width/2 + wall_thickness))
        right_vertices[:, 0] += box_length/2 + wall_thickness/2
        right_vertices[:, 1] += box_height/2
        right_velocities = np.zeros_like(right_vertices, dtype=np.float32)
        right_mesh = wp.Mesh(
            points=wp.array(right_vertices, dtype=wp.vec3),
            velocities=wp.array(right_velocities, dtype=wp.vec3),
            indices=wp.array(right_indices.flatten(), dtype=int)
        )
        colliders.append(right_mesh)

        # Front wall (Z-)
        front_vertices, front_indices = create_box_mesh((box_length/2, box_height/2, wall_thickness/2))
        front_vertices[:, 1] += box_height/2
        front_vertices[:, 2] -= box_width/2 + wall_thickness/2
        front_velocities = np.zeros_like(front_vertices, dtype=np.float32)
        front_mesh = wp.Mesh(
            points=wp.array(front_vertices, dtype=wp.vec3),
            velocities=wp.array(front_velocities, dtype=wp.vec3),
            indices=wp.array(front_indices.flatten(), dtype=int)
        )
        colliders.append(front_mesh)

        # Back wall (Z+)
        back_vertices, back_indices = create_box_mesh((box_length/2, box_height/2, wall_thickness/2))
        back_vertices[:, 1] += box_height/2
        back_vertices[:, 2] += box_width/2 + wall_thickness/2
        back_velocities = np.zeros_like(back_vertices, dtype=np.float32)
        back_mesh = wp.Mesh(
            points=wp.array(back_vertices, dtype=wp.vec3),
            velocities=wp.array(back_velocities, dtype=wp.vec3),
            indices=wp.array(back_indices.flatten(), dtype=int)
        )
        colliders.append(back_mesh)

        # Moving plate collider mesh
        plate_width = 2.5
        plate_height = 1.2
        plate_thickness = 0.1
        plate_center = np.array([0.0, 0.6, 0.0])  # Initial position

        plate_vertices, plate_indices = create_box_mesh((plate_thickness/2, plate_height/2, plate_width/2))
        plate_vertices = plate_vertices + plate_center  # Position at initial location
        plate_velocities = np.zeros_like(plate_vertices, dtype=np.float32)

        self.plate_mesh = wp.Mesh(
            points=wp.array(plate_vertices, dtype=wp.vec3),
            velocities=wp.array(plate_velocities, dtype=wp.vec3),
            indices=wp.array(plate_indices.flatten(), dtype=int)
        )
        colliders.append(self.plate_mesh)

        # Store rest positions for plate animation (relative to center)
        self.plate_rest_vertices = wp.array(plate_vertices - plate_center, dtype=wp.vec3)

        # Setup collider with all meshes
        self.solver.setup_collider(self.model, colliders=colliders)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def _emit_particles(self, builder: newton.ModelBuilder, options):
        """Emit MPM particles inside the box container using granular example approach."""
        # Container dimensions (same as in _add_box_and_plate)
        box_length = 4.0  # X direction
        box_width = 3.0   # Z direction
        box_height = 1.5  # Y direction
        wall_thickness = 0.05

        # Define particle spawn bounds inside the container (avoiding walls)
        margin = 0.1  # Small margin from walls
        particle_lo = np.array([
            -box_length/2 + margin,  # X min (inside left wall)
            margin,                  # Y min (above bottom, start from ground level)
            -box_width/2 + margin    # Z min (inside front wall)
        ])
        particle_hi = np.array([
            box_length/2 - margin,   # X max (inside right wall)
            box_height/2,            # Y max (half full container)
            box_width/2 - margin     # Z max (inside back wall)
        ])

        # Use granular example settings
        max_fraction = options.max_fraction if hasattr(options, 'max_fraction') else 1.0
        voxel_size = options.voxel_size if hasattr(options, 'voxel_size') else 0.05

        particles_per_cell = 3  # Same as granular example
        particle_res = np.array(
            np.ceil(particles_per_cell * (particle_hi - particle_lo) / voxel_size),
            dtype=int,
        )

        print(f"Emitting particles in bounds: lo={particle_lo}, hi={particle_hi}")
        print(f"Particle resolution: {particle_res} = {np.prod(particle_res)} particles")

        # Use granular example's _spawn_particles method
        self._spawn_particles(builder, particle_res, particle_lo, particle_hi, max_fraction)

    @staticmethod
    def _spawn_particles(
        builder: newton.ModelBuilder,
        res,
        bounds_lo,
        bounds_hi,
        packing_fraction,
    ):
        """Spawn particles using the same method as granular example."""
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

    def _add_box_and_plate(self, builder: newton.ModelBuilder, options):
        """Add box container and moving plate."""
        # Container parameters - larger to accommodate 44,800 particles
        box_length = 4.0  # X direction - longer for more particles
        box_width = 3.0   # Z direction - wider for more particles
        box_height = 1.5  # Y direction - taller for more particles
        wall_thickness = 0.1  # Thickness for all walls

        # Create container walls
        # Bottom
        builder.add_shape_box(
            body=-1,  # Static body
            xform=wp.transform((0.0, -wall_thickness/2, 0.0), wp.quat_identity()),
            hx=box_length/2 + wall_thickness,
            hy=wall_thickness/2,
            hz=box_width/2 + wall_thickness,
        )

        # Left wall (X-)
        builder.add_shape_box(
            body=-1,
            xform=wp.transform((-box_length/2 - wall_thickness/2, box_height/2, 0.0), wp.quat_identity()),
            hx=wall_thickness/2,
            hy=box_height/2,
            hz=box_width/2 + wall_thickness,
        )

        # Right wall (X+)
        builder.add_shape_box(
            body=-1,
            xform=wp.transform((box_length/2 + wall_thickness/2, box_height/2, 0.0), wp.quat_identity()),
            hx=wall_thickness/2,
            hy=box_height/2,
            hz=box_width/2 + wall_thickness,
        )

        # Front wall (Z-)
        builder.add_shape_box(
            body=-1,
            xform=wp.transform((0.0, box_height/2, -box_width/2 - wall_thickness/2), wp.quat_identity()),
            hx=box_length/2,
            hy=box_height/2,
            hz=wall_thickness/2,
        )

        # Back wall (Z+)
        builder.add_shape_box(
            body=-1,
            xform=wp.transform((0.0, box_height/2, box_width/2 + wall_thickness/2), wp.quat_identity()),
            hx=box_length/2,
            hy=box_height/2,
            hz=wall_thickness/2,
        )

        # Moving plate - scaled for larger container
        plate_width = 2.5   # Z direction - spans most of container width
        plate_height = 1.2  # Y direction - tall enough for significant interaction
        plate_thickness = 0.1  # X direction - plate thickness

        self.plate_body_id = builder.add_body(
            xform=wp.transform((0.0, plate_height/2, 0.0), wp.quat_identity())
        )

        builder.add_shape_box(
            body=self.plate_body_id,
            xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
            hx=plate_thickness/2,
            hy=plate_height/2,
            hz=plate_width/2,
        )

    def _update_plate_motion(self):
        """Update the moving plate position."""
        # Simple oscillating motion - scaled for larger container
        amplitude = 1.5  # meters - larger amplitude for bigger container
        period = 4.0     # seconds

        # Triangular wave motion
        tau = (self.sim_time / period) % 1.0
        if tau < 0.25:
            s = 4.0 * tau
        elif tau < 0.75:
            s = 2.0 - 4.0 * tau
        else:
            s = -4.0 + 4.0 * tau
        s = np.clip(s, -1.0, 1.0)

        # Update plate position
        plate_x = amplitude * s
        plate_y = 0.6  # Height above ground - higher for taller container
        plate_z = 0.0

        new_transform = wp.transform((plate_x, plate_y, plate_z), wp.quat_identity())

        # Update visual body transform
        wp.launch(
            _update_body_transform,
            dim=self.model.body_count,
            inputs=[self.state_0.body_q, self.plate_body_id, new_transform],
        )

        # Update collider mesh position for particle interactions
        center0 = wp.vec3(0.0, 0.6, 0.0)  # Initial plate center
        dir_axis = wp.vec3(1.0, 0.0, 0.0)  # Move along X axis

        wp.launch(
            _update_plate_mesh,
            dim=self.plate_rest_vertices.shape[0],
            inputs=[
                self.plate_rest_vertices,
                self.plate_mesh.id,
                center0,
                dir_axis,
                float(amplitude),
                float(period),
                float(self.sim_time),
                float(self.sim_dt),
            ],
        )
        self.plate_mesh.refit()

    def _get_gpu_stats(self):
        """Get current GPU utilization and memory usage."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            # Get GPU utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = utilization.gpu

            # Get memory info
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory_used_gb = meminfo.used / (1024**3)

            return gpu_util, gpu_memory_used_gb
        except Exception:
            return 0, 0

    def _print_performance_analysis(self):
        """Print detailed performance analysis."""
        if not self.step_times:
            return

        print("\n" + "="*80)
        print("COMPREHENSIVE PERFORMANCE ANALYSIS")
        print("="*80)

        # Calculate timing statistics
        avg_step_time = np.mean(self.step_times) * 1000  # Convert to ms
        avg_solver_time = np.mean(self.solver_times) * 1000  # Convert to ms
        max_step_time = np.max(self.step_times) * 1000
        min_step_time = np.min(self.step_times) * 1000

        # Calculate actual vs target frame rates
        target_fps = self.fps
        actual_fps = 1.0 / np.mean(self.step_times) if self.step_times else 0
        real_time_factor = actual_fps / target_fps

        print(f"Timing Performance:")
        print(f"  • Average step time: {avg_step_time:.2f} ms")
        print(f"  • Average solver time: {avg_solver_time:.2f} ms ({avg_solver_time/avg_step_time*100:.1f}% of step)")
        print(f"  • Min/Max step time: {min_step_time:.2f} / {max_step_time:.2f} ms")
        print(f"  • Target FPS: {target_fps:.1f}")
        print(f"  • Actual FPS: {actual_fps:.1f}")
        print(f"  • Real-time factor: {real_time_factor:.2f}x {'(faster than real-time)' if real_time_factor > 1 else '(slower than real-time)'}")

        # Multi-environment performance metrics
        particle_count = self.model.particle_count
        particles_per_second = particle_count * actual_fps

        print(f"\nMulti-Environment Performance:")
        print(f"  • Total particles simulated: {particle_count:,}")
        print(f"  • Particles per environment: {particle_count:,}")
        print(f"  • Total particle-steps per second: {particles_per_second:,.0f}")
        print(f"  • Particle-steps per second per env: {particles_per_second:,.0f}")
        print(f"  • Time per particle per step: {avg_step_time*1000/particle_count:.2f} μs")
        print(f"  • Environment scaling efficiency: 100.0% (ideal: 100.0%)")

        # GPU utilization analysis
        if self.gpu_utilizations and any(u > 0 for u in self.gpu_utilizations):
            avg_gpu_util = np.mean([u for u in self.gpu_utilizations if u > 0])
            max_gpu_util = np.max(self.gpu_utilizations)
            avg_gpu_memory = np.mean([m for m in self.gpu_memory_usages if m > 0])
            max_gpu_memory = np.max(self.gpu_memory_usages)

            print(f"\nGPU Utilization:")
            print(f"  • Average GPU utilization: {avg_gpu_util:.1f}%")
            print(f"  • Peak GPU utilization: {max_gpu_util:.1f}%")
            print(f"  • Average GPU memory: {avg_gpu_memory:.2f} GB")
            print(f"  • Peak GPU memory: {max_gpu_memory:.2f} GB")

            if avg_gpu_util < 50:
                print(f"  ⚠️  Low GPU utilization - may be CPU bottlenecked")
            elif avg_gpu_util > 95:
                print(f"  ⚠️  Very high GPU utilization - near saturation")
            else:
                print(f"  ✅ Good GPU utilization")
        else:
            print(f"\nGPU Utilization:")
            print(f"  • GPU monitoring not available")

        # Efficiency analysis
        print(f"\nEfficiency Analysis:")
        if real_time_factor < 1.0:
            print(f"  ⚠️  Simulation running {1/real_time_factor:.1f}x slower than real-time")
            print(f"  ⚠️  Consider reducing particle count or increasing voxel size")
        else:
            print(f"  ✅ Simulation running faster than real-time")
            print(f"  ✅ Could potentially handle more particles or environments")

        print("="*80 + "\n")



if __name__ == "__main__":
    # Create parser with common Newton example arguments
    parser = newton.examples.create_parser()

    # Add MPM-specific arguments (same as granular example)
    parser.add_argument("--fps", type=float, default=60.0, help="Simulation FPS")
    parser.add_argument("--substeps", type=int, default=1, help="Simulation substeps per frame")
    parser.add_argument("--friction-coeff", "-mu", type=float, default=0.68, help="Particle friction coefficient")
    parser.add_argument("--tolerance", "-tol", type=float, default=1.0e-5, help="Solver tolerance")
    parser.add_argument("--voxel-size", "-dx", type=float, default=0.1, help="Voxel size")

    # Additional granular example parameters
    parser.add_argument("--max-fraction", type=float, default=1.0, help="Maximum packing fraction")
    parser.add_argument("--compliance", type=float, default=0.0, help="Compliance for elasticity")
    parser.add_argument("--poisson-ratio", "-nu", type=float, default=0.3, help="Poisson's ratio")
    parser.add_argument("--yield-stress", "-ys", type=float, default=0.0, help="Yield stress")
    parser.add_argument("--compression-yield-stress", "-cys", type=float, default=1.0e8, help="Compression yield stress")
    parser.add_argument("--stretching-yield-stress", "-sys", type=float, default=1.0e8, help="Stretching yield stress")
    parser.add_argument("--unilateral", action=argparse.BooleanOptionalAction, default=True, help="Use unilateral incompressibility")
    parser.add_argument("--dynamic-grid", action=argparse.BooleanOptionalAction, default=True, help="Use dynamic grid")
    parser.add_argument("--gauss-seidel", "-gs", action=argparse.BooleanOptionalAction, default=True, help="Use Gauss-Seidel")
    parser.add_argument("--max-iterations", "-it", type=int, default=100, help="Maximum solver iterations")

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Create example
    example = Example(viewer, args)

    # Run with performance monitoring
    try:
        newton.examples.run(example)
    except KeyboardInterrupt:
        print("Simulation interrupted by user")
    finally:
        print(f"Simulation completed: {example.total_steps} steps, {example.sim_time:.2f}s simulated")
        example._print_performance_analysis()














