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

"""
Dynamic Sand Plow Simulation with Mesh-to-Particle Conversion

This example demonstrates dynamic conversion between mesh and particle representations
for sand excavation simulation. Sand starts as mesh geometry and converts to MPM
particles when disturbed by the plow, then converts back to mesh when settled.

Features:
- Dynamic mesh-to-particle conversion based on proximity to plow
- Material Point Method (MPM) physics for realistic sand behavior
- Particle-to-mesh reconstruction for performance optimization
- Real-time visualization of conversion process
- Smooth particle generation with jitter for realistic appearance
"""

import sys
import math
import numpy as np
import warp as wp
from typing import Tuple

wp.config.enable_backward = False

import newton
import newton.examples
from newton.solvers import SolverImplicitMPM

# Import our custom modules
try:
    from dynamic_sand_plow_design import DynamicSandManager, SandState, ConversionZone
    from mesh_sphere_intersection_simple import MeshSphereIntersector
    from mesh_particle_conversion import MeshParticleConverter, ParticleProperties
    from particle_mesh_reconstruction import ParticleMeshReconstructor
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    print("Make sure the following files are in the same directory:")
    print("- dynamic_sand_plow_design.py")
    print("- mesh_sphere_intersection_simple.py")
    print("- mesh_particle_conversion.py")
    print("- particle_mesh_reconstruction.py")
    sys.exit(1)

# Plow movement kernels (similar to original sand_plow example)
@wp.kernel
def update_kinematic_mesh(
    rest_points: wp.array(dtype=wp.vec3),
    mesh_id: wp.uint64,
    R: wp.mat33,
    t: wp.vec3,
    dt: float,
):
    v = wp.tid()
    m = wp.mesh_get(mesh_id)
    p0 = m.points[v] + dt * m.velocities[v]
    rp = rest_points[v]
    np1 = wp.vec3(
        R[0,0]*rp[0] + R[0,1]*rp[1] + R[0,2]*rp[2] + t[0],
        R[1,0]*rp[0] + R[1,1]*rp[1] + R[1,2]*rp[2] + t[1],
        R[2,0]*rp[0] + R[2,1]*rp[1] + R[2,2]*rp[2] + t[2],
    )
    m.velocities[v] = (np1 - p0) / dt
    m.points[v] = p0

@wp.kernel
def update_body_transform(
    body_q: wp.array(dtype=wp.transform),
    body_id: int,
    new_transform: wp.transform,
):
    tid = wp.tid()
    if tid == body_id:
        body_q[tid] = new_transform

# Helper functions from original example
def mat3_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    a = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = a
    c, s, C = math.cos(angle_rad), math.sin(angle_rad), 1.0 - math.cos(angle_rad)
    return np.array([
        [x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, z*z*C + c  ],
    ], dtype=np.float32)

def make_box(center, size):
    cx, cy, cz = center
    sx, sy, sz = np.array(size, dtype=np.float32) * 0.5
    v = np.array([
        [cx - sx, cy - sy, cz - sz],  # 0
        [cx + sx, cy - sy, cz - sz],  # 1
        [cx - sx, cy + sy, cz - sz],  # 2
        [cx + sx, cy + sy, cz - sz],  # 3
        [cx - sx, cy - sy, cz + sz],  # 4
        [cx + sx, cy - sy, cz + sz],  # 5
        [cx - sx, cy + sy, cz + sz],  # 6
        [cx + sx, cy + sy, cz + sz],  # 7
    ], dtype=np.float32)
    t = np.array([
        [1,5,7],[1,7,3],   # +X
        [4,0,2],[4,2,6],   # -X
        [2,3,7],[2,7,6],   # +Y
        [4,5,1],[4,1,0],   # -Y
        [5,4,6],[5,6,7],   # +Z
        [0,1,3],[0,3,2],   # -Z
    ], dtype=np.int32)
    return v, t

def merge_meshes(parts):
    verts, tris, off = [], [], 0
    for v, t in parts:
        verts.append(v)
        tris.append(t + off)
        off += v.shape[0]
    V = np.vstack(verts).astype(np.float32)
    T = np.vstack(tris).astype(np.int32)
    return V, T

class Example:
    def __init__(self, viewer, options):
        # Setup simulation parameters first (following granular example pattern)
        self.fps = options.fps
        self.frame_dt = 1.0 / self.fps

        # Group related attributes by prefix
        self.sim_time = 0.0
        self.sim_substeps = options.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        # Save a reference to the viewer
        self.viewer = viewer
        self.device = wp.get_device()

        # Initialize dynamic sand management system
        sand_bounds = (
            np.array([-2.0, 0.0, -0.3], dtype=np.float32),  # min bounds
            np.array([2.0, 0.2, 2.0], dtype=np.float32)     # max bounds
        )

        self.sand_manager = DynamicSandManager(
            sand_bounds=sand_bounds,
            region_size=0.5,
            max_particles=options.max_particles,
            conversion_radius=options.conversion_radius
        )

        # Initialize conversion components
        self.intersector = MeshSphereIntersector(device=self.device)
        self.converter = MeshParticleConverter(
            particle_properties=ParticleProperties(density=2000.0, friction=options.friction_coeff),
            device=self.device
        )
        self.reconstructor = ParticleMeshReconstructor(device=self.device)

        # Build Newton model
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        builder.add_ground_plane()
        builder.gravity = wp.vec3(0.0, options.gravity, 0.0)
        
        # Pre-allocate maximum particles (all initially dormant) using improved method
        Example.emit_particles(builder, options)

        # Plow setup (similar to original example)
        self.conversion_radius = options.conversion_radius
        self.plow_speed = options.plow_speed
        self.plow_finished = False

        # Motion bounds
        self.x0 = sand_bounds[0][0]  # start position
        self.x1 = sand_bounds[1][0]  # end position
        self.plow_y = 0.0
        self.plow_z = 1.0

        # Create plow geometry
        self._create_plow_geometry(builder, options.plow_pitch_deg)

        # Create simple sand box geometry (instead of complex terrain)
        self._create_sand_box_geometry(builder, sand_bounds)
        
        # MPM solver setup using options pattern
        options.grid_padding = 0 if options.dynamic_grid else 5
        options.yield_stresses = wp.vec3(
            options.yield_stress,
            -options.stretching_yield_stress,
            options.compression_yield_stress,
        )

        # Finalize model
        self.model = builder.finalize()
        self.model.particle_mu = options.friction_coeff

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.solver = SolverImplicitMPM(self.model, options)
        self.solver.setup_collider(self.model, colliders=[self.plow_mesh])

        self.solver.enrich_state(self.state_0)
        self.solver.enrich_state(self.state_1)
        
        self.viewer.set_model(self.model)
        self.viewer.show_particles = True

        # Initialize plow position
        self._update_plow_position(self.x0, self.plow_z, self.frame_dt)

        # Initialize sand terrain as simple box mesh
        self._initialize_sand_terrain()

        # Track conversion state
        self.last_conversion_check = 0.0
        self.conversion_check_interval = 0.1  # Check every 0.1 seconds

    @staticmethod
    def emit_particles(builder, options):
        """
        Emit particles using improved generation method from granular example.
        Creates a pre-allocated pool of dormant particles with smooth distribution.
        """
        # Create particle pool far away from simulation (dormant state)
        dormant_position = wp.vec3(10000.0, 10000.0, 10000.0)

        # Use options values for particle generation
        voxel_size = options.voxel_size
        max_particles = options.max_particles
        particles_per_cell = options.particles_per_cell

        # Calculate particle spacing based on voxel size and particles per cell
        radius = voxel_size / (2.0 * particles_per_cell**(1.0/3.0))
        spacing = 2.0 * radius

        # Create regular grid of particles
        particles_per_dim = int(max_particles**(1.0/3.0)) + 1

        points = []
        for i in range(particles_per_dim):
            for j in range(particles_per_dim):
                for k in range(particles_per_dim):
                    if len(points) >= max_particles:
                        break

                    x = dormant_position[0] + i * spacing
                    y = dormant_position[1] + j * spacing
                    z = dormant_position[2] + k * spacing

                    points.append([x, y, z])

                if len(points) >= max_particles:
                    break
            if len(points) >= max_particles:
                break

        # Convert to numpy array and add jitter for smoother appearance
        # This is the key improvement from the granular example
        points = np.array(points[:max_particles], dtype=np.float32)

        # Add random jitter to particle positions (from granular example)
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        points += 2.0 * radius * (rng.random(points.shape) - 0.5)

        # Add particles to builder
        for point in points:
            builder.add_particle(
                pos=wp.vec3(point[0], point[1], point[2]),
                vel=wp.vec3(0.0, 0.0, 0.0),
                mass=1.0
            )

    def _create_sand_box_geometry(self, builder, sand_bounds):
        """Create simple rectangular sand box geometry instead of complex terrain"""
        min_bounds, max_bounds = sand_bounds

        # Create a simple box mesh for the sand region
        # This will be converted to particles when the plow approaches
        vertices = np.array([
            # Bottom face
            [min_bounds[0], min_bounds[1], min_bounds[2]],
            [max_bounds[0], min_bounds[1], min_bounds[2]],
            [max_bounds[0], min_bounds[1], max_bounds[2]],
            [min_bounds[0], min_bounds[1], max_bounds[2]],
            # Top face
            [min_bounds[0], max_bounds[1], min_bounds[2]],
            [max_bounds[0], max_bounds[1], min_bounds[2]],
            [max_bounds[0], max_bounds[1], max_bounds[2]],
            [min_bounds[0], max_bounds[1], max_bounds[2]],
        ], dtype=np.float32)

        # Box indices (12 triangles)
        indices = np.array([
            # Bottom face
            0, 1, 2, 0, 2, 3,
            # Top face
            4, 6, 5, 4, 7, 6,
            # Front face
            0, 4, 5, 0, 5, 1,
            # Back face
            2, 6, 7, 2, 7, 3,
            # Left face
            0, 3, 7, 0, 7, 4,
            # Right face
            1, 5, 6, 1, 6, 2,
        ], dtype=np.int32)

        # Store the sand box mesh for visualization
        self.sand_box_vertices = vertices
        self.sand_box_indices = indices

    def _initialize_sand_terrain(self):
        """Initialize sand regions as simple box meshes"""
        print("Initializing sand terrain as simple boxes...")

        # Create simple box mesh for each region instead of complex terrain
        for x_regions in self.sand_manager.regions:
            for y_regions in x_regions:
                for region in y_regions:
                    # Generate simple box mesh for this region
                    vertices, indices = self._create_simple_box_mesh(
                        region.bounds_min, region.bounds_max
                    )

                    region.mesh_vertices = vertices
                    region.mesh_indices = indices
                    region.state = SandState.MESH

        print(f"Initialized {len(self.sand_manager.regions) * len(self.sand_manager.regions[0]) * len(self.sand_manager.regions[0][0])} sand regions as simple boxes")

    def _create_simple_box_mesh(self, min_bounds: np.ndarray, max_bounds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create simple box mesh for a sand region"""
        # Create a simple box mesh for the sand region
        vertices = np.array([
            # Bottom face
            [min_bounds[0], min_bounds[1], min_bounds[2]],
            [max_bounds[0], min_bounds[1], min_bounds[2]],
            [max_bounds[0], min_bounds[1], max_bounds[2]],
            [min_bounds[0], min_bounds[1], max_bounds[2]],
            # Top face
            [min_bounds[0], max_bounds[1], min_bounds[2]],
            [max_bounds[0], max_bounds[1], min_bounds[2]],
            [max_bounds[0], max_bounds[1], max_bounds[2]],
            [min_bounds[0], max_bounds[1], max_bounds[2]],
        ], dtype=np.float32)

        # Box indices (12 triangles)
        indices = np.array([
            # Bottom face
            0, 1, 2, 0, 2, 3,
            # Top face
            4, 6, 5, 4, 7, 6,
            # Front face
            0, 4, 5, 0, 5, 1,
            # Back face
            2, 6, 7, 2, 7, 3,
            # Left face
            0, 3, 7, 0, 7, 4,
            # Right face
            1, 5, 6, 1, 6, 2,
        ], dtype=np.int32)

        return vertices, indices

    def _create_region_terrain(self, min_bounds: np.ndarray, max_bounds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create terrain mesh for a sand region"""
        resolution = 8  # Grid resolution for mesh

        x = np.linspace(min_bounds[0], max_bounds[0], resolution)
        z = np.linspace(min_bounds[2], max_bounds[2], resolution)

        vertices = []
        indices = []

        # Generate heightfield vertices with some terrain variation
        for i, x_val in enumerate(x):
            for j, z_val in enumerate(z):
                # Create varied terrain height
                base_height = (min_bounds[1] + max_bounds[1]) * 0.5
                variation = 0.02 * np.sin(x_val * 5) * np.cos(z_val * 5)
                y_val = base_height + variation

                vertices.append([x_val, y_val, z_val])

        # Generate triangle indices
        for i in range(resolution - 1):
            for j in range(resolution - 1):
                # Two triangles per quad
                v0 = i * resolution + j
                v1 = v0 + 1
                v2 = v0 + resolution
                v3 = v2 + 1

                indices.extend([v0, v1, v2, v1, v3, v2])

        return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.int32)
        
    def _spawn_particle_pool(self, builder: newton.ModelBuilder, max_particles: int, voxel_size: float):
        """Pre-allocate particle pool - all initially dormant (moved far away)"""
        # Place all particles far away initially
        dormant_position = np.array([10000.0, 10000.0, 10000.0])
        
        for i in range(max_particles):
            builder.add_particle(
                pos=dormant_position,
                vel=(0.0, 0.0, 0.0),
                mass=0.001,  # Small mass for dormant particles
                radius=voxel_size * 0.5,
                flags=0  # Inactive flag
            )
        
        print(f"Pre-allocated {max_particles} particles in dormant pool")
    
    def _create_plow_geometry(self, builder: newton.ModelBuilder, plow_pitch_deg: float):
        """Create plow geometry (similar to original example)"""
        # Plow dimensions
        plow_width = 1.2
        bottom_len_x = 0.50
        bottom_thick = 0.08
        top_height_y = 0.60
        top_thick_x = 0.08
        
        # Bottom and top plate centers
        bottom_center = np.array([0.0, 0.14, 0.0], dtype=np.float32)
        bottom_size = np.array([bottom_len_x, bottom_thick, plow_width], dtype=np.float32)
        
        top_center = np.array([-0.15, bottom_center[1] + 0.5*top_height_y + 0.07, 0.0], dtype=np.float32)
        top_size = np.array([top_thick_x, top_height_y, plow_width], dtype=np.float32)
        
        # Create mesh geometry
        v_bottom, t_bottom = make_box([0,0,0], bottom_size)
        v_top, t_top = make_box([0,0,0], top_size)
        
        # Apply pitch rotation
        pitch = math.radians(plow_pitch_deg)
        self.plow_rotation = mat3_from_axis_angle(np.array([0,0,1], dtype=np.float32), pitch)
        
        v_bottom = v_bottom + bottom_center
        v_top = v_top + top_center
        
        V, T = merge_meshes([(v_bottom, t_bottom), (v_top, t_top)])
        indices_flat = T.reshape(-1).astype(np.int32)
        vels = np.zeros_like(V, dtype=np.float32)
        
        # Create Warp mesh for collision
        self.plow_mesh = wp.Mesh(
            points=wp.array(V, dtype=wp.vec3),
            indices=wp.array(indices_flat, dtype=int),
            velocities=wp.array(vels, dtype=wp.vec3),
        )
        self.plow_rest = wp.array(V, dtype=wp.vec3)
        
        # Create visual body
        def mat33_from_np_rowmajor(M: np.ndarray) -> wp.mat33:
            return wp.mat33(
                float(M[0,0]), float(M[0,1]), float(M[0,2]),
                float(M[1,0]), float(M[1,1]), float(M[1,2]),
                float(M[2,0]), float(M[2,1]), float(M[2,2]),
            )
        
        self.plow_body_id = builder.add_body(
            xform=wp.transform(wp.vec3(self.x0, self.plow_y, self.plow_z), wp.quat_identity())
        )
        
        # Add collision shapes
        builder.add_shape_box(
            self.plow_body_id,
            hx=bottom_size[0]*0.5, hy=bottom_size[1]*0.5, hz=bottom_size[2]*0.5,
            xform=wp.transform(
                wp.vec3(*bottom_center),
                wp.quat_from_matrix(mat33_from_np_rowmajor(self.plow_rotation))
            ),
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
        )
        
        builder.add_shape_box(
            self.plow_body_id,
            hx=top_size[0]*0.5, hy=top_size[1]*0.5, hz=top_size[2]*0.5,
            xform=wp.transform(
                wp.vec3(*top_center),
                wp.quat_from_matrix(mat33_from_np_rowmajor(self.plow_rotation))
            ),
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
        )
    
    def _update_plow_position(self, x_pos: float, z_pos: float, dt: float):
        """Update plow position and collision mesh"""
        R = self.plow_rotation
        t = np.array([x_pos, self.plow_y, z_pos], dtype=np.float32)
        
        # Update collision mesh
        wp.launch(
            update_kinematic_mesh,
            dim=self.plow_rest.shape[0],
            inputs=[self.plow_rest, self.plow_mesh.id, wp.mat33(*R), wp.vec3(*t), float(dt)],
        )
        self.plow_mesh.refit()
        
        # Update visual body
        new_tf = wp.transform(wp.vec3(x_pos, self.plow_y, z_pos), wp.quat_identity())
        if self.plow_body_id < self.model.body_count:
            wp.launch(
                update_body_transform,
                dim=self.model.body_count,
                inputs=[self.state_0.body_q, self.plow_body_id, new_tf],
            )
    
    def _update_conversion_zones(self):
        """Update mesh-particle conversion based on plow position"""
        # Get current plow position
        current_plow_pos = np.array([
            self.x0 + self.plow_speed * self.sim_time,
            self.plow_y,
            self.plow_z
        ])
        
        # Update conversion zone
        self.sand_manager.update_conversion_zone(current_plow_pos)
        
        # Get affected regions
        affected_regions = self.sand_manager.get_affected_regions(
            current_plow_pos, self.conversion_radius
        )
        
        # Process each affected region
        for region in affected_regions:
            if region.state == SandState.MESH:
                # Convert mesh to particles
                self._convert_region_to_particles(region)
            elif region.state == SandState.PARTICLES:
                # Check if plow has moved away
                distance = np.linalg.norm(current_plow_pos - 
                    (region.bounds_min + region.bounds_max) * 0.5)
                
                if distance > self.conversion_radius * 1.5:  # Hysteresis
                    self._convert_region_to_mesh(region)
    
    def _convert_region_to_particles(self, region):
        """Convert a mesh region to particles"""
        if region.mesh_vertices is None:
            return
        
        # Get available particle indices
        available_indices = list(self.sand_manager.dormant_particles)[:1000]  # Limit for performance
        
        if not available_indices:
            return  # No particles available
        
        # Convert mesh to particles
        used_indices, particle_positions = self.converter.convert_mesh_to_particles(
            sphere_center=self.sand_manager.conversion_zone.center,
            sphere_radius=self.conversion_radius,
            mesh_vertices=region.mesh_vertices,
            mesh_indices=region.mesh_indices,
            particle_spacing=0.05,
            available_particle_indices=available_indices
        )
        
        if len(used_indices) > 0:
            # Activate particles in Newton state
            self.converter.activate_particles(used_indices, particle_positions, self.model, self.state_0)
            
            # Update sand manager state
            region.state = SandState.PARTICLES
            region.particle_indices = used_indices
            
            # Update particle pools
            for idx in used_indices:
                self.sand_manager.dormant_particles.discard(idx)
                self.sand_manager.active_particles.add(idx)
            
            print(f"Converted region {region.id} to {len(used_indices)} particles")
    
    def _convert_region_to_mesh(self, region):
        """Convert particles back to mesh"""
        if region.particle_indices is None:
            return
        
        # Extract particle positions
        particle_positions = self.reconstructor.extract_particle_positions(
            region.particle_indices, self.state_0
        )
        
        # Reconstruct mesh
        if len(particle_positions) > 0:
            vertices, indices = self.reconstructor.reconstruct_mesh_from_particles(
                particle_positions,
                (region.bounds_min, region.bounds_max),
                method="heightfield",
                resolution=16
            )
            
            # Smooth the mesh
            vertices = self.reconstructor.smooth_mesh(vertices, indices, iterations=2)
            
            region.mesh_vertices = vertices
            region.mesh_indices = indices
        
        # Deactivate particles
        self.converter.deactivate_particles(region.particle_indices, self.model, self.state_0)
        
        # Update particle pools
        for idx in region.particle_indices:
            self.sand_manager.active_particles.discard(idx)
            self.sand_manager.dormant_particles.add(idx)
        
        # Update region state
        region.state = SandState.MESH
        region.particle_indices = None
        
        print(f"Converted region {region.id} back to mesh")
    
    def step(self):
        """Main simulation step"""
        # Update plow position
        if not self.plow_finished:
            x_current = self.x0 + self.plow_speed * self.sim_time
            
            if x_current >= self.x1:
                x_current = self.x1
                self.plow_finished = True
            
            self._update_plow_position(x_current, self.plow_z, self.frame_dt)
        
        # Update conversion zones periodically
        if self.sim_time - self.last_conversion_check >= self.conversion_check_interval:
            self._update_conversion_zones()
            self.last_conversion_check = self.sim_time
        
        # Run MPM simulation
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, contacts=None, control=None, dt=self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        
        self.sim_time += self.frame_dt
    
    def render(self):
        """Render the simulation"""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)

        # Visualize sand regions
        self._render_sand_regions()

        self.viewer.end_frame()

    def _render_sand_regions(self):
        """Render sand regions as meshes for visualization"""

        # Create a combined sand terrain mesh for better visualization
        all_vertices = []
        all_indices = []
        vertex_offset = 0

        for x_regions in self.sand_manager.regions:
            for y_regions in x_regions:
                for region in y_regions:
                    # Only include regions that are in MESH state and have valid mesh data
                    if (region.state == SandState.MESH and
                        region.mesh_vertices is not None and
                        region.mesh_indices is not None and
                        len(region.mesh_vertices) > 0 and
                        len(region.mesh_indices) > 0):

                        # Validate mesh data before adding
                        vertices = region.mesh_vertices
                        indices = region.mesh_indices

                        # Check if indices are valid for this mesh
                        if np.max(indices) < len(vertices):
                            # Add vertices
                            all_vertices.extend(vertices.tolist())

                            # Add indices with offset
                            region_indices = indices + vertex_offset
                            all_indices.extend(region_indices.tolist())

                            vertex_offset += len(vertices)

        # Render combined sand terrain
        if len(all_vertices) > 0 and len(all_indices) > 0:
            try:
                vertices_np = np.array(all_vertices, dtype=np.float32)
                indices_np = np.array(all_indices, dtype=np.int32)

                # Validate mesh consistency
                max_vertex_index = len(vertices_np) - 1
                if len(indices_np) > 0 and np.max(indices_np) > max_vertex_index:
                    print(f"Warning: Mesh indices out of range. Max index: {np.max(indices_np)}, Max vertex: {max_vertex_index}")
                    return

                # Ensure indices are valid triangles
                if len(indices_np) % 3 != 0:
                    print(f"Warning: Invalid triangle count. Indices: {len(indices_np)}")
                    return

                # Generate normals for better lighting
                normals_np = self._compute_vertex_normals(vertices_np, indices_np)

                vertices_wp = wp.array(vertices_np, dtype=wp.vec3, device=self.device)
                indices_wp = wp.array(indices_np.flatten(), dtype=wp.int32, device=self.device)
                normals_wp = wp.array(normals_np, dtype=wp.vec3, device=self.device)

                self.viewer.log_mesh(
                    "/sand/terrain",
                    vertices_wp,
                    indices_wp,
                    normals=normals_wp,
                    hidden=False,
                    backface_culling=False
                )
            except Exception as e:
                print(f"Warning: Could not render sand terrain: {e}")

        # Log conversion zone visualization
        self._render_conversion_zone()

        # Render ground plane for reference
        self._render_ground_plane()

    def _render_conversion_zone(self):
        """Render the conversion zone as a wireframe sphere"""

        try:
            # Create a simple sphere mesh for the conversion zone
            sphere_center = self.sand_manager.conversion_zone.center
            sphere_radius = self.conversion_radius

            # Generate sphere vertices (simple icosphere)
            vertices, indices = self._create_sphere_mesh(sphere_center, sphere_radius, resolution=8)

            if len(vertices) > 0:
                vertices_wp = wp.array(vertices, dtype=wp.vec3, device=self.device)
                indices_wp = wp.array(indices.flatten(), dtype=wp.int32, device=self.device)

                self.viewer.log_mesh(
                    "/conversion_zone/sphere",
                    vertices_wp,
                    indices_wp,
                    hidden=False,
                    backface_culling=False
                )
        except Exception as e:
            # Skip if visualization fails
            pass

    def _create_sphere_mesh(self, center: np.ndarray, radius: float, resolution: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """Create a simple sphere mesh for visualization"""
        vertices = []
        indices = []

        # Generate vertices using spherical coordinates
        for i in range(resolution + 1):
            theta = np.pi * i / resolution  # 0 to pi
            for j in range(2 * resolution + 1):
                phi = 2 * np.pi * j / (2 * resolution)  # 0 to 2*pi

                x = center[0] + radius * np.sin(theta) * np.cos(phi)
                y = center[1] + radius * np.cos(theta)
                z = center[2] + radius * np.sin(theta) * np.sin(phi)

                vertices.append([x, y, z])

        # Generate triangle indices
        for i in range(resolution):
            for j in range(2 * resolution):
                # Current ring
                v0 = i * (2 * resolution + 1) + j
                v1 = i * (2 * resolution + 1) + (j + 1) % (2 * resolution + 1)

                # Next ring
                v2 = (i + 1) * (2 * resolution + 1) + j
                v3 = (i + 1) * (2 * resolution + 1) + (j + 1) % (2 * resolution + 1)

                # Two triangles per quad
                if i > 0:  # Skip top cap
                    indices.extend([v0, v1, v2])
                if i < resolution - 1:  # Skip bottom cap
                    indices.extend([v1, v3, v2])

        return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.int32)

    def _render_ground_plane(self):
        """Render a ground plane for visual reference"""

        try:
            # Create a simple ground plane
            size = 5.0
            vertices = np.array([
                [-size, -0.1, -size],
                [size, -0.1, -size],
                [-size, -0.1, size],
                [size, -0.1, size],
            ], dtype=np.float32)

            indices = np.array([0, 1, 2, 1, 3, 2], dtype=np.int32)

            vertices_wp = wp.array(vertices, dtype=wp.vec3, device=self.device)
            indices_wp = wp.array(indices, dtype=wp.int32, device=self.device)

            self.viewer.log_mesh(
                "/ground/plane",
                vertices_wp,
                indices_wp,
                hidden=False,
                backface_culling=False
            )
        except Exception as e:
            # Skip if visualization fails
            pass

    def _compute_vertex_normals(self, vertices: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """Compute vertex normals for mesh lighting"""
        if len(vertices) == 0:
            return np.array([], dtype=np.float32).reshape(0, 3)

        normals = np.zeros_like(vertices)

        # Compute face normals and accumulate to vertices
        if len(indices) >= 3:
            triangles = indices.reshape(-1, 3)
            for tri in triangles:
                # Validate triangle indices
                if (tri[0] < len(vertices) and tri[1] < len(vertices) and tri[2] < len(vertices) and
                    tri[0] >= 0 and tri[1] >= 0 and tri[2] >= 0):

                    v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]

                    # Compute face normal
                    edge1 = v1 - v0
                    edge2 = v2 - v0
                    face_normal = np.cross(edge1, edge2)

                    # Normalize
                    norm = np.linalg.norm(face_normal)
                    if norm > 1e-8:  # Avoid division by very small numbers
                        face_normal /= norm

                        # Accumulate to vertices
                        normals[tri[0]] += face_normal
                        normals[tri[1]] += face_normal
                        normals[tri[2]] += face_normal

        # Normalize vertex normals
        for i in range(len(normals)):
            norm = np.linalg.norm(normals[i])
            if norm > 1e-8:
                normals[i] /= norm
            else:
                normals[i] = np.array([0, 1, 0])  # Default up normal

        return normals.astype(np.float32)


if __name__ == "__main__":
    import argparse

    # Create argument parser following Newton examples pattern
    parser = newton.examples.create_parser()

    # Add MPM solver arguments (from granular example)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--max-fraction", type=float, default=1.0)
    parser.add_argument("--compliance", type=float, default=0.0)
    parser.add_argument("--poisson-ratio", "-nu", type=float, default=0.3)
    parser.add_argument("--friction-coeff", "-mu", type=float, default=0.55)
    parser.add_argument("--yield-stress", "-ys", type=float, default=0.0)
    parser.add_argument("--compression-yield-stress", "-cys", type=float, default=1.0e8)
    parser.add_argument("--stretching-yield-stress", "-sys", type=float, default=1.0e8)
    parser.add_argument("--unilateral", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dynamic-grid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gauss-seidel", "-gs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-iterations", "-it", type=int, default=250)
    parser.add_argument("--tolerance", "-tol", type=float, default=1.0e-5)
    parser.add_argument("--voxel-size", "-dx", type=float, default=0.05)
    parser.add_argument("--gravity", type=float, default=-9.81)
    parser.add_argument("--particles-per-cell", type=float, default=3.0)

    # Add dynamic sand plow specific arguments
    parser.add_argument("--plow-speed", type=float, default=0.5, help="Speed of the plow movement")
    parser.add_argument("--conversion-radius", type=float, default=0.5, help="Radius of conversion zone")
    parser.add_argument("--max-particles", type=int, default=50000, help="Maximum number of particles")
    parser.add_argument("--plow-pitch-deg", type=float, default=-18.0, help="Plow pitch angle in degrees")

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Create example and run
    example = Example(viewer, args)

    newton.examples.run(example)
