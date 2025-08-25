# SPDX-License-Identifier: Apache-2.0
# Dynamic Sand Plow — Mesh-to-Particle Conversion with MPM
"""
Enhanced sand plow simulation with dynamic mesh-to-particle conversion.

This example demonstrates:
1. Sand starting as static mesh geometry
2. Dynamic conversion to MPM particles when plow approaches (1m diameter sphere)
3. Realistic sand simulation using Material Point Method
4. Conversion back to mesh when plow moves away
5. Mesh reconstruction reflecting deformed sand state

The conversion sphere follows the plow and triggers mesh-particle transitions
for realistic excavation simulation while maintaining performance.
"""

import sys
import math
import argparse
import numpy as np
import warp as wp
from typing import Tuple

wp.config.enable_backward = False

import newton
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

class DynamicSandPlowExample:
    def __init__(
        self,
        stage_path="example_dynamic_sand_plow.usd",
        voxel_size=0.05,
        particles_per_cell=2.0,
        tolerance=1e-5,
        headless=False,
        sand_friction=0.55,
        dynamic_grid=True,
        plow_pitch_deg=-18.0,
        plow_speed=0.5,
        conversion_radius=0.5,  # 1m diameter sphere
        max_particles=50000,
    ):
        self.device = wp.get_device()
        
        # Initialize dynamic sand management system
        sand_bounds = (
            np.array([-2.0, 0.0, -0.3], dtype=np.float32),  # min bounds
            np.array([2.0, 0.2, 2.0], dtype=np.float32)     # max bounds
        )
        
        self.sand_manager = DynamicSandManager(
            sand_bounds=sand_bounds,
            region_size=0.5,
            max_particles=max_particles,
            conversion_radius=conversion_radius
        )
        
        # Initialize conversion components
        self.intersector = MeshSphereIntersector(device=self.device)
        self.converter = MeshParticleConverter(
            particle_properties=ParticleProperties(density=2000.0, friction=sand_friction),
            device=self.device
        )
        self.reconstructor = ParticleMeshReconstructor(device=self.device)
        
        # Build Newton model
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        builder.add_ground_plane()
        builder.gravity = wp.vec3(0.0, -9.81, 0.0)
        
        # Timing
        self.sim_time = 0.0
        self.frame_dt = 1.0/60.0
        self.sim_substeps = 5
        self.sim_dt = self.frame_dt / self.sim_substeps
        
        # Pre-allocate maximum particles (all initially dormant)
        self._spawn_particle_pool(builder, max_particles, voxel_size)
        
        # Plow setup (similar to original example)
        self.conversion_radius = conversion_radius
        self.plow_speed = float(plow_speed)
        self.plow_finished = False
        
        # Motion bounds
        self.x0 = sand_bounds[0][0]  # start position
        self.x1 = sand_bounds[1][0]  # end position
        self.plow_y = 0.0
        self.plow_z = 1.0
        
        # Create plow geometry
        self._create_plow_geometry(builder, plow_pitch_deg)
        
        # MPM solver setup
        opt = SolverImplicitMPM.Options()
        opt.voxel_size = float(voxel_size)
        opt.max_fraction = 1.0
        opt.tolerance = float(tolerance)
        opt.unilateral = True
        opt.max_iterations = 250
        opt.gauss_seidel = True
        opt.dynamic_grid = bool(dynamic_grid)
        opt.yield_stresses = (0.0, -1.0e8, 1.0e8)
        
        if not dynamic_grid:
            opt.grid_padding = 5
        
        # Finalize model
        self.model = builder.finalize()
        self.model.particle_mu = float(sand_friction)
        
        self.mpm = SolverImplicitMPM(self.model, opt)
        self.mpm.setup_collider(self.model, [self.plow_mesh])
        
        # States
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.mpm.enrich_state(self.state_0)
        self.mpm.enrich_state(self.state_1)
        
        # Renderer setup following Newton examples pattern
        self.renderer = None
        if not headless:
            if stage_path and stage_path.endswith('.usd'):
                # Use ViewerUSD for USD export
                self.renderer = newton.viewer.ViewerUSD(stage_path, fps=60)
                self.renderer.set_model(self.model)
                print(f"Using ViewerUSD renderer, will save to: {stage_path}")
            else:
                # Use RendererOpenGL for interactive viewing
                self.renderer = newton.viewer.RendererOpenGL(self.model, stage_path or "Dynamic Sand Plow")
                print("Using RendererOpenGL for interactive viewing")

            # Enable particle visualization
            if hasattr(self.renderer, 'show_particles'):
                self.renderer.show_particles = True
        
        # Initialize plow position
        self._update_plow_position(self.x0, self.plow_z, self.frame_dt)

        # Initialize sand regions with proper mesh geometry
        self._initialize_sand_terrain()
        
        # Track conversion state
        self.last_conversion_check = 0.0
        self.conversion_check_interval = 0.1  # Check every 0.1 seconds

    def _initialize_sand_terrain(self):
        """Initialize sand regions with proper terrain mesh"""
        print("Initializing sand terrain...")

        # Create a heightfield terrain for each region
        for x_regions in self.sand_manager.regions:
            for y_regions in x_regions:
                for region in y_regions:
                    # Generate terrain mesh for this region
                    vertices, indices = self._create_region_terrain(
                        region.bounds_min, region.bounds_max
                    )

                    region.mesh_vertices = vertices
                    region.mesh_indices = indices
                    region.state = SandState.MESH

        print(f"Initialized {len(self.sand_manager.regions) * len(self.sand_manager.regions[0]) * len(self.sand_manager.regions[0][0])} sand regions")

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
            self.mpm.step(self.state_0, self.state_1, contacts=None, control=None, dt=self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        
        self.sim_time += self.frame_dt
    
    def render(self):
        """Render the simulation"""
        if self.renderer is None:
            return

        self.renderer.begin_frame(self.sim_time)

        # Handle different renderer types
        if hasattr(self.renderer, 'log_state'):
            # ViewerUSD interface
            self.renderer.log_state(self.state_0)
        else:
            # RendererOpenGL interface
            self.renderer.render(self.state_0)

        # Visualize sand regions (only for RendererOpenGL)
        if hasattr(self.renderer, 'log_mesh'):
            self._render_sand_regions()

        self.renderer.end_frame()

    def _render_sand_regions(self):
        """Render sand regions as meshes for visualization"""
        if self.renderer is None:
            return

        # Create a combined sand terrain mesh for better visualization
        all_vertices = []
        all_indices = []
        vertex_offset = 0

        for x_regions in self.sand_manager.regions:
            for y_regions in x_regions:
                for region in y_regions:
                    if region.mesh_vertices is not None and len(region.mesh_vertices) > 0:
                        # Only show mesh regions, hide particle regions
                        if region.state == SandState.MESH:
                            # Add vertices
                            all_vertices.extend(region.mesh_vertices.tolist())

                            # Add indices with offset
                            region_indices = region.mesh_indices + vertex_offset
                            all_indices.extend(region_indices.tolist())

                            vertex_offset += len(region.mesh_vertices)

        # Render combined sand terrain
        if len(all_vertices) > 0 and len(all_indices) > 0:
            try:
                vertices_np = np.array(all_vertices, dtype=np.float32)
                indices_np = np.array(all_indices, dtype=np.int32)

                # Generate normals for better lighting
                normals_np = self._compute_vertex_normals(vertices_np, indices_np)

                vertices_wp = wp.array(vertices_np, dtype=wp.vec3, device=self.device)
                indices_wp = wp.array(indices_np.flatten(), dtype=wp.int32, device=self.device)
                normals_wp = wp.array(normals_np, dtype=wp.vec3, device=self.device)

                self.renderer.log_mesh(
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
        if self.renderer is None:
            return

        try:
            # Create a simple sphere mesh for the conversion zone
            sphere_center = self.sand_manager.conversion_zone.center
            sphere_radius = self.conversion_radius

            # Generate sphere vertices (simple icosphere)
            vertices, indices = self._create_sphere_mesh(sphere_center, sphere_radius, resolution=8)

            if len(vertices) > 0:
                vertices_wp = wp.array(vertices, dtype=wp.vec3, device=self.device)
                indices_wp = wp.array(indices.flatten(), dtype=wp.int32, device=self.device)

                self.renderer.log_mesh(
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
        if self.renderer is None:
            return

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

            self.renderer.log_mesh(
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
        normals = np.zeros_like(vertices)

        # Compute face normals and accumulate to vertices
        triangles = indices.reshape(-1, 3)
        for tri in triangles:
            v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]

            # Compute face normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normal = np.cross(edge1, edge2)

            # Normalize
            norm = np.linalg.norm(face_normal)
            if norm > 0:
                face_normal /= norm

            # Accumulate to vertices
            normals[tri[0]] += face_normal
            normals[tri[1]] += face_normal
            normals[tri[2]] += face_normal

        # Normalize vertex normals
        for i in range(len(normals)):
            norm = np.linalg.norm(normals[i])
            if norm > 0:
                normals[i] /= norm
            else:
                normals[i] = np.array([0, 1, 0])  # Default up normal

        return normals.astype(np.float32)

    def save(self):
        """Save the USD file if using USD renderer"""
        if self.renderer is not None:
            from newton.viewer import ViewerUSD
            if isinstance(self.renderer, ViewerUSD):
                # ViewerUSD uses close() to save the file
                self.renderer.close()
                print("USD file saved successfully!")
            elif hasattr(self.renderer, 'save'):
                # RendererOpenGL or other renderers with save() method
                self.renderer.save()
                print("File saved successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--stage-path", type=lambda x: None if x == "None" else str(x),
                        default="example_dynamic_sand_plow.usd")
    parser.add_argument("--num-frames", type=int, default=1200)
    parser.add_argument("--voxel-size", "-dx", type=float, default=0.05)
    parser.add_argument("--particles-per-cell", "-ppc", type=float, default=3.0)
    parser.add_argument("--sand-friction", "-mu", type=float, default=0.55)
    parser.add_argument("--tolerance", "-tol", type=float, default=1e-5)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dynamic-grid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plow-pitch-deg", type=float, default=-18.0)
    parser.add_argument("--plow-speed", type=float, default=0.5)
    parser.add_argument("--conversion-radius", type=float, default=0.5)
    parser.add_argument("--max-particles", type=int, default=50000)
    args = parser.parse_known_args()[0]
    
    if wp.get_device(args.device).is_cpu:
        print("Error: This example requires a GPU.")
        sys.exit(1)
    
    with wp.ScopedDevice(args.device):
        ex = DynamicSandPlowExample(
            stage_path=args.stage_path,
            voxel_size=args.voxel_size,
            tolerance=args.tolerance,
            headless=args.headless,
            sand_friction=args.sand_friction,
            dynamic_grid=args.dynamic_grid,
            plow_pitch_deg=args.plow_pitch_deg,
            plow_speed=args.plow_speed,
            conversion_radius=args.conversion_radius,
            max_particles=args.max_particles,
        )
        
        for frame in range(args.num_frames):
            ex.step()
            ex.render()
            
            if frame % 60 == 0:  # Print status every second
                active_particles = len(ex.sand_manager.active_particles)
                print(f"Frame {frame}: {active_particles} active particles")
        
        # Save USD file if using USD renderer
        ex.save()

        print("Dynamic sand plow simulation completed!")
