# SPDX-License-Identifier: Apache-2.0
# Dynamic Sand Plow Design Architecture
"""
Design document and architecture for dynamic mesh-to-particle conversion
in Newton MPM simulation.

This file outlines the design approach for implementing dynamic sand conversion
where sand starts as mesh geometry and converts to particles when disturbed
by a plow, then converts back to mesh when undisturbed.
"""

import numpy as np
import warp as wp
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class SandState(Enum):
    """States that sand regions can be in"""
    MESH = "mesh"           # Static mesh representation
    PARTICLES = "particles" # Active MPM particles
    TRANSITIONING = "transitioning"  # Converting between states

@dataclass
class SandRegion:
    """Represents a region of sand that can be in different states"""
    id: int
    bounds_min: np.ndarray  # 3D bounding box minimum
    bounds_max: np.ndarray  # 3D bounding box maximum
    state: SandState
    mesh_vertices: Optional[np.ndarray] = None
    mesh_indices: Optional[np.ndarray] = None
    particle_indices: Optional[List[int]] = None  # Indices into global particle array
    last_update_time: float = 0.0

@dataclass
class ConversionZone:
    """Spherical zone that triggers mesh-to-particle conversion"""
    center: np.ndarray      # 3D center position
    radius: float           # Sphere radius
    last_position: np.ndarray  # Previous position for movement tracking

class DynamicSandManager:
    """
    Manages dynamic conversion between mesh and particle representations.
    
    Key Design Decisions:
    1. Pre-allocate maximum number of particles needed
    2. Use particle flags to activate/deactivate particles
    3. Maintain spatial grid for efficient region queries
    4. Track sand regions with state information
    """
    
    def __init__(self, 
                 sand_bounds: Tuple[np.ndarray, np.ndarray],
                 region_size: float = 0.5,
                 max_particles: int = 100000,
                 conversion_radius: float = 0.5):
        """
        Initialize the dynamic sand management system.
        
        Args:
            sand_bounds: (min_bounds, max_bounds) for total sand area
            region_size: Size of each sand region for spatial partitioning
            max_particles: Maximum number of particles to pre-allocate
            conversion_radius: Radius of conversion sphere
        """
        self.sand_bounds = sand_bounds
        self.region_size = region_size
        self.max_particles = max_particles
        self.conversion_radius = conversion_radius
        
        # Initialize spatial grid of sand regions
        self.regions = self._create_spatial_grid()
        
        # Particle management
        self.particle_pool_size = max_particles
        self.active_particles = set()  # Set of active particle indices
        self.dormant_particles = set(range(max_particles))  # Available particles
        
        # Conversion zone tracking
        self.conversion_zone = ConversionZone(
            center=np.zeros(3),
            radius=conversion_radius,
            last_position=np.zeros(3)
        )
        
    def _create_spatial_grid(self) -> List[List[List[SandRegion]]]:
        """Create 3D grid of sand regions for spatial partitioning"""
        min_bounds, max_bounds = self.sand_bounds
        grid_size = np.ceil((max_bounds - min_bounds) / self.region_size).astype(int)
        
        regions = []
        region_id = 0
        
        for x in range(grid_size[0]):
            x_regions = []
            for y in range(grid_size[1]):
                y_regions = []
                for z in range(grid_size[2]):
                    # Calculate region bounds
                    region_min = min_bounds + np.array([x, y, z]) * self.region_size
                    region_max = region_min + self.region_size
                    
                    # Create region with initial mesh state
                    region = SandRegion(
                        id=region_id,
                        bounds_min=region_min,
                        bounds_max=region_max,
                        state=SandState.MESH
                    )
                    
                    # Generate initial mesh for this region
                    region.mesh_vertices, region.mesh_indices = self._generate_region_mesh(
                        region_min, region_max
                    )
                    
                    y_regions.append(region)
                    region_id += 1
                x_regions.append(y_regions)
            regions.append(x_regions)
            
        return regions
    
    def _generate_region_mesh(self, min_bounds: np.ndarray, max_bounds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate mesh geometry for a sand region"""
        # Simple heightfield mesh generation
        # In practice, this would create a more sophisticated sand surface
        
        resolution = 10  # Grid resolution for mesh
        x = np.linspace(min_bounds[0], max_bounds[0], resolution)
        z = np.linspace(min_bounds[2], max_bounds[2], resolution)
        
        vertices = []
        indices = []
        
        # Generate heightfield vertices
        for i, x_val in enumerate(x):
            for j, z_val in enumerate(z):
                # Simple height function - could be more complex terrain
                y_val = min_bounds[1] + 0.1 * np.sin(x_val) * np.cos(z_val)
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
    
    def update_conversion_zone(self, plow_position: np.ndarray):
        """Update the position of the conversion zone based on plow movement"""
        self.conversion_zone.last_position = self.conversion_zone.center.copy()
        self.conversion_zone.center = plow_position.copy()
    
    def get_affected_regions(self, sphere_center: np.ndarray, sphere_radius: float) -> List[SandRegion]:
        """Get all sand regions that intersect with the given sphere"""
        affected_regions = []
        
        # Convert sphere position to grid coordinates
        min_bounds, _ = self.sand_bounds
        grid_pos = (sphere_center - min_bounds) / self.region_size
        grid_radius = sphere_radius / self.region_size
        
        # Check regions in a bounding box around the sphere
        grid_min = np.floor(grid_pos - grid_radius).astype(int)
        grid_max = np.ceil(grid_pos + grid_radius).astype(int)
        
        for x in range(max(0, grid_min[0]), min(len(self.regions), grid_max[0] + 1)):
            for y in range(max(0, grid_min[1]), min(len(self.regions[0]), grid_max[1] + 1)):
                for z in range(max(0, grid_min[2]), min(len(self.regions[0][0]), grid_max[2] + 1)):
                    region = self.regions[x][y][z]
                    
                    # Check if sphere intersects region bounds
                    if self._sphere_intersects_box(sphere_center, sphere_radius, 
                                                 region.bounds_min, region.bounds_max):
                        affected_regions.append(region)
        
        return affected_regions
    
    def _sphere_intersects_box(self, sphere_center: np.ndarray, sphere_radius: float,
                              box_min: np.ndarray, box_max: np.ndarray) -> bool:
        """Check if sphere intersects with axis-aligned bounding box"""
        # Find closest point on box to sphere center
        closest_point = np.clip(sphere_center, box_min, box_max)
        
        # Check if distance to closest point is within sphere radius
        distance_sq = np.sum((sphere_center - closest_point) ** 2)
        return distance_sq <= sphere_radius ** 2
    
    def convert_mesh_to_particles(self, region: SandRegion) -> List[int]:
        """Convert a mesh region to particles and return particle indices"""
        if region.state != SandState.MESH:
            return []
        
        # Sample particles from mesh volume
        particle_positions = self._sample_particles_from_mesh(
            region.mesh_vertices, region.mesh_indices
        )
        
        # Allocate particles from dormant pool
        allocated_particles = []
        for pos in particle_positions:
            if len(self.dormant_particles) == 0:
                break  # No more particles available
                
            particle_idx = self.dormant_particles.pop()
            self.active_particles.add(particle_idx)
            allocated_particles.append(particle_idx)
        
        # Update region state
        region.state = SandState.PARTICLES
        region.particle_indices = allocated_particles
        
        return allocated_particles
    
    def convert_particles_to_mesh(self, region: SandRegion, particle_positions: np.ndarray):
        """Convert particles back to mesh representation"""
        if region.state != SandState.PARTICLES or region.particle_indices is None:
            return
        
        # Reconstruct mesh from particle positions
        region.mesh_vertices, region.mesh_indices = self._reconstruct_mesh_from_particles(
            particle_positions
        )
        
        # Return particles to dormant pool
        for particle_idx in region.particle_indices:
            self.active_particles.discard(particle_idx)
            self.dormant_particles.add(particle_idx)
        
        # Update region state
        region.state = SandState.MESH
        region.particle_indices = None
    
    def _sample_particles_from_mesh(self, vertices: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """Sample particle positions from mesh volume"""
        # Simple volume sampling - could be more sophisticated
        # For now, sample particles in a grid within the mesh bounding box
        
        min_bounds = np.min(vertices, axis=0)
        max_bounds = np.max(vertices, axis=0)
        
        # Sample particles in 3D grid
        particles_per_axis = 8  # Adjust density as needed
        x = np.linspace(min_bounds[0], max_bounds[0], particles_per_axis)
        y = np.linspace(min_bounds[1], max_bounds[1], particles_per_axis)
        z = np.linspace(min_bounds[2], max_bounds[2], particles_per_axis)
        
        positions = []
        for x_val in x:
            for y_val in y:
                for z_val in z:
                    positions.append([x_val, y_val, z_val])
        
        return np.array(positions, dtype=np.float32)
    
    def _reconstruct_mesh_from_particles(self, particle_positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Reconstruct mesh surface from particle positions"""
        # Simple mesh reconstruction - in practice would use more sophisticated algorithms
        # like marching cubes or surface reconstruction
        
        # For now, create a simple convex hull or bounding box mesh
        min_bounds = np.min(particle_positions, axis=0)
        max_bounds = np.max(particle_positions, axis=0)
        
        # Generate simple box mesh representing the deformed sand
        return self._generate_region_mesh(min_bounds, max_bounds)


# Warp kernels for particle state management
@wp.kernel
def activate_particles(
    particle_flags: wp.array(dtype=wp.int32),
    particle_positions: wp.array(dtype=wp.vec3),
    particle_velocities: wp.array(dtype=wp.vec3),
    new_positions: wp.array(dtype=wp.vec3),
    particle_indices: wp.array(dtype=wp.int32),
):
    """Activate dormant particles at specified positions"""
    tid = wp.tid()
    if tid < particle_indices.shape[0]:
        idx = particle_indices[tid]
        particle_flags[idx] = 1  # Active flag
        particle_positions[idx] = new_positions[tid]
        particle_velocities[idx] = wp.vec3(0.0, 0.0, 0.0)

@wp.kernel
def deactivate_particles(
    particle_flags: wp.array(dtype=wp.int32),
    particle_positions: wp.array(dtype=wp.vec3),
    particle_indices: wp.array(dtype=wp.int32),
):
    """Deactivate particles by moving them far away and setting inactive flag"""
    tid = wp.tid()
    if tid < particle_indices.shape[0]:
        idx = particle_indices[tid]
        particle_flags[idx] = 0  # Inactive flag
        particle_positions[idx] = wp.vec3(1000.0, 1000.0, 1000.0)  # Move far away
