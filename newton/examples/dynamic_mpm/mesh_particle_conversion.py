# SPDX-License-Identifier: Apache-2.0
# Mesh-to-Particle Conversion for Dynamic Sand Plow
"""
Functions to convert mesh regions into MPM particles and vice versa,
including volume sampling, particle property calculation, and mesh modification.
"""

import numpy as np
import warp as wp
from typing import List, Tuple, Optional
import newton
from dataclasses import dataclass

@dataclass
class ParticleProperties:
    """Properties for newly created particles"""
    density: float = 2000.0  # kg/m³ (sand density)
    friction: float = 0.6
    cohesion: float = 0.0
    young_modulus: float = 1e6
    poisson_ratio: float = 0.3

@wp.kernel
def sample_particles_in_sphere_kernel(
    sphere_center: wp.vec3,
    sphere_radius: float,
    mesh_vertices: wp.array(dtype=wp.vec3),
    mesh_indices: wp.array(dtype=wp.int32),
    sample_positions: wp.array(dtype=wp.vec3),
    sample_results: wp.array(dtype=wp.int32),  # 1 if valid particle position, 0 otherwise
    particle_spacing: float,
):
    """
    Kernel to sample particle positions within a sphere that intersects with mesh.
    
    Generates a regular grid of potential particle positions within the sphere
    and tests which ones are valid (inside or near the mesh surface).
    """
    tid = wp.tid()
    
    if tid < sample_positions.shape[0]:
        pos = sample_positions[tid]
        
        # Check if position is within sphere
        distance_to_center = wp.length(pos - sphere_center)
        if distance_to_center > sphere_radius:
            sample_results[tid] = 0
            return
        
        # Check if position is near mesh surface
        # Simple approach: check distance to nearest mesh vertex
        min_distance_sq = float(1e10)  # Declare as dynamic variable
        for i in range(mesh_vertices.shape[0]):
            vertex = mesh_vertices[i]
            dist_sq = wp.length_sq(pos - vertex)
            min_distance_sq = wp.min(min_distance_sq, dist_sq)
        
        # Consider position valid if it's close enough to mesh
        threshold_distance = particle_spacing * 1.5
        if wp.sqrt(min_distance_sq) <= threshold_distance:
            sample_results[tid] = 1
        else:
            sample_results[tid] = 0

@wp.kernel
def update_particle_properties_kernel(
    particle_indices: wp.array(dtype=wp.int32),
    particle_positions: wp.array(dtype=wp.vec3),
    particle_velocities: wp.array(dtype=wp.vec3),
    particle_masses: wp.array(dtype=wp.float32),
    particle_radii: wp.array(dtype=wp.float32),
    particle_flags: wp.array(dtype=wp.int32),
    new_positions: wp.array(dtype=wp.vec3),
    particle_mass: float,
    particle_radius: float,
):
    """
    Kernel to update properties of newly activated particles.
    """
    tid = wp.tid()
    
    if tid < particle_indices.shape[0]:
        idx = particle_indices[tid]
        
        # Set particle properties
        particle_positions[idx] = new_positions[tid]
        particle_velocities[idx] = wp.vec3(0.0, 0.0, 0.0)
        particle_masses[idx] = particle_mass
        particle_radii[idx] = particle_radius
        particle_flags[idx] = 1  # Active flag

@wp.kernel
def activate_particles_simple_kernel(
    particle_indices: wp.array(dtype=wp.int32),
    new_positions: wp.array(dtype=wp.vec3),
    particle_positions: wp.array(dtype=wp.vec3),
    particle_velocities: wp.array(dtype=wp.vec3),
):
    """
    Simple kernel to activate particles at specified positions.
    """
    tid = wp.tid()

    if tid < particle_indices.shape[0]:
        idx = particle_indices[tid]

        if idx < particle_positions.shape[0]:
            # Set particle position and velocity
            particle_positions[idx] = new_positions[tid]
            particle_velocities[idx] = wp.vec3(0.0, 0.0, 0.0)

@wp.kernel
def deactivate_particles_simple_kernel(
    particle_indices: wp.array(dtype=wp.int32),
    particle_positions: wp.array(dtype=wp.vec3),
    particle_velocities: wp.array(dtype=wp.vec3),
):
    """
    Simple kernel to deactivate particles by moving them far away.
    """
    tid = wp.tid()

    if tid < particle_indices.shape[0]:
        idx = particle_indices[tid]

        if idx < particle_positions.shape[0]:
            # Move particle far away and stop it
            particle_positions[idx] = wp.vec3(10000.0, 10000.0, 10000.0)
            particle_velocities[idx] = wp.vec3(0.0, 0.0, 0.0)

class MeshParticleConverter:
    """
    Handles conversion between mesh and particle representations for dynamic sand simulation.
    
    Provides methods to:
    1. Convert mesh regions to particles within a spherical zone
    2. Sample particle positions from mesh geometry
    3. Calculate appropriate particle properties
    4. Manage particle activation/deactivation
    """
    
    def __init__(self,
                 particle_properties: ParticleProperties = None,
                 device = None):
        self.device = device or wp.get_device()
        self.particle_props = particle_properties or ParticleProperties()
        
    def convert_mesh_to_particles(
        self,
        sphere_center: np.ndarray,
        sphere_radius: float,
        mesh_vertices: np.ndarray,
        mesh_indices: np.ndarray,
        particle_spacing: float = 0.05,
        available_particle_indices: List[int] = None
    ) -> Tuple[List[int], np.ndarray]:
        """
        Convert mesh region within sphere to particles.

        Args:
            sphere_center: 3D center of conversion sphere
            sphere_radius: Radius of conversion sphere
            mesh_vertices: Mesh vertex positions (N, 3)
            mesh_indices: Mesh triangle indices (M, 3)
            particle_spacing: Desired spacing between particles
            available_particle_indices: List of available particle indices to use

        Returns:
            Tuple of (used_particle_indices, particle_positions)
        """
        # Generate candidate particle positions in sphere
        candidate_positions = self._generate_sphere_grid(
            sphere_center, sphere_radius, particle_spacing
        )

        if len(candidate_positions) == 0:
            return [], np.array([])

        # CPU-based filtering to avoid kernel issues
        valid_positions = []
        threshold_distance = particle_spacing * 1.5

        for pos in candidate_positions:
            # Check if position is within sphere
            distance_to_center = np.linalg.norm(pos - sphere_center)
            if distance_to_center > sphere_radius:
                continue

            # Check if position is near mesh surface
            distances_to_vertices = np.linalg.norm(mesh_vertices - pos, axis=1)
            min_distance = np.min(distances_to_vertices)

            if min_distance <= threshold_distance:
                valid_positions.append(pos)

        valid_positions = np.array(valid_positions, dtype=np.float32)

        # Limit to available particle indices
        if available_particle_indices is not None:
            max_particles = min(len(valid_positions), len(available_particle_indices))
            valid_positions = valid_positions[:max_particles]
            used_indices = available_particle_indices[:max_particles]
        else:
            used_indices = list(range(len(valid_positions)))

        return used_indices, valid_positions
    
    def activate_particles(
        self,
        particle_indices: List[int],
        particle_positions: np.ndarray,
        model: newton.Model,
        state: newton.State
    ):
        """
        Activate particles in the Newton model and state with given positions.

        Args:
            particle_indices: Indices of particles to activate
            particle_positions: Positions for the particles (N, 3)
            model: Newton model object containing particle properties
            state: Newton state object to modify
        """
        if len(particle_indices) == 0:
            return

        # Use a simple kernel to activate particles
        if len(particle_indices) > 0:
            # Convert to Warp arrays
            indices_wp = wp.array(particle_indices, dtype=wp.int32, device=self.device)
            positions_wp = wp.array(particle_positions, dtype=wp.vec3, device=self.device)

            # Launch activation kernel
            wp.launch(
                activate_particles_simple_kernel,
                dim=len(particle_indices),
                inputs=[
                    indices_wp,
                    positions_wp,
                    state.particle_q,
                    state.particle_qd,
                ],
                device=self.device
            )
    
    def deactivate_particles(
        self,
        particle_indices: List[int],
        model: newton.Model,
        state: newton.State
    ):
        """
        Deactivate particles in the Newton model and state.

        Args:
            particle_indices: Indices of particles to deactivate
            model: Newton model object containing particle properties
            state: Newton state object to modify
        """
        if len(particle_indices) == 0:
            return

        # Use simple kernel to deactivate particles
        if len(particle_indices) > 0:
            # Convert to Warp arrays
            indices_wp = wp.array(particle_indices, dtype=wp.int32, device=self.device)

            # Launch deactivation kernel
            wp.launch(
                deactivate_particles_simple_kernel,
                dim=len(particle_indices),
                inputs=[
                    indices_wp,
                    state.particle_q,
                    state.particle_qd,
                ],
                device=self.device
            )
    
    def _generate_sphere_grid(
        self,
        center: np.ndarray,
        radius: float,
        spacing: float
    ) -> np.ndarray:
        """
        Generate a regular grid of positions within a sphere.
        
        Args:
            center: Sphere center (3,)
            radius: Sphere radius
            spacing: Grid spacing
            
        Returns:
            Array of positions within sphere (N, 3)
        """
        # Create bounding box
        min_bound = center - radius
        max_bound = center + radius
        
        # Generate grid points
        num_points = int(2 * radius / spacing) + 1
        x = np.linspace(min_bound[0], max_bound[0], num_points)
        y = np.linspace(min_bound[1], max_bound[1], num_points)
        z = np.linspace(min_bound[2], max_bound[2], num_points)
        
        # Create meshgrid and filter points within sphere
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        positions = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
        
        # Filter points within sphere
        distances = np.linalg.norm(positions - center, axis=1)
        valid_mask = distances <= radius
        
        return positions[valid_mask].astype(np.float32)
    
    def estimate_particle_count(
        self,
        sphere_center: np.ndarray,
        sphere_radius: float,
        mesh_vertices: np.ndarray,
        particle_spacing: float = 0.05
    ) -> int:
        """
        Estimate how many particles would be needed for a sphere-mesh intersection.
        
        Args:
            sphere_center: 3D center of sphere
            sphere_radius: Radius of sphere
            mesh_vertices: Mesh vertex positions
            particle_spacing: Desired particle spacing
            
        Returns:
            Estimated number of particles needed
        """
        # Simple estimation based on sphere volume and mesh bounds
        sphere_volume = (4.0/3.0) * np.pi * sphere_radius**3
        particle_volume = particle_spacing**3
        
        # Estimate intersection volume (simplified)
        mesh_bounds_min = np.min(mesh_vertices, axis=0)
        mesh_bounds_max = np.max(mesh_vertices, axis=0)
        
        # Check if sphere intersects mesh bounds
        sphere_min = sphere_center - sphere_radius
        sphere_max = sphere_center + sphere_radius
        
        # Calculate intersection volume with mesh bounding box
        intersection_min = np.maximum(sphere_min, mesh_bounds_min)
        intersection_max = np.minimum(sphere_max, mesh_bounds_max)
        
        if np.any(intersection_min >= intersection_max):
            return 0  # No intersection
        
        intersection_volume = np.prod(intersection_max - intersection_min)
        
        # Estimate particles needed (with some safety factor)
        estimated_particles = int(intersection_volume / particle_volume * 0.6)  # 60% packing efficiency
        
        return max(0, estimated_particles)
    
    def get_particle_properties_for_sand(self) -> dict:
        """
        Get appropriate particle properties for sand simulation.
        
        Returns:
            Dictionary of particle properties for MPM solver
        """
        return {
            'density': self.particle_props.density,
            'friction': self.particle_props.friction,
            'cohesion': self.particle_props.cohesion,
            'young_modulus': self.particle_props.young_modulus,
            'poisson_ratio': self.particle_props.poisson_ratio
        }
