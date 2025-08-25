# SPDX-License-Identifier: Apache-2.0
# Particle-to-Mesh Reconstruction for Dynamic Sand Plow
"""
Algorithms to reconstruct mesh geometry from particle positions when the sphere
moves away, including surface reconstruction and mesh updating for realistic
sand deformation representation.
"""

import numpy as np
import warp as wp
from typing import List, Tuple, Optional
from scipy.spatial import ConvexHull, Delaunay
from scipy.interpolate import griddata
import newton

@wp.kernel
def extract_particle_positions_kernel(
    particle_indices: wp.array(dtype=wp.int32),
    particle_positions: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    output_positions: wp.array(dtype=wp.vec3),
    output_count: wp.array(dtype=wp.int32),
):
    """
    Extract positions of active particles for mesh reconstruction.
    """
    tid = wp.tid()
    
    if tid < particle_indices.shape[0]:
        idx = particle_indices[tid]
        
        # Only extract active particles
        if particle_flags[idx] > 0:
            output_positions[tid] = particle_positions[idx]
            wp.atomic_add(output_count, 0, 1)
        else:
            output_positions[tid] = wp.vec3(0.0, 0.0, 0.0)

@wp.kernel
def smooth_mesh_vertices_kernel(
    vertices: wp.array(dtype=wp.vec3),
    vertex_neighbors: wp.array(dtype=wp.int32),  # Flattened neighbor indices
    neighbor_counts: wp.array(dtype=wp.int32),   # Number of neighbors per vertex
    neighbor_offsets: wp.array(dtype=wp.int32),  # Offset into neighbor array
    smoothed_vertices: wp.array(dtype=wp.vec3),
    smoothing_factor: float,
):
    """
    Apply Laplacian smoothing to mesh vertices for more natural surface.
    """
    tid = wp.tid()
    
    if tid < vertices.shape[0]:
        vertex = vertices[tid]
        neighbor_count = neighbor_counts[tid]
        
        if neighbor_count > 0:
            # Calculate average of neighboring vertices
            neighbor_sum = wp.vec3(0.0, 0.0, 0.0)
            offset = neighbor_offsets[tid]
            
            for i in range(neighbor_count):
                neighbor_idx = vertex_neighbors[offset + i]
                if neighbor_idx >= 0 and neighbor_idx < vertices.shape[0]:
                    neighbor_sum += vertices[neighbor_idx]
            
            neighbor_avg = neighbor_sum / float(neighbor_count)
            
            # Apply smoothing
            smoothed_vertices[tid] = vertex + smoothing_factor * (neighbor_avg - vertex)
        else:
            smoothed_vertices[tid] = vertex

class ParticleMeshReconstructor:
    """
    Reconstructs mesh geometry from particle positions using various algorithms.
    
    Provides methods for:
    1. Surface reconstruction from particle point clouds
    2. Heightfield generation from particles
    3. Mesh smoothing and optimization
    4. Volume-preserving reconstruction
    """
    
    def __init__(self, device = None):
        self.device = device or wp.get_device()
        
    def reconstruct_mesh_from_particles(
        self,
        particle_positions: np.ndarray,
        original_bounds: Tuple[np.ndarray, np.ndarray],
        method: str = "heightfield",
        resolution: int = 32
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct mesh from particle positions.
        
        Args:
            particle_positions: Particle positions (N, 3)
            original_bounds: (min_bounds, max_bounds) of original mesh region
            method: Reconstruction method ("heightfield", "convex_hull", "alpha_shape")
            resolution: Grid resolution for heightfield method
            
        Returns:
            Tuple of (vertices, indices) for reconstructed mesh
        """
        if len(particle_positions) == 0:
            return self._create_empty_mesh(original_bounds)
        
        if method == "heightfield":
            return self._reconstruct_heightfield(particle_positions, original_bounds, resolution)
        elif method == "convex_hull":
            return self._reconstruct_convex_hull(particle_positions)
        elif method == "alpha_shape":
            return self._reconstruct_alpha_shape(particle_positions)
        else:
            raise ValueError(f"Unknown reconstruction method: {method}")
    
    def _reconstruct_heightfield(
        self,
        particle_positions: np.ndarray,
        original_bounds: Tuple[np.ndarray, np.ndarray],
        resolution: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct mesh using heightfield interpolation.
        
        Creates a heightfield surface by interpolating particle heights
        over a regular grid. Good for terrain-like surfaces.
        """
        min_bounds, max_bounds = original_bounds
        
        # Create grid for heightfield
        x_grid = np.linspace(min_bounds[0], max_bounds[0], resolution)
        z_grid = np.linspace(min_bounds[2], max_bounds[2], resolution)
        xx, zz = np.meshgrid(x_grid, z_grid, indexing='ij')
        
        # Interpolate heights from particles
        if len(particle_positions) >= 3:
            # Use particle positions for interpolation
            particle_xy = particle_positions[:, [0, 2]]  # X-Z coordinates
            particle_heights = particle_positions[:, 1]  # Y coordinates
            
            # Interpolate heights on grid
            grid_points = np.stack([xx.flatten(), zz.flatten()], axis=1)
            interpolated_heights = griddata(
                particle_xy, particle_heights, grid_points,
                method='linear', fill_value=min_bounds[1]
            )
            heights = interpolated_heights.reshape(resolution, resolution)
        else:
            # Fallback: flat surface at minimum height
            heights = np.full((resolution, resolution), min_bounds[1])
        
        # Generate vertices
        vertices = []
        for i in range(resolution):
            for j in range(resolution):
                x = xx[i, j]
                y = heights[i, j]
                z = zz[i, j]
                vertices.append([x, y, z])
        
        vertices = np.array(vertices, dtype=np.float32)
        
        # Generate triangle indices
        indices = []
        for i in range(resolution - 1):
            for j in range(resolution - 1):
                # Two triangles per quad
                v0 = i * resolution + j
                v1 = v0 + 1
                v2 = v0 + resolution
                v3 = v2 + 1
                
                # Triangle 1: v0, v1, v2
                indices.extend([v0, v1, v2])
                # Triangle 2: v1, v3, v2
                indices.extend([v1, v3, v2])
        
        indices = np.array(indices, dtype=np.int32)
        
        return vertices, indices
    
    def _reconstruct_convex_hull(
        self,
        particle_positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct mesh using 3D convex hull.
        
        Creates a convex mesh that encloses all particles.
        Simple but may not preserve concave features.
        """
        if len(particle_positions) < 4:
            # Need at least 4 points for 3D convex hull
            return self._create_simple_box_mesh(particle_positions)
        
        try:
            hull = ConvexHull(particle_positions)
            vertices = hull.points[hull.vertices].astype(np.float32)
            indices = hull.simplices.flatten().astype(np.int32)
            return vertices, indices
        except Exception:
            # Fallback to simple box mesh
            return self._create_simple_box_mesh(particle_positions)
    
    def _reconstruct_alpha_shape(
        self,
        particle_positions: np.ndarray,
        alpha: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct mesh using alpha shapes (simplified version).
        
        Creates a more detailed surface that can capture concave features.
        This is a simplified implementation - full alpha shapes are more complex.
        """
        if len(particle_positions) < 4:
            return self._create_simple_box_mesh(particle_positions)
        
        try:
            # Use Delaunay triangulation as approximation
            # In practice, would implement proper alpha shapes
            tri = Delaunay(particle_positions)
            
            # Filter triangles based on circumradius (alpha criterion)
            valid_simplices = []
            for simplex in tri.simplices:
                # Calculate circumradius of tetrahedron
                points = particle_positions[simplex]
                circumradius = self._calculate_circumradius(points)
                
                if circumradius <= alpha:
                    valid_simplices.append(simplex)
            
            if len(valid_simplices) == 0:
                return self._create_simple_box_mesh(particle_positions)
            
            # Extract surface triangles
            vertices = particle_positions.astype(np.float32)
            indices = np.array(valid_simplices).flatten().astype(np.int32)
            
            return vertices, indices
            
        except Exception:
            return self._create_simple_box_mesh(particle_positions)
    
    def _calculate_circumradius(self, points: np.ndarray) -> float:
        """Calculate circumradius of a tetrahedron (simplified)."""
        if len(points) != 4:
            return float('inf')
        
        # Simplified circumradius calculation
        # In practice, would use proper geometric calculation
        center = np.mean(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        return np.max(distances)
    
    def _create_simple_box_mesh(
        self,
        particle_positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create a simple box mesh around particles."""
        if len(particle_positions) == 0:
            # Default small box
            min_bounds = np.array([-0.1, -0.1, -0.1])
            max_bounds = np.array([0.1, 0.1, 0.1])
        else:
            min_bounds = np.min(particle_positions, axis=0)
            max_bounds = np.max(particle_positions, axis=0)
            
            # Add small margin
            margin = 0.05
            min_bounds -= margin
            max_bounds += margin
        
        # Create box vertices
        vertices = np.array([
            [min_bounds[0], min_bounds[1], min_bounds[2]],  # 0
            [max_bounds[0], min_bounds[1], min_bounds[2]],  # 1
            [min_bounds[0], max_bounds[1], min_bounds[2]],  # 2
            [max_bounds[0], max_bounds[1], min_bounds[2]],  # 3
            [min_bounds[0], min_bounds[1], max_bounds[2]],  # 4
            [max_bounds[0], min_bounds[1], max_bounds[2]],  # 5
            [min_bounds[0], max_bounds[1], max_bounds[2]],  # 6
            [max_bounds[0], max_bounds[1], max_bounds[2]],  # 7
        ], dtype=np.float32)
        
        # Create box faces (12 triangles)
        indices = np.array([
            # Bottom face
            0, 1, 2, 1, 3, 2,
            # Top face
            4, 6, 5, 5, 6, 7,
            # Front face
            0, 2, 4, 2, 6, 4,
            # Back face
            1, 5, 3, 3, 5, 7,
            # Left face
            0, 4, 1, 1, 4, 5,
            # Right face
            2, 3, 6, 3, 7, 6,
        ], dtype=np.int32)
        
        return vertices, indices
    
    def _create_empty_mesh(
        self,
        original_bounds: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create an empty mesh (flat surface) when no particles are present."""
        min_bounds, max_bounds = original_bounds
        
        # Create flat surface at minimum height
        vertices = np.array([
            [min_bounds[0], min_bounds[1], min_bounds[2]],
            [max_bounds[0], min_bounds[1], min_bounds[2]],
            [min_bounds[0], min_bounds[1], max_bounds[2]],
            [max_bounds[0], min_bounds[1], max_bounds[2]],
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 1, 3, 2], dtype=np.int32)
        
        return vertices, indices
    
    def smooth_mesh(
        self,
        vertices: np.ndarray,
        indices: np.ndarray,
        iterations: int = 3,
        smoothing_factor: float = 0.1
    ) -> np.ndarray:
        """
        Apply Laplacian smoothing to mesh for more natural appearance.
        
        Args:
            vertices: Mesh vertices (N, 3)
            indices: Mesh triangle indices (M, 3)
            iterations: Number of smoothing iterations
            smoothing_factor: Strength of smoothing (0-1)
            
        Returns:
            Smoothed vertices
        """
        if len(vertices) == 0:
            return vertices
        
        # Build vertex neighbor information
        neighbor_lists = [[] for _ in range(len(vertices))]
        
        # Extract edges from triangles
        triangles = indices.reshape(-1, 3)
        for tri in triangles:
            for i in range(3):
                v1, v2 = tri[i], tri[(i + 1) % 3]
                neighbor_lists[v1].append(v2)
                neighbor_lists[v2].append(v1)
        
        # Remove duplicates and convert to arrays
        for i in range(len(neighbor_lists)):
            neighbor_lists[i] = list(set(neighbor_lists[i]))
        
        # Apply smoothing iterations
        smoothed_vertices = vertices.copy()
        
        for _ in range(iterations):
            new_vertices = smoothed_vertices.copy()
            
            for i, vertex in enumerate(smoothed_vertices):
                neighbors = neighbor_lists[i]
                if len(neighbors) > 0:
                    neighbor_avg = np.mean(smoothed_vertices[neighbors], axis=0)
                    new_vertices[i] = vertex + smoothing_factor * (neighbor_avg - vertex)
            
            smoothed_vertices = new_vertices
        
        return smoothed_vertices.astype(np.float32)
    
    def extract_particle_positions(
        self,
        particle_indices: List[int],
        state: newton.State
    ) -> np.ndarray:
        """
        Extract positions of specified particles from Newton state.

        Args:
            particle_indices: Indices of particles to extract
            state: Newton state object

        Returns:
            Array of particle positions (N, 3)
        """
        if len(particle_indices) == 0:
            return np.array([]).reshape(0, 3)

        # Simple CPU-based extraction to avoid kernel issues
        valid_positions = []

        # Get particle positions directly from state
        particle_q_np = state.particle_q.numpy()

        for idx in particle_indices:
            if idx < len(particle_q_np):
                pos = particle_q_np[idx]
                # Filter out particles that are far away (deactivated)
                if not (abs(pos[0]) > 1000 or abs(pos[1]) > 1000 or abs(pos[2]) > 1000):
                    valid_positions.append([pos[0], pos[1], pos[2]])

        return np.array(valid_positions, dtype=np.float32) if valid_positions else np.array([]).reshape(0, 3)
