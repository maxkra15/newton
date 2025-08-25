# SPDX-License-Identifier: Apache-2.0
# Simplified Mesh-Sphere Intersection Detection for Dynamic Sand Plow
"""
Simplified CPU-based algorithms for detecting intersections between spherical 
conversion zones and mesh geometry. Avoids complex Warp kernels that have 
compilation issues.
"""

import numpy as np
from typing import List, Tuple, Optional

def sphere_triangle_intersection_cpu(
    sphere_center: np.ndarray,
    sphere_radius: float,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray
) -> bool:
    """
    CPU-based sphere-triangle intersection test.
    
    Args:
        sphere_center: 3D center of sphere
        sphere_radius: Radius of sphere
        v0, v1, v2: Triangle vertices
        
    Returns:
        True if sphere intersects triangle, False otherwise
    """
    # Find closest point on triangle to sphere center
    closest_point = closest_point_on_triangle_cpu(sphere_center, v0, v1, v2)
    
    # Check distance
    distance_sq = np.sum((sphere_center - closest_point) ** 2)
    return distance_sq <= sphere_radius * sphere_radius

def closest_point_on_triangle_cpu(
    point: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray
) -> np.ndarray:
    """
    Find the closest point on a triangle to a given point (CPU version).
    
    Uses barycentric coordinates to determine the closest point.
    """
    # Triangle edges
    edge0 = v1 - v0
    edge1 = v2 - v0
    v0_to_point = v0 - point
    
    a = np.dot(edge0, edge0)
    b = np.dot(edge0, edge1)
    c = np.dot(edge1, edge1)
    d = np.dot(edge0, v0_to_point)
    e = np.dot(edge1, v0_to_point)
    
    det = a * c - b * b
    s = b * e - c * d
    t = b * d - a * e
    
    if s + t < det:
        if s < 0.0:
            if t < 0.0:
                # Region 4
                if d < 0.0:
                    t = 0.0
                    s = np.clip(-d / a, 0.0, 1.0)
                else:
                    s = 0.0
                    t = np.clip(-e / c, 0.0, 1.0)
            else:
                # Region 3
                s = 0.0
                t = np.clip(-e / c, 0.0, 1.0)
        elif t < 0.0:
            # Region 5
            t = 0.0
            s = np.clip(-d / a, 0.0, 1.0)
        else:
            # Region 0
            inv_det = 1.0 / det
            s *= inv_det
            t *= inv_det
    else:
        if s < 0.0:
            # Region 2
            tmp0 = b + d
            tmp1 = c + e
            if tmp1 > tmp0:
                numer = tmp1 - tmp0
                denom = a - 2.0 * b + c
                s = np.clip(numer / denom, 0.0, 1.0)
                t = 1.0 - s
            else:
                t = np.clip(-e / c, 0.0, 1.0)
                s = 0.0
        elif t < 0.0:
            # Region 6
            tmp0 = b + e
            tmp1 = a + d
            if tmp1 > tmp0:
                numer = tmp1 - tmp0
                denom = a - 2.0 * b + c
                t = np.clip(numer / denom, 0.0, 1.0)
                s = 1.0 - t
            else:
                s = np.clip(-d / a, 0.0, 1.0)
                t = 0.0
        else:
            # Region 1
            numer = c + e - b - d
            if numer <= 0.0:
                s = 0.0
            else:
                denom = a - 2.0 * b + c
                s = np.clip(numer / denom, 0.0, 1.0)
            t = 1.0 - s
    
    return v0 + s * edge0 + t * edge1

class MeshSphereIntersector:
    """
    Simplified mesh-sphere intersection detection using CPU-based algorithms.
    
    Provides both exact triangle-based intersection and approximate methods
    for performance optimization.
    """
    
    def __init__(self, device = None):
        self.device = device  # Keep for compatibility but use CPU
        
    def check_intersection_exact(
        self,
        sphere_center: np.ndarray,
        sphere_radius: float,
        mesh_vertices: np.ndarray,
        mesh_indices: np.ndarray
    ) -> bool:
        """
        Exact sphere-mesh intersection using triangle tests.
        
        Args:
            sphere_center: 3D center of sphere
            sphere_radius: Radius of sphere
            mesh_vertices: Mesh vertex array (N, 3)
            mesh_indices: Mesh triangle indices (M, 3)
            
        Returns:
            True if sphere intersects mesh, False otherwise
        """
        # Reshape indices to triangles
        triangles = mesh_indices.reshape(-1, 3)
        
        # Test each triangle
        for tri in triangles:
            v0 = mesh_vertices[tri[0]]
            v1 = mesh_vertices[tri[1]]
            v2 = mesh_vertices[tri[2]]
            
            if sphere_triangle_intersection_cpu(sphere_center, sphere_radius, v0, v1, v2):
                return True
        
        return False
    
    def check_intersection_raycast(
        self,
        sphere_center: np.ndarray,
        sphere_radius: float,
        mesh_vertices: np.ndarray,
        mesh_indices: np.ndarray,
        num_rays: int = 26
    ) -> bool:
        """
        Approximate sphere-mesh intersection using raycast sampling.
        
        Simplified version that checks if sphere center is close to mesh vertices.
        
        Args:
            sphere_center: 3D center of sphere
            sphere_radius: Radius of sphere
            mesh_vertices: Mesh vertex array
            mesh_indices: Mesh triangle indices
            num_rays: Number of rays to cast (ignored in simplified version)
            
        Returns:
            True if sphere is close to mesh, False otherwise
        """
        # Simple distance check to mesh vertices
        distances = np.linalg.norm(mesh_vertices - sphere_center, axis=1)
        min_distance = np.min(distances)
        
        return min_distance <= sphere_radius * 1.5  # Add some tolerance
    
    def _generate_sphere_directions(self, num_directions: int) -> np.ndarray:
        """
        Generate uniformly distributed directions on a sphere.
        
        Uses the golden spiral method for uniform distribution.
        """
        directions = []
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle in radians
        
        for i in range(num_directions):
            # Golden spiral method
            y = 1 - (i / float(num_directions - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)
            
            theta = golden_angle * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            directions.append([x, y, z])
        
        return np.array(directions, dtype=np.float32)
    
    def get_intersection_volume(
        self,
        sphere_center: np.ndarray,
        sphere_radius: float,
        mesh_vertices: np.ndarray,
        mesh_indices: np.ndarray,
        resolution: int = 32
    ) -> float:
        """
        Estimate the volume of intersection between sphere and mesh.
        
        Uses voxel sampling within the sphere to estimate intersection volume.
        Useful for determining how much material to convert to particles.
        
        Args:
            sphere_center: 3D center of sphere
            sphere_radius: Radius of sphere
            mesh_vertices: Mesh vertex array
            mesh_indices: Mesh triangle indices
            resolution: Voxel resolution for sampling
            
        Returns:
            Estimated intersection volume
        """
        # Create voxel grid within sphere
        voxel_size = (2 * sphere_radius) / resolution
        voxel_volume = voxel_size ** 3
        
        # Sample points within sphere
        sample_points = []
        for x in range(resolution):
            for y in range(resolution):
                for z in range(resolution):
                    # Convert to world coordinates
                    local_pos = np.array([x, y, z]) * voxel_size - sphere_radius
                    world_pos = sphere_center + local_pos
                    
                    # Check if point is within sphere
                    if np.linalg.norm(local_pos) <= sphere_radius:
                        sample_points.append(world_pos)
        
        if not sample_points:
            return 0.0
        
        # Count how many sample points are inside the mesh
        # Simplified version: check if point is close to mesh vertices
        inside_count = 0
        for point in sample_points:
            distances = np.linalg.norm(mesh_vertices - point, axis=1)
            min_distance = np.min(distances)
            
            # Simple heuristic: consider inside if close to mesh
            if min_distance <= voxel_size:
                inside_count += 1
        
        return inside_count * voxel_volume
    
    def check_sphere_box_intersection(
        self,
        sphere_center: np.ndarray,
        sphere_radius: float,
        box_min: np.ndarray,
        box_max: np.ndarray
    ) -> bool:
        """
        Check if sphere intersects with axis-aligned bounding box.
        
        Args:
            sphere_center: 3D center of sphere
            sphere_radius: Radius of sphere
            box_min: Minimum corner of box
            box_max: Maximum corner of box
            
        Returns:
            True if sphere intersects box, False otherwise
        """
        # Find closest point on box to sphere center
        closest_point = np.clip(sphere_center, box_min, box_max)
        
        # Check if distance to closest point is within sphere radius
        distance_sq = np.sum((sphere_center - closest_point) ** 2)
        return distance_sq <= sphere_radius ** 2
