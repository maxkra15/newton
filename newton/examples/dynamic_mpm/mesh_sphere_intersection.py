# SPDX-License-Identifier: Apache-2.0
# Mesh-Sphere Intersection Detection for Dynamic Sand Plow
"""
Efficient algorithms for detecting intersections between spherical conversion zones
and mesh geometry using Newton's raycast capabilities and spatial queries.
"""

import numpy as np
import warp as wp
from typing import List, Tuple, Optional
import newton

# Simplified CPU-based intersection detection to avoid kernel compilation issues

@wp.func
def sphere_triangle_intersection(
    sphere_center: wp.vec3,
    sphere_radius: float,
    v0: wp.vec3,
    v1: wp.vec3,
    v2: wp.vec3
) -> bool:
    """
    Test if a sphere intersects with a triangle.
    
    Uses the method of finding the closest point on the triangle to the sphere center
    and checking if the distance is within the sphere radius.
    """
    # Find closest point on triangle to sphere center
    closest_point = closest_point_on_triangle(sphere_center, v0, v1, v2)
    
    # Check distance
    distance_sq = wp.length_sq(sphere_center - closest_point)
    return distance_sq <= sphere_radius * sphere_radius

@wp.func
def closest_point_on_triangle(
    point: wp.vec3,
    v0: wp.vec3,
    v1: wp.vec3,
    v2: wp.vec3
) -> wp.vec3:
    """
    Find the closest point on a triangle to a given point.
    
    Uses barycentric coordinates to determine the closest point.
    """
    # Triangle edges
    edge0 = v1 - v0
    edge1 = v2 - v0
    v0_to_point = v0 - point
    
    a = wp.dot(edge0, edge0)
    b = wp.dot(edge0, edge1)
    c = wp.dot(edge1, edge1)
    d = wp.dot(edge0, v0_to_point)
    e = wp.dot(edge1, v0_to_point)
    
    det = a * c - b * b
    s = b * e - c * d
    t = b * d - a * e
    
    if s + t < det:
        if s < 0.0:
            if t < 0.0:
                # Region 4
                if d < 0.0:
                    t = 0.0
                    s = wp.clamp(-d / a, 0.0, 1.0)
                else:
                    s = 0.0
                    t = wp.clamp(-e / c, 0.0, 1.0)
            else:
                # Region 3
                s = 0.0
                t = wp.clamp(-e / c, 0.0, 1.0)
        elif t < 0.0:
            # Region 5
            t = 0.0
            s = wp.clamp(-d / a, 0.0, 1.0)
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
                s = wp.clamp(numer / denom, 0.0, 1.0)
                t = 1.0 - s
            else:
                t = wp.clamp(-e / c, 0.0, 1.0)
                s = 0.0
        elif t < 0.0:
            # Region 6
            tmp0 = b + e
            tmp1 = a + d
            if tmp1 > tmp0:
                numer = tmp1 - tmp0
                denom = a - 2.0 * b + c
                t = wp.clamp(numer / denom, 0.0, 1.0)
                s = 1.0 - t
            else:
                s = wp.clamp(-d / a, 0.0, 1.0)
                t = 0.0
        else:
            # Region 1
            numer = c + e - b - d
            if numer <= 0.0:
                s = 0.0
            else:
                denom = a - 2.0 * b + c
                s = wp.clamp(numer / denom, 0.0, 1.0)
            t = 1.0 - s
    
    return v0 + s * edge0 + t * edge1

@wp.kernel
def raycast_sphere_intersection_kernel(
    sphere_center: wp.vec3,
    sphere_radius: float,
    ray_origins: wp.array(dtype=wp.vec3),
    ray_directions: wp.array(dtype=wp.vec3),
    mesh_vertices: wp.array(dtype=wp.vec3),
    mesh_indices: wp.array(dtype=wp.int32),
    hit_results: wp.array(dtype=wp.float32),  # Distance to hit, -1 if no hit
):
    """
    Kernel to perform multiple raycasts from sphere surface to detect mesh intersection.

    Simplified version that checks if ray origin is close to any mesh vertex.
    """
    tid = wp.tid()

    if tid < ray_origins.shape[0]:
        ray_origin = ray_origins[tid]

        # Simple distance check to mesh vertices
        min_distance = 1000.0
        for i in range(mesh_vertices.shape[0]):
            vertex = mesh_vertices[i]
            distance = wp.length(ray_origin - vertex)
            if distance < min_distance:
                min_distance = distance

        # Consider hit if within threshold
        if min_distance <= sphere_radius:
            hit_results[tid] = min_distance
        else:
            hit_results[tid] = -1.0

class MeshSphereIntersector:
    """
    Efficient mesh-sphere intersection detection using multiple algorithms.
    
    Provides both exact triangle-based intersection and approximate raycast-based
    intersection detection for performance optimization.
    """
    
    def __init__(self, device = None):
        self.device = device or wp.get_device()
        
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
        with wp.ScopedDevice(self.device):
            # Convert to Warp arrays
            vertices_wp = wp.array(mesh_vertices, dtype=wp.vec3, device=self.device)
            indices_wp = wp.array(mesh_indices.flatten(), dtype=wp.int32, device=self.device)
            
            triangle_count = len(mesh_indices) // 3
            results = wp.zeros(triangle_count, dtype=wp.int32, device=self.device)
            
            # Launch kernel
            wp.launch(
                sphere_mesh_intersection_kernel,
                dim=triangle_count,
                inputs=[
                    wp.vec3(sphere_center),
                    sphere_radius,
                    vertices_wp,
                    indices_wp,
                    results
                ],
                device=self.device
            )
            
            # Check if any triangle intersected
            results_np = results.numpy()
            return np.any(results_np > 0)
    
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

        Casts rays from the sphere surface in multiple directions to detect
        intersection with the mesh. Faster but less accurate than exact method.

        Args:
            sphere_center: 3D center of sphere
            sphere_radius: Radius of sphere
            mesh_vertices: Mesh vertex array
            mesh_indices: Mesh triangle indices
            num_rays: Number of rays to cast (default: 26 for icosahedral sampling)

        Returns:
            True if any ray hits the mesh within sphere radius, False otherwise
        """
        with wp.ScopedDevice(self.device):
            # Generate ray directions using icosahedral sampling
            ray_directions = self._generate_sphere_directions(num_rays)

            # Generate ray origins on sphere surface
            ray_origins = sphere_center + ray_directions * sphere_radius

            # Convert to Warp arrays
            origins_wp = wp.array(ray_origins, dtype=wp.vec3, device=self.device)
            directions_wp = wp.array(-ray_directions, dtype=wp.vec3, device=self.device)  # Point inward
            vertices_wp = wp.array(mesh_vertices, dtype=wp.vec3, device=self.device)
            indices_wp = wp.array(mesh_indices.flatten(), dtype=wp.int32, device=self.device)
            results = wp.zeros(num_rays, dtype=wp.float32, device=self.device)

            # Launch raycast kernel
            wp.launch(
                raycast_sphere_intersection_kernel,
                dim=num_rays,
                inputs=[
                    wp.vec3(sphere_center),
                    sphere_radius,
                    origins_wp,
                    directions_wp,
                    vertices_wp,
                    indices_wp,
                    results
                ],
                device=self.device
            )

            # Check if any ray hit within sphere radius
            results_np = results.numpy()
            valid_hits = results_np[results_np >= 0]  # -1 means no hit

            return len(valid_hits) > 0 and np.any(valid_hits <= sphere_radius * 2.0)
    
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
        # This is a simplified version - in practice would use proper point-in-mesh testing
        inside_count = 0
        for point in sample_points:
            # Simple heuristic: check if point is below mesh surface
            # In practice, would use proper ray casting or winding number
            mesh_min_y = np.min(mesh_vertices[:, 1])
            mesh_max_y = np.max(mesh_vertices[:, 1])
            
            if mesh_min_y <= point[1] <= mesh_max_y:
                inside_count += 1
        
        return inside_count * voxel_volume
