# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Confine reconstructed water to the appropriate side of the moving PET wall."""

import numpy as np
import warp as wp

from newton.geometry import ParticleSurface


def compute_normals(vertices: wp.array[wp.vec3], indices: wp.array[int], normals: wp.array[wp.vec3]):
    """Reuse GPU output buffers for area-weighted bottle and liquid normals."""
    normals.zero_()
    wp.launch(_accumulate_normals, len(indices) // 3, inputs=[vertices, indices, normals], device=vertices.device)
    wp.launch(_normalize_normals, len(normals), inputs=[normals], device=vertices.device)


@wp.kernel
def _particle_sides(mesh: wp.uint64, positions: wp.array[wp.vec3], sides: wp.array[float]):
    i = wp.tid()
    query = wp.mesh_query_point_sign_winding_number(mesh, positions[i], 0.05)
    sides[i] = wp.where(query.result, query.sign, 1.0)


@wp.kernel
def _confine_field(
    volume: wp.uint64,
    coordinates: wp.array[wp.vec3i],
    field: wp.array[float],
    mesh: wp.uint64,
    shell_mesh: wp.uint64,
    shell_triangle_count: int,
    grid: wp.uint64,
    positions: wp.array[wp.vec3],
    sides: wp.array[float],
    voxel_size: float,
    kernel_radius: float,
    search_radius: float,
    margin: float,
    floor: float,
):
    i = wp.tid()
    ijk = coordinates[i]
    index = wp.volume_lookup_index(volume, ijk[0], ijk[1], ijk[2])
    if index < 0:
        return
    point = voxel_size * wp.vec3(float(ijk[0]), float(ijk[1]), float(ijk[2]))
    value = wp.max(field[index], (floor - point[2]) / kernel_radius)
    field[index] = value
    query = wp.mesh_query_point_no_sign(shell_mesh, point, search_radius)
    if not query.result:
        return
    signed_query = wp.mesh_query_point_sign_winding_number(mesh, point, search_radius)
    if signed_query.face >= shell_triangle_count:
        # The actual opening moves and tilts with the shell. Its temporary
        # classification cap must never act as a water boundary.
        return
    closest = wp.mesh_eval_position(shell_mesh, query.face, query.u, query.v)
    distance = wp.length(point - closest)
    # Far inside the bottle, or far outside the liquid, the zero contour cannot
    # intersect the wall. Avoid unnecessary neighbor queries in those regions.
    if distance > margin + 2.0 * voxel_size and (signed_query.sign < 0.0 or value > 0.0):
        return
    nearest = int(-1)
    nearest_distance = search_radius * search_radius
    for particle in wp.hash_grid_query(grid, point, search_radius):
        delta = point - positions[particle]
        squared_distance = wp.dot(delta, delta)
        if squared_distance < nearest_distance:
            nearest = particle
            nearest_distance = squared_distance
    if nearest >= 0:
        # Negative field values denote liquid. Intersect with the interior for
        # contained particles and with the exterior for actual spilled water.
        boundary = (margin - sides[nearest] * signed_query.sign * distance) / kernel_radius
        field[index] = wp.max(value, boundary)


class BottleSurface:
    """Clip an anisotropic density field before marching cubes, in SI units.

    A temporary neck cap supplies a robust winding-number sign. It contributes
    no collision or rendered geometry, and its faces never clip the liquid.
    The nearest fluid particle determines which side of the PET wall to keep.
    """

    def __init__(
        self,
        shell_positions,
        shell_triangles,
        *,
        segments,
        spacing,
        voxel_size,
        margin,
        floor,
        max_grid_cells,
        device,
    ):
        self.device = device
        self.voxel_size = voxel_size
        self.kernel_radius = 3.0 * spacing
        self.search_radius = 4.0 * spacing + 2.0 * voxel_size
        # Resolve the inset even when the reconstruction voxels are coarser
        # than the physical wall thickness.
        self.margin = max(margin, 0.5 * voxel_size)
        self.floor = floor
        self.shell_triangle_count = len(shell_triangles)
        top = len(shell_positions) - segments
        cap = np.array([[top, top + j, top + j + 1] for j in range(1, segments - 1)], dtype=np.int32)
        indices = np.concatenate((shell_triangles, cap)).flatten()
        self.mesh = wp.Mesh(
            points=shell_positions,
            indices=wp.array(indices, dtype=int, device=device),
            support_winding_number=True,
        )
        self.shell_mesh = wp.Mesh(
            points=shell_positions,
            indices=wp.array(np.asarray(shell_triangles).flatten(), dtype=int, device=device),
        )
        self.grid = wp.HashGrid(64, 64, 128, device=device)
        self.sides = None
        self.vertex_sides = None
        self.coordinates = None
        self.surface = ParticleSurface(
            voxel_size=voxel_size,
            kernel_radius=self.kernel_radius,
            threshold=0.25,
            smooth_lambda=0.0,
            anisotropic=True,
            anisotropy_min_neighbors=8,
            anisotropy_ratio=4.0,
            anisotropy_binning=True,
            anisotropy_strength=0.6,
            kernel_scale=0.65,
            field_mode="sdf",
            field_smooth_iterations=1,
            # Smoothing after clipping would move vertices back through the wall.
            mesh_smooth_iterations=0,
            max_grid_cells=max_grid_cells,
            device=device,
        )

    def extract(self, positions, radii):
        self.mesh.refit()
        self.shell_mesh.refit()
        if self.sides is None or len(self.sides) != len(positions):
            self.sides = wp.empty(len(positions), dtype=float, device=self.device)
            self.grid.reserve(len(positions))
        wp.launch(_particle_sides, len(positions), inputs=[self.mesh.id, positions, self.sides], device=self.device)
        self.grid.build(positions, self.kernel_radius)
        field = self.surface.update_field(positions, radii)
        if field is not None:
            count = field.volume.get_voxel_count()
            if self.coordinates is None or len(self.coordinates) < count:
                self.coordinates = wp.empty(2 * count, dtype=wp.vec3i, device=self.device)
            field.volume.get_voxels(out=self.coordinates)
            wp.launch(
                _confine_field,
                count,
                inputs=[
                    field.volume.id,
                    self.coordinates,
                    field.voxel_data,
                    self.mesh.id,
                    self.shell_mesh.id,
                    self.shell_triangle_count,
                    self.grid.id,
                    positions,
                    self.sides,
                    self.voxel_size,
                    self.kernel_radius,
                    self.search_radius,
                    self.margin,
                    self.floor,
                ],
                device=self.device,
            )
        mesh = self.surface.resurface()
        vertices, indices, normals = mesh.to_arrays()
        if vertices is not None:
            if self.vertex_sides is None or len(self.vertex_sides) < len(vertices):
                self.vertex_sides = wp.empty(2 * len(vertices), dtype=float, device=self.device)
            # A sub-voxel crease can cross a marching-cubes triangle even when
            # all field nodes are on the correct side. Guard the final vertices.
            wp.launch(
                _confine_vertices,
                len(vertices),
                inputs=[
                    self.mesh.id,
                    self.shell_mesh.id,
                    self.shell_triangle_count,
                    self.grid.id,
                    positions,
                    self.sides,
                    vertices,
                    self.search_radius,
                    self.margin,
                    self.floor,
                    self.vertex_sides,
                ],
                device=self.device,
            )
            compute_normals(vertices, indices, normals)
        return mesh


@wp.kernel
def _confine_vertices(
    mesh: wp.uint64,
    shell_mesh: wp.uint64,
    shell_triangle_count: int,
    grid: wp.uint64,
    positions: wp.array[wp.vec3],
    sides: wp.array[float],
    vertices: wp.array[wp.vec3],
    search_radius: float,
    margin: float,
    floor: float,
    vertex_sides: wp.array[float],
):
    i = wp.tid()
    point = vertices[i]
    point[2] = wp.max(point[2], floor)
    vertex_sides[i] = 0.0
    nearest = int(-1)
    nearest_distance = search_radius * search_radius
    for particle in wp.hash_grid_query(grid, point, search_radius):
        squared_distance = wp.dot(point - positions[particle], point - positions[particle])
        if squared_distance < nearest_distance:
            nearest = particle
            nearest_distance = squared_distance
    if nearest < 0:
        vertices[i] = point
        return
    side = sides[nearest]
    # Keep the emitting component fixed while correcting the vertex. Choosing
    # its side again after displacement can switch to a different fluid source.
    vertex_sides[i] = side
    for _ in range(4):
        query = wp.mesh_query_point_no_sign(shell_mesh, point, search_radius)
        if not query.result:
            break
        signed_query = wp.mesh_query_point_sign_winding_number(mesh, point, search_radius)
        if signed_query.face >= shell_triangle_count:
            vertex_sides[i] = 0.0
            break
        closest = wp.mesh_eval_position(shell_mesh, query.face, query.u, query.v)
        offset = point - closest
        distance = wp.length(offset)
        if side * signed_query.sign * distance >= margin:
            break
        normal = wp.mesh_eval_face_normal(shell_mesh, query.face)
        if distance > 1.0e-8:
            normal = signed_query.sign * offset / distance
        point = closest + side * (margin + 1.0e-6) * normal
        point[2] = wp.max(point[2], floor)
    vertices[i] = point


@wp.kernel
def _accumulate_normals(vertices: wp.array[wp.vec3], indices: wp.array[int], normals: wp.array[wp.vec3]):
    i = wp.tid()
    a = indices[3 * i]
    b = indices[3 * i + 1]
    c = indices[3 * i + 2]
    normal = wp.cross(vertices[b] - vertices[a], vertices[c] - vertices[a])
    wp.atomic_add(normals, a, normal)
    wp.atomic_add(normals, b, normal)
    wp.atomic_add(normals, c, normal)


@wp.kernel
def _normalize_normals(normals: wp.array[wp.vec3]):
    i = wp.tid()
    normals[i] = wp.normalize(normals[i])
