# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compatibility helpers for particle hash-grid queries."""

import inspect

import warp as wp


def _supports_grouped_hash_grid() -> bool:
    """Return whether Warp exposes the complete grouped-grid Python API."""
    try:
        reserve_parameters = inspect.signature(wp.HashGrid.reserve).parameters
        build_parameters = inspect.signature(wp.HashGrid.build).parameters
    except (TypeError, ValueError):
        return False
    return "grouped" in reserve_parameters and "groups" in build_parameters


_HASH_GRID_GROUPING_SUPPORTED = _supports_grouped_hash_grid()


if _HASH_GRID_GROUPING_SUPPORTED:

    @wp.func
    def particle_grid_query(
        grid: wp.uint64,
        position: wp.vec3,
        radius: float,
        world: wp.int32,
        grouped: wp.bool,
    ):
        """Query one world from a grouped grid, or all points from a legacy grid."""
        if grouped:
            return wp.hash_grid_query(grid, position, radius, world)
        return wp.hash_grid_query(grid, position, radius)

else:

    @wp.func
    def particle_grid_query(
        grid: wp.uint64,
        position: wp.vec3,
        radius: float,
        world: wp.int32,
        grouped: wp.bool,
    ):
        """Query a legacy grid without referencing Warp's grouped overload."""
        return wp.hash_grid_query(grid, position, radius)
