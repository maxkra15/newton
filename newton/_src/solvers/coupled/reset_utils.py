# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Utilities shared by coupled-solver reset implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    from ...sim import Model


@wp.kernel(enable_backward=False)
def _copy_selected_float(
    entity_world: wp.array[int],
    world_mask: wp.array[wp.bool],
    src: wp.array[float],
    dst: wp.array[float],
):
    entity = wp.tid()
    world = entity_world[entity]
    if world >= 0 and world_mask[world]:
        dst[entity] = src[entity]


@wp.kernel(enable_backward=False)
def _copy_selected_int(
    entity_world: wp.array[int],
    world_mask: wp.array[wp.bool],
    src: wp.array[int],
    dst: wp.array[int],
):
    entity = wp.tid()
    world = entity_world[entity]
    if world >= 0 and world_mask[world]:
        dst[entity] = src[entity]


@wp.kernel(enable_backward=False)
def _copy_selected_vec3(
    entity_world: wp.array[int],
    world_mask: wp.array[wp.bool],
    src: wp.array[wp.vec3],
    dst: wp.array[wp.vec3],
):
    entity = wp.tid()
    world = entity_world[entity]
    if world >= 0 and world_mask[world]:
        dst[entity] = src[entity]


@wp.kernel(enable_backward=False)
def _copy_selected_spatial_vector(
    entity_world: wp.array[int],
    world_mask: wp.array[wp.bool],
    src: wp.array[wp.spatial_vector],
    dst: wp.array[wp.spatial_vector],
):
    entity = wp.tid()
    world = entity_world[entity]
    if world >= 0 and world_mask[world]:
        dst[entity] = src[entity]


@wp.kernel(enable_backward=False)
def _copy_selected_transform(
    entity_world: wp.array[int],
    world_mask: wp.array[wp.bool],
    src: wp.array[wp.transform],
    dst: wp.array[wp.transform],
):
    entity = wp.tid()
    world = entity_world[entity]
    if world >= 0 and world_mask[world]:
        dst[entity] = src[entity]


@wp.kernel(enable_backward=False)
def _zero_selected_float(
    entity_world: wp.array[int],
    world_mask: wp.array[wp.bool],
    values: wp.array[float],
):
    entity = wp.tid()
    world = entity_world[entity]
    if world >= 0 and world_mask[world]:
        values[entity] = 0.0


@wp.kernel(enable_backward=False)
def _zero_selected_int(
    entity_world: wp.array[int],
    world_mask: wp.array[wp.bool],
    values: wp.array[int],
):
    entity = wp.tid()
    world = entity_world[entity]
    if world >= 0 and world_mask[world]:
        values[entity] = 0


@wp.kernel(enable_backward=False)
def _zero_selected_vec3(
    entity_world: wp.array[int],
    world_mask: wp.array[wp.bool],
    values: wp.array[wp.vec3],
):
    entity = wp.tid()
    world = entity_world[entity]
    if world >= 0 and world_mask[world]:
        values[entity] = wp.vec3()


@wp.kernel(enable_backward=False)
def _zero_selected_spatial_vector(
    entity_world: wp.array[int],
    world_mask: wp.array[wp.bool],
    values: wp.array[wp.spatial_vector],
):
    entity = wp.tid()
    world = entity_world[entity]
    if world >= 0 and world_mask[world]:
        values[entity] = wp.spatial_vector()


@wp.kernel(enable_backward=False)
def _zero_selected_transform(
    entity_world: wp.array[int],
    world_mask: wp.array[wp.bool],
    values: wp.array[wp.transform],
):
    entity = wp.tid()
    world = entity_world[entity]
    if world >= 0 and world_mask[world]:
        values[entity] = wp.transform()


@wp.kernel(enable_backward=False)
def _copy_selected_mapped_float(
    global_indices: wp.array[int],
    global_to_local: wp.array[int],
    local_entity_world: wp.array[int],
    world_mask: wp.array[wp.bool],
    src: wp.array[float],
    dst: wp.array[float],
):
    global_id = global_indices[wp.tid()]
    local_id = global_to_local[global_id]
    if local_id >= 0:
        world = local_entity_world[local_id]
        if world >= 0 and world_mask[world]:
            dst[global_id] = src[local_id]


@wp.kernel(enable_backward=False)
def _copy_selected_mapped_vec3(
    global_indices: wp.array[int],
    global_to_local: wp.array[int],
    local_entity_world: wp.array[int],
    world_mask: wp.array[wp.bool],
    src: wp.array[wp.vec3],
    dst: wp.array[wp.vec3],
):
    global_id = global_indices[wp.tid()]
    local_id = global_to_local[global_id]
    if local_id >= 0:
        world = local_entity_world[local_id]
        if world >= 0 and world_mask[world]:
            dst[global_id] = src[local_id]


@wp.kernel(enable_backward=False)
def _copy_selected_mapped_spatial_vector(
    global_indices: wp.array[int],
    global_to_local: wp.array[int],
    local_entity_world: wp.array[int],
    world_mask: wp.array[wp.bool],
    src: wp.array[wp.spatial_vector],
    dst: wp.array[wp.spatial_vector],
):
    global_id = global_indices[wp.tid()]
    local_id = global_to_local[global_id]
    if local_id >= 0:
        world = local_entity_world[local_id]
        if world >= 0 and world_mask[world]:
            dst[global_id] = src[local_id]


@wp.kernel(enable_backward=False)
def _copy_selected_mapped_transform(
    global_indices: wp.array[int],
    global_to_local: wp.array[int],
    local_entity_world: wp.array[int],
    world_mask: wp.array[wp.bool],
    src: wp.array[wp.transform],
    dst: wp.array[wp.transform],
):
    global_id = global_indices[wp.tid()]
    local_id = global_to_local[global_id]
    if local_id >= 0:
        world = local_entity_world[local_id]
        if world >= 0 and world_mask[world]:
            dst[global_id] = src[local_id]


_COPY_SELECTED_KERNELS = {
    wp.float32: _copy_selected_float,
    wp.int32: _copy_selected_int,
    wp.vec3: _copy_selected_vec3,
    wp.spatial_vector: _copy_selected_spatial_vector,
    wp.transform: _copy_selected_transform,
}

_ZERO_SELECTED_KERNELS = {
    wp.float32: _zero_selected_float,
    wp.int32: _zero_selected_int,
    wp.vec3: _zero_selected_vec3,
    wp.spatial_vector: _zero_selected_spatial_vector,
    wp.transform: _zero_selected_transform,
}

_COPY_SELECTED_MAPPED_KERNELS = {
    wp.float32: _copy_selected_mapped_float,
    wp.vec3: _copy_selected_mapped_vec3,
    wp.spatial_vector: _copy_selected_mapped_spatial_vector,
    wp.transform: _copy_selected_mapped_transform,
}


def copy_selected_rows(
    src: wp.array,
    dst: wp.array,
    entity_world: wp.array,
    world_mask: wp.array,
) -> None:
    """Copy rows whose non-negative entity world is selected by ``world_mask``."""
    if src.dtype != dst.dtype:
        raise TypeError(f"Cannot copy selected rows between {src.dtype} and {dst.dtype} arrays.")
    try:
        kernel = _COPY_SELECTED_KERNELS[src.dtype]
    except KeyError as error:
        raise TypeError(f"Unsupported selected-row dtype {src.dtype}.") from error
    wp.launch(kernel, dim=src.shape[0], inputs=[entity_world, world_mask, src, dst], device=dst.device)


def zero_selected_rows(values: wp.array, entity_world: wp.array, world_mask: wp.array) -> None:
    """Zero rows whose non-negative entity world is selected by ``world_mask``."""
    try:
        kernel = _ZERO_SELECTED_KERNELS[values.dtype]
    except KeyError as error:
        raise TypeError(f"Unsupported selected-row dtype {values.dtype}.") from error
    wp.launch(kernel, dim=values.shape[0], inputs=[entity_world, world_mask, values], device=values.device)


def copy_selected_mapped_rows(
    src: wp.array,
    dst: wp.array,
    global_indices: wp.array,
    global_to_local: wp.array,
    local_entity_world: wp.array,
    world_mask: wp.array,
) -> None:
    """Scatter selected entry-local rows into parent-global rows."""
    if src.dtype != dst.dtype:
        raise TypeError(f"Cannot copy selected rows between {src.dtype} and {dst.dtype} arrays.")
    try:
        kernel = _COPY_SELECTED_MAPPED_KERNELS[src.dtype]
    except KeyError as error:
        raise TypeError(f"Unsupported selected mapped-row dtype {src.dtype}.") from error
    wp.launch(
        kernel,
        dim=global_indices.shape[0],
        inputs=[global_indices, global_to_local, local_entity_world, world_mask, src, dst],
        device=dst.device,
    )


def validate_reset_world_mask(model: Model, world_mask: wp.array | None) -> wp.array | None:
    """Validate and return a parent-model reset world mask.

    ``None`` denotes a full reset. A partial reset mask must be a one-dimensional
    Warp boolean array on the model device with one element per model world.
    """
    if world_mask is None:
        return None
    if not isinstance(world_mask, wp.array):
        raise TypeError("'world_mask' must be a Warp array or None.")
    if world_mask.dtype != wp.bool:
        raise TypeError("'world_mask' must have dtype bool.")
    if world_mask.ndim != 1:
        raise ValueError("'world_mask' must be one-dimensional.")
    if world_mask.device != model.device:
        raise ValueError(f"'world_mask' device {world_mask.device} does not match model device {model.device}.")
    if world_mask.shape[0] != model.world_count:
        raise ValueError(
            f"'world_mask' length {world_mask.shape[0]} does not match model world_count {model.world_count}."
        )
    return world_mask
