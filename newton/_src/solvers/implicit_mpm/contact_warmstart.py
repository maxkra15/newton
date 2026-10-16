# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact history independent of the mutable sparse FEM topology."""

import warp as wp

from .rasterized_collisions import environment_from_offsets


@wp.func
def _node_coordinate(position: wp.vec3, inverse_spacing: float):
    scaled = inverse_spacing * position
    return wp.vec3i(int(wp.round(scaled[0])), int(wp.round(scaled[1])), int(wp.round(scaled[2])))


@wp.func
def _node_hash(coordinate: wp.vec3i, environment: int):
    # Arithmetic wraps in uint32; equality is checked against the complete key.
    value = wp.uint32(coordinate[0]) * wp.uint32(73856093)
    value = value ^ (wp.uint32(coordinate[1]) * wp.uint32(19349663))
    value = value ^ (wp.uint32(coordinate[2]) * wp.uint32(83492791))
    return value ^ (wp.uint32(environment) * wp.uint32(-1640531535))


@wp.kernel
def _save_contacts(
    positions: wp.array[wp.vec3],
    environment_offsets: wp.array[int],
    collider_ids: wp.array[int],
    normals: wp.array[wp.vec3],
    impulses: wp.array[wp.vec3],
    inverse_spacing: float,
    dt: float,
    slots: wp.array[int],
    cached_coordinates: wp.array[wp.vec3i],
    cached_environments: wp.array[int],
    cached_colliders: wp.array[int],
    cached_normals: wp.array[wp.vec3],
    cached_impulses: wp.array[wp.vec3],
    cached_dt: wp.array[float],
):
    i = wp.tid()
    if i == 0:
        cached_dt[0] = dt

    impulse = impulses[i]
    if (
        collider_ids[i] < 0
        or wp.length_sq(normals[i]) < 0.5
        or not wp.isfinite(normals[i])
        or not wp.isfinite(positions[i])
        or not wp.isfinite(impulse)
    ):
        return
    if wp.length_sq(impulse) == 0.0:
        return

    coordinate = _node_coordinate(positions[i], inverse_spacing)
    environment = int(0)
    if environment_offsets:
        environment = environment_from_offsets(i, environment_offsets)
    cached_coordinates[i] = coordinate
    cached_environments[i] = environment
    cached_colliders[i] = collider_ids[i]
    cached_normals[i] = wp.normalize(normals[i])
    cached_impulses[i] = impulse

    mask = wp.uint32(slots.shape[0] - 1)
    slot = int(_node_hash(coordinate, environment) & mask)
    # Each node has one unique key. Insertion does not read concurrently written
    # metadata; the next kernel launch provides the required memory ordering.
    for _ in range(slots.shape[0]):
        if wp.atomic_cas(slots, slot, -1, i) == -1:
            break
        slot = (slot + 1) & int(mask)


@wp.kernel
def _load_contacts(
    positions: wp.array[wp.vec3],
    environment_offsets: wp.array[int],
    collider_ids: wp.array[int],
    normals: wp.array[wp.vec3],
    inverse_spacing: float,
    dt: float,
    slots: wp.array[int],
    cached_coordinates: wp.array[wp.vec3i],
    cached_environments: wp.array[int],
    cached_colliders: wp.array[int],
    cached_normals: wp.array[wp.vec3],
    cached_impulses: wp.array[wp.vec3],
    cached_dt: wp.array[float],
    impulses: wp.array[wp.vec3],
):
    i = wp.tid()
    impulses[i] = wp.vec3(0.0)
    previous_dt = cached_dt[0]
    if (
        previous_dt <= 0.0
        or dt <= 0.0
        or not wp.isfinite(previous_dt)
        or not wp.isfinite(dt)
        or collider_ids[i] < 0
        or wp.length_sq(normals[i]) < 0.5
        or not wp.isfinite(normals[i])
        or not wp.isfinite(positions[i])
    ):
        return
    ratio = dt / previous_dt
    # A drastic timestep change is a safer cold start than an amplified impulse.
    if ratio < 0.25 or ratio > 4.0:
        return

    coordinate = _node_coordinate(positions[i], inverse_spacing)
    environment = int(0)
    if environment_offsets:
        environment = environment_from_offsets(i, environment_offsets)
    mask = wp.uint32(slots.shape[0] - 1)
    slot = int(_node_hash(coordinate, environment) & mask)
    for _ in range(slots.shape[0]):
        entry = slots[slot]
        if entry == -1:
            break
        if cached_coordinates[entry] == coordinate and cached_environments[entry] == environment:
            if (
                cached_colliders[entry] == collider_ids[i]
                and wp.dot(cached_normals[entry], wp.normalize(normals[i])) >= 0.95
            ):
                impulses[i] = ratio * cached_impulses[entry]
            break
        slot = (slot + 1) & int(mask)


@wp.kernel
def _reset_contacts(
    world_mask: wp.array[wp.bool],
    environments: wp.array[int],
    colliders: wp.array[int],
):
    i = wp.tid()
    environment = environments[i]
    if world_mask[world_mask.shape[0] - 1] or (
        environment >= 0 and environment < world_mask.shape[0] - 1 and world_mask[environment]
    ):
        colliders[i] = -1


class ContactWarmstartCache:
    """Keep the immediately previous S2 contact impulses in a bounded hash table.

    Grid nodes lie on the half-voxel lattice. Keys include the FEM environment,
    so overlapping isolated worlds never share history. Collider identity and
    normal agreement are checked separately. All allocations precede capture;
    table contents, including the previous timestep, update on every replay.
    """

    def __init__(self, node_capacity: int, *, device: wp.Device):
        slot_count = 1 << (max(2 * node_capacity, 2) - 1).bit_length()
        self.slots = wp.full(slot_count, -1, dtype=int, device=device)
        self.coordinates = wp.empty(node_capacity, dtype=wp.vec3i, device=device)
        self.environments = wp.full(node_capacity, -1, dtype=int, device=device)
        self.colliders = wp.full(node_capacity, -1, dtype=int, device=device)
        self.normals = wp.empty(node_capacity, dtype=wp.vec3, device=device)
        self.impulses = wp.empty(node_capacity, dtype=wp.vec3, device=device)
        self.dt = wp.zeros(1, dtype=float, device=device)

    def _arguments(self) -> list[wp.array]:
        return [
            self.slots,
            self.coordinates,
            self.environments,
            self.colliders,
            self.normals,
            self.impulses,
            self.dt,
        ]

    def load(
        self,
        positions: wp.array[wp.vec3],
        environment_offsets: wp.array[int] | None,
        collider_ids: wp.array[int],
        normals: wp.array[wp.vec3],
        impulses: wp.array[wp.vec3],
        *,
        voxel_size: float,
        dt: float,
    ) -> None:
        wp.launch(
            _load_contacts,
            dim=impulses.shape[0],
            inputs=[positions, environment_offsets, collider_ids, normals, 2.0 / voxel_size, dt, *self._arguments()],
            outputs=[impulses],
            device=impulses.device,
        )

    def save(
        self,
        positions: wp.array[wp.vec3],
        environment_offsets: wp.array[int] | None,
        collider_ids: wp.array[int],
        normals: wp.array[wp.vec3],
        impulses: wp.array[wp.vec3],
        *,
        voxel_size: float,
        dt: float,
    ) -> None:
        if impulses.shape[0] > self.impulses.shape[0]:
            raise RuntimeError("MPM contact warm-start node capacity was exceeded.")
        self.slots.fill_(-1)
        wp.launch(
            _save_contacts,
            dim=impulses.shape[0],
            inputs=[positions, environment_offsets, collider_ids, normals, impulses, 2.0 / voxel_size, dt],
            outputs=self._arguments(),
            device=impulses.device,
        )

    def reset(self, world_mask: wp.array[wp.bool] | None = None) -> None:
        if world_mask is None:
            self.slots.fill_(-1)
            self.dt.zero_()
        else:
            wp.launch(
                _reset_contacts,
                dim=self.colliders.shape[0],
                inputs=[world_mask, self.environments, self.colliders],
                device=self.colliders.device,
            )
