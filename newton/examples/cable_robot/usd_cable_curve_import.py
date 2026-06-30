# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""USD BasisCurves cable importer with explicit PhysicsFixedJoint heads.

This importer parses a BasisCurves cable description and builds Newton rigid cable segments
with optional mesh heads connected by fixed joints (scheme 2):

1. Cable segments are created with ``ModelBuilder.add_rod_graph`` (capsule segments + cable joints).
2. Head parts are authored by explicit ``PhysicsFixedJoint`` prims.
3. Each head body is connected to the cable by the authored fixed-joint frame.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import warp as wp

import newton

try:
    from pxr import Usd, UsdGeom  # type: ignore
except ImportError as exc:
    raise ImportError("This importer requires USD Python bindings (`pxr`).") from exc


@dataclass
class CableCurveImportResult:
    cable_body_ids: list[int]
    cable_joint_ids: list[int]
    head_body_ids: list[int]
    head_fixed_joint_ids: list[int]
    fixed_body_ids: list[int]


@dataclass
class _AuthoredFixedJoint:
    prim_path: str
    body0_path: str
    body1_path: str
    local_pos0_m: wp.vec3
    local_pos1_m: wp.vec3
    local_rot0: wp.quat
    local_rot1: wp.quat


def _as_matrix_world(prim: Usd.Prim) -> np.ndarray:  # type: ignore[name-defined]
    xformable = UsdGeom.Xformable(prim)
    mat = np.asarray(xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=np.float64).reshape(4, 4)
    return mat


def _transform_points_rowvec(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    pts_out = pts_h @ np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    return pts_out[:, :3].astype(np.float32, copy=False)


def _fan_triangulate_faces(face_counts: np.ndarray, face_indices: np.ndarray) -> np.ndarray:
    counts = np.asarray(face_counts, dtype=np.int32).reshape(-1)
    indices = np.asarray(face_indices, dtype=np.int32).reshape(-1)
    if counts.size == 0:
        return np.zeros((0,), dtype=np.int32)
    if int(np.sum(counts)) != int(indices.shape[0]):
        raise ValueError("Invalid face data: sum(face_counts) does not match face index count.")

    tris: list[int] = []
    offset = 0
    for _n in counts:
        n = int(_n)
        if n < 3:
            offset += n
            continue
        base = int(indices[offset])
        for i in range(n - 2):
            i1 = int(indices[offset + i + 1])
            i2 = int(indices[offset + i + 2])
            tris.extend([base, i1, i2])
        offset += n
    return np.asarray(tris, dtype=np.int32)


def _find_first_mesh_prim(root_prim: Usd.Prim) -> Usd.Prim | None:  # type: ignore[name-defined]
    stack = [root_prim]
    while stack:
        prim = stack.pop()
        if prim.GetTypeName() == "Mesh":
            return prim
        children = list(prim.GetChildren())
        stack.extend(reversed(children))
    return None


def _meters_per_unit(stage: Usd.Stage) -> float:  # type: ignore[name-defined]
    if UsdGeom.StageHasAuthoredMetersPerUnit(stage):
        return float(UsdGeom.GetStageMetersPerUnit(stage))
    return 1.0


def _get_world_transform_m(prim: Usd.Prim, meters_per_unit: float) -> wp.transform:  # type: ignore[name-defined]
    mat = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=np.float32)
    pos, rot, _ = wp.transform_decompose(wp.mat44(mat.T))
    pos_m = wp.vec3(
        float(pos[0]) * meters_per_unit,
        float(pos[1]) * meters_per_unit,
        float(pos[2]) * meters_per_unit,
    )
    return wp.transform(pos_m, wp.normalize(rot))


def _get_single_relationship_target(prim: Usd.Prim, rel_name: str) -> str:  # type: ignore[name-defined]
    rel = prim.GetRelationship(rel_name)
    if not rel:
        raise ValueError(f"Joint '{prim.GetPath()}' is missing relationship '{rel_name}'.")
    targets = rel.GetTargets()
    if len(targets) != 1:
        raise ValueError(
            f"Joint '{prim.GetPath()}' relationship '{rel_name}' must contain exactly one target, got {len(targets)}."
        )
    return str(targets[0])


def _get_required_vec3_attr_m(prim: Usd.Prim, attr_name: str, meters_per_unit: float) -> wp.vec3:  # type: ignore[name-defined]
    attr = prim.GetAttribute(attr_name)
    if not attr or not attr.HasAuthoredValue():
        raise ValueError(f"Joint '{prim.GetPath()}' is missing required attribute '{attr_name}'.")
    val = attr.Get()
    vec = np.asarray(val, dtype=np.float64).reshape(3)
    if not np.isfinite(vec).all():
        raise ValueError(f"Joint '{prim.GetPath()}' attribute '{attr_name}' contains non-finite values: {val}.")
    return wp.vec3(
        float(vec[0]) * meters_per_unit,
        float(vec[1]) * meters_per_unit,
        float(vec[2]) * meters_per_unit,
    )


def _get_required_quat_attr(prim: Usd.Prim, attr_name: str) -> wp.quat:  # type: ignore[name-defined]
    attr = prim.GetAttribute(attr_name)
    if not attr or not attr.HasAuthoredValue():
        raise ValueError(f"Joint '{prim.GetPath()}' is missing required attribute '{attr_name}'.")
    val = attr.Get()
    q = wp.quat(float(val.imaginary[0]), float(val.imaginary[1]), float(val.imaginary[2]), float(val.real))
    qn = wp.normalize(q)
    if float(wp.length(qn)) <= 0.0:
        raise ValueError(f"Joint '{prim.GetPath()}' attribute '{attr_name}' is invalid quaternion: {val}.")
    return qn


def _collect_authored_fixed_joints_for_curve(
    stage: Usd.Stage,  # type: ignore[name-defined]
    curve_parent_path: str,
    meters_per_unit: float,
) -> list[_AuthoredFixedJoint]:
    joints: list[_AuthoredFixedJoint] = []
    for prim in stage.Traverse():
        if prim.GetTypeName() != "PhysicsFixedJoint":
            continue

        body0_path = _get_single_relationship_target(prim, "physics:body0")
        if body0_path != curve_parent_path:
            continue

        body1_path = _get_single_relationship_target(prim, "physics:body1")
        local_pos0_m = _get_required_vec3_attr_m(prim, "physics:localPos0", meters_per_unit)
        local_pos1_m = _get_required_vec3_attr_m(prim, "physics:localPos1", meters_per_unit)
        local_rot0 = _get_required_quat_attr(prim, "physics:localRot0")
        local_rot1 = _get_required_quat_attr(prim, "physics:localRot1")

        joints.append(
            _AuthoredFixedJoint(
                prim_path=str(prim.GetPath()),
                body0_path=body0_path,
                body1_path=body1_path,
                local_pos0_m=local_pos0_m,
                local_pos1_m=local_pos1_m,
                local_rot0=local_rot0,
                local_rot1=local_rot1,
            )
        )
    return joints


def _load_mesh_from_body_prim(
    stage: Usd.Stage,  # type: ignore[name-defined]
    body_prim_path: str,
    meters_per_unit: float,
) -> tuple[newton.Mesh, Usd.Prim]:  # type: ignore[name-defined]
    body_prim = stage.GetPrimAtPath(body_prim_path)
    if not body_prim or not body_prim.IsValid():
        raise ValueError(f"Invalid body prim path '{body_prim_path}' for fixed-joint head attachment.")

    mesh_prim = _find_first_mesh_prim(body_prim)
    if mesh_prim is None:
        raise ValueError(f"No Mesh prim found under body prim '{body_prim_path}'.")

    mesh = UsdGeom.Mesh(mesh_prim)
    points_raw = mesh.GetPointsAttr().Get()
    face_indices_raw = mesh.GetFaceVertexIndicesAttr().Get()
    face_counts_raw = mesh.GetFaceVertexCountsAttr().Get()
    if points_raw is None or face_indices_raw is None or face_counts_raw is None:
        raise ValueError(f"Mesh '{mesh_prim.GetPath()}' is missing points/faces data.")

    body_world = _as_matrix_world(body_prim)
    world_body = np.linalg.inv(body_world)
    mesh_world = _as_matrix_world(mesh_prim)
    mesh_body = mesh_world @ world_body

    points_local = _transform_points_rowvec(np.asarray(points_raw, dtype=np.float32), mesh_body)
    tri_indices = _fan_triangulate_faces(np.asarray(face_counts_raw), np.asarray(face_indices_raw))
    if tri_indices.shape[0] == 0:
        raise ValueError(f"Mesh '{mesh_prim.GetPath()}' has no triangulated faces.")

    points_local_m = (points_local * meters_per_unit).astype(np.float32, copy=False)
    return newton.Mesh(points_local_m, tri_indices), body_prim


def _match_anchor_to_edge(
    anchor_world_m: np.ndarray,
    points_m: np.ndarray,
    edges: list[tuple[int, int]],
) -> tuple[int, float, float]:
    best_edge_idx = -1
    best_t = 0.0
    best_dist = float("inf")

    for e_idx, (u, v) in enumerate(edges):
        p0 = points_m[u]
        p1 = points_m[v]
        d = p1 - p0
        d2 = float(np.dot(d, d))
        if d2 <= 0.0:
            raise ValueError(f"Edge ({u}, {v}) has zero length while matching fixed-joint anchor.")

        t = float(np.dot(anchor_world_m - p0, d) / d2)
        if t < 0.0:
            t = 0.0
        if t > 1.0:
            t = 1.0
        proj = p0 + t * d
        dist = float(np.linalg.norm(anchor_world_m - proj))
        if dist < best_dist:
            best_dist = dist
            best_t = t
            best_edge_idx = e_idx

    if best_edge_idx < 0:
        raise RuntimeError("Internal error: failed to match fixed-joint anchor to any cable edge.")
    return best_edge_idx, best_t, best_dist


def _quat_from_segment_direction(direction: np.ndarray) -> wp.quat:
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    n = np.linalg.norm(d)
    if n <= 0.0:
        raise ValueError("Cannot build orientation from zero-length direction vector.")
    d = d / n
    q = wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), wp.vec3(float(d[0]), float(d[1]), float(d[2])))
    return wp.normalize(q)


def _ordered_polyline_indices(
    point_count: int,
    edges: list[tuple[int, int]],
    curve_prim_path: str,
) -> list[int]:
    if point_count < 2:
        raise ValueError(f"BasisCurves '{curve_prim_path}' must contain at least 2 points.")
    if len(edges) != point_count - 1:
        raise ValueError(
            f"BasisCurves '{curve_prim_path}' cannot be resampled because it is not a single polyline: "
            f"points={point_count}, edges={len(edges)}."
        )

    adjacency: list[list[int]] = [[] for _ in range(point_count)]
    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)

    endpoints = [idx for idx, neighbors in enumerate(adjacency) if len(neighbors) == 1]
    if len(endpoints) != 2 or any(len(neighbors) > 2 for neighbors in adjacency):
        raise ValueError(f"BasisCurves '{curve_prim_path}' cannot be resampled because it is not a simple chain.")

    start = 0 if 0 in endpoints else endpoints[0]
    order = [start]
    previous = -1
    current = start
    while len(order) < point_count:
        candidates = [idx for idx in adjacency[current] if idx != previous]
        if len(candidates) != 1:
            raise ValueError(f"BasisCurves '{curve_prim_path}' cannot be resampled because the chain is ambiguous.")
        previous, current = current, candidates[0]
        order.append(current)

    return order


def _resample_polyline(
    points_m: np.ndarray,
    edges: list[tuple[int, int]],
    fixed_point_indices: list[int],
    resample_segments: int | None,
    curve_prim_path: str,
) -> tuple[np.ndarray, list[tuple[int, int]], list[int]]:
    if resample_segments is None or int(resample_segments) <= 0:
        return points_m, edges, fixed_point_indices

    segment_count = int(resample_segments)
    if segment_count < 1:
        raise ValueError("resample_segments must be positive when provided.")

    order = _ordered_polyline_indices(points_m.shape[0], edges, curve_prim_path)
    ordered_points = points_m[order]
    segment_lengths = np.linalg.norm(ordered_points[1:] - ordered_points[:-1], axis=1)
    if np.any(segment_lengths <= 0.0):
        raise ValueError(f"BasisCurves '{curve_prim_path}' has zero-length polyline segments.")

    arc_lengths = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = float(arc_lengths[-1])
    target_arc_lengths = np.linspace(0.0, total_length, segment_count + 1)
    resampled_points = np.empty((segment_count + 1, 3), dtype=np.float64)

    source_segment = 0
    for target_index, target_arc in enumerate(target_arc_lengths):
        while source_segment + 1 < arc_lengths.shape[0] and arc_lengths[source_segment + 1] < target_arc:
            source_segment += 1
        if source_segment + 1 >= arc_lengths.shape[0]:
            resampled_points[target_index] = ordered_points[-1]
        else:
            start_arc = arc_lengths[source_segment]
            end_arc = arc_lengths[source_segment + 1]
            alpha = 0.0 if end_arc <= start_arc else (target_arc - start_arc) / (end_arc - start_arc)
            resampled_points[target_index] = (1.0 - alpha) * ordered_points[source_segment] + alpha * ordered_points[
                source_segment + 1
            ]

    original_arc_by_index = {point_index: arc_lengths[ordered_index] for ordered_index, point_index in enumerate(order)}
    fixed_resampled_indices = []
    for point_index in fixed_point_indices:
        original_arc = original_arc_by_index.get(point_index)
        if original_arc is None:
            continue
        resampled_index = int(round((original_arc / total_length) * segment_count))
        fixed_resampled_indices.append(max(0, min(segment_count, resampled_index)))

    resampled_edges = [(idx, idx + 1) for idx in range(segment_count)]
    return resampled_points, resampled_edges, sorted(set(fixed_resampled_indices))


def add_cable_from_usd_curve(
    builder: newton.ModelBuilder,
    source_usd_path: str,
    curve_prim_path: str = "/World/cable/curve_0",
    *,
    cable_label: str | None = None,
    cable_cfg: newton.ShapeConfig | None = None,
    stretch_stiffness: float = 1.0e9,
    stretch_damping: float = 0.0,
    bend_stiffness: float = 0.0,
    bend_damping: float = 0.0,
    wrap_in_articulation: bool = True,
    head_shape_mode: str = "mesh",
    head_cfg: newton.ShapeConfig | None = None,
    head_mass: float = 0.0,
    resample_segments: int | None = None,
) -> CableCurveImportResult:
    """Import a BasisCurves cable with optional rigid mesh heads.

    Args:
        builder: Target model builder to mutate in-place.
        source_usd_path: USD file containing the cable BasisCurves prim.
        curve_prim_path: Prim path to the BasisCurves object.
        cable_label: Label prefix for cable bodies/joints.
        cable_cfg: Shape config for cable capsule segments.
        stretch_stiffness: Cable stretch stiffness [N/m].
        stretch_damping: Cable stretch damping.
        bend_stiffness: Cable bend stiffness [N*m].
        bend_damping: Cable bend damping.
        wrap_in_articulation: Whether to wrap cable joints in articulation(s).
        head_shape_mode: Either ``"mesh"`` or ``"convex_hull"`` for head shapes.
        head_cfg: Shape config for head meshes.
        head_mass: Initial mass [kg] for each head body before shape mass contribution.
        resample_segments: Optional number of uniform cable segments to build from
            the authored curve. ``None`` or non-positive values keep authored
            points and connections.

    Returns:
        Imported cable/head body and joint indices.
    """
    if head_shape_mode not in ("mesh", "convex_hull"):
        raise ValueError(f"Unsupported head_shape_mode '{head_shape_mode}'. Expected 'mesh' or 'convex_hull'.")

    stage = Usd.Stage.Open(source_usd_path)
    if stage is None:
        raise RuntimeError(f"Failed to open curve USD stage: '{source_usd_path}'.")

    curve_prim = stage.GetPrimAtPath(curve_prim_path)
    if not curve_prim or not curve_prim.IsValid():
        raise ValueError(f"Curve prim '{curve_prim_path}' is not valid in stage '{source_usd_path}'.")
    if curve_prim.GetTypeName() != "BasisCurves":
        raise ValueError(f"Prim '{curve_prim_path}' is type '{curve_prim.GetTypeName()}', expected 'BasisCurves'.")

    points_attr = curve_prim.GetAttribute("points")
    points_raw = points_attr.Get() if points_attr else None
    if points_raw is None:
        raise ValueError(f"BasisCurves '{curve_prim_path}' is missing 'points'.")

    widths_attr = curve_prim.GetAttribute("widths")
    widths_raw = widths_attr.Get() if widths_attr else None
    if widths_raw is None:
        raise ValueError(f"BasisCurves '{curve_prim_path}' is missing 'widths'.")

    connections_attr = curve_prim.GetAttribute("connections")
    connections_raw = connections_attr.Get() if connections_attr else None
    fixed_points_attr = curve_prim.GetAttribute("fixed_points")
    fixed_points_raw = fixed_points_attr.Get() if fixed_points_attr else None

    points = np.asarray(points_raw, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] < 2:
        raise ValueError(f"BasisCurves '{curve_prim_path}' must contain at least 2 points.")

    mpu = _meters_per_unit(stage)
    points_m = points * float(mpu)

    widths = np.asarray(widths_raw, dtype=np.float64).reshape(-1)
    if widths.shape[0] == 0:
        raise ValueError(f"BasisCurves '{curve_prim_path}' has empty 'widths'.")
    if float(np.min(widths)) <= 0.0:
        raise ValueError(f"BasisCurves '{curve_prim_path}' has non-positive width values.")
    radius_m = float(widths[0]) * 0.5 * float(mpu)

    edges: list[tuple[int, int]] = []
    if connections_raw is None:
        for i in range(points_m.shape[0] - 1):
            edges.append((i, i + 1))
    else:
        for pair in connections_raw:
            u = int(pair[0])
            v = int(pair[1])
            if u < 0 or u >= points_m.shape[0] or v < 0 or v >= points_m.shape[0]:
                raise ValueError(f"Connection ({u}, {v}) is out of point range [0, {points_m.shape[0] - 1}].")
            if u == v:
                raise ValueError(f"Connection ({u}, {v}) is invalid (self-edge).")
            edges.append((u, v))
    if len(edges) == 0:
        raise ValueError(f"BasisCurves '{curve_prim_path}' has no valid connections.")

    fixed_point_indices: list[int] = []
    if fixed_points_raw is not None:
        fixed_points = np.asarray(fixed_points_raw, dtype=np.int64).reshape(-1)
        for idx_raw in fixed_points:
            idx = int(idx_raw)
            if idx < 0 or idx >= points_m.shape[0]:
                raise ValueError(
                    f"Fixed point index {idx} in '{curve_prim_path}' is out of point range [0, {points_m.shape[0] - 1}]."
                )
            fixed_point_indices.append(idx)

    points_m, edges, fixed_point_indices = _resample_polyline(
        points_m,
        edges,
        fixed_point_indices,
        resample_segments,
        curve_prim_path,
    )

    node_positions_wp = [wp.vec3(float(p[0]), float(p[1]), float(p[2])) for p in points_m]
    rod_cfg = builder.default_shape_cfg if cable_cfg is None else cable_cfg
    cable_body_ids, cable_joint_ids = builder.add_rod_graph(
        node_positions=node_positions_wp,
        edges=edges,
        radius=radius_m,
        cfg=rod_cfg,
        stretch_stiffness=stretch_stiffness,
        stretch_damping=stretch_damping,
        bend_stiffness=bend_stiffness,
        bend_damping=bend_damping,
        label=cable_label,
        wrap_in_articulation=wrap_in_articulation,
    )

    fixed_body_ids: list[int] = []
    if len(fixed_point_indices) > 0:
        fixed_points_set = set(fixed_point_indices)
        fixed_body_seen: set[int] = set()
        for edge_idx, (u, v) in enumerate(edges):
            if u in fixed_points_set or v in fixed_points_set:
                body_id = int(cable_body_ids[edge_idx])
                if body_id not in fixed_body_seen:
                    fixed_body_seen.add(body_id)
                    fixed_body_ids.append(body_id)

    edge_lengths: list[float] = []
    for u, v in edges:
        seg = points_m[v] - points_m[u]
        length = float(np.linalg.norm(seg))
        if length <= 0.0:
            raise ValueError(f"Edge ({u}, {v}) has zero length.")
        edge_lengths.append(length)

    head_body_ids: list[int] = []
    head_fixed_joint_ids: list[int] = []

    curve_parent_path = str(curve_prim.GetParent().GetPath())
    authored_fixed_joints = _collect_authored_fixed_joints_for_curve(stage, curve_parent_path, float(mpu))

    head_models_attr = curve_prim.GetAttribute("cable_head_models")
    if head_models_attr and head_models_attr.HasAuthoredValue():
        raise ValueError(
            "Legacy head schema 'cable_head_models/cable_head_ranges_i' is no longer supported. "
            "Please author explicit PhysicsFixedJoint entries with physics:body0/body1/localPos/localRot."
        )

    head_shape_cfg = builder.default_shape_cfg if head_cfg is None else head_cfg
    curve_base_label = cable_label if cable_label is not None else str(curve_prim.GetPath())

    attach_tolerance_m = max(2.0 * radius_m, 1.0e-4)
    body1_to_head_body: dict[str, int] = {}

    for authored_idx, authored in enumerate(authored_fixed_joints):
        head_body = body1_to_head_body.get(authored.body1_path)
        if head_body is None:
            head_mesh, body1_prim = _load_mesh_from_body_prim(stage, authored.body1_path, float(mpu))
            head_xform = _get_world_transform_m(body1_prim, float(mpu))
            head_label = f"{curve_base_label}:authored_head_{authored_idx}:{authored.body1_path}"
            head_body = builder.add_link(xform=head_xform, mass=head_mass, label=head_label)
            head_body_ids.append(head_body)
            body1_to_head_body[authored.body1_path] = head_body

            head_shape_label = f"{head_label}:shape"
            if head_shape_mode == "mesh":
                builder.add_shape_mesh(
                    body=head_body,
                    xform=wp.transform(),
                    mesh=head_mesh,
                    scale=wp.vec3(1.0, 1.0, 1.0),
                    cfg=head_shape_cfg,
                    label=head_shape_label,
                )
            else:
                builder.add_shape_convex_hull(
                    body=head_body,
                    xform=wp.transform(),
                    mesh=head_mesh,
                    scale=wp.vec3(1.0, 1.0, 1.0),
                    cfg=head_shape_cfg,
                    label=head_shape_label,
                )

        body0_prim = stage.GetPrimAtPath(authored.body0_path)
        if not body0_prim or not body0_prim.IsValid():
            raise ValueError(
                f"Fixed joint '{authored.prim_path}' references invalid body0 prim '{authored.body0_path}'."
            )
        body0_world = _get_world_transform_m(body0_prim, float(mpu))
        anchor_world_p = wp.transform_point(body0_world, authored.local_pos0_m)
        anchor_world_q = wp.normalize(wp.transform_get_rotation(body0_world) * authored.local_rot0)
        anchor_world_np = np.asarray(
            [float(anchor_world_p[0]), float(anchor_world_p[1]), float(anchor_world_p[2])], dtype=np.float64
        )

        matched_edge_idx, edge_t, edge_dist = _match_anchor_to_edge(anchor_world_np, points_m, edges)
        if edge_dist > attach_tolerance_m:
            raise ValueError(
                f"Fixed joint '{authored.prim_path}' anchor is too far from cable edges: "
                f"distance={edge_dist:.6e} m, tolerance={attach_tolerance_m:.6e} m."
            )

        edge_u, edge_v = edges[matched_edge_idx]
        seg_vec = points_m[edge_v] - points_m[edge_u]
        seg_q = _quat_from_segment_direction(seg_vec)
        parent_q = wp.normalize(wp.quat_inverse(seg_q) * anchor_world_q)
        parent_z = float(edge_t) * float(edge_lengths[matched_edge_idx])
        parent_xform = wp.transform(wp.vec3(0.0, 0.0, parent_z), parent_q)
        child_xform = wp.transform(authored.local_pos1_m, authored.local_rot1)

        fixed_label = f"{curve_base_label}:fixed:{authored.prim_path}"
        fixed_joint = builder.add_joint_fixed(
            parent=int(cable_body_ids[matched_edge_idx]),
            child=head_body,
            parent_xform=parent_xform,
            child_xform=child_xform,
            label=fixed_label,
            collision_filter_parent=True,
            enabled=True,
        )
        head_fixed_joint_ids.append(fixed_joint)

    return CableCurveImportResult(
        cable_body_ids=[int(v) for v in cable_body_ids],
        cable_joint_ids=[int(v) for v in cable_joint_ids],
        head_body_ids=head_body_ids,
        head_fixed_joint_ids=head_fixed_joint_ids,
        fixed_body_ids=fixed_body_ids,
    )


def _default_curve_usd_path() -> str:
    return os.path.realpath(
        os.path.join(os.path.dirname(__file__), "assets", "version3", "SRA_curve", "cable_SRA_curve02.usda")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse BasisCurves cable USD and build Newton cable + rigid heads.")
    parser.add_argument("--usd-path", type=str, default=_default_curve_usd_path())
    parser.add_argument("--curve-prim-path", type=str, default="/World/cable/curve_0")
    parser.add_argument("--head-shape-mode", choices=["mesh", "convex_hull"], default="mesh")
    parser.add_argument("--wrap-in-articulation", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    builder = newton.ModelBuilder()
    result = add_cable_from_usd_curve(
        builder,
        source_usd_path=args.usd_path,
        curve_prim_path=args.curve_prim_path,
        cable_label="curve_cable",
        wrap_in_articulation=bool(args.wrap_in_articulation),
        head_shape_mode=args.head_shape_mode,
    )

    print(
        "[usd_curve_import] summary: "
        f"usd={args.usd_path} curve={args.curve_prim_path} "
        f"cable_bodies={len(result.cable_body_ids)} "
        f"cable_joints={len(result.cable_joint_ids)} "
        f"head_bodies={len(result.head_body_ids)} "
        f"head_fixed_joints={len(result.head_fixed_joint_ids)} "
        f"fixed_bodies={len(result.fixed_body_ids)}"
    )
