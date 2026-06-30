# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Waterhose Insertion (proxy coupling)
#
# An RBY1 robot grasps a VBD waterhose connector, inserts it into a
# refrigerator socket, releases it, and backs away. SolverMuJoCo simulates
# the robot, SolverVBD simulates the hose, and SolverCoupledProxy transfers
# gripper contact between them.
#
# Standalone: depends only on Newton and Warp (no IsaacLab).
#
# Command: python -m newton.examples waterhose_insert
###########################################################################

from __future__ import annotations

import numpy as np


def _connector_metrics(
    tip_position: np.ndarray,
    connector_axis: np.ndarray,
    socket_position: np.ndarray,
    socket_axis: np.ndarray,
) -> tuple[float, float, float]:
    """Return connector axial depth, radial error, and socket-axis alignment."""

    socket_axis = np.asarray(socket_axis, dtype=np.float64)
    connector_axis = np.asarray(connector_axis, dtype=np.float64)
    socket_axis /= np.linalg.norm(socket_axis)
    connector_axis /= np.linalg.norm(connector_axis)
    offset = np.asarray(tip_position, dtype=np.float64) - np.asarray(socket_position, dtype=np.float64)
    depth = float(np.dot(offset, socket_axis))
    radial_error = float(np.linalg.norm(offset - depth * socket_axis))
    alignment = float(np.dot(connector_axis, socket_axis))
    return depth, radial_error, alignment
