# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from newton.examples.multiphysics.example_waterhose_insert import _connector_metrics


class TestWaterhoseGeometry(unittest.TestCase):
    def test_seated_connector_metrics(self):
        depth, radial_error, alignment = _connector_metrics(
            tip_position=np.array([0.0, 0.0, 0.002]),
            connector_axis=np.array([0.0, 0.0, 1.0]),
            socket_position=np.zeros(3),
            socket_axis=np.array([0.0, 0.0, 1.0]),
        )

        self.assertAlmostEqual(depth, 0.002)
        self.assertAlmostEqual(radial_error, 0.0)
        self.assertAlmostEqual(alignment, 1.0)

    def test_radial_error_is_orthogonal_to_socket_axis(self):
        depth, radial_error, _ = _connector_metrics(
            tip_position=np.array([0.012, -0.016, 0.004]),
            connector_axis=np.array([0.0, 0.0, 1.0]),
            socket_position=np.zeros(3),
            socket_axis=np.array([0.0, 0.0, 2.0]),
        )

        self.assertAlmostEqual(depth, 0.004)
        self.assertAlmostEqual(radial_error, 0.020)

    def test_reversed_connector_axis_is_misaligned(self):
        _, _, alignment = _connector_metrics(
            tip_position=np.zeros(3),
            connector_axis=np.array([0.0, 0.0, -2.0]),
            socket_position=np.zeros(3),
            socket_axis=np.array([0.0, 0.0, 1.0]),
        )

        self.assertAlmostEqual(alignment, -1.0)


if __name__ == "__main__":
    unittest.main()
