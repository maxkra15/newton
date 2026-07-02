# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton.examples.vbd import example_vbd_tablecloth as tablecloth


class _SliderUI:
    def __init__(self, values):
        self.values = iter(values)
        self.labels = []

    def text(self, _text):
        pass

    def slider_float(self, label, _value, _minimum, _maximum):
        self.labels.append(label)
        return True, next(self.values)


class TestVBDTablecloth(unittest.TestCase):
    def test_tableware_densities_are_material_values(self):
        self.assertEqual(tablecloth.PARAMS["plate_density"], 2400.0)
        self.assertEqual(tablecloth.PARAMS["glass_density"], 2500.0)
        self.assertEqual(tablecloth.PARAMS["fork_density"], 8000.0)

    def test_pull_distances_accumulate_across_speed_changes(self):
        self.assertTrue(
            hasattr(tablecloth, "advance_pull_distances"),
            "tablecloth example must provide advance_pull_distances",
        )

        speeds = wp.array([1.0, 2.0], dtype=float, device="cpu")
        distances = wp.zeros(2, dtype=float, device="cpu")

        wp.launch(
            tablecloth.advance_pull_distances,
            dim=2,
            inputs=[speeds, 0.25, 1.0],
            outputs=[distances],
            device="cpu",
        )
        np.testing.assert_allclose(distances.numpy(), [0.25, 0.50])

        speeds.assign(np.array([0.0, 1.0], dtype=np.float32))
        wp.launch(
            tablecloth.advance_pull_distances,
            dim=2,
            inputs=[speeds, 0.25, 1.0],
            outputs=[distances],
            device="cpu",
        )
        np.testing.assert_allclose(distances.numpy(), [0.25, 0.75])

        speeds.assign(np.array([10.0, 10.0], dtype=np.float32))
        wp.launch(
            tablecloth.advance_pull_distances,
            dim=2,
            inputs=[speeds, 1.0, 1.0],
            outputs=[distances],
            device="cpu",
        )
        np.testing.assert_allclose(distances.numpy(), [1.0, 1.0])

    def test_pulled_edge_descends_after_table_clearance(self):
        particle_indices = wp.array([0], dtype=wp.int32, device="cpu")
        lane_indices = wp.array([0], dtype=wp.int32, device="cpu")
        rest_positions = wp.array([[0.0, 0.0, 1.0]], dtype=wp.vec3, device="cpu")
        speeds = wp.array([2.0], dtype=float, device="cpu")
        distances = wp.array([0.60], dtype=float, device="cpu")
        particle_q_0 = wp.zeros(1, dtype=wp.vec3, device="cpu")
        particle_q_1 = wp.zeros(1, dtype=wp.vec3, device="cpu")
        particle_qd_0 = wp.zeros(1, dtype=wp.vec3, device="cpu")
        particle_qd_1 = wp.zeros(1, dtype=wp.vec3, device="cpu")

        wp.launch(
            tablecloth.move_pulled_edges,
            dim=1,
            inputs=[
                particle_indices,
                lane_indices,
                rest_positions,
                speeds,
                distances,
                1.25,
                0.10,
                0.30,
                1,
            ],
            outputs=[particle_q_0, particle_q_1, particle_qd_0, particle_qd_1],
            device="cpu",
        )

        np.testing.assert_allclose(particle_q_0.numpy(), [[0.60, 0.0, 1.0 - 0.30 * 0.50 / 1.15]], rtol=1.0e-6)
        np.testing.assert_allclose(particle_qd_0.numpy(), [[2.0, 0.0, -2.0 * 0.30 / 1.15]], rtol=1.0e-6)

    def test_gui_updates_each_table_pull_speed(self):
        example = tablecloth.Example.__new__(tablecloth.Example)
        example.pull_speeds = list(tablecloth.PARAMS["pull_speeds"])
        example.pull_speeds_device = wp.array(example.pull_speeds, dtype=float, device="cpu")
        ui = _SliderUI([0.5, 1.0, 2.0, 3.0, 5.5])

        example.gui(ui)

        self.assertEqual(ui.labels, [f"Table {i + 1}" for i in range(5)])
        np.testing.assert_allclose(example.pull_speeds, [0.5, 1.0, 2.0, 3.0, 5.5])
        np.testing.assert_allclose(example.pull_speeds_device.numpy(), example.pull_speeds)


if __name__ == "__main__":
    unittest.main()
