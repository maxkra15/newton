# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import newton
from newton.examples.multiphysics import example_waterhose_insert as waterhose
from newton.examples.multiphysics.example_waterhose_insert import _connector_metrics
from newton.solvers import SolverMuJoCo, SolverVBD


class TestWaterhoseGeometry(unittest.TestCase):
    def test_robot_keeps_authored_collision_links(self):
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        SolverVBD.register_custom_attributes(builder, dahl_defaults_enabled=False)

        example = waterhose.Example.__new__(waterhose.Example)
        _, _, robot_shapes, finger_bodies = example._add_robot(builder, load_visual_shapes=False)
        collision_bit = int(newton.ShapeFlags.COLLIDE_SHAPES)
        collision_bodies = {
            builder.shape_body[shape] for shape in robot_shapes if builder.shape_flags[shape] & collision_bit
        }
        collision_labels = {str(builder.body_label[body]).rsplit("/", 1)[-1] for body in collision_bodies}

        self.assertEqual(len(finger_bodies), 2)
        self.assertGreater(len(collision_bodies), len(finger_bodies))
        self.assertIn("right_arm_wrist_pitch", collision_labels)
        self.assertIn("right_gripper_base", collision_labels)

    def test_reference_scene_is_rebased_onto_zero_ground(self):
        self.assertEqual(waterhose.GROUND_HEIGHT, 0.0)
        self.assertEqual(getattr(waterhose, "SCENE_VERTICAL_OFFSET", None), 1.05)
        self.assertAlmostEqual(float(waterhose.FRIDGE_POSITION[2]), 1.55)
        robot_position = waterhose.wp.transform_get_translation(waterhose.ROBOT_TRANSFORM)
        self.assertAlmostEqual(float(robot_position[2]), 0.05)
        self.assertAlmostEqual(float(waterhose.SOCKET_POSITION[2]), 1.33698)

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
