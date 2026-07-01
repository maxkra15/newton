# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import newton
from newton.examples.multiphysics import example_waterhose_insert as waterhose
from newton.examples.multiphysics.example_waterhose_insert import _connector_metrics
from newton.solvers import SolverMuJoCo, SolverVBD


class TestWaterhoseGeometry(unittest.TestCase):
    def test_lift_phase_precedes_carry(self):
        self.assertEqual(getattr(waterhose, "LIFT", None), 7)
        self.assertEqual(waterhose.Example.Phase.LIFT + 1, waterhose.Example.Phase.CARRY)
        self.assertEqual(waterhose.DONE, 14)

    def test_lift_target_preserves_xy_and_reaches_preinsert_height(self):
        phase_count = waterhose.DONE + 1
        start = waterhose.wp.transform(waterhose.wp.vec3(1.0, 2.0, 3.0), waterhose.wp.quat_identity())
        identity = waterhose.wp.transform()

        phase = waterhose.wp.array([waterhose.LIFT], dtype=int, device="cpu")
        target_position = waterhose.wp.zeros(1, dtype=waterhose.wp.vec3, device="cpu")
        target_rotation = waterhose.wp.zeros(1, dtype=waterhose.wp.vec4, device="cpu")
        gripper_blend = waterhose.wp.zeros(1, dtype=float, device="cpu")

        waterhose.wp.launch(
            waterhose._update_state_machine,
            dim=1,
            inputs=[
                waterhose.wp.array([start, start], dtype=waterhose.wp.transform, device="cpu"),
                phase,
                waterhose.wp.array([2.0], dtype=float, device="cpu"),
                waterhose.wp.ones(phase_count, dtype=float, device="cpu"),
                waterhose.wp.array([start], dtype=waterhose.wp.transform, device="cpu"),
                waterhose.wp.array([start], dtype=waterhose.wp.transform, device="cpu"),
                waterhose.wp.zeros(1, dtype=waterhose.wp.vec3, device="cpu"),
                waterhose.wp.zeros(phase_count, dtype=waterhose.wp.vec3, device="cpu"),
                waterhose.wp.zeros(phase_count, dtype=waterhose.wp.vec3, device="cpu"),
                waterhose.wp.zeros(phase_count, dtype=waterhose.wp.vec3, device="cpu"),
                waterhose.wp.zeros(phase_count, dtype=waterhose.wp.vec3, device="cpu"),
                waterhose.wp.zeros(phase_count, dtype=float, device="cpu"),
                waterhose.wp.zeros(phase_count, dtype=int, device="cpu"),
                waterhose.wp.zeros(phase_count, dtype=waterhose.wp.vec3, device="cpu"),
                waterhose.wp.zeros(phase_count, dtype=waterhose.wp.vec3, device="cpu"),
                waterhose.wp.zeros(phase_count, dtype=waterhose.wp.vec3, device="cpu"),
                waterhose.wp.zeros(phase_count, dtype=waterhose.wp.vec3, device="cpu"),
                0,
                1,
                identity,
                identity,
                waterhose.wp.vec3(4.0, 5.0, 6.0),
                waterhose.wp.quat_identity(),
                waterhose.wp.vec3(),
                waterhose.CONNECTOR_TIP_LENGTH,
                waterhose.wp.quat_identity(),
                waterhose.wp.vec3(),
                0.01,
                target_position,
                target_rotation,
                gripper_blend,
            ],
            device="cpu",
        )

        expected_z = 6.0 - 0.018 - waterhose.CONNECTOR_TIP_LENGTH
        np.testing.assert_allclose(target_position.numpy()[0], [1.0, 2.0, expected_z], atol=1.0e-6)
        self.assertEqual(int(phase.numpy()[0]), waterhose.CARRY)

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
