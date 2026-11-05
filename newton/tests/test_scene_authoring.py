# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Physical and transactional contracts of the opt-in authoring scene."""

import importlib.util
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from newton.viewer import SceneEditor, ViewerNull


class TestSceneAuthoring(unittest.TestCase):
    def make_editor(self):
        """Create a real CPU scene without a graphical or network dependency."""
        editor = SceneEditor(ViewerNull(), device="cpu")
        self.addCleanup(editor.close)
        return editor

    def test_fast_edits_preserve_storage_and_gesture_undo(self):
        """Apply pose and appearance edits without recompiling or replacing arrays."""
        editor = self.make_editor()
        object_id = editor.add_primitive("box", position=(0, 0, 2))
        editor.commit()
        model, state, poses = editor.model, editor.state_0, editor.state_0.body_q
        editor.begin_edit()
        for index in range(8):
            editor.update_object(object_id, position=(index * 0.1, 0, 2), color=(0.2, 0.4, 0.6), friction=0.8)
            editor.before_frame()
        editor.end_edit()
        self.assertIs(editor.model, model)
        self.assertIs(editor.state_0, state)
        self.assertIs(editor.state_0.body_q, poses)
        self.assertEqual(editor.rebuild_count, 1)
        np.testing.assert_allclose(editor.get_pose(object_id)[0], (0.7, 0, 2), atol=1e-6)
        np.testing.assert_allclose(editor.model.shape_color.numpy()[1], (0.2, 0.4, 0.6))
        self.assertAlmostEqual(editor.model.shape_material_mu.numpy()[1], 0.8, places=6)
        editor.undo()
        editor.before_frame()
        np.testing.assert_allclose(editor.get_pose(object_id)[0], (0, 0, 2))
        editor.redo()
        editor.before_frame()
        np.testing.assert_allclose(editor.get_pose(object_id)[0], (0.7, 0, 2), atol=1e-6)
        editor.reset_scene()
        np.testing.assert_allclose(editor.get_pose(object_id)[0], (0.7, 0, 2), atol=1e-6)
        self.assertEqual(editor.rebuild_count, 1)

    def test_batched_topology_preserves_unedited_physical_state(self):
        """Add and remove objects once while retaining an existing falling body's state."""
        editor = self.make_editor()
        removed = editor.add_primitive("box", position=(2, 0, 3))
        first = editor.add_primitive("sphere", position=(0, 0, 3), size=(0.5, 0.5, 0.5))
        editor.commit()
        for _ in range(4):
            editor.step()
        index = editor.body_indices(first)[0]
        before_q = editor.state_0.body_q.numpy()[index].copy()
        before_qd = editor.state_0.body_qd.numpy()[index].copy()
        editor.control.joint_f[6:7].fill_(0.7)
        editor.begin_edit()
        editor.add_primitive("cylinder", position=(3, 0, 2))
        editor.add_primitive("capsule", position=(4, 0, 2), size=(0.5, 0.5, 1.5))
        editor.remove_object(removed)
        editor.before_frame()
        self.assertEqual(editor.rebuild_count, 1)
        editor.end_edit()
        editor.before_frame()
        self.assertEqual(editor.rebuild_count, 2)
        self.assertEqual(editor.model.body_count, 3)
        index = editor.body_indices(first)[0]
        np.testing.assert_array_equal(editor.state_0.body_q.numpy()[index], before_q)
        np.testing.assert_array_equal(editor.state_0.body_qd.numpy()[index], before_qd)
        self.assertAlmostEqual(editor.control.joint_f.numpy()[0], 0.7, places=6)
        editor.step()
        self.assertLess(editor.get_pose(first)[0][2], before_q[2])

    def test_authored_objects_collide_in_one_physical_scene(self):
        """Settle a dynamic sphere on an authored static box rather than the ground."""
        editor = self.make_editor()
        support = editor.add_primitive("box", position=(0, 0, 0.5), dynamic=False)
        sphere = editor.add_primitive("sphere", position=(0, 0, 2), size=(0.5, 0.5, 0.5))
        editor.commit()
        for _ in range(120):
            editor.step()
        height = editor.get_pose(sphere)[0][2]
        self.assertGreater(height, 1.20)
        self.assertLess(height, 1.30)
        np.testing.assert_allclose(editor.get_pose(support)[0], (0, 0, 0.5), atol=1e-6)
        self.assertEqual(editor.rebuild_count, 1)

    def test_structural_edits_preserve_live_pose_and_authored_reset(self):
        """Resize and change mass of a settled prop at its live location, then reset it."""
        editor = self.make_editor()
        object_id = editor.add_primitive("box", position=(0, 0, 4))
        editor.commit()
        editor.update_object(object_id, position=(0, 0, 3))
        editor.before_frame()
        for _ in range(90):
            editor.step()
        position, rotation = editor.get_pose(object_id)
        self.assertLess(position[2], 0.6)
        editor.begin_edit()
        for index in range(10):
            editor.update_object(object_id, size=(1 + index * 0.02, 1, 1), mass=2)
            editor.before_frame()
        self.assertEqual(editor.rebuild_count, 1)
        editor.end_edit()
        editor.before_frame()
        np.testing.assert_allclose(editor.get_pose(object_id)[0], position, atol=1e-6)
        np.testing.assert_allclose(editor.get_pose(object_id)[1], rotation, atol=1e-6)
        self.assertAlmostEqual(editor.model.body_mass.numpy()[0], 2, places=6)
        editor.update_object(object_id, dynamic=False)
        editor.before_frame()
        np.testing.assert_allclose(editor.get_pose(object_id)[0], position, atol=1e-6)
        editor.update_object(object_id, position=(2, 0, 3), mass=3)
        editor.before_frame()
        np.testing.assert_allclose(editor.get_pose(object_id)[0], (2, 0, 3), atol=1e-6)
        editor.reset_scene()
        np.testing.assert_allclose(editor.get_pose(object_id)[0], (2, 0, 3), atol=1e-6)

    @unittest.skipUnless(importlib.util.find_spec("pxr"), "Requires the USD importer dependency")
    def test_failed_usd_transaction_keeps_working_runtime(self):
        """Reject an unsupported USD without losing the existing simulation."""
        editor = self.make_editor()
        object_id = editor.add_primitive("box", position=(0, 0, 2))
        editor.commit()
        model, state = editor.model, editor.state_0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsupported.usda"
            path.write_text(
                '#usda 1.0\n( metersPerUnit = 1 )\ndef Points "Unsupported" (prepend apiSchemas = ["NewtonParticleAPI"]) {}\n'
            )
            invalid = editor.add_usd(path)
            with self.assertRaisesRegex(ValueError, "rigid USD"):
                editor.commit()
            self.assertIs(editor.model, model)
            self.assertIs(editor.state_0, state)
            self.assertEqual(editor.rebuild_count, 1)
            editor.step()
            self.assertLess(editor.get_pose(object_id)[0][2], 2)
            editor.remove_object(invalid)
            editor.commit()
            self.assertEqual(editor.model.body_count, 1)
            # A discarded failed preparation must not block unrelated valid edits.
            obsolete = editor.add_usd(path)
            editor.remove_object(obsolete)
            editor.add_primitive("sphere", position=(2, 0, 2))
            deadline = time.monotonic() + 3
            while editor.busy and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertFalse(editor.busy)
            editor.before_frame()
            self.assertIsNone(editor.error)
            self.assertEqual(editor.model.body_count, 2)

    @unittest.skipUnless(importlib.util.find_spec("pxr"), "Requires the USD importer dependency")
    def test_static_usd_plane_supports_dynamic_objects(self):
        """Keep an imported zero-volume static collider physical and editable."""
        from pxr import Usd, UsdGeom, UsdPhysics

        editor = self.make_editor()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plane.usda"
            stage = Usd.Stage.CreateNew(str(path))
            UsdGeom.SetStageMetersPerUnit(stage, 1)
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            plane = UsdGeom.Plane.Define(stage, "/Plane")
            plane.CreateAxisAttr(UsdGeom.Tokens.z)
            UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
            stage.GetRootLayer().Save()
            plane_id = editor.add_usd(path, position=(0, 0, 1))
            sphere = editor.add_primitive("sphere", position=(0, 0, 2), size=(0.5, 0.5, 0.5))
            editor.commit()
            self.assertFalse(editor.get_object(plane_id).dynamic)
            for _ in range(90):
                editor.step()
            self.assertGreater(editor.get_pose(sphere)[0][2], 1.20)
            self.assertLess(editor.get_pose(sphere)[0][2], 1.30)
            np.testing.assert_allclose(editor.get_pose(plane_id)[0], (0, 0, 1), atol=1e-6)

    @unittest.skipUnless(importlib.util.find_spec("pxr"), "Requires the USD importer dependency")
    def test_usd_roundtrip_and_plain_geometry_get_real_physics(self):
        """Import a bare USD cube as a physical prop and reopen its authored scene."""
        editor = self.make_editor()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "cube.usda"
            source.write_text(
                '#usda 1.0\n(\n metersPerUnit = 1\n upAxis = "Z"\n)\ndef Cube "Cube" {\n double size = 0.4\n}\n'
            )
            object_id = editor.add_usd(source, position=(0, 0, 2))
            editor.commit()
            self.assertEqual(editor.model.body_count, 1)
            editor.step()
            self.assertLess(editor.get_pose(object_id)[0][2], 2)
            for _ in range(90):
                editor.step()
            self.assertGreater(editor.get_pose(object_id)[0][2], 0.15)
            self.assertLess(editor.get_pose(object_id)[0][2], 0.30)
            output = Path(directory) / "scene.usda"
            editor.save_usd(output)
            restored = self.make_editor()
            restored.load_usd(output)
            restored.commit()
            self.assertEqual(restored.objects, editor.objects)
            self.assertEqual(restored.model.body_count, 1)
            np.testing.assert_allclose(restored.get_pose(object_id)[0], (0, 0, 2), atol=1e-6)

            editor.begin_edit()
            for index in range(16):
                editor.update_object(object_id, size=(1 + index * 0.01,) * 3)
                editor.before_frame()
                self.assertFalse(editor.busy, "USD resizing must not prepare intermediate drag samples")
            editor.end_edit()
            editor.commit()
            self.assertEqual(editor.rebuild_count, 2)

            # Undo snapshots created while metadata loads retain authored mass semantics.
            from pxr import Usd, UsdGeom, UsdPhysics

            native_source = Path(directory) / "native.usda"
            stage = Usd.Stage.CreateNew(str(native_source))
            UsdGeom.SetStageMetersPerUnit(stage, 1)
            cube = UsdGeom.Cube.Define(stage, "/Body")
            cube.AddTranslateOp().Set((0, 0, 0.4))
            UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(2)
            stage.GetRootLayer().Save()
            native = self.make_editor()
            native_id = native.add_usd(native_source, position=(0, 0, 2))
            native.update_object(native_id, position=(0, 0, 3))
            native.commit()
            native.undo()
            native.before_frame()
            self.assertEqual(native.get_object(native_id).mass, 2)
            self.assertAlmostEqual(native.model.body_mass.numpy()[0], 2, places=6)
            self.assertEqual(native.rebuild_count, 1)
            placement = native.get_pose(native_id)[0]
            native.update_object(native_id, size=(2, 2, 2))
            native.commit()
            np.testing.assert_allclose(native.get_pose(native_id)[0], placement, atol=1e-6)
            explicit = self.make_editor()
            explicit_id = explicit.add_usd(native_source)
            explicit.update_object(explicit_id, mass=1)
            explicit.commit()
            self.assertEqual(explicit.get_object(explicit_id).mass, 1)
            self.assertAlmostEqual(explicit.model.body_mass.numpy()[0], 1, places=6)

    def test_empty_scene_and_invalid_values(self):
        """Keep an empty ground usable and reject invalid edits before changing the document."""
        editor = self.make_editor()
        editor.commit()
        editor.step()
        editor.reset_scene()
        self.assertEqual(editor.model.body_count, 0)
        for kind, size in (("sphere", (1, 2, 1)), ("capsule", (1, 1, 0.5)), ("box", (-1, 1, 1))):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                editor.add_primitive(kind, size=size)
        self.assertFalse(editor.objects)
        self.assertEqual(editor.rebuild_count, 1)


if __name__ == "__main__":
    unittest.main()
