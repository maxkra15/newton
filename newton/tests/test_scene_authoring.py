# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Physical and transactional contracts of the opt-in authoring scene."""

import importlib.util
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import warp as wp

from newton import CollisionPipeline, ModelBuilder
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
        """Settle on an authored support and insert overlapping drops without launching its occupant."""
        editor = self.make_editor()
        support = editor.add_primitive("box", position=(0, 0, 0.5), dynamic=False)
        sphere = editor.add_primitive("sphere", position=(0, 0, 2), size=(0.5, 0.5, 0.5))
        editor.add_primitive("box", position=(4, 0, 10), size=(2, 2, 0.5), dynamic=False)
        editor.commit()
        for _ in range(120):
            editor.step()
        height = editor.get_pose(sphere)[0][2]
        self.assertGreater(height, 1.20)
        self.assertLess(height, 1.30)
        np.testing.assert_allclose(editor.get_pose(support)[0], (0, 0, 0.5), atol=1e-6)
        self.assertEqual(editor.rebuild_count, 1)
        index = editor.body_indices(sphere)[0]
        editor.state_0.body_qd[index : index + 1].fill_(wp.spatial_vector(10, 0, 0, 0, 0, 0))
        for _ in range(12):
            editor.step()
        for state in (editor.state_0, editor.state_1):
            state.body_qd[index : index + 1].zero_()
        live = editor.get_pose(sphere)[0]
        self.assertGreater(live[0], 1.5)
        height = live[2]
        before_q, before_v = editor.state_0.body_q.numpy().copy(), editor.state_0.body_qd.numpy().copy()
        first = editor.add_primitive("box", position=(live[0], live[1], 0.5))
        second = editor.add_primitive("box", position=(live[0], live[1], 0.5))
        reset_drop = editor.add_primitive("box", position=(0, 0, 0.5))
        overhead_drop = editor.add_primitive("box", position=(4, 0, 2))
        for object_id in (first, second, reset_drop, overhead_drop):
            editor.place_object(object_id)
        editor.commit()
        np.testing.assert_array_equal(editor.state_0.body_q.numpy()[: len(before_q)], before_q)
        np.testing.assert_array_equal(editor.state_0.body_qd.numpy()[: len(before_v)], before_v)
        self.assertGreater(editor.get_object(first).position[2], height + 0.7)
        self.assertLess(editor.get_object(first).position[2], height + 0.8)
        self.assertGreater(editor.get_object(second).position[2] - editor.get_object(first).position[2], 1.0)
        self.assertGreater(editor.get_object(reset_drop).position[2], 2.7)
        self.assertGreater(editor.get_object(overhead_drop).position[2], 10.75)
        self.assertLess(editor.get_object(overhead_drop).position[2], 10.8)
        self.assertEqual(editor.rebuild_count, 2)
        editor.step()
        self.assertLess(np.linalg.norm(editor.state_0.body_qd.numpy()[:, :3], axis=1).max(), 1.0)
        self.assertTrue(np.isfinite(editor.state_0.body_q.numpy()).all())
        resolved = {obj.id: obj.position for obj in editor.objects}
        for removed in (overhead_drop, reset_drop, second):
            first_body = editor.body_indices(first)[0]
            first_pose = editor.state_0.body_q.numpy()[first_body].copy()
            first_velocity = editor.state_0.body_qd.numpy()[first_body].copy()
            editor.undo()
            editor.before_frame()
            self.assertNotIn(removed, {obj.id for obj in editor.objects})
            self.assertEqual(editor.get_object(first).position, resolved[first])
            first_body = editor.body_indices(first)[0]
            np.testing.assert_array_equal(editor.state_0.body_q.numpy()[first_body], first_pose)
            np.testing.assert_array_equal(editor.state_0.body_qd.numpy()[first_body], first_velocity)
        for restored_id in (second, reset_drop, overhead_drop):
            editor.redo()
            editor.before_frame()
            self.assertEqual(editor.get_object(restored_id).position, resolved[restored_id])
            self.assertEqual(editor.get_object(first).position, resolved[first])
        editor.reset_scene()
        for _ in range(60):
            editor.step()
            self.assertLess(np.linalg.norm(editor.state_0.body_qd.numpy()[:, :3], axis=1).max(), 10.0)
        if importlib.util.find_spec("pxr"):
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "safe-reset.usda"
                editor.save_usd(output)
                restored = self.make_editor()
                restored.load_usd(output)
                restored.commit()
                for _ in range(60):
                    restored.step()
                    self.assertLess(np.linalg.norm(restored.state_0.body_qd.numpy()[:, :3], axis=1).max(), 10.0)

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
            # A minute tilt must use this footprint's plane equation, rather
            # than the collision half-space's million-metre lateral extent.
            editor.update_object(plane_id, rotation=tuple(wp.quat_from_axis_angle(wp.vec3(1, 0, 0), 0.001)))
            editor.before_frame()
            placed = editor.add_primitive("box", position=(3, 0, 0.5))
            editor.place_object(placed)
            editor.commit()
            self.assertGreater(editor.get_object(placed).position[2], 1.5)
            self.assertLess(editor.get_object(placed).position[2], 1.6)

    @unittest.skipUnless(importlib.util.find_spec("pxr"), "Requires the USD importer dependency")
    def test_safe_usd_placement_uses_bounds_and_rolls_back(self):
        """Place offset assets as a batch and retain unresolved authoring data on failed publication."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        editor = self.make_editor()
        editor.commit()
        original_model = editor.model
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "offset.usda"
            stage = Usd.Stage.CreateNew(str(source))
            UsdGeom.SetStageMetersPerUnit(stage, 1)
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            cube = UsdGeom.Cube.Define(stage, "/Asset/Cube")
            cube.CreateSizeAttr(0.4)
            cube.AddTranslateOp().Set(Gf.Vec3d(0.4, -0.3, -0.6))
            stage.GetRootLayer().Save()
            with (
                mock.patch.object(editor.viewer, "log_lines", wraps=editor.viewer.log_lines) as lines,
                mock.patch.object(
                    ModelBuilder, "finalize", autospec=True, side_effect=ModelBuilder.finalize
                ) as finalize,
            ):
                for x in (2.0, 2.1, 2.2):
                    deadline = time.monotonic() + 5
                    pivot = None
                    while pivot is None and time.monotonic() < deadline:
                        pivot = editor.preview_usd(source, position=(x, -1, 0), scale=(2, 0.5, 3), placement=True)
                        editor.before_frame()
                        self.assertFalse(editor.busy)
                        self.assertEqual(editor.objects, ())
                        self.assertIs(editor.model, original_model)
                        self.assertEqual(editor.rebuild_count, 1)
                        if pivot is None:
                            time.sleep(0.005)
                    self.assertIsNotNone(pivot, editor.error)
                    editor.render()
                    vertices = np.concatenate([array.numpy() for array in lines.call_args.args[1:3]])
                    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
                    np.testing.assert_allclose(upper - lower, (0.8, 0.2, 1.2), atol=1e-6)
                    np.testing.assert_allclose((lower + upper) / 2 - pivot, (0.8, -0.15, -1.8), atol=1e-6)
                    self.assertGreaterEqual(lower[2], 0)
                    self.assertLess(lower[2], 0.03)
                finalize.assert_not_called()
                editor.clear_preview()
            plane_path = Path(directory) / "plane.usda"
            plane_stage = Usd.Stage.CreateNew(str(plane_path))
            UsdGeom.SetStageMetersPerUnit(plane_stage, 1)
            UsdGeom.SetStageUpAxis(plane_stage, UsdGeom.Tokens.z)
            plane = UsdGeom.Plane.Define(plane_stage, "/Plane")
            UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
            plane_stage.GetRootLayer().Save()
            first = editor.add_usd(source, position=(0, 0, 0))
            second = editor.add_usd(source, position=(0, 0, 0))
            invalid = editor.add_usd(plane_path)
            for object_id in (first, second, invalid):
                editor.place_object(object_id)
            with self.assertRaisesRegex(ValueError, "bounded geometry"):
                editor.commit()
            self.assertIs(editor.model, original_model)
            self.assertEqual(editor.get_object(first).position, (0, 0, 0))
            self.assertEqual(editor.get_object(second).position, (0, 0, 0))
            editor.remove_object(invalid)
            explicit = editor.add_primitive("box", position=(4, 0, 0.123))
            editor.commit()
            self.assertGreater(editor.get_object(first).position[2], 0.8)
            self.assertGreater(editor.get_object(second).position[2] - editor.get_object(first).position[2], 0.4)
            self.assertEqual(editor.get_object(explicit).position, (4, 0, 0.123))
            self.assertEqual(editor.rebuild_count, 2)
            editor.step()
            for object_id in (first, second):
                index = editor.body_indices(object_id)[0]
                self.assertLess(np.linalg.norm(editor.state_0.body_qd.numpy()[index, :3]), 1.0)
            output = Path(directory) / "placed.usda"
            editor.save_usd(output)
            restored = self.make_editor()
            restored.load_usd(output)
            restored.commit()
            self.assertEqual(restored.objects, editor.objects)
            editor.reset_scene()
            for object_id in (first, second):
                np.testing.assert_allclose(
                    editor.get_pose(object_id)[0], editor.get_object(object_id).position, atol=1e-6
                )

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

            model = editor.model
            with (
                mock.patch.object(editor, "_prepare_template", wraps=editor._prepare_template) as prepare,
                mock.patch.object(editor.viewer, "log_lines", wraps=editor.viewer.log_lines) as lines,
            ):
                editor.begin_edit()
                for index in range(16):
                    scale = (1 + index * 0.01,) * 3
                    editor.update_object(object_id, size=scale)
                    pivot = editor.preview_usd(source, position=editor.get_pose(object_id)[0], scale=scale)
                    self.assertFalse(editor.busy, "USD resize previews must not start intermediate preparation")
                    self.assertIsNotNone(pivot)
                    editor.before_frame()
                    editor.render()
                    vertices = np.concatenate([array.numpy() for array in lines.call_args.args[1:3]])
                    np.testing.assert_allclose(
                        vertices.max(axis=0) - vertices.min(axis=0), np.asarray(scale) * 0.4, atol=1e-6
                    )
                    self.assertIs(editor.model, model)
                    self.assertEqual(editor.rebuild_count, 1)
                prepare.assert_not_called()
                editor.end_edit()
                editor.commit()
                prepare.assert_called_once()
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

    @unittest.skipUnless(importlib.util.find_spec("pxr"), "Requires the USD importer dependency")
    def test_topology_changes_preserve_fixed_joint_asset_controls(self):
        """Move jointed controls across topology edits without copying empty joint spans."""
        from pxr import Usd, UsdGeom, UsdPhysics

        editor = self.make_editor()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixed.usda"
            stage = Usd.Stage.CreateNew(str(path))
            UsdGeom.SetStageMetersPerUnit(stage, 1)
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            root = UsdGeom.Xform.Define(stage, "/Assembly")
            UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
            for name, height in (("Base", 0.0), ("Child", 0.2)):
                body = UsdGeom.Cube.Define(stage, f"/Assembly/{name}")
                body.CreateSizeAttr(0.1)
                body.AddTranslateOp().Set((0, 0, height))
                UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
                UsdPhysics.CollisionAPI.Apply(body.GetPrim())
                UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(1)
            joint = UsdPhysics.FixedJoint.Define(stage, "/Assembly/Weld")
            joint.CreateBody0Rel().SetTargets(["/Assembly/Base"])
            joint.CreateBody1Rel().SetTargets(["/Assembly/Child"])
            joint.CreateLocalPos0Attr((0, 0, 0.2))
            stage.GetRootLayer().Save()
            leading = editor.add_primitive("box", position=(-3, 0, 1))
            asset = editor.add_usd(path, position=(0, 0, 2))
            editor.commit()
            old_pose = editor.state_0.body_q.numpy()[list(editor.body_indices(asset))].copy()
            expected = {}
            for field in ("joint_f", "joint_act", "joint_target_q", "joint_target_qd"):
                values = getattr(editor.control, field, None)
                if values is not None:
                    data = np.arange(values.size, dtype=np.float32) / 100 + 0.1
                    values.assign(data)
                    expected[field] = data[7 if field == "joint_target_q" else 6 :].copy()
            editor.remove_object(leading)
            trailing = editor.add_primitive("box", position=(3, 0, 1))
            editor.commit()
            np.testing.assert_array_equal(editor.state_0.body_q.numpy()[list(editor.body_indices(asset))], old_pose)
            for field, data in expected.items():
                actual = getattr(editor.control, field).numpy()
                np.testing.assert_array_equal(actual[: len(data)], data)
                np.testing.assert_array_equal(
                    actual[len(data) :], getattr(editor.model.control(), field).numpy()[len(data) :]
                )
            editor.remove_object(trailing)
            editor.commit()
            for field, data in expected.items():
                np.testing.assert_array_equal(getattr(editor.control, field).numpy(), data)
            self.assertEqual(editor.rebuild_count, 3)

    def test_contact_capacity_recovers_before_solving(self):
        """Grow an undersized native contact pipeline while retaining the physical runtime."""
        editor = self.make_editor()
        editor.add_primitive("box", position=(0, 0, 0.5))
        editor.commit()
        model, solver, control = editor.model, editor.solver, editor.control
        states = {id(editor.state_0), id(editor.state_1)}
        poses = {id(editor.state_0.body_q), id(editor.state_1.body_q)}
        undersized = CollisionPipeline(model, rigid_contact_max=1)
        editor._runtime.pipeline = editor.collision_pipeline = undersized
        editor._runtime.contacts = editor.contacts = undersized.contacts()
        editor.viewer.set_collision_pipeline(undersized)
        with self.assertLogs("newton", level="WARNING") as messages:
            editor.step()
        self.assertTrue(any("Expanded rigid contact capacity" in line for line in messages.output))
        self.assertGreater(editor.contacts.rigid_contact_max, 1)
        self.assertGreaterEqual(int(editor.contacts.rigid_contact_count.numpy()[0]), 4)
        self.assertLessEqual(int(editor.contacts.rigid_contact_count.numpy()[0]), editor.contacts.rigid_contact_max)
        self.assertEqual(model.rigid_contact_max, editor.contacts.rigid_contact_max)
        self.assertEqual(
            model._get_contact_capacity(), (editor.contacts.rigid_contact_max, editor.contacts.soft_contact_max)
        )
        self.assertIs(editor.collision_pipeline, editor._runtime.pipeline)
        self.assertIs(editor.contacts, editor._runtime.contacts)
        self.assertIs(editor.model, model)
        self.assertIs(editor.solver, solver)
        self.assertIs(editor.control, control)
        self.assertEqual({id(editor.state_0), id(editor.state_1)}, states)
        self.assertEqual({id(editor.state_0.body_q), id(editor.state_1.body_q)}, poses)
        self.assertTrue(np.isfinite(editor.state_0.body_q.numpy()).all())
        self.assertEqual(editor.rebuild_count, 1)

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
