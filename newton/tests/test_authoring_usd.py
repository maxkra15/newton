# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise scene document interchange through actual USD composition/import."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import newton
from newton._src.viewer.authoring_usd import NotAuthoringSceneError, load_scene, prepare_usd, save_scene
from newton.tests.unittest_utils import USD_AVAILABLE


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestAuthoringUsd(unittest.TestCase):
    @staticmethod
    def object(kind="box", **changes):
        values = {
            "id": "box-id",
            "name": "Box",
            "kind": SimpleNamespace(value=kind),
            "position": (1.0, 2.0, 3.0),
            "rotation": (0.0, 0.0, 0.0, 1.0),
            "size": (2.0, 2.0, 4.0),
            "color": (0.25, 0.5, 0.75),
            "mass": 3.0,
            "friction": 0.42,
            "dynamic": True,
            "source": None,
            "visible": True,
        }
        values.update(changes)
        return SimpleNamespace(**values)

    @staticmethod
    def asset(path, *, units=1.0, rigid=True, variants=False):
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(stage, units)
        UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
        root = UsdGeom.Xform.Define(stage, "/Asset")
        stage.SetDefaultPrim(root.GetPrim())
        cube = UsdGeom.Cube.Define(stage, "/Asset/Body")
        cube.CreateSizeAttr(2.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(0.0, 2.0, 0.0))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        if rigid:
            UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
            UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(2.0)
        if variants:
            UsdPhysics.MassAPI(cube.GetPrim()).GetMassAttr().Clear()
            for name in ("Newton", "MuJoCo", "PhysX"):
                variant = root.GetPrim().GetVariantSets().AddVariantSet(name)
                for selection in ("Enabled", "Disabled"):
                    variant.AddVariant(selection)
                variant.SetVariantSelection("Enabled" if name != "Newton" else "Disabled")
            variant = root.GetPrim().GetVariantSets().GetVariantSet("Newton")
            variant.SetVariantSelection("Enabled")
            with variant.GetVariantEditContext():
                UsdPhysics.MassAPI(cube.GetPrim()).GetMassAttr().Set(7.0)
            variant.SetVariantSelection("Disabled")
        stage.GetRootLayer().Save()
        return stage

    def test_primitives_roundtrip_and_native_physics(self):
        """Preserve definitions and standard geometry, mass, pose, and friction."""
        from pxr import Usd, UsdGeom

        objects = [
            self.object(),
            self.object("sphere", id="sphere", size=(2.0, 2.0, 2.0), position=(4.0, 0.0, 0.0)),
            self.object("capsule", id="capsule"),
            self.object("cylinder", id="cylinder", dynamic=False, visible=False),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Scene.USDA"
            save_scene(path, objects)
            definitions = load_scene(path)
            self.assertEqual([item["id"] for item in definitions], [item.id for item in objects])
            self.assertEqual(definitions[2]["kind"], "capsule")
            self.assertEqual(definitions[2]["size"], [2.0, 2.0, 4.0])
            self.assertFalse(definitions[3]["visible"])
            stage = Usd.Stage.Open(str(path))
            capsule = UsdGeom.Capsule(stage.GetPrimAtPath("/World/Object_2/Geometry"))
            self.assertEqual(capsule.GetHeightAttr().Get(), 2.0)
            self.assertEqual(capsule.GetRadiusAttr().Get(), 1.0)
            builder = newton.ModelBuilder()
            builder.add_usd(stage, load_visual_shapes=False)
            self.assertEqual(builder.body_count, 4)
            self.assertEqual(builder.shape_count, 5)
            np.testing.assert_allclose(builder.body_mass, [3.0] * 4)
            cylinder_body = builder.body_label.index("/World/Object_3")
            self.assertTrue(builder.body_flags[cylinder_body] & newton.BodyFlags.KINEMATIC)
            np.testing.assert_allclose(builder.body_q[0].p, (1.0, 2.0, 3.0))
            ground = builder.shape_label.index("/World/Ground")
            self.assertEqual(builder.shape_body[ground], -1)
            self.assertEqual(builder.shape_type[ground], newton.GeoType.PLANE)
            np.testing.assert_allclose(builder.shape_scale[ground], (0.0, 0.0, 0.0))
            self.assertAlmostEqual(builder.shape_material_mu[ground], 0.7, places=6)
            np.testing.assert_allclose(
                [value for index, value in enumerate(builder.shape_material_mu) if index != ground], [0.42] * 4
            )
            box_shape = builder.shape_label.index("/World/Object_0/Geometry")
            np.testing.assert_allclose(builder.shape_scale[box_shape], (1.0, 1.0, 2.0))
            np.testing.assert_allclose(builder.shape_color[box_shape], objects[0].color, atol=1.0e-6)

    def test_asset_variant_units_axis_and_source_isolation(self):
        """Apply Newton variants and SI conversion without editing source layers."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Asset.usda"
            stage = self.asset(path, variants=True)
            previous = path.read_bytes()
            template = prepare_usd(str(path), scale=(2.0, 2.0, 2.0))
            self.assertEqual(template.body_count, 1)
            self.assertEqual(template.body_mass[0], 7.0)
            np.testing.assert_allclose(template.body_q[0].p, (0.0, 0.0, 4.0), atol=1.0e-6)
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(
                stage.GetDefaultPrim().GetVariantSets().GetVariantSet("Newton").GetVariantSelection(), "Disabled"
            )
            template.body_mass[0] = 99.0
            self.assertEqual(prepare_usd(str(path)).body_mass[0], 7.0)
            visual = Path(directory) / "centimetres.usda"
            self.asset(visual, units=0.01, rigid=False)
            converted = prepare_usd(str(visual))
            np.testing.assert_allclose(converted.shape_transform[0].p, (0.0, 0.0, 0.02), atol=1.0e-6)
            np.testing.assert_allclose(converted.shape_scale[0], (0.01, 0.01, 0.01), atol=1.0e-6)
            rigid = Path(directory) / "rigid-centimetres.usda"
            self.asset(rigid, units=0.01)
            with self.assertRaisesRegex(ValueError, "metersPerUnit=1"):
                prepare_usd(str(rigid))

    def test_reference_composition_roundtrip(self):
        """Keep referenced rigid topology, scale, and object physics in exported USD."""
        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.usda"
            source_stage = self.asset(source)
            source_stage.CreateClassPrim("/Prototype").SetTypeName("Xform")
            prototype = UsdGeom.Cube.Define(source_stage, "/Prototype/Cube")
            prototype.GetPrim().CreateRelationship("guide:target").SetTargets(["/Prototype/Cube"])
            instance = UsdGeom.Xform.Define(source_stage, "/Asset/Geometry/Instance")
            variant = instance.GetPrim().GetVariantSets().AddVariantSet("Newton")
            variant.AddVariant("Disabled")
            variant.AddVariant("Enabled")
            variant.SetVariantSelection("Enabled")
            with variant.GetVariantEditContext():
                instance.GetPrim().GetReferences().AddInternalReference("/Prototype")
                instance.GetPrim().SetInstanceable(True)
            variant.SetVariantSelection("Disabled")
            source_stage.GetRootLayer().Save()
            path = Path(directory) / "scene.usda"
            save_scene(path, [self.object("usd", source=str(source), size=(2.0, 2.0, 2.0), color=(1.0, 1.0, 1.0))])
            definition = load_scene(path)[0]
            self.assertEqual(definition["source"], str(source))
            stage = Usd.Stage.Open(str(path))
            asset = stage.GetPrimAtPath("/World/Object_0/Asset/Asset")
            self.assertTrue(asset.HasAuthoredReferences())
            body = stage.GetPrimAtPath("/World/Object_0/Asset/Asset/Body")
            self.assertTrue(body.HasAPI(UsdPhysics.RigidBodyAPI))
            nested = stage.GetPrimAtPath("/World/Object_0/Asset/Asset/Geometry/Instance/Cube")
            self.assertFalse(nested.IsInstanceProxy())
            self.assertEqual(nested.GetRelationship("guide:target").GetTargets(), [nested.GetPath()])
            builder = newton.ModelBuilder()
            builder.add_usd(stage, load_visual_shapes=False)
            self.assertEqual(builder.body_count, 1)
            self.assertAlmostEqual(builder.body_mass[0], 3.0)
            shape = builder.shape_label.index("/World/Object_0/Asset/Asset/Body")
            self.assertAlmostEqual(builder.shape_material_mu[shape], 0.42, places=6)
            np.testing.assert_allclose(builder.body_q[0].p, (1.0, 2.0, 7.0), atol=1.0e-6)

    def test_multiroot_joint_and_inertia_references(self):
        """Preserve cross-root joints and rescale authored inertia with total mass."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "multiroot.usda"
            stage = Usd.Stage.CreateNew(str(source))
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            for name, mass in (("A", 2.0), ("B", 4.0)):
                cube = UsdGeom.Cube.Define(stage, f"/{name}")
                UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
                inertia = UsdPhysics.MassAPI.Apply(cube.GetPrim())
                inertia.CreateMassAttr(mass)
                inertia.CreateDiagonalInertiaAttr(Gf.Vec3f(mass))
            joint = UsdPhysics.FixedJoint.Define(stage, "/Joint")
            joint.CreateBody0Rel().SetTargets(["/A"])
            joint.CreateBody1Rel().SetTargets(["/B"])
            stage.GetRootLayer().Save()
            template = prepare_usd(str(source))
            path = Path(directory) / "scene.usda"
            save_scene(path, [self.object("usd", source=str(source), size=(1.0, 1.0, 1.0), mass=12.0)])
            exported = Usd.Stage.Open(str(path))
            joint = UsdPhysics.FixedJoint(exported.GetPrimAtPath("/World/Object_0/Asset/Joint"))
            self.assertEqual(str(joint.GetBody0Rel().GetTargets()[0]), "/World/Object_0/Asset/A")
            self.assertEqual(str(joint.GetBody1Rel().GetTargets()[0]), "/World/Object_0/Asset/B")
            builder = newton.ModelBuilder()
            builder.add_usd(exported, collapse_fixed_joints=False, load_visual_shapes=False)
            self.assertEqual(builder.joint_count, template.joint_count)
            self.assertEqual(builder.body_count, template.body_count)
            np.testing.assert_allclose(builder.body_mass, np.asarray(template.body_mass) * 2.0)
            np.testing.assert_allclose(builder.body_inertia, np.asarray(template.body_inertia) * 2.0)

    def test_preserve_native_physics_and_promote_visual_asset(self):
        """Preserve untouched rigid physics and give bare visual props a physical root."""
        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "rigid.usda"
            self.asset(source)
            template = prepare_usd(str(source))
            path = Path(directory) / "scene.usda"
            save_scene(
                path,
                [
                    self.object(
                        "usd",
                        source=str(source),
                        size=(1.0, 1.0, 1.0),
                        color=(1.0, 1.0, 1.0),
                        mass=99.0,
                        dynamic=False,
                        overrides=(),
                    )
                ],
            )
            builder = newton.ModelBuilder()
            builder.add_usd(str(path), load_visual_shapes=False)
            self.assertAlmostEqual(builder.body_mass[0], template.body_mass[0])
            self.assertFalse(builder.body_flags[0] & newton.BodyFlags.KINEMATIC)
            shape = builder.shape_label.index("/World/Object_0/Asset/Asset/Body")
            self.assertAlmostEqual(builder.shape_material_mu[shape], template.shape_material_mu[0])
            self.assertEqual(load_scene(path)[0]["overrides"], [])
            visual = Path(directory) / "visual.usda"
            stage = self.asset(visual, rigid=False)
            stage.GetPrimAtPath("/Asset/Body").RemoveAPI(UsdPhysics.CollisionAPI)
            UsdGeom.BasisCurves.Define(stage, "/Asset/grasp_identifier_curve").CreatePurposeAttr(UsdGeom.Tokens.guide)
            stage.GetRootLayer().Save()
            promoted = Path(directory) / "promoted.usda"
            save_scene(
                promoted,
                [self.object("usd", source=str(visual), size=(1.0, 1.0, 1.0), color=(1.0, 1.0, 1.0), overrides=())],
            )
            builder = newton.ModelBuilder()
            builder.add_usd(str(promoted), load_visual_shapes=False)
            self.assertEqual(builder.body_count, 1)
            self.assertAlmostEqual(builder.body_mass[0], 3.0)
            shape = builder.shape_label.index("/World/Object_0/Asset/Asset/Body")
            self.assertEqual(builder.shape_body[shape], 0)
            self.assertTrue(builder.shape_flags[shape] & newton.ShapeFlags.COLLIDE_SHAPES)
            exported = Usd.Stage.Open(str(promoted))
            guide = exported.GetPrimAtPath("/World/Object_0/Asset/Asset/grasp_identifier_curve")
            self.assertFalse(guide.HasAPI(UsdPhysics.CollisionAPI))

    def test_scalar_material_tint_and_unsupported_graph_save(self):
        """Tint scalar materials and retain the old file when a graph cannot be exported."""
        from pxr import Gf, Sdf, UsdShade

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "material.usda"
            stage = self.asset(source)
            material = UsdShade.Material.Define(stage, "/Asset/Material")
            shader = UsdShade.Shader.Define(stage, "/Asset/Material/Surface")
            shader.CreateIdAttr("UsdPreviewSurface")
            color = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
            color.Set(Gf.Vec3f(0.1, 0.2, 0.3))
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/Asset/Body")).Bind(material)
            stage.GetRootLayer().Save()
            template = prepare_usd(str(source))
            path = Path(directory) / "scene.usda"
            tint = (0.5, 0.75, 0.9)
            obj = self.object("usd", source=str(source), size=(1.0, 1.0, 1.0), color=tint)
            save_scene(path, [obj])
            builder = newton.ModelBuilder()
            builder.add_usd(str(path), load_visual_shapes=False)
            shape = builder.shape_label.index("/World/Object_0/Asset/Asset/Body")
            np.testing.assert_allclose(
                builder.shape_color[shape], np.asarray(template.shape_color[0]) * tint, atol=1.0e-6
            )
            previous = path.read_bytes()
            reader = UsdShade.Shader.Define(stage, "/Asset/Material/Reader")
            reader.CreateIdAttr("UsdPrimvarReader_float3")
            color.ConnectToSource(reader.ConnectableAPI(), "result")
            stage.GetRootLayer().Save()
            with self.assertRaisesRegex(ValueError, "connected color"):
                save_scene(path, [obj])
            self.assertEqual(path.read_bytes(), previous)
            obj.color = (1.0, 1.0, 1.0)
            save_scene(path, [obj])

    def test_reject_deformable_content_before_partial_import(self):
        """Allow visual guides but reject explicit deformable simulation in rigid assets."""
        from pxr import Gf, Sdf, UsdGeom

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.usda"
            stage = self.asset(path)
            guide = UsdGeom.BasisCurves.Define(stage, "/Asset/grasp_identifier_curve")
            guide.CreatePurposeAttr(UsdGeom.Tokens.guide)
            guide.CreateTypeAttr(UsdGeom.Tokens.linear)
            guide.CreateCurveVertexCountsAttr([2])
            guide.CreatePointsAttr([Gf.Vec3f(0.0), Gf.Vec3f(1.0)])
            points = UsdGeom.Points.Define(stage, "/Asset/GuidePoints")
            points.CreatePurposeAttr(UsdGeom.Tokens.guide)
            points.CreatePointsAttr([Gf.Vec3f(0.0)])
            stage.GetRootLayer().Save()
            template = prepare_usd(str(path))
            self.assertEqual(template.body_count, 1)
            self.assertEqual(template.shape_count, 1)
            self.assertEqual(template.particle_count, 0)
            points.GetPrim().SetMetadata(
                "apiSchemas", Sdf.TokenListOp.Create(prependedItems=["NewtonPointsDeformableSimAPI"])
            )
            stage.GetRootLayer().Save()
            with self.assertRaisesRegex(ValueError, "rigid USD assets only"):
                prepare_usd(str(path))

    def test_failed_save_preserves_destination_and_external_detection(self):
        """Leave saved scenes intact when a referenced asset cannot be prepared."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.usda"
            save_scene(path, [self.object()])
            previous = path.read_bytes()
            with self.assertRaises((ValueError, RuntimeError)):
                save_scene(path, [self.object("usd", source=str(Path(directory) / "missing.usda"))])
            self.assertEqual(path.read_bytes(), previous)
            external = Path(directory) / "external.usda"
            self.asset(external)
            with self.assertRaises(NotAuthoringSceneError):
                load_scene(external)


if __name__ == "__main__":
    unittest.main()
