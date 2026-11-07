# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Native authoring interaction coverage, separate from CPU document tests."""

import hashlib
import importlib.util
import json
import threading
import time
import unittest
from io import BytesIO
from pathlib import Path
from unittest import mock

import numpy as np
import warp as wp

from newton._src.viewer.gui.viewer_gui import ViewerGui
from newton.examples.basic.example_scene_editor import Example
from newton.tests import test_viewer_assets
from newton.tests.test_viewer_get_frame import _make_headless_viewer_gl_or_skip


class TestSceneAuthoringUI(unittest.TestCase):
    def test_drag_commit_undo_and_workspace_lifetime(self):
        """Deliver a native palette drag once and retain its presenter through model swaps."""
        try:
            from imgui_bundle import imgui
        except ImportError as error:
            self.skipTest(f"Viewer UI dependencies not available: {error}")

        previous_context = imgui.get_current_context()
        context = imgui.create_context()

        def restore_context():
            imgui.destroy_context(context)
            if previous_context is not None:
                imgui.set_current_context(previous_context)

        self.addCleanup(restore_context)
        with wp.ScopedDevice("cpu"):
            viewer = _make_headless_viewer_gl_or_skip(self, width=1280, height=900)
            self.addCleanup(viewer.close)
            viewer.gui = ViewerGui(viewer, viewer.renderer.window, workspace_enabled=True)
            self.assertTrue(viewer.gui.is_available)
            viewer.gui.ui.io.set_ini_filename("")
            editor = Example(viewer, Example.create_parser().parse_args(["--empty", "--solver", "xpbd"]))
            self.addCleanup(editor.close)
            editor.add_primitive("box", position=(-1, 0, 1))
            editor.commit()
            viewer.set_camera(pos=(5, -7, 4), pitch=-23, yaw=125)

            def render(count=1, *, simulate=False):
                for _ in range(count):
                    editor.before_frame()
                    viewer.gui._workspace_example = editor
                    if viewer.gui.workspace is not None:
                        viewer.gui.workspace.bind(editor)
                    if simulate and viewer.should_step() and viewer.gui.workspace.before_step():
                        editor.step()
                    if simulate:
                        viewer.gui.workspace.render_example()
                    else:
                        editor.render()
                    self.assertFalse(viewer.gui.ui.section_errors)

            def native_mouse(x, y, down=None, *, simulate=False):
                window = viewer.renderer.window
                native_y = window.height - y
                window.dispatch_event("on_mouse_motion", x, native_y, 0, 0)
                if down is not None:
                    window.dispatch_event("on_mouse_press" if down else "on_mouse_release", x, native_y, 1, 0)
                window.dispatch_events()
                render(simulate=simulate)

            def mouse(x, y, down=None):
                viewer.gui.ui.io.add_mouse_pos_event(x, y)
                if down is not None:
                    viewer.gui.ui.io.add_mouse_button_event(0, down)
                render()

            def click(x, y):
                mouse(x, y)
                mouse(x, y, True)
                mouse(x, y, False)
                render(3)

            render(4)
            workspace = viewer.gui.workspace
            panel = workspace.authoring
            self.assertIsNotNone(panel)
            self.assertIsNone(panel.assets)

            def scene_pixel(point):
                clip = (
                    viewer.camera.get_projection_matrix().reshape(4, 4).T
                    @ viewer.camera.get_view_matrix().reshape(4, 4).T
                    @ np.array((*point, 1.0))
                )
                ndc = clip[:3] / clip[3]
                x, y, width, height = workspace.rect
                return x + (ndc[0] + 1) * width / 2, y + (1 - ndc[1]) * height / 2

            def click_scene_point(point):
                pixel = scene_pixel(point)
                native_mouse(*pixel)
                native_mouse(*pixel, True)
                native_mouse(*pixel, False)
                render(2)

            # Deliver actual viewport clicks through the native release handler,
            # including the raycast used before authoring handles become active.
            click_scene_point((0, 0, 0))
            self.assertEqual(workspace.selected, ("shape", 0))
            obj = editor.objects[0]
            click_scene_point(obj.position)
            self.assertEqual(workspace.selected, ("body", editor.body_indices(obj.id)[0]))
            viewer.show_ground = False
            click_scene_point((0, 0, 0))
            self.assertIsNone(workspace.selected)
            viewer.show_ground = True

            initial_count, initial_builds, initial_model = len(editor.objects), editor.rebuild_count, editor.model
            # Fixed window dimensions and a fresh ImGui context make the first
            # palette card stable. All input goes through ImGui's real queue.
            mouse(74, 210)
            mouse(74, 210, True)
            mouse(90, 215)
            x, y, width, height = workspace.rect
            target = x + width * 0.45, y + height * 0.5
            mouse(*target)
            render(3)
            self.assertEqual(len(editor.objects), initial_count)
            self.assertEqual(editor.rebuild_count, initial_builds)
            self.assertIs(editor.model, initial_model)
            mouse(*target, False)
            render(4)
            self.assertEqual(len(editor.objects), initial_count + 1)
            self.assertEqual(editor.rebuild_count, initial_builds + 1)
            self.assertIs(workspace.authoring, panel)

            click(94, 94)  # Undo from the authoring toolbar.
            self.assertEqual(len(editor.objects), initial_count)
            click(122, 94)  # Redo from the same toolbar.
            self.assertEqual(len(editor.objects), initial_count + 1)
            self.assertIs(workspace.authoring, panel)

            from pyglet.window import key

            builds = editor.rebuild_count
            viewer.gui.handle_key_press(key.H)
            render(2)
            self.assertFalse(viewer.gui.show_ui)
            viewer.gui.handle_key_press(key.H)
            render(3)
            self.assertTrue(viewer.gui.show_ui)
            self.assertEqual(editor.rebuild_count, builds)
            self.assertIs(workspace.authoring, panel)
            self.assertGreater(np.ptp(viewer.get_frame(render_ui=True).numpy()), 0)

            # H must also finish a numeric drag. Otherwise its open gesture
            # indefinitely defers structural publication while the UI is hidden.
            edited = editor.objects[-1]
            editor.select(edited.id)
            render(4)
            builds = editor.rebuild_count
            mouse(1020, 358)
            mouse(1020, 358, True)
            mouse(1045, 358)
            render(2)
            self.assertNotEqual(editor.get_object(edited.id).size, edited.size)
            self.assertEqual(editor.rebuild_count, builds)
            viewer.gui.handle_key_press(key.H)
            render(3)
            self.assertFalse(viewer.gui.show_ui)
            self.assertFalse(editor.pending)
            self.assertEqual(editor.rebuild_count, builds + 1)
            viewer.gui.handle_key_press(key.H)
            render(2)
            self.assertTrue(viewer.gui.show_ui)

            # Keep a real viewport handle active across physics/frame boundaries,
            # rather than publishing only one edit before releasing the mouse.
            workspace.history.set_recording(True)
            workspace.history.playback_speed = 2.0
            render(simulate=True)
            self.assertTrue(workspace.history.frames)
            original = editor.get_object(edited.id)
            gx, gy = scene_pixel(editor.get_pose(edited.id)[0])
            gy -= 20
            sim_time = editor.sim_time
            viewer._paused = False
            native_mouse(gx, gy, simulate=True)
            self.assertTrue(viewer.gui._gizmo_hovered)
            native_mouse(gx, gy, True, simulate=True)
            self.assertTrue(panel._gizmo_active)
            self.assertTrue(viewer.is_paused())
            model, builds = editor.model, editor.rebuild_count
            last_z = original.position[2]
            deadline = time.monotonic() + 2.1
            moved = 0
            while time.monotonic() < deadline:
                moved += 1
                native_mouse(gx, gy - 2 * moved, simulate=True)
                self.assertEqual(editor.selected_id, edited.id)
                self.assertTrue(panel._gizmo_active)
                self.assertTrue(viewer.is_paused())
                self.assertIs(editor.model, model)
                self.assertEqual(editor.rebuild_count, builds)
                self.assertTrue(workspace.history.recording)
                self.assertTrue(workspace.history.frames)
                self.assertEqual(workspace.history.playback_speed, 2.0)
                position = editor.get_object(edited.id).position
                self.assertGreater(position[2], last_z)
                last_z = position[2]
                time.sleep(0.05)
            native_mouse(gx, gy - 2 * moved, False, simulate=True)
            render(2, simulate=True)
            self.assertFalse(panel._gizmo_active)
            self.assertFalse(viewer.is_paused())
            self.assertTrue(workspace.history.recording)
            self.assertEqual(workspace.history.playback_speed, 2.0)
            self.assertGreater(editor.sim_time, sim_time)
            self.assertGreater(editor.get_object(edited.id).position[2], original.position[2])
            editor.undo()
            render()
            self.assertEqual(editor.get_object(edited.id).position, original.position)

            if importlib.util.find_spec("pxr") is not None:
                # Reuse the real HTTP package/cache fixture. Only worker timing
                # is controlled: parsing, metadata, USD import and native input
                # all follow their production paths.
                assets = test_viewer_assets.TestAssetLibrary()
                assets.setUp()
                self.addCleanup(assets.doCleanups)
                # The service fixture checks bytes; this workflow also imports
                # its USD and decodes its thumbnail, so supply complete content.
                assets.files["simready_usd/payloads/mesh.usda"] = (
                    b'#usda 1.0\n(defaultPrim = "Root")\ndef Xform "Root" {\n def Cube "Mesh" {}\n}\n'
                )
                from pyglet.image import SolidColorImagePattern

                image_bytes = BytesIO()
                SolidColorImagePattern((100, 180, 200, 255)).create_image(8, 8).save("thumbnail.png", file=image_bytes)
                assets.files["simready_usd/.thumbs/256x256/root.usda.png"] = image_bytes.getvalue()
                bom_path = ".metadata/com.nvidia.simready.packaging.bom.json"
                bom = json.loads(assets.files[bom_path])
                for item in bom["items"]:
                    data = assets.files[item["relative_path"]]
                    item.update(size=len(data), hash={"sha256": hashlib.sha256(data).hexdigest()})
                assets.files[bom_path] = json.dumps(bom).encode()
                manifest_path = "com.nvidia.simready.packaging.json"
                manifest = json.loads(assets.files[manifest_path])
                manifest["metadata"][0]["hash"]["sha256"] = hashlib.sha256(assets.files[bom_path]).hexdigest()
                assets.files[manifest_path] = json.dumps(manifest).encode()
                replacement_path = Path(assets.directory.name) / "replacement.usda"
                replacement_objects = editor.objects
                editor.save_usd(replacement_path)
                record = assets.record()
                assets.library.download(record)
                assets.until(lambda: assets.library.ready(record) is not None)
                release_thumbnail = threading.Event()
                self.addCleanup(release_thumbnail.set)
                thumbnail = assets.library._thumbnail

                def deferred_thumbnail(record):
                    release_thumbnail.wait(5)
                    return thumbnail(record)

                with mock.patch.object(assets.library, "_thumbnail", deferred_thumbnail):
                    panel.assets = assets.library
                    panel.query = panel._submitted_query = "sponge"

                    native_mouse(95, 123)
                    native_mouse(95, 123, True)
                    native_mouse(95, 123, False)
                    render(3)
                    native_mouse(75, 290)
                    native_mouse(75, 290, True)
                    native_mouse(75, 290, False)
                    render(2)
                    self.assertIsNotNone(panel._selected_asset)
                    self.assertEqual(panel._selected_asset.id, record.id)
                    self.assertIsNone(editor.selected_id)
                    before = len(editor.objects)
                    # A quick second press is also a potential double click.
                    # It must remain a drag until its release decides otherwise.
                    native_mouse(75, 290, True)
                    self.assertEqual(len(editor.objects), before)
                    release_thumbnail.set()
                    assets.until(lambda: not assets.library.loading)
                    render(3)
                    self.assertIsNotNone(panel._selected_asset)
                    self.assertEqual(panel._selected_asset.id, record.id)
                    self.assertEqual(panel._selected_asset.license, "CC-BY-4.0")
                    native_mouse(100, 295)
                    native_mouse(*target)
                    render(3)
                    self.assertEqual(len(editor.objects), before)
                    native_mouse(*target, False)
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline and (len(editor.objects) == before or editor.pending):
                        render()
                        time.sleep(0.01)
                    self.assertIsNone(editor.error)
                    self.assertEqual(len(editor.objects), before + 1)
                    imported = editor.objects[-1]
                    self.assertEqual(imported.kind.value, "usd")
                    self.assertTrue(editor.body_indices(imported.id))
                    self.assertTrue(panel._textures)
                    before = len(editor.objects)
                    # A different point starts a fresh double-click sequence
                    # without sleeping for ImGui's multi-click timeout.
                    native_mouse(110, 290)
                    native_mouse(110, 290, True)
                    native_mouse(110, 290, False)
                    native_mouse(110, 290, True)
                    self.assertEqual(len(editor.objects), before)
                    native_mouse(110, 290, False)
                    render(3)
                    self.assertEqual(len(editor.objects), before + 1)
                    self.assertTrue(editor.body_indices(editor.objects[-1].id))

                    native_mouse(75, 290)
                    native_mouse(75, 290, True)
                    native_mouse(75, 290, False)
                    render(3)
                    self.assertIsNotNone(panel._selected_asset)
                    self.assertEqual(panel._selected_asset.id, record.id)
                    # The large inspector preview is also a placement source.
                    before = len(editor.objects)
                    builds = editor.rebuild_count
                    model = editor.model
                    with mock.patch.object(viewer, "log_lines", wraps=viewer.log_lines) as lines:
                        native_mouse(1080, 222)
                        native_mouse(1080, 222, True)
                        native_mouse(1060, 230)
                        native_mouse(*target)
                        render(3)
                        outlines = [
                            call
                            for call in lines.call_args_list
                            if call.args[0] == "__scene_authoring/placement" and call.args[1] is not None
                        ]
                        self.assertTrue(outlines)
                        vertices = np.concatenate([array.numpy() for array in outlines[-1].args[1:3]])
                        lower, upper = vertices.min(axis=0), vertices.max(axis=0)
                        np.testing.assert_allclose(upper - lower, (0.02, 0.02, 0.02), atol=1e-6)
                    self.assertEqual(len(editor.objects), before)
                    self.assertEqual(editor.rebuild_count, builds)
                    self.assertIs(editor.model, model)
                    native_mouse(*target, False)
                    render(4)
                    self.assertIsNone(editor.error)
                    self.assertEqual(len(editor.objects), before + 1)
                    self.assertEqual(editor.rebuild_count, builds + 1)
                    self.assertEqual(editor.objects[-1].kind.value, "usd")
                    self.assertTrue(editor.body_indices(editor.objects[-1].id))
                    center = editor.get_pose(editor.objects[-1].id)[0]
                    np.testing.assert_allclose(lower, np.asarray(center) - 0.01, atol=1e-5)
                    np.testing.assert_allclose(upper, np.asarray(center) + 0.01, atol=1e-5)

                def queue_download():
                    # Request the same real package with an incomplete cache;
                    # hold its response until another interaction owns the UI.
                    (assets.library._package(record) / "complete.json").unlink()
                    assets.library._verified.pop(record.id, None)
                    assets.requests.clear()
                    assets.hold.set()
                    assets.release.clear()
                    panel._place_asset(record, (-3, 0, 1))
                    assets.until(lambda: any(path.endswith("LICENSE.txt") for path in assets.requests))
                    self.assertEqual(len(panel._pending_imports), 1)

                def complete_download():
                    assets.release.set()
                    assets.until(lambda: assets.library.ready(record) is not None)

                def await_import(count):
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline and (len(editor.objects) != count or editor.pending):
                        render()
                        time.sleep(0.01)
                    self.assertIsNone(editor.error)
                    self.assertEqual(len(editor.objects), count)
                    self.assertFalse(editor.pending)

                # A finished transfer must not mutate a historical scene.
                viewer._paused = True
                queue_download()
                history = workspace.history
                history.set_recording(True)
                history.capture()
                editor.step()
                history.capture()
                history.select(0)
                self.assertTrue(history.replaying)
                document = editor.objects
                complete_download()
                render(2, simulate=True)
                self.assertEqual(editor.objects, document)
                self.assertEqual(len(panel._pending_imports), 1)
                history.select(len(history.frames) - 1)
                await_import(len(document) + 1)

                # Likewise, retain the completed request until a real held
                # viewport gesture releases its single undo transaction.
                queue_download()
                editor.select(edited.id)
                render(3)
                gx, gy = scene_pixel(editor.get_pose(edited.id)[0])
                gy -= 20
                native_mouse(gx, gy, simulate=True)
                native_mouse(gx, gy, True, simulate=True)
                native_mouse(gx, gy - 4, simulate=True)
                self.assertTrue(panel._gizmo_active)
                self.assertIsNotNone(editor._edit_before)
                count = len(editor.objects)
                complete_download()
                render(2, simulate=True)
                self.assertEqual(len(editor.objects), count)
                self.assertEqual(len(panel._pending_imports), 1)
                self.assertTrue(panel._gizmo_active)
                native_mouse(gx, gy - 4, False, simulate=True)
                await_import(count + 1)

                # Opening a document supersedes the old drop request while its
                # verified package remains available in the shared cache.
                queue_download()
                editor.load_usd(replacement_path)
                complete_download()
                await_import(len(replacement_objects))
                self.assertEqual(editor.objects, replacement_objects)
                self.assertFalse(panel._pending_imports)
                self.assertIsNotNone(assets.library.ready(record))

            workspace.bind(None)
            self.assertIsNone(workspace.authoring)
            self.assertIsNone(panel._file_drop_handler)


if __name__ == "__main__":
    unittest.main()
