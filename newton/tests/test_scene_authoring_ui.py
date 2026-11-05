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
            editor = Example(viewer, Example.create_parser().parse_args(["--empty"]))
            self.addCleanup(editor.close)
            editor.add_primitive("box", position=(-1, 0, 1))
            editor.commit()
            viewer.set_camera(pos=(5, -7, 4), pitch=-23, yaw=125)

            def render(count=1):
                for _ in range(count):
                    editor.before_frame()
                    viewer.gui._workspace_example = editor
                    if viewer.gui.workspace is not None:
                        viewer.gui.workspace.bind(editor)
                    editor.render()
                    self.assertFalse(viewer.gui.ui.section_errors)

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

                    def native_mouse(x, y, down=None):
                        window = viewer.renderer.window
                        native_y = window.height - y
                        window.dispatch_event("on_mouse_motion", x, native_y, 0, 0)
                        if down is not None:
                            window.dispatch_event("on_mouse_press" if down else "on_mouse_release", x, native_y, 1, 0)
                        window.dispatch_events()
                        render()

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
                    native_mouse(1080, 222)
                    native_mouse(1080, 222, True)
                    native_mouse(1060, 230)
                    native_mouse(*target)
                    render(3)
                    self.assertEqual(len(editor.objects), before)
                    self.assertEqual(editor.rebuild_count, builds)
                    native_mouse(*target, False)
                    render(4)
                    self.assertIsNone(editor.error)
                    self.assertEqual(len(editor.objects), before + 1)
                    self.assertEqual(editor.rebuild_count, builds + 1)
                    self.assertEqual(editor.objects[-1].kind.value, "usd")
                    self.assertTrue(editor.body_indices(editor.objects[-1].id))

            workspace.bind(None)
            self.assertIsNone(workspace.authoring)
            self.assertIsNone(panel._file_drop_handler)


if __name__ == "__main__":
    unittest.main()
