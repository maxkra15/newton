# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Native authoring interaction coverage, separate from CPU document tests."""

import unittest

import numpy as np
import warp as wp

from newton._src.viewer.gui.viewer_gui import ViewerGui
from newton.examples.basic.example_scene_editor import Example
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

            workspace.bind(None)
            self.assertIsNone(workspace.authoring)
            self.assertIsNone(panel._file_drop_handler)


if __name__ == "__main__":
    unittest.main()
