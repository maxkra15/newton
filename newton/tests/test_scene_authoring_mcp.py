# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise the optional scene bridge over the SDK's real HTTP transport."""

import asyncio
import base64
import importlib.util
import json
import socket
import tempfile
import threading
import time
import unittest
from concurrent.futures import Future
from io import BytesIO
from pathlib import Path
from unittest import mock
from urllib.parse import urlsplit

import numpy as np
import warp as wp

from newton._src.viewer.gui.viewer_gui import ViewerGui
from newton.examples.basic.example_scene_editor import Example
from newton.tests.test_viewer_get_frame import _make_headless_viewer_gl_or_skip
from newton.viewer import SceneEditor, SceneEditorMCP, ViewerNull

_HAS_MCP = importlib.util.find_spec("mcp") is not None and importlib.util.find_spec("uvicorn") is not None

if _HAS_MCP:
    import httpx2
    from mcp import Client
    from mcp.client.streamable_http import streamable_http_client


class _SDKClient:
    """Keep one authenticated SDK session off the scene's owning thread."""

    def __init__(self, url, token):
        self.url, self.token = url, token
        self.ready = Future()
        self.error = None
        self.closed = False
        self.thread = threading.Thread(target=self._run, name="test-scene-mcp-client", daemon=True)
        self.thread.start()

    def _run(self):
        try:
            asyncio.run(self._connect())
        except BaseException as error:
            if not self.ready.done():
                self.ready.set_exception(error)
            else:
                self.error = error

    async def _connect(self):
        self.loop = asyncio.get_running_loop()
        self.stop = asyncio.Event()
        async with httpx2.AsyncClient(
            headers={"Authorization": f"Bearer {self.token}"}, timeout=httpx2.Timeout(3, read=30)
        ) as http:
            async with Client(
                streamable_http_client(self.url, http_client=http), read_timeout_seconds=30
            ) as self.session:
                self.protocol_version = self.session.protocol_version
                self.ready.set_result(True)
                await self.stop.wait()

    def call(self, name, arguments):
        return asyncio.run_coroutine_threadsafe(self.session.call_tool(name, arguments), self.loop)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if hasattr(self, "loop") and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.stop.set)
        self.thread.join(5)
        if self.thread.is_alive():
            raise RuntimeError("MCP client did not shut down")
        if self.error is not None:
            raise self.error


class _SceneSession:
    """Pump the same frame boundaries as the application while awaiting requests."""

    def __init__(self, test, viewer, *, native=False):
        self.test, self.viewer, self.native = test, viewer, native
        self.directory = tempfile.TemporaryDirectory()
        test.addCleanup(self.directory.cleanup)
        if native:
            args = Example.create_parser().parse_args(
                ["--empty", "--mcp-port", "0", "--mcp-info", str(Path(self.directory.name) / "connection.json")]
            )
            self.editor = Example(viewer, args)
        else:
            self.editor = SceneEditor(viewer, device="cpu")
        test.addCleanup(self.editor.close)
        self.editor.add_primitive("box", name="Initial", position=(0, 0, 1))
        self.editor.commit()
        self.bridge = (
            self.editor._mcp
            if native
            else SceneEditorMCP(self.editor, port=0, info_file=Path(self.directory.name) / "connection.json")
        )
        test.addCleanup(self.bridge.close)
        self.info = json.loads(self.bridge.info_file.read_text())
        self.client = _SDKClient(self.bridge.url, self.info["token"])
        test.addCleanup(self.client.close)
        self.wait(self.client.ready)

    def pump(self):
        if not self.native:
            self.bridge.before_frame()
        self.editor.before_frame()
        gui = getattr(self.viewer, "gui", None)
        if gui is not None:
            gui._workspace_example = self.editor
            if gui.workspace is not None:
                gui.workspace.bind(self.editor)
        if self.native and self.viewer.should_step() and (gui is None or gui.workspace.before_step()):
            self.editor.step()
        if gui is not None and gui.workspace is not None:
            gui.workspace.render_example()
        else:
            self.editor.render()
        if not self.native:
            self.bridge.after_frame()

    def wait(self, future):
        deadline = time.monotonic() + 40
        while not future.done() and time.monotonic() < deadline:
            self.pump()
            time.sleep(0.001)
        self.test.assertTrue(future.done(), "MCP request did not complete at a frame boundary")
        return future.result()

    def call(self, name, arguments=None):
        return self.wait(self.client.call(name, arguments or {}))

    def data(self, name, arguments=None):
        result = self.call(name, arguments)
        self.test.assertFalse(result.is_error, str(result.content))
        self.test.assertIsInstance(result.structured_content, dict)
        return result.structured_content


@unittest.skipUnless(_HAS_MCP, "Requires the optional MCP SDK and HTTP server")
class TestSceneAuthoringMCP(unittest.TestCase):
    def test_atomic_batches_guards_recovery_and_shutdown(self):
        """Apply real MCP batches once, reject unsafe edits, and recover from a failed USD."""
        with wp.ScopedDevice("cpu"):
            viewer = ViewerNull()
            self.addCleanup(viewer.close)
            scene = _SceneSession(self, viewer)
            self.assertGreaterEqual(scene.client.protocol_version, "2026-07-28")
            self.assertEqual(scene.bridge.info_file.stat().st_mode & 0o777, 0o600)
            self.assertEqual(httpx2.post(scene.bridge.url).status_code, 401)
            self.assertEqual(
                httpx2.post(
                    scene.bridge.url,
                    headers={"Authorization": f"Bearer {scene.info['token']}", "Origin": "https://foreign.invalid"},
                    json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
                ).status_code,
                403,
            )
            initial = scene.data("get_scene")
            revision, generation = initial["revision"], initial["model_generation"]
            self.assertEqual(len(initial["objects"]), 1)
            batch = scene.data(
                "apply_edits",
                {
                    "expected_revision": revision,
                    "edits": [
                        {
                            "operation": "add_primitive",
                            "kind": "box",
                            "properties": {"name": "Batch box", "position": [2, 0, 1]},
                        },
                        {
                            "operation": "add_primitive",
                            "kind": "sphere",
                            "properties": {"name": "Batch sphere", "position": [4, 0, 1]},
                        },
                    ],
                },
            )
            self.assertEqual(len(batch["changed_object_ids"]), 2)
            self.assertEqual(batch["model_generation"], generation + 1)
            self.assertEqual(scene.editor.model.body_count, 3)
            for _ in range(4):
                scene.pump()
            state = scene.data("get_scene")
            self.assertEqual(len(state["objects"]), 3)
            self.assertEqual(scene.editor.rebuild_count, generation + 1)
            first = batch["changed_object_ids"][0]
            before = scene.editor.objects
            for edits, expected_revision in (
                ([], state["revision"]),
                ([{"operation": "remove", "object_id": first}], revision),
                (
                    [
                        {"operation": "update", "object_id": first, "properties": {"name": "Must roll back"}},
                        {"operation": "remove", "object_id": "missing"},
                    ],
                    state["revision"],
                ),
            ):
                self.assertTrue(
                    scene.call("apply_edits", {"edits": edits, "expected_revision": expected_revision}).is_error
                )
                self.assertEqual(scene.editor.objects, before)
                self.assertEqual(scene.editor.revision, state["revision"])
                self.assertEqual(scene.editor.rebuild_count, generation + 1)
            scene.editor.begin_edit()
            try:
                self.assertTrue(
                    scene.call(
                        "apply_edits",
                        {
                            "edits": [{"operation": "remove", "object_id": first}],
                            "expected_revision": state["revision"],
                        },
                    ).is_error
                )
            finally:
                scene.editor.end_edit()
            self.assertEqual(scene.editor.objects, before)
            undone = scene.data("undo_redo", {"direction": "undo", "expected_revision": state["revision"]})
            self.assertEqual(scene.editor.model.body_count, 1)
            scene.data("undo_redo", {"direction": "redo", "expected_revision": undone["revision"]})
            self.assertEqual(scene.editor.model.body_count, 3)
            page = scene.data("get_scene", {"offset": 1, "limit": 1})
            self.assertEqual(page["total_objects"], 3)
            self.assertEqual(len(page["objects"]), 1)
            changes = [
                {
                    "operation": "update",
                    "object_id": first,
                    "properties": {"position": [2, 0, 1.5], "color": [0.25, 0.5, 0.75]},
                },
                {"operation": "remove", "object_id": batch["changed_object_ids"][1]},
            ]
            mixed = scene.data(
                "apply_edits",
                {"expected_revision": page["revision"], "edits": changes},
            )
            self.assertEqual(scene.editor.model.body_count, 2)
            np.testing.assert_allclose(scene.editor.get_pose(first)[0], (2, 0, 1.5))
            self.assertEqual(scene.editor.get_object(first).color, (0.25, 0.5, 0.75))
            undone = scene.data("undo_redo", {"direction": "undo", "expected_revision": mixed["revision"]})
            changes[0]["properties"]["color"] = [0.75, 0.5, 0.25]
            scene.data("apply_edits", {"expected_revision": undone["revision"], "edits": changes})
            self.assertFalse(scene.editor.can_redo)
            self.assertEqual(scene.editor.model.body_count, 2)
            self.assertEqual(scene.editor.get_object(first).color, (0.75, 0.5, 0.25))
            revision, document = scene.editor.revision, scene.editor.objects
            cancelled = scene.client.call(
                "apply_edits",
                {
                    "expected_revision": revision,
                    "edits": [{"operation": "add_primitive", "kind": "box", "properties": {"name": "Cancelled"}}],
                },
            )
            # Wait for actual receipt, then for SDK cancellation propagation,
            # before allowing the owning thread to drain a scene command.
            deadline = time.monotonic() + 5
            while scene.bridge._queue.empty() and time.monotonic() < deadline:
                time.sleep(0.001)
            with scene.bridge._queue.mutex:
                self.assertTrue(scene.bridge._queue.queue)
                request = scene.bridge._queue.queue[0]
            cancelled.cancel()
            self.assertTrue(request.cancelled.wait(5))
            for _ in range(4):
                scene.pump()
            self.assertEqual(scene.editor.objects, document)
            self.assertEqual(scene.editor.revision, revision)
            for tool, arguments in (
                ("set_camera", {"position": [3, -4, 2], "pitch": -20, "yaw": 125}),
                ("set_playing", {"playing": False}),
                ("step_simulation", {"frames": 2}),
                ("get_viewport", {}),
            ):
                self.assertTrue(scene.call(tool, arguments).is_error)
            if importlib.util.find_spec("pxr") is not None:
                path = Path(scene.directory.name) / "scene.usda"
                scene.data("save_scene", {"path": str(path)})
                contents = path.read_bytes()
                self.assertTrue(contents.startswith(b"#usda"))
                self.assertTrue(scene.call("save_scene", {"path": str(path)}).is_error)
                self.assertEqual(path.read_bytes(), contents)
                invalid = Path(scene.directory.name) / "unsupported.usda"
                invalid.write_text('#usda 1.0\ndef TetMesh "Unsupported" {}\n')
                model = scene.editor.model
                state = scene.data("get_scene")
                pending = scene.data(
                    "apply_edits",
                    {
                        "expected_revision": state["revision"],
                        "edits": [{"operation": "add_usd", "source": str(invalid)}],
                    },
                )
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    state = scene.data("get_scene")
                    if state["error"]:
                        break
                self.assertIsNotNone(state["error"])
                self.assertTrue(state["pending"])
                self.assertIs(scene.editor.model, model)
                scene.data("undo_redo", {"direction": "undo", "expected_revision": pending["revision"]})
                self.assertFalse(scene.editor.pending)
                self.assertEqual(scene.editor.model.body_count, 2)
            scene.client.close()
            address = urlsplit(scene.bridge.url)
            scene.bridge.close()
            scene.bridge.close()
            self.assertFalse(scene.bridge.info_file.exists())
            with self.assertRaises(OSError):
                socket.create_connection((address.hostname, address.port), timeout=0.2)


@unittest.skipUnless(_HAS_MCP, "Requires the optional MCP SDK and HTTP server")
class TestSceneAuthoringMCPViewport(unittest.TestCase):
    def test_owner_thread_pixels_camera_and_paused_steps(self):
        """Capture real PNG pixels on the GL owner and execute each requested step once."""
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
            viewer = _make_headless_viewer_gl_or_skip(self, width=640, height=480)
            self.addCleanup(viewer.close)
            viewer.gui = ViewerGui(viewer, viewer.renderer.window, workspace_enabled=True)
            viewer.gui.ui.io.set_ini_filename("")
            viewer._paused = True
            scene = _SceneSession(self, viewer, native=True)
            history = viewer.gui.workspace.history
            history.set_recording(True)
            scene.pump()
            scene.data("set_camera", {"position": [3, -4, 2], "pitch": -20, "yaw": 125})
            np.testing.assert_allclose(viewer.camera.pos, (3, -4, 2))
            scene.data("set_playing", {"playing": True})
            self.assertFalse(viewer.is_paused())
            self.assertTrue(scene.call("step_simulation", {"frames": 1}).is_error)
            scene.data("set_playing", {"playing": False})
            before = scene.editor.sim_time
            scene.data("step_simulation", {"frames": 2})
            self.assertAlmostEqual(scene.editor.sim_time - before, scene.editor.frame_dt * 2)
            history.set_recording(False)
            scene.data("step_simulation", {"frames": 2})
            self.assertAlmostEqual(scene.editor.sim_time - before, scene.editor.frame_dt * 4)
            live_time = scene.editor.sim_time
            live_position = scene.editor.get_pose(scene.editor.objects[0].id)[0]
            history.select(0)
            replay = scene.data("get_scene")
            self.assertTrue(replay["history_replaying"])
            self.assertAlmostEqual(replay["sim_time"], live_time)
            self.assertAlmostEqual(replay["display_time"], history.frames[0].time)
            np.testing.assert_allclose(replay["objects"][0]["live_position"], live_position)
            self.assertTrue(
                scene.call(
                    "apply_edits",
                    {
                        "expected_revision": replay["revision"],
                        "edits": [{"operation": "remove", "object_id": replay["objects"][0]["id"]}],
                    },
                ).is_error
            )
            owner, captures = threading.get_ident(), []
            get_frame = viewer.get_frame

            def capture(*args, **kwargs):
                self.assertEqual(threading.get_ident(), owner)
                frame = get_frame(*args, **kwargs)
                captures.append(frame.numpy().copy())
                return frame

            revision = scene.editor.revision
            with mock.patch.object(viewer, "get_frame", side_effect=capture):
                result = scene.call("get_viewport", {"max_size": 640, "include_ui": False})
                self.assertFalse(result.is_error, str(result.content))
                self.assertEqual(len(captures), 1)
                image = next(content for content in result.content if content.type == "image")
                self.assertEqual(image.mime_type, "image/png")
                from pyglet import image as pyglet_image

                decoded = pyglet_image.load("viewport.png", file=BytesIO(base64.b64decode(image.data)))
                pixels = np.frombuffer(decoded.get_data("RGB", -decoded.width * 3), dtype=np.uint8).reshape(
                    decoded.height, decoded.width, 3
                )
                np.testing.assert_array_equal(pixels, captures[0])
                self.assertGreater(np.ptp(pixels), 0)
                self.assertEqual((result.structured_content["width"], result.structured_content["height"]), (640, 480))
                self.assertAlmostEqual(result.structured_content["sim_time"], live_time)
                self.assertAlmostEqual(result.structured_content["display_time"], history.frames[0].time)
                self.assertEqual(scene.editor.revision, revision)
                self.assertAlmostEqual(scene.editor.sim_time, live_time)
            history.select(len(history.frames) - 1)
