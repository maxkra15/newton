# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build and edit a physical scene with primitives, USD, and SimReady assets.

Command: python -m newton.examples scene_editor
"""

import subprocess
from pathlib import Path

import newton.examples
from newton.viewer import SceneEditor


class Example(SceneEditor):
    """Own physics explicitly so the editor can safely replace compiled scenes."""

    @staticmethod
    def create_parser():
        """Expose the same scene options to the CLI and workspace example browser."""
        parser = newton.examples.create_parser()
        parser.add_argument("--scene", help="Open an authored USD scene or import a USD asset.")
        parser.add_argument("--empty", action="store_true", help="Start with only the ground plane.")
        return parser

    def __init__(self, viewer, args):
        super().__init__(viewer)
        if args.scene:
            self.load_usd(args.scene)
        elif not args.empty:
            self.add_primitive(
                "box", name="Plinth", position=(0, 0, 0.2), size=(2, 2, 0.4), color=(0.28, 0.33, 0.39), dynamic=False
            )
            self.add_primitive(
                "box", name="Blue box", position=(-0.4, 0, 1.1), size=(0.6, 0.6, 0.6), color=(0.22, 0.6, 0.85)
            )
            self.add_primitive(
                "sphere", name="Coral sphere", position=(0.45, 0.1, 1.5), size=(0.5, 0.5, 0.5), color=(0.95, 0.43, 0.32)
            )
        self.commit()
        viewer.set_camera(pos=(5.0, -7.0, 4.5), pitch=-23.0, yaw=125.0)
        window = getattr(getattr(viewer, "renderer", None), "window", None)
        if window is not None:
            revision = newton.__version__
            root = Path(__file__).resolve().parents[3]
            if (root / ".git").exists():
                try:
                    revision = subprocess.check_output(
                        ["git", "-C", str(root), "describe", "--always", "--dirty", "--abbrev=12", "--exclude=*"],
                        text=True,
                        stderr=subprocess.DEVNULL,
                        timeout=1,
                    ).strip()
                except (OSError, subprocess.SubprocessError):
                    pass
            window.set_caption(f"Newton Scene Editor [{revision}]")


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    try:
        newton.examples.run(example, args)
    finally:
        example.close()
