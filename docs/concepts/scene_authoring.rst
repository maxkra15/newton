.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. _scene-authoring:

Scene Authoring
===============

.. experimental::

   :class:`newton.viewer.SceneEditor` and the ``scene_editor`` application are
   experimental. Their authoring API and saved editor metadata may change.

The scene editor combines the Newton workspace with an explicitly owned XPBD
simulation. Added objects participate in physics. It supports rigid primitives
and local USD assets, including packages browsed from NVIDIA's SimReady catalog.

Launch and edit
---------------

The ``examples`` extra includes the native viewer and USD import dependencies:

.. code-block:: console

   uv run --extra examples -m newton.examples scene_editor
   uv run --extra examples -m newton.examples scene_editor --empty
   uv run --extra examples -m newton.examples scene_editor --scene scene.usda

The left panel has three tabs:

* **Create** adds boxes, spheres, capsules, and cylinders. Click a card to place
  an object near the viewport center, or drag it onto the ground plane. A
  placement outline follows the pointer. Scale, color, and Dynamic configure
  new primitives. **Import USD** and local file drops add USD assets.
* **Assets** searches the SimReady catalog. Select a card to inspect its source
  and available license information. Drag a card into the viewport,
  double-click it, or choose **Place in scene** to download and import it.
* **Scene** lists authored objects and filters them by name. Select an object
  here or in the viewport. Its context menu offers Duplicate and Delete.

The right panel edits the selected object's name, position, rotation, dimensions,
color, visibility, friction, mass, and Dynamic setting. Move and Rotate handles
edit its live pose directly in the viewport. A handle gesture pauses simulation
temporarily, then restores its previous playback state. **Visible** controls
rendering; collision remains enabled. A fixed object remains a physical collider.

The authoring toolbar provides Open, Save, Undo, Redo, and Reset. Undo and Redo
operate on authored edits; they do not rewind unrelated simulation motion.
Undoing a pose edit restores its previous authored reset pose, which can differ
from the object's live position immediately before the edit.
Reset restores the authored poses. Save writes the authored scene, rather than
a recording of its current simulated state. Open restores an editor document;
opening an ordinary USD file imports it as an asset. Press **H** to hide or show
the workspace, and use the timeline controls to pause or advance physics.
Return from history replay to the live frame before authoring.
The workspace's **Restart example** action also resets physics while retaining
the authored objects, rather than recreating the initial demonstration.

SimReady packages and offline use
---------------------------------

The browser reads the public
`SimReady submission registry <https://github.com/NVIDIA-Omniverse/simready-central/blob/main/submissions/nvidia/simready-assets>`_.
Each registered asset names a pinned Hugging Face revision. Only visible
thumbnails are requested; a full package is downloaded when placement is
requested. Downloads keep the package's relative paths, referenced files,
metadata, and license files, and verify the package before importing it.
Review the source package's license for its permitted uses.

Catalog loading, cache verification, thumbnails, and package transfers run in
background workers. Native texture uploads happen on the UI thread, with a
bounded thumbnail cache. Downloaded packages use Newton's asset cache. Set
``HF_HUB_OFFLINE=1`` to prevent network requests; previously cached catalog
entries and complete packages remain available. A missing package then reports
an error rather than silently importing an incomplete asset. Local USD import
does not require the catalog or a Hugging Face account.

Fast edits and structural changes
---------------------------------

Pose, color, visibility, and friction edits update existing Model and State
buffers. Primitive dimensions, USD scale, mass, motion type, and object
additions or removals require a replacement runtime. A drag groups these
structural edits into one commit on release. Its outline is transient viewer
geometry and does not rerun ``ModelBuilder`` for each pointer movement.

USD preparation runs in a worker and caches read-only builder templates. At a
safe frame boundary, the editor constructs a candidate Model, States, Control,
collision pipeline, and XPBD solver before adopting them. Failed compilation
keeps the previous simulation available and displays the error. Stable authored
IDs identify objects across model replacement; compiled body indices must be
resolved again after a commit.
The current simulation continues during downloads and background preparation.

This ownership boundary is deliberate. The editor does not mutate arbitrary
examples' private solvers or captured CUDA graphs. Other solvers, replicated
training environments, deformable authoring, and editing an existing Isaac Lab
simulation are outside this application's supported scope.

The design follows the distinction between inexpensive live edits and explicit
recompilation found in
`RoboSimStudio's parameter tiers <https://github.com/robosimstudio/robosimstudio/blob/d474a18b0b70c8f8fae3bbd8ce43a8456e57969b/robosimstudio/specs/params.py#L26-L33>`_.
`mjlab's model-expansion path <https://github.com/mujocolab/mjlab/blob/dcdf6ce519abfd2ea45e9503ae6fb650b75bb869/src/mjlab/sim/sim.py#L441-L459>`_
also shows why replacing simulation arrays requires an owner that can refresh
dependent bridges, sensors, and graphs. Newton's editor owns its full runtime
and makes no assumptions about another application's graph lifecycle.

USD interchange and limits
--------------------------

Primitive dimensions and positions are in metres, mass is in kilograms, and
rotations are XYZW quaternions in the Python API. Capsule height includes both
hemispheres. USD scale is multiplicative in the editor's Z-up frame. Native USD
import preserves supported rigid bodies, joints, colliders, and visuals.
Geometry without an authored body is promoted to a rigid prop with collision
geometry. SimReady backend variants select Newton when available and disable
alternative backend variants in an anonymous session layer; source files stay
unchanged.

Assets containing authored rigid-body, joint, or mass data must declare
``metersPerUnit=1`` and ``kilogramsPerUnit=1``. Geometry without authored physics
can be converted from other stage length units. Particle and deformable
simulation, legacy ``TetMesh`` simulation, and ``resetXformStack`` are rejected.
Harmless grasp-marker curves and point annotations are permitted. Dynamic
planes and heightfields are unsupported.

Export uses standard OpenUSD geometry and physics schemas, with editor metadata
for stable IDs and authored values. USD assets remain references to their
source packages, so retain those packages when moving a saved scene. White
asset tint preserves source materials. Exporting a nonwhite tint supports
``displayColor`` and scalar ``UsdPreviewSurface`` color; connected color
textures and other shader graphs require white tint and otherwise raise an
error. The native viewer remains subject to its existing material and texture
support.

Embedding the experimental API
------------------------------

Import the editor through its public module:

.. testcode:: scene-authoring

   from newton.viewer import SceneEditor, ViewerNull

   viewer = ViewerNull()
   editor = SceneEditor(viewer, device="cpu")
   box = editor.add_primitive("box", position=(0, 0, 1), size=(0.5, 0.5, 0.5))
   editor.before_frame()
   editor.begin_edit()
   editor.update_object(box, size=(0.7, 0.5, 0.5))
   editor.end_edit()
   editor.before_frame()
   editor.step()
   editor.render()
   editor.close()
   viewer.close()

Call :meth:`~newton.viewer.SceneEditor.before_frame` before workspace binding,
stepping, or rendering, including while paused. Newton's example runner calls
it automatically for an example exposing ``scene_editor``. Use
:meth:`~newton.viewer.SceneEditor.begin_edit` and
:meth:`~newton.viewer.SceneEditor.end_edit` to group a gesture, or disable
``auto_apply`` and call :meth:`~newton.viewer.SceneEditor.request_commit` for an
explicit structural commit. Call :meth:`~newton.viewer.SceneEditor.close` when
the application is disposed to stop queued preparation.
