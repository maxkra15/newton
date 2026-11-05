# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""An opt-in, document-owned rigid scene with transactional runtime replacement."""

from __future__ import annotations

import dataclasses
import enum
import logging
import math
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from ..geometry import GeoType, ShapeFlags
from ..sim import BodyFlags, CollisionPipeline, JointType, ModelBuilder, eval_ik
from ..sim.collide import compute_shape_aabbs
from ..solvers.xpbd import SolverXPBD
from .authoring_usd import NotAuthoringSceneError, load_scene, prepare_usd, save_scene

if TYPE_CHECKING:
    from ..sim import Contacts, Control, Model, State
    from .viewer import ViewerBase

logger = logging.getLogger(__name__)


@wp.kernel
def _move_bodies(
    start: int,
    delta: wp.transform,
    initial_delta: wp.transform,
    initial: wp.array[wp.transform],
    poses: wp.array[wp.transform],
    other: wp.array[wp.transform],
    velocity: wp.array[wp.spatial_vector],
    other_velocity: wp.array[wp.spatial_vector],
    clear_velocity: bool,
):
    index = start + wp.tid()
    pose = wp.transform_multiply(delta, poses[index])
    poses[index] = pose
    other[index] = pose
    if clear_velocity:
        velocity[index] = wp.spatial_vector()
        other_velocity[index] = wp.spatial_vector()
    initial[index] = wp.transform_multiply(initial_delta, initial[index])


@wp.kernel
def _move_world_shapes(
    start: int,
    delta: wp.transform,
    bodies: wp.array[int],
    transforms: wp.array[wp.transform],
):
    index = start + wp.tid()
    if bodies[index] == -1:
        transforms[index] = wp.transform_multiply(delta, transforms[index])


@wp.kernel
def _move_world_anchors(
    start: int,
    delta: wp.transform,
    parents: wp.array[int],
    kinds: wp.array[int],
    anchors: wp.array[wp.transform],
):
    index = start + wp.tid()
    if parents[index] == -1 and kinds[index] != int(JointType.FREE):
        anchors[index] = wp.transform_multiply(delta, anchors[index])


@wp.kernel
def _appearance(
    start: int,
    tint: wp.vec3,
    visible: bool,
    friction: float,
    base_colors: wp.array[wp.vec3],
    base_opacity: wp.array[float],
    colors: wp.array[wp.vec3],
    opacity: wp.array[float],
    coefficients: wp.array[float],
):
    index = start + wp.tid()
    colors[index] = wp.cw_mul(base_colors[index], tint)
    opacity[index] = wp.where(visible, base_opacity[index], 0.0)
    coefficients[index] = friction


@wp.kernel
def _copy_bodies(
    old_indices: wp.array[int],
    new_indices: wp.array[int],
    deltas: wp.array[wp.transform],
    clear_velocity: wp.array[bool],
    old_q: wp.array[wp.transform],
    old_qd: wp.array[wp.spatial_vector],
    q: wp.array[wp.transform],
    qd: wp.array[wp.spatial_vector],
    other_q: wp.array[wp.transform],
    other_qd: wp.array[wp.spatial_vector],
):
    thread = wp.tid()
    old = old_indices[thread]
    new = new_indices[thread]
    pose = wp.transform_multiply(deltas[thread], old_q[old])
    velocity = old_qd[old]
    if clear_velocity[thread]:
        velocity = wp.spatial_vector()
    q[new] = pose
    qd[new] = velocity
    other_q[new] = pose
    other_qd[new] = velocity


@dataclasses.dataclass(frozen=True)
class _Binding:
    bodies: tuple[int, int]
    shapes: tuple[int, int]
    joints: tuple[int, int]
    reference: tuple[float, ...] | None
    follows_body: bool


@dataclasses.dataclass
class _Runtime:
    model: Model
    state_0: State
    state_1: State
    control: Control
    solver: SolverXPBD
    pipeline: CollisionPipeline
    contacts: Contacts
    bindings: dict[str, _Binding]
    objects: dict[str, Any]
    base_colors: wp.array[wp.vec3]
    base_opacity: wp.array[float]
    base_friction: wp.array[float]


class SceneEditor:
    """Author a rigid scene while owning its simulation and compiled resources.

    This entire class is **experimental**. It is an opt-in XPBD scene host,
    rather than a way to mutate another application's private solver or graphs.
    Positions and dimensions use metres, rotations use XYZW quaternions, and
    masses use kilograms. Pose, color, visibility, and friction edits retain
    Model/State storage. Add/remove, geometry, mass, and motion-type changes
    create one candidate runtime when committed, never on every drag sample.

    Use :meth:`before_frame` before stepping or rendering, including when
    paused. The example runner does this automatically. Exported documents use
    standard OpenUSD geometry and physics schemas. USD conversion is optional
    and loaded only when importing/exporting an asset.
    """

    class Shape(enum.Enum):
        """Geometry categories supported by the experimental authoring scene."""

        BOX = "box"
        SPHERE = "sphere"
        CAPSULE = "capsule"
        CYLINDER = "cylinder"
        USD = "usd"

    @dataclasses.dataclass(frozen=True)
    class Object:
        """Immutable authored values; live simulation poses remain in State."""

        id: str
        """Stable document identity, retained across runtime replacements."""
        name: str
        """Display name, independent of physical body labels."""
        kind: SceneEditor.Shape
        """Primitive type or a referenced USD asset."""
        position: tuple[float, float, float] = (0.0, 0.0, 1.0)
        """Authored reset placement in the Z-up scene [m]."""
        rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
        """Authored reset orientation as an XYZW quaternion."""
        size: tuple[float, float, float] = (1.0, 1.0, 1.0)
        """Full primitive XYZ dimensions [m], or multiplicative USD scale."""
        color: tuple[float, float, float] = (0.55, 0.72, 0.85)
        """RGB tint in [0, 1]; white preserves a USD asset's original colors."""
        mass: float = 1.0
        """Primitive mass or total USD rigid-body mass [kg]."""
        friction: float = 0.6
        """Nonnegative surface friction coefficient."""
        dynamic: bool = True
        """Whether physics advances the object; false makes it kinematic."""
        source: str | None = None
        """Absolute root USD path, present only for USD assets."""
        visible: bool = True
        """Rendering visibility, independent of collision participation."""
        overrides: tuple[str, ...] = ()
        """Explicit USD mass, friction, or dynamic overrides; otherwise retain native values."""

    def __init__(self, viewer: ViewerBase, *, device: str | wp.Device | None = None) -> None:
        """Create an empty document without compiling a model or starting network work.

        Args:
            viewer: Viewer receiving the owned model, state, and interaction forces.
            device: Simulation device; defaults to the current Warp device.
        """
        self.viewer = viewer
        self.device = wp.get_device(device)
        self.scene_editor = self
        self.frame_dt = 1.0 / 60.0
        self.sim_time = 0.0
        self.sim_substeps = 4
        self.model = self.state_0 = self.state_1 = self.control = self.solver = self.contacts = None
        self.collision_pipeline = None
        self.selected_id: str | None = None
        self.auto_apply = True
        self.revision = 0
        self.rebuild_count = 0
        self.last_build_ms = 0.0
        self.error: str | None = None
        self.status = "Add a primitive or import a USD asset."
        self._objects: tuple[SceneEditor.Object, ...] = ()
        self._runtime: _Runtime | None = None
        self._topology_dirty = True
        self._fast_pending: dict[str, SceneEditor.Object] = {}
        self._placement_requests: set[str] = set()
        self._undo: list[tuple[SceneEditor.Object, ...]] = []
        self._redo: list[tuple[SceneEditor.Object, ...]] = []
        self._edit_before: tuple[SceneEditor.Object, ...] | None = None
        self._requested_commit = False
        self._attempted_revision = -1
        self._templates: dict[tuple[Any, ...], ModelBuilder] = {}
        self._preparing: dict[tuple[Any, ...], Future] = {}
        self._executor: ThreadPoolExecutor | None = None
        self._closed = False
        self._preview: tuple | None = None
        self._preview_last: tuple | None = None
        self._preview_buffers = None

    @property
    def objects(self) -> tuple[Object, ...]:
        """Return authored objects in stable document order."""
        return self._objects

    @property
    def pending(self) -> bool:
        """Report document changes awaiting publication to the physical scene."""
        return self._topology_dirty or bool(self._fast_pending)

    @property
    def busy(self) -> bool:
        """Report outstanding background USD preparation."""
        return any(not future.done() for future in self._preparing.values())

    @property
    def can_undo(self) -> bool:
        """Report whether an authored edit can be undone."""
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        """Report whether an authored edit can be redone."""
        return bool(self._redo)

    def get_object(self, object_id: str) -> Object:
        """Resolve a stable document ID, rejecting stale compiled indices."""
        return next((obj for obj in self._objects if obj.id == object_id), None) or self._missing(object_id)

    @staticmethod
    def _missing(object_id: str):
        raise KeyError(f"Unknown scene object {object_id!r}")

    def select(self, object_id: str | None) -> None:
        """Select an authored object without changing simulation state."""
        if object_id is not None:
            self.get_object(object_id)
        self.selected_id = object_id

    @classmethod
    def _validate(cls, obj: Object) -> Object:
        position = cls._vector(obj.position, 3, "position")
        rotation = cls._vector(obj.rotation, 4, "rotation")
        norm = math.sqrt(sum(value * value for value in rotation))
        if norm < 1.0e-8:
            raise ValueError("Rotation must be a nonzero XYZW quaternion")
        rotation = tuple(value / norm for value in rotation)
        size = cls._vector(obj.size, 3, "size")
        color = cls._vector(obj.color, 3, "color")
        if any(value <= 0 for value in size):
            raise ValueError("Dimensions must be positive")
        if any(value < 0 or value > 1 for value in color):
            raise ValueError("Color components must be between zero and one")
        if not math.isfinite(obj.mass) or obj.mass <= 0:
            raise ValueError("Mass must be finite and positive")
        if not math.isfinite(obj.friction) or obj.friction < 0:
            raise ValueError("Friction must be finite and nonnegative")
        if obj.kind == cls.Shape.SPHERE and not np.allclose(size, size[0]):
            raise ValueError("A sphere requires equal X, Y, and Z diameters")
        if obj.kind in (cls.Shape.CAPSULE, cls.Shape.CYLINDER) and not math.isclose(size[0], size[1]):
            raise ValueError("Capsules and cylinders require equal X and Y diameters")
        if obj.kind == cls.Shape.CAPSULE and size[2] < size[0]:
            raise ValueError("Capsule total height must be at least its diameter")
        if not obj.name.strip():
            raise ValueError("Object name cannot be empty")
        overrides = tuple(sorted(set(obj.overrides)))
        if set(overrides) - {"mass", "friction", "dynamic"}:
            raise ValueError("Unknown USD physics override")
        if not isinstance(obj.dynamic, bool) or not isinstance(obj.visible, bool):
            raise ValueError("Dynamic and visible flags must be booleans")
        return dataclasses.replace(
            obj, position=position, rotation=rotation, size=size, color=color, overrides=overrides
        )

    @staticmethod
    def _vector(values, count: int, name: str) -> tuple:
        result = tuple(float(value) for value in values)
        if len(result) != count or not all(math.isfinite(value) for value in result):
            raise ValueError(f"{name} requires {count} finite components")
        return result

    def _remember(self) -> None:
        if self._edit_before is None:
            self._undo.append(self._objects)
            del self._undo[:-100]
        self._redo.clear()

    def _changed(self) -> None:
        self.revision += 1
        self.error = None

    def add_primitive(
        self,
        kind: Shape | str,
        *,
        position: tuple[float, float, float] = (0.0, 0.0, 1.0),
        size: tuple[float, float, float] = (1.0, 1.0, 1.0),
        name: str | None = None,
        color: tuple[float, float, float] = (0.55, 0.72, 0.85),
        dynamic: bool = True,
        mass: float = 1.0,
    ) -> str:
        """Stage a primitive without finalizing a model.

        Args:
            kind: Box, sphere, capsule, or cylinder category.
            position: Initial center in the Z-up scene [m].
            size: Full XYZ extent [m]. Capsule height includes both hemispheres.
            name: Display name, defaulting to the shape category.
            color: RGB color in [0, 1].
            dynamic: Whether physics advances the object; false makes it kinematic.
            mass: Total mass [kg].

        Returns:
            Stable object ID, valid across structural commits.
        """
        kind = self.Shape(kind)
        if kind == self.Shape.USD:
            raise ValueError("Use add_usd() for USD assets")
        obj = self._validate(
            self.Object(
                uuid.uuid4().hex,
                name or kind.value.title(),
                kind,
                position=position,
                size=size,
                color=color,
                dynamic=dynamic,
                mass=mass,
            )
        )
        self._remember()
        self._objects += (obj,)
        self.selected_id = obj.id
        self._topology_dirty = True
        self._changed()
        return obj.id

    def add_usd(
        self,
        source: str | Path,
        *,
        position: tuple[float, float, float] = (0.0, 0.0, 1.0),
        name: str | None = None,
    ) -> str:
        """Stage a local USD package; preparation runs independently of UI drawing.

        Args:
            source: Root USD path; referenced dependencies must be available locally.
            position: Initial root placement in the Z-up scene [m].
            name: Display name, defaulting to the source filename stem.

        Returns:
            Stable object ID. Import failures are reported by :attr:`error`.
        """
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        obj = self._validate(
            self.Object(
                uuid.uuid4().hex,
                name or path.stem,
                self.Shape.USD,
                position=position,
                color=(1.0, 1.0, 1.0),
                source=str(path),
            )
        )
        self._remember()
        self._objects += (obj,)
        self.selected_id = obj.id
        self._topology_dirty = True
        self._changed()
        self._schedule_template(obj)
        return obj.id

    def update_object(self, object_id: str, **fields) -> None:
        """Queue supported edits; transforms and appearance retain compiled storage.

        Args:
            object_id: Stable authored identity.
            **fields: Editable :class:`Object` values. ID, kind, source, and the
                internal override record cannot be changed. Size, mass, and
                dynamic require a structural commit. Placement edits target
                the live pose and also update the authored reset pose.
        """
        before = self.get_object(object_id)
        allowed = {field.name for field in dataclasses.fields(before)} - {"id", "kind", "source", "overrides"}
        if unknown := set(fields) - allowed:
            raise ValueError(f"Unsupported authored fields: {', '.join(sorted(unknown))}")
        after = self._validate(dataclasses.replace(before, **fields))
        if after.kind == self.Shape.USD:
            overrides = set(before.overrides) | (set(fields) & {"mass", "friction", "dynamic"})
            after = dataclasses.replace(after, overrides=tuple(sorted(overrides)))
        if after == before:
            return
        self._remember()
        self._objects = tuple(after if item.id == object_id else item for item in self._objects)
        if self._structure(before) != self._structure(after):
            self._topology_dirty = True
        else:
            self._fast_pending.setdefault(object_id, before)
        self._changed()

    def place_object(self, object_id: str) -> None:
        """Queue conservative, collision-free insertion of a newly staged object.

        Raise its complete native shape bounds along the scene's up axis above
        occupied footprints when committed. Explicit adds and USD loads retain
        their exact positions unless this experimental operation is requested.
        Infinite planes require explicit placement instead.

        Args:
            object_id: Stable ID of an added object that has not yet been committed.
        """
        self.get_object(object_id)
        if self._runtime is not None and object_id in self._runtime.bindings:
            raise ValueError("Automatic placement requires a newly added object")
        if object_id not in self._placement_requests:
            self._placement_requests.add(object_id)
            self._topology_dirty = True
            self._changed()

    def remove_object(self, object_id: str) -> None:
        """Stage object removal, retaining the current physical scene until commit."""
        self.get_object(object_id)
        self._remember()
        self._objects = tuple(obj for obj in self._objects if obj.id != object_id)
        self._fast_pending.pop(object_id, None)
        self._placement_requests.discard(object_id)
        if self.selected_id == object_id:
            self.selected_id = None
        self._topology_dirty = True
        self._changed()

    def duplicate_object(self, object_id: str) -> str:
        """Duplicate an object's authored data while sharing its immutable asset template."""
        obj = self.get_object(object_id)
        position, rotation = self.get_pose(object_id)
        duplicate = dataclasses.replace(
            obj,
            id=uuid.uuid4().hex,
            name=obj.name + " copy",
            position=(position[0] + 0.4, position[1] + 0.4, position[2]),
            rotation=rotation,
        )
        self._remember()
        self._objects += (duplicate,)
        self.selected_id = duplicate.id
        self._topology_dirty = True
        self._changed()
        return duplicate.id

    def begin_edit(self) -> None:
        """Start one undoable gesture; delay structural compilation until release."""
        if self._edit_before is None:
            self._edit_before = self._objects

    def end_edit(self) -> None:
        """Finish a gesture without compiling on the UI callback stack."""
        if self._edit_before is not None and self._edit_before != self._objects:
            self._undo.append(self._edit_before)
            del self._undo[:-100]
        self._edit_before = None

    def _restore_document(self, objects: tuple[Object, ...]) -> None:
        objects = self._resolved_defaults(objects)
        old = {obj.id: obj for obj in self._objects}
        new = {obj.id: obj for obj in objects}
        structural = old.keys() != new.keys() or any(
            self._structure(old[key]) != self._structure(new[key]) for key in old.keys() & new.keys()
        )
        self._objects = objects
        self._placement_requests.intersection_update(new)
        self._topology_dirty |= structural
        for key in old.keys() & new.keys():
            if old[key] != new[key]:
                self._fast_pending.setdefault(key, old[key])
        if self.selected_id not in new:
            self.selected_id = None
        self._changed()

    def undo(self) -> None:
        """Restore authored values, including reset poses, without rewinding unrelated physics."""
        self.end_edit()
        if self._undo:
            self._redo.append(self._objects)
            self._restore_document(self._undo.pop())

    def redo(self) -> None:
        """Restore an undone authored edit."""
        self.end_edit()
        if self._redo:
            self._undo.append(self._objects)
            self._restore_document(self._redo.pop())

    @staticmethod
    def _structure(obj: Object) -> tuple:
        return obj.kind, obj.size, obj.source, obj.mass, obj.dynamic

    def _template_key(self, obj: Object) -> tuple:
        path = Path(obj.source)
        stat = path.stat()
        return str(path), obj.size, stat.st_mtime_ns, stat.st_size

    @staticmethod
    def _prepare_template(source: str, size: tuple) -> ModelBuilder:
        with wp.ScopedDevice("cpu"):
            return prepare_usd(source, scale=size)

    def _schedule_template(self, obj: Object) -> None:
        key = self._template_key(obj)
        if key in self._templates or key in self._preparing:
            return
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="newton-usd-authoring")
        self._preparing[key] = self._executor.submit(self._prepare_template, obj.source, obj.size)

    def _trim_templates(self) -> None:
        # Retain every active source, but bound old scale variants after gestures.
        active = {(obj.source, obj.size) for obj in self._objects if obj.kind == self.Shape.USD}
        for key in list(self._templates):
            if len(self._templates) <= 32:
                break
            if key[:2] not in active:
                del self._templates[key]

    def request_commit(self) -> None:
        """Request a structural commit at the next safe frame boundary, including retry."""
        self._requested_commit = True
        self._attempted_revision = -1

    def before_frame(self) -> None:
        """Publish edits before workspace/history binding, even while physics is paused."""
        if self._closed:
            return
        requested = (
            {(obj.source, obj.size) for obj in self._objects if obj.kind == self.Shape.USD} if self._preparing else ()
        )
        for key, future in list(self._preparing.items()):
            if key[:2] not in requested:
                future.cancel()
                del self._preparing[key]
                continue
            if not future.done():
                continue
            try:
                self._templates[key] = future.result()
                self._objects = self._resolved_defaults(self._objects)
            except Exception as exc:
                self.error = str(exc)
                self._attempted_revision = self.revision
            del self._preparing[key]
        if self._fast_pending:
            self._apply_fast_edits()
        if self._topology_dirty and self._edit_before is None and (self.auto_apply or self._requested_commit):
            if self._attempted_revision == self.revision:
                return
            try:
                for obj in self._objects:
                    if obj.kind == self.Shape.USD:
                        self._schedule_template(obj)
                if self.busy:
                    return
                self._attempted_revision = self.revision
                self.commit()
            except Exception as exc:
                self._attempted_revision = self.revision
                self.error = str(exc)
                self.status = "Could not apply scene changes. The previous simulation is retained."
                logger.error("Could not compile authored scene: %s", exc)

    def _template(self, obj: Object) -> ModelBuilder:
        key = self._template_key(obj)
        if key not in self._templates:
            future = self._preparing.pop(key, None)
            self._templates[key] = (
                future.result() if future is not None else self._prepare_template(obj.source, obj.size)
            )
            self._objects = self._resolved_defaults(self._objects)
        return self._templates[key]

    def _resolved_defaults(self, objects: tuple[Object, ...]) -> tuple[Object, ...]:
        result = []
        for original in objects:
            obj = original
            template = next(
                (
                    template
                    for key, template in reversed(self._templates.items())
                    if obj.kind == self.Shape.USD and key[:2] == (obj.source, obj.size)
                ),
                None,
            )
            if template is not None:
                mass = sum(template.body_mass)
                defaults = {}
                if mass > 0 and "mass" not in obj.overrides:
                    defaults["mass"] = mass
                if template.body_count and "dynamic" not in obj.overrides:
                    defaults["dynamic"] = any(flag & int(BodyFlags.DYNAMIC) for flag in template.body_flags)
                elif not template.body_count and "dynamic" not in obj.overrides:
                    defaults["dynamic"] = not any(
                        kind in (GeoType.PLANE, GeoType.HFIELD) for kind in template.shape_type
                    )
                if template.shape_count and "friction" not in obj.overrides:
                    defaults["friction"] = float(template.shape_material_mu[0])
                obj = dataclasses.replace(obj, **defaults)
            result.append(obj)
        return tuple(result)

    def _add_primitive(self, builder: ModelBuilder, obj: Object) -> None:
        x, y, z = obj.size
        radius = x * 0.5
        if obj.kind == self.Shape.BOX:
            volume = x * y * z
        elif obj.kind == self.Shape.SPHERE:
            volume = 4.0 * math.pi * radius**3 / 3.0
        elif obj.kind == self.Shape.CYLINDER:
            volume = math.pi * radius**2 * z
        else:
            volume = math.pi * radius**2 * (z - x) + 4.0 * math.pi * radius**3 / 3.0
        body = builder.add_body(
            xform=wp.transform(obj.position, obj.rotation), label=f"objects/{obj.id}/body", is_kinematic=not obj.dynamic
        )
        cfg = ModelBuilder.ShapeConfig(density=obj.mass / volume, mu=obj.friction)
        common = {"cfg": cfg, "color": (1.0, 1.0, 1.0), "label": f"objects/{obj.id}/shape"}
        if obj.kind == self.Shape.BOX:
            builder.add_shape_box(body, hx=x * 0.5, hy=y * 0.5, hz=z * 0.5, **common)
        elif obj.kind == self.Shape.SPHERE:
            builder.add_shape_sphere(body, radius=radius, **common)
        elif obj.kind == self.Shape.CYLINDER:
            builder.add_shape_cylinder(body, radius=radius, half_height=z * 0.5, **common)
        else:
            builder.add_shape_capsule(body, radius=radius, half_height=(z - x) * 0.5, **common)

    def _add_asset(self, builder: ModelBuilder, obj: Object) -> None:
        template = self._template(obj)
        obj = self.get_object(obj.id)
        if template.particle_count:
            raise ValueError("Scene authoring currently supports rigid USD assets, not particles or deformables")
        start = builder.body_count
        if not template.body_count:
            if not template.shape_count:
                raise ValueError("The USD asset contains no supported rigid geometry")
            body = builder.add_body(
                xform=wp.transform(obj.position, obj.rotation),
                label=f"objects/{obj.id}/body",
                is_kinematic=not obj.dynamic,
            )
            for index in range(template.shape_count):
                if template.shape_type[index] in (GeoType.PLANE, GeoType.HFIELD) and obj.dynamic:
                    raise ValueError("Planes and heightfields must be imported as static scene geometry")
                flags = template.shape_flags[index]
                cfg = ModelBuilder.ShapeConfig(
                    mu=obj.friction,
                    has_shape_collision=True,
                    has_particle_collision=bool(flags & int(ShapeFlags.COLLIDE_PARTICLES)),
                    is_visible=bool(flags & int(ShapeFlags.VISIBLE)),
                )
                builder.add_shape(
                    body=body,
                    type=template.shape_type[index],
                    xform=template.shape_transform[index],
                    scale=template.shape_scale[index],
                    src=template.shape_source[index],
                    cfg=cfg,
                    color=template.shape_color[index],
                    opacity=template.shape_opacity[index],
                    label=f"objects/{obj.id}/shape_{index}",
                )
            mass = builder.body_mass[body]
            if mass <= 0:
                if not obj.dynamic:
                    return
                raise ValueError("The USD asset has no finite-volume collision geometry for a rigid prop")
            ratio = obj.mass / mass
            builder.body_mass[body] *= ratio
            builder.body_inertia[body] = np.asarray(builder.body_inertia[body]) * ratio
            return
        builder.add_builder(template, xform=wp.transform(obj.position, obj.rotation), label_prefix=f"objects/{obj.id}/")
        for index in range(start, builder.body_count):
            if "dynamic" in obj.overrides:
                builder.body_flags[index] = int(BodyFlags.DYNAMIC if obj.dynamic else BodyFlags.KINEMATIC)
        if "mass" in obj.overrides:
            previous = sum(builder.body_mass[start:])
            if previous <= 0:
                raise ValueError("This USD has no positive rigid-body mass to override")
            ratio = obj.mass / previous
            for index in range(start, builder.body_count):
                builder.body_mass[index] *= ratio
                builder.body_inertia[index] = np.asarray(builder.body_inertia[index]) * ratio

    def _build(self) -> _Runtime:
        builder = ModelBuilder()
        builder.add_ground_plane(cfg=ModelBuilder.ShapeConfig(mu=0.7), color=(0.28, 0.31, 0.34))
        bindings = {}
        for obj in self._objects:
            b, s, j = builder.body_count, builder.shape_count, builder.joint_count
            if obj.kind == self.Shape.USD:
                self._add_asset(builder, obj)
            else:
                self._add_primitive(builder, obj)
            reference = None
            follows = False
            if builder.body_count > b:
                placement = wp.transform(obj.position, obj.rotation)
                reference = tuple(wp.transform_multiply(wp.transform_inverse(placement), builder.body_q[b]))
                roots = [k for k in range(j, builder.joint_count) if builder.joint_parent[k] == -1]
                follows = len(roots) == 1 and builder.joint_type[roots[0]] == JointType.FREE
                follows &= all(body >= b for body in builder.shape_body[s:])
            bindings[obj.id] = _Binding(
                (b, builder.body_count), (s, builder.shape_count), (j, builder.joint_count), reference, follows
            )
        model = builder.finalize(device=self.device)
        runtime = _Runtime(
            model,
            model.state(),
            model.state(),
            model.control(),
            SolverXPBD(model, iterations=8),
            None,
            None,
            bindings,
            {obj.id: obj for obj in self._objects},
            wp.clone(model.shape_color),
            wp.clone(model.shape_opacity),
            wp.clone(model.shape_material_mu),
        )
        runtime.pipeline = CollisionPipeline(model)
        runtime.contacts = runtime.pipeline.contacts()
        for obj in self._objects:
            self._apply_appearance(runtime, obj)
        return runtime

    def _preserve_state(self, old: _Runtime, new: _Runtime) -> None:
        labels = {label: index for index, label in enumerate(old.model.body_label)}
        old_indices, new_indices, deltas, clear_velocity = [], [], [], []
        for object_id, obj in new.objects.items():
            previous_obj = old.objects.get(object_id)
            if previous_obj is None or (obj.kind, obj.source) != (previous_obj.kind, previous_obj.source):
                continue
            moved = (obj.position, obj.rotation) != (previous_obj.position, previous_obj.rotation)
            delta = wp.transform_identity()
            old_binding, new_binding = old.bindings[object_id], new.bindings[object_id]
            changed_pivot = (
                old_binding.follows_body and new_binding.follows_body and old_binding.reference != new_binding.reference
            )
            if moved or changed_pivot:
                position, rotation = self.get_pose(object_id)
                current = wp.transform(position, rotation)
                target = wp.transform(obj.position, obj.rotation) if moved else current
                if changed_pivot:
                    target = wp.transform_multiply(
                        target, wp.transform(new_binding.reference[:3], new_binding.reference[3:])
                    )
                    current = wp.transform_multiply(
                        current, wp.transform(old_binding.reference[:3], old_binding.reference[3:])
                    )
                delta = wp.transform_multiply(target, wp.transform_inverse(current))
            start, end = new.bindings[object_id].bodies
            for index in range(start, end):
                previous = labels.get(new.model.body_label[index])
                if previous is not None:
                    old_indices.append(previous)
                    new_indices.append(index)
                    deltas.append(delta)
                    clear_velocity.append(moved or not obj.dynamic)
        if old_indices:
            wp.launch(
                _copy_bodies,
                len(old_indices),
                inputs=[
                    wp.array(old_indices, dtype=int, device=self.device),
                    wp.array(new_indices, dtype=int, device=self.device),
                    wp.array(deltas, dtype=wp.transform, device=self.device),
                    wp.array(clear_velocity, dtype=bool, device=self.device),
                    old.state_0.body_q,
                    old.state_0.body_qd,
                    new.state_0.body_q,
                    new.state_0.body_qd,
                    new.state_1.body_q,
                    new.state_1.body_qd,
                ],
                device=self.device,
            )
            self._sync_coordinates(new)
        previous_joints = {name: index for index, name in enumerate(old.model.joint_label)}
        old_q, old_d = old.model.joint_q_start.numpy(), old.model.joint_qd_start.numpy()
        new_q, new_d = new.model.joint_q_start.numpy(), new.model.joint_qd_start.numpy()
        for object_id in new.objects:
            if object_id not in old.objects:
                continue
            for joint in range(*new.bindings[object_id].joints):
                previous = previous_joints.get(new.model.joint_label[joint])
                if previous is None:
                    continue
                for field in ("joint_f", "joint_act", "joint_target_q", "joint_target_qd"):
                    source, target = getattr(old.control, field, None), getattr(new.control, field, None)
                    if source is None or target is None:
                        continue
                    coordinate = field == "joint_target_q"
                    starts_old, starts_new = (old_q, new_q) if coordinate else (old_d, new_d)
                    count = int(starts_new[joint + 1] - starts_new[joint])
                    if count != int(starts_old[previous + 1] - starts_old[previous]):
                        raise ValueError("An unchanged joint changed control layout during compilation")
                    if count == 0:
                        continue
                    wp.copy(
                        target,
                        source,
                        dest_offset=int(starts_new[joint]),
                        src_offset=int(starts_old[previous]),
                        count=count,
                    )

    @staticmethod
    def _sync_coordinates(runtime: _Runtime) -> None:
        if runtime.model.joint_count:
            eval_ik(runtime.model, runtime.state_0, runtime.state_0.joint_q, runtime.state_0.joint_qd)
            eval_ik(runtime.model, runtime.state_1, runtime.state_1.joint_q, runtime.state_1.joint_qd)

    def _publish(self, runtime: _Runtime) -> None:
        self._runtime = runtime
        self.model, self.state_0, self.state_1 = runtime.model, runtime.state_0, runtime.state_1
        self.control, self.solver = runtime.control, runtime.solver
        self.collision_pipeline, self.contacts = runtime.pipeline, runtime.contacts

    def _place_candidate(self, runtime: _Runtime) -> None:
        requested = [obj.id for obj in self._objects if obj.id in self._placement_requests]
        if not requested:
            return
        model, pipeline, contacts = runtime.model, runtime.pipeline, runtime.contacts
        snapshots = []
        for poses in (runtime.state_0.body_q, model.body_q):
            wp.launch(
                compute_shape_aabbs,
                model.shape_count,
                inputs=[
                    poses,
                    model.shape_transform,
                    model.shape_body,
                    model.shape_type,
                    model.shape_scale,
                    model.shape_collision_radius,
                    model.shape_source_ptr,
                    model.shape_margin,
                    model.shape_gap,
                    model.shape_collision_aabb_lower,
                    model.shape_collision_aabb_upper,
                    contacts.contact_counters,
                    contacts.contact_generation,
                    pipeline.broad_phase_pair_count,
                    contacts.contact_counters.shape[0],
                ],
                outputs=[
                    pipeline.narrow_phase.shape_aabb_lower,
                    pipeline.narrow_phase.shape_aabb_upper,
                    pipeline.geom_data,
                    pipeline.geom_transform,
                ],
                device=self.device,
                record_tape=False,
            )
            # CPU numpy() shares Warp storage; preserve LIVE before RESET writes
            # the same buffers. Keep two intervals, not a swept union of poses.
            snapshots.append(
                tuple(
                    array.numpy().copy()
                    for array in (
                        pipeline.narrow_phase.shape_aabb_lower,
                        pipeline.narrow_phase.shape_aabb_upper,
                        pipeline.geom_transform,
                    )
                )
            )
        lower, upper, transforms = (np.concatenate([snapshot[index] for snapshot in snapshots]) for index in range(3))
        types, scales = np.tile(model.shape_type.numpy(), 2), np.tile(model.shape_scale.numpy(), (2, 1))
        margins, search_gap = np.tile(model.shape_margin.numpy(), 2), np.tile(model.shape_gap.numpy(), 2)
        lower += search_gap[:, None]
        upper -= search_gap[:, None]
        infinite = (types == GeoType.PLANE) & (scales[:, 0] == 0) & (scales[:, 1] == 0)
        colliding = np.tile((model.shape_flags.numpy() & int(ShapeFlags.COLLIDE_SHAPES)) != 0, 2)
        occupied = colliding.copy()
        for object_id in requested:
            start, end = runtime.bindings[object_id].shapes
            occupied[start:end] = occupied[start + model.shape_count : end + model.shape_count] = False
        axis, clearance = int(model.up_axis), 0.001
        lateral = [index for index in range(3) if index != axis]
        for object_id in requested:
            binding, obj = runtime.bindings[object_id], runtime.objects[object_id]
            shapes = slice(*binding.shapes)
            rows = np.r_[np.arange(shapes.start, shapes.stop), np.arange(shapes.start, shapes.stop) + model.shape_count]
            if shapes.start == shapes.stop or infinite[shapes].any():
                raise ValueError("Automatic placement requires bounded geometry; place infinite planes explicitly")
            lo, hi = lower[shapes].min(axis=0), upper[shapes].max(axis=0)
            if not np.isfinite([lo, hi]).all():
                raise ValueError("Automatic placement requires finite shape bounds")
            footprint = occupied & np.all(upper[:, lateral] >= lo[lateral] - clearance, axis=1)
            footprint &= np.all(lower[:, lateral] <= hi[lateral] + clearance, axis=1)
            supports = [0.0]
            for shape in np.flatnonzero(footprint & infinite):
                anchor = transforms[shape, :3]
                normal = wp.quat_rotate(wp.quat(*transforms[shape, 3:]), wp.vec3(0, 0, 1))
                if normal[axis] <= 1.0e-6:
                    raise ValueError("Automatic placement against a vertical or downward plane is unsupported")
                # Evaluate the plane over this footprint, not its million-metre AABB.
                rise = sum(max(normal[j] * (anchor[j] - lo[j]), normal[j] * (anchor[j] - hi[j])) for j in lateral)
                supports.append(anchor[axis] + rise / normal[axis] + margins[shape])
            height = float(max(supports))
            lift = max(0.0, height + clearance - float(lo[axis]))
            for shape in sorted(np.flatnonzero(footprint & ~infinite), key=lambda shape: lower[shape, axis]):
                if (
                    float(lo[axis]) + lift < float(upper[shape, axis]) + clearance
                    and float(hi[axis]) + lift > float(lower[shape, axis]) - clearance
                ):
                    lift = max(lift, float(upper[shape, axis]) + clearance - float(lo[axis]))
            if lift:
                translation = [0.0, 0.0, 0.0]
                translation[axis] = lift
                delta = wp.transform(translation, wp.quat_identity())
                start, end = binding.bodies
                if end > start:
                    wp.launch(
                        _move_bodies,
                        end - start,
                        inputs=[
                            start,
                            delta,
                            delta,
                            model.body_q,
                            runtime.state_0.body_q,
                            runtime.state_1.body_q,
                            runtime.state_0.body_qd,
                            runtime.state_1.body_qd,
                            False,
                        ],
                        device=self.device,
                    )
                wp.launch(
                    _move_world_shapes,
                    shapes.stop - shapes.start,
                    inputs=[shapes.start, delta, model.shape_body, model.shape_transform],
                    device=self.device,
                )
                start, end = binding.joints
                if end > start:
                    wp.launch(
                        _move_world_anchors,
                        end - start,
                        inputs=[start, delta, model.joint_parent, model.joint_type, model.joint_X_p],
                        device=self.device,
                    )
                position = tuple(value + translation[index] for index, value in enumerate(obj.position))
                runtime.objects[object_id] = dataclasses.replace(obj, position=position)
                lower[rows, axis] += lift
                upper[rows, axis] += lift
            occupied[rows] = colliding[rows]
        self._sync_coordinates(runtime)
        eval_ik(model, model, model.joint_q, model.joint_qd)
        self._collide(runtime)

    def _resolve_placement_history(self, runtime: _Runtime) -> None:
        # Pending additions can appear in later undo snapshots before a batch
        # commits. Keep their insertion baseline aligned with the resolved pose.
        offsets = {
            obj.id: tuple(
                after - before for after, before in zip(runtime.objects[obj.id].position, obj.position, strict=True)
            )
            for obj in self._objects
            if obj.id in self._placement_requests and runtime.objects[obj.id].position != obj.position
        }
        if not offsets:
            return

        def translated(objects):
            return tuple(
                dataclasses.replace(
                    obj,
                    position=tuple(value + shift for value, shift in zip(obj.position, offsets[obj.id], strict=True)),
                )
                if obj.id in offsets
                else obj
                for obj in objects
            )

        self._undo = [translated(objects) for objects in self._undo]
        self._redo = [translated(objects) for objects in self._redo]
        if self._edit_before is not None:
            self._edit_before = translated(self._edit_before)

    def commit(self) -> bool:
        """Build and adopt one candidate, keeping the old runtime on failure.

        This explicit operation may block for first-use kernel compilation.
        Interactive callers should use request_commit()/before_frame() so it
        never happens within an ImGui draw or a captured physics operation.
        """
        if self._closed:
            raise RuntimeError("Scene editor is closed")
        if not self._topology_dirty:
            self._apply_fast_edits()
            return False
        started = time.perf_counter()
        old = self._runtime
        candidate = self._build()
        if old is not None:
            self._preserve_state(old, candidate)
        self._place_candidate(candidate)
        # Surface asynchronous build errors before replacing the displayed scene.
        wp.synchronize_device(self.device)
        try:
            self.viewer.set_model(candidate.model)
            self.viewer.set_control(candidate.control)
            self.viewer.set_solver(candidate.solver)
            self.viewer.set_collision_pipeline(candidate.pipeline)
        except Exception:
            if old is not None:
                self.viewer.set_model(old.model)
                self.viewer.set_control(old.control)
                self.viewer.set_solver(old.solver)
                self.viewer.set_collision_pipeline(old.pipeline)
            raise
        self._publish(candidate)
        self._resolve_placement_history(candidate)
        self._objects = tuple(candidate.objects[obj.id] for obj in self._objects)
        self._placement_requests.difference_update(candidate.bindings)
        self._topology_dirty = self._requested_commit = False
        self._fast_pending.clear()
        self._trim_templates()
        self.rebuild_count += 1
        self.last_build_ms = (time.perf_counter() - started) * 1000.0
        self.error = None
        self.status = f"{len(self._objects)} objects · applied in {self.last_build_ms:.1f} ms"
        self._preview_last = None
        return True

    def body_indices(self, object_id: str) -> tuple[int, ...]:
        """Resolve current body indices; callers must not retain them across commits."""
        binding = self._runtime.bindings.get(object_id) if self._runtime else None
        return tuple(range(*binding.bodies)) if binding else ()

    def get_pose(self, object_id: str) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        """Read a selected free prop's live placement; scene assemblies use their authored root."""
        obj = self.get_object(object_id)
        runtime = self._runtime
        binding = runtime.bindings.get(object_id) if runtime else None
        if binding and binding.follows_body:
            index = binding.bodies[0]
            values = runtime.state_0.body_q[index : index + 1].numpy()[0]
            placement = wp.transform_multiply(
                wp.transform(values[:3], values[3:]),
                wp.transform_inverse(wp.transform(binding.reference[:3], binding.reference[3:])),
            )
            return tuple(wp.transform_get_translation(placement)), tuple(wp.transform_get_rotation(placement))
        return obj.position, obj.rotation

    @staticmethod
    def _apply_appearance(runtime: _Runtime, obj: Object) -> None:
        start, end = runtime.bindings[obj.id].shapes
        if end > start:
            wp.launch(
                _appearance,
                end - start,
                inputs=[
                    start,
                    wp.vec3(obj.color),
                    obj.visible,
                    obj.friction,
                    runtime.base_colors,
                    runtime.base_opacity,
                    runtime.model.shape_color,
                    runtime.model.shape_opacity,
                    runtime.model.shape_material_mu,
                ],
                device=runtime.model.device,
            )
            if obj.kind == SceneEditor.Shape.USD and "friction" not in obj.overrides:
                wp.copy(
                    runtime.model.shape_material_mu,
                    runtime.base_friction,
                    dest_offset=start,
                    src_offset=start,
                    count=end - start,
                )

    def _apply_fast_edits(self) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        current = {obj.id: obj for obj in self._objects}
        moved = False
        for object_id, previous in list(self._fast_pending.items()):
            if object_id not in current or object_id not in runtime.bindings:
                continue
            obj = current[object_id]
            if self._structure(obj) != self._structure(runtime.objects[object_id]):
                continue
            if (obj.position, obj.rotation) != (previous.position, previous.rotation):
                moved = True
                binding = runtime.bindings[object_id]
                live_position, live_rotation = self.get_pose(object_id)
                if not binding.follows_body:
                    live_position, live_rotation = previous.position, previous.rotation
                target = wp.transform(obj.position, obj.rotation)
                delta = wp.transform_multiply(target, wp.transform_inverse(wp.transform(live_position, live_rotation)))
                initial_delta = wp.transform_multiply(
                    target, wp.transform_inverse(wp.transform(previous.position, previous.rotation))
                )
                start, end = binding.bodies
                if end > start:
                    wp.launch(
                        _move_bodies,
                        end - start,
                        inputs=[
                            start,
                            delta,
                            initial_delta,
                            runtime.model.body_q,
                            runtime.state_0.body_q,
                            runtime.state_1.body_q,
                            runtime.state_0.body_qd,
                            runtime.state_1.body_qd,
                            True,
                        ],
                        device=self.device,
                    )
                start, end = binding.shapes
                if end > start:
                    wp.launch(
                        _move_world_shapes,
                        end - start,
                        inputs=[start, initial_delta, runtime.model.shape_body, runtime.model.shape_transform],
                        device=self.device,
                    )
                start, end = binding.joints
                if end > start:
                    wp.launch(
                        _move_world_anchors,
                        end - start,
                        inputs=[
                            start,
                            initial_delta,
                            runtime.model.joint_parent,
                            runtime.model.joint_type,
                            runtime.model.joint_X_p,
                        ],
                        device=self.device,
                    )
                self._sync_coordinates(runtime)
                eval_ik(runtime.model, runtime.model, runtime.model.joint_q, runtime.model.joint_qd)
            if (obj.color, obj.visible, obj.friction, obj.overrides) != (
                previous.color,
                previous.visible,
                previous.friction,
                previous.overrides,
            ):
                self._apply_appearance(runtime, obj)
            runtime.objects[object_id] = obj
        if moved:
            self._collide(runtime)
        self._fast_pending.clear()
        # Previous snapshots use the old authored poses and materials. Drop
        # them without clearing the selection or interrupting a held gizmo.
        gui = getattr(self.viewer, "gui", None)
        workspace = getattr(gui, "workspace", None)
        if workspace is not None and workspace.example is self:
            workspace.reset_history()

    def _collide(self, runtime: _Runtime) -> bool:
        """Detect complete contacts before solving, growing owned buffers if needed."""
        pipeline, contacts = runtime.pipeline, runtime.contacts
        try:
            for attempt in range(4):
                pipeline.collide(runtime.state_0, contacts)
                counters = contacts.contact_counters.numpy()
                count = int(counters[0])
                if count < 0:
                    raise RuntimeError("Rigid contact count exceeded supported indexing")
                if count <= contacts.rigid_contact_max and not counters[2:].any():
                    if pipeline is not runtime.pipeline:
                        if runtime is self._runtime:
                            self.viewer.set_collision_pipeline(pipeline)
                        previous = runtime.contacts.rigid_contact_max
                        runtime.pipeline, runtime.contacts = pipeline, contacts
                        runtime.model.rigid_contact_max = contacts.rigid_contact_max
                        runtime.model.soft_contact_max = contacts.soft_contact_max
                        runtime.model._contact_capacity_initialized = True
                        if runtime is self._runtime:
                            self.collision_pipeline, self.contacts = pipeline, contacts
                        logger.warning(
                            "Expanded rigid contact capacity from %d to %d; retained all %d contacts",
                            previous,
                            contacts.rigid_contact_max,
                            count,
                        )
                    return True
                if attempt == 3:
                    raise RuntimeError(
                        f"Rigid contacts remain incomplete after three growth attempts (count={count}, capacity={contacts.rigid_contact_max}, overflow={counters[2:].tolist()})"
                    )
                pipeline = CollisionPipeline(
                    runtime.model,
                    rigid_contact_max=max(2 * contacts.rigid_contact_max, 2 * count),
                    soft_contact_max=contacts.soft_contact_max,
                    publish_capacity=False,
                )
                contacts = pipeline.contacts()
        except (MemoryError, RuntimeError, ValueError) as error:
            message = f"Contact detection could not produce a complete buffer: {error}"
            if runtime is not self._runtime:
                raise RuntimeError(message) from error
            self.error = message
            self.status = "Physics paused; the current scene is retained."
            # An overflowing count must not reach display kernels either.
            runtime.contacts.contact_counters.zero_()
            logger.error(message)
            if not hasattr(self.viewer, "_paused"):
                raise RuntimeError(message) from error
            self.viewer._paused = True
        return False

    def reset_scene(self) -> None:
        """Restore the authored reset poses without rebuilding model or solver."""
        if self._runtime is not None:
            runtime = self._runtime
            if runtime.model.body_count:
                for state in (runtime.state_0, runtime.state_1):
                    wp.copy(state.body_q, runtime.model.body_q)
                    state.body_qd.zero_()
                    state.body_f.zero_()
            self._sync_coordinates(runtime)
            self._collide(runtime)
            self.sim_time = 0.0
            workspace = getattr(getattr(self.viewer, "gui", None), "workspace", None)
            if workspace is not None and workspace.example is self:
                workspace.reset_history()

    def step(self) -> None:
        """Step owned XPBD physics; topology never changes during these launches."""
        runtime = self._runtime
        if runtime is None:
            self.before_frame()
            runtime = self._runtime
        if runtime is None:
            return
        dt = self.frame_dt / self.sim_substeps
        for substep in range(self.sim_substeps):
            runtime.state_0.clear_forces()
            self.viewer.apply_forces(runtime.state_0)
            if not self._collide(runtime):
                self._sync_coordinates(runtime)
                self._publish(runtime)
                self.sim_time += substep * dt
                return
            runtime.solver.step(runtime.state_0, runtime.state_1, runtime.control, runtime.contacts, dt)
            runtime.state_0, runtime.state_1 = runtime.state_1, runtime.state_0
        self._sync_coordinates(runtime)
        self._publish(runtime)
        self.sim_time += self.frame_dt

    def render(self) -> None:
        """Render live or history-selected state with optional placement preview."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self._draw_preview()
        self.viewer.end_frame()

    def preview_primitive(self, kind: Shape | str, *, position, size=(1.0, 1.0, 1.0), color=(0.3, 0.85, 1.0)) -> None:
        """Update a transient placement outline without modifying the document/model."""
        self._preview = self.Shape(kind).value, tuple(position), tuple(size), tuple(color)

    def clear_preview(self) -> None:
        """Hide the transient placement outline on the next render."""
        self._preview = None

    def _draw_preview(self) -> None:
        name = "__scene_authoring/placement"
        if self._preview is None:
            if self._preview_last is not None:
                self.viewer.log_lines(name, None, None, None, hidden=True)
            self._preview_last = None
            return
        if self._preview != self._preview_last:
            _, position, size, _color = self._preview
            corners = np.asarray(
                [(x, y, z) for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)], dtype=np.float32
            )
            corners = corners * np.asarray(size) + np.asarray(position)
            edges = [(i, j) for i in range(8) for j in range(i + 1, 8) if (i ^ j) in (1, 2, 4)]
            starts = np.asarray([corners[a] for a, _ in edges], dtype=np.float32)
            ends = np.asarray([corners[b] for _, b in edges], dtype=np.float32)
            if self._preview_buffers is None:
                self._preview_buffers = (
                    wp.empty(12, dtype=wp.vec3, device=self.device),
                    wp.empty(12, dtype=wp.vec3, device=self.device),
                )
            self._preview_buffers[0].assign(starts)
            self._preview_buffers[1].assign(ends)
            self._preview_last = self._preview
        self.viewer.log_lines(name, *self._preview_buffers, self._preview[3])

    def save_usd(self, path: str | Path) -> None:
        """Export the authored document using standard USD geometry/physics schemas."""
        save_scene(path, self._objects)
        self.status = f"Saved {Path(path).name}"

    def load_usd(self, path: str | Path) -> None:
        """Open an editor document, or import an ordinary USD as a scene asset."""
        try:
            records = load_scene(path)
        except NotAuthoringSceneError:
            self.add_usd(path)
            return
        objects = tuple(
            self._validate(self.Object(**{**record, "kind": self.Shape(record["kind"])})) for record in records
        )
        if len({obj.id for obj in objects}) != len(objects):
            raise ValueError("Scene document contains duplicate object IDs")
        self._remember()
        self._restore_document(objects)
        for obj in objects:
            if obj.kind == self.Shape.USD:
                self._schedule_template(obj)

    def close(self) -> None:
        """Cancel pending preparation; retain already compiled state for caller inspection."""
        if self._closed:
            return
        self._closed = True
        for future in self._preparing.values():
            future.cancel()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def test_final(self) -> None:
        """Validate finite physical state and one compiled scene for example test mode."""
        if self._runtime is None:
            raise AssertionError("The authoring scene was not initialized")
        if self.model.body_count and not np.isfinite(self._runtime.state_0.body_q.numpy()).all():
            raise AssertionError("The authored scene contains non-finite body poses")
