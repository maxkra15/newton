# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import ctypes
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.viewer.gl.fluid import DiffuseBatch, FluidBatch, _pack_fluid_vertices
from newton._src.viewer.viewer_gl import ViewerGL
from newton.examples.fluid.example_fluid_sph_interactive_tank import Example
from newton.tests.unittest_utils import add_function_test, get_test_devices
from newton.viewer import ViewerNull


class _FluidGLProbe:
    """Captures the CPU fluid vertex upload without creating an OpenGL context."""

    GL_ARRAY_BUFFER = 1

    def __init__(self):
        self.data = None

    def glBindBuffer(self, _target, _buffer):
        pass

    def glBufferSubData(self, _target, _offset, size, pointer):
        raw = ctypes.string_at(pointer, size)
        self.data = np.frombuffer(raw, dtype=np.float32).copy()


class _DiffuseGLProbe:
    """Minimal OpenGL stand-in for diffuse buffer growth and uploads."""

    GL_ARRAY_BUFFER = 1
    GL_DYNAMIC_DRAW = 2
    GL_FLOAT = 3
    GL_FALSE = 0
    GLuint = ctypes.c_uint

    def __init__(self):
        self._next_name = 1
        self.uploads = []

    def _generate_name(self, name):
        name.value = self._next_name
        self._next_name += 1

    def glGenVertexArrays(self, _count, name):
        self._generate_name(name)

    def glGenBuffers(self, _count, name):
        self._generate_name(name)

    def glBindVertexArray(self, _array):
        pass

    def glBindBuffer(self, _target, _buffer):
        pass

    def glBufferData(self, _target, _size, _data, _usage):
        pass

    def glVertexAttribPointer(self, _index, _size, _type, _normalized, _stride, _pointer):
        pass

    def glEnableVertexAttribArray(self, _index):
        pass

    def glDeleteVertexArrays(self, _count, _array):
        pass

    def glDeleteBuffers(self, _count, _buffer):
        pass

    def glBufferSubData(self, _target, _offset, size, pointer):
        raw = ctypes.string_at(pointer, size)
        self.uploads.append(np.frombuffer(raw, dtype=np.float32).copy().reshape(-1, 4))


def _make_cpu_fluid_batch(capacity):
    batch = FluidBatch.__new__(FluidBatch)
    batch._gl = _FluidGLProbe()
    batch.capacity = capacity
    batch.count = 0
    batch.vbo = 0
    batch._ensure_capacity = lambda count: None
    return batch


def _make_diffuse_batch():
    """Create a diffuse batch without allocating OpenGL resources."""
    batch = DiffuseBatch.__new__(DiffuseBatch)
    batch._host_positions = np.zeros((0, 4), dtype=np.float32)
    batch._host_velocities = np.zeros((0, 4), dtype=np.float32)
    batch._ensure_capacity = lambda count: None
    batch._upload = lambda: None
    return batch


class _DiffuseBatchProbe:
    """Captures ViewerGL diffuse updates without creating an OpenGL context."""

    def __init__(self):
        self.update_args = None
        self.update_kwargs = None

    def update(self, *args, **kwargs):
        self.update_args = args
        self.update_kwargs = kwargs


class _LogFluidProbe(ViewerNull):
    """Captures fluid and point logging calls for ViewerBase particle routing."""

    def __init__(self):
        super().__init__(num_frames=1)
        self.logged_fluid = None
        self.logged_points = None

    def log_fluid(
        self,
        name,
        points,
        radii=None,
        radius_scale=1.0,
        color=(0.113, 0.425, 0.55, 0.8),
        ior=1.0,
        blur_radius_world=None,
        anisotropy=None,
        anisotropy_secondary=None,
        anisotropy_tertiary=None,
        hidden=False,
        worlds=None,
    ):
        self.logged_fluid = {
            "name": name,
            "points": points,
            "radii": radii,
            "radius_scale": radius_scale,
            "color": color,
            "ior": ior,
            "blur_radius_world": blur_radius_world,
            "anisotropy": anisotropy,
            "hidden": hidden,
            "worlds": worlds,
        }

    def log_points(self, name, points, radii=None, colors=None, hidden=False):
        self.logged_points = {"name": name, "points": points, "radii": radii, "hidden": hidden}


class TestViewerFluid(unittest.TestCase):
    @staticmethod
    def _build_model(flags_list):
        builder = newton.ModelBuilder()
        for i, flag in enumerate(flags_list):
            builder.add_particle(
                pos=(float(i), 0.0, 0.0),
                vel=(0.0, 0.0, 0.0),
                mass=1.0,
                radius=0.1,
                flags=flag,
            )
        return builder.finalize(device="cpu")

    def test_show_fluid_routes_active_particles_to_log_fluid(self):
        active = int(newton.ParticleFlags.ACTIVE)
        model = self._build_model([active, 0, active])
        state = model.state()
        viewer = _LogFluidProbe()

        viewer.set_model(model)
        viewer.show_fluid = True
        viewer.show_particles = False
        viewer._log_particles(state)

        self.assertIsNotNone(viewer.logged_fluid)
        self.assertEqual(viewer.logged_fluid["name"], "/model/fluid")
        self.assertFalse(viewer.logged_fluid["hidden"])
        self.assertEqual(viewer.logged_fluid["color"], viewer.fluid_color)
        self.assertEqual(viewer.logged_fluid["ior"], viewer.fluid_ior)
        np.testing.assert_allclose(viewer.logged_fluid["points"].numpy()[:, 0], [0.0, 2.0], atol=1.0e-6)
        self.assertIsNotNone(viewer.logged_fluid["worlds"])
        np.testing.assert_array_equal(viewer.logged_fluid["worlds"].numpy(), [-1, -1])
        self.assertIsNotNone(viewer.logged_points)
        self.assertEqual(viewer.logged_points["name"], "/model/particles")
        self.assertIsNone(viewer.logged_points["points"])
        self.assertTrue(viewer.logged_points["hidden"])

    def test_show_fluid_preserves_world_alignment_when_compacting_active_particles(self):
        active = int(newton.ParticleFlags.ACTIVE)
        local = newton.ModelBuilder()
        local.add_particle(
            pos=(1.0, 0.0, 0.0),
            vel=(0.0, 0.0, 0.0),
            mass=1.0,
            radius=0.1,
            flags=active,
        )
        local.add_particle(
            pos=(2.0, 0.0, 0.0),
            vel=(0.0, 0.0, 0.0),
            mass=1.0,
            radius=0.1,
            flags=0,
        )

        scene = newton.ModelBuilder()
        scene.add_particle(
            pos=(-1.0, 0.0, 0.0),
            vel=(0.0, 0.0, 0.0),
            mass=1.0,
            radius=0.1,
            flags=active,
        )
        scene.replicate(local, 2)
        model = scene.finalize(device="cpu")
        state = model.state()
        viewer = _LogFluidProbe()

        viewer.set_model(model)
        viewer.show_fluid = True
        viewer.show_particles = False
        viewer._log_particles(state)

        np.testing.assert_allclose(viewer.logged_fluid["points"].numpy()[:, 0], [-1.0, 1.0, 1.0], atol=1.0e-6)
        np.testing.assert_array_equal(viewer.logged_fluid["worlds"].numpy(), [-1, 0, 1])

    def test_cpu_fluid_batch_applies_world_offsets_and_visibility(self):
        batch = _make_cpu_fluid_batch(5)
        points = np.array(
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        worlds = np.array([0, 1, -1, 2, -1], dtype=np.int32)
        offsets = np.array([[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)
        visible = np.array([1, 0], dtype=np.int32)
        q1 = wp.array([(1.0, 0.0, 0.0, 1.0)] * 4 + [(1.0, 0.0, 0.0, 0.0)], dtype=wp.vec4, device="cpu")
        q2 = wp.array([(0.0, 1.0, 0.0, 1.0)] * 5, dtype=wp.vec4, device="cpu")
        q3 = wp.array([(0.0, 0.0, 1.0, 1.0)] * 5, dtype=wp.vec4, device="cpu")

        batch.update(
            points,
            0.1,
            anisotropy=q1,
            anisotropy_secondary=q2,
            anisotropy_tertiary=q3,
            worlds=worlds,
            world_offsets=offsets,
            visible_worlds_mask=visible,
        )

        data = batch._gl.data.reshape(5, 16)
        np.testing.assert_allclose(
            data[:, :3], [[-9.0, 0.0, 0.0], [12.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0], [5.0, 0.0, 0.0]]
        )
        np.testing.assert_allclose(data[:, 3], [0.1, 0.0, 0.1, 0.0, 0.0])

    def test_cpu_fluid_batch_ignores_world_state_without_world_ids(self):
        batch = _make_cpu_fluid_batch(1)
        points = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

        batch.update(
            points,
            0.2,
            world_offsets=np.zeros((1, 2), dtype=np.float32),
            visible_worlds_mask=np.zeros((1, 2), dtype=np.int32),
        )

        data = batch._gl.data.reshape(1, 16)
        np.testing.assert_allclose(data[0, :4], [1.0, 2.0, 3.0, 0.2])

    def test_cpu_fluid_batch_validates_world_array_shapes(self):
        batch = _make_cpu_fluid_batch(2)
        points = np.zeros((2, 3), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "worlds must have shape"):
            batch.update(points, 0.1, worlds=np.array([0], dtype=np.int32))
        with self.assertRaisesRegex(ValueError, "world_offsets must have shape"):
            batch.update(
                points,
                0.1,
                worlds=np.array([0, 1], dtype=np.int32),
                world_offsets=np.zeros((2, 2), dtype=np.float32),
            )
        with self.assertRaisesRegex(ValueError, "visible_worlds_mask must be one-dimensional"):
            batch.update(
                points,
                0.1,
                worlds=np.array([0, 1], dtype=np.int32),
                visible_worlds_mask=np.ones((2, 1), dtype=np.int32),
            )

    def test_diffuse_world_offsets_visibility_and_alignment(self):
        batch = _make_diffuse_batch()
        positions = np.array(
            [
                [1.0, 0.0, 0.0, 1.0],
                [2.0, 0.0, 0.0, 1.0],
                [3.0, 0.0, 0.0, 1.0],
                [4.0, 0.0, 0.0, 1.0],
                [5.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        original_positions = positions.copy()
        velocities = np.array(
            [
                [10.0, 0.0, 0.0, 0.0],
                [20.0, 0.0, 0.0, 0.0],
                [30.0, 0.0, 0.0, 0.0],
                [40.0, 0.0, 0.0, 0.0],
                [50.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        worlds = np.array([0, 1, 2, -1, 0], dtype=np.int32)
        offsets = np.array([[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)
        visible = np.array([1, 0], dtype=np.int32)

        batch.update(
            positions,
            velocities,
            worlds=worlds,
            world_offsets=offsets,
            visible_worlds_mask=visible,
        )

        self.assertEqual(batch.count, 2)
        np.testing.assert_allclose(
            batch._host_positions,
            [[-9.0, 0.0, 0.0, 1.0], [4.0, 0.0, 0.0, 1.0]],
        )
        np.testing.assert_allclose(
            batch._host_velocities,
            [[10.0, 0.0, 0.0, 0.0], [40.0, 0.0, 0.0, 0.0]],
        )
        np.testing.assert_array_equal(positions, original_positions)

    def test_diffuse_empty_visibility_mask_keeps_all_live_particles(self):
        batch = _make_diffuse_batch()
        positions = np.array(
            [[1.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0], [3.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

        batch.update(
            positions,
            None,
            worlds=np.array([0, 2, -1], dtype=np.int32),
            world_offsets=np.array([[5.0, 0.0, 0.0]], dtype=np.float32),
            visible_worlds_mask=np.empty(0, dtype=np.int32),
        )

        self.assertEqual(batch.count, 3)
        np.testing.assert_allclose(
            batch._host_positions[:, :3],
            [[6.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        )
        np.testing.assert_array_equal(batch._host_velocities, np.zeros_like(positions))

    def test_diffuse_legacy_update_filters_only_dead_particles(self):
        batch = _make_diffuse_batch()
        positions = np.array(
            [[1.0, 2.0, 3.0, 1.0], [4.0, 5.0, 6.0, 0.0], [7.0, 8.0, 9.0, 0.5]],
            dtype=np.float32,
        )
        velocities = np.arange(12, dtype=np.float32).reshape(3, 4)

        batch.update(positions, velocities)

        self.assertEqual(batch.count, 2)
        np.testing.assert_array_equal(batch._host_positions, positions[[0, 2]])
        np.testing.assert_array_equal(batch._host_velocities, velocities[[0, 2]])

    def test_diffuse_growth_preserves_filtered_data_and_hidden_state(self):
        gl = _DiffuseGLProbe()
        batch = DiffuseBatch(gl, capacity=1)
        batch.hidden = True
        positions = np.array(
            [[1.0, 2.0, 3.0, 1.0], [4.0, 5.0, 6.0, 0.0], [7.0, 8.0, 9.0, 0.5]],
            dtype=np.float32,
        )
        velocities = np.arange(12, dtype=np.float32).reshape(3, 4)

        batch.update(positions, velocities)

        self.assertEqual(batch.capacity, 2)
        self.assertEqual(batch.count, 2)
        self.assertTrue(batch.hidden)
        self.assertEqual(len(gl.uploads), 2)
        np.testing.assert_array_equal(gl.uploads[0], positions[[0, 2]])
        np.testing.assert_array_equal(gl.uploads[1], velocities[[0, 2]])

    def test_diffuse_update_validates_indexed_arrays(self):
        batch = _make_diffuse_batch()
        positions = np.zeros((2, 4), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "positions must have shape"):
            batch.update(np.zeros((2, 3), dtype=np.float32), None)
        with self.assertRaisesRegex(ValueError, "velocities must have shape"):
            batch.update(positions, np.zeros((1, 4), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "worlds must have shape"):
            batch.update(positions, None, worlds=np.array([0], dtype=np.int32))
        with self.assertRaisesRegex(TypeError, "worlds must use an integer dtype"):
            batch.update(positions, None, worlds=np.array([0.0, 1.0], dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "world_offsets must have shape"):
            batch.update(
                positions,
                None,
                worlds=np.array([0, 1], dtype=np.int32),
                world_offsets=np.zeros((2, 2), dtype=np.float32),
            )
        with self.assertRaisesRegex(ValueError, "visible_worlds_mask must be one-dimensional"):
            batch.update(
                positions,
                None,
                worlds=np.array([0, 1], dtype=np.int32),
                visible_worlds_mask=np.ones((2, 1), dtype=np.int32),
            )

    def test_viewer_gl_forwards_diffuse_world_state(self):
        batch = _DiffuseBatchProbe()
        viewer = ViewerGL.__new__(ViewerGL)
        viewer.fluid_diffuse = {"diffuse": batch}
        viewer.world_offsets = object()
        viewer._visible_worlds_mask = object()
        positions = np.zeros((1, 4), dtype=np.float32)
        velocities = np.ones((1, 4), dtype=np.float32)
        worlds = np.array([0], dtype=np.int32)

        viewer.log_fluid_diffuse("diffuse", positions, velocities, worlds=worlds)

        self.assertEqual(batch.update_args, (positions, velocities))
        self.assertIs(batch.update_kwargs["worlds"], worlds)
        self.assertIs(batch.update_kwargs["world_offsets"], viewer.world_offsets)
        self.assertIs(batch.update_kwargs["visible_worlds_mask"], viewer._visible_worlds_mask)

    def test_switching_from_fluid_to_particles_hides_fluid_batch(self):
        active = int(newton.ParticleFlags.ACTIVE)
        model = self._build_model([active])
        state = model.state()
        viewer = _LogFluidProbe()

        viewer.set_model(model)
        viewer.show_fluid = True
        viewer._log_particles(state)
        self.assertFalse(viewer.logged_fluid["hidden"])

        viewer.logged_fluid = None
        viewer.logged_points = None
        viewer.show_fluid = False
        viewer.show_particles = True
        viewer._log_particles(state)

        self.assertIsNotNone(viewer.logged_fluid)
        self.assertIsNone(viewer.logged_fluid["points"])
        self.assertTrue(viewer.logged_fluid["hidden"])
        self.assertIsNotNone(viewer.logged_points)
        self.assertFalse(viewer.logged_points["hidden"])

    def test_default_log_fluid_falls_back_to_points(self):
        active = int(newton.ParticleFlags.ACTIVE)
        model = self._build_model([active])
        state = model.state()
        viewer = _LogFluidProbe()

        ViewerNull.log_fluid(viewer, "fallback", state.particle_q, radii=0.2, hidden=False)

        self.assertIsNotNone(viewer.logged_points)
        self.assertEqual(viewer.logged_points["name"], "fallback")
        self.assertFalse(viewer.logged_points["hidden"])

    def test_interactive_tank_parser_defaults(self):
        args = Example.create_parser().parse_args([])

        self.assertEqual(args.render_mode, "fluid")
        self.assertEqual(args.substeps, 3)
        self.assertEqual(args.pbf_iterations, 3)
        self.assertGreater(args.box_count, 0)
        self.assertGreater(args.pick_stiffness, 0.0)
        self.assertTrue(args.show_diffuse)
        self.assertGreater(args.fluid_diffuse_max_particles, 0)
        self.assertAlmostEqual(args.buoyancy_scale, 1.0)
        self.assertGreater(args.box_linear_drag, 0.0)
        self.assertGreater(args.box_quadratic_drag, 0.0)
        self.assertGreater(args.box_floor_stiffness, 0.0)
        self.assertGreater(min(args.box_density_fractions), 0.0)
        self.assertGreater(max(args.box_density_fractions), 1.0)
        self.assertEqual(len(args.fluid_color), 4)
        self.assertGreater(args.fluid_radius_scale, 1.0)
        self.assertGreater(args.fluid_blur_radius, 0.0)
        self.assertGreater(args.foam_radius, 0.0)
        self.assertGreater(args.foam_motion_blur, 0.0)

    def test_interactive_tank_rollout_floats_and_sinks_boxes_by_density(self):
        args = Example.create_parser().parse_args(
            [
                "--viewer",
                "null",
                "--no-show-bounds",
                "--box-count",
                "3",
                "--box-density-fractions",
                "0.30",
                "0.60",
                "1.60",
                "--spacing",
                "0.08",
                "--radius",
                "0.06",
                "--smoothing-length",
                "0.172",
                "--shape-collision-distance",
                "0.06",
                "--shape-collision-margin",
                "0.003",
                "--particle-collision-margin",
                "0.003",
                "--fluid-carve-clearance",
                "0.08",
                "--dim-x",
                "34",
                "--dim-y",
                "22",
                "--dim-z",
                "6",
                "--emit-lower",
                "-1.24",
                "-0.78",
                "0.06",
                "--fluid-diffuse-max-particles",
                "0",
                "--fluid-render-update-interval",
                "1",
            ]
        )
        viewer = ViewerNull(num_frames=1)
        example = Example(viewer, args)

        max_speed = 0.0
        for _frame in range(240):
            example.step()
            wp.synchronize()
            body_qd = example.state_0.body_qd.numpy()[example.box_body_ids]
            max_speed = max(max_speed, float(np.linalg.norm(body_qd[:, :3], axis=1).max()))

        body_q = example.state_0.body_q.numpy()[example.box_body_ids]
        half_z = np.asarray([float(h[2]) for h in example.box_half_extents], dtype=np.float32)
        box_surface = example.box_water_height.numpy()

        self.assertTrue(np.all(np.isfinite(body_q)))
        self.assertLess(max_speed, 4.0)

        # Boxes lighter than water float partially submerged with a draft that
        # roughly tracks their density fraction.
        for box_idx, target_fraction in ((0, 0.30), (1, 0.60)):
            surface = float(box_surface[box_idx])
            bottom = float(body_q[box_idx, 2] - half_z[box_idx])
            top = float(body_q[box_idx, 2] + half_z[box_idx])
            self.assertLess(bottom, surface, f"floating box {box_idx} lost contact with the water")
            self.assertGreater(top, surface - 0.10, f"floating box {box_idx} is fully submerged")
            submerged_fraction = (surface - bottom) / (2.0 * float(half_z[box_idx]))
            self.assertLess(
                abs(submerged_fraction - target_fraction),
                0.35,
                f"floating box {box_idx} draft {submerged_fraction:.2f} far from density fraction",
            )

        # The denser-than-water box fully submerges and settles near the floor.
        sinker_top = float(body_q[2, 2] + half_z[2])
        sinker_bottom = float(body_q[2, 2] - half_z[2])
        self.assertLess(sinker_top, float(box_surface[2]) + 0.02, "dense box did not fully submerge")
        self.assertLess(sinker_bottom, args.bounds_lower[2] + 0.25, "dense box did not sink to the tank floor")


def test_fluid_surface_world_offsets_and_visibility(test, device):
    points = wp.array([(1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
    worlds = wp.array([0, 1, -1], dtype=wp.int32, device=device)
    offsets = wp.array([(-10.0, 0.0, 0.0), (10.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
    visible = wp.array([1, 0], dtype=wp.int32, device=device)
    radii = wp.array([0.1, 0.1, 0.1], dtype=wp.float32, device=device)
    dummy4 = wp.empty(0, dtype=wp.vec4, device=device)
    packed = wp.zeros(3 * 16, dtype=wp.float32, device=device)

    wp.launch(
        _pack_fluid_vertices,
        dim=3,
        inputs=[
            points,
            radii,
            1,
            0.0,
            1.0,
            dummy4,
            dummy4,
            dummy4,
            0,
            worlds,
            offsets,
            visible,
            1,
            packed,
        ],
        device=device,
    )
    data = packed.numpy().reshape(3, 16)
    np.testing.assert_allclose(data[:, :3], [[-9.0, 0.0, 0.0], [12.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    np.testing.assert_allclose(data[:, 3], [0.1, 0.0, 0.1])

    empty_worlds = wp.empty(0, dtype=wp.int32, device=device)
    empty_offsets = wp.empty(0, dtype=wp.vec3, device=device)
    empty_radii = wp.empty(0, dtype=wp.float32, device=device)
    legacy = wp.zeros(3 * 16, dtype=wp.float32, device=device)
    wp.launch(
        _pack_fluid_vertices,
        dim=3,
        inputs=[
            points,
            empty_radii,
            0,
            0.1,
            1.0,
            dummy4,
            dummy4,
            dummy4,
            0,
            empty_worlds,
            empty_offsets,
            empty_worlds,
            0,
            legacy,
        ],
        device=device,
    )
    legacy_data = legacy.numpy().reshape(3, 16)
    np.testing.assert_allclose(legacy_data[:, :3], points.numpy())
    np.testing.assert_allclose(legacy_data[:, 3], [0.1, 0.1, 0.1])


add_function_test(
    TestViewerFluid,
    "test_fluid_surface_world_offsets_and_visibility",
    test_fluid_surface_world_offsets_and_visibility,
    devices=get_test_devices(mode="basic"),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
