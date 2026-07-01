# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the coupled solver prototype."""

import unittest
from dataclasses import FrozenInstanceError
from typing import ClassVar
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.coupled.interface import CouplingEndpointKind, CouplingInterface
from newton._src.solvers.mujoco.equality import _add_equality_constraint
from newton.solvers import (
    SolverBase,
    SolverImplicitMPM,
    SolverMuJoCo,
    SolverSemiImplicit,
    SolverVBD,
    SolverXPBD,
)
from newton.solvers.experimental import coupled as coupled_api
from newton.solvers.experimental.coupled import (
    ModelView,
    SolverCoupled,
    SolverCoupledProxy,
)


@wp.kernel(enable_backward=False)
def _write_proxy_body_wrench_kernel(
    body_local_to_proxy_global: wp.array[int],
    out_body_f: wp.array[wp.spatial_vector],
):
    local_body = wp.tid()
    global_body = body_local_to_proxy_global[local_body]
    if global_body >= 0:
        out_body_f[global_body] = wp.spatial_vector(wp.vec3(1.0, 2.0, 3.0), wp.vec3(4.0, 5.0, 6.0))


@wp.kernel(enable_backward=False)
def _kick_proxy_particle_kernel(particle_qd: wp.array[wp.vec3]):
    particle_qd[0] = particle_qd[0] + wp.vec3(0.0, 2.0, 0.0)


@wp.kernel(enable_backward=False)
def _write_proxy_particle_force_kernel(
    particle_local_to_proxy_global: wp.array[int],
    out_particle_f: wp.array[wp.vec3],
):
    local_particle = wp.tid()
    global_particle = particle_local_to_proxy_global[local_particle]
    if global_particle >= 0:
        out_particle_f[global_particle] = wp.vec3(0.0, 7.0, 0.0)


class _BodyForceRecordingSolver(SolverBase, CouplingInterface):
    """Test solver that records body forces and otherwise copies state."""

    instances: ClassVar[list] = []

    def __init__(self, model):
        super().__init__(model)
        self.input_body_f = []
        self.instances.append(self)

    def step(self, state_in, state_out, control, contacts, dt):
        del control, contacts, dt
        self.input_body_f.append(state_in.body_f.numpy().copy())
        wp.copy(state_out.body_q, state_in.body_q)
        wp.copy(state_out.body_qd, state_in.body_qd)


class _ParticleForceRecordingSolver(SolverBase, CouplingInterface):
    """Test solver that records particle forces and otherwise copies state."""

    instances: ClassVar[list] = []

    def __init__(self, model):
        super().__init__(model)
        self.input_particle_f = []
        self.instances.append(self)

    def step(self, state_in, state_out, control, contacts, dt):
        del control, contacts, dt
        self.input_particle_f.append(state_in.particle_f.numpy().copy())
        wp.copy(state_out.particle_q, state_in.particle_q)
        wp.copy(state_out.particle_qd, state_in.particle_qd)


class _ControlRecordingSolver(SolverBase, CouplingInterface):
    """Test solver that records entry-local control arrays."""

    instances: ClassVar[list] = []

    def __init__(self, model):
        super().__init__(model)
        self.joint_f = []
        self.joint_target_q = []
        self.joint_target_qd = []
        self.instances.append(self)

    def step(self, state_in, state_out, control, contacts, dt):
        del contacts, dt
        self.joint_f.append(None if control is None or control.joint_f is None else control.joint_f.numpy().copy())
        self.joint_target_q.append(
            None if control is None or control.joint_target_q is None else control.joint_target_q.numpy().copy()
        )
        self.joint_target_qd.append(
            None if control is None or control.joint_target_qd is None else control.joint_target_qd.numpy().copy()
        )
        if state_in.body_q is not None and state_out.body_q is not None:
            wp.copy(state_out.body_q, state_in.body_q)
            wp.copy(state_out.body_qd, state_in.body_qd)
        if state_in.joint_q is not None and state_out.joint_q is not None:
            wp.copy(state_out.joint_q, state_in.joint_q)
            wp.copy(state_out.joint_qd, state_in.joint_qd)


class _InPlaceRecordingParticleSolver(SolverBase, CouplingInterface):
    """Test solver that records whether it was stepped in-place."""

    instances: ClassVar[dict[str, "_InPlaceRecordingParticleSolver"]] = {}

    def __init__(self, model):
        super().__init__(model)
        self.in_place_calls = []
        self.dt_values = []
        self.instances[model.name] = self

    def step(self, state_in, state_out, control, contacts, dt):
        del control, contacts
        self.in_place_calls.append(state_in is state_out)
        self.dt_values.append(dt)
        if state_in is not state_out:
            wp.copy(state_out.particle_q, state_in.particle_q)
            wp.copy(state_out.particle_qd, state_in.particle_qd)
        wp.launch(_kick_proxy_particle_kernel, dim=1, inputs=[state_out.particle_qd], device=self.model.device)


class _ProxyParticleKickSolver(SolverBase, CouplingInterface):
    """Destination test solver that applies a fixed impulse to proxy particle 0."""

    def step(self, state_in, state_out, control, contacts, dt):
        del control, contacts, dt
        wp.copy(state_out.particle_q, state_in.particle_q)
        wp.copy(state_out.particle_qd, state_in.particle_qd)
        wp.launch(_kick_proxy_particle_kernel, dim=1, inputs=[state_out.particle_qd], device=self.model.device)


class _ProxyParticleHookSolver(SolverBase, CouplingInterface):
    """Destination test solver that exposes particle proxy rewind/harvest hooks."""

    instances: ClassVar[list] = []

    def __init__(self, model):
        super().__init__(model)
        self.rewind_calls = 0
        self.harvest_calls = 0
        self.instances.append(self)

    def coupling_rewind_proxy_particle(
        self,
        particle_local_to_proxy_global,
        state,
        coupling_forces,
        particle_gravity_acceleration,
        dt,
    ):
        del particle_local_to_proxy_global, state, coupling_forces, particle_gravity_acceleration, dt
        self.rewind_calls += 1

    def coupling_harvest_proxy_particle_forces(
        self,
        particle_local_to_proxy_global,
        out_particle_f,
        *,
        particle_qd_before,
        state,
        state_out,
        contacts,
        dt,
    ):
        del particle_qd_before, state, state_out, contacts, dt
        self.harvest_calls += 1
        wp.launch(
            _write_proxy_particle_force_kernel,
            dim=particle_local_to_proxy_global.shape[0],
            inputs=[particle_local_to_proxy_global, out_particle_f],
            device=self.model.device,
        )

    def step(self, state_in, state_out, control, contacts, dt):
        del control, contacts, dt
        wp.copy(state_out.particle_q, state_in.particle_q)
        wp.copy(state_out.particle_qd, state_in.particle_qd)


class _ZeroingProxyParticleHookSolver(_ProxyParticleHookSolver):
    """Destination test solver that clears proxy particle feedback before writing."""

    def coupling_harvest_proxy_particle_forces(
        self,
        particle_local_to_proxy_global,
        out_particle_f,
        *,
        particle_qd_before,
        state,
        state_out,
        contacts,
        dt,
    ):
        out_particle_f.zero_()
        super().coupling_harvest_proxy_particle_forces(
            particle_local_to_proxy_global,
            out_particle_f,
            particle_qd_before=particle_qd_before,
            state=state,
            state_out=state_out,
            contacts=contacts,
            dt=dt,
        )


class _ProxyBodyHookSolver(SolverBase, CouplingInterface):
    """Destination test solver that writes proxy-indexed body feedback."""

    instances: ClassVar[list] = []

    def __init__(self, model):
        super().__init__(model)
        self.harvest_calls = 0
        self.instances.append(self)

    def coupling_harvest_proxy_wrenches(
        self,
        body_local_to_proxy_global,
        out_body_f,
        *,
        body_qd_before,
        state,
        state_out,
        contacts,
        dt,
    ):
        del body_qd_before, state, state_out, contacts, dt
        self.harvest_calls += 1
        wp.launch(
            _write_proxy_body_wrench_kernel,
            dim=body_local_to_proxy_global.shape[0],
            inputs=[body_local_to_proxy_global, out_body_f],
            device=self.model.device,
        )

    def step(self, state_in, state_out, control, contacts, dt):
        del control, contacts, dt
        wp.copy(state_out.body_q, state_in.body_q)
        wp.copy(state_out.body_qd, state_in.body_qd)


class _AffineBodyForceSourceSolver(SolverBase, CouplingInterface):
    """Map the input body-force x component to output linear velocity."""

    def step(self, state_in, state_out, control, contacts, dt):
        del control, contacts, dt
        wp.copy(state_out.body_q, state_in.body_q)
        body_qd = state_in.body_qd.numpy().copy()
        body_qd[:, 0] = state_in.body_f.numpy()[:, 0]
        state_out.body_qd.assign(body_qd)


class _AffineProxyBodyFeedbackSolver(SolverBase, CouplingInterface):
    """Return the scalar affine feedback map H(x) = -2x + 1."""

    def coupling_rewind_proxy_body(
        self,
        body_local_to_proxy_global,
        state,
        coupling_forces,
        body_gravity_acceleration,
        dt,
    ):
        del body_local_to_proxy_global, state, coupling_forces, body_gravity_acceleration, dt

    def coupling_harvest_proxy_wrenches(
        self,
        body_local_to_proxy_global,
        out_body_f,
        *,
        body_qd_before,
        state,
        state_out,
        contacts,
        dt,
    ):
        del state, state_out, contacts, dt
        proxy_ids = body_local_to_proxy_global.numpy()
        velocity = body_qd_before.numpy()
        force = np.zeros_like(out_body_f.numpy())
        for local_body, proxy_id in enumerate(proxy_ids):
            if proxy_id >= 0:
                force[proxy_id, 0] = -2.0 * velocity[local_body, 0] + 1.0
        out_body_f.assign(force)

    def step(self, state_in, state_out, control, contacts, dt):
        del control, contacts, dt
        wp.copy(state_out.body_q, state_in.body_q)
        wp.copy(state_out.body_qd, state_in.body_qd)


class _StepCountingCopySolver(SolverBase, CouplingInterface):
    """Test solver that records how many times it is stepped."""

    instances: ClassVar[dict[str, "_StepCountingCopySolver"]] = {}

    def __init__(self, model):
        super().__init__(model)
        self.step_count = 0
        self.dt_values = []
        self.model_notify_flags = []
        self.instances[model.name] = self

    def notify_model_changed(self, flags: int) -> None:
        self.model_notify_flags.append(int(flags))

    def step(self, state_in, state_out, control, contacts, dt):
        del control, contacts
        self.step_count += 1
        self.dt_values.append(dt)
        if state_in.body_q is not None and state_out.body_q is not None:
            wp.copy(state_out.body_q, state_in.body_q)
            wp.copy(state_out.body_qd, state_in.body_qd)
        if state_in.particle_q is not None and state_out.particle_q is not None:
            wp.copy(state_out.particle_q, state_in.particle_q)
            wp.copy(state_out.particle_qd, state_in.particle_qd)


class _ResetRecordingCopySolver(_StepCountingCopySolver):
    """Copy solver that records reset masks and flags without changing state."""

    instances: ClassVar[dict[str, "_ResetRecordingCopySolver"]] = {}

    def __init__(self, model):
        super().__init__(model)
        self.reset_calls = []

    def reset(self, state, world_mask=None, flags=None):
        self.reset_calls.append((state, world_mask, flags))


class _ConfigurableFailingCopySolver(_StepCountingCopySolver):
    """Copy solver that can fail on one selected step call."""

    instances: ClassVar[dict[str, "_ConfigurableFailingCopySolver"]] = {}

    def __init__(self, model):
        super().__init__(model)
        self.fail_on_step = None

    def step(self, state_in, state_out, control, contacts, dt):
        next_step = self.step_count + 1
        if self.fail_on_step == next_step:
            self.step_count = next_step
            raise RuntimeError(f"intentional failure in {self.model.name}")
        super().step(state_in, state_out, control, contacts, dt)


class _BodyVelocityKickSolver(_StepCountingCopySolver):
    """Copy solver that leaves a recognizable source velocity for proxy sync."""

    def step(self, state_in, state_out, control, contacts, dt):
        super().step(state_in, state_out, control, contacts, dt)
        body_qd = state_out.body_qd.numpy()
        body_qd[0, 0] = 42.0
        state_out.body_qd.assign(body_qd)


class _GraphCaptureRecordingSolver(_StepCountingCopySolver):
    """Copy solver with configurable graph support and preparation recording."""

    def __init__(self, model, supported: bool = True):
        super().__init__(model)
        self.supported = supported
        self.prepared_contacts = []

    @property
    def supports_cuda_graph_capture(self) -> bool:
        return self.supported

    def prepare_cuda_graph_capture(self, contacts=None) -> None:
        self.prepared_contacts.append(contacts)


class _ContactRecordingCopySolver(_StepCountingCopySolver):
    """Copy solver that records rigid contact shape ids seen by step()."""

    instances: ClassVar[dict[str, "_ContactRecordingCopySolver"]] = {}

    def __init__(self, model):
        super().__init__(model)
        self.rigid_shape0_steps = []
        self.rigid_shape1_steps = []
        self.step_contacts = []

    def step(self, state_in, state_out, control, contacts, dt):
        self.step_contacts.append(contacts)
        if contacts is not None and contacts.rigid_contact_count is not None:
            contact_count = int(contacts.rigid_contact_count.numpy()[0])
            self.rigid_shape0_steps.append(contacts.rigid_contact_shape0.numpy()[:contact_count].copy())
            self.rigid_shape1_steps.append(contacts.rigid_contact_shape1.numpy()[:contact_count].copy())
        super().step(state_in, state_out, control, contacts, dt)


class _ContactRecordingBodyHarvestSolver(_ContactRecordingCopySolver):
    """Contact-recording solver with custom body proxy contact hooks."""

    instances: ClassVar[dict[str, "_ContactRecordingBodyHarvestSolver"]] = {}

    def __init__(self, model):
        super().__init__(model)
        self.harvest_contacts = []

    def coupling_prepare_proxy_contacts(self, state, contacts, *, contacts_freshly_detected=False):
        del state, contacts_freshly_detected
        return contacts

    def coupling_harvest_proxy_wrenches(
        self,
        body_local_to_proxy_global,
        out_body_f,
        *,
        body_qd_before,
        state,
        state_out,
        contacts,
        dt,
    ):
        del body_local_to_proxy_global, out_body_f, body_qd_before, state, state_out, dt
        self.harvest_contacts.append(contacts)


class _DroppingProxyContactsSolver(_ContactRecordingBodyHarvestSolver):
    """Proxy destination solver that intentionally disables contact solving."""

    instances: ClassVar[dict[str, "_DroppingProxyContactsSolver"]] = {}

    def coupling_prepare_proxy_contacts(self, state, contacts, *, contacts_freshly_detected=False):
        del state, contacts, contacts_freshly_detected
        return None


class _ReplacingProxyContactsSolver(_ContactRecordingBodyHarvestSolver):
    """Proxy destination solver that returns a configured replacement buffer."""

    instances: ClassVar[dict[str, "_ReplacingProxyContactsSolver"]] = {}

    def __init__(self, model, replacement_contacts):
        super().__init__(model)
        self.replacement_contacts = replacement_contacts

    def coupling_prepare_proxy_contacts(self, state, contacts, *, contacts_freshly_detected=False):
        del state, contacts, contacts_freshly_detected
        return self.replacement_contacts


class _FakeProxyCollisionPipeline:
    """Minimal collision pipeline used to test proxy-coupler scheduling."""

    _UNSET_WORLD_MASK = object()

    def __init__(self, device, contacts=None, *, supports_cuda_graph_capture=True):
        self.contacts_obj = contacts if contacts is not None else newton.Contacts(0, 0, device=device)
        self.contacts_calls = 0
        self.collide_calls = 0
        self.collide_states = []
        self.collide_body_qd = []
        self.prepare_calls = 0
        self.reset_masks = []
        self.reset_arities = []
        self.supports_cuda_graph_capture = supports_cuda_graph_capture

    def contacts(self):
        self.contacts_calls += 1
        return self.contacts_obj

    def collide(self, state, contacts):
        self.collide_calls += 1
        self.collide_states.append(state)
        self.collide_body_qd.append(None if state.body_qd is None else state.body_qd.numpy().copy())
        self.last_contacts = contacts

    def prepare_cuda_graph_capture(self):
        self.prepare_calls += 1

    def reset_contact_matching(self, world_mask=_UNSET_WORLD_MASK):
        has_world_mask = world_mask is not self._UNSET_WORLD_MASK
        self.reset_arities.append(int(has_world_mask))
        self.reset_masks.append(world_mask if has_world_mask else None)


class _NoMaskResetCollisionPipeline(_FakeProxyCollisionPipeline):
    """Legacy duck provider whose optional reset hook accepts no mask."""

    def __init__(self, device, contacts=None):
        super().__init__(device, contacts=contacts)
        self.no_mask_reset_calls = 0

    def reset_contact_matching(self):
        self.no_mask_reset_calls += 1


class _TypeErrorResetCollisionPipeline(_FakeProxyCollisionPipeline):
    """Provider used to ensure hook-internal TypeError is never swallowed."""

    def reset_contact_matching(self, world_mask=None):
        del world_mask
        raise TypeError("provider reset failed internally")


class TestModelView(unittest.TestCase):
    """Test ModelView attribute delegation and overrides."""

    def setUp(self):
        builder = newton.ModelBuilder()
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_body(mass=2.0, inertia=wp.mat33(np.eye(3)))
        builder.add_shape_sphere(body=0, radius=0.1)
        builder.add_shape_sphere(body=1, radius=0.2)
        self.model = builder.finalize(device="cpu")

    def test_fallback_to_parent(self):
        """Unoverridden attributes should come from the parent model."""
        view = ModelView(self.model, "test")
        self.assertEqual(view.body_count, 2)
        self.assertIs(view.body_q, self.model.body_q)
        self.assertEqual(view.device, self.model.device)

    def test_override(self):
        """Overridden attributes should take precedence."""
        view = ModelView(self.model, "test")
        new_mass = wp.zeros(2, dtype=float, device="cpu")
        view.body_inv_mass = new_mass

        self.assertIs(view.body_inv_mass, new_mass)
        # Parent unchanged
        self.assertIsNot(self.model.body_inv_mass, new_mass)

    def test_count_override_slices_frequency_arrays(self):
        """Frequency-matched arrays should follow view-local counts."""
        view = ModelView(self.model, "test")
        view.body_count = 1
        view.shape_count = 1

        self.assertEqual(view.body_mass.shape[0], 1)
        self.assertEqual(view.body_inv_mass.shape[0], 1)
        self.assertEqual(view.shape_flags.shape[0], 1)
        self.assertEqual(self.model.body_mass.shape[0], 2)

    def test_zero_count_override_exposes_empty_frequency_arrays(self):
        """Zero-count views should expose empty arrays, not parent arrays."""
        builder = newton.ModelBuilder()
        builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0)
        model = builder.finalize(device="cpu")
        view = ModelView(model, "test")
        view.particle_count = 0

        self.assertEqual(view.particle_mass.shape[0], 0)
        self.assertEqual(view.particle_inv_mass.shape[0], 0)
        self.assertEqual(model.particle_mass.shape[0], 1)

    def test_disable_body_dynamics(self):
        """disable_body_dynamics should zero inverse inertia without changing flags."""
        view = ModelView(self.model, "test")
        indices = wp.array([1], dtype=int, device="cpu")
        view.disable_body_dynamics(indices)

        mass = view.body_mass.numpy()
        inertia = view.body_inertia.numpy()
        inv_mass = view.body_inv_mass.numpy()
        inv_inertia = view.body_inv_inertia.numpy()
        flags = view.body_flags.numpy()
        parent_flags = self.model.body_flags.numpy()
        dynamic = int(newton.BodyFlags.DYNAMIC)
        kinematic = int(newton.BodyFlags.KINEMATIC)
        # Body 0 should be unchanged (non-zero)
        self.assertGreater(mass[0], 0.0)
        self.assertGreater(inv_mass[0], 0.0)
        self.assertNotEqual(flags[0] & dynamic, 0)
        self.assertEqual(flags[0] & kinematic, 0)
        # Body 1 should keep forward inertial metadata but become immovable.
        self.assertEqual(mass[1], self.model.body_mass.numpy()[1])
        self.assertEqual(inv_mass[1], 0.0)
        np.testing.assert_allclose(inertia[1], self.model.body_inertia.numpy()[1])
        np.testing.assert_allclose(inv_inertia[1], np.zeros((3, 3)))
        self.assertNotEqual(flags[1] & dynamic, 0)
        self.assertEqual(flags[1] & kinematic, 0)
        self.assertNotEqual(parent_flags[1] & dynamic, 0)
        self.assertEqual(parent_flags[1] & kinematic, 0)

    def test_disable_joints_rewrites_cable_type_in_view(self):
        """disable_joints should expose disabled cable joints as D6 in the view."""
        builder = newton.ModelBuilder(gravity=0.0)
        parent = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        child = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        joint = builder.add_joint_cable(
            parent=parent,
            child=child,
            parent_xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.5, 0.0, 0.0), wp.quat_identity()),
        )
        model = builder.finalize(device="cpu")
        view = ModelView(model, "test")

        view.disable_joints(wp.array([joint], dtype=int, device="cpu"))

        self.assertFalse(bool(view.joint_enabled.numpy()[joint]))
        self.assertEqual(int(view.joint_type.numpy()[joint]), int(newton.JointType.D6))
        self.assertEqual(int(model.joint_type.numpy()[joint]), int(newton.JointType.CABLE))
        np.testing.assert_array_equal(view.joint_dof_dim.numpy()[joint], model.joint_dof_dim.numpy()[joint])

    def test_zero_particle_mass(self):
        """zero_particle_mass should zero forward and inverse mass arrays."""
        builder = newton.ModelBuilder()
        builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0)
        builder.add_particle(pos=(0.1, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=2.0)
        model = builder.finalize(device="cpu")
        view = ModelView(model, "test")

        view.zero_particle_mass(wp.array([1], dtype=int, device="cpu"))

        np.testing.assert_allclose(view.particle_mass.numpy(), [1.0, 0.0])
        np.testing.assert_allclose(view.particle_inv_mass.numpy(), [1.0, 0.0])
        np.testing.assert_allclose(model.particle_mass.numpy(), [1.0, 2.0])

    def test_set_body_inertial_properties(self):
        """set_body_inertial_properties should replace mass and full inertia."""
        view = ModelView(self.model, "test")
        indices = wp.array([1], dtype=int, device="cpu")
        target_mass = wp.array([4.0], dtype=float, device="cpu")
        target_inertia_np = np.array([[[2.0, 0.25, 0.0], [0.25, 3.0, 0.5], [0.0, 0.5, 5.0]]])
        target_inertia = wp.array(target_inertia_np, dtype=wp.mat33, device="cpu")

        view.set_body_inertial_properties(indices, target_mass, target_inertia)

        np.testing.assert_allclose(view.body_mass.numpy()[1], 4.0)
        np.testing.assert_allclose(view.body_inv_mass.numpy()[1], 0.25)
        np.testing.assert_allclose(view.body_inertia.numpy()[1], target_inertia_np[0])
        np.testing.assert_allclose(view.body_inv_inertia.numpy()[1], np.linalg.inv(target_inertia_np[0]), rtol=1.0e-6)

    def test_mark_proxy_bodies(self):
        """mark_proxy_bodies should mark only the view-local body flags."""
        view = ModelView(self.model, "test")
        indices = wp.array([1], dtype=int, device="cpu")
        view.mark_proxy_bodies(indices)

        view_flags = view.body_flags.numpy()
        parent_flags = self.model.body_flags.numpy()
        self.assertEqual(view_flags[0] & int(newton.BodyFlags.PROXY), 0)
        self.assertNotEqual(view_flags[1] & int(newton.BodyFlags.PROXY), 0)
        self.assertEqual(parent_flags[1] & int(newton.BodyFlags.PROXY), 0)

    def test_disable_particles(self):
        """disable_particles should clear only view-local active flags."""
        builder = newton.ModelBuilder()
        builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0)
        builder.add_particle(pos=(0.1, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0)
        model = builder.finalize(device="cpu")

        view = ModelView(model, "test")
        indices = wp.array([1], dtype=int, device="cpu")
        view.disable_particles(indices)

        active = int(newton.ParticleFlags.ACTIVE)
        view_flags = view.particle_flags.numpy()
        parent_flags = model.particle_flags.numpy()
        self.assertNotEqual(view_flags[0] & active, 0)
        self.assertEqual(view_flags[1] & active, 0)
        self.assertNotEqual(parent_flags[1] & active, 0)

    def test_state_creation(self):
        """view.state() should create a valid State."""
        view = ModelView(self.model, "test")
        state = view.state()
        self.assertEqual(state.body_count, 2)

    def test_state_creation_uses_view_overrides(self):
        """view.state() should clone state-relevant view-local arrays."""
        view = ModelView(self.model, "test")
        body_qd = self.model.body_qd.numpy()
        body_qd[1, 0] = 3.0
        view.body_qd = wp.array(body_qd, dtype=wp.spatial_vector, device="cpu")

        state = view.state()

        np.testing.assert_allclose(state.body_qd.numpy()[1, 0], 3.0)
        self.assertIsNot(state.body_qd, view.body_qd)
        np.testing.assert_allclose(state.body_f.numpy(), np.zeros_like(body_qd))

    def test_state_creation_respects_view_count_overrides(self):
        """view.state() should size state arrays from view-local counts."""
        self.model.request_state_attributes("body_qdd", "body_parent_f")
        view = ModelView(self.model, "test")
        view.body_count = 1

        state = view.state()

        self.assertEqual(state.body_count, 1)
        self.assertEqual(state.body_qd.shape[0], 1)
        self.assertEqual(state.body_f.shape[0], 1)
        self.assertEqual(state.body_qdd.shape[0], 1)
        self.assertEqual(state.body_parent_f.shape[0], 1)

    def test_state_creation_respects_view_zero_count(self):
        """view.state() should clear state fields hidden by view-local counts."""
        view = ModelView(self.model, "test")
        view.body_count = 0

        state = view.state()

        self.assertIsNone(state.body_q)
        self.assertIsNone(state.body_qd)
        self.assertIsNone(state.body_f)

    def test_set_body_mass_rejects_static_to_dynamic_without_inertia(self):
        """set_body_mass should not create finite mass with zero inertia."""
        builder = newton.ModelBuilder()
        builder.add_body(mass=0.0, inertia=wp.mat33(0.0))
        model = builder.finalize(device="cpu")
        view = ModelView(model, "test")

        with self.assertRaisesRegex(ValueError, "set_body_inertial_properties"):
            view.set_body_mass(wp.array([0], dtype=int, device="cpu"), wp.array([1.0], dtype=float, device="cpu"))

    def test_setattr_rejects_unknown_name(self):
        view = ModelView(self.model, "test")
        with self.assertRaisesRegex(AttributeError, "no such attribute"):
            view.not_a_model_field = 0

    def test_setattr_rejects_dtype_mismatch(self):
        view = ModelView(self.model, "test")
        with self.assertRaisesRegex(TypeError, "body_inv_mass"):
            view.body_inv_mass = wp.zeros(2, dtype=int, device="cpu")

    def test_setattr_rejects_ndim_mismatch(self):
        view = ModelView(self.model, "test")
        with self.assertRaisesRegex(TypeError, "body_inv_mass"):
            view.body_inv_mass = wp.zeros((2, 2), dtype=float, device="cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_setattr_rejects_device_mismatch(self):
        view = ModelView(self.model, "test")
        with self.assertRaisesRegex(TypeError, "body_inv_mass"):
            view.body_inv_mass = wp.zeros(2, dtype=float, device="cuda")

    def test_setattr_rejects_wrong_python_type(self):
        view = ModelView(self.model, "test")
        with self.assertRaisesRegex(TypeError, "body_count"):
            view.body_count = "two"

    def test_setattr_allows_none_when_parent_is_array(self):
        view = ModelView(self.model, "test")
        view.body_inv_mass = None
        self.assertIsNone(view.body_inv_mass)


class TestSolverCoupledGraphCapture(unittest.TestCase):
    """Test CUDA graph capability aggregation and preparation forwarding."""

    @staticmethod
    def _recording_factory(record, name: str, *, supported: bool = True):
        def factory(model):
            solver = _GraphCaptureRecordingSolver(model, supported=supported)
            record[name] = solver
            return solver

        return factory

    def test_nested_cuda_graph_capability_is_aggregated(self):
        """An unsupported nested leaf should reject capture for every parent."""
        model = newton.ModelBuilder().finalize(device="cpu")
        leaves = {}
        nested_solvers = []

        def nested_factory(view):
            nested = SolverCoupled(
                model=view,
                entries=[
                    SolverCoupled.Entry(
                        name="supported",
                        solver=self._recording_factory(leaves, "supported"),
                    ),
                    SolverCoupled.Entry(
                        name="unsupported",
                        solver=self._recording_factory(leaves, "unsupported", supported=False),
                    ),
                ],
            )
            nested_solvers.append(nested)
            return nested

        coupled = SolverCoupled(
            model=model,
            entries=[SolverCoupled.Entry(name="nested", solver=nested_factory)],
        )

        self.assertFalse(nested_solvers[0].supports_cuda_graph_capture)
        self.assertFalse(coupled.supports_cuda_graph_capture)

    def test_prepare_cuda_graph_capture_forwards_filtered_contacts_recursively(self):
        """Preparation should pass exact filtered buffers without stepping or changing state."""
        builder = newton.ModelBuilder()
        body_left = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        body_right = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        shape_left = builder.add_shape_sphere(body=body_left, radius=0.1)
        shape_right = builder.add_shape_sphere(body=body_right, radius=0.1)
        model = builder.finalize(device="cpu")

        leaves = {}
        nested_solvers = []

        def nested_factory(view):
            nested = SolverCoupled(
                model=view,
                entries=[
                    SolverCoupled.Entry(
                        name="left",
                        solver=self._recording_factory(leaves, "left"),
                        bodies=[body_left],
                        shapes=[shape_left],
                    ),
                    SolverCoupled.Entry(
                        name="right",
                        solver=self._recording_factory(leaves, "right"),
                        bodies=[body_right],
                        shapes=[shape_right],
                    ),
                ],
            )
            nested_solvers.append(nested)
            return nested

        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="nested",
                    solver=nested_factory,
                    bodies=[body_left, body_right],
                    shapes=[shape_left, shape_right],
                )
            ],
        )
        nested = nested_solvers[0]
        contacts = newton.Contacts(rigid_contact_max=4, soft_contact_max=0, device=model.device)
        contacts.rigid_contact_count.fill_(2)
        contacts.rigid_contact_shape0.assign(np.array([shape_left, shape_right, -1, -1], dtype=np.int32))
        contacts.rigid_contact_shape1.assign(np.array([shape_left, shape_right, -1, -1], dtype=np.int32))

        model_body_q_before = model.body_q.numpy().copy()
        coupled_state_before = coupled.entry_state("nested", "input").body_q.numpy().copy()
        nested_state_before = {
            name: nested.entry_state(name, "input").body_q.numpy().copy() for name in ("left", "right")
        }

        coupled.prepare_cuda_graph_capture(contacts)

        outer_filtered = coupled.entry_contacts("nested", contacts)
        outer_generation = int(outer_filtered.contact_generation.numpy()[0])
        left_filtered = nested.entry_contacts("left", outer_filtered)
        right_filtered = nested.entry_contacts("right", outer_filtered)
        self.assertIs(leaves["left"].prepared_contacts[0], left_filtered)
        self.assertIs(leaves["right"].prepared_contacts[0], right_filtered)
        self.assertEqual(int(left_filtered.rigid_contact_count.numpy()[0]), 1)
        self.assertEqual(int(right_filtered.rigid_contact_count.numpy()[0]), 1)
        self.assertEqual(int(left_filtered.rigid_contact_shape0.numpy()[0]), shape_left)
        self.assertEqual(int(right_filtered.rigid_contact_shape0.numpy()[0]), shape_right)
        self.assertEqual(leaves["left"].step_count, 0)
        self.assertEqual(leaves["right"].step_count, 0)
        np.testing.assert_array_equal(model.body_q.numpy(), model_body_q_before)
        np.testing.assert_array_equal(coupled.entry_state("nested", "input").body_q.numpy(), coupled_state_before)
        for name in ("left", "right"):
            np.testing.assert_array_equal(nested.entry_state(name, "input").body_q.numpy(), nested_state_before[name])

        replacement_contacts = newton.Contacts(rigid_contact_max=4, soft_contact_max=0, device=model.device)
        replacement_contacts.rigid_contact_count.fill_(1)
        replacement_contacts.rigid_contact_shape0.assign(np.array([shape_right, -1, -1, -1], dtype=np.int32))
        replacement_contacts.rigid_contact_shape1.assign(np.array([shape_right, -1, -1, -1], dtype=np.int32))

        coupled.prepare_cuda_graph_capture(replacement_contacts)

        self.assertIs(leaves["left"].prepared_contacts[1], left_filtered)
        self.assertIs(leaves["right"].prepared_contacts[1], right_filtered)
        self.assertGreater(int(outer_filtered.contact_generation.numpy()[0]), outer_generation)
        self.assertEqual(int(left_filtered.rigid_contact_count.numpy()[0]), 0)
        self.assertEqual(int(right_filtered.rigid_contact_count.numpy()[0]), 1)

        replacement_contacts.clear()
        replacement_contacts.rigid_contact_count.fill_(1)
        replacement_contacts.rigid_contact_shape0.assign(np.array([shape_left, -1, -1, -1], dtype=np.int32))
        replacement_contacts.rigid_contact_shape1.assign(np.array([shape_left, -1, -1, -1], dtype=np.int32))

        coupled.prepare_cuda_graph_capture(replacement_contacts)

        self.assertEqual(int(left_filtered.rigid_contact_count.numpy()[0]), 1)
        self.assertEqual(int(right_filtered.rigid_contact_count.numpy()[0]), 0)

    def test_implicit_mpm_cuda_graph_capability_requires_fixed_topology(self):
        """Resolved fixed-grid MPM supports capture while dynamic grids do not."""
        solver = SolverImplicitMPM.__new__(SolverImplicitMPM)

        for grid_type, expected in (("sparse", False), ("dense", False), ("fixed", True)):
            with self.subTest(grid_type=grid_type):
                solver.grid_type = grid_type
                self.assertEqual(solver.supports_cuda_graph_capture, expected)


class TestSolverCoupledEntryCollision(unittest.TestCase):
    """Test entry-local collision provider construction and scheduling."""

    @staticmethod
    def _build_proxy_model():
        builder = newton.ModelBuilder(gravity=0.0)
        source_body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        source_shape = builder.add_shape_sphere(
            body=source_body,
            radius=0.1,
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.25),
        )
        destination_body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        destination_shape = builder.add_shape_sphere(
            body=destination_body,
            radius=0.1,
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.75),
        )
        return builder.finalize(device="cpu"), source_body, source_shape, destination_body, destination_shape

    @staticmethod
    def _build_multi_world_contact_model():
        world = newton.ModelBuilder(gravity=0.0)
        body_a = world.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0)))
        world.add_shape_sphere(body=body_a, radius=0.1)
        body_b = world.add_body(xform=wp.transform(wp.vec3(0.19, 0.0, 0.0)))
        world.add_shape_sphere(body=body_b, radius=0.1)

        builder = newton.ModelBuilder(gravity=0.0)
        global_body_a = builder.add_body(xform=wp.transform(wp.vec3(10.0, 0.0, 0.0)))
        builder.add_shape_sphere(body=global_body_a, radius=0.1)
        global_body_b = builder.add_body(xform=wp.transform(wp.vec3(10.19, 0.0, 0.0)))
        builder.add_shape_sphere(body=global_body_b, radius=0.1)
        builder.add_world(world)
        builder.add_world(world, xform=wp.transform(wp.vec3(2.0, 0.0, 0.0)))
        return builder.finalize(device="cpu")

    @staticmethod
    def _contact_worlds(model, contacts, count):
        shape_world = model.shape_world.numpy()
        shape0 = contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = contacts.rigid_contact_shape1.numpy()[:count]
        world0 = shape_world[shape0]
        return np.where(world0 >= 0, world0, shape_world[shape1])

    def test_pipeline_uses_final_view_and_refreshes_on_outer_cadence(self):
        """The provider should see the compact final view and refresh once per outer step."""
        _ContactRecordingCopySolver.instances.clear()
        builder = newton.ModelBuilder(gravity=0.0)
        body_a = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        body_b = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_shape_sphere(body=body_a, radius=0.1)
        shape_b = builder.add_shape_sphere(body=body_b, radius=0.1)
        model = builder.finalize(device="cpu")

        pipelines = []
        final_view_snapshots = []

        def configure_view(view):
            friction = view.shape_material_mu.numpy().copy()
            friction[shape_b] = 3.5
            view.shape_material_mu = wp.array(friction, dtype=wp.float32, device=model.device)

        def collision_pipeline(view):
            final_view_snapshots.append(
                (view, int(view.shape_count), int(view.shape_contact_pair_count), view.shape_material_mu.numpy().copy())
            )
            pipeline = _FakeProxyCollisionPipeline(model.device)
            pipelines.append(pipeline)
            return pipeline

        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="entry",
                    solver=_ContactRecordingCopySolver,
                    bodies=[body_b],
                    shapes=[shape_b],
                    configure_view=configure_view,
                    substeps=3,
                    preserve_shape_ids=False,
                    collision_pipeline=collision_pipeline,
                    collide_interval=2,
                )
            ],
        )

        self.assertEqual(len(pipelines), 1)
        pipeline = pipelines[0]
        self.assertEqual(pipeline.contacts_calls, 1)
        final_view, shape_count, pair_count, friction = final_view_snapshots[0]
        self.assertIs(final_view, coupled.view("entry"))
        self.assertEqual(shape_count, 1)
        self.assertEqual(pair_count, 0)
        np.testing.assert_allclose(friction, [3.5])

        state = model.state()
        outer_contacts = newton.Contacts(0, 0, device=model.device)
        for _ in range(3):
            coupled.step(state, state, control=None, contacts=outer_contacts, dt=0.03)

        solver = _ContactRecordingCopySolver.instances["entry"]
        self.assertEqual(pipeline.collide_calls, 2)
        self.assertEqual(len(solver.step_contacts), 9)
        self.assertTrue(all(contacts is pipeline.contacts_obj for contacts in solver.step_contacts))
        self.assertTrue(all(state is coupled.entry_state("entry", "input") for state in pipeline.collide_states))

    def test_factory_returning_none_falls_back_to_outer_contacts(self):
        """A disabled provider should preserve the compatible outer-contact path."""
        _ContactRecordingCopySolver.instances.clear()
        model, source_body, source_shape, destination_body, destination_shape = self._build_proxy_model()
        factory_views = []
        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="entry",
                    solver=_ContactRecordingCopySolver,
                    bodies=[source_body, destination_body],
                    shapes=[source_shape, destination_shape],
                    collision_pipeline=factory_views.append,
                )
            ],
        )
        outer_contacts = newton.Contacts(1, 0, device=model.device)
        outer_contacts.rigid_contact_count.fill_(1)
        outer_contacts.rigid_contact_shape0.fill_(source_shape)
        outer_contacts.rigid_contact_shape1.fill_(destination_shape)

        coupled.step(model.state(), model.state(), control=None, contacts=outer_contacts, dt=0.01)

        self.assertEqual(factory_views, [coupled.view("entry")])
        fallback_contacts = _ContactRecordingCopySolver.instances["entry"].step_contacts[0]
        self.assertIsNotNone(fallback_contacts)
        self.assertIs(fallback_contacts, coupled.entry_contacts("entry", outer_contacts))
        self.assertEqual(int(fallback_contacts.rigid_contact_count.numpy()[0]), 1)
        np.testing.assert_array_equal(fallback_contacts.rigid_contact_shape0.numpy()[:1], [source_shape])
        np.testing.assert_array_equal(fallback_contacts.rigid_contact_shape1.numpy()[:1], [destination_shape])

    def test_factory_returning_none_does_not_forward_parent_contacts_to_compact_view(self):
        """Compact entries must not receive parent-global contact IDs as local IDs."""
        _ContactRecordingCopySolver.instances.clear()
        builder = newton.ModelBuilder(gravity=0.0)
        bodies = []
        shapes = []
        for x in range(3):
            body = builder.add_body(
                xform=wp.transform(wp.vec3(float(x), 0.0, 0.0), wp.quat_identity()),
                mass=1.0,
                inertia=wp.mat33(np.eye(3)),
            )
            bodies.append(body)
            shapes.append(builder.add_shape_sphere(body=body, radius=0.1))
        model = builder.finalize(device="cpu")
        factory_views = []
        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="compact",
                    solver=_ContactRecordingCopySolver,
                    bodies=bodies[1:],
                    shapes=shapes[1:],
                    preserve_shape_ids=False,
                    collision_pipeline=factory_views.append,
                )
            ],
        )
        outer_contacts = newton.Contacts(1, 0, device=model.device)
        outer_contacts.rigid_contact_count.fill_(1)
        outer_contacts.rigid_contact_shape0.fill_(shapes[1])
        outer_contacts.rigid_contact_shape1.fill_(shapes[2])

        self.assertIsNone(coupled.entry_contacts("compact", outer_contacts))
        coupled.step(model.state(), model.state(), control=None, contacts=outer_contacts, dt=0.01)

        self.assertEqual(factory_views, [coupled.view("compact")])
        self.assertIsNone(_ContactRecordingCopySolver.instances["compact"].step_contacts[0])

    def test_invalid_collision_configuration_is_rejected(self):
        model = newton.ModelBuilder().finalize(device="cpu")

        with self.assertRaisesRegex(ValueError, "collide_interval.*collision_pipeline"):
            SolverCoupled(
                model=model,
                entries=[SolverCoupled.Entry("entry", _StepCountingCopySolver, collide_interval=1)],
            )

        for interval in (0, -1, 1.5, True, "2"):
            with self.subTest(interval=interval), self.assertRaisesRegex(ValueError, "collide_interval.*>= 1"):
                SolverCoupled(
                    model=model,
                    entries=[
                        SolverCoupled.Entry(
                            "entry",
                            _StepCountingCopySolver,
                            collision_pipeline=lambda view: _FakeProxyCollisionPipeline(view.device),
                            collide_interval=interval,
                        )
                    ],
                )

        with self.assertRaisesRegex(TypeError, "collision_pipeline.*callable"):
            SolverCoupled(
                model=model,
                entries=[SolverCoupled.Entry("entry", _StepCountingCopySolver, collision_pipeline=object())],
            )

        with self.assertRaisesRegex(TypeError, r"contacts\(\).*collide\(\)"):
            SolverCoupled(
                model=model,
                entries=[
                    SolverCoupled.Entry(
                        "entry",
                        _StepCountingCopySolver,
                        collision_pipeline=lambda view: object(),
                    )
                ],
            )

    def test_positional_entry_compatibility_and_raw_solver_identity(self):
        """Appending provider fields must not shift old positional fields or wrap solvers."""
        entry = SolverCoupled.Entry(
            "entry",
            _StepCountingCopySolver,
            (),
            (),
            (),
            (),
            None,
            2,
            True,
            False,
        )
        self.assertEqual(entry.substeps, 2)
        self.assertTrue(entry.in_place)
        self.assertFalse(entry.preserve_shape_ids)
        self.assertIsNone(entry.collision_pipeline)
        self.assertIsNone(entry.collide_interval)

        model = newton.ModelBuilder().finalize(device="cpu")
        solver_instances = []
        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="entry",
                    solver=lambda view: solver_instances.append(_StepCountingCopySolver(view)) or solver_instances[-1],
                    collision_pipeline=lambda view: _FakeProxyCollisionPipeline(view.device),
                )
            ],
        )

        self.assertIs(coupled.solver("entry"), solver_instances[0])
        self.assertIsInstance(coupled.solver("entry"), CouplingInterface)

    def test_algorithm_contacts_override_entry_provider(self):
        """The filter_contacts=False seam should retain explicit algorithm contacts."""
        _ContactRecordingCopySolver.instances.clear()
        model = newton.ModelBuilder().finalize(device="cpu")
        entry_contacts = newton.Contacts(0, 0, device=model.device)
        algorithm_contacts = newton.Contacts(0, 0, device=model.device)

        class AlgorithmContactsCoupled(SolverCoupled):
            def _step_coupled(self, state_in, state_out, control, contacts, dt):
                del state_in, state_out, contacts
                self._step_entry(
                    self._entries["entry"],
                    control,
                    algorithm_contacts,
                    dt,
                    filter_contacts=False,
                )

        coupled = AlgorithmContactsCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="entry",
                    solver=_ContactRecordingCopySolver,
                    collision_pipeline=lambda view: _FakeProxyCollisionPipeline(
                        view.device,
                        contacts=entry_contacts,
                    ),
                )
            ],
        )

        coupled.step(model.state(), model.state(), control=None, contacts=None, dt=0.01)

        self.assertIs(_ContactRecordingCopySolver.instances["entry"].step_contacts[0], algorithm_contacts)

    def test_entry_provider_contacts_bypass_parent_filtering(self):
        """Provider contacts already use the entry namespace and must remain untouched."""
        _ContactRecordingCopySolver.instances.clear()
        model, _source_body, source_shape, destination_body, destination_shape = self._build_proxy_model()
        provider_contacts = newton.Contacts(1, 0, device=model.device)
        provider_contacts.rigid_contact_count.fill_(1)
        provider_contacts.rigid_contact_shape0.fill_(source_shape)
        provider_contacts.rigid_contact_shape1.fill_(destination_shape)
        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="entry",
                    solver=_ContactRecordingCopySolver,
                    bodies=[destination_body],
                    shapes=[destination_shape],
                    collision_pipeline=lambda view: _FakeProxyCollisionPipeline(
                        view.device,
                        contacts=provider_contacts,
                    ),
                )
            ],
        )

        coupled.step(
            model.state(),
            model.state(),
            control=None,
            contacts=newton.Contacts(1, 0, device=model.device),
            dt=0.01,
        )

        solver = _ContactRecordingCopySolver.instances["entry"]
        self.assertIs(solver.step_contacts[0], provider_contacts)
        self.assertEqual(int(provider_contacts.rigid_contact_count.numpy()[0]), 1)

    def test_failed_step_does_not_consume_collision_provider_cadence(self):
        """Retrying a failed outer step must repeat all scheduled collision refreshes."""
        _ConfigurableFailingCopySolver.instances.clear()
        model, source_body, source_shape, destination_body, destination_shape = self._build_proxy_model()
        entry_pipeline = _FakeProxyCollisionPipeline(model.device)
        mapping_pipeline = _FakeProxyCollisionPipeline(model.device)
        coupled = SolverCoupledProxy(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="source",
                    solver=_StepCountingCopySolver,
                    bodies=[source_body],
                    shapes=[source_shape],
                    collision_pipeline=lambda view: entry_pipeline,
                    collide_interval=2,
                ),
                SolverCoupled.Entry(
                    name="destination",
                    solver=_ConfigurableFailingCopySolver,
                    bodies=[destination_body],
                    shapes=[destination_shape],
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="source",
                        destination="destination",
                        bodies=[source_body],
                        collision_pipeline=lambda view: mapping_pipeline,
                        collide_interval=2,
                    )
                ]
            ),
        )
        solver = _ConfigurableFailingCopySolver.instances["destination"]
        solver.fail_on_step = 1
        state = model.state()
        key = ("source", "destination")

        with self.assertRaisesRegex(RuntimeError, "intentional failure in destination"):
            coupled.step(state, state, control=None, contacts=None, dt=0.01)

        self.assertEqual(entry_pipeline.collide_calls, 1)
        self.assertEqual(mapping_pipeline.collide_calls, 1)
        self.assertEqual(coupled._entries["source"].collide_counter, 0)
        self.assertEqual(coupled.get_proxy_collision_state()[key], 0)
        self.assertEqual(coupled.contact_streams(), ())

        solver.fail_on_step = None
        coupled.step(state, state, control=None, contacts=None, dt=0.01)

        self.assertEqual(entry_pipeline.collide_calls, 2)
        self.assertEqual(mapping_pipeline.collide_calls, 2)
        self.assertEqual(coupled._entries["source"].collide_counter, 1)
        self.assertEqual(coupled.get_proxy_collision_state()[key], 1)

        solver.fail_on_step = solver.step_count + 1
        with self.assertRaisesRegex(RuntimeError, "intentional failure in destination"):
            coupled.step(state, state, control=None, contacts=None, dt=0.01)

        self.assertEqual(entry_pipeline.collide_calls, 2)
        self.assertEqual(mapping_pipeline.collide_calls, 2)
        self.assertEqual(coupled._entries["source"].collide_counter, 1)
        self.assertEqual(coupled.get_proxy_collision_state()[key], 1)

        solver.fail_on_step = None
        coupled.step(state, state, control=None, contacts=None, dt=0.01)

        self.assertEqual(entry_pipeline.collide_calls, 2)
        self.assertEqual(mapping_pipeline.collide_calls, 2)
        self.assertEqual(coupled._entries["source"].collide_counter, 2)
        self.assertEqual(coupled.get_proxy_collision_state()[key], 2)

    def test_reset_clears_provider_contacts_and_cadence(self):
        _ContactRecordingCopySolver.instances.clear()
        model = newton.ModelBuilder().finalize(device="cpu")
        contacts = newton.Contacts(1, 0, device=model.device)
        contacts.rigid_contact_count.fill_(1)
        pipeline = _FakeProxyCollisionPipeline(model.device, contacts=contacts)
        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="entry",
                    solver=_ContactRecordingCopySolver,
                    collision_pipeline=lambda view: pipeline,
                    collide_interval=3,
                )
            ],
        )
        state = model.state()
        coupled.step(state, state, control=None, contacts=None, dt=0.01)
        self.assertEqual(pipeline.collide_calls, 1)

        coupled.reset(state)

        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)
        coupled.step(state, state, control=None, contacts=None, dt=0.01)
        self.assertEqual(pipeline.collide_calls, 2)

    def test_entry_provider_reset_forwards_mask_clears_contacts_and_forces_cadence(self):
        model = self._build_multi_world_contact_model()
        contacts = newton.Contacts(1, 0, device=model.device)
        contacts.rigid_contact_count.fill_(1)
        pipeline = _FakeProxyCollisionPipeline(model.device, contacts=contacts)
        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="entry",
                    solver=_ResetRecordingCopySolver,
                    bodies=list(range(model.body_count)),
                    shapes=list(range(model.shape_count)),
                    collision_pipeline=lambda view: pipeline,
                    collide_interval=3,
                )
            ],
        )
        state = model.state()
        entry = coupled._entries["entry"]

        coupled.step(state, state, control=None, contacts=None, dt=0.01)
        self.assertEqual(pipeline.collide_calls, 1)
        self.assertEqual(entry.collide_counter, 1)
        coupled._entry_contact_sources["entry"] = contacts

        world_mask = wp.array([True, False], dtype=wp.bool, device=model.device)
        coupled.reset(state, world_mask=world_mask)

        self.assertEqual(pipeline.reset_masks, [world_mask])
        self.assertEqual(pipeline.reset_arities, [1])
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)
        self.assertEqual(entry.collide_counter, 0)
        self.assertEqual(coupled._entry_contact_sources, {})
        coupled.step(state, state, control=None, contacts=None, dt=0.01)
        self.assertEqual(pipeline.collide_calls, 2, "Reset must force a fresh collision despite interval=3")

        contacts.rigid_contact_count.fill_(1)
        coupled.reset(state)
        self.assertEqual(pipeline.reset_masks, [world_mask, None])
        self.assertEqual(pipeline.reset_arities, [1, 0])
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)
        self.assertEqual(entry.collide_counter, 0)

    def test_entry_provider_reset_hook_supports_no_mask_and_propagates_internal_type_error(self):
        model = self._build_multi_world_contact_model()
        world_mask = wp.array([True, False], dtype=wp.bool, device=model.device)

        legacy_pipeline = _NoMaskResetCollisionPipeline(model.device)
        legacy_coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="legacy",
                    solver=_ResetRecordingCopySolver,
                    bodies=list(range(model.body_count)),
                    shapes=list(range(model.shape_count)),
                    collision_pipeline=lambda view: legacy_pipeline,
                )
            ],
        )
        legacy_coupled.reset(model.state(), world_mask=world_mask)
        self.assertEqual(legacy_pipeline.no_mask_reset_calls, 1)

        failing_pipeline = _TypeErrorResetCollisionPipeline(model.device)
        failing_coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="failing",
                    solver=_ResetRecordingCopySolver,
                    bodies=list(range(model.body_count)),
                    shapes=list(range(model.shape_count)),
                    collision_pipeline=lambda view: failing_pipeline,
                )
            ],
        )
        with self.assertRaisesRegex(TypeError, "provider reset failed internally"):
            failing_coupled.reset(model.state(), world_mask=world_mask)

    def test_reset_clears_provider_contact_matching_history(self):
        _ContactRecordingCopySolver.instances.clear()
        builder = newton.ModelBuilder(gravity=0.0)
        body_a = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0)))
        shape_a = builder.add_shape_sphere(body=body_a, radius=0.1)
        body_b = builder.add_body(xform=wp.transform(wp.vec3(0.19, 0.0, 0.0)))
        shape_b = builder.add_shape_sphere(body=body_b, radius=0.1)
        model = builder.finalize(device="cpu")
        pipelines = []

        def collision_pipeline(view):
            pipeline = newton.CollisionPipeline(view, broad_phase="nxn", contact_matching="latest")
            pipelines.append(pipeline)
            return pipeline

        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="entry",
                    solver=_ContactRecordingCopySolver,
                    bodies=[body_a, body_b],
                    shapes=[shape_a, shape_b],
                    collision_pipeline=collision_pipeline,
                )
            ],
        )
        state = model.state()

        coupled.step(state, state, control=None, contacts=None, dt=0.01)
        coupled.step(state, state, control=None, contacts=None, dt=0.01)

        contacts = _ContactRecordingCopySolver.instances["entry"].step_contacts[-1]
        contact_count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(contact_count, 0)
        np.testing.assert_array_equal(
            contacts.rigid_contact_match_index.numpy()[:contact_count],
            np.arange(contact_count, dtype=np.int32),
        )

        coupled.reset(state)
        coupled.step(state, state, control=None, contacts=None, dt=0.01)

        self.assertEqual(len(pipelines), 1)
        contacts = _ContactRecordingCopySolver.instances["entry"].step_contacts[-1]
        contact_count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(contact_count, 0)
        self.assertTrue(np.all(contacts.rigid_contact_match_index.numpy()[:contact_count] == -1))

    def test_entry_provider_partial_reset_preserves_other_world_contact_history(self):
        for matching_mode in ("latest", "sticky"):
            with self.subTest(contact_matching=matching_mode):
                model = self._build_multi_world_contact_model()
                pipelines = []

                def collision_pipeline(view, matching_mode=matching_mode, pipelines=pipelines):
                    pipeline = newton.CollisionPipeline(
                        view,
                        broad_phase="nxn",
                        contact_matching=matching_mode,
                        verify_buffers=False,
                    )
                    pipelines.append(pipeline)
                    return pipeline

                coupled = SolverCoupled(
                    model=model,
                    entries=[
                        SolverCoupled.Entry(
                            name="entry",
                            solver=_ResetRecordingCopySolver,
                            bodies=list(range(model.body_count)),
                            shapes=list(range(model.shape_count)),
                            collision_pipeline=collision_pipeline,
                            collide_interval=2,
                        )
                    ],
                )
                state = model.state()
                entry = coupled._entries["entry"]

                for _ in range(3):
                    coupled.step(state, state, control=None, contacts=None, dt=0.01)

                contacts = entry.collision_contacts
                contact_count = int(contacts.rigid_contact_count.numpy()[0])
                self.assertEqual(contact_count, 3)
                worlds_before = self._contact_worlds(model, contacts, contact_count)
                np.testing.assert_array_equal(np.sort(worlds_before), [-1, 0, 1])
                self.assertTrue(np.all(contacts.rigid_contact_match_index.numpy()[:contact_count] >= 0))
                normals_before = contacts.rigid_contact_normal.numpy()[:contact_count].copy()

                world_mask = wp.array([True, False], dtype=wp.bool, device=model.device)
                coupled.reset(state, world_mask=world_mask)
                self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)
                self.assertEqual(entry.collide_counter, 0)

                body_world = model.body_world.numpy()
                world0_bodies = np.flatnonzero(body_world == 0)
                q = state.body_q.numpy()
                q[world0_bodies[1]][1] += 0.0001
                state.body_q.assign(q)

                coupled.step(state, state, control=None, contacts=None, dt=0.01)

                self.assertEqual(entry.collide_counter, 1)
                contact_count = int(contacts.rigid_contact_count.numpy()[0])
                self.assertEqual(contact_count, 3)
                worlds_after = self._contact_worlds(model, contacts, contact_count)
                match_index = contacts.rigid_contact_match_index.numpy()[:contact_count]
                self.assertTrue(np.all(match_index[worlds_after == 0] == newton.geometry.MATCH_NOT_FOUND))
                self.assertTrue(np.all(match_index[worlds_after != 0] >= 0))
                if matching_mode == "sticky":
                    normals_after = contacts.rigid_contact_normal.numpy()[:contact_count]
                    self.assertFalse(
                        np.array_equal(normals_after[worlds_after == 0], normals_before[worlds_before == 0]),
                        "Selected sticky contacts must keep fresh geometry instead of replaying invalid history",
                    )

                self.assertEqual(len(pipelines), 1)

    def test_graph_support_and_preparation_include_entry_provider(self):
        _GraphCaptureRecordingSolver.instances.clear()
        model = newton.ModelBuilder().finalize(device="cpu")
        pipeline = _FakeProxyCollisionPipeline(model.device, supports_cuda_graph_capture=False)
        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="entry",
                    solver=_GraphCaptureRecordingSolver,
                    collision_pipeline=lambda view: pipeline,
                )
            ],
        )

        self.assertFalse(coupled.supports_cuda_graph_capture)
        coupled.prepare_cuda_graph_capture(newton.Contacts(0, 0, device=model.device))

        solver = _GraphCaptureRecordingSolver.instances["entry"]
        self.assertEqual(pipeline.prepare_calls, 1)
        self.assertEqual(pipeline.collide_calls, 0)
        self.assertEqual(solver.step_count, 0)
        self.assertEqual(solver.prepared_contacts, [pipeline.contacts_obj])
        self.assertNotIn("entry", coupled._entry_contact_buffers)

        class MinimalPipeline:
            def __init__(self):
                self.contacts_obj = newton.Contacts(0, 0, device=model.device)

            def contacts(self):
                return self.contacts_obj

            def collide(self, state, contacts):
                del state, contacts

        supported = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="supported",
                    solver=_GraphCaptureRecordingSolver,
                    collision_pipeline=lambda view: MinimalPipeline(),
                )
            ],
        )
        self.assertTrue(supported.supports_cuda_graph_capture)
        supported.prepare_cuda_graph_capture()
        self.assertEqual(_GraphCaptureRecordingSolver.instances["supported"].step_count, 0)

        cadenced = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="cadenced",
                    solver=_GraphCaptureRecordingSolver,
                    collision_pipeline=lambda view: MinimalPipeline(),
                    collide_interval=2,
                )
            ],
        )
        self.assertFalse(cadenced.supports_cuda_graph_capture)

    def test_graph_support_includes_mapping_provider_and_nested_solver(self):
        model, source_body, source_shape, destination_body, destination_shape = self._build_proxy_model()

        def make_coupled(*, provider_supported=True, collide_interval=1, solver_supported=True):
            pipeline = _FakeProxyCollisionPipeline(
                model.device,
                supports_cuda_graph_capture=provider_supported,
            )
            coupled = SolverCoupledProxy(
                model=model,
                entries=[
                    SolverCoupled.Entry(
                        name="source",
                        solver=_GraphCaptureRecordingSolver,
                        bodies=[source_body],
                        shapes=[source_shape],
                    ),
                    SolverCoupled.Entry(
                        name="destination",
                        solver=lambda view: _GraphCaptureRecordingSolver(view, supported=solver_supported),
                        bodies=[destination_body],
                        shapes=[destination_shape],
                    ),
                ],
                coupling=SolverCoupledProxy.Config(
                    proxies=[
                        SolverCoupledProxy.Proxy(
                            source="source",
                            destination="destination",
                            bodies=[source_body],
                            collision_pipeline=lambda view: pipeline,
                            collide_interval=collide_interval,
                        )
                    ]
                ),
            )
            return coupled

        self.assertTrue(make_coupled().supports_cuda_graph_capture)
        self.assertFalse(make_coupled(provider_supported=False).supports_cuda_graph_capture)
        self.assertFalse(make_coupled(collide_interval=2).supports_cuda_graph_capture)
        self.assertFalse(make_coupled(solver_supported=False).supports_cuda_graph_capture)

    def test_graph_preparation_includes_mapping_provider_without_simulation(self):
        _GraphCaptureRecordingSolver.instances.clear()
        model, source_body, source_shape, destination_body, destination_shape = self._build_proxy_model()
        mapping_contacts = newton.Contacts(2, 1, device=model.device)
        pipeline = _FakeProxyCollisionPipeline(model.device, contacts=mapping_contacts)
        coupled = SolverCoupledProxy(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="source",
                    solver=_GraphCaptureRecordingSolver,
                    bodies=[source_body],
                    shapes=[source_shape],
                ),
                SolverCoupled.Entry(
                    name="destination",
                    solver=_GraphCaptureRecordingSolver,
                    bodies=[destination_body],
                    shapes=[destination_shape],
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="source",
                        destination="destination",
                        bodies=[source_body],
                        collision_pipeline=lambda view: pipeline,
                    )
                ]
            ),
        )
        state_before = {
            name: coupled.entry_state(name, "input").body_q.numpy().copy() for name in ("source", "destination")
        }

        coupled.prepare_cuda_graph_capture()

        source_solver = _GraphCaptureRecordingSolver.instances["source"]
        destination_solver = _GraphCaptureRecordingSolver.instances["destination"]
        self.assertEqual(pipeline.prepare_calls, 1)
        self.assertEqual(pipeline.contacts_calls, 1)
        self.assertEqual(pipeline.collide_calls, 0)
        self.assertEqual(source_solver.prepared_contacts, [None])
        self.assertEqual(destination_solver.prepared_contacts, [None, mapping_contacts])
        self.assertEqual(source_solver.step_count, 0)
        self.assertEqual(destination_solver.step_count, 0)
        self.assertEqual(coupled._proxy_collision_configs[("source", "destination")].collide_counter, 0)
        self.assertEqual(coupled._proxy_contact_stream_buffers, {})
        for name in ("source", "destination"):
            np.testing.assert_array_equal(coupled.entry_state(name, "input").body_q.numpy(), state_before[name])

    def test_mapping_provider_collide_interval_requires_exact_positive_int(self):
        model, source_body, source_shape, destination_body, destination_shape = self._build_proxy_model()

        for interval in (0, -1, 1.5, True, "2"):
            with self.subTest(interval=interval), self.assertRaisesRegex(ValueError, "collide_interval.*>= 1"):
                SolverCoupledProxy(
                    model=model,
                    entries=[
                        SolverCoupled.Entry(
                            name="source",
                            solver=_StepCountingCopySolver,
                            bodies=[source_body],
                            shapes=[source_shape],
                        ),
                        SolverCoupled.Entry(
                            name="destination",
                            solver=_StepCountingCopySolver,
                            bodies=[destination_body],
                            shapes=[destination_shape],
                        ),
                    ],
                    coupling=SolverCoupledProxy.Config(
                        proxies=[
                            SolverCoupledProxy.Proxy(
                                source="source",
                                destination="destination",
                                bodies=[source_body],
                                collision_pipeline=lambda view: _FakeProxyCollisionPipeline(view.device),
                                collide_interval=interval,
                            )
                        ]
                    ),
                )

    def test_proxy_destination_uses_entry_provider_after_sync_once_per_outer_step(self):
        """Proxy iterations should refresh destination entry contacts after proxy state sync."""
        _BodyVelocityKickSolver.instances.clear()
        _ContactRecordingBodyHarvestSolver.instances.clear()
        model, source_body, source_shape, destination_body, destination_shape = self._build_proxy_model()
        pipeline = _FakeProxyCollisionPipeline(model.device)
        source_solvers = []
        coupled = SolverCoupledProxy(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="source",
                    solver=lambda view: source_solvers.append(_BodyVelocityKickSolver(view)) or source_solvers[-1],
                    bodies=[source_body],
                    shapes=[source_shape],
                ),
                SolverCoupled.Entry(
                    name="destination",
                    solver=_ContactRecordingBodyHarvestSolver,
                    bodies=[destination_body],
                    shapes=[destination_shape],
                    collision_pipeline=lambda view: pipeline,
                    collide_interval=2,
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="source",
                        destination="destination",
                        bodies=[source_body],
                    )
                ],
                iterations=3,
            ),
        )
        state = model.state()

        for _ in range(3):
            coupled.step(state, state, control=None, contacts=None, dt=0.01)

        destination_solver = _ContactRecordingBodyHarvestSolver.instances["destination"]
        self.assertIs(coupled.solver("source"), source_solvers[0])
        self.assertEqual(pipeline.collide_calls, 2)
        self.assertEqual(len(destination_solver.step_contacts), 9)
        self.assertTrue(all(contacts is pipeline.contacts_obj for contacts in destination_solver.step_contacts))
        self.assertTrue(np.isclose(pipeline.collide_body_qd[0][:, 0], 42.0).any())
        self.assertIsNone(coupled.get_proxy_contacts("source", "destination"))

    def test_bidirectional_entry_providers_refresh_before_each_entry_first_step(self):
        """A proxy cycle must not step its first source before that source's provider."""
        events = []

        class EventSolver(_StepCountingCopySolver):
            def __init__(self, model):
                super().__init__(model)
                self.proxy_contact_freshness = []

            def coupling_prepare_proxy_contacts(self, state, contacts, *, contacts_freshly_detected=False):
                del state
                self.proxy_contact_freshness.append(contacts_freshly_detected)
                return contacts

            def step(self, state_in, state_out, control, contacts, dt):
                events.append(("step", self.model.name))
                super().step(state_in, state_out, control, contacts, dt)

        class EventPipeline(_FakeProxyCollisionPipeline):
            def __init__(self, device, name):
                super().__init__(device)
                self.name = name

            def collide(self, state, contacts):
                events.append(("collide", self.name))
                super().collide(state, contacts)

        model, body_a, shape_a, body_b, shape_b = self._build_proxy_model()
        pipelines = {}

        def provider(name):
            def factory(view):
                pipeline = EventPipeline(view.device, name)
                pipelines[name] = pipeline
                return pipeline

            return factory

        coupled = SolverCoupledProxy(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="A",
                    solver=EventSolver,
                    bodies=[body_a],
                    shapes=[shape_a],
                    collision_pipeline=provider("A"),
                ),
                SolverCoupled.Entry(
                    name="B",
                    solver=EventSolver,
                    bodies=[body_b],
                    shapes=[shape_b],
                    collision_pipeline=provider("B"),
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(source="A", destination="B", bodies=[body_a]),
                    SolverCoupledProxy.Proxy(source="B", destination="A", bodies=[body_b]),
                ],
                iterations=2,
            ),
        )

        coupled.step(model.state(), model.state(), control=None, contacts=None, dt=0.01)

        self.assertEqual(
            events,
            [
                ("collide", "A"),
                ("step", "A"),
                ("collide", "B"),
                ("step", "B"),
                ("step", "B"),
                ("step", "A"),
                ("step", "A"),
                ("step", "B"),
                ("step", "B"),
                ("step", "A"),
            ],
        )
        self.assertEqual(pipelines["A"].collide_calls, 1)
        self.assertEqual(pipelines["B"].collide_calls, 1)
        self.assertEqual(coupled._entries["A"].collide_counter, 1)
        self.assertEqual(coupled._entries["B"].collide_counter, 1)
        self.assertEqual(coupled.solver("A").step_count, 4)
        self.assertEqual(coupled.solver("B").step_count, 4)
        self.assertEqual(coupled.solver("A").proxy_contact_freshness, [True, False])
        self.assertEqual(coupled.solver("B").proxy_contact_freshness, [True, False])

    def test_mapping_proxy_provider_reset_forwards_mask_clears_contacts_and_forces_cadence(self):
        world = newton.ModelBuilder(gravity=0.0)
        source_body = world.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        world.add_shape_sphere(body=source_body, radius=0.1)
        destination_body = world.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        world.add_shape_sphere(body=destination_body, radius=0.1)
        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_world(world)
        builder.add_world(world)
        model = builder.finalize(device="cpu")
        source_bodies = [0, 2]
        destination_bodies = [1, 3]
        source_shapes = [0, 2]
        destination_shapes = [1, 3]

        contacts = newton.Contacts(1, 0, device=model.device)
        contacts.rigid_contact_count.fill_(1)
        pipeline = _FakeProxyCollisionPipeline(model.device, contacts=contacts)
        coupled = SolverCoupledProxy(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="source",
                    solver=_StepCountingCopySolver,
                    bodies=source_bodies,
                    shapes=source_shapes,
                ),
                SolverCoupled.Entry(
                    name="destination",
                    solver=_ContactRecordingBodyHarvestSolver,
                    bodies=destination_bodies,
                    shapes=destination_shapes,
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="source",
                        destination="destination",
                        bodies=source_bodies,
                        collision_pipeline=lambda view: pipeline,
                        collide_interval=3,
                    )
                ]
            ),
        )
        state = model.state()
        key = ("source", "destination")
        config = coupled._proxy_collision_configs[key]

        coupled.step(state, state, control=None, contacts=None, dt=0.01)
        self.assertEqual(pipeline.collide_calls, 1)
        self.assertEqual(config.collide_counter, 1)
        self.assertIn(key, coupled._proxy_contact_stream_buffers)

        world_mask = wp.array([True, False], dtype=wp.bool, device=model.device)
        coupled.reset(state, world_mask=world_mask)

        self.assertEqual(pipeline.reset_masks, [world_mask])
        self.assertEqual(pipeline.reset_arities, [1])
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)
        self.assertEqual(config.collide_counter, 0)
        self.assertEqual(coupled._proxy_contact_stream_buffers, {})
        coupled.step(state, state, control=None, contacts=None, dt=0.01)
        self.assertEqual(pipeline.collide_calls, 2, "Reset must force a fresh proxy collision despite interval=3")

        contacts.rigid_contact_count.fill_(1)
        coupled.reset(state)
        self.assertEqual(pipeline.reset_masks, [world_mask, None])
        self.assertEqual(pipeline.reset_arities, [1, 0])
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)
        self.assertEqual(config.collide_counter, 0)

    def test_proxy_rejects_entry_and_mapping_collision_providers_for_same_direction(self):
        model, source_body, source_shape, destination_body, destination_shape = self._build_proxy_model()

        with self.assertRaisesRegex(ValueError, "both.*collision"):
            SolverCoupledProxy(
                model=model,
                entries=[
                    SolverCoupled.Entry(
                        name="source",
                        solver=_StepCountingCopySolver,
                        bodies=[source_body],
                        shapes=[source_shape],
                    ),
                    SolverCoupled.Entry(
                        name="destination",
                        solver=_ContactRecordingBodyHarvestSolver,
                        bodies=[destination_body],
                        shapes=[destination_shape],
                        collision_pipeline=lambda view: _FakeProxyCollisionPipeline(view.device),
                    ),
                ],
                coupling=SolverCoupledProxy.Config(
                    proxies=[
                        SolverCoupledProxy.Proxy(
                            source="source",
                            destination="destination",
                            bodies=[source_body],
                            collision_pipeline=lambda view: _FakeProxyCollisionPipeline(view.device),
                        )
                    ]
                ),
            )

    def test_proxy_rejects_self_direction_before_provider_ambiguity(self):
        """A self-directed proxy is invalid even when it would also share a provider."""
        model, source_body, source_shape, destination_body, destination_shape = self._build_proxy_model()

        with self.assertRaisesRegex(ValueError, "source and destination entries must differ"):
            SolverCoupledProxy(
                model=model,
                entries=[
                    SolverCoupled.Entry(
                        name="source",
                        solver=_StepCountingCopySolver,
                        bodies=[source_body],
                        shapes=[source_shape],
                    ),
                    SolverCoupled.Entry(
                        name="destination",
                        solver=_ContactRecordingBodyHarvestSolver,
                        bodies=[destination_body],
                        shapes=[destination_shape],
                        collision_pipeline=lambda view: _FakeProxyCollisionPipeline(view.device),
                    ),
                ],
                coupling=SolverCoupledProxy.Config(
                    proxies=[
                        SolverCoupledProxy.Proxy(
                            source="source",
                            destination="destination",
                            bodies=[source_body],
                        ),
                        SolverCoupledProxy.Proxy(
                            source="destination",
                            destination="destination",
                            bodies=[destination_body],
                            proxy_bodies=[source_body],
                        ),
                    ]
                ),
            )

    def test_proxy_rejects_entry_provider_shared_by_multiple_sources(self):
        """Keep the provider-ambiguity guard covered for a future multi-source proxy loop."""
        coupled = SolverCoupledProxy.__new__(SolverCoupledProxy)
        destination = mock.Mock(collision_pipeline=object())
        coupled._entries = {"destination": destination}
        coupled._proxy_groups = {
            ("source_a", "destination"): {},
            ("source_b", "destination"): {},
        }
        coupled._proxy_collision_configs = {}

        with self.assertRaisesRegex(ValueError, "multiple proxy sources"):
            coupled._validate_proxy_collision_providers()

    def test_proxy_destination_configure_view_is_copy_on_write_for_pipeline(self):
        """Destination proxy material overrides should not leak into the parent or source view."""
        model, source_body, source_shape, destination_body, destination_shape = self._build_proxy_model()
        parent_friction = model.shape_material_mu.numpy().copy()
        pipeline_views = []

        def configure_destination(view):
            friction = view.shape_material_mu.numpy().copy()
            friction[source_shape] = 4.0
            view.shape_material_mu = wp.array(friction, dtype=wp.float32, device=model.device)

        def collision_pipeline(view):
            pipeline_views.append((view, view.shape_material_mu.numpy().copy()))
            return _FakeProxyCollisionPipeline(view.device)

        coupled = SolverCoupledProxy(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="source",
                    solver=_StepCountingCopySolver,
                    bodies=[source_body],
                    shapes=[source_shape],
                ),
                SolverCoupled.Entry(
                    name="destination",
                    solver=_ContactRecordingBodyHarvestSolver,
                    bodies=[destination_body],
                    shapes=[destination_shape],
                    configure_view=configure_destination,
                    collision_pipeline=collision_pipeline,
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="source",
                        destination="destination",
                        bodies=[source_body],
                    )
                ]
            ),
        )

        np.testing.assert_array_equal(model.shape_material_mu.numpy(), parent_friction)
        self.assertEqual(float(coupled.view("source").shape_material_mu.numpy()[source_shape]), 0.25)
        self.assertEqual(float(coupled.view("destination").shape_material_mu.numpy()[source_shape]), 4.0)
        self.assertIs(pipeline_views[0][0], coupled.view("destination"))
        self.assertEqual(float(pipeline_views[0][1][source_shape]), 4.0)


class TestCoupledContactStreams(unittest.TestCase):
    """Test public zero-copy contact stream discovery."""

    @staticmethod
    def _build_proxy_replacement_solver(replacement_contacts, *, provider_rigid_max=2, provider_soft_max=1):
        model, source_body, source_shape, destination_body, destination_shape = (
            TestSolverCoupledEntryCollision._build_proxy_model()
        )
        provider_contacts = newton.Contacts(provider_rigid_max, provider_soft_max, device=model.device)
        pipeline = _FakeProxyCollisionPipeline(model.device, contacts=provider_contacts)
        coupled = SolverCoupledProxy(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="source",
                    solver=_StepCountingCopySolver,
                    bodies=[source_body],
                    shapes=[source_shape],
                ),
                SolverCoupled.Entry(
                    name="destination",
                    solver=lambda view: _ReplacingProxyContactsSolver(view, replacement_contacts),
                    bodies=[destination_body],
                    shapes=[destination_shape],
                    collision_pipeline=lambda view: pipeline,
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[SolverCoupledProxy.Proxy(source="source", destination="destination", bodies=[source_body])]
            ),
        )
        return model, coupled, provider_contacts

    def test_public_types_are_frozen_and_diagnostics_report_overflow(self):
        self.assertTrue(hasattr(coupled_api, "CoupledContactStream"))
        self.assertTrue(hasattr(coupled_api, "CoupledContactDiagnostics"))
        contacts = newton.Contacts(rigid_contact_max=2, soft_contact_max=1, device="cpu")
        contacts.rigid_contact_count.fill_(5)
        contacts.soft_contact_count.fill_(3)

        class CountingCounters:
            def __init__(self):
                self.read_count = 0

            def numpy(self):
                self.read_count += 1
                return np.array([5, 3], dtype=np.int32)

        packed_counters = CountingCounters()
        contacts.contact_counters = packed_counters

        stream = coupled_api.CoupledContactStream(
            name="outer",
            kind="outer",
            contacts=contacts,
            forces_available=False,
        )

        self.assertEqual(
            stream.diagnostics(),
            coupled_api.CoupledContactDiagnostics(
                rigid_count=5,
                rigid_capacity=2,
                rigid_overflow=3,
                soft_count=3,
                soft_capacity=1,
                soft_overflow=2,
            ),
        )
        self.assertEqual(packed_counters.read_count, 1)
        with self.assertRaises(FrozenInstanceError):
            stream.name = "replacement"
        with self.assertRaises(FrozenInstanceError):
            stream.diagnostics().rigid_count = 0

    def test_base_streams_are_stable_zero_copy_and_allocation_free(self):
        model = newton.ModelBuilder().finalize(device="cpu")
        coupled = SolverCoupled(
            model=model,
            entries=[SolverCoupled.Entry(name="entry", solver=_StepCountingCopySolver)],
        )
        self.assertTrue(hasattr(coupled, "contact_streams"))
        state = model.state()
        outer_contacts = newton.Contacts(rigid_contact_max=2, soft_contact_max=1, device=model.device)

        coupled.step(state, state, control=None, contacts=outer_contacts, dt=0.01)
        filtered_contacts = coupled._entry_contact_buffers["entry"]
        generations_before = (
            int(outer_contacts.contact_generation.numpy()[0]),
            int(filtered_contacts.contact_generation.numpy()[0]),
        )

        allocation_error = AssertionError("contact_streams allocated a Warp array")
        with (
            mock.patch.object(wp, "array", side_effect=allocation_error),
            mock.patch.object(wp, "zeros", side_effect=allocation_error),
            mock.patch.object(wp, "full", side_effect=allocation_error),
            mock.patch.object(wp, "empty", side_effect=allocation_error),
        ):
            streams = coupled.contact_streams()

        self.assertEqual([stream.name for stream in streams], ["outer", "entry/entry"])
        self.assertEqual([stream.kind for stream in streams], ["outer", "entry"])
        self.assertIs(streams[0].contacts, outer_contacts)
        self.assertIs(streams[1].contacts, filtered_contacts)
        self.assertIsNone(streams[0].shape_local_to_parent)
        self.assertIsNone(streams[0].particle_local_to_parent)
        self.assertIs(streams[1].shape_local_to_parent, coupled._entries["entry"].shape_local_to_global)
        self.assertIs(streams[1].particle_local_to_parent, coupled._entries["entry"].particle_local_to_global)
        self.assertTrue(all(not stream.forces_available for stream in streams))
        self.assertEqual(
            (
                int(outer_contacts.contact_generation.numpy()[0]),
                int(filtered_contacts.contact_generation.numpy()[0]),
            ),
            generations_before,
        )

        replacement_contacts = newton.Contacts(0, 0, device=model.device)
        replacement_streams = coupled.contact_streams(replacement_contacts)
        self.assertEqual([stream.name for stream in replacement_streams], ["outer"])
        self.assertIs(replacement_streams[0].contacts, replacement_contacts)

        coupled.reset(state)
        self.assertEqual(coupled.contact_streams(), ())

    def test_entry_provider_stream_exposes_compact_shape_map(self):
        builder = newton.ModelBuilder(gravity=0.0)
        body_a = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        body_b = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_shape_sphere(body=body_a, radius=0.1)
        shape_b = builder.add_shape_sphere(body=body_b, radius=0.1)
        model = builder.finalize(device="cpu")
        provider_contacts = newton.Contacts(1, 0, device=model.device)
        pipeline = _FakeProxyCollisionPipeline(model.device, contacts=provider_contacts)

        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="compact",
                    solver=_StepCountingCopySolver,
                    bodies=[body_b],
                    shapes=[shape_b],
                    preserve_shape_ids=False,
                    collision_pipeline=lambda view: pipeline,
                )
            ],
        )

        entry = coupled._entries["compact"]
        self.assertTrue(hasattr(entry, "shape_local_to_global"))
        self.assertTrue(hasattr(entry, "shape_global_to_local"))
        streams = coupled.contact_streams()
        self.assertEqual([stream.name for stream in streams], ["entry/compact"])
        stream = streams[0]
        self.assertEqual(stream.kind, "entry")
        self.assertIs(stream.contacts, provider_contacts)
        self.assertIs(stream.shape_local_to_parent, entry.shape_local_to_global)
        self.assertIs(stream.particle_local_to_parent, entry.particle_local_to_global)
        np.testing.assert_array_equal(stream.shape_local_to_parent.numpy(), np.array([shape_b], dtype=np.int32))
        np.testing.assert_array_equal(entry.shape_global_to_local.numpy(), np.array([-1, 0], dtype=np.int32))
        self.assertFalse(stream.forces_available)

    def test_proxy_streams_include_directional_provider_alias(self):
        _StepCountingCopySolver.instances.clear()
        _ContactRecordingBodyHarvestSolver.instances.clear()
        builder = newton.ModelBuilder(gravity=0.0)
        source_body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        source_shape = builder.add_shape_sphere(body=source_body, radius=0.1)
        destination_body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        destination_shape = builder.add_shape_sphere(body=destination_body, radius=0.1)
        model = builder.finalize(device="cpu")
        provider_contacts = newton.Contacts(1, 0, device=model.device)
        pipeline = _FakeProxyCollisionPipeline(model.device, contacts=provider_contacts)
        coupled = SolverCoupledProxy(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="source",
                    solver=_StepCountingCopySolver,
                    bodies=[source_body],
                    shapes=[source_shape],
                ),
                SolverCoupled.Entry(
                    name="destination",
                    solver=_ContactRecordingBodyHarvestSolver,
                    bodies=[destination_body],
                    shapes=[destination_shape],
                    collision_pipeline=lambda view: pipeline,
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[SolverCoupledProxy.Proxy(source="source", destination="destination", bodies=[source_body])]
            ),
        )

        coupled.step(model.state(), model.state(), control=None, contacts=None, dt=0.01)
        streams = coupled.contact_streams()

        self.assertEqual(
            [(stream.name, stream.kind) for stream in streams],
            [("entry/destination", "entry"), ("proxy/source/destination", "proxy")],
        )
        entry_stream, proxy_stream = streams
        self.assertIs(entry_stream.contacts, provider_contacts)
        self.assertIs(proxy_stream.contacts, provider_contacts)
        self.assertEqual((proxy_stream.source, proxy_stream.destination), ("source", "destination"))
        self.assertIs(
            proxy_stream.shape_local_to_parent,
            coupled._entries["destination"].shape_local_to_global,
        )
        self.assertIs(
            proxy_stream.particle_local_to_parent,
            coupled._entries["destination"].particle_local_to_global,
        )
        self.assertFalse(entry_stream.forces_available)
        self.assertFalse(proxy_stream.forces_available)

    def test_failed_proxy_step_hides_all_streams_until_success(self):
        _ConfigurableFailingCopySolver.instances.clear()
        model, body_a, shape_a, body_b, shape_b = TestSolverCoupledEntryCollision._build_proxy_model()
        coupled = SolverCoupledProxy(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="A",
                    solver=_ConfigurableFailingCopySolver,
                    bodies=[body_a],
                    shapes=[shape_a],
                ),
                SolverCoupled.Entry(
                    name="B",
                    solver=_ConfigurableFailingCopySolver,
                    bodies=[body_b],
                    shapes=[shape_b],
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(source="A", destination="B", bodies=[body_a]),
                    SolverCoupledProxy.Proxy(source="B", destination="A", bodies=[body_b]),
                ]
            ),
        )
        state = model.state()
        successful_contacts = newton.Contacts(1, 0, device=model.device)
        coupled.step(state, state, control=None, contacts=successful_contacts, dt=0.01)
        self.assertIs(coupled.contact_streams()[0].contacts, successful_contacts)

        solver_a = _ConfigurableFailingCopySolver.instances["A"]
        solver_a.fail_on_step = solver_a.step_count + 2
        failed_contacts = newton.Contacts(2, 0, device=model.device)
        with self.assertRaisesRegex(RuntimeError, "intentional failure in A"):
            coupled.step(state, state, control=None, contacts=failed_contacts, dt=0.01)

        self.assertEqual(coupled.contact_streams(), ())
        self.assertEqual(coupled.contact_streams(failed_contacts), ())

        solver_a.fail_on_step = None
        recovered_contacts = newton.Contacts(3, 0, device=model.device)
        coupled.step(state, state, control=None, contacts=recovered_contacts, dt=0.01)
        recovered_streams = coupled.contact_streams()
        self.assertIs(recovered_streams[0].contacts, recovered_contacts)
        self.assertIn("proxy/A/B", [stream.name for stream in recovered_streams])
        self.assertIn("proxy/B/A", [stream.name for stream in recovered_streams])

    def test_proxy_mapping_pipeline_stream_preserves_get_proxy_contacts(self):
        builder = newton.ModelBuilder(gravity=0.0)
        source_body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        source_shape = builder.add_shape_sphere(body=source_body, radius=0.1)
        destination_body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        destination_shape = builder.add_shape_sphere(body=destination_body, radius=0.1)
        model = builder.finalize(device="cpu")
        proxy_contacts = newton.Contacts(1, 0, device=model.device)
        pipeline = _FakeProxyCollisionPipeline(model.device, contacts=proxy_contacts)
        coupled = SolverCoupledProxy(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="source",
                    solver=_StepCountingCopySolver,
                    bodies=[source_body],
                    shapes=[source_shape],
                ),
                SolverCoupled.Entry(
                    name="destination",
                    solver=_ContactRecordingBodyHarvestSolver,
                    bodies=[destination_body],
                    shapes=[destination_shape],
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="source",
                        destination="destination",
                        bodies=[source_body],
                        collision_pipeline=lambda view: pipeline,
                    )
                ]
            ),
        )

        streams = coupled.contact_streams()

        self.assertEqual([stream.name for stream in streams], ["proxy/source/destination"])
        self.assertIs(streams[0].contacts, proxy_contacts)
        self.assertIs(coupled.get_proxy_contacts("source", "destination"), proxy_contacts)

    def test_proxy_default_direction_is_absent_before_first_step(self):
        model, source_body, source_shape, destination_body, destination_shape = (
            TestSolverCoupledEntryCollision._build_proxy_model()
        )
        coupled = SolverCoupledProxy(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="source",
                    solver=_StepCountingCopySolver,
                    bodies=[source_body],
                    shapes=[source_shape],
                ),
                SolverCoupled.Entry(
                    name="destination",
                    solver=_StepCountingCopySolver,
                    bodies=[destination_body],
                    shapes=[destination_shape],
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[SolverCoupledProxy.Proxy(source="source", destination="destination", bodies=[source_body])]
            ),
        )

        try:
            streams = coupled.contact_streams()
        except UnboundLocalError as error:
            self.fail(f"contact_streams read an uninitialized direction buffer: {error}")

        self.assertEqual(streams, ())

    def test_proxy_post_reset_fallback_does_not_reuse_prior_direction_buffer(self):
        model, body_a, shape_a, body_b, shape_b = TestSolverCoupledEntryCollision._build_proxy_model()
        provider_contacts = newton.Contacts(1, 0, device=model.device)
        pipeline = _FakeProxyCollisionPipeline(model.device, contacts=provider_contacts)
        coupled = SolverCoupledProxy(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="A",
                    solver=_StepCountingCopySolver,
                    bodies=[body_a],
                    shapes=[shape_a],
                ),
                SolverCoupled.Entry(
                    name="B",
                    solver=_StepCountingCopySolver,
                    bodies=[body_b],
                    shapes=[shape_b],
                    collision_pipeline=lambda view: pipeline,
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(source="A", destination="B", bodies=[body_a]),
                    SolverCoupledProxy.Proxy(source="B", destination="A", bodies=[body_b]),
                ]
            ),
        )
        state = model.state()
        coupled.step(state, state, control=None, contacts=None, dt=0.01)

        coupled.reset(state)
        streams = coupled.contact_streams()

        self.assertEqual(
            [stream.name for stream in streams],
            ["entry/B", "proxy/A/B"],
        )
        self.assertIs(streams[0].contacts, provider_contacts)
        self.assertIs(streams[1].contacts, provider_contacts)

    def test_proxy_outer_contact_path_exposes_actual_directional_buffer(self):
        _StepCountingCopySolver.instances.clear()
        _ContactRecordingBodyHarvestSolver.instances.clear()
        builder = newton.ModelBuilder(gravity=0.0)
        source_body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        source_shape = builder.add_shape_sphere(body=source_body, radius=0.1)
        destination_body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        destination_shape = builder.add_shape_sphere(body=destination_body, radius=0.1)
        model = builder.finalize(device="cpu")
        coupled = SolverCoupledProxy(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="source",
                    solver=_StepCountingCopySolver,
                    bodies=[source_body],
                    shapes=[source_shape],
                ),
                SolverCoupled.Entry(
                    name="destination",
                    solver=_ContactRecordingBodyHarvestSolver,
                    bodies=[destination_body],
                    shapes=[destination_shape],
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[SolverCoupledProxy.Proxy(source="source", destination="destination", bodies=[source_body])]
            ),
        )
        outer_contacts = newton.Contacts(1, 0, device=model.device)

        coupled.step(model.state(), model.state(), control=None, contacts=outer_contacts, dt=0.01)
        streams = {stream.name: stream for stream in coupled.contact_streams()}

        self.assertIn("proxy/source/destination", streams)
        actual_contacts = _ContactRecordingBodyHarvestSolver.instances["destination"].step_contacts[0]
        self.assertIs(streams["proxy/source/destination"].contacts, actual_contacts)
        self.assertIs(actual_contacts, coupled._entry_contact_buffers["destination"])

    def test_proxy_stream_omits_direction_when_prepare_hook_returns_none(self):
        _StepCountingCopySolver.instances.clear()
        _DroppingProxyContactsSolver.instances.clear()
        builder = newton.ModelBuilder(gravity=0.0)
        source_body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        source_shape = builder.add_shape_sphere(body=source_body, radius=0.1)
        destination_body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        destination_shape = builder.add_shape_sphere(body=destination_body, radius=0.1)
        model = builder.finalize(device="cpu")
        provider_contacts = newton.Contacts(1, 0, device=model.device)
        pipeline = _FakeProxyCollisionPipeline(model.device, contacts=provider_contacts)
        coupled = SolverCoupledProxy(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="source",
                    solver=_StepCountingCopySolver,
                    bodies=[source_body],
                    shapes=[source_shape],
                ),
                SolverCoupled.Entry(
                    name="destination",
                    solver=_DroppingProxyContactsSolver,
                    bodies=[destination_body],
                    shapes=[destination_shape],
                    collision_pipeline=lambda view: pipeline,
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[SolverCoupledProxy.Proxy(source="source", destination="destination", bodies=[source_body])]
            ),
        )

        coupled.step(model.state(), model.state(), control=None, contacts=None, dt=0.01)
        streams = coupled.contact_streams()

        self.assertEqual([stream.name for stream in streams], ["entry/destination"])
        self.assertIsNone(_DroppingProxyContactsSolver.instances["destination"].step_contacts[0])

    def test_proxy_distinct_replacement_uses_destination_stream_metadata(self):
        replacement_contacts = newton.Contacts(2, 1, device="cpu")
        model, coupled, provider_contacts = self._build_proxy_replacement_solver(replacement_contacts)

        coupled.step(model.state(), model.state(), control=None, contacts=None, dt=0.01)
        streams = {stream.name: stream for stream in coupled.contact_streams()}

        self.assertIs(streams["entry/destination"].contacts, provider_contacts)
        proxy_stream = streams["proxy/source/destination"]
        self.assertIs(proxy_stream.contacts, replacement_contacts)
        self.assertEqual(proxy_stream.kind, "proxy")
        self.assertEqual((proxy_stream.source, proxy_stream.destination), ("source", "destination"))
        self.assertIs(proxy_stream.shape_local_to_parent, coupled._entries["destination"].shape_local_to_global)
        self.assertIs(
            proxy_stream.particle_local_to_parent,
            coupled._entries["destination"].particle_local_to_global,
        )

    def test_proxy_replacement_requires_contacts_type(self):
        model, coupled, _ = self._build_proxy_replacement_solver(object())

        try:
            with self.assertRaisesRegex(TypeError, "coupling_prepare_proxy_contacts.*Contacts"):
                coupled.step(model.state(), model.state(), control=None, contacts=None, dt=0.01)
        except AttributeError as error:
            self.fail(f"invalid proxy contacts reached the destination solver: {error}")

    def test_proxy_replacement_requires_matching_capacities(self):
        replacement_contacts = newton.Contacts(1, 0, device="cpu")
        model, coupled, _ = self._build_proxy_replacement_solver(replacement_contacts)

        with self.assertRaisesRegex(ValueError, "coupling_prepare_proxy_contacts.*capacities"):
            coupled.step(model.state(), model.state(), control=None, contacts=None, dt=0.01)


class TestSolverCoupledReset(unittest.TestCase):
    """World-mask reset validation and entry forwarding."""

    @staticmethod
    def _devices():
        devices = ["cpu"]
        if wp.is_cuda_available():
            devices.append("cuda:0")
        return devices

    @staticmethod
    def _transforms(values):
        values = np.asarray(values, dtype=np.float32)
        transforms = np.zeros((len(values), 7), dtype=np.float32)
        transforms[:, 0] = values
        transforms[:, 6] = 1.0
        return transforms

    @staticmethod
    def _build_recording_coupled(device="cpu"):
        template = newton.ModelBuilder(gravity=0.0)
        template.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder = newton.ModelBuilder(gravity=0.0)
        builder.replicate(template, world_count=2)
        model = builder.finalize(device=device)
        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="recording",
                    solver=_ResetRecordingCopySolver,
                    bodies=[0, 1],
                )
            ],
        )
        return model, coupled

    @staticmethod
    def _build_masked_coupled(device):
        template = newton.ModelBuilder(gravity=0.0)
        for x in (0.0, 1.0):
            body = template.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            joint = template.add_joint_revolute(parent=-1, child=body, axis=(0.0, 0.0, 1.0))
            template.add_articulation([joint])
            template.add_particle(pos=(x, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0)

        builder = newton.ModelBuilder(gravity=0.0)
        builder.replicate(template, world_count=2)
        model = builder.finalize(device=device)
        model.request_state_attributes("body_qdd", "body_parent_f")
        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="rigid",
                    solver=_ResetRecordingCopySolver,
                    bodies=[1, 3],
                    joints=[1, 3],
                    substeps=2,
                ),
                SolverCoupled.Entry(
                    name="particles",
                    solver=_ResetRecordingCopySolver,
                    particles=[1, 3],
                    substeps=2,
                ),
            ],
        )
        return model, coupled

    @staticmethod
    def _seed_spatial(array, values):
        rows = np.repeat(np.asarray(values, dtype=np.float32)[:, None], 6, axis=1)
        array.assign(rows)
        return rows

    @staticmethod
    def _seed_vec3(array, values):
        rows = np.repeat(np.asarray(values, dtype=np.float32)[:, None], 3, axis=1)
        array.assign(rows)
        return rows

    def _assert_invalid_mask_does_not_mutate(self, world_mask, error_type, message):
        model, coupled = self._build_recording_coupled()
        state = model.state()
        entry = coupled._entries["recording"]
        entry.state_0.body_qd.assign(np.full((2, 6), 3.0, dtype=np.float32))
        entry.state_1.body_qd.assign(np.full((2, 6), 7.0, dtype=np.float32))
        state_0_before = entry.state_0.body_qd.numpy().copy()
        state_1_before = entry.state_1.body_qd.numpy().copy()
        stream_marker = object()
        coupled._last_outer_contacts = stream_marker
        coupled._contact_streams_valid = True
        coupled._entry_output_state_valid = True

        with self.assertRaisesRegex(error_type, message):
            coupled.reset(state, world_mask=world_mask)

        self.assertIs(coupled._last_outer_contacts, stream_marker)
        self.assertTrue(coupled._contact_streams_valid)
        self.assertTrue(coupled._entry_output_state_valid)
        self.assertEqual(entry.solver.reset_calls, [])
        np.testing.assert_array_equal(entry.state_0.body_qd.numpy(), state_0_before)
        np.testing.assert_array_equal(entry.state_1.body_qd.numpy(), state_1_before)

    def test_reset_rejects_invalid_world_masks_before_mutation(self):
        cases = (
            ("type", [True, False], TypeError, "Warp array"),
            (
                "dtype",
                wp.array([1, 0], dtype=wp.int32, device="cpu"),
                TypeError,
                "dtype.*bool",
            ),
            (
                "ndim",
                wp.array(np.array([[True, False]], dtype=np.bool_), dtype=wp.bool, device="cpu"),
                ValueError,
                "one-dimensional",
            ),
            (
                "length",
                wp.array([True], dtype=wp.bool, device="cpu"),
                ValueError,
                "length.*world_count",
            ),
        )
        for name, world_mask, error_type, message in cases:
            with self.subTest(name=name):
                self._assert_invalid_mask_does_not_mutate(world_mask, error_type, message)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_reset_rejects_world_mask_on_wrong_device_before_mutation(self):
        self._assert_invalid_mask_does_not_mutate(
            wp.array([True, False], dtype=wp.bool, device="cuda:0"),
            ValueError,
            "device.*model device",
        )

    def test_reset_forwards_exact_parent_mask_and_flags(self):
        model, coupled = self._build_recording_coupled()
        state = model.state()
        world_mask = wp.array([False, True], dtype=wp.bool, device=model.device)
        flags = newton.StateFlags.BODY_Q

        coupled.reset(state, world_mask=world_mask, flags=flags)

        reset_state, forwarded_mask, forwarded_flags = coupled._entries["recording"].solver.reset_calls[0]
        self.assertIs(reset_state, coupled._entries["recording"].state_0)
        self.assertIs(forwarded_mask, world_mask)
        self.assertIs(forwarded_flags, flags)

    def test_partial_reset_syncs_selected_compact_rows_and_preserves_unselected_rows(self):
        for device in self._devices():
            with self.subTest(device=device):
                model, coupled = self._build_masked_coupled(device)
                state = model.state()
                body_q = self._transforms([10.0, 20.0, 30.0, 40.0])
                body_qd = np.arange(24, dtype=np.float32).reshape(4, 6) + 10.0
                body_f = np.arange(24, dtype=np.float32).reshape(4, 6) + 100.0
                particle_q = np.arange(12, dtype=np.float32).reshape(4, 3) + 20.0
                particle_qd = np.arange(12, dtype=np.float32).reshape(4, 3) + 40.0
                particle_f = np.arange(12, dtype=np.float32).reshape(4, 3) + 60.0
                joint_q = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
                joint_qd = np.array([11.0, 12.0, 13.0, 14.0], dtype=np.float32)
                state.body_q.assign(body_q)
                state.body_qd.assign(body_qd)
                state.body_f.assign(body_f)
                state.particle_q.assign(particle_q)
                state.particle_qd.assign(particle_qd)
                state.particle_f.assign(particle_f)
                state.joint_q.assign(joint_q)
                state.joint_qd.assign(joint_qd)

                rigid = coupled._entries["rigid"]
                particles = coupled._entries["particles"]
                np.testing.assert_array_equal(rigid.body_local_to_global.numpy(), [1, 3])
                np.testing.assert_array_equal(rigid.joint_coord_local_to_global.numpy(), [1, 3])

                rigid_public_before = {}
                rigid_transient_before = {}
                for index, entry_state in enumerate((rigid.state_1, rigid.state_tmp), start=1):
                    q = self._transforms([100.0 * index + 1.0, 100.0 * index + 2.0])
                    qd = np.full((2, 6), 100.0 * index + 3.0, dtype=np.float32)
                    jq = np.array([100.0 * index + 4.0, 100.0 * index + 5.0], dtype=np.float32)
                    jqd = np.array([100.0 * index + 6.0, 100.0 * index + 7.0], dtype=np.float32)
                    entry_state.body_q.assign(q)
                    entry_state.body_qd.assign(qd)
                    entry_state.joint_q.assign(jq)
                    entry_state.joint_qd.assign(jqd)
                    rigid_public_before[index] = (q, qd, jq, jqd)
                    rigid_transient_before[index] = {
                        "body_f": self._seed_spatial(entry_state.body_f, [10.0 * index + 1.0, 10.0 * index + 2.0]),
                        "body_qdd": self._seed_spatial(entry_state.body_qdd, [10.0 * index + 3.0, 10.0 * index + 4.0]),
                        "body_parent_f": self._seed_spatial(
                            entry_state.body_parent_f, [10.0 * index + 5.0, 10.0 * index + 6.0]
                        ),
                    }
                self._seed_spatial(rigid.state_0.body_qdd, [31.0, 32.0])
                self._seed_spatial(rigid.state_0.body_parent_f, [33.0, 34.0])

                particle_public_before = {}
                particle_force_before = {}
                for index, entry_state in enumerate((particles.state_1, particles.state_tmp), start=1):
                    q = np.full((4, 3), 200.0 * index + 1.0, dtype=np.float32)
                    qd = np.full((4, 3), 200.0 * index + 2.0, dtype=np.float32)
                    entry_state.particle_q.assign(q)
                    entry_state.particle_qd.assign(qd)
                    particle_public_before[index] = (q, qd)
                    particle_force_before[index] = self._seed_vec3(
                        entry_state.particle_f,
                        [20.0 * index + row for row in range(4)],
                    )

                world_mask = wp.array([True, False], dtype=wp.bool, device=device)
                coupled.reset(state, world_mask=world_mask)

                for index, entry_state in enumerate((rigid.state_1, rigid.state_tmp), start=1):
                    q_before, qd_before, jq_before, jqd_before = rigid_public_before[index]
                    expected_q = q_before.copy()
                    expected_q[0] = body_q[1]
                    expected_qd = qd_before.copy()
                    expected_qd[0] = body_qd[1]
                    expected_jq = jq_before.copy()
                    expected_jq[0] = joint_q[1]
                    expected_jqd = jqd_before.copy()
                    expected_jqd[0] = joint_qd[1]
                    np.testing.assert_array_equal(entry_state.body_q.numpy(), expected_q)
                    np.testing.assert_array_equal(entry_state.body_qd.numpy(), expected_qd)
                    np.testing.assert_array_equal(entry_state.joint_q.numpy(), expected_jq)
                    np.testing.assert_array_equal(entry_state.joint_qd.numpy(), expected_jqd)
                    for name, before in rigid_transient_before[index].items():
                        expected = before.copy()
                        expected[0] = 0.0
                        np.testing.assert_array_equal(getattr(entry_state, name).numpy(), expected)

                expected_state_0_body_f = body_f[[1, 3]].copy()
                expected_state_0_body_f[0] = 0.0
                np.testing.assert_array_equal(rigid.state_0.body_f.numpy(), expected_state_0_body_f)
                np.testing.assert_array_equal(rigid.state_0.body_qdd.numpy()[0], 0.0)
                np.testing.assert_array_equal(rigid.state_0.body_qdd.numpy()[1], 32.0)
                np.testing.assert_array_equal(rigid.state_0.body_parent_f.numpy()[0], 0.0)
                np.testing.assert_array_equal(rigid.state_0.body_parent_f.numpy()[1], 34.0)

                selected_particles = np.array([True, True, False, False])
                for index, entry_state in enumerate((particles.state_1, particles.state_tmp), start=1):
                    q_before, qd_before = particle_public_before[index]
                    expected_q = q_before.copy()
                    expected_qd = qd_before.copy()
                    expected_q[selected_particles] = particle_q[selected_particles]
                    expected_qd[selected_particles] = particle_qd[selected_particles]
                    np.testing.assert_array_equal(entry_state.particle_q.numpy(), expected_q)
                    np.testing.assert_array_equal(entry_state.particle_qd.numpy(), expected_qd)
                    expected_f = particle_force_before[index].copy()
                    expected_f[selected_particles] = 0.0
                    np.testing.assert_array_equal(entry_state.particle_f.numpy(), expected_f)

                expected_state_0_particle_f = particle_f.copy()
                expected_state_0_particle_f[selected_particles] = 0.0
                np.testing.assert_array_equal(particles.state_0.particle_f.numpy(), expected_state_0_particle_f)
                np.testing.assert_array_equal(state.body_q.numpy(), body_q)
                np.testing.assert_array_equal(state.body_qd.numpy(), body_qd)
                np.testing.assert_array_equal(state.particle_q.numpy(), particle_q)
                np.testing.assert_array_equal(state.particle_qd.numpy(), particle_qd)
                np.testing.assert_array_equal(state.joint_q.numpy(), joint_q)
                np.testing.assert_array_equal(state.joint_qd.numpy(), joint_qd)

    def test_partial_reset_respects_state_flags_but_always_invalidates_selected_transients(self):
        model, coupled = self._build_masked_coupled("cpu")
        state = model.state()
        body_q = self._transforms([10.0, 20.0, 30.0, 40.0])
        body_qd = np.full((4, 6), 20.0, dtype=np.float32)
        particle_q = np.full((4, 3), 30.0, dtype=np.float32)
        particle_qd = np.arange(12, dtype=np.float32).reshape(4, 3) + 40.0
        joint_q = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        joint_qd = np.array([11.0, 12.0, 13.0, 14.0], dtype=np.float32)
        state.body_q.assign(body_q)
        state.body_qd.assign(body_qd)
        state.particle_q.assign(particle_q)
        state.particle_qd.assign(particle_qd)
        state.joint_q.assign(joint_q)
        state.joint_qd.assign(joint_qd)

        rigid = coupled._entries["rigid"]
        particles = coupled._entries["particles"]
        for entry_state in (rigid.state_1, rigid.state_tmp):
            entry_state.body_q.assign(self._transforms([101.0, 102.0]))
            entry_state.body_qd.assign(np.full((2, 6), 103.0, dtype=np.float32))
            entry_state.joint_q.assign(np.array([104.0, 105.0], dtype=np.float32))
            entry_state.joint_qd.assign(np.array([106.0, 107.0], dtype=np.float32))
            self._seed_spatial(entry_state.body_f, [108.0, 109.0])
        for entry_state in (particles.state_1, particles.state_tmp):
            entry_state.particle_q.assign(np.full((4, 3), 201.0, dtype=np.float32))
            entry_state.particle_qd.assign(np.full((4, 3), 202.0, dtype=np.float32))
            self._seed_vec3(entry_state.particle_f, [203.0, 204.0, 205.0, 206.0])

        flags = newton.StateFlags.BODY_Q | newton.StateFlags.PARTICLE_QD | newton.StateFlags.JOINT_Q
        coupled.reset(state, wp.array([True, False], dtype=wp.bool, device="cpu"), flags=flags)

        for entry_state in (rigid.state_1, rigid.state_tmp):
            np.testing.assert_array_equal(entry_state.body_q.numpy()[0], body_q[1])
            np.testing.assert_array_equal(entry_state.body_q.numpy()[1], self._transforms([102.0])[0])
            np.testing.assert_array_equal(entry_state.body_qd.numpy(), 103.0)
            np.testing.assert_array_equal(entry_state.joint_q.numpy(), [joint_q[1], 105.0])
            np.testing.assert_array_equal(entry_state.joint_qd.numpy(), [106.0, 107.0])
            np.testing.assert_array_equal(entry_state.body_f.numpy()[0], 0.0)
            np.testing.assert_array_equal(entry_state.body_f.numpy()[1], 109.0)
        for entry_state in (particles.state_1, particles.state_tmp):
            np.testing.assert_array_equal(entry_state.particle_q.numpy(), 201.0)
            np.testing.assert_array_equal(entry_state.particle_qd.numpy()[:2], particle_qd[:2])
            np.testing.assert_array_equal(entry_state.particle_qd.numpy()[2:], 202.0)
            np.testing.assert_array_equal(entry_state.particle_f.numpy()[:2], 0.0)
            np.testing.assert_array_equal(entry_state.particle_f.numpy()[2, :], 205.0)
            np.testing.assert_array_equal(entry_state.particle_f.numpy()[3, :], 206.0)
        np.testing.assert_array_equal(state.body_q.numpy(), body_q)
        np.testing.assert_array_equal(state.body_qd.numpy(), body_qd)
        np.testing.assert_array_equal(state.particle_q.numpy(), particle_q)
        np.testing.assert_array_equal(state.particle_qd.numpy(), particle_qd)
        np.testing.assert_array_equal(state.joint_q.numpy(), joint_q)
        np.testing.assert_array_equal(state.joint_qd.numpy(), joint_qd)

    def test_partial_reset_excludes_global_rows_and_full_reset_keeps_bulk_parity(self):
        for device in self._devices():
            with self.subTest(device=device):
                template = newton.ModelBuilder(gravity=0.0)
                template.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
                builder = newton.ModelBuilder(gravity=0.0)
                builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
                builder.add_world(template)
                builder.add_world(template)
                model = builder.finalize(device=device)
                model.request_state_attributes("body_qdd", "body_parent_f")
                coupled = SolverCoupled(
                    model,
                    [
                        SolverCoupled.Entry(
                            name="all",
                            solver=_ResetRecordingCopySolver,
                            bodies=[0, 1, 2],
                            joints=[0, 1, 2],
                            substeps=2,
                        )
                    ],
                )
                entry = coupled._entries["all"]
                np.testing.assert_array_equal(entry.view.body_world.numpy(), [-1, 0, 1])
                state = model.state()
                body_q = self._transforms([10.0, 20.0, 30.0])
                body_qd = np.arange(18, dtype=np.float32).reshape(3, 6)
                joint_q = np.arange(model.joint_coord_count, dtype=np.float32) + 10.0
                joint_qd = np.arange(model.joint_dof_count, dtype=np.float32) + 20.0
                state.body_q.assign(body_q)
                state.body_qd.assign(body_qd)
                state.joint_q.assign(joint_q)
                state.joint_qd.assign(joint_qd)

                partial_q_before = {}
                partial_joint_q_before = {}
                for index, entry_state in enumerate((entry.state_1, entry.state_tmp), start=1):
                    q = self._transforms([100.0 * index + 1.0, 100.0 * index + 2.0, 100.0 * index + 3.0])
                    jq = np.arange(model.joint_coord_count, dtype=np.float32) + 100.0 * index
                    entry_state.body_q.assign(q)
                    entry_state.body_qd.assign(np.full((3, 6), 100.0 * index + 4.0, dtype=np.float32))
                    entry_state.joint_q.assign(jq)
                    entry_state.joint_qd.assign(np.arange(model.joint_dof_count, dtype=np.float32) + 200.0 * index)
                    partial_q_before[index] = q
                    partial_joint_q_before[index] = jq

                world_mask = wp.array([True, False], dtype=wp.bool, device=device)
                coupled.reset(state, world_mask=world_mask)

                for index, entry_state in enumerate((entry.state_1, entry.state_tmp), start=1):
                    expected_q = partial_q_before[index].copy()
                    expected_q[1] = body_q[1]
                    np.testing.assert_array_equal(entry_state.body_q.numpy(), expected_q)
                    expected_joint_q = partial_joint_q_before[index].copy()
                    expected_joint_q[7:14] = joint_q[7:14]
                    np.testing.assert_array_equal(entry_state.joint_q.numpy(), expected_joint_q)
                np.testing.assert_array_equal(state.body_q.numpy(), body_q)
                np.testing.assert_array_equal(state.joint_q.numpy(), joint_q)

                for entry_state in (entry.state_1, entry.state_tmp):
                    entry_state.body_q.assign(self._transforms([901.0, 902.0, 903.0]))
                    entry_state.body_qd.assign(np.full((3, 6), 904.0, dtype=np.float32))
                    entry_state.joint_q.assign(np.full(model.joint_coord_count, 905.0, dtype=np.float32))
                    entry_state.joint_qd.assign(np.full(model.joint_dof_count, 906.0, dtype=np.float32))
                    self._seed_spatial(entry_state.body_f, [907.0, 908.0, 909.0])
                    self._seed_spatial(entry_state.body_qdd, [910.0, 911.0, 912.0])
                    self._seed_spatial(entry_state.body_parent_f, [913.0, 914.0, 915.0])

                coupled.reset(state)

                for entry_state in (entry.state_1, entry.state_tmp):
                    np.testing.assert_array_equal(entry_state.body_q.numpy(), body_q)
                    np.testing.assert_array_equal(entry_state.body_qd.numpy(), body_qd)
                    np.testing.assert_array_equal(entry_state.joint_q.numpy(), joint_q)
                    np.testing.assert_array_equal(entry_state.joint_qd.numpy(), joint_qd)
                    np.testing.assert_array_equal(entry_state.body_f.numpy(), 0.0)
                    np.testing.assert_array_equal(entry_state.body_qdd.numpy(), 0.0)
                    np.testing.assert_array_equal(entry_state.body_parent_f.numpy(), 0.0)
                self.assertIs(entry.solver.reset_calls[-1][1], None)


class TestSolverCoupledBasic(unittest.TestCase):
    """Test SolverCoupled with two SemiImplicit solvers (simplest case)."""

    def setUp(self):
        builder = newton.ModelBuilder()

        # Two bodies: body 0 owned by solver A, body 1 owned by solver B
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_body(mass=2.0, inertia=wp.mat33(np.eye(3)))
        builder.add_shape_sphere(body=0, radius=0.1)
        builder.add_shape_sphere(body=1, radius=0.2)

        self.model = builder.finalize(device="cpu")

    def test_entry_control_arrays_are_mapped_to_local_dofs(self):
        """Entry solvers should receive control arrays in their local DOF namespace."""
        _ControlRecordingSolver.instances.clear()
        builder = newton.ModelBuilder()
        body_a = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        body_b = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        joint_a = builder.add_joint_revolute(parent=-1, child=body_a, axis=(0.0, 0.0, 1.0))
        joint_b = builder.add_joint_revolute(parent=-1, child=body_b, axis=(0.0, 0.0, 1.0))
        builder.add_articulation([joint_a])
        builder.add_articulation([joint_b])
        model = builder.finalize(device="cpu")

        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(name="A", solver=_ControlRecordingSolver, bodies=[body_a], joints=[joint_a]),
                SolverCoupled.Entry(name="B", solver=_ControlRecordingSolver, bodies=[body_b], joints=[joint_b]),
            ],
        )
        control = model.control()
        control.joint_f.assign(np.array([3.0, 7.0], dtype=np.float32))
        control.joint_target_q.assign(np.array([11.0, 13.0], dtype=np.float32))

        coupled.step(model.state(), model.state(), control, contacts=None, dt=1.0 / 60.0)

        solver_a, solver_b = _ControlRecordingSolver.instances
        np.testing.assert_array_equal(solver_a.joint_f[0], np.array([3.0], dtype=np.float32))
        np.testing.assert_array_equal(solver_b.joint_f[0], np.array([7.0], dtype=np.float32))
        np.testing.assert_array_equal(solver_a.joint_target_q[0], np.array([11.0], dtype=np.float32))
        np.testing.assert_array_equal(solver_b.joint_target_q[0], np.array([13.0], dtype=np.float32))

    def test_notify_model_changed_refreshes_view_inertial_masks(self):
        """Runtime parent inertial edits should refresh derived view masks."""
        coupled = SolverCoupled(
            model=self.model,
            entries=[
                SolverCoupled.Entry(name="A", solver=_StepCountingCopySolver, bodies=[0]),
                SolverCoupled.Entry(name="B", solver=_StepCountingCopySolver, bodies=[1]),
            ],
        )

        self.model.body_inv_mass.assign(np.array([0.25, 0.125], dtype=np.float32))
        coupled.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)

        view_a_inv_mass = coupled.view("A").body_inv_mass.numpy()
        view_b_inv_mass = coupled.view("B").body_inv_mass.numpy()
        np.testing.assert_allclose(view_a_inv_mass, [0.25])
        np.testing.assert_allclose(view_b_inv_mass, [0.125])

    def test_entry_shapes_filter_shape_contact_pairs(self):
        """Entry shape masks should prune explicit contact pairs in each view."""
        self.assertEqual(self.model.shape_contact_pair_count, 1)

        coupled = SolverCoupled(
            model=self.model,
            entries=[
                SolverCoupled.Entry(name="A", solver=SolverSemiImplicit, bodies=[0], shapes=[0]),
                SolverCoupled.Entry(name="B", solver=SolverSemiImplicit, bodies=[1], shapes=[1]),
            ],
        )

        collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
        view_a = coupled.view("A")
        view_b = coupled.view("B")
        flags_a = view_a.shape_flags.numpy()
        flags_b = view_b.shape_flags.numpy()

        self.assertEqual(view_a.shape_flags.shape[0], self.model.shape_count)
        self.assertNotEqual(int(flags_a[0]) & collide, 0)
        self.assertEqual(int(flags_a[1]) & collide, 0)
        np.testing.assert_array_equal(view_a.shape_body.numpy(), np.array([0, -1], dtype=np.int32))
        self.assertEqual(view_a.shape_contact_pair_count, 0)

        self.assertEqual(view_b.shape_flags.shape[0], self.model.shape_count)
        self.assertEqual(int(flags_b[0]) & collide, 0)
        self.assertNotEqual(int(flags_b[1]) & collide, 0)
        np.testing.assert_array_equal(view_b.shape_body.numpy(), np.array([-1, 0], dtype=np.int32))
        self.assertEqual(view_b.shape_contact_pair_count, 0)

        self.assertEqual(self.model.shape_contact_pair_count, 1)

    def test_entries_preserve_global_shape_ids_by_default(self):
        """Entry shape views should keep global shape arrays with hidden dummies."""
        coupled = SolverCoupled(
            model=self.model,
            entries=[
                SolverCoupled.Entry(
                    name="A",
                    solver=SolverSemiImplicit,
                    bodies=[0],
                    shapes=[0],
                ),
                SolverCoupled.Entry(name="B", solver=SolverSemiImplicit, bodies=[1], shapes=[1]),
            ],
        )

        view_a = coupled.view("A")
        flags = view_a.shape_flags.numpy()
        collide = int(newton.ShapeFlags.COLLIDE_SHAPES)

        self.assertEqual(view_a.body_count, 1)
        self.assertEqual(view_a.shape_count, self.model.shape_count)
        self.assertEqual(view_a.shape_flags.shape[0], self.model.shape_count)
        np.testing.assert_array_equal(view_a.shape_body.numpy(), np.array([0, -1], dtype=np.int32))
        self.assertEqual(view_a.body_shapes, {-1: [], 0: [0]})
        self.assertNotEqual(int(flags[0]) & collide, 0)
        self.assertEqual(int(flags[1]) & collide, 0)
        self.assertEqual(view_a.shape_contact_pair_count, 0)

    def test_particle_entry_without_shapes_keeps_global_static_shapes(self):
        """Particle-only entries should inherit global static shapes by default."""
        builder = newton.ModelBuilder()
        ground_shape = builder.add_ground_plane()
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        dynamic_shape = builder.add_shape_sphere(body=body, radius=0.1)
        particle = builder.add_particle(pos=(0.0, 0.0, 0.5), vel=(0.0, 0.0, 0.0), mass=1.0)
        model = builder.finalize(device="cpu")

        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(name="particles", solver=SolverSemiImplicit, particles=[particle]),
            ],
        )

        view = coupled.view("particles")
        flags = view.shape_flags.numpy()
        collide_particles = int(newton.ShapeFlags.COLLIDE_PARTICLES)

        self.assertEqual(view.shape_count, model.shape_count)
        self.assertEqual(view.body_shapes[-1], [ground_shape])
        self.assertNotIn(dynamic_shape, view.body_shapes[-1])
        self.assertNotEqual(int(flags[ground_shape]) & collide_particles, 0)
        self.assertEqual(int(flags[dynamic_shape]) & collide_particles, 0)
        body_shape_ids = np.array(view.body_shapes[-1], dtype=int)
        particle_collider_shapes = body_shape_ids[(flags[body_shape_ids] & collide_particles) > 0]
        np.testing.assert_array_equal(particle_collider_shapes, np.array([ground_shape], dtype=int))

    def test_particles_keep_global_connectivity_while_rigid_domains_compact(self):
        """Particle identity mappings must not prevent independent rigid compaction."""
        builder = newton.ModelBuilder()
        for mass in (1.0, 2.0, 3.0):
            body = builder.add_body(mass=mass, inertia=wp.mat33(np.eye(3)))
            builder.add_shape_sphere(body=body, radius=0.1)
        for x in (0.0, 1.0):
            builder.add_particle(pos=(x, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0)
        builder.add_spring(0, 1, ke=1.0, kd=0.1, control=0.0)
        model = builder.finalize(device="cpu")

        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="mixed",
                    solver=SolverSemiImplicit,
                    bodies=[2],
                    particles=[0],
                    shapes=[2],
                    preserve_shape_ids=False,
                )
            ],
        )
        view = coupled.view("mixed")

        self.assertEqual(view.body_count, 1)
        self.assertEqual(view.shape_count, 1)
        np.testing.assert_allclose(view.body_mass.numpy(), model.body_mass.numpy()[2:3])
        np.testing.assert_array_equal(view.shape_body.numpy(), [0])
        self.assertEqual(view.body_shapes, {-1: [], 0: [0]})

        self.assertEqual(view.particle_count, model.particle_count)
        self.assertEqual(view.spring_count, model.spring_count)
        np.testing.assert_array_equal(view.spring_indices.numpy(), [0, 1])

    def test_entry_can_compact_shape_ids_when_requested(self):
        """Entry views should still support compact local shape ids by opt-out."""
        coupled = SolverCoupled(
            model=self.model,
            entries=[
                SolverCoupled.Entry(
                    name="A",
                    solver=SolverSemiImplicit,
                    bodies=[0],
                    shapes=[0],
                    preserve_shape_ids=False,
                ),
                SolverCoupled.Entry(
                    name="B",
                    solver=SolverSemiImplicit,
                    bodies=[1],
                    shapes=[1],
                    preserve_shape_ids=False,
                ),
            ],
        )

        view_a = coupled.view("A")
        view_b = coupled.view("B")

        self.assertEqual(view_a.shape_count, 1)
        self.assertEqual(view_a.shape_flags.shape[0], 1)
        np.testing.assert_array_equal(view_a.shape_body.numpy(), np.array([0], dtype=np.int32))
        self.assertEqual(view_b.shape_count, 1)
        self.assertEqual(view_b.shape_flags.shape[0], 1)
        np.testing.assert_array_equal(view_b.shape_body.numpy(), np.array([0], dtype=np.int32))

    def test_preserved_global_shape_ids_remap_hidden_shapes_in_mixed_views(self):
        """Preserved shape ids should not leave hidden shapes attached to omitted bodies."""
        builder = newton.ModelBuilder()
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_shape_sphere(body=0, radius=0.1)
        builder.add_shape_sphere(body=1, radius=0.1)
        builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0)
        model = builder.finalize(device="cpu")

        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="A",
                    solver=SolverSemiImplicit,
                    bodies=[0],
                    particles=[0],
                    shapes=[0],
                ),
                SolverCoupled.Entry(name="B", solver=SolverSemiImplicit, bodies=[1], shapes=[1]),
            ],
        )

        view_a = coupled.view("A")

        self.assertEqual(view_a.body_count, 1)
        self.assertEqual(view_a.particle_count, 1)
        self.assertEqual(view_a.shape_count, model.shape_count)
        np.testing.assert_array_equal(view_a.shape_body.numpy(), np.array([0, -1], dtype=np.int32))
        self.assertEqual(view_a.body_shapes, {-1: [], 0: [0]})

    def test_proxy_shape_visibility_keeps_proxy_contact_pairs(self):
        """Proxy destination views should keep shape pairs touching proxy bodies."""
        coupled = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupled.Entry(name="A", solver=SolverSemiImplicit, bodies=[0], shapes=[0]),
                SolverCoupled.Entry(name="B", solver=SolverSemiImplicit, bodies=[1], shapes=[1]),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(source="A", destination="B", bodies=[0]),
                ],
            ),
        )

        collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
        view_a = coupled.view("A")
        view_b = coupled.view("B")

        self.assertEqual(view_a.shape_contact_pair_count, 0)
        self.assertNotEqual(int(view_b.shape_flags.numpy()[0]) & collide, 0)
        self.assertNotEqual(int(view_b.shape_flags.numpy()[1]) & collide, 0)
        self.assertEqual(view_b.shape_contact_pair_count, 1)
        np.testing.assert_array_equal(view_b.shape_contact_pairs.numpy(), np.array([[0, 1]], dtype=np.int32))

    def test_proxy_harvest_uses_filtered_preserved_shape_contacts(self):
        """Custom proxy harvest should receive the contacts used by the step."""
        _StepCountingCopySolver.instances.clear()
        _ContactRecordingBodyHarvestSolver.instances.clear()

        builder = newton.ModelBuilder(gravity=0.0)
        ground_shape = builder.add_ground_plane()
        src_body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        src_shape = builder.add_shape_sphere(body=src_body, radius=0.1)
        dst_body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        dst_shape = builder.add_shape_sphere(body=dst_body, radius=0.1)
        hidden_body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        hidden_shape = builder.add_shape_sphere(body=hidden_body, radius=0.1)
        model = builder.finalize(device="cpu")

        coupled = SolverCoupledProxy(
            model=model,
            entries=[
                SolverCoupled.Entry(name="src", solver=_StepCountingCopySolver, bodies=[src_body], shapes=[src_shape]),
                SolverCoupled.Entry(
                    name="dst",
                    solver=_ContactRecordingBodyHarvestSolver,
                    bodies=[dst_body],
                    shapes=[ground_shape, dst_shape],
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(source="src", destination="dst", bodies=[src_body]),
                ],
            ),
        )

        contacts = newton.Contacts(2, 0, device=model.device)
        contacts.rigid_contact_count.assign(np.array([2], dtype=np.int32))
        contacts.rigid_contact_shape0.assign(np.array([ground_shape, ground_shape], dtype=np.int32))
        contacts.rigid_contact_shape1.assign(np.array([dst_shape, hidden_shape], dtype=np.int32))

        coupled.step(model.state(), model.state(), control=None, contacts=contacts, dt=1.0 / 60.0)

        dst_solver = _ContactRecordingBodyHarvestSolver.instances["dst"]
        self.assertEqual(len(dst_solver.step_contacts), 1)
        self.assertEqual(len(dst_solver.harvest_contacts), 1)
        self.assertIs(dst_solver.harvest_contacts[0], dst_solver.step_contacts[0])
        self.assertIsNot(dst_solver.step_contacts[0], contacts)
        self.assertEqual(int(dst_solver.step_contacts[0].rigid_contact_count.numpy()[0]), 1)
        np.testing.assert_array_equal(dst_solver.rigid_shape1_steps[0], np.array([dst_shape], dtype=np.int32))

    def test_proxy_collision_contacts_bypass_preserved_shape_filter(self):
        """Proxy-local contacts are already generated in the destination view."""
        _StepCountingCopySolver.instances.clear()
        _ContactRecordingBodyHarvestSolver.instances.clear()

        builder = newton.ModelBuilder(gravity=0.0)
        ground_shape = builder.add_ground_plane()
        src_body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        src_shape = builder.add_shape_sphere(body=src_body, radius=0.1)
        dst_body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        dst_shape = builder.add_shape_sphere(body=dst_body, radius=0.1)
        model = builder.finalize(device="cpu")

        proxy_contacts = newton.Contacts(1, 0, device=model.device)
        proxy_contacts.rigid_contact_count.assign(np.array([1], dtype=np.int32))
        proxy_contacts.rigid_contact_shape0.assign(np.array([ground_shape], dtype=np.int32))
        proxy_contacts.rigid_contact_shape1.assign(np.array([dst_shape], dtype=np.int32))

        def make_pipeline(view):
            del view
            return _FakeProxyCollisionPipeline(model.device, contacts=proxy_contacts)

        coupled = SolverCoupledProxy(
            model=model,
            entries=[
                SolverCoupled.Entry(name="src", solver=_StepCountingCopySolver, bodies=[src_body], shapes=[src_shape]),
                SolverCoupled.Entry(
                    name="dst",
                    solver=_ContactRecordingBodyHarvestSolver,
                    bodies=[dst_body],
                    shapes=[ground_shape, dst_shape],
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="src",
                        destination="dst",
                        bodies=[src_body],
                        collision_pipeline=make_pipeline,
                    ),
                ],
            ),
        )

        coupled.step(model.state(), model.state(), control=None, contacts=None, dt=1.0 / 60.0)

        dst_solver = _ContactRecordingBodyHarvestSolver.instances["dst"]
        self.assertEqual(len(dst_solver.step_contacts), 1)
        self.assertEqual(len(dst_solver.harvest_contacts), 1)
        self.assertIs(dst_solver.step_contacts[0], proxy_contacts)
        self.assertIs(dst_solver.harvest_contacts[0], proxy_contacts)
        self.assertIs(coupled.get_proxy_contacts("src", "dst"), proxy_contacts)

    def test_duplicate_shape_ownership_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "owned by more than one"):
            SolverCoupled(
                model=self.model,
                entries=[
                    SolverCoupled.Entry(name="A", solver=SolverSemiImplicit, bodies=[0], shapes=[0]),
                    SolverCoupled.Entry(name="B", solver=SolverSemiImplicit, bodies=[1], shapes=[0]),
                ],
            )

    def test_step(self):
        """SolverCoupled.step() should advance both bodies."""
        coupled = SolverCoupled(
            model=self.model,
            entries=[
                SolverCoupled.Entry(name="A", solver=SolverSemiImplicit, bodies=[0]),
                SolverCoupled.Entry(name="B", solver=SolverSemiImplicit, bodies=[1]),
            ],
        )

        state_0 = self.model.state()
        state_1 = self.model.state()
        contacts = self.model.collide(state_0)

        # Step and check bodies moved (due to gravity)
        coupled.step(state_0, state_1, control=None, contacts=contacts, dt=1.0 / 60.0)

        q0_before = state_0.body_q.numpy()
        q1_after = state_1.body_q.numpy()

        # Bodies should have fallen under gravity
        for i in range(2):
            self.assertFalse(
                np.allclose(q0_before[i], q1_after[i]),
                f"Body {i} did not move after step",
            )

    def test_entry_in_place_steps_same_state(self):
        """Entries can opt into same-object state input/output stepping."""
        _InPlaceRecordingParticleSolver.instances.clear()
        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.0)
        model = builder.finalize(device="cpu")

        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="particles",
                    solver=lambda v: _InPlaceRecordingParticleSolver(model=v),
                    particles=[0],
                    in_place=True,
                ),
            ],
        )

        state = model.state()

        coupled.step(state, state, control=None, contacts=None, dt=1.0 / 60.0)

        solver = _InPlaceRecordingParticleSolver.instances["particles"]
        self.assertEqual(solver.in_place_calls, [True])
        np.testing.assert_allclose(state.particle_qd.numpy()[0], np.array([0.0, 2.0, 0.0]))

    def test_entry_in_place_substeps_same_state(self):
        """In-place entries can substep without allocating scratch states."""
        _InPlaceRecordingParticleSolver.instances.clear()
        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.0)
        model = builder.finalize(device="cpu")

        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="particles",
                    solver=lambda v: _InPlaceRecordingParticleSolver(model=v),
                    particles=[0],
                    substeps=3,
                    in_place=True,
                ),
            ],
        )

        state = model.state()
        coupled.step(state, state, control=None, contacts=None, dt=0.3)

        solver = _InPlaceRecordingParticleSolver.instances["particles"]
        self.assertEqual(solver.in_place_calls, [True, True, True])
        np.testing.assert_allclose(solver.dt_values, [0.1, 0.1, 0.1])
        np.testing.assert_allclose(state.particle_qd.numpy()[0], np.array([0.0, 6.0, 0.0]))

    def test_particle_views_deactivate_non_owned_particles(self):
        """Each particle owner view should expose only its owned particles as active."""
        builder = newton.ModelBuilder()
        builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0)
        builder.add_particle(pos=(0.1, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0)
        model = builder.finalize(device="cpu")

        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(name="A", solver=SolverSemiImplicit, particles=[0]),
                SolverCoupled.Entry(name="B", solver=SolverSemiImplicit, particles=[1]),
            ],
        )

        active = int(newton.ParticleFlags.ACTIVE)
        view_a_flags = coupled.view("A").particle_flags.numpy()
        view_b_flags = coupled.view("B").particle_flags.numpy()
        parent_flags = model.particle_flags.numpy()

        self.assertEqual(view_a_flags.shape[0], 2)
        self.assertNotEqual(view_a_flags[0] & active, 0)
        self.assertEqual(view_a_flags[1] & active, 0)
        self.assertEqual(view_b_flags[0] & active, 0)
        self.assertNotEqual(view_b_flags[1] & active, 0)
        self.assertNotEqual(parent_flags[0] & active, 0)
        self.assertNotEqual(parent_flags[1] & active, 0)

    def test_proxy_destination_view_marks_proxy_flags(self):
        """Proxy destination views should expose proxy bodies through body_flags."""
        coupled = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupled.Entry(name="A", solver=SolverSemiImplicit, bodies=[0]),
                SolverCoupled.Entry(name="B", solver=SolverSemiImplicit, bodies=[1]),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(source="A", destination="B", bodies=[0]),
                ],
            ),
        )

        view_a = coupled.view("A")
        view_b = coupled.view("B")
        proxy_flag = int(newton.BodyFlags.PROXY)

        self.assertEqual(view_a.body_flags.numpy()[0] & proxy_flag, 0)
        self.assertNotEqual(view_b.body_flags.numpy()[0] & proxy_flag, 0)
        self.assertEqual(self.model.body_flags.numpy()[0] & proxy_flag, 0)
        self.assertGreater(view_b.body_inv_mass.numpy()[0], 0.0)

    def test_proxy_coupling_rejects_more_than_two_entries(self):
        """Generic proxy coupling is currently limited to one solver pair."""
        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        model = builder.finalize(device="cpu")

        with self.assertRaisesRegex(ValueError, "at most two solver entries"):
            SolverCoupledProxy(
                model=model,
                entries=[
                    SolverCoupled.Entry(name="a", solver=SolverSemiImplicit, bodies=[0]),
                    SolverCoupled.Entry(name="b", solver=SolverSemiImplicit, bodies=[1]),
                    SolverCoupled.Entry(name="c", solver=SolverSemiImplicit, bodies=[2]),
                ],
                coupling=SolverCoupledProxy.Config(
                    proxies=[
                        SolverCoupledProxy.Proxy(source="a", destination="b", bodies=[0]),
                    ],
                ),
            )

    def test_proxy_coupling_rejects_invalid_numerical_config(self):
        entries = [
            SolverCoupled.Entry(name="A", solver=SolverSemiImplicit, bodies=[0]),
            SolverCoupled.Entry(name="B", solver=SolverSemiImplicit, bodies=[1]),
        ]

        for mass_scale in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(mass_scale=mass_scale), self.assertRaisesRegex(ValueError, "mass_scale"):
                SolverCoupledProxy(
                    model=self.model,
                    entries=entries,
                    coupling=SolverCoupledProxy.Config(
                        proxies=[
                            SolverCoupledProxy.Proxy(
                                source="A",
                                destination="B",
                                bodies=[0],
                                mass_scale=mass_scale,
                            )
                        ]
                    ),
                )

        for iterations in (0, -1, 1.5, float("nan")):
            with self.subTest(iterations=iterations), self.assertRaisesRegex(ValueError, "iterations"):
                SolverCoupledProxy(
                    model=self.model,
                    entries=entries,
                    coupling=SolverCoupledProxy.Config(
                        proxies=[SolverCoupledProxy.Proxy(source="A", destination="B", bodies=[0])],
                        iterations=iterations,
                    ),
                )

        for collide_interval in (0, -1, 1.5, float("nan")):
            with (
                self.subTest(collide_interval=collide_interval),
                self.assertRaisesRegex(ValueError, "collide_interval"),
            ):
                SolverCoupledProxy(
                    model=self.model,
                    entries=entries,
                    coupling=SolverCoupledProxy.Config(
                        proxies=[
                            SolverCoupledProxy.Proxy(
                                source="A",
                                destination="B",
                                bodies=[0],
                                collision_pipeline=lambda model: None,
                                collide_interval=collide_interval,
                            )
                        ]
                    ),
                )

    def test_proxy_coupling_rejects_unowned_source_body(self):
        with self.assertRaisesRegex(ValueError, "owned by source entry"):
            SolverCoupledProxy(
                model=self.model,
                entries=[
                    SolverCoupled.Entry(name="A", solver=SolverSemiImplicit, bodies=[0]),
                    SolverCoupled.Entry(name="B", solver=SolverSemiImplicit, bodies=[1]),
                ],
                coupling=SolverCoupledProxy.Config(
                    proxies=[SolverCoupledProxy.Proxy(source="A", destination="B", bodies=[1])]
                ),
            )

    def test_proxy_coupling_rejects_destination_owned_proxy_body(self):
        """Proxy body ids must not alias bodies owned by the destination."""
        builder = newton.ModelBuilder(gravity=0.0)
        body0 = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        body1 = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        model = builder.finalize(device="cpu")

        with self.assertRaisesRegex(ValueError, "owned by destination entry"):
            SolverCoupledProxy(
                model=model,
                entries=[
                    SolverCoupled.Entry(name="src", solver=SolverSemiImplicit, bodies=[body0]),
                    SolverCoupled.Entry(name="dst", solver=SolverSemiImplicit, bodies=[body1]),
                ],
                coupling=SolverCoupledProxy.Config(
                    proxies=[
                        SolverCoupledProxy.Proxy(
                            source="src",
                            destination="dst",
                            bodies=[body0],
                            proxy_bodies=[body1],
                        ),
                    ],
                ),
            )

    def test_proxy_coupling_rejects_destination_owned_proxy_particle(self):
        """Proxy particle ids must not alias particles owned by the destination."""
        builder = newton.ModelBuilder(gravity=0.0)
        particle0 = builder.add_particle(
            pos=(0.0, 0.0, 0.0),
            vel=(0.0, 0.0, 0.0),
            mass=1.0,
        )
        particle1 = builder.add_particle(
            pos=(1.0, 0.0, 0.0),
            vel=(0.0, 0.0, 0.0),
            mass=1.0,
        )
        model = builder.finalize(device="cpu")

        with self.assertRaisesRegex(ValueError, "owned by destination entry"):
            SolverCoupledProxy(
                model=model,
                entries=[
                    SolverCoupled.Entry(name="src", solver=SolverSemiImplicit, particles=[particle0]),
                    SolverCoupled.Entry(name="dst", solver=SolverSemiImplicit, particles=[particle1]),
                ],
                coupling=SolverCoupledProxy.Config(
                    proxies=[
                        SolverCoupledProxy.Proxy(
                            source="src",
                            destination="dst",
                            particles=[particle0],
                            proxy_particles=[particle1],
                        ),
                    ],
                ),
            )


class TestSolverCoupledProxyMappingValidation(unittest.TestCase):
    """Cross-record Proxy mappings reject ambiguous destination aliases."""

    @staticmethod
    def _build_model():
        builder = newton.ModelBuilder(gravity=0.0)
        source_body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        proxy_body_a = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        proxy_body_b = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        source_joint = builder.add_joint_prismatic(parent=-1, child=source_body, axis=(1.0, 0.0, 0.0))
        proxy_joint_a = builder.add_joint_prismatic(parent=-1, child=proxy_body_a, axis=(1.0, 0.0, 0.0))
        proxy_joint_b = builder.add_joint_prismatic(parent=-1, child=proxy_body_b, axis=(1.0, 0.0, 0.0))
        builder.add_articulation([source_joint])
        builder.add_articulation([proxy_joint_a])
        builder.add_articulation([proxy_joint_b])
        source_particle = builder.add_particle(
            pos=(0.0, 0.0, 0.0),
            vel=(0.0, 0.0, 0.0),
            mass=1.0,
            radius=0.0,
        )
        proxy_particle_a = builder.add_particle(
            pos=(1.0, 0.0, 0.0),
            vel=(0.0, 0.0, 0.0),
            mass=1.0,
            radius=0.0,
        )
        proxy_particle_b = builder.add_particle(
            pos=(2.0, 0.0, 0.0),
            vel=(0.0, 0.0, 0.0),
            mass=1.0,
            radius=0.0,
        )
        model = builder.finalize(device="cpu")
        ids = {
            "source_body": source_body,
            "proxy_body_a": proxy_body_a,
            "proxy_body_b": proxy_body_b,
            "source_joint": source_joint,
            "proxy_joint_a": proxy_joint_a,
            "proxy_joint_b": proxy_joint_b,
            "source_particle": source_particle,
            "proxy_particle_a": proxy_particle_a,
            "proxy_particle_b": proxy_particle_b,
        }
        return model, ids

    @staticmethod
    def _make_coupled(model, ids, proxies):
        return SolverCoupledProxy(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="source",
                    solver=_StepCountingCopySolver,
                    bodies=[ids["source_body"]],
                    particles=[ids["source_particle"]],
                    joints=[ids["source_joint"]],
                ),
                SolverCoupled.Entry(name="destination", solver=_StepCountingCopySolver),
            ],
            coupling=SolverCoupledProxy.Config(proxies=proxies),
        )

    def test_cross_record_destination_proxy_ids_must_be_unique(self):
        model, ids = self._build_model()
        source_body = ids["source_body"]
        source_particle = ids["source_particle"]
        source_joint = ids["source_joint"]

        overlapping = {
            "body": [
                SolverCoupledProxy.Proxy(
                    source="source",
                    destination="destination",
                    bodies=[source_body],
                    proxy_bodies=[ids["proxy_body_a"]],
                ),
                SolverCoupledProxy.Proxy(
                    source="source",
                    destination="destination",
                    bodies=[source_body],
                    proxy_bodies=[ids["proxy_body_a"]],
                ),
            ],
            "particle": [
                SolverCoupledProxy.Proxy(
                    source="source",
                    destination="destination",
                    particles=[source_particle],
                    proxy_particles=[ids["proxy_particle_a"]],
                ),
                SolverCoupledProxy.Proxy(
                    source="source",
                    destination="destination",
                    particles=[source_particle],
                    proxy_particles=[ids["proxy_particle_a"]],
                ),
            ],
            "joint": [
                SolverCoupledProxy.Proxy(
                    source="source",
                    destination="destination",
                    bodies=[source_body],
                    proxy_bodies=[ids["proxy_body_a"]],
                    joints=[source_joint],
                    proxy_joints=[ids["proxy_joint_a"]],
                ),
                SolverCoupledProxy.Proxy(
                    source="source",
                    destination="destination",
                    bodies=[source_body],
                    proxy_bodies=[ids["proxy_body_b"]],
                    joints=[source_joint],
                    proxy_joints=[ids["proxy_joint_a"]],
                ),
            ],
        }

        for entity_kind, proxies in overlapping.items():
            with (
                self.subTest(entity_kind=entity_kind),
                self.assertRaisesRegex(
                    ValueError,
                    rf"Proxy destination {entity_kind}.*multiple Proxy records",
                ),
            ):
                self._make_coupled(model, ids, proxies)

    def test_source_entities_can_fan_out_to_distinct_destination_proxy_ids(self):
        model, ids = self._build_model()
        proxies = [
            SolverCoupledProxy.Proxy(
                source="source",
                destination="destination",
                bodies=[ids["source_body"]],
                proxy_bodies=[ids["proxy_body_a"]],
                particles=[ids["source_particle"]],
                proxy_particles=[ids["proxy_particle_a"]],
                joints=[ids["source_joint"]],
                proxy_joints=[ids["proxy_joint_a"]],
            ),
            SolverCoupledProxy.Proxy(
                source="source",
                destination="destination",
                bodies=[ids["source_body"]],
                proxy_bodies=[ids["proxy_body_b"]],
                particles=[ids["source_particle"]],
                proxy_particles=[ids["proxy_particle_b"]],
                joints=[ids["source_joint"]],
                proxy_joints=[ids["proxy_joint_b"]],
            ),
        ]

        coupled = self._make_coupled(model, ids, proxies)

        self.assertEqual(len(coupled._proxy_mappings), 2)
        self.assertEqual(len(coupled._proxy_particle_mappings), 2)
        self.assertEqual(len(coupled._proxy_joint_mappings), 2)


class TestSolverMuJoCoCouplingHooks(unittest.TestCase):
    """MuJoCo-specific coupling hook behavior."""

    def test_effective_inertia_preserves_anisotropic_free_body_inertia(self):
        try:
            SolverMuJoCo.import_mujoco()
        except ImportError as exc:
            self.skipTest(str(exc))

        builder = newton.ModelBuilder(gravity=0.0)
        body = builder.add_link(
            mass=2.0,
            inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.5),
        )
        joint = builder.add_joint_free(child=body)
        builder.add_articulation([joint])
        model = builder.finalize(device="cpu")
        solver = SolverMuJoCo(model=model, iterations=1, disable_contacts=True)

        endpoint_kind = wp.array([int(CouplingEndpointKind.BODY)], dtype=int, device=model.device)
        endpoint_index = wp.array([body], dtype=int, device=model.device)
        endpoint_local_pos = wp.zeros(1, dtype=wp.vec3, device=model.device)
        effective_mass = wp.empty(1, dtype=float, device=model.device)
        effective_inertia = wp.empty(1, dtype=wp.mat33, device=model.device)
        solver.coupling_eval_effective_mass_block(
            endpoint_kind,
            endpoint_index,
            endpoint_local_pos,
            effective_mass,
            effective_inertia,
        )

        np.testing.assert_allclose(effective_mass.numpy(), model.body_mass.numpy(), rtol=1.0e-5)
        np.testing.assert_allclose(effective_inertia.numpy(), model.body_inertia.numpy(), rtol=1.0e-5)

    def test_gravity_acceleration_hook_uses_body_gravcomp(self):
        try:
            SolverMuJoCo.import_mujoco()
        except ImportError as exc:
            self.skipTest(str(exc))

        builder = newton.ModelBuilder(gravity=-10.0, up_axis=newton.Axis.Z)
        SolverMuJoCo.register_custom_attributes(builder)

        body0 = builder.add_link(
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
            custom_attributes={"mujoco:gravcomp": 0.0},
        )
        body1 = builder.add_link(
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
            custom_attributes={"mujoco:gravcomp": 0.5},
        )
        body2 = builder.add_link(
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
            custom_attributes={"mujoco:gravcomp": 1.0},
        )
        builder.add_shape_box(body=body0, hx=0.05, hy=0.05, hz=0.05)
        builder.add_shape_box(body=body1, hx=0.05, hy=0.05, hz=0.05)
        builder.add_shape_box(body=body2, hx=0.05, hy=0.05, hz=0.05)
        joint0 = builder.add_joint_revolute(parent=-1, child=body0, axis=(0.0, 0.0, 1.0))
        joint1 = builder.add_joint_revolute(parent=body0, child=body1, axis=(0.0, 1.0, 0.0))
        joint2 = builder.add_joint_revolute(parent=body1, child=body2, axis=(1.0, 0.0, 0.0))
        builder.add_articulation([joint0, joint1, joint2])
        model = builder.finalize(device="cpu")

        solver = SolverMuJoCo(model=model, iterations=1, disable_contacts=True)
        body_acceleration = wp.empty(model.body_count, dtype=wp.vec3, device=model.device)
        solver.coupling_eval_gravity_acceleration(body_acceleration, None)

        np.testing.assert_allclose(
            body_acceleration.numpy(),
            np.array([[0.0, 0.0, -10.0], [0.0, 0.0, -5.0], [0.0, 0.0, 0.0]], dtype=np.float32),
            atol=1.0e-6,
        )

        model.mujoco.gravcomp.assign(np.array([0.25, 0.5, 0.75], dtype=np.float32))
        solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
        solver.coupling_eval_gravity_acceleration(body_acceleration, None)

        np.testing.assert_allclose(
            body_acceleration.numpy(),
            np.array([[0.0, 0.0, -7.5], [0.0, 0.0, -5.0], [0.0, 0.0, -2.5]], dtype=np.float32),
            atol=1.0e-6,
        )


class TestSolverCoupledProxyJoints(unittest.TestCase):
    """Proxy joints preserve source drive commands in destination solves."""

    def test_cross_world_joint_proxy_mapping_is_rejected(self):
        builder = newton.ModelBuilder(gravity=0.0)
        builder.begin_world()
        source_body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        source_joint = builder.add_joint_revolute(parent=-1, child=source_body, axis=(0.0, 0.0, 1.0))
        builder.add_articulation([source_joint])
        builder.end_world()
        builder.begin_world()
        proxy_body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        proxy_joint = builder.add_joint_revolute(parent=-1, child=proxy_body, axis=(0.0, 0.0, 1.0))
        builder.add_articulation([proxy_joint])
        builder.end_world()
        model = builder.finalize(device="cpu")

        with self.assertRaisesRegex(ValueError, "source joint.*same world"):
            SolverCoupledProxy(
                model=model,
                entries=[
                    SolverCoupled.Entry(
                        name="src",
                        solver=_ControlRecordingSolver,
                        bodies=[source_body],
                        joints=[source_joint],
                    ),
                    SolverCoupled.Entry(name="dst", solver=_ControlRecordingSolver, bodies=[proxy_body]),
                ],
                coupling=SolverCoupledProxy.Config(
                    proxies=[
                        SolverCoupledProxy.Proxy(
                            source="src",
                            destination="dst",
                            joints=[source_joint],
                            proxy_joints=[proxy_joint],
                        )
                    ]
                ),
            )

    def test_aliased_proxy_joint_copies_control_target_each_iteration(self):
        _ControlRecordingSolver.instances.clear()
        builder = newton.ModelBuilder(gravity=0.0)
        source_body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        proxy_body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        source_joint = builder.add_joint_prismatic(parent=-1, child=source_body, axis=(1.0, 0.0, 0.0))
        proxy_joint = builder.add_joint_prismatic(parent=-1, child=proxy_body, axis=(1.0, 0.0, 0.0))
        builder.add_articulation([source_joint])
        builder.add_articulation([proxy_joint])
        model = builder.finalize(device="cpu")

        coupled = SolverCoupledProxy(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="src",
                    solver=_ControlRecordingSolver,
                    bodies=[source_body],
                    joints=[source_joint],
                ),
                SolverCoupled.Entry(name="dst", solver=_ControlRecordingSolver, bodies=[proxy_body]),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="src",
                        destination="dst",
                        joints=[source_joint],
                        proxy_joints=[proxy_joint],
                    )
                ],
                iterations=3,
            ),
        )
        control = model.control()
        control.joint_target_q.assign(np.array([0.25, 0.75], dtype=np.float32))
        control.joint_target_qd.assign(np.array([0.5, 1.5], dtype=np.float32))

        coupled.step(model.state(), model.state(), control, contacts=None, dt=1.0 / 60.0)

        source_solver, destination_solver = _ControlRecordingSolver.instances
        self.assertEqual(len(source_solver.joint_target_q), 3)
        self.assertEqual(len(destination_solver.joint_target_q), 3)
        for target_q, target_qd in zip(
            destination_solver.joint_target_q,
            destination_solver.joint_target_qd,
            strict=True,
        ):
            np.testing.assert_array_equal(target_q, np.array([0.25], dtype=np.float32))
            np.testing.assert_array_equal(target_qd, np.array([0.5], dtype=np.float32))


class TestSolverCoupledMuJoCoVBDMultiEnv(unittest.TestCase):
    """Regression tests for multi-world MuJoCo/VBD solver partitions."""

    def test_compacted_multi_world_articulation_end_is_rebased(self):
        """articulation_end must be rebased to local joint ids, matching articulation_start.

        Regression: compaction rebased articulation_start but left articulation_end as
        global joint indices, so a non-first-world articulation got an out-of-bounds
        end (e.g. end=9 in an 8-joint view), corrupting solver FK (fixed base displaced).
        """
        world_count = 2
        template = newton.ModelBuilder(gravity=0.0)

        # Articulation A: fixed base + one revolute link (the "rigid" entry).
        base = template.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), label="base")
        jf = template.add_joint_fixed(parent=-1, child=base)
        link = template.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), label="link")
        jr = template.add_joint_revolute(parent=base, child=link, axis=(0.0, 0.0, 1.0))
        template.add_articulation([jf, jr])
        # Articulation B: a free body owned by the other entry.
        free_body = template.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), label="free")
        jfree = template.add_joint_free(child=free_body)
        template.add_articulation([jfree])

        builder = newton.ModelBuilder(gravity=0.0)
        builder.replicate(template, world_count=world_count)
        builder.color()
        model = builder.finalize(device="cpu")

        bpw, jpw = template.body_count, template.joint_count

        def expand(ids, stride):
            return [w * stride + i for w in range(world_count) for i in ids]

        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="rigid",
                    solver=SolverSemiImplicit,
                    bodies=expand([base, link], bpw),
                    joints=expand([jf, jr], jpw),
                ),
                SolverCoupled.Entry(
                    name="free",
                    solver=SolverSemiImplicit,
                    bodies=expand([free_body], bpw),
                    joints=expand([jfree], jpw),
                ),
            ],
        )

        view = coupled.view("rigid")
        starts = view.articulation_start.numpy()
        ends = view.articulation_end.numpy()
        # Two articulations (one per world), each spanning 2 joints in the 4-joint view.
        self.assertEqual(starts.tolist(), [0, 2, 4])
        self.assertEqual(ends.tolist(), [2, 4])
        # End indices must stay within the compacted joint range (no OOB).
        self.assertTrue(all(e <= view.joint_count for e in ends))


class TestSolverCoupledBodyProxyInertia(unittest.TestCase):
    """Body proxy mappings install full proxy inertia tensors."""

    @staticmethod
    def _entry_body_local(coupled: SolverCoupledProxy, entry_name: str, body_id: int) -> int:
        return int(coupled._entries[entry_name].body_global_to_local.numpy()[body_id])

    def test_body_proxy_aitken_relaxation_converges_affine_fixed_point(self):
        builder = newton.ModelBuilder(gravity=0.0)
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        model = builder.finalize(device="cpu")

        coupled = SolverCoupledProxy(
            model=model,
            entries=[
                SolverCoupled.Entry(name="src", solver=_AffineBodyForceSourceSolver, bodies=[body]),
                SolverCoupled.Entry(name="dst", solver=_AffineProxyBodyFeedbackSolver),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="src",
                        destination="dst",
                        bodies=[body],
                        proxy_relaxation_mode="aitken",
                        proxy_relaxation=1.0,
                        proxy_relaxation_min=0.1,
                        proxy_relaxation_max=1.0,
                    )
                ],
                iterations=3,
            ),
        )

        coupled.step(model.state(), model.state(), control=None, contacts=None, dt=1.0)

        mapping = coupled._proxy_mappings[0]
        np.testing.assert_allclose(mapping.coupling_forces.numpy()[body, 0], 1.0 / 3.0, atol=1.0e-6)
        np.testing.assert_allclose(mapping.aitken_relaxation.numpy()[0], 1.0 / 3.0, atol=1.0e-6)

    def test_duplicate_body_proxy_mapping_ids_are_rejected(self):
        builder = newton.ModelBuilder(gravity=0.0)
        for _ in range(3):
            builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        model = builder.finalize(device="cpu")

        with self.assertRaisesRegex(ValueError, "Duplicate source body"):
            SolverCoupledProxy(
                model=model,
                entries=[
                    SolverCoupled.Entry(name="src", solver=_StepCountingCopySolver, bodies=[0, 1]),
                    SolverCoupled.Entry(name="dst", solver=_StepCountingCopySolver),
                ],
                coupling=SolverCoupledProxy.Config(
                    proxies=[
                        SolverCoupledProxy.Proxy(
                            source="src",
                            destination="dst",
                            bodies=[0, 0],
                            proxy_bodies=[1, 2],
                        ),
                    ],
                ),
            )

        with self.assertRaisesRegex(ValueError, "Duplicate proxy body"):
            SolverCoupledProxy(
                model=model,
                entries=[
                    SolverCoupled.Entry(name="src", solver=_StepCountingCopySolver, bodies=[0, 1]),
                    SolverCoupled.Entry(name="dst", solver=_StepCountingCopySolver),
                ],
                coupling=SolverCoupledProxy.Config(
                    proxies=[
                        SolverCoupledProxy.Proxy(
                            source="src",
                            destination="dst",
                            bodies=[0, 1],
                            proxy_bodies=[2, 2],
                        ),
                    ],
                ),
            )

    def test_cross_world_body_proxy_mapping_is_rejected(self):
        builder = newton.ModelBuilder(gravity=0.0)
        builder.begin_world()
        source_body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.end_world()
        builder.begin_world()
        proxy_body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.end_world()
        model = builder.finalize(device="cpu")

        with self.assertRaisesRegex(ValueError, "same world"):
            SolverCoupledProxy(
                model=model,
                entries=[
                    SolverCoupled.Entry(name="src", solver=_StepCountingCopySolver, bodies=[source_body]),
                    SolverCoupled.Entry(name="dst", solver=_StepCountingCopySolver, bodies=[proxy_body]),
                ],
                coupling=SolverCoupledProxy.Config(
                    proxies=[
                        SolverCoupledProxy.Proxy(
                            source="src",
                            destination="dst",
                            bodies=[source_body],
                            proxy_bodies=[proxy_body],
                        ),
                    ],
                ),
            )

    def test_body_proxy_maps_proxy_indexed_feedback_to_source(self):
        _BodyForceRecordingSolver.instances.clear()
        _ProxyBodyHookSolver.instances.clear()

        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        model = builder.finalize(device="cpu")

        coupled = SolverCoupledProxy(
            model=model,
            entries=[
                SolverCoupled.Entry(name="src", solver=_BodyForceRecordingSolver, bodies=[0]),
                SolverCoupled.Entry(name="dst", solver=_ProxyBodyHookSolver, bodies=[1]),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="src",
                        destination="dst",
                        bodies=[0],
                        proxy_bodies=[2],
                    ),
                ],
            ),
        )

        state_0 = model.state()
        state_1 = model.state()
        dt = 0.5

        coupled.step(state_0, state_1, control=None, contacts=None, dt=dt)
        coupled.step(state_1, state_0, control=None, contacts=None, dt=dt)

        src_solver = _BodyForceRecordingSolver.instances[-1]
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertEqual(src_solver.input_body_f[1].shape[0], 1)
        np.testing.assert_allclose(src_solver.input_body_f[1][0], expected, atol=1.0e-6)

    def test_body_proxy_feedback_relaxation_blends_next_step_force_input(self):
        _BodyForceRecordingSolver.instances.clear()
        _ProxyBodyHookSolver.instances.clear()

        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        model = builder.finalize(device="cpu")

        coupled = SolverCoupledProxy(
            model=model,
            entries=[
                SolverCoupled.Entry(name="src", solver=_BodyForceRecordingSolver, bodies=[0]),
                SolverCoupled.Entry(name="dst", solver=_ProxyBodyHookSolver, bodies=[1]),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="src",
                        destination="dst",
                        bodies=[0],
                        proxy_bodies=[2],
                        proxy_relaxation=0.25,
                    ),
                ],
            ),
        )

        state_0 = model.state()
        state_1 = model.state()
        dt = 0.5

        coupled.step(state_0, state_1, control=None, contacts=None, dt=dt)
        coupled.step(state_1, state_0, control=None, contacts=None, dt=dt)

        src_solver = _BodyForceRecordingSolver.instances[-1]
        expected = 0.25 * np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        np.testing.assert_allclose(src_solver.input_body_f[1][0], expected, atol=1.0e-6)


class TestSolverCoupledParticleProxy(unittest.TestCase):
    """Particle proxy mappings keep proxy particles dynamic in the destination view."""

    def setUp(self):
        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=2.0, radius=0.0)
        builder.add_particle(pos=(1.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=2.0, radius=0.0)
        self.model = builder.finalize(device="cpu")

    def _make_coupled(self, dst_solver=_ProxyParticleKickSolver):
        return SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupled.Entry(name="src", solver=_ParticleForceRecordingSolver, particles=[0]),
                SolverCoupled.Entry(name="dst", solver=dst_solver, particles=[1]),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="src",
                        destination="dst",
                        particles=[0],
                        mass_scale=0.5,
                    ),
                ],
            ),
        )

    def test_duplicate_particle_proxy_mapping_ids_are_rejected(self):
        builder = newton.ModelBuilder(gravity=0.0)
        for i in range(3):
            builder.add_particle(pos=(float(i), 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.0)
        model = builder.finalize(device="cpu")

        with self.assertRaisesRegex(ValueError, "Duplicate source particle"):
            SolverCoupledProxy(
                model=model,
                entries=[
                    SolverCoupled.Entry(name="src", solver=_ParticleForceRecordingSolver, particles=[0, 1]),
                    SolverCoupled.Entry(name="dst", solver=_ProxyParticleKickSolver, particles=[2]),
                ],
                coupling=SolverCoupledProxy.Config(
                    proxies=[
                        SolverCoupledProxy.Proxy(
                            source="src",
                            destination="dst",
                            particles=[0, 0],
                            proxy_particles=[1, 2],
                        ),
                    ],
                ),
            )

        with self.assertRaisesRegex(ValueError, "Duplicate proxy particle"):
            SolverCoupledProxy(
                model=model,
                entries=[
                    SolverCoupled.Entry(name="src", solver=_ParticleForceRecordingSolver, particles=[0, 1]),
                    SolverCoupled.Entry(name="dst", solver=_ProxyParticleKickSolver, particles=[2]),
                ],
                coupling=SolverCoupledProxy.Config(
                    proxies=[
                        SolverCoupledProxy.Proxy(
                            source="src",
                            destination="dst",
                            particles=[0, 1],
                            proxy_particles=[2, 2],
                        ),
                    ],
                ),
            )

    def test_cross_world_particle_proxy_mapping_is_rejected(self):
        builder = newton.ModelBuilder(gravity=0.0)
        builder.begin_world()
        source_particle = builder.add_particle(
            pos=(0.0, 0.0, 0.0),
            vel=(0.0, 0.0, 0.0),
            mass=1.0,
        )
        builder.end_world()
        builder.begin_world()
        proxy_particle = builder.add_particle(
            pos=(1.0, 0.0, 0.0),
            vel=(0.0, 0.0, 0.0),
            mass=1.0,
        )
        builder.end_world()
        model = builder.finalize(device="cpu")

        with self.assertRaisesRegex(ValueError, "source particle.*same world"):
            SolverCoupledProxy(
                model=model,
                entries=[
                    SolverCoupled.Entry(name="src", solver=_ParticleForceRecordingSolver, particles=[source_particle]),
                    SolverCoupled.Entry(name="dst", solver=_ProxyParticleKickSolver),
                ],
                coupling=SolverCoupledProxy.Config(
                    proxies=[
                        SolverCoupledProxy.Proxy(
                            source="src",
                            destination="dst",
                            particles=[source_particle],
                            proxy_particles=[proxy_particle],
                        )
                    ]
                ),
            )

    def test_proxy_destination_view_keeps_and_scales_particle_mass(self):
        _ParticleForceRecordingSolver.instances.clear()
        coupled = self._make_coupled()

        src_view = coupled.view("src")
        dst_view = coupled.view("dst")

        self.assertEqual(src_view.particle_inv_mass.shape[0], 2)
        self.assertEqual(src_view.particle_inv_mass.numpy()[1], 0.0)
        np.testing.assert_allclose(dst_view.particle_mass.numpy(), [1.0, 2.0])
        np.testing.assert_allclose(dst_view.particle_inv_mass.numpy(), [1.0, 0.5])
        np.testing.assert_allclose(self.model.particle_mass.numpy(), [2.0, 2.0])

    def test_particle_proxy_feedback_is_applied_on_next_step(self):
        _ParticleForceRecordingSolver.instances.clear()
        coupled = self._make_coupled()

        state_0 = self.model.state()
        state_1 = self.model.state()
        control = self.model.control()
        dt = 0.5

        coupled.step(state_0, state_1, control=None, contacts=None, dt=dt)
        coupled.step(state_1, state_0, control=control, contacts=None, dt=dt)

        solver = _ParticleForceRecordingSolver.instances[-1]
        self.assertEqual(len(solver.input_particle_f), 2)
        np.testing.assert_allclose(solver.input_particle_f[0][0], np.zeros(3), atol=1.0e-6)
        np.testing.assert_allclose(solver.input_particle_f[1][0], np.array([0.0, 4.0, 0.0]), atol=1.0e-6)

    def test_particle_proxy_feedback_relaxation_handles_zeroing_custom_harvest(self):
        _ParticleForceRecordingSolver.instances.clear()
        _ProxyParticleHookSolver.instances.clear()

        coupled = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupled.Entry(name="src", solver=_ParticleForceRecordingSolver, particles=[0]),
                SolverCoupled.Entry(name="dst", solver=_ZeroingProxyParticleHookSolver, particles=[1]),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="src",
                        destination="dst",
                        particles=[0],
                        mass_scale=0.5,
                        proxy_relaxation=0.25,
                    ),
                ],
            ),
        )

        state_0 = self.model.state()
        state_1 = self.model.state()
        dt = 0.5

        coupled.step(state_0, state_1, control=None, contacts=None, dt=dt)
        coupled.step(state_1, state_0, control=None, contacts=None, dt=dt)

        solver = _ParticleForceRecordingSolver.instances[-1]
        self.assertEqual(len(solver.input_particle_f), 2)
        np.testing.assert_allclose(solver.input_particle_f[0][0], np.zeros(3), atol=1.0e-6)
        np.testing.assert_allclose(solver.input_particle_f[1][0], np.array([0.0, 1.75, 0.0]), atol=1.0e-6)

    def test_particle_proxy_feedback_overrelaxation_is_applied_on_next_step(self):
        _ParticleForceRecordingSolver.instances.clear()
        _ProxyParticleHookSolver.instances.clear()

        coupled = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupled.Entry(name="src", solver=_ParticleForceRecordingSolver, particles=[0]),
                SolverCoupled.Entry(name="dst", solver=_ZeroingProxyParticleHookSolver, particles=[1]),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="src",
                        destination="dst",
                        particles=[0],
                        mass_scale=0.5,
                        proxy_relaxation=1.5,
                    ),
                ],
            ),
        )

        state_0 = self.model.state()
        state_1 = self.model.state()
        dt = 0.5

        coupled.step(state_0, state_1, control=None, contacts=None, dt=dt)
        coupled.step(state_1, state_0, control=None, contacts=None, dt=dt)

        solver = _ParticleForceRecordingSolver.instances[-1]
        self.assertEqual(len(solver.input_particle_f), 2)
        np.testing.assert_allclose(solver.input_particle_f[0][0], np.zeros(3), atol=1.0e-6)
        np.testing.assert_allclose(solver.input_particle_f[1][0], np.array([0.0, 10.5, 0.0]), atol=1.0e-6)

    def test_particle_proxy_aitken_relaxation_kernels(self):
        coupled = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupled.Entry(name="src", solver=_ParticleForceRecordingSolver, particles=[0]),
                SolverCoupled.Entry(name="dst", solver=_ProxyParticleHookSolver, particles=[1]),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="src",
                        destination="dst",
                        particles=[0],
                        proxy_relaxation_mode="aitken",
                    )
                ],
                iterations=2,
            ),
        )

        coupled.step(self.model.state(), self.model.state(), control=None, contacts=None, dt=0.5)

        mapping = coupled._proxy_particle_mappings[0]
        np.testing.assert_allclose(mapping.coupling_forces.numpy()[0], np.array([0.0, 7.0, 0.0]), atol=1.0e-6)
        self.assertTrue(np.isfinite(mapping.aitken_relaxation.numpy()[0]))

    def test_particle_proxy_maps_proxy_indexed_feedback_to_source(self):
        _ParticleForceRecordingSolver.instances.clear()
        _ProxyParticleHookSolver.instances.clear()

        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=2.0, radius=0.0)
        builder.add_particle(pos=(1.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=2.0, radius=0.0)
        builder.add_particle(pos=(2.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=2.0, radius=0.0)
        model = builder.finalize(device="cpu")

        coupled = SolverCoupledProxy(
            model=model,
            entries=[
                SolverCoupled.Entry(name="src", solver=_ParticleForceRecordingSolver, particles=[0]),
                SolverCoupled.Entry(name="dst", solver=_ProxyParticleHookSolver, particles=[1]),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="src",
                        destination="dst",
                        particles=[0],
                        proxy_particles=[2],
                    ),
                ],
            ),
        )

        state_0 = model.state()
        state_1 = model.state()
        dt = 0.5

        coupled.step(state_0, state_1, control=None, contacts=None, dt=dt)
        coupled.step(state_1, state_0, control=None, contacts=None, dt=dt)

        src_solver = _ParticleForceRecordingSolver.instances[-1]
        self.assertEqual(src_solver.input_particle_f[1].shape[0], 3)
        np.testing.assert_allclose(src_solver.input_particle_f[1][0], np.array([0.0, 7.0, 0.0]), atol=1.0e-6)
        np.testing.assert_allclose(src_solver.input_particle_f[1][2], np.zeros(3), atol=1.0e-6)

    def test_proxy_destination_view_marks_proxy_particle_flags(self):
        coupled = self._make_coupled()

        src_view = coupled.view("src")
        dst_view = coupled.view("dst")
        proxy_flag = int(newton.ParticleFlags.PROXY)

        self.assertEqual(src_view.particle_flags.numpy()[0] & proxy_flag, 0)
        self.assertNotEqual(dst_view.particle_flags.numpy()[0] & proxy_flag, 0)
        self.assertEqual(self.model.particle_flags.numpy()[0] & proxy_flag, 0)

    def test_xpbd_ignores_proxy_proxy_particle_contacts(self):
        flags = int(newton.ParticleFlags.ACTIVE) | int(newton.ParticleFlags.PROXY)
        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_particle(pos=(-0.02, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.05, flags=flags)
        builder.add_particle(pos=(0.02, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.05, flags=flags)
        model = builder.finalize(device="cpu")
        solver = SolverXPBD(model=model, iterations=4, soft_contact_relaxation=1.0)

        state_0 = model.state()
        state_1 = model.state()
        contacts = model.contacts()
        q_before = state_0.particle_q.numpy().copy()

        solver.step(state_0, state_1, control=None, contacts=contacts, dt=1.0 / 60.0)

        np.testing.assert_allclose(state_1.particle_q.numpy(), q_before, atol=1.0e-6)

    def test_xpbd_ignores_proxy_static_particle_contacts(self):
        proxy_flags = int(newton.ParticleFlags.ACTIVE) | int(newton.ParticleFlags.PROXY)
        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_particle(pos=(-0.02, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.05, flags=proxy_flags)
        builder.add_particle(
            pos=(0.02, 0.0, 0.0),
            vel=(0.0, 0.0, 0.0),
            mass=0.0,
            radius=0.05,
            flags=int(newton.ParticleFlags.ACTIVE),
        )
        model = builder.finalize(device="cpu")
        solver = SolverXPBD(model=model, iterations=4, soft_contact_relaxation=1.0)

        state_0 = model.state()
        state_1 = model.state()
        contacts = model.contacts()
        q_before = state_0.particle_q.numpy().copy()

        solver.step(state_0, state_1, control=None, contacts=contacts, dt=1.0 / 60.0)

        np.testing.assert_allclose(state_1.particle_q.numpy(), q_before, atol=1.0e-6)

    def test_xpbd_ignores_proxy_particle_proxy_body_contacts(self):
        proxy_particle_flags = int(newton.ParticleFlags.ACTIVE) | int(newton.ParticleFlags.PROXY)
        builder = newton.ModelBuilder(gravity=0.0)
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_shape_sphere(body=body, radius=0.05)
        builder.add_particle(
            pos=(0.08, 0.0, 0.0),
            vel=(0.0, 0.0, 0.0),
            mass=1.0,
            radius=0.05,
            flags=proxy_particle_flags,
        )
        model = builder.finalize(device="cpu")
        view = ModelView(model, "xpbd")
        view.mark_proxy_bodies(wp.array([body], dtype=int, device=model.device))
        solver = SolverXPBD(model=view, iterations=4, soft_contact_relaxation=1.0)

        state_0 = model.state()
        state_1 = model.state()
        contacts = model.collide(state_0)
        self.assertGreater(int(contacts.soft_contact_count.numpy()[0]), 0)
        q_before = state_0.particle_q.numpy().copy()

        solver.step(state_0, state_1, control=None, contacts=contacts, dt=1.0 / 60.0)

        np.testing.assert_allclose(state_1.particle_q.numpy(), q_before, atol=1.0e-6)


class TestSolverCoupledProxyDiagnostics(unittest.TestCase):
    """Proxy feedback diagnostics expose live buffers without copying them."""

    @staticmethod
    def _build_coupled(*, include_bodies=True, include_particles=True):
        builder = newton.ModelBuilder(gravity=0.0)
        for _ in range(4):
            builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        for i in range(4):
            builder.add_particle(
                pos=(float(i), 0.0, 0.0),
                vel=(0.0, 0.0, 0.0),
                mass=1.0,
                radius=0.0,
            )
        model = builder.finalize(device="cpu")

        proxies = []
        for source_id, proxy_id, relaxation, relaxation_mode, relaxation_max in (
            (0, 1, 0.25, "fixed", 1.0),
            (2, 3, 0.75, "aitken", 0.9),
        ):
            proxies.append(
                SolverCoupledProxy.Proxy(
                    source="src",
                    destination="dst",
                    bodies=[source_id] if include_bodies else (),
                    proxy_bodies=[proxy_id] if include_bodies else None,
                    particles=[source_id] if include_particles else (),
                    proxy_particles=[proxy_id] if include_particles else None,
                    proxy_relaxation=relaxation,
                    proxy_relaxation_mode=relaxation_mode,
                    proxy_relaxation_min=0.1,
                    proxy_relaxation_max=relaxation_max,
                )
            )

        coupled = SolverCoupledProxy(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="src",
                    solver=_StepCountingCopySolver,
                    bodies=[0, 2] if include_bodies else (),
                    particles=[0, 2] if include_particles else (),
                ),
                SolverCoupled.Entry(name="dst", solver=_StepCountingCopySolver),
            ],
            coupling=SolverCoupledProxy.Config(proxies=proxies, iterations=2),
        )
        return coupled

    def test_proxy_diagnostics_exports_frozen_borrowed_feedback_records(self):
        for name in ("ProxyBodyFeedback", "ProxyParticleFeedback", "ProxyRelaxationDiagnostics"):
            self.assertTrue(hasattr(coupled_api, name))
            self.assertIn(name, coupled_api.__all__)

        coupled = self._build_coupled()
        body_mapping_ids = tuple(id(mapping) for mapping in coupled._proxy_mappings)
        particle_mapping_ids = tuple(id(mapping) for mapping in coupled._proxy_particle_mappings)
        array_type = type(coupled._proxy_mappings[0].src_body_ids)

        with mock.patch.object(array_type, "numpy", side_effect=AssertionError("accessor synchronized to host")):
            body = coupled.proxy_body_feedback("src", "dst")
            particle = coupled.proxy_particle_feedback("src", "dst")

        self.assertIsInstance(body, coupled_api.ProxyBodyFeedback)
        self.assertEqual((body.source, body.destination), ("src", "dst"))
        self.assertEqual(len(body.forces), 2)
        for index, mapping in enumerate(coupled._proxy_mappings):
            self.assertIs(body.source_body_ids[index], mapping.src_body_ids)
            self.assertIs(body.proxy_body_ids[index], mapping.proxy_body_ids_global)
            self.assertIs(body.forces[index], mapping.coupling_forces)
        np.testing.assert_array_equal(body.source_body_ids[0].numpy(), [0])
        np.testing.assert_array_equal(body.source_body_ids[1].numpy(), [2])
        np.testing.assert_array_equal(body.proxy_body_ids[0].numpy(), [1])
        np.testing.assert_array_equal(body.proxy_body_ids[1].numpy(), [3])
        self.assertTrue(all(forces.shape == (coupled.model.body_count,) for forces in body.forces))

        self.assertIsInstance(particle, coupled_api.ProxyParticleFeedback)
        self.assertEqual((particle.source, particle.destination), ("src", "dst"))
        self.assertEqual(len(particle.forces), 2)
        for index, mapping in enumerate(coupled._proxy_particle_mappings):
            self.assertIs(particle.source_particle_ids[index], mapping.src_particle_ids)
            self.assertIs(particle.proxy_particle_ids[index], mapping.proxy_particle_ids_global)
            self.assertIs(particle.forces[index], mapping.coupling_forces)
        np.testing.assert_array_equal(particle.source_particle_ids[0].numpy(), [0])
        np.testing.assert_array_equal(particle.source_particle_ids[1].numpy(), [2])
        np.testing.assert_array_equal(particle.proxy_particle_ids[0].numpy(), [1])
        np.testing.assert_array_equal(particle.proxy_particle_ids[1].numpy(), [3])
        self.assertTrue(all(forces.shape == (coupled.model.particle_count,) for forces in particle.forces))

        coupled._proxy_mappings[0].coupling_forces.fill_(2.0)
        coupled._proxy_particle_mappings[0].coupling_forces.fill_(3.0)
        coupled.reset(coupled.model.state())
        np.testing.assert_array_equal(body.forces[0].numpy(), 0.0)
        np.testing.assert_array_equal(particle.forces[0].numpy(), 0.0)
        self.assertIs(coupled.proxy_body_feedback("src", "dst").forces[0], body.forces[0])
        self.assertIs(coupled.proxy_particle_feedback("src", "dst").forces[0], particle.forces[0])

        self.assertEqual(tuple(id(mapping) for mapping in coupled._proxy_mappings), body_mapping_ids)
        self.assertEqual(tuple(id(mapping) for mapping in coupled._proxy_particle_mappings), particle_mapping_ids)
        with self.assertRaises(FrozenInstanceError):
            body.source = "replacement"
        with self.assertRaises(FrozenInstanceError):
            particle.destination = "replacement"

    def test_proxy_diagnostics_report_fixed_and_aitken_state_without_ambiguity(self):
        coupled = self._build_coupled()
        body_mappings = coupled._proxy_mappings
        particle_mappings = coupled._proxy_particle_mappings
        array_type = type(body_mappings[0].world_ids)

        with mock.patch.object(array_type, "numpy", side_effect=AssertionError("accessor synchronized to host")):
            diagnostics = coupled.proxy_relaxation_diagnostics("src", "dst")

        self.assertIsInstance(diagnostics, coupled_api.ProxyRelaxationDiagnostics)
        self.assertEqual((diagnostics.source, diagnostics.destination), ("src", "dst"))
        self.assertEqual(diagnostics.global_slot, 0)
        self.assertEqual(diagnostics.world_slot_offset, 1)
        self.assertEqual(diagnostics.body_mode, ("fixed", "aitken"))
        self.assertEqual(diagnostics.body_configured, (0.25, 0.75))
        self.assertEqual(diagnostics.body_min, (0.1, 0.1))
        self.assertEqual(diagnostics.body_max, (1.0, 0.9))
        self.assertIs(diagnostics.body_world_ids[0], body_mappings[0].world_ids)
        self.assertIs(diagnostics.body_world_ids[1], body_mappings[1].world_ids)
        self.assertIsNone(diagnostics.body_current[0])
        self.assertIsNone(diagnostics.body_has_previous[0])
        self.assertIsNone(diagnostics.body_stats[0])
        self.assertIs(diagnostics.body_current[1], body_mappings[1].aitken_relaxation)
        self.assertIs(diagnostics.body_has_previous[1], body_mappings[1].aitken_has_previous)
        self.assertIs(diagnostics.body_stats[1], body_mappings[1].aitken_stats)

        self.assertEqual(diagnostics.particle_mode, ("fixed", "aitken"))
        self.assertEqual(diagnostics.particle_configured, (0.25, 0.75))
        self.assertEqual(diagnostics.particle_min, (0.1, 0.1))
        self.assertEqual(diagnostics.particle_max, (1.0, 0.9))
        self.assertIs(diagnostics.particle_world_ids[0], particle_mappings[0].world_ids)
        self.assertIs(diagnostics.particle_world_ids[1], particle_mappings[1].world_ids)
        self.assertIsNone(diagnostics.particle_current[0])
        self.assertIsNone(diagnostics.particle_has_previous[0])
        self.assertIsNone(diagnostics.particle_stats[0])
        self.assertIs(diagnostics.particle_current[1], particle_mappings[1].aitken_relaxation)
        self.assertIs(diagnostics.particle_has_previous[1], particle_mappings[1].aitken_has_previous)
        self.assertIs(diagnostics.particle_stats[1], particle_mappings[1].aitken_stats)
        with self.assertRaises(FrozenInstanceError):
            diagnostics.global_slot = 1

    def test_proxy_diagnostics_raise_key_error_for_unknown_direction_or_missing_kind(self):
        body_only = self._build_coupled(include_particles=False)
        particle_only = self._build_coupled(include_bodies=False)

        body_diagnostics = body_only.proxy_relaxation_diagnostics("src", "dst")
        self.assertEqual(body_diagnostics.body_mode, ("fixed", "aitken"))
        for field in (
            "particle_mode",
            "particle_configured",
            "particle_min",
            "particle_max",
            "particle_world_ids",
            "particle_current",
            "particle_has_previous",
            "particle_stats",
        ):
            self.assertIsNone(getattr(body_diagnostics, field))

        particle_diagnostics = particle_only.proxy_relaxation_diagnostics("src", "dst")
        self.assertEqual(particle_diagnostics.particle_mode, ("fixed", "aitken"))
        for field in (
            "body_mode",
            "body_configured",
            "body_min",
            "body_max",
            "body_world_ids",
            "body_current",
            "body_has_previous",
            "body_stats",
        ):
            self.assertIsNone(getattr(particle_diagnostics, field))

        with self.assertRaises(KeyError):
            body_only.proxy_particle_feedback("src", "dst")
        with self.assertRaises(KeyError):
            particle_only.proxy_body_feedback("src", "dst")
        for coupled in (body_only, particle_only):
            with self.assertRaises(KeyError):
                coupled.proxy_body_feedback("unknown", "dst")
            with self.assertRaises(KeyError):
                coupled.proxy_particle_feedback("src", "unknown")
            with self.assertRaises(KeyError):
                coupled.proxy_relaxation_diagnostics("unknown", "dst")


class TestSolverCoupledProxyWorldState(unittest.TestCase):
    """Proxy feedback and Aitken state are isolated by model world."""

    @staticmethod
    def _devices():
        devices = ["cpu"]
        if wp.is_cuda_available():
            devices.append("cuda:0")
        return devices

    @staticmethod
    def _build_coupled(device):
        builder = newton.ModelBuilder(gravity=0.0)
        global_source_body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        global_proxy_body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        global_source_particle = builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.0)
        global_proxy_particle = builder.add_particle(pos=(1.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.0)

        world = newton.ModelBuilder(gravity=0.0)
        world.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        world.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        world.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.0)
        world.add_particle(pos=(1.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.0)
        builder.add_world(world)
        builder.add_world(world)
        model = builder.finalize(device=device)

        source_bodies = [global_source_body, 2, 4]
        proxy_bodies = [global_proxy_body, 3, 5]
        source_particles = [global_source_particle, 2, 4]
        proxy_particles = [global_proxy_particle, 3, 5]
        coupled = SolverCoupledProxy(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="src",
                    solver=_StepCountingCopySolver,
                    bodies=source_bodies,
                    particles=source_particles,
                ),
                SolverCoupled.Entry(name="dst", solver=_StepCountingCopySolver),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="src",
                        destination="dst",
                        bodies=source_bodies,
                        proxy_bodies=proxy_bodies,
                        particles=source_particles,
                        proxy_particles=proxy_particles,
                        proxy_relaxation_mode="aitken",
                        proxy_relaxation=1.0,
                        proxy_relaxation_min=0.1,
                        proxy_relaxation_max=1.0,
                    )
                ],
                iterations=2,
            ),
        )
        return model, coupled

    @staticmethod
    def _rows(count, width, offset):
        return np.arange(count * width, dtype=np.float32).reshape(count, width) + offset

    def test_proxy_diagnostics_expose_global_and_regular_world_slots(self):
        model, coupled = self._build_coupled("cpu")
        body = coupled._proxy_mappings[0]
        particle = coupled._proxy_particle_mappings[0]
        diagnostics = coupled.proxy_relaxation_diagnostics("src", "dst")

        self.assertEqual(diagnostics.global_slot, 0)
        self.assertEqual(diagnostics.world_slot_offset, 1)
        self.assertIs(diagnostics.body_current[0], body.aitken_relaxation)
        self.assertIs(diagnostics.particle_current[0], particle.aitken_relaxation)
        self.assertEqual(diagnostics.body_current[0].shape, (model.world_count + 1,))
        self.assertEqual(diagnostics.particle_current[0].shape, (model.world_count + 1,))
        np.testing.assert_array_equal(diagnostics.body_world_ids[0].numpy(), [-1, 0, 1])
        np.testing.assert_array_equal(diagnostics.particle_world_ids[0].numpy(), [-1, 0, 1])
        for world_id, expected_slot in ((-1, 0), (0, 1), (1, 2)):
            self.assertEqual(world_id + diagnostics.world_slot_offset, expected_slot)

    def test_partial_and_full_reset_partition_body_and_particle_proxy_state(self):
        for device in self._devices():
            with self.subTest(device=device):
                model, coupled = self._build_coupled(device)
                body = coupled._proxy_mappings[0]
                particle = coupled._proxy_particle_mappings[0]
                np.testing.assert_array_equal(body.world_ids.numpy(), [-1, 0, 1])
                np.testing.assert_array_equal(particle.world_ids.numpy(), [-1, 0, 1])
                self.assertEqual(body.aitken_stats.shape, (model.world_count + 1, 2))
                self.assertEqual(particle.aitken_stats.shape, (model.world_count + 1, 2))
                self.assertEqual(body.aitken_relaxation.shape, (model.world_count + 1,))
                self.assertEqual(particle.aitken_has_previous.shape, (model.world_count + 1,))

                body_full = self._rows(model.body_count, 6, 10.0)
                body_qd = self._rows(model.body_count, 6, 100.0)
                body_previous = self._rows(3, 6, 200.0)
                body_residual = self._rows(3, 6, 300.0)
                particle_full = self._rows(model.particle_count, 3, 20.0)
                particle_qd = self._rows(model.particle_count, 3, 120.0)
                particle_previous = self._rows(3, 3, 220.0)
                particle_residual = self._rows(3, 3, 320.0)
                body.coupling_forces.assign(body_full)
                body.proxy_qd_before.assign(body_qd)
                body.coupling_forces_previous.assign(body_previous)
                body.aitken_residual_previous.assign(body_residual)
                particle.coupling_forces.assign(particle_full)
                particle.proxy_qd_before.assign(particle_qd)
                particle.coupling_forces_previous.assign(particle_previous)
                particle.aitken_residual_previous.assign(particle_residual)
                for mapping, base in ((body, 1.0), (particle, 11.0)):
                    mapping.aitken_stats.assign(
                        np.array([[base, base + 1.0], [base + 2.0, base + 3.0], [base + 4.0, base + 5.0]])
                    )
                    mapping.aitken_relaxation.assign(np.array([0.2, 0.3, 0.4], dtype=np.float32))
                    mapping.aitken_has_previous.assign(np.array([1, 1, 1], dtype=np.int32))

                coupled.reset(
                    model.state(),
                    world_mask=wp.array([True, False], dtype=wp.bool, device=device),
                )

                for mapping, full_before, qd_before, previous_before, residual_before, entity_world in (
                    (body, body_full, body_qd, body_previous, body_residual, model.body_world.numpy()),
                    (
                        particle,
                        particle_full,
                        particle_qd,
                        particle_previous,
                        particle_residual,
                        model.particle_world.numpy(),
                    ),
                ):
                    expected_full = full_before.copy()
                    expected_full[entity_world == 0] = 0.0
                    expected_qd = qd_before.copy()
                    expected_qd[entity_world == 0] = 0.0
                    expected_previous = previous_before.copy()
                    expected_previous[1] = 0.0
                    expected_residual = residual_before.copy()
                    expected_residual[1] = 0.0
                    np.testing.assert_array_equal(mapping.coupling_forces.numpy(), expected_full)
                    np.testing.assert_array_equal(mapping.proxy_qd_before.numpy(), expected_qd)
                    np.testing.assert_array_equal(mapping.coupling_forces_previous.numpy(), expected_previous)
                    np.testing.assert_array_equal(mapping.aitken_residual_previous.numpy(), expected_residual)
                    np.testing.assert_array_equal(mapping.aitken_stats.numpy()[1], 0.0)
                    np.testing.assert_array_equal(
                        mapping.aitken_relaxation.numpy(),
                        np.array([0.2, 1.0, 0.4], dtype=np.float32),
                    )
                    np.testing.assert_array_equal(mapping.aitken_has_previous.numpy(), [1, 0, 1])

                for mapping in (body, particle):
                    mapping.coupling_forces.fill_(2.0)
                    mapping.proxy_qd_before.fill_(3.0)
                    mapping.coupling_forces_previous.fill_(4.0)
                    mapping.aitken_residual_previous.fill_(5.0)
                    mapping.aitken_stats.fill_(6.0)
                    mapping.aitken_relaxation.fill_(0.5)
                    mapping.aitken_has_previous.fill_(1)

                coupled.reset(model.state())

                for mapping in (body, particle):
                    np.testing.assert_array_equal(mapping.coupling_forces.numpy(), 0.0)
                    np.testing.assert_array_equal(mapping.proxy_qd_before.numpy(), 0.0)
                    np.testing.assert_array_equal(mapping.coupling_forces_previous.numpy(), 0.0)
                    np.testing.assert_array_equal(mapping.aitken_residual_previous.numpy(), 0.0)
                    np.testing.assert_array_equal(mapping.aitken_stats.numpy(), 0.0)
                    np.testing.assert_array_equal(mapping.aitken_relaxation.numpy(), 1.0)
                    np.testing.assert_array_equal(mapping.aitken_has_previous.numpy(), 0)

    def test_aitken_relaxation_updates_each_world_independently(self):
        for device in self._devices():
            with self.subTest(device=device):
                model, coupled = self._build_coupled(device)
                for mapping, width, blend in (
                    (coupled._proxy_mappings[0], 6, coupled._blend_proxy_body_feedback),
                    (coupled._proxy_particle_mappings[0], 3, coupled._blend_proxy_particle_feedback),
                ):
                    self.assertEqual(mapping.aitken_stats.shape, (model.world_count + 1, 2))
                    previous = np.zeros((3, width), dtype=np.float32)
                    residual_previous = np.zeros((3, width), dtype=np.float32)
                    residual_previous[1, 0] = 1.0
                    residual_previous[2, 0] = 2.0
                    raw = np.zeros_like(mapping.coupling_forces.numpy())
                    proxy_ids = mapping.proxy_body_ids_global if width == 6 else mapping.proxy_particle_ids_global
                    proxy_ids_np = proxy_ids.numpy()
                    raw[proxy_ids_np[1], 0] = 3.0
                    raw[proxy_ids_np[2], 0] = 1.0
                    mapping.coupling_forces_previous.assign(previous)
                    mapping.aitken_residual_previous.assign(residual_previous)
                    mapping.coupling_forces.assign(raw)
                    mapping.aitken_relaxation.assign(np.ones(3, dtype=np.float32))
                    mapping.aitken_has_previous.assign(np.array([0, 1, 1], dtype=np.int32))

                    blend(mapping)

                    np.testing.assert_allclose(mapping.aitken_stats.numpy()[1], [2.0, 4.0], atol=1.0e-6)
                    np.testing.assert_allclose(mapping.aitken_stats.numpy()[2], [-2.0, 1.0], atol=1.0e-6)
                    np.testing.assert_allclose(mapping.aitken_relaxation.numpy(), [1.0, 0.1, 1.0], atol=1.0e-6)
                    np.testing.assert_allclose(mapping.coupling_forces.numpy()[proxy_ids_np[1], 0], 0.3, atol=1.0e-6)
                    np.testing.assert_allclose(mapping.coupling_forces.numpy()[proxy_ids_np[2], 0], 1.0, atol=1.0e-6)


class TestSolverCoupledVBDColoring(unittest.TestCase):
    """Compaction must remap ``body_color_groups`` for VBD entries.

    A VBD entry whose global body ids are not a 0-prefix gets compacted to dense
    local indices; the color groups must be remapped global->local, or two bodies
    joined by a joint can share a color, race in VBD's parallel solve, and the
    constraint diverges.
    """

    def test_compacted_vbd_entry_color_groups_are_valid(self):
        builder = newton.ModelBuilder()
        for _ in range(5):
            builder.add_body(mass=1.0)  # each auto-adds a free joint + articulation
        fixed_joint = builder.add_joint_fixed(parent=3, child=4)
        builder.color()
        model = builder.finalize(device="cpu")

        # "dst" owns {2,3,4} (not a 0-prefix) -> compaction maps it to local 0,1,2.
        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="src",
                    solver=SolverSemiImplicit,
                    bodies=[0, 1],
                    joints=[0, 1],
                ),
                SolverCoupled.Entry(
                    name="dst",
                    solver=lambda view: SolverVBD(view, iterations=1),
                    bodies=[2, 3, 4],
                    joints=[2, 3, 4, fixed_joint],
                ),
            ],
        )

        view = coupled.view("dst")
        body_count = int(view.body_count)
        groups = [[int(x) for x in g.numpy()] for g in view.body_color_groups]
        parents = [int(x) for x in view.joint_parent.numpy()]
        children = [int(x) for x in view.joint_child.numpy()]

        # Color groups must partition the local body set.
        union = sorted(body for group in groups for body in group)
        self.assertEqual(union, list(range(body_count)), f"groups must partition local bodies; got {groups}")

        # No joint-connected pair may share a color.
        color_of = {body: color for color, group in enumerate(groups) for body in group}
        for parent, child in zip(parents, children, strict=True):
            if 0 <= parent < body_count and 0 <= child < body_count:
                self.assertNotEqual(
                    color_of.get(parent),
                    color_of.get(child),
                    f"joint-connected local bodies {parent},{child} share a color: {groups}",
                )

    def test_compacted_custom_namespace_does_not_mutate_parent(self):
        """Compacted entry namespaces must be view-local, not parent aliases."""
        builder = newton.ModelBuilder()
        SolverVBD.register_custom_attributes(builder, dahl_defaults_enabled=False)
        for _ in range(5):
            builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        soft_joint = builder.add_joint_fixed(parent=3, child=4, custom_attributes={"vbd:joint_is_hard": 0})
        builder.color()
        model = builder.finalize(device="cpu")
        model.vbd.namespace_marker = "parent metadata"

        parent_joint_is_hard = model.vbd.joint_is_hard.numpy().copy()
        vbd_joint_order = [2, 3, 4, soft_joint]

        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="src",
                    solver=SolverSemiImplicit,
                    bodies=[0, 1],
                    joints=[0, 1],
                ),
                SolverCoupled.Entry(
                    name="dst",
                    solver=lambda view: SolverVBD(view, iterations=1),
                    bodies=[2, 3, 4],
                    joints=vbd_joint_order,
                ),
            ],
        )

        np.testing.assert_array_equal(model.vbd.joint_is_hard.numpy(), parent_joint_is_hard)

        view = coupled.view("dst")
        self.assertIsNot(view.vbd, model.vbd)
        self.assertEqual(view.vbd.namespace_marker, model.vbd.namespace_marker)
        self.assertEqual(view.vbd.joint_is_hard.shape[0], view.joint_count)
        np.testing.assert_array_equal(view.vbd.joint_is_hard.numpy(), parent_joint_is_hard[vbd_joint_order])

    def test_compacted_custom_frequency_namespace_metadata_is_generic(self):
        builder = newton.ModelBuilder()
        for _ in range(4):
            builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        _add_equality_constraint(builder, constraint_type=newton.EqType.CONNECT, body1=0, body2=1)
        _add_equality_constraint(builder, constraint_type=newton.EqType.CONNECT, body1=2, body2=3)
        model = builder.finalize(device="cpu")

        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(name="src", solver=SolverSemiImplicit, bodies=[0, 1]),
                SolverCoupled.Entry(name="dst", solver=SolverSemiImplicit, bodies=[2, 3]),
            ],
        )

        view = coupled.view("dst")
        self.assertEqual(view.custom_frequency_counts["mujoco:equality_constraint"], 1)
        self.assertEqual(view.mujoco.equality_constraint_count, 1)
        self.assertEqual(view.mujoco.equality_constraint_type.shape[0], 1)
        np.testing.assert_array_equal(view.mujoco.equality_constraint_body1.numpy(), np.array([0], dtype=np.int32))
        np.testing.assert_array_equal(view.mujoco.equality_constraint_body2.numpy(), np.array([1], dtype=np.int32))
        self.assertEqual(int(view.mujoco.equality_constraint_world_start.numpy()[-1]), 1)
        self.assertNotIn("equality_constraint_count", view.overrides)
        self.assertNotIn("equality_constraint_body1", view.overrides)

    def test_metadata_projects_nonprefix_custom_references(self):
        builder = newton.ModelBuilder()
        for frequency in ("linkage", "entity", "link"):
            builder.add_custom_frequency(newton.ModelBuilder.CustomFrequency(name=frequency, namespace="test"))

        def add_attribute(name, frequency, dtype, *, references=None, assignment=None):
            builder.add_custom_attribute(
                newton.ModelBuilder.CustomAttribute(
                    name=name,
                    frequency=frequency,
                    dtype=dtype,
                    namespace="test",
                    references=references,
                    assignment=assignment or newton.Model.AttributeAssignment.MODEL,
                )
            )

        add_attribute("linkage_body0", "test:linkage", wp.int32, references="body")
        add_attribute("linkage_body1", "test:linkage", wp.int32, references="body")
        add_attribute("linkage_bodies", "test:linkage", wp.vec2i, references="body")
        add_attribute("linkage_weight", "test:linkage", wp.float32)
        add_attribute("entity_body", "test:entity", wp.int32, references="body")
        add_attribute("link_entity", "test:link", wp.int32, references="test:entity")
        add_attribute(
            "state_seed",
            newton.Model.AttributeFrequency.BODY,
            wp.float32,
            assignment=newton.Model.AttributeAssignment.STATE,
        )

        for body in range(4):
            builder.add_body(
                mass=1.0,
                inertia=wp.mat33(np.eye(3)),
                custom_attributes={"test:state_seed": float(10 + body)},
            )
            builder.add_custom_values(**{"test:entity_body": body})
            builder.add_custom_values(**{"test:link_entity": body})
        builder.add_custom_values(
            **{
                "test:linkage_body0": 0,
                "test:linkage_body1": 2,
                "test:linkage_bodies": wp.vec2i(0, 2),
                "test:linkage_weight": 2.0,
            }
        )
        builder.add_custom_values(
            **{
                "test:linkage_body0": 1,
                "test:linkage_body1": 3,
                "test:linkage_bodies": wp.vec2i(1, 3),
                "test:linkage_weight": 4.0,
            }
        )
        builder.add_custom_values(
            **{
                "test:linkage_body0": -1,
                "test:linkage_body1": 1,
                "test:linkage_bodies": wp.vec2i(-1, 1),
                "test:linkage_weight": 6.0,
            }
        )
        model = builder.finalize(device="cpu")
        model.test.namespace_marker = "parent"

        self.assertEqual(
            model._attribute_reference_frequency("test:linkage_body0"),
            newton.Model.AttributeFrequency.BODY,
        )
        self.assertEqual(model._attribute_reference_frequency("test:link_entity"), "test:entity")
        self.assertEqual(
            model.attribute_assignment.get("test:linkage_body0", newton.Model.AttributeAssignment.MODEL),
            newton.Model.AttributeAssignment.MODEL,
        )
        self.assertFalse(hasattr(model, "_attribute_descriptors"))
        for name, frequency in (
            ("body_label", newton.Model.AttributeFrequency.BODY),
            ("shape_color", newton.Model.AttributeFrequency.SHAPE),
            ("_shape_sdf_index", newton.Model.AttributeFrequency.SHAPE),
            ("tri_materials", newton.Model.AttributeFrequency.TRIANGLE),
        ):
            with self.subTest(inferred_attribute=name):
                self.assertEqual(model._resolve_attribute_frequency(name), frequency)
        self.assertEqual(
            model._resolve_attribute_frequency("joint_q"),
            newton.Model.AttributeFrequency.JOINT_COORD,
        )

        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(name="src", solver=SolverSemiImplicit, bodies=[0, 2]),
                SolverCoupled.Entry(name="dst", solver=SolverSemiImplicit, bodies=[1, 3]),
            ],
        )
        view = coupled.view("dst")

        self.assertEqual(view.test.linkage_count, 2)
        self.assertEqual(view.test.entity_count, 2)
        self.assertEqual(view.test.link_count, 2)
        np.testing.assert_array_equal(view.test.linkage_body0.numpy(), [0, -1])
        np.testing.assert_array_equal(view.test.linkage_body1.numpy(), [1, 0])
        np.testing.assert_array_equal(view.test.linkage_bodies.numpy(), [[0, 1], [-1, 0]])
        np.testing.assert_array_equal(view.test.entity_body.numpy(), [0, 1])
        np.testing.assert_array_equal(view.test.link_entity.numpy(), [0, 1])
        np.testing.assert_allclose(view.test.linkage_weight.numpy(), [4.0, 6.0])

        state = view.state()
        np.testing.assert_allclose(state.test.state_seed.numpy(), [11.0, 13.0])

        view.test.namespace_marker = "view"
        view.test.linkage_weight.fill_(7.0)
        self.assertEqual(model.test.namespace_marker, "parent")
        np.testing.assert_allclose(model.test.linkage_weight.numpy(), [2.0, 4.0, 6.0])

    def test_compaction_projects_late_registered_attribute(self):
        builder = newton.ModelBuilder()
        for _ in range(2):
            builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        model = builder.finalize(device="cpu")
        model.extra_values = wp.array([1.0, 2.0], dtype=wp.float32, device="cpu")
        model.attribute_frequency["extra_values"] = newton.Model.AttributeFrequency.BODY

        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(name="src", solver=SolverSemiImplicit, bodies=[0]),
                SolverCoupled.Entry(name="dst", solver=SolverSemiImplicit, bodies=[1]),
            ],
        )

        np.testing.assert_allclose(coupled.view("dst").extra_values.numpy(), [2.0])

    def test_compaction_rejects_misaligned_inferred_attribute(self):
        builder = newton.ModelBuilder()
        for _ in range(2):
            builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        model = builder.finalize(device="cpu")
        model.body_misaligned = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device="cpu")

        with self.assertRaisesRegex(ValueError, "body_misaligned.*expected 2 values"):
            SolverCoupled(
                model=model,
                entries=[
                    SolverCoupled.Entry(name="src", solver=SolverSemiImplicit, bodies=[0]),
                    SolverCoupled.Entry(name="dst", solver=SolverSemiImplicit, bodies=[1]),
                ],
            )

    def test_compaction_validates_custom_reference_storage(self):
        def build_model():
            builder = newton.ModelBuilder()
            builder.add_custom_frequency(newton.ModelBuilder.CustomFrequency(name="row", namespace="test"))
            builder.add_custom_attribute(
                newton.ModelBuilder.CustomAttribute(
                    name="row_body",
                    frequency="test:row",
                    dtype=wp.int32,
                    namespace="test",
                    references="body",
                )
            )
            for body in range(2):
                builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
                builder.add_custom_values(**{"test:row_body": body})
            return builder.finalize(device="cpu")

        for value, message in (
            (None, "registered value is missing"),
            (wp.array([0], dtype=wp.int32, device="cpu"), "expected 2 rows"),
        ):
            with self.subTest(message=message):
                model = build_model()
                model.test.row_body = value
                with self.assertRaisesRegex(ValueError, message):
                    SolverCoupled(
                        model=model,
                        entries=[
                            SolverCoupled.Entry(name="src", solver=SolverSemiImplicit, bodies=[0]),
                            SolverCoupled.Entry(name="dst", solver=SolverSemiImplicit, bodies=[1]),
                        ],
                    )

    def test_metadata_projects_custom_rows_across_worlds(self):
        sub_builder = newton.ModelBuilder()
        sub_builder.add_custom_frequency(newton.ModelBuilder.CustomFrequency(name="node", namespace="test"))
        for name, references in (("node_body", "body"), ("environment", "world")):
            sub_builder.add_custom_attribute(
                newton.ModelBuilder.CustomAttribute(
                    name=name,
                    frequency="test:node",
                    dtype=wp.int32,
                    namespace="test",
                    references=references,
                )
            )
        sub_builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="node_value",
                frequency="test:node",
                dtype=wp.float32,
                namespace="test",
            )
        )
        for body in range(2):
            sub_builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
            sub_builder.add_custom_values(
                **{"test:node_body": body, "test:environment": -1, "test:node_value": float(body)}
            )

        builder = newton.ModelBuilder()
        builder.add_world(sub_builder)
        builder.add_world(sub_builder)
        model = builder.finalize(device="cpu")
        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(name="src", solver=SolverSemiImplicit, bodies=[0, 2]),
                SolverCoupled.Entry(name="dst", solver=SolverSemiImplicit, bodies=[1, 3]),
            ],
        )
        view = coupled.view("dst")

        self.assertEqual(view.test.node_count, 2)
        np.testing.assert_array_equal(view.test.node_body.numpy(), [0, 1])
        np.testing.assert_array_equal(view.test.environment.numpy(), [0, 1])
        np.testing.assert_allclose(view.test.node_value.numpy(), [1.0, 1.0])
        np.testing.assert_array_equal(view.test.environment_start.numpy(), [0, 1, 2, 2])


if __name__ == "__main__":
    unittest.main()
