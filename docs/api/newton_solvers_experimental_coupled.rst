.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton.solvers.experimental.coupled
===================================

.. py:module:: newton.solvers.experimental.coupled
.. currentmodule:: newton.solvers.experimental.coupled

.. rubric:: Classes

.. autoclass:: AdmmDiagnostics
   :exclude-members: iterations, contact_count, contact_count_max, contact_overflow, interfaces

.. autoclass:: AdmmInterfaceDiagnostics
   :exclude-members: source, destination, primal_residual_norm, dual_residual_norm

.. autoclass:: CoupledContactDiagnostics

.. autoclass:: CoupledContactStream
   :exclude-members: name, kind, contacts, source, destination, shape_local_to_parent, particle_local_to_parent, forces_available

.. autoclass:: CouplingInterface

.. autoclass:: ModelView

.. autoclass:: ProxyBodyFeedback
   :exclude-members: source, destination, source_body_ids, proxy_body_ids, forces

.. autoclass:: ProxyParticleFeedback
   :exclude-members: source, destination, source_particle_ids, proxy_particle_ids, forces

.. autoclass:: ProxyRelaxationDiagnostics
   :exclude-members: source, destination, global_slot, world_slot_offset, body_mode, body_configured, body_min, body_max, body_world_ids, body_current, body_has_previous, body_stats, particle_mode, particle_configured, particle_min, particle_max, particle_world_ids, particle_current, particle_has_previous, particle_stats

.. autoclass:: SolverCoupled

.. autoclass:: SolverCoupledADMM

.. autoclass:: SolverCoupledProxy
