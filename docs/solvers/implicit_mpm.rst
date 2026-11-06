.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Implicit MPM Runtime Controls
============================

:class:`~newton.solvers.SolverImplicitMPM` can change grid resolution between
steps while preserving particle motion and constitutive history. Applications
own curriculum schedules; Newton owns grid rebuilding and state preservation.

.. experimental::

   ``set_voxel_size()``, ``set_particle_radius()``, ``contact_recovery_time``,
   and the ``state_prev`` argument to ``project_outside()`` may change without
   prior notice.

Changing resolution
-------------------

Discard captured graphs before changing resolution, then warm up and recapture
the simulation before replay:

.. code-block:: python

   graph = None
   solver.set_voxel_size(0.06, state=state)
   solver.set_particle_radius(0.02)

Voxel changes rebuild sparse or dense grids around the current positions;
fixed grids reject resizing. Particle state, applied forces and collider material
views remain intact. Grid-backed warm starts are discarded. Reserve sufficient
cells and hierarchy nodes for the finest planned resolution. A failed rebuild
restores the old resources. Previously recorded sparse-grid errors must be handled
with an explicit reset; resizing does not clear them.

Particle radius is independent of voxel spacing. Its setter changes uniform
support volume ``8 * radius**3`` and density at fixed mass. It does not split,
merge, add or remove particles. Both setters reject scales whose native volume
or reciprocal cannot be represented in float32. These are runtime operations;
existing USD authoring of initial solver and particle properties is unchanged.

Coarser grids reduce grid work but have fewer collision degrees of freedom near
thin walls. Radius changes alone need not reduce cost. Measure equal simulated
time and check task-specific contact accuracy when choosing a curriculum.

Contact handling
----------------

The default contact response is unchanged. For particle contact bases,
``Config.contact_recovery_time`` optionally recovers a fraction of a negative
gap through contact impulses:

.. code-block:: python

   config = SolverImplicitMPM.Config(collider_basis="pic512", contact_recovery_time=0.01)

Zero disables recovery. A positive time constant ``tau`` targets an outward
relative speed ``-gap * (1 - exp(-dt / tau)) / dt``. Positive-gap speculative
contacts are unchanged. This runtime numerical setting is configured in Python,
not authored as a material property in USD. It can add energy and alter friction,
deformation or geometric volume; it does not guarantee containment.

For applications explicitly using position projection, ``project_outside()``
also accepts ``state_prev``. Preserve its particle positions and collider poses
until the call; ``state_in.body_q`` must contain the end-of-step collider poses.
The sweep tests centers against rigid triangles. It is exact for linear center
motion and translating colliders; rotations require substeps. Finite-radius
sweeps and deforming collider trajectories are unsupported. Up to four contacts
and endpoint projection are applied, with remaining displacement discarded when
that budget is exhausted. Projection supplies no reaction impulses or volume
recovery and remains an explicit operation outside the MPM solve.

Moving-bowl example
-------------------

.. code-block:: console

   uv run --extra examples -m newton.examples mpm_bowl
   uv run --extra examples -m newton.examples mpm_bowl --resize-time 3 --resize-voxel-size 0.06

The CUDA example uses full MPM, sparse grids and particle contacts without
position projection. Its voxel and support-radius sliders recapture the graph
using disposable warm-up states to preserve the live simulation.
