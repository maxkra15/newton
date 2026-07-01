.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton

Coupled Solvers
===============

.. warning::

   Coupled solvers are experimental. The public API currently lives under
   :mod:`newton.solvers.experimental.coupled`, and both the API shape and the
   numerical defaults may change while the feature is refined.

Newton's coupled-solver framework lets one simulation step be split across
multiple solver backends while those backends still exchange forces, poses, and
constraint information through a shared :class:`Model`. This is useful when a
scene combines material models or algorithms that are best handled by different
solvers: for example a MuJoCo or Kamino rigid mechanism coupled to VBD cloth,
XPBD particles coupled to MPM material, or rigid bodies connected to particles
through an ADMM constraint.

The framework is exposed as an experimental namespace rather than as flat
symbols on :mod:`newton.solvers`. Import the coupled solver types directly from
that namespace:

.. code-block:: python

   from newton.solvers import SolverMuJoCo, SolverVBD
   from newton.solvers.experimental.coupled import (
       SolverCoupledADMM,
       SolverCoupled,
       SolverCoupledProxy,
   )

The main public types are:

- :class:`newton.solvers.experimental.coupled.ModelView`: a view-local overlay
  on a shared :class:`Model`.
- :class:`newton.solvers.experimental.coupled.CouplingInterface`: the hook
  protocol implemented by solvers that need custom coupled behavior.
- :class:`newton.solvers.experimental.coupled.SolverCoupled`: the shared base
  for partitioning models, distributing state, stepping entries, and
  reconciling results.
- :class:`newton.solvers.experimental.coupled.SolverCoupledProxy`: a lagged or
  staggered proxy coupling wrapper.
- :class:`newton.solvers.experimental.coupled.SolverCoupledADMM`: a fixed
  iteration ADMM coupling wrapper for model-derived joints, attachments, and
  contacts.

Shared Model, Entry Views, and Ownership
----------------------------------------

Coupled simulations start from a single :class:`Model`. Each sub-solver receives
a :class:`~newton.solvers.experimental.coupled.ModelView` rather than the raw
model. A view delegates reads to the parent model until the coupler or user
applies a view-local override. The important idea is that sub-solvers can see
the same model topology while owning only the bodies, particles, joints, or
shapes assigned to their entry.

A :class:`~newton.solvers.experimental.coupled.SolverCoupled.Entry` describes
one sub-solver:

.. code-block:: python

   entry = SolverCoupled.Entry(
       name="soft",
       solver=lambda view: SolverVBD(model=view, iterations=20),
       bodies=soft_body_ids,
       particles=cloth_particle_ids,
       shapes=cloth_shape_ids,
       substeps=2,
   )

The entry lists the objects the sub-solver owns. During construction,
``SolverCoupled`` creates a model view for every entry, deactivates non-owned
dynamic endpoints where appropriate, constructs the sub-solver by calling the
entry's ``solver(view)`` factory, and keeps per-entry input and output states.
Bind any extra solver constructor arguments in the factory itself. After a
top-level step, only owned outputs are reconciled back into the caller's shared
``state_out``. This prevents two sub-solvers from overwriting the same body or
particle unless an explicit coupling algorithm is responsible for arbitration.

The shared base also manages:

- per-entry substeps, so one solver can take smaller time intervals than
  another;
- copying public force input from ``State`` and ``Control`` into entry-local
  state;
- entry-local collision visibility and shape ownership;
- input-state notifications for solvers with private history buffers;
- fallback effective-mass estimates from public model mass and inertia arrays.

``ModelView`` applies view-local changes with copy-on-write semantics so the
coupler can hide, immobilize, or rescale endpoints without changing the parent
model. Parent-derived view masks are refreshed when relevant model-change
notifications arrive. Direct writes through returned Warp arrays are not
intercepted, so view-local edits should go through the coupled-solver API that
owns the view.

Entry-local Configuration and Collision
---------------------------------------

Use ``Entry.configure_view`` for deliberate, entry-local model overrides. The
callback runs after ownership masks are applied and before the entry collision
provider and solver are constructed. Assigning an array on the view creates an
overlay; it does not replace the parent-model array. For example, this entry
uses different friction and owns its own collision pipeline:

.. code-block:: python

   import newton
   import warp as wp

   from newton.solvers import SolverMuJoCo
   from newton.solvers.experimental.coupled import SolverCoupled

   def configure_rigid_view(view):
       friction = wp.clone(view.shape_material_mu)
       friction.fill_(0.8)
       view.shape_material_mu = friction

   rigid_entry = SolverCoupled.Entry(
       name="rigid",
       solver=lambda view: SolverMuJoCo(
           model=view,
           use_mujoco_contacts=False,
       ),
       bodies=rigid_body_ids,
       shapes=rigid_shape_ids,
       configure_view=configure_rigid_view,
       collision_pipeline=lambda view: newton.CollisionPipeline(
           view,
           contact_matching="latest",
       ),
       collide_interval=2,
   )

An entry-local provider receives the finalized :class:`ModelView`, including
the ``configure_view`` overlay, shape visibility, compact namespace when
requested, and filtered shape pairs. Its persistent contact buffer takes
precedence over outer contacts for that entry. ``collide_interval`` counts
outer coupled steps, not entry substeps; ``None`` refreshes every outer step.
Returning ``None`` from the factory keeps the outer-contact fallback only when
``preserve_shape_ids=True``. Compact entries use entry-local shape IDs, so they
receive no contacts unless the factory supplies an entry-local provider. A
reset clears the current provider contacts and cadence so the next step
refreshes collision even when the interval is greater than one.

Coupling Hooks
--------------

Some solvers keep important state outside the public :class:`State` arrays or
can report interface forces more accurately than a generic momentum fallback.
Those solvers implement
:class:`newton.solvers.experimental.coupled.CouplingInterface` hooks. The
hooks are ordinary methods with default mixin implementations. Solvers override
only the methods that need solver-specific behavior. A solver that cannot
support a hook raises :class:`NotImplementedError` from that hook instead of
silently using an invalid path.

The protocol currently covers these concepts:

- ``coupling_notify_input_state_update()`` tells a solver that public state
  arrays or force-input buffers were changed by the coupler. Its ``flags``
  argument uses :class:`newton.StateFlags`. VBD uses this to realign private
  previous-pose state after proxy synchronization or ADMM iteration restarts.
  MPM uses it to keep collider caches consistent.
- ``coupling_eval_gravity_acceleration()`` lets a solver report the body and
  particle acceleration that it applies internally for gravity-like forces.
  Proxy and ADMM couplers pass these acceleration arrays explicitly to rewind
  and harvest hooks so solvers that scale or compensate gravity can avoid
  double-applying it.
- ``coupling_rewind_proxy_body()`` and
  ``coupling_rewind_proxy_particle()`` let a
  destination solver prepare proxy velocities before a lagged proxy pass.
- ``coupling_harvest_proxy_wrenches()`` and
  ``coupling_harvest_proxy_particle_forces()`` let a destination solver
  report feedback forces from solver-native contact or transfer data.
- ``coupling_prepare_proxy_contacts()`` lets a destination solver filter or prepare
  proxy-local contacts before its step.
- ``coupling_eval_effective_mass()`` and
  ``coupling_eval_effective_mass_block()`` let a solver provide endpoint
  effective mass instead of using raw model mass and inertia.

Force injection itself is not a hook. Couplers write into public
``state.body_f``, ``state.particle_f``, and ``control.joint_f`` buffers, then
call the normal solver step. Likewise, virtual and proximal mass changes are
applied to a ``ModelView`` and refreshed through the usual
``notify_model_changed()`` path when a solver must rebuild private caches.

Contact Streams, Reset, and Graph Capture
-----------------------------------------

Contact stream selection
~~~~~~~~~~~~~~~~~~~~~~~~

A coupled solve can use several contact buffers at once. Select the buffer you
intend by its stable stream name instead of assuming that one aggregate
``Contacts`` object represents the whole solve:

.. code-block:: python

   streams = {
       stream.name: stream
       for stream in coupled.contact_streams(outer_contacts)
   }
   rigid_stream = streams["entry/rigid"]
   counts = rigid_stream.diagnostics()
   readable_rigid_rows = min(
       max(counts.rigid_count, 0),
       counts.rigid_capacity,
   )

``contact_streams()`` may report ``outer``, ``entry/<name>``,
``proxy/<source>/<destination>``, and ``admm/internal`` buffers. Enumeration is
zero-copy and performs no filtering, collision update, device allocation, or
host synchronization. Pass ``outer_contacts`` explicitly before the first
successful step; otherwise the method uses the last successfully stepped outer
buffer. A failed or incomplete step publishes no streams.

The returned descriptors borrow live buffers and optional local-to-parent ID
maps. Treat them as read-only and do not retain them beyond the coupled solver's
lifetime. ``forces_available`` is ``False`` unless the complete stream has a
standardized force representation. Calling ``diagnostics()`` is deliberately
different from enumeration: it reads the device counters on the host and
reports raw counts, capacities, and overflow. Clamp raw counts to capacity, as
above, before indexing fixed-size contact arrays.

World-selective reset
~~~~~~~~~~~~~~~~~~~~~

The reset mask is a one-dimensional Warp Boolean array on the model device,
with one element per regular world:

.. code-block:: python

   import warp as wp

   selected_worlds = wp.array(
       [True, False],
       dtype=wp.bool,
       device=coupled.device,
   )
   coupled.reset(state, world_mask=selected_worlds)

A partial reset clears selected entry state, transient forces, coupling warm
starts, and contact-matching history while preserving unselected worlds.
Global entities use world ID ``-1`` and are never selected by a partial mask.
Current contact buffers and host collision cadence are invalidated globally so
the next step performs fresh collision detection; unselected and global matcher
history remains available to that pass. Omit ``world_mask`` for a full reset.

CUDA graph lifecycle
~~~~~~~~~~~~~~~~~~~~

Query graph support after constructing the complete solver hierarchy, then
prepare persistent buffers before capture:

.. code-block:: python

   if not coupled.supports_cuda_graph_capture:
       raise RuntimeError("coupled configuration is not graph-capturable")

   coupled.prepare_cuda_graph_capture(outer_contacts)

``supports_cuda_graph_capture`` aggregates every entry and optional collision
provider. Entry collision cadence must be one for capture, and dynamic sparse
topology such as sparse implicit MPM reports unsupported. Preparation is
idempotent, recursively prepares providers and sub-solvers, and does not step
the simulation. Call it before capture whenever contact capacities determine
persistent solver storage.

Proxy Coupling
--------------

Proxy coupling represents an endpoint owned by one solver as a proxy endpoint in
another solver. The source solver owns the real object. The destination solver
receives a proxy body or proxy particle with scaled virtual inertia, solves its
own local problem against that proxy, then returns feedback to the source on a
later pass or iteration.

This is a good match for coupling algorithms that are naturally one-way within a
substep but can converge through repeated lagged iterations. Examples include a
rigid body acting as a proxy collider inside a soft-body solve, XPBD particles
driving MPM transfer particles, or VBD reporting contact forces back to a rigid
source body.

A proxy pair is declared with
:class:`newton.solvers.experimental.coupled.SolverCoupledProxy.Proxy`:

.. code-block:: python

   solver = SolverCoupledProxy(
       model,
       entries=[rigid_entry, soft_entry],
       coupling=SolverCoupledProxy.Config(
           proxies=[
               SolverCoupledProxy.Proxy(
                   source="rigid",
                   destination="soft",
                   bodies=robot_body_ids,
                   proxy_bodies=robot_proxy_body_ids,
                   particles=(),
                   proxy_particles=(),
                   mass_scale=0.25,
                   mode="lagged",
                   proxy_relaxation=0.5,
               )
           ],
           iterations=4,
       ),
   )

``source`` and ``destination`` name entries. ``bodies`` and ``particles`` are
source endpoints. ``proxy_bodies`` and ``proxy_particles`` name the
corresponding destination endpoints. If a proxy list is ``None``, the source
indices are reused in the destination view. ``mass_scale`` scales proxy body
mass/inertia and proxy particle mass in that destination view.
``proxy_relaxation`` blends harvested feedback with the previously lagged
feedback buffer after each destination solve. Values below ``1`` underrelax the
update, ``1`` keeps the harvested feedback unchanged, and values above ``1``
overrelax it.

Two proxy modes are available through the ``mode`` string:

- ``LAGGED`` synchronizes the source begin pose and end velocity into the
  destination proxy, rewinds destination proxy velocity by previously applied
  feedback, public force input, and gravity, then steps the destination. This is
  the most common mode for relaxed fixed-point coupling.
- ``STAGGERED`` synchronizes the source end pose and velocity into the
  destination and skips the generic lagged rewind. This is useful when the
  scheduling already gives the destination a current source state.

After the destination step, the coupler harvests feedback. If the destination
solver implements a body or particle harvest hook, that hook can report
contact-native forces or transfer impulses. Otherwise the shared fallback
estimates feedback from proxy momentum change. The fallback is convenient for
simple particle proxy cases, but contact-rich or solver-private interactions are
usually better served by a custom harvest hook.

Proxy-local collision detection is optional. A proxy can provide a
``collision_pipeline`` factory that receives the destination ``ModelView``. If
the factory returns a pipeline, the coupler owns a persistent contact buffer and
refreshes it at ``collide_interval``. If the factory returns ``None`` or no
factory is supplied, the destination solve uses contacts passed to the outer
``step()`` call when the destination preserves parent shape IDs. A compact
destination receives no outer contacts unless it has a local provider.

The generic proxy loop currently supports at most two solver entries. Within
that limit, body and particle mappings are grouped by ``(source,
destination)``. One source step and one destination step are performed for each
solver pair and proxy iteration, so a single proxy declaration can carry both
body and particle mappings around the same destination solve.

ADMM Coupling
-------------

ADMM coupling is the symmetric coupling path. Instead of placing a virtual proxy
inside another solver, it constructs interface rows between endpoints owned by
different entries. Each iteration restores entry states, applies a proximal
velocity target when configured, lets sub-solvers advance, solves local
interface rows, updates dual variables, and splats equal and opposite coupling
forces back to endpoint force buffers.

Compared with proxy coupling, ADMM is less invasive for sub-solvers: entries do
not need to represent proxy bodies or particles, filter proxy contacts, or
harvest proxy-native feedback. The tradeoff is that the coupler must implement
each supported interface row explicitly, so every cross-solver joint, attachment,
and contact type needs ADMM row support. Transient contacts and stiff
attachments also generally need several coupling iterations per step, while the
proxy path is often useful with a single lagged or staggered pass.

The implemented ADMM wrapper discovers constraint rows from the shared model and
enables contact rows through explicit
:class:`newton.solvers.experimental.coupled.SolverCoupledADMM.ContactPair`
objects. It does not currently accept arbitrary user-authored endpoint records
as public API. Supported row sources are:

- cross-solver model joints;
- custom body-particle attachment attributes;
- internally detected rigid-rigid, rigid-particle, and particle-particle
  contacts.

Cross-solver model joints are owned by the coupler only when the two connected
bodies belong to different entries and the joint itself is not owned by either
sub-solver. This avoids solving the same constraint twice. The current generic
ADMM path supports ``BALL``, ``FIXED``, and ``REVOLUTE`` joints. Ball joints
create translational anchor-coincidence rows. Fixed joints add angular rows.
Revolute joints preserve the hinge axis and can add a dry-friction row from
model joint friction. Prismatic, distance, and D6 joint rows are not yet part of
the experimental API.

Body-particle attachments cover interfaces that cannot be represented by a
model joint because one endpoint is a particle. The helper
``SolverCoupledADMM.add_body_particle_attachment()`` registers and fills custom
attributes under ``coupling:body_particle_attachment`` with body id, particle id,
body-local point, stiffness, damping, and enabled state. Importers can author the
same custom attributes directly. Rows whose endpoints are unowned or owned by
the same entry are ignored; only cross-solver attachments are coupled by ADMM.

Contact coupling is enabled by adding one or more ``ContactPair`` values to
``SolverCoupledADMM.Config.contact_pairs``. A contact pair names two entries.
``SolverCoupledADMM.auto_detect_contact_pairs(entries)`` can build the complete
pair list for every distinct entry combination.

For enabled contact pairs, the coupler owns private detection data and builds
rows from solver ownership: particle-shape rows between particle entries and
shapes on bodies owned by other entries, rigid-rigid rows from cross-entry shape
pairs, and particle-particle rows from cross-entry particle sets through a
private hash-grid stream. Friction is read from model material properties such
as ``shape_material_mu`` and ``Model.particle_mu`` at row-fill time; it is not a
``ContactPair`` field. Contact rows use an isotropic Coulomb
maximum-dissipation projection. They do not solve cone complementarity directly.

ADMM contact buffers are fixed-capacity device arrays. Persistent contacts
warm-start local variables and dual variables by stable contact keys across
steps. The particle-particle stream is contacts-like and hash-grid based, but it
is internal to the ADMM coupler and should not be treated as a public contact
stream.

The main ADMM parameters are:

- ``iterations``: fixed iteration count, chosen to be graph-capture friendly;
- ``rho``: penalty weight for interface rows;
- ``gamma``: proximal inertia and velocity weight;
- ``baumgarte``: positional error stabilization for attachment/contact rows;
- stiffness and damping values for model-joint and body-particle attachment
  rows;
- rigid contact matching mode, thresholds, and warm-start force scale.

When ``gamma`` is positive, the coupler scales owned body and particle masses in
each entry ``ModelView``, asks sub-solvers to refresh model-derived caches, and
shifts entry input velocities toward the previous ADMM iterate. Endpoint
effective mass uses solver hooks when available and model fallbacks otherwise.
This keeps the implementation compatible with solvers that can provide an
articulated mass estimate, such as MuJoCo Warp, while still allowing simpler
solvers to participate.

Coupling Diagnostics
--------------------

Proxy diagnostics expose the exact live feedback and Aitken arrays for one
direction. They do not copy arrays or synchronize with the host:

.. code-block:: python

   body_feedback = proxy_solver.proxy_body_feedback("rigid", "soft")
   for source_ids, proxy_ids, forces in zip(
       body_feedback.source_body_ids,
       body_feedback.proxy_body_ids,
       body_feedback.forces,
       strict=True,
   ):
       # ``forces`` is parent-model-sized and indexed at ``proxy_ids``.
       consume_body_feedback(source_ids, proxy_ids, forces)

   relaxation = proxy_solver.proxy_relaxation_diagnostics("rigid", "soft")

Body and particle feedback mappings remain separate and retain configuration
order. Consumers must treat all borrowed arrays as read-only. Fixed relaxation
uses the configured scalar. Aitken ``body_current`` and ``particle_current``
arrays use slot ``0`` for global entities and slot ``world_id + 1`` for a
regular world. A missing direction or requested entity kind raises
:class:`KeyError`.

ADMM reductions are also on demand:

.. code-block:: python

   summary = admm_solver.diagnostics()
   current_contact_count = int(summary.contact_count.numpy()[0])
   for interface in summary.interfaces:
       primal = float(interface.primal_residual_norm.numpy()[0])
       dual = float(interface.dual_residual_norm.numpy()[0])

Calling ``diagnostics()`` launches device reductions into stable, read-only
one-element arrays without changing simulation state. It adds no work to an
ordinary ``step()``. Reading those arrays with ``numpy()`` synchronizes with the
host. ``contact_count`` and ``contact_count_max`` are capacity-clamped;
``contact_overflow`` reports raw excess rows.

Choosing Proxy or ADMM Coupling
-------------------------------

Use proxy coupling when one solver can reasonably treat the other solver's
endpoint as an obstacle, transfer participant, or virtual body over a substep.
Proxy coupling is often easier to tune for collider-style interactions and can
reuse destination solver contact machinery. It is also the path that currently
supports MPM transfer-active proxy particles and deformable collider particles.

Use ADMM coupling when the interface should be represented as a symmetric
constraint or frictional contact between entries. ADMM is better suited for
cross-solver joints, body-particle attachments, and contact rows that need equal
and opposite forces. It is more structured, but it also has more tuning
parameters and a narrower set of supported row types.

The two approaches share the same base concepts: model views, ownership,
entry-local state, force-buffer injection, input-state notifications, and
effective-mass hooks. A scene can often be formulated either way, but the
numerical behavior will differ. Proxy coupling behaves like a relaxed
fixed-point iteration over solver-specific dynamics. ADMM behaves like a fixed
iteration constrained optimization split over the entry solvers and interface
rows.

Solver-Specific Behavior
------------------------

Coupled solvers rely on solver-specific hooks only where generic public
model/state behavior is insufficient.

VBD uses proxy contact preparation, body-proxy harvesting, and input-state
notifications. The notification hook keeps private previous-body state aligned
when proxy poses are synchronized or ADMM iterations restart. The harvest path
reduces final rigid-rigid and body-particle contact forces onto proxy bodies
instead of relying on aggregate momentum differences.

VBD can update several structural joint modes in one validated transaction:

.. code-block:: python

   from newton.solvers import SolverVBD

   vbd.set_joint_constraint_modes(
       [shoulder_joint, elbow_joint],
       hard=[True, False],
   )
   vbd.set_joint_constraint_modes(
       cable_joint_ids,
       hard=False,
       slot=SolverVBD.JointSlot.BEND,
   )

All indices, modes, and structural slots are validated before device state is
changed. The call updates captured buffers in place; switching a slot to soft
clears its persistent lambda and reference state. Call this API before CUDA
graph capture.

Implicit MPM supports proxy body and proxy particle rewind/harvest hooks.
Transfer-active proxy particles can participate in P2G/G2P momentum transfer
while being excluded from material volume, stress, strain, and constitutive
updates. Deformable collider particles registered through collider-particle ids
use collider impulse collection rather than material transfer.

XPBD understands proxy particles and proxy bodies in particle contact kernels.
Owned particles may collide with destination proxy particles, but proxy-proxy,
proxy-static, and proxy-particle versus proxy-body contacts are filtered so the
destination solve does not create feedback between two proxy endpoints or
against immovable particles.

MuJoCo provides GPU effective-mass hooks from MuJoCo Warp data so proxy virtual
inertia and ADMM endpoint weights can use articulated mass estimates rather than
raw body mass. MuJoCo currently marks proxy impulse harvesting as unsupported
because the hook set does not expose a MuJoCo contact-only feedback path.

SemiImplicit preserves externally injected and coupled particle forces by
accumulating particle contact forces into the existing ``particle_f`` buffer.
Kamino accepts ``ModelView`` instances in solver construction and marks proxy
impulse harvesting as unsupported until it exposes a contact-only feedback path.

Current Limitations
-------------------

The coupled-solver framework is useful today, but it is still experimental:

- Proxy stability is tuning-sensitive. Virtual inertia scale, contact
  stiffness, solver iterations, and lagged versus staggered scheduling strongly
  affect damping and convergence.
- Generic momentum harvesting is only a fallback. Solver-private contact modes
  should expose custom harvest hooks where possible.
- ADMM contact detection is internal and does not consume arbitrary caller
  :class:`Contacts` rows as a public interface stream.
- ADMM joint support is limited to ball, fixed, and revolute rows.
- Particle-particle ADMM contacts use a private stream, not a public contact API.
- Effective-mass weighting falls back to simple model mass/inertia where no
  custom hook is available.
- USD ownership, automatic coupled-solver construction, and high-level tuning
  guidance are not part of the experimental public API yet.

Treat coupled solvers as an advanced feature for controlled experiments and
solver integration work. Prefer focused regression tests and explicit scene
tuning when using them in new examples.
