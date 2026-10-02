.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Implicit MPM and fast particle training
=======================================

:class:`~newton.solvers.SolverImplicitMPM` simulates granular and
elasto-plastic materials on a background grid. The moving-bowl example uses
a capacity-bounded sparse NanoVDB grid, a kinematic bowl mesh, and particle
state shared with :class:`~newton.solvers.SolverXPBD`.

.. experimental::

   The moving-bowl particle-to-MPM training workflow, runtime voxel resizing,
   and the ``state_prev`` swept-projection option are experimental.

Run the example
---------------

.. code-block:: console

   uv run --extra examples -m newton.examples mpm_bowl
   uv run --extra examples -m newton.examples mpm_bowl --solver particles
   uv run --extra examples -m newton.examples mpm_bowl --solver particles --switch-time 3
   uv run --extra examples -m newton.examples mpm_bowl --resize-time 3 --resize-voxel-size 0.06

The example requires CUDA. Its controls change fidelity, grid resolution,
particle radius, friction, and particle cohesion during the simulation.
Even substep counts allow CUDA graph capture; ``--no-use-cuda-graph`` runs
the same pipeline without capture. Configuration changes rebuild the graph
between frames. Bowl motion uses a device clock so graph replay advances
the prescribed trajectory.

`Warp 1.17's hash-grid build
<https://github.com/NVIDIA/warp/blob/v1.17.0/warp/native/hashgrid.cpp>`_
uses a shared host descriptor when recording hash-grid builds in a
CUDA graph. Interleaving builds of different hash-grid instances with particle
graph replay can change its results. This example operates one scene; when
comparing independent particle scenes with this Warp version, run each scene's
captured rollout before building a different hash grid.

Fidelity choices
----------------

The fast stage reuses Newton's XPBD particle contacts and Warp's spatial hash
grid for neighbor searches. It integrates gravity, resolves local particle
overlap and friction, and optionally attracts nearby particles. It skips MPM
transfer, matrix assembly, and the global stress solve. This provides a cheap
single-world granular approximation for policy pretraining. Its behavior
requires validation against the target MPM material and task.

The MPM stage uses the same particle count, positions, velocities, and masses.
Changing fidelity initializes a fresh MPM strain reference and clears stress
history; deformation accumulated during a particle stage has no corresponding
MPM constitutive history. Switching preserves the current particle motion.
This supports policy fine-tuning with full MPM, while physical equivalence
between the stages depends on the workload and is not established by the example.

.. list-table:: Material controls in the example
   :header-rows: 1
   :widths: 25 35 40

   * - Control
     - Fast particles
     - Full MPM
   * - ``--friction``
     - Particle contact friction coefficient
     - Pressure-dependent material friction coefficient
   * - ``--particle-cohesion``
     - Attraction range [m]
     - Unused; calibrate tensile and shear strength separately
   * - ``--yield-stress``
     - Unused
     - Deviatoric yield offset [Pa]
   * - ``--yield-pressure``
     - Unused
     - Compressive pressure cap [Pa]
   * - ``--tensile-yield-ratio``
     - Unused
     - Tensile pressure cap divided by compressive pressure cap
   * - ``--young-modulus``
     - Unused
     - Elastic stiffness [Pa]
   * - ``--viscosity``
     - Unused
     - Viscosity [Pa s]

Particle attraction range and continuum yield stress have different units and
semantics. There is no universal conversion between them. Calibrate them with
task measurements such as settling height, slip, and retained mass. The
example's collision projection treats particle centers as points; XPBD uses
the radii for particle-to-particle contacts.

Performance measurements
------------------------

Run the included benchmark with:

.. code-block:: console

   uv run --extra dev asv/benchmarks/simulation/bench_mpm_bowl.py

The following measurements used an RTX 4090, Warp 1.17.0, a 0.08 m sparse grid,
60 Hz frames, two substeps per frame, a configured MPM iteration budget of 100,
and two XPBD iterations. Each time is the median of three runs, each averaging 30
synchronized frames following warm-up. It includes bowl motion, grid or neighbor
rebuilds, integration, and swept collision projection. Rendering and compilation
are excluded. Both fidelities use the same initial scene and simulated time per
frame.

Run on an otherwise idle GPU. Concurrent workloads on this shared device
produced 4.4 ms particle frames and 92--98 ms MPM frames in a later run.
The table below reports the earlier measurements; absolute timings depend on
the available GPU and host capacity.

.. list-table:: Frame time [ms]
   :header-rows: 1
   :widths: 15 15 20 20 15

   * - Particles
     - Graph capture
     - Fast particles
     - Full MPM
     - Speedup
   * - 6,645
     - Enabled
     - 1.03
     - 16.29
     - 15.8x
   * - 60,952
     - Enabled
     - 1.54
     - 19.44
     - 12.6x
   * - 6,645
     - Disabled
     - 1.13
     - 26.96
     - 23.9x
   * - 60,952
     - Disabled
     - 1.70
     - 27.66
     - 16.3x

These compare two physical models. They are evidence for a fidelity/performance
tradeoff, not a speedup of the identical MPM equations. With full MPM alone,
graph replay reduced frame time by about 1.4--1.7x in these runs. Contact density,
convergence, grid capacity, material stiffness, and scene motion affect timings.

Why full MPM costs more
-----------------------

Newton's implicit MPM pipeline bins particles into grid cells with Warp FEM
``PicQuadrature``, transfers mass and momentum, samples collider signed
distances, assembles strain and compliance operators, solves coupled contact
and rheology constraints, and transfers velocity and strain back to particles.
Sparse storage bounds the spatial domain but does not remove operator assembly
or iterative stress/contact work.

Warp's installed ``examples/fem/example_apic_fluid.py`` uses the same NanoVDB
and particle-quadrature infrastructure. It assembles a divergence operator and
solves a pressure Schur complement. Newton extends this pattern to continuum
stress, plastic yield, viscosity, and coupled frictional contact. Warp's
``examples/core/example_dem.py`` illustrates the cheaper spatial-hash neighbor
path, but its stiff explicit penalty contacts use many substeps. The bowl's
fast stage uses XPBD positional contacts instead of that penalty-force method.

The relevant upstream examples are `Warp APIC fluid
<https://github.com/NVIDIA/warp/blob/v1.17.0/warp/examples/fem/example_apic_fluid.py>`_
and `Warp DEM
<https://github.com/NVIDIA/warp/blob/v1.17.0/warp/examples/core/example_dem.py>`_.
Warp documents the underlying `mesh ray queries
<https://nvidia.github.io/warp/v1.17/language_reference/_generated/warp.mesh_query_ray.html>`_.

The following sections describe the operations behind those costs. Equations
describe active particles with positive mass and the example's PIC quadrature;
GIMP distributes a particle's support across additional quadrature samples.
Internal cell-volume normalizations are stated where they affect the formulas.

Substep data flow
-----------------

.. mermaid::

   flowchart TD
       P["Persistent particle state"] --> F{"Fidelity"}
       F -->|MPM| G["Sparse cells and APIC transfer"]
       G --> A["Assemble strain, compliance, and collider operators"]
       A --> S["Coupled stress and frictional contact iterations"]
       S --> U["Update particle velocity and material history"]
       F -->|Particles| X["Ballistic integration and hash-grid neighbors"]
       X --> L["Local particle position iterations"]
       U --> C["Swept mesh projection"]
       L --> C
       C --> N["Next particle state"]
       N --> P

The prescribed bowl poses are sampled at the start and end of every substep.
Both routes use the same meshes and saved start state for the final projection.
Only the MPM route uses the FEM grid and advances constitutive history.

Particle state and discretization
---------------------------------

Particles carry the material history while the grid carries the velocity and
stress solve for the current step. A particle's grid cell is computed from its
position and the voxel edge length :math:`h`. Warp's ``PicQuadrature`` bins
particles into the active cells and uses their support volumes as integration
measures.

.. list-table:: Persistent particle quantities
   :header-rows: 1
   :widths: 30 70

   * - Quantity
     - Role
   * - ``particle_q``, ``particle_qd``
     - Position :math:`x_p` [m] and velocity :math:`v_p` [m/s].
   * - ``particle_mass``, ``particle_radius``
     - Mass :math:`m_p` [kg] and support/contact radius :math:`r_p` [m].
   * - ``mpm.particle_qd_grad``
     - Velocity gradient :math:`C_p` [1/s], also used for the APIC affine
       velocity prediction.
   * - ``mpm.particle_elastic_strain``
     - Elastic deformation matrix :math:`F_{e,p}`. Despite the attribute
       name, its undeformed value is the identity matrix.
   * - ``mpm.particle_Jp``
     - Plastic volume ratio used by the hardening/softening history.
   * - ``mpm.particle_stress``
     - Material stress [Pa], including a particle-backed warm-start guess.
   * - ``mpm.particle_transform``
     - A separate accumulated deformation frame for grain rendering.

Newton derives support volume and density from the model:

.. math::

   V_p = 8r_p^3,\qquad \rho_p = \frac{m_p}{V_p}.

This is a cubical support-volume convention; it is not the sphere volume
:math:`4\pi r_p^3/3`. The bowl emits particles at spacing :math:`2r_p` and
assigns mass :math:`\rho_0(2r_p)^3`, which is consistent with this convention.

The example uses three different approximation spaces:

.. list-table:: Grid fields
   :header-rows: 1
   :widths: 20 15 65

   * - Field
     - Basis
     - Consequence
   * - Velocity
     - ``Q1``
     - Trilinear interpolation; a particle couples to the eight corner nodes
       of its cell.
   * - Strain/stress
     - ``P0``
     - One constant symmetric-tensor sample per active cell, with six tensor
       components.
   * - Collider data
     - ``S2``
     - Quadratic serendipity samples provide more detailed collider response
       than the velocity grid. An interpolation operator couples the spaces.

The sparse grid stores occupied cells and their shared nodes, rather than a
dense box containing all empty space. It still needs particle binning, FEM
space partitions, coefficient integration, and the stress/contact solve.
Particles in the same cell share continuum stress DOFs; this is a different
interaction model from independent sphere contacts.

Mass and momentum transfer
--------------------------

Let :math:`w_{ip}=N_i(x_p)` be the velocity basis weight at particle :math:`p`,
and :math:`x_i` the position of grid node :math:`i`. Before constraints, the
mass and APIC momentum transfers are:

.. math::

   \bar m_i &= \frac{1}{h^3}\sum_p w_{ip}m_p,\\
   \bar p_i &= \frac{1}{h^3}\sum_p w_{ip}m_p
       \left[v_p+\Delta t\,g_p+C_p(x_i-x_p)\right],\\
   u_i^* &= \frac{\bar p_i}{\bar m_i+d_{\mathrm{air}}},
       \qquad d_{\mathrm{air}}=\mathtt{air\_drag}\,\Delta t.

The kernels integrate particle density against the quadrature volume, giving
the same :math:`m_p=\rho_pV_p` factors. The :math:`h^{-3}` normalization
keeps the arrays in the solver's cell-volume convention.
``integrate_mass``, ``integrate_velocity``, and ``integrate_velocity_apic``
perform these integrations; ``free_velocity`` applies the regularized inverse
mass. The air term damps low-occupancy/background nodes.

The affine term carries subparticle velocity variation. Omitting it gives a
PIC transfer, which can introduce more numerical dissipation. Choosing PIC
still leaves the grid assembly and rheology solve in place; the fast XPBD
stage skips that entire pipeline.

The coupled implicit solve
--------------------------

Newton builds a strain operator :math:`B`, an elastic compliance operator
:math:`A_e`, and a collider interpolation operator :math:`J`. The strain
operator includes :math:`\Delta t`, gradients of the velocity basis, particle
quadrature, and the cell-volume normalization. Its stored three-component
stencils are expanded into symmetric-tensor operations by the kernels.

With these assembly factors absorbed into the symbols, stress and contact
updates change the grid velocity schematically as:

.. math::

   u = u^* + M^{-1}B^\mathsf{T}\sigma
             + M^{-1}J^\mathsf{T}\lambda.

Here :math:`M^{-1}` is the regularized lumped inverse mass,
:math:`\sigma` uses the solver's compression-positive stress convention, and
:math:`\lambda` is the collider impulse. The constitutive update combines
velocity-induced strain, the previous elastic strain, compliance, and plastic
flow. Its stress-response operator contains:

.. math::

   D = A_e + BM^{-1}B^\mathsf{T}.

This is the Delassus operator: a stress change alters grid velocity, which
alters strain at other material samples. It explains why local material
updates remain globally coupled. Newton already applies stress-driven
velocity changes without assembling this complete global matrix.
It assembles the strain/compliance stencils and factorizes local diagonal
blocks for the nonlinear solve.

For the bowl's ``Q1`` velocity basis, ``solver="auto"`` chooses colored
Gauss-Seidel. Strain nodes within a color can update in parallel without
sharing velocity DOFs; colors are processed in sequence. The local stress
update resolves the yield/flow rule, and collider contact updates constrain
relative normal velocity and tangential slip. The solve repeats these updates
and checks stress-update residuals. A lower residual improves constitutive
and contact convergence, but does not test whether a particle crossed a mesh
between two states.

Residual checking is batched. The conditional CUDA graph checks after groups
of five stress/contact sweeps. The host loop uses the solver's own group size:
25 for Gauss-Seidel and 50 for Jacobi. Both paths shorten the last group to
respect ``max_iterations``, including budgets of one or two. Convergence can
stop the solve at an earlier group boundary. A zero budget skips the nonlinear
loop. CUDA graph replay resets its iteration counter before each solve.

This branch fixes an earlier discrepancy: CUDA rounded small budgets up to
five, while the host loop skipped a budget smaller than its group size. Use
this corrected behavior when comparing low-iteration approximations.
``max_iterations`` bounds the final nonlinear solve; chained warm-start stages
add their own work. For example, ``("jacobi", "gs")`` adds five Jacobi smoother
sweeps before the Gauss-Seidel budget. Assess tolerance, iteration budget,
residuals, and the complete frame cost together.

There are existing ``gs-soa`` and ``gs-batched`` variants. The former changes
matrix memory layout; the latter trades some within-batch sequential
dependence for parallel work. Wider ``B2``/``B3`` velocity stencils select the
batched variant automatically. Measure each variant at matching material,
tolerance, and trajectory. A linear-only CG/CR/GMRES solve is not a replacement
for the bowl's frictional contact and nonlinear granular flow rule.

Elasticity, yielding, and viscosity
-----------------------------------

Finite elasticity uses the Hencky strain measure. If
:math:`F_e=U\operatorname{diag}(s)V^\mathsf{T}`, the previous elastic
strain contribution is formed from:

.. math::

   \varepsilon_e
      = U\operatorname{diag}(\log s)U^\mathsf{T}.

The isotropic stress-to-strain relationship is:

.. math::

   \mathcal{C}(\sigma)
     = \frac{(1+\nu)\sigma-\nu\operatorname{tr}(\sigma)I}{E},

where :math:`E` is Young's modulus [Pa] and :math:`\nu` is Poisson's ratio.
The assembly rotates this relationship into the current deformation frame
and includes the material damping factor. Stiffnesses at the solver's
:math:`10^{12}` threshold are treated as effectively infinite. The example's
default :math:`E=10^{15}` therefore selects the stiff granular limit; use a
finite lower stiffness to exercise elastic deformation.

The yield surface stores six interpolated parameters: compressive pressure
bound, tensile pressure bound, shear yield stress, pressure-dependent friction,
dilatancy, and viscosity. The normal component is bounded, and the admissible
deviatoric stress magnitude is a capped, piecewise pressure-dependent limit.
Tensor components use Warp's orthonormal symmetric-tensor mapping, and the
packed pressure bounds include a factor of :math:`\sqrt{3/2}`. Raw tensor
entries and packed normal/deviatoric coordinates must use those conversions.
``yield_stress`` contributes a cohesive shear offset. ``tensile_yield_ratio``
controls the tensile pressure bound, so cohesive shear strength and tensile
strength are separate material choices.

Hardening scales the pressure and shear bounds through the stored plastic
volume ratio; it does not scale the elastic modulus. Viscosity is passed into
the discrete flow rule as :math:`\eta/\Delta t`. Changing the timestep
therefore changes both temporal resolution and the coefficients of the
discrete material solve. These are continuum parameters, unlike an XPBD
neighbor attraction distance.

Grid-to-particle update and material history
--------------------------------------------

After convergence, the solved velocity field is sampled at each particle:

.. math::

   v_p^{n+1} &= \sum_i w_{ip}u_i,\\
   C_p^{n+1} &= \sum_i u_i\otimes\nabla N_i(x_p),\\
   x_p^{n+1} &= x_p^n+\Delta t\,v_p^{n+1}.

The implementation clamps sampled velocity to ``model.particle_max_velocity``
before advection. This is a speed bound, not a timestep chosen from voxel size.
GIMP combines multiple quadrature contributions with support-volume weights.

The elastic deformation update combines the solved elastic increment with the
skew-symmetric velocity-gradient increment. Plastic volume is updated
multiplicatively from the trace of the plastic increment, using the configured
hardening/softening rates and bounded volume changes. Particle stress is
interpolated and projected onto the updated yield surface. The rendering
deformation frame is advanced separately from this elastic history.

The example performs swept projection after these updates. On a contact, the
projector removes the symmetric part of the stored velocity gradient, retaining
its rigid rotation component. This is an additional approximate contact
correction; it does not rerun the stress solve after moving a particle.

What the fast particle stage computes
-------------------------------------

The fast stage first performs ballistic integration. It builds a Warp hash
grid from the predicted positions once per substep, with cell size
:math:`2r_{\max}+\kappa`, where :math:`\kappa` is ``particle_cohesion``.
Particle :math:`i` queries neighbors within
:math:`r_i+r_{\max}+\kappa`. The contact kernel traverses nearby buckets
instead of testing every particle pair.

For a pair, define the signed gap and inverse masses:

.. math::

   c_{ij}=\|x_i-x_j\|-r_i-r_j,\qquad
   n_{ij}=\frac{x_i-x_j}{\|x_i-x_j\|},\qquad
   w_i=\frac{1}{m_i}.

Ignoring the tangential correction, the contribution to particle :math:`i`
is:

.. math::

   \Delta x_i
      = -\omega\,\frac{w_i}{w_i+w_j}\,c_{ij}\,n_{ij},
      \qquad c_{ij}\leq\kappa,

where :math:`\omega` is the contact relaxation factor. Negative gaps push
overlapping particles apart; positive gaps within the cohesion range attract
them. Zero distance and zero combined inverse mass are skipped.
For overlapping contacts, tangential correction is limited by the friction
coefficient times overlap and the tangential motion over the substep.
Corrections are accumulated atomically, applied over the requested position
iterations, and used to update particle velocity.

This path stores no continuum stress or elastic/plastic strain history.
It has no material matrix assembly or global yield solve. Its cost is roughly
neighbor construction plus :math:`O(PkI)` local work for :math:`P` particles,
:math:`k` nearby candidates, and :math:`I` position iterations. This estimate
assumes bounded neighbor occupancy; large radii, cohesion ranges, or severe
overlap increase :math:`k`. The hash grid is reused across those iterations,
so large corrections and large timesteps can still miss new particle contacts.

The example passes no external shape-contact array to XPBD. The common MPM
mesh projector handles the bowl after either solver. Fast frames do not
rebuild an MPM grid, although the MPM solver remains allocated for its collider
data and for a later fidelity switch.

Both stages are suitable for simulation-based policy training. The current MPM
kernel modules and swept projector disable backward generation; this example
does not provide gradients through the simulation.

Fidelity handoff and calibration
--------------------------------

``Example.set_solver_type()`` resets MPM history in both ping-pong state
buffers. It keeps particle position, velocity, mass, radius, and material
configuration. Elastic and rendering deformation matrices return to identity,
stress and velocity gradient return to zero, and plastic volume returns to its
configured initial value. Grid/contact warm starts are cleared. The simulation
clock and prescribed bowl trajectory continue.

This defines a new material reference at the handoff configuration. It avoids
using stale continuum stress after an XPBD rollout, but it cannot recover the
deformation or plastic loading path that a full MPM rollout would have followed.
Keep observations and actions consistent across fidelities, and validate the
handoff against task outcomes before relying on policy transfer.

Useful calibration measurements include retained mass under the same bowl
trajectory, settling height, angle of repose, slip onset, and response to a
short acceleration pulse. Match physical control intervals and total mass.
Tune fast pair friction/cohesion against those measurements, then fine-tune
with the target MPM yield, elasticity, and viscosity settings. The mesh's wall
friction is initialized at construction; the UI friction slider adjusts pair
friction and the MPM material coefficient.

Where further performance work should go
----------------------------------------

The current implementation uses the existing full MPM equations and adds a
separate approximate particle stage. Further changes should be measured
individually:

1. **Reduce launch overhead.** Capture the frame and keep time-varying inputs in
   device arrays. This retains the chosen equations and solver tolerance.
2. **Reduce assembly traffic.** Prototype direct applications of strain and
   collider stencils, or fuse transfers with shared particle/cell reads. The
   target is the assembled :math:`B`/:math:`J` data and intermediate arrays;
   the complete :math:`D` matrix is already avoided.
3. **Improve convergence.** Compare existing layouts and preconditioners at
   matching residual thresholds. Fewer sweeps at a worse residual represent
   a fidelity change.
4. **Coarsen resolution selectively.** Coarser voxels reduce active DOFs but
   change the discretization. Keep the policy's action period fixed while
   testing voxel size, particle density, and substep count separately.

Measure assembly, solve, transfer, and projection time independently, then
check end-to-end frame time. Kernel fusion can trade launch count for register
pressure or atomic contention. Require convergence and trajectory comparisons
before claiming a speedup of the same physical model. Particle-count reduction
is a separate resampling problem that must preserve mass and momentum.

Collision tunneling
-------------------

An endpoint signed-distance query can miss a thin wall entirely when a particle
finishes beyond the query range or on the opposite side of the wall. Reducing
MPM iterations also leaves larger contact residuals. Implicit stability does
not guarantee collision containment or timestep accuracy.

Pass the start-of-step state to
:meth:`~newton.solvers.SolverImplicitMPM.project_outside` to test the trajectory:

.. code-block:: python

   # Save before integration: XPBD may overwrite its input position buffer.
   wp.copy(previous.particle_q, state_in.particle_q)
   wp.copy(previous.body_q, state_in.body_q)
   solver.step(state_in, state_out, None, None, dt)
   # Set state_out.body_q to the actual end-of-step collider poses here.
   mpm_solver.project_outside(state_out, state_out, dt, state_prev=previous)

The sweep finds the first entering triangle along the particle center's relative
trajectory, carries the contact point with the collider, and permits frictional
slip. It catches static-wall crossings and translating walls that move through
stationary particles without increasing rheology iterations. The same projector
works on particle states produced by XPBD.

The centerline intersection test is exact for straight particle trajectories
and translating rigid meshes. Rotation is approximated by a straight segment
between the two local-frame endpoints, so large rotations require substeps.
Up to four swept contacts are resolved, followed by endpoint projection. If
that budget is exhausted, the remaining displacement is discarded. This is
an approximate contact integrator, and particle radii are absent from the sweep.
Deforming meshes are unsupported, and the correction does not apply reaction
impulses to rigid bodies. It suits the prescribed kinematic bowl; dynamic
two-way coupling requires contact impulses
from the coupled solve. Large timesteps can still cause particle-particle
crossings, inaccurate packing, or missed rotational contacts.

Swept contact algorithm
-----------------------

Let :math:`T_0,T_1` be the collider's start and end rigid transforms and
:math:`x_0,x_1` the particle's start and predicted end positions. The first
query uses the local-frame segment:

.. math::

   a=T_0^{-1}x_0,\qquad b=T_1^{-1}x_1,\qquad
   \ell(t)=a+t(b-a),\quad 0\leq t\leq1.

The ray direction is :math:`b-a`, without normalization, so the hit parameter
is a fraction of the substep. Each applicable collider is queried through its
mesh BVH. The nearest entering face is selected, using stable collider and
face-material IDs. A start point slightly inside a wall can be repaired by a
closest-point query before retrying the ray.

For a local hit point :math:`c_\ell` and face normal :math:`n_\ell`, the
contact is carried to the collider's end pose:

.. math::

   c=T_1c_\ell,\qquad n=R_1n_\ell,\qquad
   v_c=\frac{T_1c_\ell-T_0c_\ell}{\Delta t}+v_{\mathrm{mesh}}.

Here :math:`R_1` is the end rotation, and any mesh surface velocity is rotated
into the world frame. The new projection path uses the supplied poses for
finite-difference velocity; it does not additionally extrapolate them from
``body_qd``.

For a relative displacement or velocity :math:`q`, define
:math:`q_n=q\cdot n` and :math:`q_T=q-q_nn`. The isotropic Coulomb response
implemented by the projector is:

.. math::

   \mathcal{P}_{\mu}(q)=
   \begin{cases}
     q, & q_n\geq0,\\
     \max\left(0,1-\dfrac{\mu(-q_n)}{\|q_T\|}\right)q_T,
       & q_n<0,\ \|q_T\|>0,\\
     0, & q_n<0,\ \|q_T\|=0.
   \end{cases}

It removes incoming normal motion and limits tangential slip using that normal
correction. With material thickness :math:`d_m` and projection threshold
:math:`\theta_m`, the update is:

.. math::

   \delta &= \max(10^{-6}\ {\rm m},d_m-\theta_m),\\
   x' &= c+\delta n+\mathcal{P}_{\mu}(x_1-c),\\
   v' &= v_c+\mathcal{P}_{\mu}(v-v_c).

The remaining tangential path is queried again against end-pose geometry.
This catches a second wall or bowl facet after the first collision. Four
contact passes bound the work; the last pass discards remaining displacement
if all four passes hit. A final endpoint signed-distance query repairs local
penetration into the offset surface.

The first ray has no voxel-length limit, which is why it catches a complete
thin-wall crossing that an endpoint query can miss. The endpoint and
initial-penetration repairs still use the configured closest-point query range.
This procedure is a conservative motion truncation at the contact budget,
rather than a continuous-time impact solve with a remaining-time integrator.
Its friction and gradient corrections can dissipate motion.

For translation, the relative trajectory is a straight segment. Under rotation,
the exact trajectory in the mesh frame is curved. A chord between its endpoints
can miss an obstacle on that curve. Monitor angular travel
:math:`\|\omega\|\Delta t` and split large rotations into smaller steps.
Also monitor particle travel relative to diameter,
:math:`\|v_i-v_j\|\Delta t/(r_i+r_j)`, when packing accuracy matters: the mesh
sweep does not resolve particle-particle crossings.

Change resolution and radius in flight
--------------------------------------

:meth:`~newton.solvers.SolverImplicitMPM.set_voxel_size` rebuilds a sparse or
dense grid around the current particle positions:

.. code-block:: python

   mpm_solver.set_voxel_size(0.06, state=state)

Positions, velocities, particle stress, elastic strain, plastic volume, and
deformation frames retain their values. Grid warm starts are discarded.
Collider margins remain in meters. This is a runtime resolution choice;
initial authored resolution continues to use ``NewtonMPMSceneAPI.voxelSize``.
Fixed grids cannot be resized with this method. Call outside graph capture,
discard old graphs, and capture again. Reserve enough active-cell capacity for
the finer grid; a failed capacity check restores the original solver resources.
Inspect :meth:`~newton.solvers.SolverImplicitMPM.check_sparse_grid_rebuild_status`
after replay to detect later capacity overflow.

Particle radii can already be changed through the model:

.. code-block:: python

   model.particle_radius.fill_(radius)
   model.particle_max_radius = radius  # XPBD neighbor-search bound
   mpm_solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)

Newton derives MPM support volume as ``8 * radius**3`` and density as mass divided
by this volume. Keeping mass fixed therefore changes density. Changing particle
radius changes the modeled material and contact spacing; changing voxel size
changes the numerical resolution. Neither operation changes particle count or
respawns particles. Recapture graphs after model changes that replace cached
buffers or alter scalar neighbor-search parameters. Particle radius and model
material properties use their existing model attributes and USD schemas.

Resolution changes and sparse capacity
--------------------------------------

Changing :math:`h` changes the integer cell coordinates, shared grid nodes,
shape-function gradients, and operator dimensions. ``set_voxel_size()``
therefore allocates a new grid, partitions, scratch fields, and last-step grid
history around the current positions. It retains the previous rigid-body poses
used by backward-difference collider velocity. The new resources are committed
only after allocation and capacity checks succeed; failure restores the old
voxel size, resources, and status arrays.

Particle-backed history remains in the current state. This makes
``warmstart_mode="particles"`` useful for changing topology: stresses can be
integrated onto the new grid. Grid-backed warm starts belong to the old
discretization and are discarded. Rebuildable sparse grids already require
particle-backed rather than grid-backed stress warm starts.

There are independent limits for active cells and NanoVDB hierarchy nodes.
The bowl reserves 4,096 active cells, 128 leaf nodes, 64 lower nodes, and 16
upper nodes. A spatially scattered cloud can consume more hierarchy nodes
than a compact cloud with the same active-cell count. Those particular limits
are for the bowl, rather than a general particle-count capacity formula.

For a well-sampled filled region of volume :math:`V`, grid work initially
scales with :math:`V/h^3`; halving :math:`h` can require about eight times as
many cells. As cells become sparsely sampled, finite particle count limits
occupancy and the quality of the continuum discretization also changes.
Shrinking voxels without adding particles does not create more material samples.

The ratio of particle spacing to voxel size controls how many particles
contribute to each stress sample. The default bowl has spacing 0.032 m and
voxels 0.08 m wide. Decreasing voxel size and decreasing particle radius are
separate experiments. If radius is halved while mass is fixed, support volume
falls by eight and computed density rises by eight. Preserving density would
require changing masses too, which changes total mass when particle count
stays fixed.

CUDA graphs and runtime changes
-------------------------------

Graph replay uses the arrays and scalar arguments recorded at capture time.
Replacing grid resources, changing the active solver branch, or changing a
Python scalar used by XPBD neighbor searches requires a new graph. The example
invalidates the old graph before each of these changes.

Capture first warms up on temporary state buffers. This initializes lazy FEM
and solver kernels without advancing the live particles. It then restores the
live states and device clock, clears stale collider/grid warm starts, and
captures the frame. An even number of substeps restores the original
ping-pong buffer ordering, so the next replay reads the previous frame's output.
Odd substep counts use the uncaptured route.

The bowl clock is a device array incremented by the captured frame. The motion
kernel reads it when the graph executes, so replay advances the trajectory.
Capturing a Python floating-point time instead would record a fixed value and
repeat the same motion. Host ``sim_time`` drives the scheduled fidelity and
resolution changes between frames. Conditional GPU graphs also control the
inner MPM convergence loop when the device supports them.

The example reads sparse rebuild status after every MPM frame. That read is a
host synchronization and is included in the benchmark. It detects later
capacity overflow even when the grid rebuild runs inside a captured frame.
Capacity reservation, graph replay, and topology validity are all needed for
the captured path; graph capture alone does not provide overflow handling.

Validation and profiling
------------------------

.. code-block:: console

   uv run --extra dev -m unittest newton.tests.test_implicit_mpm_adaptive
   uv run --extra dev -m unittest newton.tests.test_examples.TestMPMExamples -k mpm_bowl
   uv run --extra dev asv/benchmarks/simulation/bench_mpm_bowl.py

The focused tests cover complete thin-wall crossings, earliest-hit selection,
a second contact at a corner, a translating wall crossing a stationary
particle, inactive/zero-mass passthrough, and collider-world isolation.
Resolution tests cover state preservation, invalid sizes, fixed-grid rejection,
capture-time rejection, and rollback after sparse capacity overflow.
The bowl tests cover full MPM, a particle-to-MPM handoff with resizing, and
20 Hz frames with one substep in each fidelity. The low-budget MPM case requests
and executes one nonlinear iteration. Iteration-loop tests count actual device
sweeps for small and nondivisible budgets, early convergence, independent
worlds, and repeated conditional graph launches.

Profile a warmed-up rollout with fixed initial conditions, particle count,
physical frame duration, material parameters, grid settings, and convergence
thresholds. Synchronize the device when timing complete frames and exclude
compilation/rendering. Compare retained mass and trajectories as well as time.
For the fast/full comparison, record both the runtime gain and the material
behavior difference. For optimizations within full MPM, verify matching
residuals and history evolution.

Development split: Newton and Isaac Lab
-----------------------------------------

The recommended split keeps numerical operations in Newton and training
schedules in Isaac Lab. Inspection of Isaac Lab's ``develop`` branch at
``e1988b935352d14fdcaddff996c8cd7ae6990a03`` (2026-10-01) found existing
integration points:

* `MPMSolverCfg <https://github.com/isaac-sim/IsaacLab/blob/e1988b935352d14fdcaddff996c8cd7ae6990a03/source/isaaclab_newton/isaaclab_newton/physics/mpm_manager_cfg.py>`_
  already exposes sparse-grid capacities, basis choices, solver chains, warm
  starts, and optional post-step projection.
* `NewtonMPMManager <https://github.com/isaac-sim/IsaacLab/blob/e1988b935352d14fdcaddff996c8cd7ae6990a03/source/isaaclab_newton/isaaclab_newton/physics/mpm_manager.py>`_
  owns stepping, capture compatibility, sparse rebuild status, and history resets.
* `MPMObjectData <https://github.com/isaac-sim/IsaacLab/blob/e1988b935352d14fdcaddff996c8cd7ae6990a03/source/isaaclab_newton/isaaclab_newton/assets/mpm_object/mpm_object_data.py>`_
  gathers fixed-size particle position, velocity, and state buffers, and per-object
  mean position/velocity. Its reads come from Newton's particle state.
* `Curriculum terms <https://github.com/isaac-sim/IsaacLab/blob/e1988b935352d14fdcaddff996c8cd7ae6990a03/source/isaaclab/isaaclab/envs/mdp/curriculums.py>`_
  can schedule task parameters. Changing a configuration field alone does not
  rebuild a live solver or update a recorded CUDA graph.

.. list-table:: Ownership and implementation order
   :header-rows: 1
   :widths: 26 27 47

   * - Work item
     - Owner
     - Next concrete change
   * - Swept rigid collision projection
     - Newton solver; Isaac Lab stepping adapter
     - Propose the tested Newton API independently. Add previous-position and
       previous-pose scratch buffers to the Isaac Lab manager before using it.
   * - Runtime voxel size
     - Newton solver; Isaac Lab curriculum
     - Propose ``set_voxel_size`` independently. Add an Isaac Lab manager method
       that updates the live solver, its config, and graph capture together.
   * - Exact small iteration budgets
     - Newton solver
     - Submit the loop fix and regression tests separately from experimental
       predictors or material changes.
   * - Particle/MPM fidelity switching
     - Newton demonstration; Isaac Lab task/manager
     - Keep the bowl as a demonstration. Add the policy curriculum, observation
       contract, reset semantics, and robot coupling in Isaac Lab.
   * - Runtime particle radius
     - Newton model properties; Isaac Lab curriculum
     - Preserve the existing model update/notification path. Let the task choose
       when to change radius and whether the density change is intentional.
   * - Automatic particle splitting/merging
     - Future Newton solver/model feature
     - Design mass, momentum, constitutive-history transfer, and stable particle
       identity before adding a training schedule. This branch does not do it.
   * - Stress warm-start predictor and volume recovery
     - Experimental benchmark first; Newton if validated
     - Measure accuracy, contact residuals, energy, and cost on several granular
       scenes before exposing a solver configuration parameter.

Isaac Lab currently steps implicit MPM in place. Passing ``state_0`` as
``state_prev`` after that step would use already-updated positions. The swept
adapter must copy particle positions and collider poses before the substep,
then use endpoint collider poses for projection. It must also account for
prescribed kinematic motion and for a solver owned by a coupled manager.

Runtime resolution and fidelity changes must happen at a physics boundary,
outside capture. Isaac Lab's `NewtonManager <https://github.com/isaac-sim/IsaacLab/blob/e1988b935352d14fdcaddff996c8cd7ae6990a03/source/isaaclab_newton/isaaclab_newton/physics/newton_manager.py>`_
has graph invalidation and deferred recapture machinery. A manager-level
voxel-size operation should call Newton's transactional resize, update its
configuration only on success, and schedule capture before the next replay.
It should preserve the asset's position/velocity buffers and invalidate any
consumer caching grid-field pointers. Newton's voxel size is currently one
setting for the whole solver, including separate worlds; a curriculum should
therefore change it for the batch together.

The particle/MPM training manager needs temporary XPBD working buffers because
XPBD can overwrite its input positions. Keep particle IDs, count, mass,
position/velocity layout, action units, observation ordering, policy timestep,
and reward definitions stable. Publish the same asset data after either
solver. Reset both MPM material-state buffers when returning from the particle
stage, using the current configuration as the new strain reference. That
preserves particle motion but introduces a deliberate material-history reset.

The bowl and its stress predictor are single-world experiments. The current
XPBD particle-pair kernel queries spatial neighbors without a world-ID filter.
Training with overlapping world coordinates therefore needs world filtering
in Newton or physically separated neighbor-query regions. Per-environment
material randomization also needs an adapter: XPBD's particle-pair friction
and attraction coefficients are model scalars, while MPM's material fields
are particle arrays. A shared observation layout alone does not resolve
either difference.

For force observations, define a fixed probe or robot-wrench interface in SI
units, with the same frame, averaging interval, filtering, and clipping in
both stages. The fast stage can supply an analytical approximation; the full
stage must read the appropriate MPM/rigid coupling reaction. The bowl's swept
projector does not return reaction impulses to the bowl, and Isaac Lab's
direct MPM manager is not a robot dynamics solver. An MPM stress tensor is
also not a net robot force. Robot fine-tuning needs a coupling configuration
and a verified wrench reduction before force observations can be treated as
equivalent. This integration is planned; no policy-training or wrench adapter
is implemented by the bowl example.

Granular material matching should include tensile strength. Pair attraction
range, tensile pressure cap, and deviatoric yield offset describe different
effects. A tensile cap is
``tensile_yield_ratio * yield_pressure``; a nonzero ratio with the default
``1e15 Pa`` pressure cap would imply an enormous tensile strength. Choose
finite pressure and tensile caps together, then calibrate shear offset and
friction using settling, shear, pull-apart, and discharge measurements.
The example exposes both controls so they can be explored independently.

Low-iteration and volume-recovery experiments
----------------------------------------------

The standalone experiment compares identical two-second moving-bowl rollouts
at 6,645 particles and two sparse-grid voxel sizes:

.. code-block:: console

   uv run --extra dev asv/benchmarks/simulation/bench_mpm_low_iterations.py \
     --frames 120 --voxel-sizes 0.08 0.12 --output /tmp/mpm-low-iterations.json

   uv run --extra dev asv/benchmarks/simulation/bench_mpm_low_iterations.py \
     --paired --cases gs2 gs5 gs20 pic_captured2 pic_captured20 \
     --cell-capacity 1024 --output /tmp/mpm-paired.json

   uv run -m newton.examples mpm_bowl --iterations 2 --rheology-solver jacobi gs
   uv run -m newton.examples mpm_bowl --iterations 2 --warmstart-mode none

It fixes ``critical_fraction=1`` and ``tolerance=0`` to exercise the packing law
and execute the requested number of final sweeps. It compares 2/5/10/20 sweeps
to a 100-sweep reference at the same voxel size, cold starts, particle-backed
stress history, existing Jacobi/CR chains, an XPBD stress predictor, and a
bounded packing bias. Compilation and the first ten frames are excluded from
timing. Measurement copies, sparse rebuild status, swept projection, and
prescribed motion are included. This is a short trajectory comparison against
another discretized simulation, rather than a physical ground-truth test.

XPBD stress seed
^^^^^^^^^^^^^^^^

The experimental predictor copies the current state to temporary buffers and
runs two XPBD particle iterations. At each iteration it estimates the pair
constraint force from its displacement correction:

.. math::

   \mathbf f_{ij} \simeq
   \frac{\omega}{(m_i^{-1}+m_j^{-1})\Delta t^2}
   (\delta\mathbf x_t-\delta\mathbf x_n),
   \qquad
   \boldsymbol\sigma_i^{\mathrm{seed}} =
   \frac{1}{2V_i}\sum_{j,I}
   \operatorname{sym}(\mathbf f_{ij}^{I}\otimes\mathbf x_{ij}^{I}).

Here :math:`\omega` is XPBD relaxation, :math:`I` indexes predictor iterations,
and :math:`V_i=8r_i^3` matches MPM's reference particle volume. Compressive
stress uses Newton's positive-pressure convention. The half factor shares
each pair's virial between its two particles. The seed is rasterized through
the same particle quadrature as a normal particle stress warm start, then
projected onto MPM's yield surface by existing preprocessing.

The predictor contributes an initial stress guess only. Its advanced
positions/velocities are discarded; MPM still integrates the real state for
one timestep and retains its elastic/plastic history. Gravity is consequently
not applied twice to the live state. This initial prototype replaces the
temporal stress guess and estimates particle-particle stress only. It does
not predict wall-contact impulses, and its pair law is not a general mapping
for elastic solids, snow, or fluids. If useful beyond the benchmark, the
production implementation should reuse XPBD contact calculations and expose
an explicit stress/impulse seed hook, rather than duplicate its pair law.

Bounded packing bias
^^^^^^^^^^^^^^^^^^^^

The existing critical-fraction law adds a positive void allowance
:math:`\max(\phi_c V_{\mathrm{free}}-V_p,0)`. It clamps an overfilled cell's
allowance to zero; it does not request expansion to recover packing lost in
earlier steps. Collision projection can increase particle concentration
without appearing in the grid's strain solve.

The experimental Baumgarte-style extension also measures that excess:

.. math::

   e = \max(V_p-\phi_c V_{\mathrm{free}},0),
   \qquad
   o = \max(\phi_c V_{\mathrm{free}}-V_p,0)
       -\min(\beta e,c_{\max}V_p).

The prototype uses :math:`\beta=0.2` and :math:`c_{\max}=0.05`. Volumes here are
the existing normalized FEM integrals. The signed offset is mapped with
Newton's ``unilateral_offset_to_strain_rhs`` and inserted into the strain
right-hand side before the coupled solve. Existing postprocessing removes
the offset from physical material strain. The negative component keeps
cohesion enabled; only actual void allowance follows the existing cohesion
disable rule. The strain matrix already includes :math:`\Delta t`, so the
offset is a per-step volume correction; dividing it by timestep again would
give the wrong scaling.

This uses current packing as an error signal and therefore reacts on the
next solve to concentration introduced by projection. It does not measure
only projection-induced error, and it does not increase grid contact DOFs.
It can also react to quadrature noise, ordinary underconvergence, or a changed
grid resolution. Bounded correction prevents asking for complete recovery in
one step, but does not guarantee energy stability. `Baumgarte stabilization
in Box2D <https://box2d.org/posts/2024/02/solver2d/>`_ provides the analogous
position-error velocity bias and discusses its energy/jitter tradeoff; applying
that idea to MPM packing here is an experimental extension.

Before merging a volume-recovery feature, compare this density signal with a
projection-displacement divergence signal, test a timestep-aware recovery
timescale, and evaluate damped or split position correction. Test static
piles, driven walls, narrow gaps, large timesteps, resolution changes, and
compliant/cohesive materials. Check packing, retained mass, contact work,
particle kinetic energy, elastic strain, and wall-force bias together. A
lower packing error alone is insufficient evidence of improved fidelity.

Particle-based contact sampling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The experiment also compares ``collider_basis="pic"`` with the default
``"S2"`` basis. This changes the sampling of rigid contact while retaining
the Q1 velocity grid, P0 strain basis, APIC transfer, constitutive model, and
particle stress history. It does not replace MPM with particle dynamics.

There are two differences relevant to a low iteration count. Contact is
sampled at particles instead of the serendipity grid nodes. Additionally,
the existing point-basis impulse history is stored by particle index and can
be copied into a rebuilt grid's quadrature ordering. The rebuildable S2 path
currently clears its grid impulse guess because the previous grid topology
has been overwritten. The quality experiment changes both effects together;
it does not isolate the contribution of contact sampling from warm starting.

Warp 1.17.0 computes an unbounded point basis's maximum points per cell using
a device-to-host readback. That readback prevents outer frame capture during
grid rebuild. Newton already accepts ``"picN"`` with an integer upper bound;
``collider_basis="pic512"`` bypasses the readback through the existing public
configuration. The captured benchmark uses that form and records actual
maximum cell occupancy on the GPU. If occupancy exceeds the bound, its status
check raises and the rollout must be discarded. A bound without an overflow
check could silently omit contact contributions.

The measured maximum was 49 particles per cell at 8 cm voxels and 131 at
12 cm voxels. These values apply to this rollout, particle spacing, and
material, not arbitrary piles. A coarse grid can increase particles per cell
even while reducing active cell count. A production change should add a
checked point-basis capacity to Newton's existing status mechanism, including
graph replay and per-world tests. The benchmark's subclass is a prototype of
the check. Larger particle counts need separate cost and memory measurements.

An alternative worth testing next is a warm start for S2 contact impulses
keyed by stable world, collider, and grid-node coordinates, with validation
when contact normals or active colliders change. It must gather from a cache
that survives topology rebuild; interpolating through the overwritten grid
is unsafe. That experiment could retain the default contact discretization
and avoid the extra XPBD prediction. It is not implemented here.

Recorded findings and next decisions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The records below were collected on an RTX 4090 with Warp 1.17.0, using the
two-second protocol above. Other GPU jobs were present during several runs.
Absolute timings vary substantially with that load; do not compare throughput
across the serial and paired records. All records, including containment
failures and the different load conditions, are available as
:download:`raw experiment data <mpm_bowl_experiments.json>`.

One lower-load serial run at 8 cm voxels produced:

.. list-table:: Default S2 contact basis, 6,645 particles
   :header-rows: 1
   :widths: 40 20 20 20

   * - Method
     - Median frame [ms]
     - Position RMS vs. 100 sweeps [cm]
     - Grid packing excess [%]
   * - GS 100
     - 14.60
     - 0.00
     - 32.64
   * - GS 20
     - 7.13
     - 2.61
     - 43.08
   * - GS 5
     - 5.74
     - 7.00
     - 52.97
   * - GS 2
     - 5.60
     - 11.23
     - 68.17
   * - XPBD stress seed + GS 2
     - 6.50
     - 8.33
     - 52.32
   * - Packing bias + GS 2
     - 5.30
     - 10.72
     - 25.53

Here packing excess is
``sum(max(particle_volume - free_volume, 0)) / sum(particle_volume)``
on the last substep's assembled grid. It is a quadrature/packing diagnostic,
not a measurement of lost particle mass or conserved physical volume. The
nonzero 100-sweep value illustrates why this metric needs independent physical
validation. Position errors match particle identities at the rollout endpoint.

The XPBD stress seed improved two-sweep position error by about 26%, but five
ordinary GS sweeps were cheaper and more accurate in this run. The seed also
increased projection RMS from 0.207 to 0.268 mm. At 12 cm voxels, the same
comparison gave 11.00 cm without the seed and 8.50 cm with it. Replacing the
temporal stress guess with this raw pair estimate is consequently not a good
production default. Future predictors should consider contact impulses,
temporal/predicted stress blending, and cost against a five-sweep control.

The Jacobi-five-sweep/GS-two-sweep chain did not improve endpoint error in
this scene. CR-two-sweep/GS-two-sweep failed the coarse-grid containment check
with 48 particles outside below the rim and a 13.07 m/s maximum speed. That
records an unsuccessful rollout; it does not by itself distinguish a wall
crossing from particles launched through the open rim. The zero-sweep
diagnostic collapsed the pile and required projection for every particle.
It is not a useful substitute for the fast XPBD environment.

The packing bias reduced its grid error substantially but increased motion:
at 8 cm, RMS speed rose from 0.420 to 0.517 m/s for the two-sweep comparison,
and projection RMS rose to 0.568 mm. The 20-sweep bias also moved the result
farther from the unbiased 100-sweep reference. This prototype should remain
experimental until recovery timescale, damping, energy, and wall-force tests
show a favorable tradeoff.

Particle-based contacts gave a more promising low-sweep approximation:

.. list-table:: Endpoint position RMS against each basis's own 100-sweep reference
   :header-rows: 1
   :widths: 40 30 30

   * - Method
     - 8 cm voxels [cm]
     - 12 cm voxels [cm]
   * - S2, GS 2
     - 11.23
     - 11.00
   * - pic512, GS 2
     - 1.92
     - 1.17
   * - pic512, GS 20
     - 0.38
     - 0.26

Both captured and uncaptured point-basis runs passed containment and had
matching recorded quality metrics. The point-basis 100-sweep result itself
differs from the S2 reference by 3.02/3.23 cm, so the table compares convergence
within each contact discretization. It does not establish greater physical
accuracy for point contacts. Paired timing alternates cases in randomized
order each frame. Under concurrent GPU load, S2 and point-contact two-sweep
frames were approximately equal in cost (about 100 ms in one such run), with
100-sweep frames around 128 ms. Isolated timing on an idle GPU and larger
particle counts is needed before recommending a throughput advantage.

Reducing reserved active cells from 4,096 to 1,024 preserved all recorded
quality metrics for these two voxel sizes and passed the rebuild checks.
It reduces reserved grid work and storage, but these runs do not isolate its
timing gain from changing GPU load. Reserve capacity for the entire intended
motion and resolution schedule, then keep checking overflow during replay.
The example's larger default also supports its finer-grid resize schedule;
the smaller tested bound is not a general replacement.

The resulting implementation order is: propose runtime voxel changes, swept
projection, and the exact iteration-budget fix as separate Newton changes;
keep the XPBD/MPM demonstration available for pretraining; test checked point
contact capacity and an S2 impulse cache next; refine volume recovery only
after measuring energy and force effects. Isaac Lab should own the shared
observation/wrench adapter and the curriculum that calls those numerical
operations. Automatic particle splitting/merging remains a separate design
task rather than a radius update.
