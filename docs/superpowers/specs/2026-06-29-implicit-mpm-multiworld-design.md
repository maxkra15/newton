# Implicit MPM Multi-World Isolation Design

## Summary

Newton's implicit MPM solver currently assembles every particle into one spatial
FEM topology. Particles from colocated Newton worlds therefore share grid nodes,
mass, momentum, material averages, strain, stress, and collider responses.

This change makes multi-world isolation the default while retaining one
SolverImplicitMPM instance, monolithic particle arrays, and batched GPU work.
It uses Warp's native multi-environment FEM geometry to produce topologically
independent worlds at coincident physical coordinates.

The implementation is confined to Newton. Newton main at commit
7b630e096060cce6bbbc98545f2634945cbb55b3 locks
warp-lang 1.15.0.dev20260612, which already contains Warp's GH-1407
multi-environment FEM implementation from commit
6d0d21c2f41d3659b194e03d3f2cc334320cdbb2. Neither Warp source nor Newton's
Warp dependency changes.

## Goals

- Make independent MPM physics the default for models with multiple Newton
  worlds.
- Keep worlds physically colocated while separating all FEM cells and degrees
  of freedom.
- Support sparse, dense, and fixed MPM grids.
- Preserve PIC and GIMP integration, all supported rheology solvers, warm
  starts, collider coupling, direct projection, and grain rendering.
- Keep one solver object and one monolithic block-diagonal system.
- Ensure world-local colliders affect only their world and global static or
  kinematic colliders affect every world.
- Add deterministic regression coverage against independent one-world
  references.
- Add an idiomatic quick ASV benchmark comparing one multi-world solver with
  multiple independent solver instances.

## Non-Goals

- Changing Warp.
- Making NanoVDB construction from particles CUDA-graph-capturable.
- Implementing in-place sparse matrix compression.
- Sharing dynamic particles or dynamic rigid bodies across otherwise
  independent worlds.
- Supporting selective per-world reset of persistent MPM warm-start state.
- Adding a new rendered example; tests, documentation, and the benchmark are
  sufficient for this change.

## Public API and Compatibility

SolverImplicitMPM.Config gains:

- separate_worlds: bool = True

When separate_worlds is true and model.world_count is greater than one, the
solver uses isolated multi-environment FEM topology. When it is false, the
existing shared-grid behavior remains available as an explicit compatibility
escape hatch.

Single-world behavior remains unchanged. In particular, particles assigned to
Newton's global world, particle_world equal to -1, are mapped implicitly to the
single FEM environment.

An isolated multi-world solver rejects any global particle with a clear
ValueError. A single dynamic particle state cannot participate independently
in several FEM environments without duplication and an undefined reduction of
the resulting updates.

SolverImplicitMPM.setup_collider gains an optional keyword-only
collider_world_ids argument. Each entry assigns the corresponding custom
collider to a Newton world. A custom mesh without a world ID remains global.
Body-backed colliders infer their world from body_world. Supplied IDs must be
-1 or in the range from 0 through model.world_count minus one.

Global static and kinematic colliders are supported and apply to every
environment. An isolated multi-world solver rejects a global collider backed by
a body with nonzero mass because its single rigid state and inertia would
couple the worlds through accumulated impulses.

Existing return contracts for collect_collider_impulses and
collider_body_index remain unchanged.

## Solver World Layout

At construction, SolverImplicitMPM resolves an internal isolation flag:

- false for single-world models;
- Config.separate_worlds for multi-world models.

For isolated models, model.particle_world is the particle FEM environment
array. ModelBuilder's validated particle_world_start array supplies contiguous
per-world particle ranges, including empty ranges. Construction validates that
there are no global particles before allocating FEM state.

No per-world SolverImplicitMPM objects or per-world state copies are created.
All material and state arrays remain indexed by the original global particle
index, and PicQuadrature sample qp_index continues to address that index.

## Grid Construction

### Dense and fixed grids

The existing union particle AABB determines one common physical domain.
Grid3D is constructed with env_count equal to model.world_count when isolation
is active. Warp duplicates its topology, not its physical coordinates, for
each environment.

Dynamic dense mode rebuilds the grid as it does today. Fixed mode creates the
multi-environment grid once and reuses it.

Dense active-cell marking includes the particle environment in every FEM
lookup. The explicit geometry partition therefore activates only the matching
environment's cell even when worlds overlap exactly.

### Sparse grids

Each world receives a one-dimensional array of local integer voxel
coordinates derived from its particle range. Configured grid padding is
expanded independently within each environment.

The arrays are passed together to Nanogrid.from_environment_voxels with Warp's
default generated packing offsets and guard cells. Warp builds one NanoVDB
volume, records per-cell environment metadata, and hides packed translations
from physical FEM coordinates.

Empty worlds are represented by empty voxel arrays. Sparse multi-environment
construction remains CUDA-only and outside graph capture, consistent with the
current sparse solver path and Warp's API.

## Particle Binning and Integration

PIC mode constructs PicQuadrature directly from world-space particle
positions, particle measures, and model.particle_world as env_indices.
use_domain_element_indices remains enabled to preserve current memory behavior.
Newton's redundant manual PIC location kernel is removed from this path.

GIMP supplies explicit cell-index, coordinate, and fraction arrays, so Warp
does not accept env_indices at PicQuadrature construction. Newton's custom GIMP
corner lookups instead pass particle_world for every lookup.

Dense active-cell lookup receives the same environment ID. No environment is
inferred from physical position.

Grain rendering repeats each source particle's environment for its grain
samples and passes those environment IDs to its PicQuadrature. This prevents
render-grain interpolation from becoming ambiguous on multi-environment grids.

## Function Spaces and Linear Algebra

Velocity, collider, and strain space partitions set environment_first when
isolation is active and retain with_halo false. Their fields remain single
arrays, but each world's entries are contiguous and topologically disconnected.

FEM assembly then produces block-diagonal mass, strain, compliance, collision,
and rigidity operators without adding world checks to every physical
integrand.

The strain partition's environment node offsets are multiplied by six to
produce scalar coefficient offsets for the vec6 stress unknown. These offsets
are passed to both the Warp LinearOperator and its preconditioner for CG, CR,
and GMRES. Krylov coefficients are therefore computed independently per world.

Newton's nonlinear GS and Jacobi paths already update constraints locally once
the topology is disconnected. Their residual reduction becomes
environment-aware and uses the worst environment for the shared termination
condition. This matches Warp's batched-solver convention: environments share
an outer iteration count but never share physical coefficients or updates.

Warm-start fields remain one multi-environment field. Warp's environment-aware
NonconformingField lookup transfers values only within the same environment
when a sparse grid changes.

## Collider Isolation

Collider setup records an environment for every packed collider:

- body-backed collider: model.body_world for that body;
- static shapes attached to body -1: model.shape_world;
- custom mesh without an explicit ID: global world -1;
- custom mesh with collider_world_ids: the supplied world.

Static shapes attached to body -1 are grouped by shape_world before mesh
merging. This prevents one mesh from mixing global and world-local faces.

Collider arrays retain stable collider IDs. Separate indirection arrays group
those IDs into global and per-world ranges so a node does not scan colliders
from every other world. Per-collider face offsets preserve material lookup
when queries traverse the indirection.

The rasterization kernel derives each collider node's environment from the
environment-first space partition offsets. collision_sdf searches only global
colliders and colliders belonging to that environment. project_outside passes
the particle environment directly and applies identical filtering.

Consequently, rigidity matrices contain only the matching world's dynamic body
blocks. Global colliders are static or kinematic and cannot create a shared
dynamic impulse path.

## Errors and Validation

Construction raises ValueError for:

- global particles in an isolated multi-world model;
- collider world IDs outside the valid range;
- world-local collider metadata that cannot be aligned with the solver model;
- global dynamic colliders in isolated mode.

The errors identify the offending feature and explain the supported
alternative: replicate particles or dynamic bodies into each world, use a
static or kinematic global collider, or disable separate_worlds explicitly for
legacy coupled behavior.

Changing world topology, particle membership, or collider world membership
after solver construction requires reconstructing the solver. Existing
material-only notify_model_changed behavior is unchanged.

## Tests

Tests remain in newton/tests/test_implicit_mpm.py and use unittest conventions.

### Reference-equivalence test

Two particle blocks occupy identical physical coordinates in different worlds
and receive distinct gravity or initial velocity. The multi-world model is
stepped for several frames and each particle_world_start slice is compared with
a freshly constructed one-world reference using NumPy assert_allclose.

The test uses deterministic zero-jitter particles, PIC transfer, a fixed
iteration budget, and no collider. It covers dense mode on CPU and CUDA and
sparse mode on CUDA. A focused GIMP variant exercises explicit
environment-aware corner lookups. Multiple steps exercise warm-start transfer.

### Isolation and policy tests

- Replicating one setup into multiple coincident worlds does not change any
  per-world result.
- A global particle in an isolated multi-world model raises the documented
  error, while existing single-world global-particle cases continue to step.
- Two coincident world-local colliders with different motion affect only their
  own particles.
- A global static ground affects every world.
- A global dynamic collider raises the documented error.
- project_outside uses the same world filtering.
- At least one linear rheology solver exercises batched coefficient offsets.
- Grain rendering can update samples on a multi-environment velocity field.

CUDA-only cases use Newton's CUDA test-device registration and skip normally
when no CUDA device is available. Tests do not synchronize immediately before
array numpy conversion because the conversion already synchronizes.

## Benchmark

Add asv/benchmarks/simulation/bench_implicit_mpm.py with a
FastImplicitMPMMultiworld benchmark discoverable by Newton's Fast GPU CI
filter.

The benchmark compares two correct layouts for world counts 1, 8, and 32:

- multiworld: one isolated model, solver, and state pair;
- independent: the same number of one-world models, solvers, and state pairs
  stepped sequentially.

Both layouts use coincident deterministic particle blocks, identical material
and solver settings, the same total number of world-steps, and tolerance zero
for a fixed iteration count. Setup performs an untimed warmup and
synchronization. Timed work performs several steps and one synchronization at
the tail, never one synchronization per world.

The benchmark exposes raw time and milliseconds per world-step. It makes no
timing assertion in tests. The module follows existing Newton ASV conventions,
uses skip_benchmark_if when CUDA is unavailable, and is directly runnable with
newton.utils.run_benchmark. GPU peak-memory claims are excluded because ASV
peakmem reports host RSS rather than CUDA allocation peaks.

## Documentation and Changelog

The public solver and Config docstrings describe default isolation, the legacy
escape hatch, global-particle rejection, and collider rules. The worlds concept
documentation notes that implicit MPM now supports physically coincident
independent worlds and that sparse mode is preferable for heterogeneous or
physically separated bounds.

CHANGELOG.md receives an Added entry describing independent multi-world
SolverImplicitMPM grids, solves, and collider filtering.

## Acceptance Criteria

- Existing single-world implicit MPM tests pass unchanged.
- Isolated coincident worlds match independent one-world references within the
  documented numerical tolerance.
- Changing one world's particles, gravity, material, or colliders does not
  alter another world's assembled physics.
- Sparse, dense, and fixed paths retain their existing supported devices and
  behavior.
- Global-particle and global-dynamic-collider policies fail early with clear
  errors.
- The benchmark runs through the standard Newton benchmark harness and compares
  batched multi-world execution against independent solvers.
- Pre-commit checks pass for every modified file.
- Warp is not modified, Newton's Warp dependency is not changed, and nothing is
  pushed.
