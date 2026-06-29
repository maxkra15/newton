# Fluid multi-world isolation design

## Status

Proposed implementation design for Newton branch `eric-heiden/flex-fluid` at
`e3848572`, with a companion change to Warp's `HashGrid` implementation.

## Objective

Make particle and fluid simulation in colocated Newton worlds numerically
isolated and scalable for vectorized reinforcement-learning workloads, while
preserving the current single-world behavior and CUDA-graph execution model.
Make the ViewerGL fluid path apply Newton world offsets so the isolated worlds
can also be inspected visually.

## Confirmed problem

Newton represents environment membership with `Model.particle_world` and keeps
replicated environments physically colocated. `SolverXPBD` and `SolverSPH`
currently build one `wp.HashGrid` over every particle and their neighbor kernels
do not test the world IDs. Particles from separate worlds therefore contribute
to each other's density, position corrections, viscosity, vorticity, contacts,
render smoothing, and diffuse-particle behavior.

A minimal two-world reproduction with one coincident XPBD fluid particle per
world separates the particles by about 0.0225 m after one iteration, while the
one-world baseline remains stationary. This demonstrates state corruption, not
only excess query work.

The existing `wp.HashGrid` API cannot prune a query by group. A user-space world
test restores correctness, but a query in `W` colocated environments still
visits approximately `W` copies of each neighborhood.

## Scope and decomposition

The work is split into separately reviewable changes:

1. **Newton correctness:** world-filter every particle-neighbor path and add
   multi-world regression tests. This works with released Warp and is the
   numerical oracle for later optimization.
2. **Warp scalability:** add optional grouping to `wp.HashGrid`, retaining the
   exact legacy path when groups are absent.
3. **Newton acceleration:** use grouped builds and queries when a model can use
   the fast isolation contract, with the correctness filter retained
   defensively.
4. **Viewer integration:** propagate particle world IDs to ViewerGL fluid and
   diffuse batches, apply display offsets, and honor visible-world filtering.
5. **Particle reordering:** prevent global Morton sorting from invalidating
   world ranges; initially disable it for multi-world models, then add a
   segmented per-world reorder as a separate optimization.

GPU-only diffuse compaction and depth sorting are useful renderer optimizations,
but are not required for environment isolation and are outside this change.

## Environment semantics

Newton's existing compatibility rule is:

```text
same world OR either entity belongs to global world -1
```

The correctness-only path will preserve that rule. The grouped fast path will
be enabled only when every particle belongs to a non-negative local world. If a
multi-world model contains global particles, Newton will retain the ungrouped
grid and compatibility filter. This avoids changing established behavior while
keeping the common replicated-RL layout fast.

Fluid material settings remain solver-wide. This work supports homogeneous
replicas and does not introduce per-world rest density, smoothing length, rest
distance, or material coefficients.

## Warp API

Extend `HashGrid` without changing existing call sites:

```python
grid.reserve(point_count, grouped=True)
grid.build(points, radius, groups=groups)

# Existing all-groups behavior.
query = wp.hash_grid_query(grid.id, point, radius)

# New exact group query.
query = wp.hash_grid_query(grid.id, point, radius, group)
```

`groups` must be a contiguous `wp.array[wp.int32]` on the same device as the
points, with the same number of entries. `groups=None` uses the current build
and query implementation unchanged. A group ID may be any `int32` value. A
query for a group that is not present returns no points.

`reserve(..., grouped=True)` preallocates all grouped-build storage and the
64-bit radix-sort temporary capacity needed by later builds. This makes the
first grouped build safe inside CUDA graph capture. Calling grouped `build()`
without grouped capacity may allocate in the same way the current unreserved
build does; Newton will reserve before capture.

## Warp native representation

Keep the existing dense spatial `cell_starts` and `cell_ends` arrays. For a
grouped build, create a 64-bit key per point with spatial cell as the major key
and a bit-preserving normalized `int32` group as the minor key. Radix-sort these
keys with original point IDs.

Because all entries for one `(cell, group)` are contiguous:

- the existing three-argument query uses the full cell range and therefore
  preserves all-groups behavior;
- the four-argument query performs lower/upper-bound searches for the requested
  group inside each visited cell range;
- `hash_grid_point_id()` continues to return original point IDs in a
  spatially coherent order;
- memory remains `O(points + cells)`, rather than
  `O(groups * cells)`.

The ungrouped descriptor and 32-bit radix-sort path remain unchanged. Grouped
storage is allocated only after grouped reserve/build is requested. Host and
CUDA implementations must have matching semantics for float16, float32, and
float64 grids.

The implementation will add separate grouped native entry points rather than
silently changing the existing C API signatures. The Python wrapper selects the
appropriate reserve/update function and holds a reference to the group array
for the duration of the call.

## Newton solver integration

### Correctness filter

Add `particle_world` to all relevant XPBD and SPH kernels. Reject an
incompatible candidate before distance calculations, density accumulation,
neighbor caps, covariance accumulation, or atomic updates. The affected paths
include:

- XPBD particle-particle contacts;
- XPBD fluid lambda and delta passes;
- XPBD viscosity and vorticity passes;
- SPH density, pressure, force, PBF, and velocity-smoothing passes;
- render smoothing and anisotropy;
- diffuse advection and spawning.

Particle-shape collision already applies the correct compatibility rule and
will not be changed except where tests expose a mismatch.

### Grouped fast path

At solver initialization, determine whether all particles have non-negative
world IDs and the model has more than one world. If so:

- reserve the grid in grouped mode;
- pass `model.particle_world` to every build;
- construct neighbor queries with the source particle's world ID.

If the model is single-world or contains global particles, use the legacy grid
query and the compatibility filter. The same kernel may select the grouped or
ungrouped query with a scalar launch argument, provided Warp generates the same
query type for both overloads; otherwise define two small query-kernel variants
around shared functions.

The filter remains active in grouped kernels as a defensive invariant and test
oracle. It should have negligible cost after structural pruning.

### Reordering

`SolverXPBD.reorder_particles()` currently globally Morton-sorts positions and
permutes `particle_world`, which interleaves worlds while leaving
`particle_world_start` stale. Until segmented reordering is implemented, the
method will be a documented no-op when `model.world_count > 1`.

A later optimization may sort independently within each immutable world range,
including explicit front/tail segments for global particles. It must update or
preserve every per-particle array and keep `particle_world_start` valid.

## ViewerGL integration

Add optional `worlds` arguments to `ViewerBase.log_fluid()`,
`ViewerGL.log_fluid()`, and `log_fluid_diffuse()`. When supplied, ViewerGL will:

- add the viewer's display-only world offset while packing fluid vertices;
- collapse or omit particles from hidden worlds;
- apply the same offset/filter logic to diffuse positions;
- leave global particles unshifted.

The high-level particle logger and XPBD/SPH examples will pass
`model.particle_world`; diffuse calls will pass the solver's diffuse-world
array. The CUDA-to-OpenGL surface path stays device-native. The existing
diffuse CPU sorting path may apply offsets after its host transfer until the
separate GPU sorting optimization is implemented.

## Error handling and compatibility

- Existing ungrouped Warp and Newton use remains source-compatible.
- Group arrays with the wrong dtype, device, contiguity, rank, or length raise
  descriptive Python errors before native code runs.
- A grouped query on an ungrouped build returns an empty query in release builds
  and asserts in native debug builds; documentation states that callers must
  build with groups first.
- Empty point sets and absent groups are valid.
- Group membership may change on every grouped build.
- No public Newton symbol is removed or renamed.

## Tests

### Warp

Add CPU/CUDA tests for:

- exact same-group results with physically overlapping groups;
- preservation of the all-groups query on a grouped build;
- absent, sparse, negative, and extreme `int32` group IDs;
- empty/single-point grids and hash-wrapped cells;
- float16/float32/float64 grids;
- validation failures;
- repeated group-membership changes;
- multi-stream execution;
- grouped reserve/build/query inside CUDA graph capture;
- unchanged legacy query results.

### Newton

Add tests that:

- reproduce the coincident two-world XPBD failure and compare with a one-world
  baseline;
- compare every slice of a small colocated multi-world fluid against
  independently simulated models;
- verify mixed fluid/solid particle contacts do not cross worlds;
- preserve local/global compatibility on the fallback path;
- isolate viscosity, vorticity, rendering covariance, and diffuse behavior;
- verify multi-world reorder is safe and leaves world metadata valid;
- verify ViewerGL packing applies offsets and visible-world filtering;
- capture and replay the grouped solver step on CUDA.

Tests follow the repository's `unittest` convention. The correctness test is
written first and observed failing before production changes.

## Performance validation

Benchmark 1, 8, 64, and 256 colocated worlds with fixed particles per world.
Measure grid build and neighbor kernels separately, and record candidate visits,
wall time, peak memory, and capture-time allocations. Compare:

1. ungrouped with user-space filtering;
2. grouped HashGrid;
3. grouped particle BVH as an experimental reference.

Numerical output must match independent-world baselines. Candidate visits and
query time for the grouped grid should scale approximately linearly with world
count. No timing threshold is placed in unit tests.

## Delivery structure

The work should be reviewed as at least two changesets:

1. Warp grouped HashGrid API, native implementation, documentation, and tests.
2. Newton correctness filters, grouped integration, reordering guard, ViewerGL
   world handling, examples, changelog, and tests.

Newton's grouped integration can be developed against an editable local Warp
build, but its final dependency requirement must target a released Warp version
that contains the new API. The correctness-only Newton commit remains usable
with Warp 1.14 and can be reviewed or landed independently.
