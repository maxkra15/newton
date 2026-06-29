# Multi-World Sparse Implicit MPM for RL

**Status:** Approved for implementation on 2026-06-29

## Summary

This project combines Newton's isolated multi-world implicit MPM solver with
Warp's rebuildable NanoVDB topology so that a sparse implicit MPM step can be
recorded inside an outer CUDA graph. It then consumes that paired Newton/Warp
stack from IsaacLab and adapts the Franka Pour task to use one isolated MPM
environment per RL world.

The work is split across three repositories. Warp owns capture-safe dynamic
topology and environment partitions. Newton owns physical isolation, per-world
solver state, capacity policy, and MPM capture semantics. IsaacLab owns RL
replication, asynchronous reset orchestration, dependency selection, and the
Franka Pour task. Each repository remains independently testable and keeps its
own branch and commit history.

## Goals

1. Preserve strict Newton-world isolation for MPM particles, grids, colliders,
   rheology solves, warm-start data, and scratch storage.
2. Make capacity-bounded sparse implicit MPM steps safe to record in an outer
   CUDA graph and replay while particle positions and active grid topology
   change.
3. Retain fixed-grid outer capture and the legacy shared-grid mode without
   changing their public behavior.
4. Give overflow a deterministic, inspectable failure path outside capture;
   capacity exhaustion must never be accepted as a valid simulation result.
5. Make the IsaacLab Franka Pour task reproducible from a clean checkout and
   configure it through public Newton/IsaacLab APIs rather than private solver
   internals.
6. Demonstrate overlapping-world isolation, asynchronous reset safety,
   eager-versus-captured equivalence, repeated replay, and useful total-step
   performance measurements.

## Non-goals

- Capturing the entire policy, observation, reward, and host-driven reset loop
  in one CUDA graph.
- Supporting variable particle counts after solver initialization.
- Growing sparse capacities during graph replay.
- Making dense grids graph-capturable; dense bounds remain host-derived.
- Replacing the Franka Pour reward, curriculum, or robot-control strategy.
- Refactoring unrelated IsaacLab Newton tasks or generalizing the coupled
  solver beyond what the Pour integration requires.
- Publishing branches, pushing commits, or opening pull requests as part of
  this implementation.

## Repository and Branch Topology

The implementation uses isolated worktrees and never modifies the existing
dirty checkouts.

| Repository | Integration branch | Base | Inputs |
|---|---|---|---|
| Warp | `max/warp-max` | `origin/main` at `b8596fd51` | `max/capture-environment-partition` at `675c15b7f`; `max/s2-rebuildable-edge-grid` at `2eb92885e` |
| Newton | `max/newton-max` | `origin/main` at `49ca73b9` | `max/coupled-public-apis` at `9d70911e`; `max/implicit-mpm-cuda-graph` at `d61ea124`; `sparse-rebuildable` at `a8601c24` |
| IsaacLab | `max/franka-pour-multiworld-mpm` | committed `max/newton-coupling-manager` at `80d2b8b42b` | selected uncommitted Franka Pour task files from the dirty coupling checkout, plus the final immutable Newton/Warp revisions |

`max/newton-max` is an integration branch, not a proposed squashed upstream PR
boundary. The coupled solver history is an existing dependency. The MPM work
must remain reviewable as a coherent layer on top of that dependency so it can
later be split or rebased without mixing IsaacLab task code into Newton.

## Architecture

### Warp: capture-safe topology and partitions

Warp combines two independent capabilities:

1. A rebuildable `wp.Volume`/`fem.Nanogrid` reserves fixed voxel, node, leaf,
   and edge capacities before capture and rebuilds active topology in place.
2. FEM environment partitions and temporary storage retain stable allocations
   and object lifetimes throughout capture and replay.

The merged implementation must preserve one allocation contract: every buffer
whose address is used by a captured kernel is allocated before capture and
remains alive until the captured graph is destroyed. A topology rebuild may
change active counts and index contents, but not storage addresses or declared
capacities. Rebuild status is written to a device-side bit mask. Host-side
inspection is allowed after replay, never from within capture.

Automatically generated environment packing offsets are recomputed on-device
from the current masked point bounds during each rebuild. The offset array and
its bounds/scan scratch retain fixed addresses, while their values may change.
This prevents independently moving environments from aliasing the same packed
NanoVDB coordinates. Explicit caller-provided offsets remain fixed and make
the caller responsible for non-overlapping spatial bounds. Environment count,
guard width, alignment, and buffer capacities remain structural invariants.

The environment-partition path and rebuildable-edge path overlap in Warp
context allocation and FEM tests. Conflict resolution must keep both lifetime
retention and all S2 edge-topology capacity checks. The native Warp library is
rebuilt from the merged source before any Newton test; a stale `warp.so` is not
a valid test artifact.

### Newton: isolated rebuildable sparse grids

Newton retains `SolverImplicitMPM.Config.separate_worlds=True` as the default
for multi-world models. Particles must be contiguous by world and every local
particle must have a valid `particle_world`. Global particles remain supported
only in a single-world model or with `separate_worlds=False`.

In isolated sparse mode, the solver constructs one logical FEM environment per
Newton world. A world's particle range determines only that world's NanoVDB
topology. Particle-to-grid transfer, active-cell compression, collider queries,
rheology, warm-start fields, and grid-to-particle transfer all carry the same
environment identity. Global static colliders may be deliberately visible to
all worlds; dynamic body-backed colliders must belong to exactly one world.

`max_active_cell_count` remains the total fixed capacity of the batched FEM
geometry partition. Multi-world callers derive it from a per-world estimate,
normally `cells_per_world * world_count`. Keeping a shared reserve matches
Warp's environment-partition contract, avoids artificial per-world waste, and
preserves one batched solve. The packed sparse topology stores only each
world's active local voxels plus guard regions; it does not cover the physical
distance between replicated scenes. Documentation must distinguish the
caller's per-world sizing estimate from Newton's total capacity.

A sparse step is eligible for outer capture only when all of these invariants
hold:

- CUDA and conditional graphs are supported;
- `grid_type == "sparse"`;
- `max_active_cell_count > 0`;
- `grid_padding == 0`;
- Warp exposes rebuildable NanoVDB support for every required basis topology,
  including S2 edges when the selected strain or collider basis needs them;
- particle count, world count, world ordering, solver configuration, and
  reserved capacities remain fixed across replay;
- automatic packed-environment offsets may change, but their array shape,
  guard width, and alignment remain fixed;
- the nonlinear/linear solver combination has no host convergence checks in
  the captured path.

The existing fixed-grid Jacobi capture path remains supported. Unsupported
sparse configurations run eagerly and retain the current allocating behavior;
they do not pretend to be capture-safe.

The batched grid records rebuild failures into persistent device status.
Newton accumulates the status across substeps and exposes a public post-replay
check that reports the exhausted capacity class. Overflow must not trigger
host synchronization while recording or replaying the graph. The next safe
host boundary raises a descriptive exception recommending a larger total
capacity or a smaller active domain.

### Newton: coupled solver and graph ownership

Franka Pour needs MJWarp for articulated rigid dynamics and implicit MPM for
the material. The clean coupled-solver lineage is merged before the MPM
features. Its graph-capability hooks become the single source of truth for
whether every subsolver can participate in capture.

Only one layer owns a CUDA graph. When IsaacLab requests internal physics
capture, `NewtonManager` captures the coupled physics step and replays it.
When an application records a larger outer graph, the manager leaves the
solver eager-but-capture-safe and does not launch or create a nested graph.
The sparse MPM entry reports capture support only when the invariants above
are satisfied. Coupled capture support is the conjunction of all entry
capabilities.

### IsaacLab: RL world mapping and Franka Pour

IsaacLab already creates one Newton builder world per replicated RL
environment. `MPMObject` already records equal-size particle slices and
presents runtime data as `[num_envs, particles_per_env, ...]`. The task must
use that mapping and must not hand-pack a shared simulation grid.

The Franka Pour task is copied from the dirty coupling checkout into a clean
branch together with its registration, changelog fragment, and focused tests.
Scratch scripts, videos, absolute paths, `.pth` files, and unrelated dirty
changes are excluded.

The task keeps its proxy-coupled model:

- MJWarp owns the Franka, dynamic source cup, and rigid receiver geometry.
- Implicit MPM owns the material and particle-collision copies.
- Per-world proxy data transfers the source-cup pose from MJWarp to MPM.

The MPM configuration exposes `separate_worlds=True`, `grid_type="sparse"`,
and a resolved per-environment cell estimate. Existing `cells_per_env`, minimum,
and explicit total override settings are reduced to one deterministic function.
Without an explicit total override, the task passes
`max(cells_per_env, minimum_cells_per_env) * num_envs` as Newton's total
`max_active_cell_count`.

Reset remains outside the captured physics graph. A public manager/asset reset
path accepts a fixed-size environment mask and restores all selected MPM
history: position, velocity, elastic strain, deformation/velocity gradients,
stress, plastic volume, and particle-backed warm-start fields. The next sparse
rebuild starts with cleared global status. The reset must leave non-selected
environments bitwise unchanged at the API boundary. The task no longer reaches
through `NewtonManager._solver._entries`.

The stable Torch-based `DirectRLEnv` loop remains the first supported RL path.
The experimental all-Warp RL graph is only changed if needed to forward the
existing `env_mask` to deformable assets; its policy/reward loop is outside
this project's capture claim.

## Dependency Reproducibility

IsaacLab metadata must consume a mutually compatible Newton/Warp pair. All
duplicated dependency declarations, including wheel-builder metadata, are
updated together. Local development may point to isolated source worktrees,
but the test launcher must print and assert the imported `newton.__file__`,
`warp.__file__`, Warp version, and native library location. The documented
reproducible form is an immutable Newton Git revision plus a Warp wheel built
from the final `max/warp-max` revision. No absolute workstation path is
committed.

## Testing Strategy

All behavior changes follow red-green-refactor. Existing feature-branch tests
are retained, then integration tests are added for interactions the separate
branches could not cover.

### Warp

- Capture and replay an environment partition while active membership changes
  within fixed capacity.
- Rebuild a multi-environment NanoVDB with changing voxel and S2 edge topology
  under capture.
- Move one environment beyond its initial packed extent and verify automatic
  offsets change without packed-coordinate aliasing or pointer replacement.
- Exercise exact-capacity success and one-over-capacity status for voxels,
  nodes, leaves, and edges.
- Verify capture-owned temporary objects remain valid through repeated replay.

### Newton

- Two worlds occupy identical local coordinates but have different particle
  motion; eager and captured sparse results match within solver tolerance.
- Moving a collider in one world cannot affect particles or grid fields in
  another world.
- Each world may change sparse topology independently over repeated graph
  replay without reallocating capture-owned storage.
- Empty worlds, inactive particles, exact capacity, and global overflow under
  uneven per-world activity are covered.
- Global sparse-capacity overflow is detected after replay and cannot be
  mistaken for a successful captured step.
- Fixed-grid capture and `separate_worlds=False` regressions continue to pass.
- Coupled graph capability is false if any entry is not capture-safe and true
  for the supported MJWarp plus sparse-MPM configuration.

### IsaacLab and Franka Pour

- Clean-checkout configuration and Gym registration tests.
- Two or more overlapping replicated environments produce correct world IDs,
  particle slices, collider ownership, and derived total capacity.
- Resetting one environment restores all of its MPM history while leaving a
  neighboring environment unchanged.
- Repeated captured physics replay matches eager physics for a short,
  deterministic rollout.
- A moving source cup affects only its own material world.
- A reduced headless smoke test initializes the coupled task, steps it, resets
  a subset, and steps again without private solver access.

GPU-dependent tests use explicit capability skips when CUDA, conditional graph
support, or rebuildable native Warp support is unavailable. A skip must name
the missing capability and cannot mask an import or configuration failure.

## Benchmarking

Benchmarks report total physics-step wall time after warm-up, not a selected
kernel time. They compare:

1. legacy shared sparse allocation;
2. isolated eager rebuildable sparse grids;
3. isolated captured rebuildable sparse grids;
4. isolated captured fixed grids as a reference.

For each mode, report environment count, particles per environment, per-world
sizing estimate, total reserved capacity, peak GPU memory, median step time,
tail latency, and steps per second. Scaling points use the same physical scene
and particle count per world. Setup, graph recording, and first-time
compilation are reported separately from steady-state replay. Claims are
limited to the measured hardware and configuration; an 8x kernel reduction is
never described as an 8x end-to-end speedup unless total-step measurements show
it.

## Error Handling and Diagnostics

- Invalid or non-contiguous particle world IDs fail during solver creation.
- Dynamic global colliders fail during isolated solver creation.
- Unsupported capture configurations expose a precise capability reason and
  remain usable eagerly.
- Capacity exhaustion is accumulated on device and raised at the next explicit
  host status check with the exhausted capacity class.
- Dependency mismatch diagnostics show imported source and native-library
  locations before tests run.
- Selective reset validates mask shape and device and rejects particle-count or
  world-count changes after initialization.

## Acceptance Criteria

The implementation is complete when:

1. Clean isolated Warp, Newton, and IsaacLab branches contain only scoped
   changes and preserve their original dirty checkouts.
2. The merged Warp native library builds and its focused partition/NanoVDB
   capture tests pass.
3. Newton's focused implicit-MPM and coupled-capability suites pass, including
   a real multi-world rebuildable sparse outer-capture regression.
4. The clean Franka Pour branch installs against the exact paired revisions,
   its focused tests pass, and its headless multi-environment smoke test either
   passes on available hardware or reports a specific unavailable runtime
   capability.
5. Benchmark output distinguishes total-step throughput, graph setup cost, and
   memory use, with no unsupported performance claim.
6. No branch is pushed and no pull request is created.
