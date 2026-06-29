# Coupled-Solver Public Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add public, vectorized-integration-safe lifecycle, contact, diagnostics, and VBD control APIs to Newton's experimental coupled solvers.

**Architecture:** Keep `SolverCoupled` responsible for entry ownership, local collision selection, state mapping, and lifecycle aggregation. Proxy and ADMM retain algorithm-specific buffers but expose immutable descriptors and honor the existing world-mask contract. All additions remain under the existing experimental namespace, with `SolverBase` receiving only generic CUDA graph lifecycle hooks.

**Tech Stack:** Python 3.12, Warp device kernels and arrays, Newton `ModelView`/`Contacts`/`SolverBase`, `unittest`, Sphinx, uv, pre-commit.

---

## File structure

- `newton/_src/solvers/solver.py`: generic graph capability and preparation hooks.
- `newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py`: dynamic graph-capability override.
- `newton/_src/solvers/coupled/contact_stream.py`: public immutable stream descriptors and on-demand count diagnostics.
- `newton/_src/solvers/coupled/reset_utils.py`: reusable world-mask validation and masked array kernels.
- `newton/_src/solvers/coupled/solver_coupled.py`: entry-local collision, stream enumeration, graph aggregation, and base masked reset.
- `newton/_src/solvers/coupled/solver_coupled_proxy.py`: masked Proxy history, per-world Aitken state, streams, and feedback diagnostics.
- `newton/_src/solvers/coupled/solver_coupled_admm.py`: masked ADMM state and on-demand diagnostics.
- `newton/_src/solvers/coupled/__init__.py`: public experimental exports.
- `newton/_src/solvers/vbd/solver_vbd.py`: batched joint constraint-mode API.
- `newton/tests/test_coupled_solver.py`: Base/Proxy, collision, stream, graph, and view-isolation regressions.
- `newton/tests/test_admm_coupled_solver.py`: ADMM reset and diagnostics regressions.
- `newton/tests/test_solver_vbd.py`: VBD batch API regressions.
- `docs/concepts/coupling.rst`, `CHANGELOG.md`: public contract and release notes.

### Task 1: Generic CUDA graph lifecycle

**Files:**
- Modify: `newton/_src/solvers/solver.py`
- Modify: `newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py`
- Modify: `newton/_src/solvers/coupled/solver_coupled.py`
- Modify: `newton/_src/solvers/vbd/solver_vbd.py`
- Test: `newton/tests/test_coupled_solver.py`
- Test: `newton/tests/test_solver_vbd.py`

- [ ] **Step 1: Write graph-capability tests**

Add a recording solver with an overridable capability and preparation counter,
then add `TestSolverCoupledGraphCapture` tests equivalent to:

```python
class _GraphRecordingSolver(_StepCountingCopySolver):
    def __init__(self, model, supported=True):
        super().__init__(model)
        self.supported = supported
        self.prepare_calls = 0

    @property
    def supports_cuda_graph_capture(self):
        return self.supported

    def prepare_cuda_graph_capture(self, contacts=None):
        self.prepare_calls += 1


def test_nested_graph_capability_is_aggregated(self):
    model, entries, solvers = self._two_particle_entries((True, False))
    coupled = SolverCoupled(model, entries)
    self.assertFalse(coupled.supports_cuda_graph_capture)


def test_graph_prepare_forwards_without_stepping(self):
    model, entries, solvers = self._two_particle_entries((True, True))
    coupled = SolverCoupled(model, entries)
    state_before = model.state().particle_q.numpy().copy()
    coupled.prepare_cuda_graph_capture(None)
    self.assertEqual([solver.prepare_calls for solver in solvers], [1, 1])
    self.assertEqual([solver.step_count for solver in solvers], [0, 0])
    np.testing.assert_array_equal(model.particle_q.numpy(), state_before)
```

- [ ] **Step 2: Run the tests and confirm RED**

Run:

```bash
uv run --extra dev -m newton.tests -k test_coupled_solver.TestSolverCoupledGraphCapture
```

Expected: failure because `SolverBase` and `SolverCoupled` do not define the
new lifecycle API.

- [ ] **Step 3: Add the generic hooks and coupled aggregation**

Add to `SolverBase`:

```python
@property
def supports_cuda_graph_capture(self) -> bool:
    """Return whether this solver can be stepped inside a CUDA graph."""
    return True

def prepare_cuda_graph_capture(self, contacts: Contacts | None = None) -> None:
    """Allocate graph-persistent buffers without advancing simulation state."""
    del contacts
```

Add to `SolverCoupled`:

```python
@property
def supports_cuda_graph_capture(self) -> bool:
    return all(entry.solver.supports_cuda_graph_capture for entry in self._entries.values())

def prepare_cuda_graph_capture(self, contacts: Contacts | None = None) -> None:
    self.prepare_contacts(contacts)
    for entry in self._entries.values():
        entry.solver.prepare_cuda_graph_capture(self.entry_contacts(entry.name, contacts))
```

Override `SolverImplicitMPM.supports_cuda_graph_capture` so only fixed-topology
configurations report support. Use the solver's resolved `grid_type`; sparse
topology reports `False`. Override VBD preparation to idempotently size its
rigid-rigid, rigid-particle, and warm-start buffers from the supplied contact
capacities without stepping or replacing already-large buffers. Keep
`SolverCoupled.prepare_contacts()` as a compatibility wrapper that performs the
coupled allocation portion of `prepare_cuda_graph_capture()`.

- [ ] **Step 4: Run graph and existing coupled tests**

```bash
uv run --extra dev -m newton.tests -k test_coupled_solver
uv run --extra dev -m newton.tests -k test_prepare_cuda_graph_capture_preallocates_vbd_contacts
```

Expected: all coupled tests pass.

- [ ] **Step 5: Commit**

```bash
git add newton/_src/solvers/solver.py \
  newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py \
  newton/_src/solvers/coupled/solver_coupled.py \
  newton/_src/solvers/vbd/solver_vbd.py \
  newton/tests/test_coupled_solver.py newton/tests/test_solver_vbd.py
git commit -m "Add solver graph capability hooks"
```

### Task 2: Entry-local collision pipelines

**Files:**
- Modify: `newton/_src/solvers/coupled/solver_coupled.py`
- Test: `newton/tests/test_coupled_solver.py`

- [ ] **Step 1: Write failing entry-collision tests**

Extend `_FakeProxyCollisionPipeline` or add a generic recording pipeline. Test
that an `Entry(collision_pipeline=...)`:

```python
def test_entry_collision_pipeline_preserves_solver_identity(self):
    solver_instances = []
    pipelines = []
    entry = SolverCoupled.Entry(
        name="rigid",
        solver=lambda view: solver_instances.append(_ContactRecordingCopySolver(view)) or solver_instances[-1],
        bodies=[0],
        shapes=[0],
        collision_pipeline=lambda view: pipelines.append(_FakeProxyCollisionPipeline(view)) or pipelines[-1],
        collide_interval=2,
    )
    coupled = SolverCoupled(model, [entry])
    pipeline = pipelines[0]
    coupled.step(state_0, state_1, None, outer_contacts, 0.01)
    coupled.step(state_1, state_0, None, outer_contacts, 0.01)
    self.assertIs(solver_instances[0].contacts_seen[0], pipeline.contacts_buffer)
    self.assertEqual(pipeline.collide_calls, 1)
```

Also test invalid intervals, `None` factory fallback, and that a Proxy
destination uses its entry pipeline when the mapping has no legacy provider,
rejects configuring both providers for one destination, and gives an explicit
mapping provider precedence when supplied.

- [ ] **Step 2: Run and confirm RED**

```bash
uv run --extra dev -m newton.tests -k test_coupled_solver.TestSolverCoupledEntryCollision
```

Expected: `Entry` rejects unknown `collision_pipeline` and `collide_interval`
arguments.

- [ ] **Step 3: Add runtime collision state and selection**

Add fields to `SolverCoupled.Entry` and `SolverEntry`:

```python
collision_pipeline: Callable[[ModelView], object | None] | None = None
collide_interval: int | None = None

# Runtime SolverEntry fields
collision_pipeline: object | None = None
collision_contacts: Contacts | None = None
collide_interval: int = 1
collide_counter: int = 0
```

During `_build_entries`, validate the factory and interval, create the pipeline
from the finalized view after `configure_view`, compaction, and shape-pair
filtering, validate callable `contacts` and `collide`, and allocate one
persistent buffer. Require `collide_interval` to be absent when no factory is
configured. After `_distribute_state` and before `_step_coupled`,
refresh each configured entry once according to its counter:

```python
def _refresh_entry_collision_pipelines(self) -> None:
    for entry in self._entries.values():
        if entry.collision_pipeline is None:
            continue
        if entry.collide_counter % entry.collide_interval == 0:
            entry.collision_pipeline.collide(entry.state_0, entry.collision_contacts)
        entry.collide_counter += 1
```

When `_step_entry(..., filter_contacts=True)` is used, select entry-local
contacts before outer filtering. Calls with `filter_contacts=False` retain their
existing algorithm-contact precedence. Proxy refreshes a destination entry
provider after proxy-state synchronization on iteration zero and reuses the
buffer for inner iterations. Extend graph support/preparation to the optional
provider capability methods specified by the design.

- [ ] **Step 4: Run RED tests and the coupled suite**

```bash
uv run --extra dev -m newton.tests -k test_coupled_solver.TestSolverCoupledEntryCollision
uv run --extra dev -m newton.tests -k test_coupled_solver
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add newton/_src/solvers/coupled/solver_coupled.py newton/tests/test_coupled_solver.py
git commit -m "Add entry-local collision pipelines"
```

### Task 3: Public coupled contact streams

**Files:**
- Create: `newton/_src/solvers/coupled/contact_stream.py`
- Modify: `newton/_src/solvers/coupled/solver_coupled.py`
- Modify: `newton/_src/solvers/coupled/solver_coupled_proxy.py`
- Modify: `newton/_src/solvers/coupled/solver_coupled_admm.py`
- Modify: `newton/_src/solvers/coupled/__init__.py`
- Test: `newton/tests/test_coupled_solver.py`
- Test: `newton/tests/test_admm_coupled_solver.py`

- [ ] **Step 1: Write stream and diagnostics tests**

Tests must import `CoupledContactStream` from
`newton.solvers.experimental.coupled`, enumerate stable names, and verify that
enumeration does not synchronize or copy arrays:

```python
streams = {stream.name: stream for stream in solver.contact_streams(outer_contacts)}
self.assertEqual(set(streams), {"outer", "entry/rigid"})
self.assertIs(streams["outer"].contacts, outer_contacts)
self.assertEqual(streams["entry/rigid"].kind, "entry")
self.assertIs(streams["entry/rigid"].shape_local_to_parent, entry.shape_local_to_global)

stats = streams["entry/rigid"].diagnostics()
self.assertEqual(stats.rigid_capacity, stream.contacts.rigid_contact_max)
self.assertEqual(stats.rigid_overflow, max(stats.rigid_count - stats.rigid_capacity, 0))
```

Add Proxy and ADMM cases for `proxy/source/destination` and `admm/internal`, and
assert `forces_available=False` for raw ADMM detection contacts.

- [ ] **Step 2: Run and confirm RED**

```bash
uv run --extra dev -m newton.tests -k test_coupled_solver.TestCoupledContactStreams
uv run --extra dev -m newton.tests -k test_admm_coupled_solver.TestAdmmContactStreams
```

Expected: import or attribute failure for `CoupledContactStream`.

- [ ] **Step 3: Implement immutable descriptors**

Create `contact_stream.py` with public dataclasses:

```python
@dataclass(frozen=True)
class CoupledContactDiagnostics:
    rigid_count: int
    rigid_capacity: int
    rigid_overflow: int
    soft_count: int
    soft_capacity: int
    soft_overflow: int


@dataclass(frozen=True)
class CoupledContactStream:
    name: str
    kind: Literal["outer", "entry", "proxy", "admm"]
    contacts: Contacts
    source: str | None = None
    destination: str | None = None
    shape_local_to_parent: wp.array[int] | None = None
    particle_local_to_parent: wp.array[int] | None = None
    forces_available: bool = False

    def diagnostics(self) -> CoupledContactDiagnostics:
        rigid_count = int(self.contacts.rigid_contact_count.numpy()[0])
        soft_count = int(self.contacts.soft_contact_count.numpy()[0])
        return CoupledContactDiagnostics(
            rigid_count,
            self.contacts.rigid_contact_max,
            max(rigid_count - self.contacts.rigid_contact_max, 0),
            soft_count,
            self.contacts.soft_contact_max,
            max(soft_count - self.contacts.soft_contact_max, 0),
        )
```

Store shape local/global maps alongside existing body/particle maps in
`SolverEntry`. Retain the last outer contacts without copying. Implement base
stream enumeration, then append Proxy mapping buffers and ADMM's internal raw
`Contacts` buffer in subclass overrides. Export both dataclasses from the
experimental package.

- [ ] **Step 4: Run stream, API, and coupled suites**

```bash
uv run --extra dev -m newton.tests -k test_coupled_solver.TestCoupledContactStreams
uv run --extra dev -m newton.tests -k test_admm_coupled_solver.TestAdmmContactStreams
uv run --extra dev -m newton.tests -k test_api
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add newton/_src/solvers/coupled/contact_stream.py \
  newton/_src/solvers/coupled/solver_coupled.py \
  newton/_src/solvers/coupled/solver_coupled_proxy.py \
  newton/_src/solvers/coupled/solver_coupled_admm.py \
  newton/_src/solvers/coupled/__init__.py \
  newton/tests/test_coupled_solver.py newton/tests/test_admm_coupled_solver.py
git commit -m "Expose coupled contact streams"
```

### Task 4: Batched VBD joint constraint modes

**Files:**
- Modify: `newton/_src/solvers/vbd/solver_vbd.py`
- Modify: `newton/_src/solvers/vbd/rigid_vbd_kernels.py`
- Test: `newton/tests/test_solver_vbd.py`

- [ ] **Step 1: Write batch parity and atomic-validation tests**

Construct a VBD model with multiple structural joints and test:

```python
solver.set_joint_constraint_modes([joint_a, joint_b], hard=False)
np.testing.assert_array_equal(solver.joint_is_hard.numpy()[selected_slots], 0)
np.testing.assert_array_equal(solver.joint_lambda_lin.numpy()[[joint_a, joint_b]], 0.0)

before = solver.joint_is_hard.numpy().copy()
with self.assertRaisesRegex(ValueError, "out of range"):
    solver.set_joint_constraint_modes([joint_a, 999], hard=False)
np.testing.assert_array_equal(solver.joint_is_hard.numpy(), before)
```

Also compare scalar `set_joint_constraint_mode` with a one-element batch and
test a boolean sequence length mismatch. Record every affected Warp-array
object before the call and assert buffer identity is preserved afterward.

- [ ] **Step 2: Run and confirm RED**

```bash
uv run --extra dev -m newton.tests -k test_solver_vbd.TestSolverVBD.test_set_joint_constraint_modes
```

Expected: missing method failure.

- [ ] **Step 3: Implement one-transaction batching**

Add:

```python
def set_joint_constraint_modes(
    self,
    joint_indices: Sequence[int],
    hard: bool | Sequence[bool],
    slot: int | None = None,
) -> None:
```

Normalize and validate all indices, modes, and slots before reading mutable
arrays. Upload joint ids and normalized modes once, then launch one device
kernel that mutates `joint_is_hard`, `joint_lambda_lin`, `joint_lambda_ang`,
`joint_C0_lin`, and `joint_C0_ang` in place. Clear lambda/C0 for every slot
switched to soft. Refactor the singular method to call
`set_joint_constraint_modes([joint_index], hard, slot)`; never replace an array
whose pointer may already be captured by a CUDA graph.

- [ ] **Step 4: Run VBD tests**

```bash
uv run --extra dev -m newton.tests -k test_solver_vbd
```

Expected: all VBD tests pass.

- [ ] **Step 5: Commit**

```bash
git add newton/_src/solvers/vbd/solver_vbd.py \
  newton/_src/solvers/vbd/rigid_vbd_kernels.py newton/tests/test_solver_vbd.py
git commit -m "Add batched VBD joint mode updates"
```

### Task 5: Base coupled masked-reset contract

**Files:**
- Create: `newton/_src/solvers/coupled/reset_utils.py`
- Modify: `newton/_src/solvers/coupled/solver_coupled.py`
- Test: `newton/tests/test_coupled_solver.py`

- [ ] **Step 1: Write failing base reset tests**

Add `TestSolverCoupledReset` with wrong dtype/device/length validation,
parent-world mask forwarding through a compact entry, selected-only transient
clearing, substep scratch synchronization, and unchanged full reset behavior.
Use two worlds and record entry state before resetting only world 0:

```python
mask = wp.array([True, False], dtype=wp.bool, device=model.device)
unselected_before = snapshot_entry_world(solver, "entry", 1)
solver.reset(state, world_mask=mask)
assert_entry_world_equal(solver, "entry", 1, unselected_before)
assert_entry_world_forces_zero(solver, "entry", 0)
```

- [ ] **Step 2: Run and confirm RED**

```bash
uv run --extra dev -m newton.tests -k test_coupled_solver.TestSolverCoupledReset
```

Expected: missing validation and whole-entry transient clearing change world 1.

- [ ] **Step 3: Add shared mask helpers**

Create `reset_utils.py` with exact mask validation and typed row kernels:

```python
def validate_reset_world_mask(model: Model, world_mask: wp.array | None) -> wp.array | None:
    if world_mask is None:
        return None
    if not isinstance(world_mask, wp.array) or world_mask.dtype != wp.bool:
        raise TypeError("world_mask must be a Warp boolean array")
    if world_mask.device != model.device:
        raise ValueError("world_mask must use the model device")
    if world_mask.ndim != 1 or world_mask.shape[0] != model.world_count:
        raise ValueError("world_mask length must equal model.world_count")
    return world_mask
```

Implement `zero_*_rows_by_world` and `copy_*_rows_by_world` kernels for float,
int, vec3, spatial-vector, and transform arrays. A negative world id is never
selected by a partial mask.

- [ ] **Step 4: Make base synchronization mask-aware**

Validate once in `SolverCoupled.reset`, forward the unchanged parent mask to
entry solvers, and change `_sync_entry_reset_state(entry, world_mask, flags)` to
copy selected body/particle/joint rows into `state_1` and `state_tmp` while
clearing only selected transient force rows. Retain existing bulk copies for
`world_mask=None`.

- [ ] **Step 5: Run and commit**

```bash
uv run --extra dev -m newton.tests -k test_coupled_solver.TestSolverCoupledReset
uv run --extra dev -m newton.tests -k test_coupled_solver
git add newton/_src/solvers/coupled/reset_utils.py \
  newton/_src/solvers/coupled/solver_coupled.py newton/tests/test_coupled_solver.py
git commit -m "Make coupled reset world-mask safe"
```

### Task 6: World-local Proxy state

**Files:**
- Modify: `newton/_src/solvers/coupled/solver_coupled_proxy.py`
- Modify: `newton/_src/solvers/coupled/proxy_utils.py`
- Test: `newton/tests/test_coupled_solver.py`

- [ ] **Step 1: Write failing Proxy isolation tests**

Add cross-world particle/joint mapping rejection and selected-only reset tests
for body feedback, particle feedback, previous forces, residuals, and velocity
snapshots. Add a two-world Aitken test proving each world can converge to a
different relaxation value and a masked reset preserves the unselected value.

- [ ] **Step 2: Run and confirm RED**

```bash
uv run --extra dev -m newton.tests -k test_coupled_solver.TestSolverCoupledProxyReset
```

Expected: particle/joint cross-world mappings are accepted, Aitken uses one
global slot, and Proxy reset clears both worlds.

- [ ] **Step 3: Partition mapping and Aitken state**

Store a world id aligned with each mapping row. Represent Aitken state with
slot 0 for world `-1` and slot `world_id + 1` otherwise:

```python
aitken_stats = wp.zeros((model.world_count + 1, 2), dtype=float, device=device)
aitken_relaxation = wp.full(model.world_count + 1, proxy_relaxation, dtype=float, device=device)
aitken_has_previous = wp.zeros(model.world_count + 1, dtype=int, device=device)
```

Update accumulation, relaxation, blend, and reset kernels to index the mapping
row's slot. Generalize entity-world validation and invoke it for bodies,
particles, and joint endpoint bodies.

- [ ] **Step 4: Honor masks in Proxy reset**

For partial reset, use indexed masked kernels for full-model coupling force and
velocity arrays, row-masked kernels for previous force/residual arrays, and
slot-masked kernels for Aitken state. Preserve the current bulk zero/fill fast
path for full reset.

- [ ] **Step 5: Run and commit**

```bash
uv run --extra dev -m newton.tests -k test_coupled_solver.TestSolverCoupledProxyReset
uv run --extra dev -m newton.tests -k test_coupled_solver
git add newton/_src/solvers/coupled/solver_coupled_proxy.py \
  newton/_src/solvers/coupled/proxy_utils.py newton/tests/test_coupled_solver.py
git commit -m "Partition proxy reset state by world"
```

### Task 7: World-selective contact matching reset

**Files:**
- Modify: `newton/_src/geometry/contact_match.py`
- Modify: `newton/_src/sim/collide.py`
- Test: `newton/tests/test_contact_matching.py`

- [ ] **Step 1: Write failing matcher tests**

Add tests that save matches for two worlds, reset world 0, and prove sticky
replay and matching are disabled only for world 0 while world 1 keeps its match.

- [ ] **Step 2: Run and confirm RED**

```bash
uv run --extra dev -m newton.tests -k test_contact_matching
```

Expected: `ContactMatcher` has only global reset behavior and
`CollisionPipeline` exposes no selective reset method.

- [ ] **Step 3: Track and invalidate previous-row worlds**

Add capacity-sized `_prev_world` and `_prev_valid` arrays. During sorted-state
save, record the effective contact world and mark the row valid. Add a kernel:

```python
@wp.kernel(enable_backward=False)
def _invalidate_prev_worlds_kernel(
    previous_world: wp.array[int],
    previous_valid: wp.array[int],
    world_mask: wp.array[bool],
):
    i = wp.tid()
    world = previous_world[i]
    if world >= 0 and world_mask[world]:
        previous_valid[i] = 0
```

Matching, sticky replay, and broken-contact reporting skip invalid rows.
Implement `ContactMatcher.reset(world_mask=None)` and public
`CollisionPipeline.reset_contact_matching(world_mask=None)`; the latter is a
no-op when matching is disabled.

- [ ] **Step 4: Run and commit**

```bash
uv run --extra dev -m newton.tests -k test_contact_matching
git add newton/_src/geometry/contact_match.py newton/_src/sim/collide.py \
  newton/tests/test_contact_matching.py
git commit -m "Reset contact matching by world"
```

### Task 8: Connect Proxy collision reset

**Files:**
- Modify: `newton/_src/solvers/coupled/solver_coupled_proxy.py`
- Test: `newton/tests/test_coupled_solver.py`

- [ ] **Step 1: Write failing Proxy contact-cache tests**

Add tests that a partial reset preserves an unselected world's match, prevents
sticky replay in the selected world, and forces collision detection on the next
step even when `collide_interval > 1`.

- [ ] **Step 2: Run and confirm RED**

```bash
uv run --extra dev -m newton.tests -k test_coupled_solver.TestSolverCoupledProxyReset
```

Expected: Proxy globally clears matching state or reuses stale selected-world
contacts.

- [ ] **Step 3: Reset Proxy providers safely**

For each `_ProxyCollisionConfig`, call
`pipeline.reset_contact_matching(world_mask)`, clear current contact counts,
and set the scalar cadence so the next outer step performs a fresh global
collision pass. The extra pass may include unselected worlds, but their matcher
history and coupling warm starts remain intact.

- [ ] **Step 4: Run and commit**

```bash
uv run --extra dev -m newton.tests -k test_coupled_solver.TestSolverCoupledProxyReset
git add newton/_src/solvers/coupled/solver_coupled_proxy.py \
  newton/tests/test_coupled_solver.py
git commit -m "Reset proxy contact caches safely"
```

### Task 9: Static ADMM masked reset

**Files:**
- Modify: `newton/_src/solvers/coupled/admm_utils.py`
- Modify: `newton/_src/solvers/coupled/solver_coupled_admm.py`
- Test: `newton/tests/test_admm_coupled_solver.py`

- [ ] **Step 1: Write failing static-group tests**

Use two worlds and cover joint attachments, body-particle attachments, angular
groups, and friction groups. Fill `u`, `lambda_`, `Jv`, and `u_target` with
world-distinct sentinels, reset world 0, and assert world 1 is unchanged. Add a
cross-world body-particle attachment rejection test and full-reset parity.

- [ ] **Step 2: Run and confirm RED**

```bash
uv run --extra dev -m newton.tests -k test_admm_coupled_solver.TestAdmmReset
```

Expected: current group reset zeros all worlds.

- [ ] **Step 3: Add static group world ids and masked reset**

Add `world_ids: wp.array[int]` to every static attachment/friction group,
populate it from parent endpoints before local remapping, and reject endpoints
from different worlds. Use row-masked kernels for `u`, `lambda_`, `Jv`, and
targets. Use entry view world arrays to copy selected primal snapshots and
clear selected force rows. Do not clear topology, effective mass, or unselected
high-water buffers.

- [ ] **Step 4: Run and commit**

```bash
uv run --extra dev -m newton.tests -k test_admm_coupled_solver.TestAdmmReset
git add newton/_src/solvers/coupled/admm_utils.py \
  newton/_src/solvers/coupled/solver_coupled_admm.py \
  newton/tests/test_admm_coupled_solver.py
git commit -m "Reset static ADMM state by world"
```

### Task 10: Dynamic ADMM contact reset

**Files:**
- Modify: `newton/_src/solvers/coupled/admm_contact_stream.py`
- Modify: `newton/_src/solvers/coupled/admm_utils.py`
- Modify: `newton/_src/solvers/coupled/solver_coupled_admm.py`
- Test: `newton/tests/test_admm_coupled_solver.py`

- [ ] **Step 1: Write failing dynamic-contact tests**

Cover rigid-rigid, rigid-particle, and particle-particle warm starts. Compare
world 1 against an identical solver that was never reset, verify reset-world
sticky contacts are not replayed, and verify partial reset preserves global
contact high-water statistics.

- [ ] **Step 2: Run and confirm RED**

```bash
uv run --extra dev -m newton.tests -k test_admm_coupled_solver.TestAdmmReset
```

Expected: dynamic duals, streams, matching, and statistics are globally reset.

- [ ] **Step 3: Track dynamic row worlds**

Add capacity-sized `world_ids` to dynamic group and `AdmmContactStream`
structures. Extend RR/RP/PP contact-fill kernels to write the endpoint world
alongside every active row.

- [ ] **Step 4: Preserve unselected warm starts**

On partial reset, zero selected rows of `u`, `lambda_`, `Jv`, `u_min`, normal
force, and normal impulse. Keep dense active rows/counts long enough for the
next refresh to snapshot selected zeros and unselected prior duals. Do not
clear max statistics. Call
`_admm_collision_pipeline.reset_contact_matching(world_mask)` and force a fresh
collision pass without discarding unselected keyed warm starts. Keep the
current complete-clear path for `world_mask=None`.

- [ ] **Step 5: Run and commit**

```bash
uv run --extra dev -m newton.tests -k test_admm_coupled_solver.TestAdmmReset
uv run --extra dev -m newton.tests -k test_admm_coupled_solver
uv run --extra dev -m newton.tests -k test_contact_matching
git add newton/_src/solvers/coupled/admm_contact_stream.py \
  newton/_src/solvers/coupled/admm_utils.py \
  newton/_src/solvers/coupled/solver_coupled_admm.py \
  newton/tests/test_admm_coupled_solver.py
git commit -m "Reset ADMM contact state by world"
```

### Task 11: Public Proxy and ADMM diagnostics

**Files:**
- Modify: `newton/_src/solvers/coupled/contact_stream.py`
- Modify: `newton/_src/solvers/coupled/solver_coupled_proxy.py`
- Modify: `newton/_src/solvers/coupled/solver_coupled_admm.py`
- Modify: `newton/_src/solvers/coupled/__init__.py`
- Test: `newton/tests/test_coupled_solver.py`
- Test: `newton/tests/test_admm_coupled_solver.py`

- [ ] **Step 1: Write public-accessor tests**

Import all result types from the public experimental package. Verify Proxy
feedback results contain parent ids and the exact read-only Warp arrays, unknown
directions raise `KeyError`, and relaxation diagnostics are per world. Verify
ADMM diagnostics return configured iterations, current/high-water contact
counts, overflow, and finite per-interface primal/dual residual norms without
changing solver arrays.

- [ ] **Step 2: Run and confirm RED**

```bash
uv run --extra dev -m newton.tests -k test_coupled_solver.TestSolverCoupledProxyDiagnostics
uv run --extra dev -m newton.tests -k test_admm_coupled_solver.TestAdmmDiagnostics
```

Expected: missing public result types and accessors.

- [ ] **Step 3: Add immutable Proxy results**

Define and export frozen `ProxyBodyFeedback`, `ProxyParticleFeedback`, and
`ProxyRelaxationDiagnostics` dataclasses. Implement:

```python
def proxy_body_feedback(self, source: str, destination: str) -> ProxyBodyFeedback
def proxy_particle_feedback(self, source: str, destination: str) -> ProxyParticleFeedback
def proxy_relaxation_diagnostics(self, source: str, destination: str) -> ProxyRelaxationDiagnostics
```

Return existing device arrays directly and document them as read-only views.

- [ ] **Step 4: Add on-demand ADMM reductions**

Define and export frozen `AdmmInterfaceDiagnostics` and `AdmmDiagnostics`.
Allocate stable one-element device arrays for current/max/overflow counts and
residual reductions. On `diagnostics()`, zero current outputs and launch
graph-capturable reductions from existing `Jv`, `u`, `u_target`, and `lambda_`
arrays. Return live read-only device arrays and leave simulation buffers
unchanged. No reduction kernel runs during ordinary `step()`.

- [ ] **Step 5: Run diagnostics and coupled suites**

```bash
uv run --extra dev -m newton.tests -k test_coupled_solver.TestSolverCoupledProxyDiagnostics
uv run --extra dev -m newton.tests -k test_admm_coupled_solver.TestAdmmDiagnostics
uv run --extra dev -m newton.tests -k test_coupled_solver
uv run --extra dev -m newton.tests -k test_admm_coupled_solver
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add newton/_src/solvers/coupled/contact_stream.py \
  newton/_src/solvers/coupled/solver_coupled_proxy.py \
  newton/_src/solvers/coupled/solver_coupled_admm.py \
  newton/_src/solvers/coupled/__init__.py \
  newton/tests/test_coupled_solver.py newton/tests/test_admm_coupled_solver.py
git commit -m "Expose coupled solver diagnostics"
```

### Task 12: Documentation, release notes, and integration verification

**Files:**
- Modify: `docs/concepts/coupling.rst`
- Modify: `CHANGELOG.md`
- Modify: generated API files selected by `docs/generate_api.py`
- Test: `newton/tests/test_generate_api.py`

- [ ] **Step 1: Add executable documentation examples**

Document entry-local collision, deliberate contact-stream selection,
copy-on-write Proxy shape overrides through `Entry.configure_view`, masked
reset semantics, graph capability/preparation, diagnostics, and batched VBD
mode changes. Use only public imports such as:

```python
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledProxy

entry = SolverCoupled.Entry(
    name="rigid",
    solver=lambda view: newton.solvers.SolverMuJoCo(view, use_mujoco_contacts=False),
    collision_pipeline=lambda view: newton.CollisionPipeline(view),
)
```

- [ ] **Step 2: Add `[Unreleased]` changelog entries**

Add imperative `Added` entries for contact streams, graph lifecycle,
diagnostics, entry collision, and VBD batching, plus a `Fixed` entry stating
that coupled partial resets preserve unselected worlds.

- [ ] **Step 3: Generate and verify public API docs**

```bash
uv run docs/generate_api.py
uv run --extra dev -m newton.tests -k test_generate_api
```

Expected: generation succeeds and generated files are current.

- [ ] **Step 4: Run focused and full verification**

```bash
uv run --extra dev -m newton.tests -k test_coupled_solver
uv run --extra dev -m newton.tests -k test_admm_coupled_solver
uv run --extra dev -m newton.tests -k test_solver_vbd
uv run --extra dev -m newton.tests -k test_collision_pipeline
uv run --extra dev -m newton.tests
uvx pre-commit run -a
git diff --check c948e5116ab40d1543858e2c99e9fe2726364925..HEAD
```

Expected: zero failures and no formatting errors.

- [ ] **Step 5: Commit documentation and generated API output**

```bash
git add docs/concepts/coupling.rst CHANGELOG.md docs/api newton/tests/test_generate_api.py
git commit -m "Document coupled integration APIs"
```

- [ ] **Step 6: Perform final review**

Review every commit and the complete diff against `c948e511`, confirm no
IsaacLab-specific code entered Newton, and confirm `git status --short` is
empty.
