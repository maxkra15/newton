# Newton Multi-World Sparse Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce `max/newton-max`, combining coupled solver APIs, isolated multi-world implicit MPM, and rebuildable sparse outer CUDA capture against `max/warp-max`.

**Architecture:** Preserve the coupled, multi-world, and sparse feature histories as merge commits. Add interaction tests before each later merge, resolve collider and grid conflicts as unions of both contracts, then add explicit capture capability, status lifecycle, selective reset, documentation, and total-step benchmarks.

**Tech Stack:** Python 3.12, Newton, Warp FEM, CUDA graphs, `unittest`, ASV, `uv`.

---

### Task 1: Merge the coupled public API lineage

**Files:**
- Worktree: `/home/maximiliank/.config/superpowers/worktrees/newton-mpm-multiworld/max-implicit-mpm-multiworld-sparse-capture`
- Branch: `max/newton-max`
- Input repository: `/home/maximiliank/Work/newton-coupled-upstream`

- [ ] **Step 1: Verify branch state and fetch the pinned coupled commit**

```bash
test "$(git branch --show-current)" = max/newton-max
test -z "$(git status --porcelain)"
git fetch /home/maximiliank/Work/newton-coupled-upstream \
  max/coupled-public-apis:refs/remotes/local/coupled-public-apis
test "$(git rev-parse refs/remotes/local/coupled-public-apis)" = 9d70911e686fb59cc666592e764cb79ffa7f3ec5
```

Expected: clean branch and exact coupled SHA.

- [ ] **Step 2: Merge the coupled lineage**

```bash
git merge --no-ff --no-edit 9d70911e686fb59cc666592e764cb79ffa7f3ec5
```

Expected: clean merge. Review automatic changes to `CHANGELOG.md`, Kamino core model, and Kamino solver.

- [ ] **Step 3: Run coupled API baselines against merged Warp**

```bash
export PYTHONPATH=/home/maximiliank/.config/superpowers/worktrees/warp-main-multiworld-reference/max-warp-max
uv run --extra dev -m newton.tests -p 'test_coupled_solver.py' -k 'cuda_graph or reset' -j 1 -v
```

Expected: coupled graph-capability and reset tests PASS.

### Task 2: Integrate multi-world collider ownership test-first

**Files:**
- Create: `newton/tests/test_implicit_mpm_multiworld_sparse.py`
- Modify during merge: `newton/_src/solvers/implicit_mpm/implicit_mpm_model.py`
- Modify during merge: `newton/_src/solvers/implicit_mpm/rasterized_collisions.py`
- Semantic review: `newton/_src/solvers/implicit_mpm/implicit_mpm_solver_kernels.py`
- Semantic review: `newton/tests/test_implicit_mpm.py`

- [ ] **Step 1: Add the failing coupled multi-world isolation regression**

Create a `unittest` module using `add_function_test` and selected CUDA devices. Build two Newton worlds with colocated MPM particles and one dynamic collider per world. Include coupled proxy/collider particle IDs. The first test must assert:

```python
config = SolverImplicitMPM.Config()
test.assertTrue(config.separate_worlds)
solver = SolverImplicitMPM(model, config=config)
test.assertEqual(solver._environment_count, 2)
np.testing.assert_array_equal(
    solver._mpm_model.collider.collider_world.numpy(),
    np.array([0, 1], dtype=np.int32),
)
test.assertEqual(solver._mpm_model.collider.query_world_offsets.shape[0], model.world_count + 2)
```

Also assert proxy/collider-only particles are absent from the active material mask while their `collider_particle_offsets` and `collider_particle_ids` remain aligned with collider IDs.

- [ ] **Step 2: Run RED**

```bash
uv run --extra dev -m newton.tests \
  -p 'test_implicit_mpm_multiworld_sparse.py' -k coupled_multiworld -j 1 -v
```

Expected: FAIL because the coupled tip has no `Config.separate_worlds` or collider world-query tables.

- [ ] **Step 3: Merge isolated multi-world MPM**

```bash
git merge --no-ff d61ea124f933b4df114df8da12d1afc7f8eeb468
```

Expected conflicts only in `implicit_mpm_model.py` and `rasterized_collisions.py`.

- [ ] **Step 4: Resolve the collider model as a union**

Keep these `Collider` fields together:

```python
collider_world: wp.array[int]
collider_face_offset: wp.array[int]
query_collider_ids: wp.array[int]
query_world_offsets: wp.array[int]
collider_particle_offsets: wp.array[int]
collider_particle_ids: wp.array[int]
```

Retain multi-world validation/query packing and effective body mass from `d61ea124`; retain `_refresh_particle_flags_and_extrema`, proxy/deformable collider particle exclusion, and deformable vertex ranges from `9d70911e`. Refresh particle flags once, then call `notify_collider_changed(effective_body_mass)` once.

- [ ] **Step 5: Run GREEN and existing multi-world/proxy suites**

```bash
uv run --extra dev -m newton.tests -p 'test_implicit_mpm_multiworld_sparse.py' -j 1 -v
uv run --extra dev -m newton.tests -p 'test_implicit_mpm.py' -k multiworld -j 1 -v
uv run --extra dev -m newton.tests -p 'test_implicit_mpm_proxy_particles.py' -j 1 -v
```

Expected: all selected tests PASS.

- [ ] **Step 6: Commit the conflict resolution and regression**

```bash
git add newton/_src/solvers/implicit_mpm/implicit_mpm_model.py \
  newton/_src/solvers/implicit_mpm/rasterized_collisions.py \
  newton/_src/solvers/implicit_mpm/implicit_mpm_solver_kernels.py \
  newton/tests/test_implicit_mpm.py \
  newton/tests/test_implicit_mpm_multiworld_sparse.py CHANGELOG.md
git commit -m "Combine coupled and isolated MPM worlds"
```

### Task 3: Integrate rebuildable sparse multi-world grids test-first

**Files:**
- Modify: `newton/tests/test_implicit_mpm_multiworld_sparse.py`
- Merge/create: `newton/tests/test_implicit_mpm_rebuildable_sparse.py`
- Modify: `newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py`
- Modify: `newton/_src/solvers/implicit_mpm/implicit_mpm_solver_kernels.py`

- [ ] **Step 1: Add the failing rebuildable sparse tests**

Add three tests using two colocated worlds, Jacobi, `grid_padding=0`, and a positive total capacity:

```python
config = SolverImplicitMPM.Config(
    separate_worlds=True,
    grid_type="sparse",
    max_active_cell_count=128,
    grid_padding=0,
    solver="jacobi",
    tolerance=0.0,
    max_iterations=5,
    strain_basis="P0",
    velocity_basis="Q1",
    collider_basis="S2",
)
solver = SolverImplicitMPM(model, config=config, enable_timers=False)
test.assertTrue(getattr(solver, "_sparse_rebuildable", False))
test.assertEqual(solver._scratchpad.grid.environment_count(), 2)
test.assertTrue(solver.supports_cuda_graph_capture)
```

The tests must cover: environment-tagged construction, independent topology changes across replay, and eager-versus-outer-captured equality after capturing a complete two-state-buffer alternation. Assert stable cell and edge grid IDs and call `check_sparse_grid_rebuild_status()` after replay.

- [ ] **Step 2: Run RED**

```bash
uv run --extra dev -m newton.tests \
  -p 'test_implicit_mpm_multiworld_sparse.py' -k rebuildable_sparse -j 1 -v
```

Expected: FAIL because sparse MPM still reallocates and its graph capability remains fixed-only.

- [ ] **Step 3: Merge the rebuildable sparse input**

```bash
git merge --no-ff a8601c24a72a1ff5fb75111e3a4ac41e0d649895
```

Expected: conflict in `solver_implicit_mpm.py`; retain both test modules and changelog entries.

- [ ] **Step 4: Use Warp's flat multi-environment rebuild API**

For isolated rebuildable sparse construction, use:

```python
grid = fem.Nanogrid.from_environment_voxels(
    positions,
    self._particle_environment,
    self._environment_count,
    point_mask=self._update_grid_point_mask(self._mpm_model.particle_flags),
    voxel_size=self.voxel_size,
    temporary_store=self.temporary_store,
    device=positions.device,
    rebuildable=True,
    max_active_voxels=self.max_active_cell_count,
    status=self._grid_status,
)
```

On replay use:

```python
grid.rebuild(
    positions,
    point_envs=self._particle_environment,
    status=self._grid_status,
    point_mask=self._update_grid_point_mask(self._mpm_model.particle_flags),
)
```

Use `_mpm_model.particle_flags`, preserve `environment_first=True` partitions and `mark_active_cells_by_environment`, preserve legacy shared/allocating sparse paths, and treat `max_active_cell_count` as total capacity.

- [ ] **Step 5: Run GREEN**

```bash
uv run --extra dev -m newton.tests \
  -p 'test_implicit_mpm_multiworld_sparse.py' -j 1 -v
uv run --extra dev -m newton.tests \
  -p 'test_implicit_mpm_rebuildable_sparse.py' -j 1 -v
```

Expected: all selected tests PASS against the built `max/warp-max` native library.

- [ ] **Step 6: Commit**

```bash
git add newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py \
  newton/_src/solvers/implicit_mpm/implicit_mpm_solver_kernels.py \
  newton/tests/test_implicit_mpm_multiworld_sparse.py \
  newton/tests/test_implicit_mpm_rebuildable_sparse.py CHANGELOG.md
git commit -m "Capture sparse MPM across isolated worlds"
```

### Task 4: Add capture capability, status lifecycle, and selective reset

**Files:**
- Modify: `newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py`
- Modify: `newton/_src/solvers/implicit_mpm/implicit_mpm_solver_kernels.py`
- Modify: `newton/tests/test_implicit_mpm_multiworld_sparse.py`
- Modify: `newton/tests/test_implicit_mpm_rebuildable_sparse.py`
- Modify: `newton/tests/test_coupled_solver.py`

- [ ] **Step 1: Add RED tests for capability and sticky status**

Assert that isolated sparse capture is true only for CUDA, enabled memory pool, conditional graphs, disabled timers, positive capacity, zero padding, supported basis topology, and Jacobi. Assert CG/CR/GMRES and multi-world Gauss-Seidel are false. Assert coupled capability is the conjunction of entry capabilities.

Add overflow lifecycle assertions:

```python
with test.assertRaisesRegex(RuntimeError, "sparse grid rebuild capacity"):
    solver.check_sparse_grid_rebuild_status()
solver.clear_sparse_grid_rebuild_status()
solver.check_sparse_grid_rebuild_status()
```

Capture two substeps with overflow only in the first and assert accumulated status remains nonzero until explicitly cleared.

- [ ] **Step 2: Add RED test for masked MPM reset**

Mutate `particle_elastic_strain`, `particle_transform`, `particle_qd_grad`, `particle_stress`, and `particle_Jp` in both worlds. Call:

```python
solver.reset(state, world_mask=wp.array([True, False], dtype=wp.bool, device=device))
```

Assert world 0 becomes identity/identity/zero/zero/one and world 1 is bitwise unchanged. Expected RED: inherited reset is a no-op.

- [ ] **Step 3: Implement the public capability contract**

Implement `supports_cuda_graph_capture` so fixed grids require fixed capacity and rebuildable sparse grids require `_sparse_rebuildable`; reject timers and unsupported multi-world solver sequences. `prepare_cuda_graph_capture()` validates support and materializes any required lazy S2 edge topology without stepping state.

- [ ] **Step 4: Implement status clear without device printing**

Remove `wp.printf` from `record_volume_rebuild_status`. Add:

```python
def clear_sparse_grid_rebuild_status(self) -> None:
    if self.model.device.is_capturing:
        raise RuntimeError("Cannot clear sparse grid rebuild status during CUDA graph capture")
    if self._grid_status is not None:
        self._grid_status.zero_()
    if self._grid_accumulated_status is not None:
        self._grid_accumulated_status.zero_()
```

Keep accumulation sticky across substeps and decode all Warp rebuild capacity flags in the host check.

- [ ] **Step 5: Implement world-selective MPM reset**

Add a Warp kernel that checks `particle_world[particle]` against `world_mask` and restores the five MPM history fields. `SolverImplicitMPM.reset()` launches it for a partial mask, fills all particles for `world_mask=None`, refreshes collider previous-pose caches from the supplied state, and clears sparse rebuild status at the explicit reset boundary. Do not alter unselected particle arrays.

- [ ] **Step 6: Run GREEN and commit**

```bash
uv run --extra dev -m newton.tests -p 'test_implicit_mpm_multiworld_sparse.py' -j 1 -v
uv run --extra dev -m newton.tests -p 'test_implicit_mpm_rebuildable_sparse.py' -j 1 -v
uv run --extra dev -m newton.tests -p 'test_coupled_solver.py' -k 'cuda_graph or reset' -j 1 -v
git add newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py \
  newton/_src/solvers/implicit_mpm/implicit_mpm_solver_kernels.py \
  newton/tests/test_implicit_mpm_multiworld_sparse.py \
  newton/tests/test_implicit_mpm_rebuildable_sparse.py newton/tests/test_coupled_solver.py
git commit -m "Define sparse MPM graph and reset lifecycle"
```

Expected: all selected tests PASS.

### Task 5: Document and benchmark end-to-end behavior

**Files:**
- Modify: `docs/concepts/worlds.rst`
- Modify: `docs/concepts/coupling.rst`
- Modify: `asv/benchmarks/simulation/bench_implicit_mpm.py`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Update documentation**

Document allocating sparse versus capacity-bounded rebuildable sparse, dynamic environment packing offsets, total capacity semantics, supported Jacobi multi-world capture, explicit status checks, reset, and unchanged dense/legacy behavior. Remove the stale statement that every sparse grid is excluded from outer capture.

- [ ] **Step 2: Extend the benchmark matrix**

Add modes:

```python
params = ([1, 8, 32], [
    "shared-sparse-allocating",
    "isolated-sparse-eager",
    "isolated-sparse-captured",
    "isolated-fixed-captured",
])
```

Time complete synchronized solver steps after warm-up. Track graph/setup time separately, median and p95 step milliseconds, steps/s, environment count, particles/environment, total capacity, live active cells, and peak CUDA memory. Capture the complete state-buffer alternation for captured modes.

- [ ] **Step 3: Run the benchmark smoke**

```bash
uv run --extra dev python asv/benchmarks/simulation/bench_implicit_mpm.py \
  --bench FastImplicitMPMMultiworld
```

Expected: every supported mode emits finite total-step and memory metrics; unsupported capabilities report explicit skips.

- [ ] **Step 4: Commit**

```bash
git add docs/concepts/worlds.rst docs/concepts/coupling.rst \
  asv/benchmarks/simulation/bench_implicit_mpm.py CHANGELOG.md
git commit -m "Document and benchmark sparse MPM worlds"
```

### Task 6: Verify Newton

**Files:**
- Verification only

- [ ] **Step 1: Run focused suites**

```bash
uv run --extra dev -m newton.tests -p 'test_implicit_mpm_multiworld_sparse.py' -j 1 -v
uv run --extra dev -m newton.tests -p 'test_implicit_mpm_rebuildable_sparse.py' -j 1 -v
uv run --extra dev -m newton.tests -p 'test_implicit_mpm.py' -k multiworld -j 1 -v
uv run --extra dev -m newton.tests -p 'test_implicit_mpm_proxy_particles.py' -j 1 -v
uv run --extra dev -m newton.tests -p 'test_coupled_solver.py' -k 'cuda_graph or reset' -j 1 -v
uv run --extra dev -m newton.tests -p 'test_admm_coupled_solver.py' -j 1 -v
```

Expected: all selected suites PASS.

- [ ] **Step 2: Run formatting and consistency gates**

```bash
uv run docs/generate_api.py
uvx pre-commit run -a
git diff --check 49ca73b9dd088e48a853aa7033f928f3bd84aa78..HEAD
git status --short
```

Expected: generated API is current, hooks pass, and the worktree is clean.
