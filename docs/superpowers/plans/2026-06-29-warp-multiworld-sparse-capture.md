# Warp Multi-World Sparse Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce `max/warp-max`, combining rebuildable NanoVDB/S2 topology with capture-safe FEM environment partitions and dynamic collision-free environment packing.

**Architecture:** Build on the NanoVDB branch first, add device-only automatic environment-offset recomputation, then prove the environment-partition path fails under capture before merging its fix branch. Rebuild Warp's native library and validate source/native ABI consistency on CUDA.

**Tech Stack:** Python 3.12, Warp Python/C++/CUDA, NanoVDB, Warp FEM, `unittest`, CUDA graphs, `uv`.

---

### Task 1: Create the isolated Warp integration branch

**Files:**
- Worktree: `/home/maximiliank/.config/superpowers/worktrees/warp-main-multiworld-reference/max-warp-max`
- Source repository: `/home/maximiliank/Work/warp-main-multiworld-reference`

- [ ] **Step 1: Verify immutable inputs**

```bash
git -C /home/maximiliank/Work/warp-main-multiworld-reference fetch --prune origin
git -C /home/maximiliank/Work/warp-main-multiworld-reference fetch --prune fork
test "$(git -C /home/maximiliank/Work/warp-main-multiworld-reference rev-parse origin/main)" = b8596fd51f6f855f32bda2d39d6c6bbc95c26e94
test "$(git -C /home/maximiliank/Work/warp-main-multiworld-reference rev-parse fork/max/s2-rebuildable-edge-grid)" = 2eb92885e93d821aefdc82392ad9219c8205e619
test "$(git -C /home/maximiliank/Work/warp-main-multiworld-reference rev-parse fork/max/capture-environment-partition)" = 675c15b7f9c72088a8bcd36529497965191e3911
```

Expected: all three `test` commands exit 0.

- [ ] **Step 2: Create an isolated branch and merge the NanoVDB stack**

```bash
git -C /home/maximiliank/Work/warp-main-multiworld-reference worktree add \
  /home/maximiliank/.config/superpowers/worktrees/warp-main-multiworld-reference/max-warp-max \
  -b max/warp-max b8596fd51f6f855f32bda2d39d6c6bbc95c26e94
git -C /home/maximiliank/.config/superpowers/worktrees/warp-main-multiworld-reference/max-warp-max \
  merge --no-ff --no-edit 2eb92885e93d821aefdc82392ad9219c8205e619
```

Expected: clean merge; `git status --short` is empty.

### Task 2: Recompute automatic environment offsets during rebuild

**Files:**
- Modify: `warp/_src/fem/geometry/nanogrid.py`
- Test: `warp/tests/fem/test_fem_multi_env.py`

- [ ] **Step 1: Add the failing moving-environment isolation test**

Add `test_nanogrid_multi_env_rebuild_updates_automatic_offsets`. Construct a rebuildable two-environment grid with initial points `[(0,0,0), (0,0,0)]`, then rebuild with environment 0 points at `x={0,4}` and environment 1 at `x=0`. Assert:

```python
initial_offsets = geo.env_offsets.numpy().copy()
geo.rebuild(moved_points, moved_envs, status=status)

test.assertEqual(int(status.numpy()[0]), wp.Volume.REBUILD_SUCCESS)
test.assertEqual(geo.cell_grid.get_active_stats().voxel_count, 3)
test.assertFalse(np.array_equal(geo.env_offsets.numpy(), initial_offsets))
test.assertGreaterEqual(geo.env_offsets.numpy()[1, 0] - 4, 3)
np.testing.assert_array_equal(
    np.sort(geo._cell_env.numpy()[:3]),
    np.array([0, 0, 1], dtype=np.int32),
)
```

Register it with `add_function_test(..., devices=cuda_devices)`.

- [ ] **Step 2: Run RED**

```bash
uv run --extra dev warp/tests/fem/test_fem_multi_env.py \
  -k nanogrid_multi_env_rebuild_updates_automatic_offsets -v
```

Expected: FAIL because the automatically generated offsets retain their initial values and two environments alias packed coordinate `x=4`.

- [ ] **Step 3: Add persistent device-side packing state**

In `Nanogrid.from_environment_voxels`, retain whether `env_offsets` was omitted and pass that flag plus `guard_cells` into `Nanogrid.__init__`. In `__init__`, allocate fixed-size device arrays only for automatic multi-environment packing:

```python
self._automatic_env_offsets = automatic_env_offsets and self.environment_count() > 1
self._environment_guard_cells = guard_cells
self._env_cell_counts = wp.empty(self.environment_count(), dtype=int, device=device)
self._env_min_x = wp.empty(self.environment_count(), dtype=int, device=device)
self._env_max_x = wp.empty(self.environment_count(), dtype=int, device=device)
self._env_spans = wp.empty(self.environment_count(), dtype=int, device=device)
self._env_starts = wp.empty(self.environment_count(), dtype=int, device=device)
```

Leave these attributes as `None` for explicit offsets and single-environment grids.

- [ ] **Step 4: Update offsets in place before packing points**

Add `_refresh_automatic_environment_offsets(points, point_envs, point_mask)`. It must reuse `_initialize_environment_bounds`, `_accumulate_environment_bounds_ijk` or `_accumulate_environment_bounds_world`, `_compute_environment_spans`, `utils.array_scan(..., inclusive=False)`, and `_compute_environment_offsets_from_starts`. Write into the existing `self._env_offsets` array; never replace it.

Call it in `Nanogrid.rebuild()` immediately before `_pack_environment_points`:

```python
if self._automatic_env_offsets:
    self._refresh_automatic_environment_offsets(points, point_envs, point_mask)
```

Explicit offsets retain existing fixed behavior.

- [ ] **Step 5: Run GREEN and neighboring tests**

```bash
uv run --extra dev warp/tests/fem/test_fem_multi_env.py \
  -k 'nanogrid_multi_env_rebuild' -v
```

Expected: moving-offset and existing rebuildable tests PASS.

- [ ] **Step 6: Commit**

```bash
git add warp/_src/fem/geometry/nanogrid.py warp/tests/fem/test_fem_multi_env.py
git commit -m "Keep rebuilt FEM environments isolated"
```

### Task 3: Integrate capture-safe environment partitions test-first

**Files:**
- Modify: `warp/tests/fem/test_fem_multi_env.py`
- Merge input: `675c15b7f9c72088a8bcd36529497965191e3911`

- [ ] **Step 1: Add the failing combined capture regression**

Add `test_nanogrid_multi_env_rebuild_partition_capture` using four points and environment IDs:

```python
points = wp.array([[0, 0, 0], [0, 0, 0], [2, 0, 0], [2, 0, 0]], dtype=wp.vec3i, device=device)
point_envs = wp.array([0, 1, 0, 1], dtype=int, device=device)
point_mask = wp.array([1, 1, 0, 0], dtype=int, device=device)
status = wp.zeros(1, dtype=wp.uint32, device=device)
geo = fem.Nanogrid.from_environment_voxels(
    points,
    point_envs,
    2,
    rebuildable=True,
    max_active_voxels=4,
    max_leaf_nodes=4,
    max_lower_nodes=4,
    max_upper_nodes=4,
    status=status,
    device=device,
)
space = fem.make_polynomial_space(geo, degree=2, element_basis=fem.ElementBasis.SERENDIPITY)
edge_grid = geo.edge_grid
cell_mask = wp.array([1, 1, 0, 0], dtype=int, device=device)
geo_partition = fem.ExplicitGeometryPartition(geo, cell_mask=cell_mask, max_cell_count=4, max_side_count=0)
space_partition = fem.make_space_partition(
    space.topology,
    geometry_partition=geo_partition,
    with_halo=False,
    environment_first=True,
    max_node_count=space.node_count(),
)
```

Warm the same rebuild sequence, record it with `wp.ScopedCapture`, change both masks to all ones, replay, and assert success, stable cell/edge grid IDs, stable partition pointers, 4 active cells, 32 vertices, 48 S2 edges, offsets changing from `[0,20,80]` to `[0,40,80]`, and balanced graph allocations through `_assert_graph_allocations_balanced`.

- [ ] **Step 2: Run RED on the NanoVDB-only checkpoint**

```bash
uv run --extra dev warp/tests/fem/test_fem_multi_env.py \
  -k nanogrid_multi_env_rebuild_partition_capture -v
```

Expected: FAIL when `EnvironmentSpacePartition.rebuild()` performs a device-to-host count read during CUDA capture.

- [ ] **Step 3: Merge the capture-safe partition stack**

```bash
git merge --no-ff --no-edit 675c15b7f9c72088a8bcd36529497965191e3911
```

If `warp/tests/fem/test_fem_multi_env.py` conflicts because of Step 1, retain both branch test additions and the new combined regression. `CHANGELOG.md` and `warp/_src/context.py` should merge automatically.

- [ ] **Step 4: Run GREEN**

```bash
uv run --extra dev warp/tests/fem/test_fem_multi_env.py \
  -k 'nanogrid_multi_env_rebuild_partition_capture or environment_space_partition_capture' -v
```

Expected: all selected tests PASS on eligible CUDA devices.

- [ ] **Step 5: Commit conflict resolution and regression**

```bash
git add CHANGELOG.md warp/_src/context.py warp/tests/fem/test_fem_multi_env.py
git commit -m "Capture rebuilt multi-world FEM partitions"
```

### Task 4: Rebuild native Warp and verify the branch

**Files:**
- Native outputs: `warp/bin/`
- Validation: source tree only; do not commit generated binaries

- [ ] **Step 1: Locate CUDA and build**

```bash
CUDA_PATH="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
test -x "$CUDA_PATH/bin/nvcc"
uv sync --extra dev
uv run build_lib.py --cuda-path="$CUDA_PATH"
```

Expected: native CPU and CUDA libraries build successfully. If `nvcc` is not on `PATH`, locate the installed toolkit under `/usr/local/cuda*` and set `CUDA_PATH` to the directory containing `bin/nvcc` before rerunning the same build command.

- [ ] **Step 2: Run focused tests**

```bash
uv run --extra dev warp/tests/geometry/test_volume_write.py -k volume_rebuild -v
uv run --extra dev warp/tests/fem/test_fem_geometry.py -k nanogrid_rebuild -v
uv run --extra dev warp/tests/fem/test_fem_multi_env.py -k 'nanogrid_multi_env or environment_space_partition' -v
uv run --extra dev warp/tests/deterministic/test_deterministic_graph_capture.py \
  -k graph_capture_deterministic_explicit_stream -v
uv run --extra dev warp/tests/test_apic.py -k borrow_temporary_bypasses_pool -v
```

Expected: all selected tests PASS; capability-based skips name the missing capability.

- [ ] **Step 3: Run quality gates**

```bash
uvx pre-commit run --files warp/_src/fem/geometry/nanogrid.py warp/tests/fem/test_fem_multi_env.py
git diff --check b8596fd51f6f855f32bda2d39d6c6bbc95c26e94..HEAD
git status --short
```

Expected: no lint errors, whitespace errors, generated tracked binaries, or uncommitted files.
