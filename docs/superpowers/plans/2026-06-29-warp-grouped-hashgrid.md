# Warp Grouped HashGrid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in grouped `wp.HashGrid` build and exact-group query that scales colocated multi-world neighbor searches while preserving the existing ungrouped path and three-argument query behavior.

**Architecture:** Keep the dense `cell_starts`/`cell_ends` index and the current 32-bit cell sort unchanged for ungrouped builds. Grouped builds allocate a separate 64-bit key buffer containing `(cell << 32) | (uint32(group) ^ 0x80000000)`, sort original point IDs by that cell-major key, and let the four-argument query binary-search one group's subrange inside each visited cell; the existing three-argument query still traverses each complete cell range. Separate grouped host/device C entry points and a typed 64-bit CUDA radix-sort reserve keep compatibility explicit and make pre-reserved grouped builds capture-safe.

**Tech Stack:** Warp Python API and builtin registration, C++17 host runtime, CUDA/CUB radix sort, Warp kernel code generation, `unittest`, Sphinx/RST, `uv`, and pre-commit.

---

## Working assumptions and file map

Run every implementation command from `/home/maximiliank/Work/warp-main-multiworld-reference`. The approved source baseline is commit `b8596fd51f6f855f32bda2d39d6c6bbc95c26e94` on branch `codex/grouped-hashgrid`; continue on that branch (or an isolated worktree at that commit) while following this plan.

The public contract is exactly:

```python
grid.reserve(point_count, grouped=True)
grid.build(points, radius, groups=groups)

all_groups = wp.hash_grid_query(grid.id, point, radius)
same_group = wp.hash_grid_query(grid.id, point, radius, group)
```

Files to modify:

- `warp/tests/geometry/test_hash_grid.py`: grouped correctness, precision, edge, validation, stream, capture, and legacy regressions. `TestHashGrid` is already imported into `default_suite` by `warp/tests/unittest_suites.py`, so that suite file must not change.
- `warp/_src/types.py:7077-7180`: optional `groups` build argument, grouped reserve selector, and Python-side validation.
- `warp/_src/context.py:6253-6280`: `ctypes` signatures for four new native entry points.
- `warp/_src/builtins.py:8332-8365`: exact-group overload for every coordinate precision.
- `warp/native/hashgrid.h`: optional grouped descriptor storage, composite-key helpers, and exact-group query bounds.
- `warp/native/hashgrid.cpp`: grouped host/device allocation, lifecycle, update dispatch, and CUDA-disabled stubs.
- `warp/native/hashgrid.cu`: CUDA composite-key generation and grouped cell-offset construction.
- `warp/native/warp.h:141-150`: exported grouped reserve/update declarations.
- `warp/native/sort.h`, `warp/native/sort.cu`, `warp/native/sort.cpp`: an internal `uint64_t` radix-sort reserve entry point, including the no-CUDA stub.
- `warp/__init__.pyi`: generated overloads; regenerate this file with `build_docs.py`, never hand-edit it.
- `docs/user_guide/runtime.rst:2131-2199`: grouped-build/query usage and preconditions.
- `CHANGELOG.md`: one `Added` entry for the public feature.
- `asv/benchmarks/grouped_hash_grid.py`: standalone CUDA scaling comparison for filtered HashGrid, grouped HashGrid,
  and grouped BVH.

No new file or dependency is needed.

### Task 1: Add the red grouped-query and validation test surface

**Files:**

- Modify: `warp/tests/geometry/test_hash_grid.py`

- [ ] **Step 1: Add a kernel that observes exact-group and legacy all-group candidates**

Place this beside `count_neighbors` and define the three concrete overloads next to the existing precision overloads:

```python
@wp.kernel
def count_group_candidates(
    grid: wp.uint64,
    radius: Any,
    grid_points: wp.array[Any],
    point_groups: wp.array[int],
    query_points: wp.array[Any],
    query_groups: wp.array[int],
    exact_counts: wp.array[int],
    all_counts: wp.array[int],
    wrong_group_counts: wp.array[int],
):
    tid = wp.tid()
    point = query_points[tid]
    group = query_groups[tid]

    exact_count = int(0)
    wrong_group_count = int(0)
    for index in wp.hash_grid_query(grid, point, radius, group):
        exact_count += 1
        if point_groups[index] != group:
            wrong_group_count += 1

    all_count = int(0)
    for index in wp.hash_grid_query(grid, point, radius):
        all_count += 1

    exact_counts[tid] = exact_count
    all_counts[tid] = all_count
    wrong_group_counts[tid] = wrong_group_count


count_group_candidates_f16 = wp.overload(
    count_group_candidates,
    [
        wp.uint64,
        wp.float16,
        wp.array[wp.vec3h],
        wp.array[int],
        wp.array[wp.vec3h],
        wp.array[int],
        wp.array[int],
        wp.array[int],
        wp.array[int],
    ],
)
count_group_candidates_f32 = wp.overload(
    count_group_candidates,
    [
        wp.uint64,
        wp.float32,
        wp.array[wp.vec3f],
        wp.array[int],
        wp.array[wp.vec3f],
        wp.array[int],
        wp.array[int],
        wp.array[int],
        wp.array[int],
    ],
)
count_group_candidates_f64 = wp.overload(
    count_group_candidates,
    [
        wp.uint64,
        wp.float64,
        wp.array[wp.vec3d],
        wp.array[int],
        wp.array[wp.vec3d],
        wp.array[int],
        wp.array[int],
        wp.array[int],
        wp.array[int],
    ],
)
```

- [ ] **Step 2: Add a reusable launch helper and core semantics test**

Add the following helper and test before the device lists. It covers overlapping groups, absent/sparse/negative/extreme IDs, the three-argument query on a grouped build, changing membership on every build, and a final legacy rebuild with `groups=None`:

```python
def run_group_candidate_query(kernel, grid, radius, points, point_groups, query_points, query_groups, device):
    query_count = query_points.shape[0]
    exact_counts = wp.empty(query_count, dtype=int, device=device)
    all_counts = wp.empty(query_count, dtype=int, device=device)
    wrong_group_counts = wp.empty(query_count, dtype=int, device=device)
    wp.launch(
        kernel,
        dim=query_count,
        inputs=[
            wp.uint64(grid.id),
            radius,
            points,
            point_groups,
            query_points,
            query_groups,
            exact_counts,
            all_counts,
            wrong_group_counts,
        ],
        device=device,
    )
    return exact_counts.numpy(), all_counts.numpy(), wrong_group_counts.numpy()


def test_hashgrid_grouped_queries(test, device):
    points = wp.zeros(8, dtype=wp.vec3, device=device)
    groups = wp.array(
        [-2147483648, -7, -7, 0, 42, 2147483647, 42, 2147483647],
        dtype=wp.int32,
        device=device,
    )
    grid = wp.HashGrid(8, 8, 8, device=device)
    grid.build(points, 1.0, groups=groups)

    exact, all_groups, wrong = run_group_candidate_query(
        count_group_candidates_f32, grid, 0.25, points, groups, points, groups, device
    )
    assert_np_equal(exact, np.array([1, 2, 2, 1, 2, 2, 2, 2], dtype=np.int32))
    assert_np_equal(all_groups, np.full(8, 8, dtype=np.int32))
    assert_np_equal(wrong, np.zeros(8, dtype=np.int32))

    absent = wp.full(8, value=1, dtype=wp.int32, device=device)
    exact, all_groups, wrong = run_group_candidate_query(
        count_group_candidates_f32, grid, 0.25, points, groups, points, absent, device
    )
    assert_np_equal(exact, np.zeros(8, dtype=np.int32))
    assert_np_equal(all_groups, np.full(8, 8, dtype=np.int32))
    assert_np_equal(wrong, np.zeros(8, dtype=np.int32))

    groups.assign([0, 0, 0, 0, 1, 1, 1, 1])
    grid.build(points, 1.0, groups=groups)
    exact, _, wrong = run_group_candidate_query(
        count_group_candidates_f32, grid, 0.25, points, groups, points, groups, device
    )
    assert_np_equal(exact, np.full(8, 4, dtype=np.int32))
    assert_np_equal(wrong, np.zeros(8, dtype=np.int32))

    groups.assign([0, 1, 2, 3, 0, 1, 2, 3])
    grid.build(points, 1.0, groups=groups)
    exact, _, wrong = run_group_candidate_query(
        count_group_candidates_f32, grid, 0.25, points, groups, points, groups, device
    )
    assert_np_equal(exact, np.full(8, 2, dtype=np.int32))
    assert_np_equal(wrong, np.zeros(8, dtype=np.int32))

    legacy_counts = wp.zeros(8, dtype=int, device=device)
    grid.build(points, 1.0)
    wp.launch(count_neighbors_f32, dim=8, inputs=[grid.id, 0.25, points, legacy_counts], device=device)
    assert_np_equal(legacy_counts.numpy(), np.full(8, 8, dtype=np.int32))
```

- [ ] **Step 3: Add multiprecision and empty/single/hash-wrap cases**

Use one test for all scalar types and one for structural edge cases:

```python
def test_hashgrid_grouped_multiprecision(test, device):
    cases = [
        (wp.float16, wp.vec3h, count_group_candidates_f16),
        (wp.float32, wp.vec3f, count_group_candidates_f32),
        (wp.float64, wp.vec3d, count_group_candidates_f64),
    ]
    groups_np = np.array([-3, -3, 11, 11, 2147483647], dtype=np.int32)
    for scalar_dtype, point_dtype, kernel in cases:
        with test.subTest(dtype=scalar_dtype.__name__):
            points = wp.zeros(5, dtype=point_dtype, device=device)
            groups = wp.array(groups_np, dtype=wp.int32, device=device)
            grid = wp.HashGrid(8, 8, 8, device=device, dtype=scalar_dtype)
            grid.build(points, 1.0, groups=groups)
            exact, all_groups, wrong = run_group_candidate_query(
                kernel, grid, scalar_dtype(0.25), points, groups, points, groups, device
            )
            assert_np_equal(exact, np.array([2, 2, 2, 2, 1], dtype=np.int32))
            assert_np_equal(all_groups, np.full(5, 5, dtype=np.int32))
            assert_np_equal(wrong, np.zeros(5, dtype=np.int32))


def test_hashgrid_grouped_edge_cases(test, device):
    grid = wp.HashGrid(4, 4, 4, device=device)
    empty_points = wp.empty(0, dtype=wp.vec3, device=device)
    empty_groups = wp.empty(0, dtype=wp.int32, device=device)
    query_points = wp.zeros(1, dtype=wp.vec3, device=device)
    query_groups = wp.array([9], dtype=wp.int32, device=device)
    grid.build(empty_points, 1.0, groups=empty_groups)
    exact, all_groups, wrong = run_group_candidate_query(
        count_group_candidates_f32,
        grid,
        0.25,
        empty_points,
        empty_groups,
        query_points,
        query_groups,
        device,
    )
    assert_np_equal(exact, np.array([0], dtype=np.int32))
    assert_np_equal(all_groups, np.array([0], dtype=np.int32))
    assert_np_equal(wrong, np.array([0], dtype=np.int32))

    single_points = wp.zeros(1, dtype=wp.vec3, device=device)
    single_groups = wp.array([-2147483648], dtype=wp.int32, device=device)
    grid.build(single_points, 1.0, groups=single_groups)
    exact, all_groups, wrong = run_group_candidate_query(
        count_group_candidates_f32,
        grid,
        0.25,
        single_points,
        single_groups,
        single_points,
        single_groups,
        device,
    )
    assert_np_equal(exact, np.array([1], dtype=np.int32))
    assert_np_equal(all_groups, np.array([1], dtype=np.int32))
    assert_np_equal(wrong, np.array([0], dtype=np.int32))

    wrapped_points = wp.array([[-0.25, 0.0, 0.0], [3.75, 0.0, 0.0]], dtype=wp.vec3, device=device)
    wrapped_groups = wp.array([-2147483648, 2147483647], dtype=wp.int32, device=device)
    grid.build(wrapped_points, 1.0, groups=wrapped_groups)
    exact, all_groups, wrong = run_group_candidate_query(
        count_group_candidates_f32,
        grid,
        0.25,
        wrapped_points,
        wrapped_groups,
        wrapped_points,
        wrapped_groups,
        device,
    )
    assert_np_equal(exact, np.array([1, 1], dtype=np.int32))
    assert_np_equal(all_groups, np.array([2, 2], dtype=np.int32))
    assert_np_equal(wrong, np.array([0, 0], dtype=np.int32))
```

- [ ] **Step 4: Add exact validation assertions**

```python
def test_hashgrid_group_validation(test, device):
    grid = wp.HashGrid(4, 4, 4, device=device)
    points = wp.zeros(4, dtype=wp.vec3, device=device)

    with test.assertRaisesRegex(TypeError, "Hash grid groups should have type int32, got float32"):
        grid.build(points, 1.0, groups=wp.zeros(4, dtype=wp.float32, device=device))
    with test.assertRaisesRegex(ValueError, "Hash grid groups must be one-dimensional, got 2 dimensions"):
        grid.build(points, 1.0, groups=wp.zeros((2, 2), dtype=wp.int32, device=device))

    group_buffer = wp.zeros(8, dtype=wp.int32, device=device)
    with test.assertRaisesRegex(ValueError, "Hash grid groups must be contiguous"):
        grid.build(points, 1.0, groups=group_buffer[::2])
    with test.assertRaisesRegex(ValueError, "Hash grid groups must have 4 entries, got 3"):
        grid.build(points, 1.0, groups=wp.zeros(3, dtype=wp.int32, device=device))

    other_devices = [candidate for candidate in devices if candidate != device]
    if other_devices:
        other_groups = wp.zeros(4, dtype=wp.int32, device=other_devices[0])
        with test.assertRaisesRegex(RuntimeError, "Hash grid groups must live on the same device as points"):
            grid.build(points, 1.0, groups=other_groups)
```

Register all four functions with `devices=devices` at the bottom of the module.

- [ ] **Step 5: Run the first test and verify the public API is red**

Run:

```bash
uv run warp/tests/geometry/test_hash_grid.py TestHashGrid.test_hashgrid_grouped_queries_cpu -v
```

Expected: `ERROR` at the first grouped build with `TypeError: HashGrid.build() got an unexpected keyword argument 'groups'`.

- [ ] **Step 6: Commit the observed-red tests**

```bash
git add warp/tests/geometry/test_hash_grid.py
git commit -s -m "Test grouped hash grid queries"
```

### Task 2: Implement grouped storage, builds, queries, and Python dispatch

**Files:**

- Modify: `warp/native/hashgrid.h`
- Modify: `warp/native/hashgrid.cpp`
- Modify: `warp/native/hashgrid.cu`
- Modify: `warp/native/warp.h`
- Modify: `warp/_src/context.py`
- Modify: `warp/_src/types.py`
- Modify: `warp/_src/builtins.py`

- [ ] **Step 1: Add optional grouped state and cell-major key helpers**

In `HashGrid_t`, keep `point_cells` and `point_ids` as the first two fields and add:

```cpp
uint64_t* point_cell_group_keys = nullptr;
int max_grouped_points = 0;
int is_grouped = 0;
```

Add these helpers to `hashgrid.h`:

```cpp
CUDA_CALLABLE inline uint32_t hash_grid_normalize_group(int group)
{
    return static_cast<uint32_t>(group) ^ 0x80000000u;
}

CUDA_CALLABLE inline uint64_t hash_grid_cell_group_key(int cell, int group)
{
    return (static_cast<uint64_t>(static_cast<uint32_t>(cell)) << 32) | hash_grid_normalize_group(group);
}

CUDA_CALLABLE inline int hash_grid_key_cell(uint64_t key)
{
    return static_cast<int>(key >> 32);
}

CUDA_CALLABLE inline int hash_grid_lower_bound(const uint64_t* keys, int first, int last, uint64_t value)
{
    while (first < last) {
        const int mid = first + (last - first) / 2;
        if (keys[mid] < value)
            first = mid + 1;
        else
            last = mid;
    }
    return first;
}

CUDA_CALLABLE inline int hash_grid_upper_bound(const uint64_t* keys, int first, int last, uint64_t value)
{
    while (first < last) {
        const int mid = first + (last - first) / 2;
        if (keys[mid] <= value)
            first = mid + 1;
        else
            last = mid;
    }
    return first;
}
```

- [ ] **Step 2: Extend the existing query type without creating a second public query type**

Add `uint32_t group_key` and `int has_group` to `hash_grid_query_t`, initialize both to zero, and route every cell transition through this helper:

```cpp
template <typename Type> CUDA_CALLABLE inline void hash_grid_query_set_cell(hash_grid_query_t<Type>& query)
{
    const int cell = hash_grid_index(query.grid, query.x, query.y, query.z);
    const int cell_start = query.grid.cell_starts[cell];
    const int cell_end = query.grid.cell_ends[cell];

    if (!query.has_group) {
        query.cell_index = cell_start;
        query.cell_end = cell_end;
        return;
    }

    const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(cell)) << 32) | query.group_key;
    query.cell_index = hash_grid_lower_bound(query.grid.point_cell_group_keys, cell_start, cell_end, key);
    query.cell_end = hash_grid_upper_bound(query.grid.point_cell_group_keys, query.cell_index, cell_end, key);
}
```

The current three-argument `hash_grid_query()` retains all existing coordinate-range setup, sets `has_group = 0`, and calls `hash_grid_query_set_cell(query)` for the initial cell. Add the exact overload:

```cpp
template <typename Type>
CUDA_CALLABLE inline hash_grid_query_t<Type> hash_grid_query(uint64_t id, vec_t<3, Type> pos, Type radius, int group)
{
    hash_grid_query_t<Type> query = hash_grid_query(id, pos, radius);
    query.has_group = 1;
    query.group_key = hash_grid_normalize_group(group);

    assert(query.grid.is_grouped && "Exact-group hash grid queries require a grouped build");
    if (!query.grid.is_grouped) {
        query.z = query.z_end + 1;
        query.cell_index = 0;
        query.cell_end = 0;
        return query;
    }

    hash_grid_query_set_cell(query);
    return query;
}
```

In `hash_grid_query_next()`, return `false` when `has_group && !grid.is_grouped`, and replace the existing two assignments after each cell transition with `hash_grid_query_set_cell(query)`. This yields an empty release-mode query and the required native debug assertion on misuse.

- [ ] **Step 3: Add grouped host allocation and update functions**

Free `point_cell_group_keys` in host/device destroy functions. Leave `hash_grid_reserve_host_impl()` and `hash_grid_update_host_impl()` on their current 32-bit arrays and sort; only set `grid->is_grouped = 0` before the legacy host rebuild.

Add grouped host reserve/update implementations with the same 1.5x capacity policy:

```cpp
template <typename Type> void hash_grid_reserve_grouped_host_impl(uint64_t id, int num_points)
{
    static const char* tag = "(native:hashgrid)";
    hash_grid_reserve_host_impl<Type>(id, num_points);
    HashGrid_t<Type>* grid = reinterpret_cast<HashGrid_t<Type>*>(id);
    if (num_points > grid->max_grouped_points) {
        wp_free_host(grid->point_cell_group_keys);
        const int num_to_alloc = num_points * 3 / 2;
        grid->point_cell_group_keys
            = static_cast<uint64_t*>(wp_alloc_host(2 * num_to_alloc * sizeof(uint64_t), tag));
        grid->max_grouped_points = num_to_alloc;
    }
}

template <typename Type>
void hash_grid_update_grouped_host_impl(
    uint64_t id,
    Type cell_width,
    const wp::array_t<vec_t<3, Type>>* points,
    const wp::array_t<int>* groups
)
{
    HashGrid_t<Type>* grid = reinterpret_cast<HashGrid_t<Type>*>(id);
    const int num_points = points->shape[0];
    hash_grid_reserve_grouped_host_impl<Type>(id, num_points);
    grid->num_points = num_points;
    grid->cell_width = cell_width;
    grid->cell_width_inv = Type(1) / cell_width;
    grid->is_grouped = 1;

    for (int i = 0; i < num_points; ++i) {
        const int cell = hash_grid_index(*grid, wp::index(*points, i));
        grid->point_cell_group_keys[i] = hash_grid_cell_group_key(cell, wp::index(*groups, i));
        grid->point_ids[i] = i;
    }
    radix_sort_pairs_host(grid->point_cell_group_keys, grid->point_ids, num_points);

    const int num_cells = grid->dim_x * grid->dim_y * grid->dim_z;
    memset(grid->cell_starts, 0, sizeof(int) * num_cells);
    memset(grid->cell_ends, 0, sizeof(int) * num_cells);
    for (int i = 0; i < num_points; ++i) {
        const int cell = hash_grid_key_cell(grid->point_cell_group_keys[i]);
        const int previous = i == 0 ? cell : hash_grid_key_cell(grid->point_cell_group_keys[i - 1]);
        if (i == 0 || cell != previous) {
            grid->cell_starts[cell] = i;
            if (i > 0)
                grid->cell_ends[previous] = i;
        }
        if (i == num_points - 1)
            grid->cell_ends[cell] = i + 1;
    }
}
```

Before dereferencing, add the same defensive `id`, rank-one, equal-length checks used by the legacy native function. Python remains the source of user-facing errors.

- [ ] **Step 4: Add the CUDA grouped rebuild**

Add a grouped key kernel and a grouped rebuild alongside the existing 32-bit functions in `hashgrid.cu`:

```cpp
template <typename Type>
__global__ void compute_cell_group_keys(
    HashGrid_t<Type> grid,
    wp::array_t<vec_t<3, Type>> points,
    wp::array_t<int> groups
)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < points.shape[0]) {
        const int cell = hash_grid_index(grid, wp::index(points, tid));
        grid.point_cell_group_keys[tid] = hash_grid_cell_group_key(cell, wp::index(groups, tid));
        grid.point_ids[tid] = tid;
    }
}

__global__ void compute_grouped_cell_offsets(
    int* cell_starts,
    int* cell_ends,
    const uint64_t* keys,
    int num_points
)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_points) {
        const int cell = hash_grid_key_cell(keys[tid]);
        const int previous = tid == 0 ? cell : hash_grid_key_cell(keys[tid - 1]);
        if (tid == 0 || cell != previous) {
            cell_starts[cell] = tid;
            if (tid > 0)
                cell_ends[previous] = tid;
        }
        if (tid == num_points - 1)
            cell_ends[cell] = tid + 1;
    }
}
```

`hash_grid_rebuild_grouped_device<Type>()` launches that key kernel, calls
`radix_sort_pairs_device(WP_CURRENT_CONTEXT, grid.point_cell_group_keys, grid.point_ids, num_points)`, zeros the dense
cell ranges, then launches `compute_grouped_cell_offsets`. Explicitly instantiate it for `half`, `float`, and `double`,
including empty stubs under `!WP_ENABLE_CUDA`.

- [ ] **Step 5: Add grouped device reserve/update and exported entry points**

`hash_grid_reserve_grouped_device_impl()` must first call `hash_grid_reserve_device_impl()` for the shared `point_ids` and legacy capacity, allocate `2 * num_to_alloc` `uint64_t` keys when grouped capacity grows, copy the descriptor to the device, and update the host descriptor map. Do not add the 64-bit sort-temp reserve yet; Task 4 adds it after its capture regression is red.

`hash_grid_update_grouped_device_impl()` follows the existing static-descriptor capture pattern, sets `is_grouped = 1`, calls `hash_grid_rebuild_grouped_device()`, and copies/records the descriptor. The legacy device update sets `is_grouped = 0` and otherwise remains unchanged.

Declare and implement these exact C entry points, dispatching all three `HashGridTypeId` cases:

```cpp
WP_API void wp_hash_grid_update_grouped_host(
    uint64_t id, int type, double cell_width, const void* points, const void* groups
);
WP_API void wp_hash_grid_reserve_grouped_host(uint64_t id, int type, int num_points);
WP_API void wp_hash_grid_update_grouped_device(
    uint64_t id, int type, double cell_width, const void* points, const void* groups
);
WP_API void wp_hash_grid_reserve_grouped_device(uint64_t id, int type, int num_points);
```

Each update case casts `points` to its matching `array_t<vec3*>` and `groups` to `const array_t<int>*`; each reserve case calls the matching templated grouped reserve.

- [ ] **Step 6: Bind and select the grouped entry points in Python**

Register both grouped update signatures as `[c_uint64, c_int, c_double, c_void_p, c_void_p]` and both grouped reserve signatures as `[c_uint64, c_int, c_int]` in `Runtime.__init__`.

Change the two public methods to:

```python
def build(self, points, radius, groups=None):
    if not types_equal(points.dtype, self._vec_type):
        raise TypeError(f"Hash grid points should have type {self._vec_type.__name__}, got {points.dtype}")
    if radius <= 0.0:
        raise ValueError(f"Hash grid cell width must be positive, got {radius}")

    if groups is not None:
        if groups.dtype != int32:
            raise TypeError(f"Hash grid groups should have type int32, got {groups.dtype}")
        if groups.ndim != 1:
            raise ValueError(f"Hash grid groups must be one-dimensional, got {groups.ndim} dimensions")
        if not groups.is_contiguous:
            raise ValueError("Hash grid groups must be contiguous")
        if groups.device != points.device:
            raise RuntimeError("Hash grid groups must live on the same device as points")
        if groups.size != points.size:
            raise ValueError(f"Hash grid groups must have {points.size} entries, got {groups.size}")

    if points.ndim > 1:
        points = points.contiguous().flatten()

    if groups is None:
        self._native_func("update")(self.id, self._type_id, radius, ctypes.byref(points.__ctype__()))
    else:
        self._native_func("update_grouped")(
            self.id,
            self._type_id,
            radius,
            ctypes.byref(points.__ctype__()),
            ctypes.byref(groups.__ctype__()),
        )
    self.reserved = True

def reserve(self, num_points, grouped=False):
    action = "reserve_grouped" if grouped else "reserve"
    self._native_func(action)(self.id, self._type_id, num_points)
    self.reserved = True
```

The local `groups` argument remains strongly referenced through the native call. Do not store group IDs in the Python object: the sorted composite keys are the persistent build result, and membership may change on every build.

- [ ] **Step 7: Register the four-argument builtin for each precision**

Inside `_add_hash_grid_query_builtins()`, add:

```python
add_builtin(
    "hash_grid_query",
    input_types={"id": uint64, "point": vec_type, "max_dist": scalar_type, "group": int},
    value_type=query_type,
    group="Geometry",
    doc=f"""Construct an exact-group point query against a :class:`warp.HashGrid`{doc_suffix}.

    The grid must first be built with ``groups``. Only points with the requested signed 32-bit ``group`` are returned.""",
    export=False,
    is_differentiable=False,
)
```

- [ ] **Step 8: Rebuild native libraries and run all non-capture grouped tests**

Run:

```bash
uv run build_lib.py
uv run warp/tests/geometry/test_hash_grid.py -v
```

Expected: native build exits `0`; all `TestHashGrid` CPU/CUDA variants pass, including every new correctness/validation test and every existing ungrouped test.

- [ ] **Step 9: Commit the core implementation**

```bash
git add warp/native/hashgrid.h warp/native/hashgrid.cpp warp/native/hashgrid.cu warp/native/warp.h \
  warp/_src/context.py warp/_src/types.py warp/_src/builtins.py
git commit -s -m "Add grouped hash grid queries"
```

### Task 3: Add multi-stream and CUDA graph capture regressions

**Files:**

- Modify: `warp/tests/geometry/test_hash_grid.py`

- [ ] **Step 1: Add a grouped multi-stream test**

Follow the existing stream test but use four streams, a fixed overlapping eight-point set, per-stream grids, grouped reserves/builds, and `count_group_candidates_f32`. Schedule every stream before reading results; assert exact counts are `4`, all-group counts are `8`, and wrong-group counts are `0` for all streams.

```python
def test_hashgrid_grouped_multiple_streams(test, device):
    with wp.ScopedDevice(device):
        points = wp.zeros(8, dtype=wp.vec3)
        groups = wp.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=wp.int32)
        streams = [wp.Stream(device=device) for _ in range(4)]
        results = []
        for stream in streams:
            with wp.ScopedStream(stream):
                grid = wp.HashGrid(8, 8, 8)
                grid.reserve(8, grouped=True)
                grid.build(points, 1.0, groups=groups)
                exact = wp.empty(8, dtype=int)
                all_groups = wp.empty(8, dtype=int)
                wrong = wp.empty(8, dtype=int)
                wp.launch(
                    count_group_candidates_f32,
                    dim=8,
                    inputs=[grid.id, 0.25, points, groups, points, groups, exact, all_groups, wrong],
                )
                results.append((grid, exact, all_groups, wrong))

        for _, exact, all_groups, wrong in results:
            assert_np_equal(exact.numpy(), np.full(8, 4, dtype=np.int32))
            assert_np_equal(all_groups.numpy(), np.full(8, 8, dtype=np.int32))
            assert_np_equal(wrong.numpy(), np.zeros(8, dtype=np.int32))
```

- [ ] **Step 2: Add a capture/replay test whose group membership changes between replays**

```python
def test_hashgrid_grouped_graph_capture(test, device):
    with wp.ScopedDevice(device):
        points = wp.zeros(8, dtype=wp.vec3)
        groups = wp.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=wp.int32)
        exact = wp.empty(8, dtype=int)
        all_groups = wp.empty(8, dtype=int)
        wrong = wp.empty(8, dtype=int)
        grid = wp.HashGrid(8, 8, 8)
        grid.reserve(8, grouped=True)

        wp.load_module(device=device)
        with wp.ScopedCapture(device=device, force_module_load=False) as capture:
            grid.build(points, 1.0, groups=groups)
            wp.launch(
                count_group_candidates_f32,
                dim=8,
                inputs=[grid.id, 0.25, points, groups, points, groups, exact, all_groups, wrong],
            )

        wp.capture_launch(capture.graph)
        assert_np_equal(exact.numpy(), np.full(8, 4, dtype=np.int32))
        assert_np_equal(all_groups.numpy(), np.full(8, 8, dtype=np.int32))
        assert_np_equal(wrong.numpy(), np.zeros(8, dtype=np.int32))

        groups.assign([0, 1, 2, 3, 0, 1, 2, 3])
        exact.zero_()
        all_groups.zero_()
        wrong.zero_()
        wp.capture_launch(capture.graph)
        assert_np_equal(exact.numpy(), np.full(8, 2, dtype=np.int32))
        assert_np_equal(all_groups.numpy(), np.full(8, 8, dtype=np.int32))
        assert_np_equal(wrong.numpy(), np.zeros(8, dtype=np.int32))
```

Register the first with `cuda_devices` and the second with `cuda_devices`.

- [ ] **Step 3: Verify stream isolation passes and capture is red**

Run:

```bash
uv run warp/tests/geometry/test_hash_grid.py \
  TestHashGrid.test_hashgrid_grouped_multiple_streams_cuda_0 \
  TestHashGrid.test_hashgrid_grouped_graph_capture_cuda_0 -v
```

Expected: multi-stream test passes; graph capture errors during the grouped radix sort because its 64-bit CUB temporary storage was not pre-reserved before capture.

- [ ] **Step 4: Commit the observed-red capture regression**

```bash
git add warp/tests/geometry/test_hash_grid.py
git commit -s -m "Test captured grouped hash grid builds"
```

### Task 4: Pre-reserve the typed 64-bit radix-sort temporary buffer

**Files:**

- Modify: `warp/native/sort.h`
- Modify: `warp/native/sort.cu`
- Modify: `warp/native/sort.cpp`
- Modify: `warp/native/hashgrid.cpp`

- [ ] **Step 1: Add a typed internal reserve function**

Declare in `sort.h`:

```cpp
void radix_sort_reserve_uint64(void* context, int n);
```

Implement in `sort.cu` using the same stream-keyed cache as all radix sorts:

```cpp
void radix_sort_reserve_uint64(void* context, int n)
{
    radix_sort_reserve_internal<uint64_t, int>(context, n, nullptr, nullptr, 0, 64);
}
```

Add the CUDA-disabled definition beside the existing reserve stub in `sort.cpp`:

```cpp
void radix_sort_reserve_uint64(void* context, int n) { }
```

- [ ] **Step 2: Call it only from grouped device reserve**

Immediately after allocating a larger grouped key buffer in `hash_grid_reserve_grouped_device_impl()`, add:

```cpp
radix_sort_reserve_uint64(WP_CURRENT_CONTEXT, num_to_alloc);
```

Do not alter `radix_sort_reserve()` or the legacy `hash_grid_reserve_device_impl()` call. This preserves the exact 32-bit ungrouped reservation path.

- [ ] **Step 3: Rebuild and verify capture plus stream parity**

Run:

```bash
uv run build_lib.py
uv run warp/tests/geometry/test_hash_grid.py \
  TestHashGrid.test_hashgrid_grouped_multiple_streams_cuda_0 \
  TestHashGrid.test_hashgrid_grouped_graph_capture_cuda_0 -v
```

Expected: both tests pass; the graph replays with counts `4` for the first membership and `2` after membership changes.

- [ ] **Step 4: Commit capture-safe reservation**

```bash
git add warp/native/sort.h warp/native/sort.cu warp/native/sort.cpp warp/native/hashgrid.cpp
git commit -s -m "Reserve grouped hash grid sort storage"
```

### Task 5: Document and generate the public API surface

**Files:**

- Modify: `warp/_src/types.py`
- Modify: `warp/_src/builtins.py`
- Modify: `docs/user_guide/runtime.rst`
- Modify: `CHANGELOG.md`
- Regenerate: `warp/__init__.pyi`

- [ ] **Step 1: Complete class/method and builtin docs**

Document that `groups` is an optional contiguous rank-one `wp.array[wp.int32]` on the points device, IDs may use the full signed range, `groups=None` keeps the legacy build, `grouped=True` reserves grouped persistent and sort storage, three arguments query every group, and four arguments require the most recent build to be grouped. State that a grouped query against an ungrouped build is invalid even though release native code returns an empty iterator.

- [ ] **Step 2: Add a runnable grouped user-guide example**

Extend the Hash Grids section with this focused pattern:

```rst
Independent point groups can share the same coordinates. Pass a contiguous
``wp.int32`` group array while building, then pass the desired group to the
four-argument query overload:

.. code-block:: python

    groups = wp.array([0, 0, 1], dtype=wp.int32, device="cuda:0")
    grid.reserve(p.shape[0], grouped=True)
    grid.build(points=p, radius=r, groups=groups)

    @wp.kernel
    def count_group_neighbors(
        grid: wp.uint64,
        points: wp.array[wp.vec3],
        groups: wp.array[int],
        radius: float,
        counts: wp.array[int],
    ):
        tid = wp.tid()
        count = int(0)
        for index in wp.hash_grid_query(grid, points[tid], radius, groups[tid]):
            if index != tid and wp.length(points[index] - points[tid]) <= radius:
                count += 1
        counts[tid] = count
```

Follow it with one sentence that the three-argument overload still returns candidates from all groups on a grouped build.

- [ ] **Step 3: Add the changelog entry**

Append under `Unreleased` → `Added`:

```markdown
- Add optional `wp.HashGrid` grouping with capture-safe grouped reservation and exact-group neighbor queries while
  preserving all-group and ungrouped behavior.
```

- [ ] **Step 4: Regenerate stubs and build docs**

Run:

```bash
uv run --extra docs build_docs.py --warnings-as-errors 2>&1 | tee /tmp/build_docs.log
```

Expected: exit `0`, no Sphinx warning from the grouped HashGrid additions, and `warp/__init__.pyi` contains six `hash_grid_query` overloads: three existing precision overloads and three corresponding overloads with `group: int32`.

- [ ] **Step 5: Commit docs and generated output**

```bash
git add warp/_src/types.py warp/_src/builtins.py docs/user_guide/runtime.rst CHANGELOG.md warp/__init__.pyi
git commit -s -m "Document grouped hash grids"
```

### Task 6: Add the standalone multi-world scaling benchmark

**Files:**

- Create: `asv/benchmarks/grouped_hash_grid.py`

- [ ] **Step 1: Add the ASV fixture and three query kernels**

Follow the existing `asv/benchmarks/spatial_query.py` convention. Define `POINTS_PER_WORLD = 64`, `RADIUS = 0.51`,
and a deterministic 4×4×4 lattice with spacing `0.5`; repeat the identical lattice for every world so all worlds are
physically colocated. Add these kernels:

```python
@wp.kernel
def query_filtered_hash_grid(
    grid: wp.uint64,
    points: wp.array[wp.vec3],
    groups: wp.array[int],
    radius: float,
    visits: wp.array[int],
    accepted: wp.array[int],
):
    tid = wp.tid()
    visit_count = int(0)
    accepted_count = int(0)
    for index in wp.hash_grid_query(grid, points[tid], radius):
        visit_count += 1
        if groups[index] == groups[tid] and wp.length(points[index] - points[tid]) <= radius:
            accepted_count += 1
    visits[tid] = visit_count
    accepted[tid] = accepted_count


@wp.kernel
def query_grouped_hash_grid(
    grid: wp.uint64,
    points: wp.array[wp.vec3],
    groups: wp.array[int],
    radius: float,
    visits: wp.array[int],
    accepted: wp.array[int],
):
    tid = wp.tid()
    visit_count = int(0)
    accepted_count = int(0)
    for index in wp.hash_grid_query(grid, points[tid], radius, groups[tid]):
        visit_count += 1
        if wp.length(points[index] - points[tid]) <= radius:
            accepted_count += 1
    visits[tid] = visit_count
    accepted[tid] = accepted_count


@wp.kernel
def query_grouped_bvh(
    bvh: wp.uint64,
    points: wp.array[wp.vec3],
    groups: wp.array[int],
    radius: float,
    visits: wp.array[int],
    accepted: wp.array[int],
):
    tid = wp.tid()
    point = points[tid]
    extent = wp.vec3(radius, radius, radius)
    root = wp.bvh_get_group_root(bvh, groups[tid])
    query = wp.bvh_query_aabb(bvh, point - extent, point + extent, root)
    visit_count = int(0)
    accepted_count = int(0)
    index = int(0)
    while wp.bvh_query_next(query, index):
        visit_count += 1
        if wp.length(points[index] - point) <= radius:
            accepted_count += 1
    visits[tid] = visit_count
    accepted[tid] = accepted_count
```

- [ ] **Step 2: Add one parameterized ASV class with all required metrics**

Define `GroupedHashGridScaling` with:

```python
class GroupedHashGridScaling:
    params = ([1, 8, 64, 256], ["filtered_hash_grid", "grouped_hash_grid", "grouped_bvh"])
    param_names = ["world_count", "strategy"]
    number = 3
    repeat = 5
    timeout = 180
```

Its `setup(world_count, strategy)` must create `64 * world_count` CUDA points and `wp.int32` group IDs, point AABBs
(`lowers == uppers == points`) for the BVH, per-point `visits`/`accepted` outputs, and exactly one selected structure:

- `filtered_hash_grid`: `HashGrid.reserve(n)`, then `build(points, RADIUS)`.
- `grouped_hash_grid`: `HashGrid.reserve(n, grouped=True)`, then `build(points, RADIUS, groups=groups)`.
- `grouped_bvh`: `Bvh(lowers, uppers, groups=groups, constructor="lbvh")`.

Provide `_rebuild()` (`grid.build(...)` or `bvh.rebuild("lbvh")`) and `_query()` dispatch helpers. Warm both once and
synchronize before ASV timing. Compute the independent NumPy single-world neighbor counts and assert that
`accepted.numpy().reshape(world_count, 64)` equals that same baseline in every row; this keeps all three strategies
numerically comparable.

Expose exactly these ASV methods:

```python
def time_build(self, world_count, strategy):
    self._rebuild()
    wp.synchronize_device(self.device)

def time_query(self, world_count, strategy):
    self._query()
    wp.synchronize_device(self.device)

def track_candidate_visits(self, world_count, strategy):
    self.visits.zero_()
    self._query()
    return int(self.visits.numpy().sum())

track_candidate_visits.unit = "candidates"

def track_memory_delta_bytes(self, world_count, strategy):
    return self.memory_delta_bytes

track_memory_delta_bytes.unit = "bytes"

def track_capture_allocations(self, world_count, strategy):
    return self.capture_allocations

track_capture_allocations.unit = "allocations"
```

In `setup()`, derive `memory_delta_bytes` with `wp.ScopedMemoryTracker(print=False)`: call
`warp._src.context.runtime.core.wp_alloc_tracker_reset()` immediately before creating/reserving/building the selected
structure, synchronize, then read
`warp._src.context.runtime.core.wp_alloc_tracker_get_current_bytes()`. After the warm-up, clear the tracker again and
capture one `_rebuild()` plus `_query()` inside `wp.ScopedCapture(force_module_load=False)`; read
the difference in `wp_alloc_tracker_get_total_alloc_count()` before and after capture into `capture_allocations`. This
exercises capture-safe HashGrid reserve/build
and the existing allocation-free grouped-BVH LBVH rebuild on the same metric. Keep timing assertions out of the file.

- [ ] **Step 3: Run the standalone benchmark and inspect scaling**

Run after committing the benchmark file so ASV can build the commit:

```bash
uvx --python 3.12 asv run -e --launch-method spawn -b GroupedHashGridScaling HEAD^!
```

Expected: ASV reports all 60 combinations (4 world counts × 3 strategies × 5 metrics); every capture-allocation result
is `0`; memory deltas are positive; grouped HashGrid and grouped BVH candidate visits grow approximately linearly with
world count, while filtered HashGrid visits grow approximately quadratically because every colocated world is visited.
Record the emitted ASV result JSON with the implementation review; do not add timing thresholds to unit tests.

- [ ] **Step 4: Commit the benchmark**

```bash
git add asv/benchmarks/grouped_hash_grid.py
git commit -s -m "Benchmark grouped hash grid scaling"
```

### Task 7: Run final Warp verification

**Files:**

- Verify only; no planned edits.

- [ ] **Step 1: Rebuild from the final native sources**

```bash
uv run build_lib.py
```

Expected: exit `0` with host and CUDA libraries rebuilt.

- [ ] **Step 2: Run the direct module and registered suite**

```bash
uv run warp/tests/geometry/test_hash_grid.py -v
uv run --extra dev -m warp.tests -s autodetect -k TestHashGrid
```

Expected: all fixed-device and generated CPU/CUDA `TestHashGrid` cases pass; no test is missing from the default suite.

- [ ] **Step 3: Run the full regression suite**

```bash
uv run --extra dev -m warp.tests -s autodetect
```

Expected: suite exits `0` with no regression in legacy hash-grid users.

- [ ] **Step 4: Run formatting, generated-file, and whitespace checks**

```bash
uvx pre-commit run --files \
  warp/tests/geometry/test_hash_grid.py warp/_src/types.py warp/_src/context.py warp/_src/builtins.py \
  warp/native/hashgrid.h warp/native/hashgrid.cpp warp/native/hashgrid.cu warp/native/warp.h \
  warp/native/sort.h warp/native/sort.cpp warp/native/sort.cu warp/__init__.pyi \
  docs/user_guide/runtime.rst CHANGELOG.md asv/benchmarks/grouped_hash_grid.py
git diff --check
git status --short
```

Expected: every hook passes, `git diff --check` emits nothing, and `git status --short` is empty. If a formatter changes a file, stage it and amend the commit that introduced that file before rerunning this step.

## Completion invariants

- `groups=None`, `reserve(num_points)`, the 32-bit key arrays, and the existing three-argument query retain their previous behavior and performance path.
- Grouped storage is allocated only by grouped reserve/build; memory stays `O(points + cells)` and no `groups * cells` table exists.
- The key is cell-major and group-minor for every signed `int32`, including both extremes.
- A grouped build supports exact-group and all-group queries from the same descriptor on host and CUDA for float16/float32/float64.
- A missing group and an empty grouped build produce empty exact queries.
- Group membership can change on every build, concurrent streams do not share temporary storage, and a pre-reserved grouped build/query captures and replays without allocation.
- Native debug builds assert on a grouped query after an ungrouped build; release builds return an empty iterator, and docs identify the call as invalid.
