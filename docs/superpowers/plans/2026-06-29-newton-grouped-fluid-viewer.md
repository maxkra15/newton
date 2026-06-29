# Newton Grouped Fluid and Viewer Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Newton's XPBD/SPH particle paths use Warp's grouped `HashGrid` fast path for eligible colocated worlds, preserve the correctness-filtered fallback, and render surface/diffuse particles with viewer world offsets and visibility filtering.

**Architecture:** This is the third, post-Warp subproject in the design. It assumes the correctness plan has already added `particle_world` checks to every neighbor kernel and the Warp plan has added grouped reserve/build/query APIs. A shared `SolverBase` lifecycle selects grouped mode only for multi-world models whose particles all have non-negative world IDs; XPBD and SPH retain the same kernels and defensive world filters while selecting a three- or four-argument query. ViewerGL carries optional world IDs into its existing surface pack kernel and diffuse host-transfer path, so surface rendering stays CUDA-to-OpenGL device-native while diffuse sorting remains deliberately CPU-based.

**Tech Stack:** Python 3.10+, Warp grouped `HashGrid`, Newton XPBD/SPH, CUDA graphs, OpenGL/CUDA interop, NumPy, `unittest`, ASV, Sphinx/reStructuredText.

---

## Scope and prerequisites

Implement this plan only after these plans have landed, in order:

1. `docs/superpowers/plans/2026-06-29-newton-particle-world-filtering.md`
2. `docs/superpowers/plans/2026-06-29-warp-grouped-hashgrid.md`

The correctness prerequisite defines `test_world_pair()` and adds the defensive world arrays/filters to all XPBD and SPH neighbor kernels. The Warp prerequisite defines these exact APIs:

```python
grid.reserve(point_count, grouped=True)
grid.build(points, radius, groups=groups)
query = wp.hash_grid_query(grid.id, point, radius, group)
```

Do not remove any correctness filter in this plan. Do not add segmented particle reordering. Do not move diffuse compaction or depth sorting to the GPU; `DiffuseBatch.update()` continues its existing host transfer, filtering, and `numpy.argsort()` path.

The Newton dependency requirement must not name an unreleased Warp version. Develop and test against the editable companion checkout; update Newton's minimum `warp-lang` version only in the release change that publishes the grouped API.

### File map

- Modify `newton/_src/solvers/solver.py`: own grouped-grid eligibility, reserve, and build selection shared by XPBD/SPH.
- Modify `newton/_src/solvers/xpbd/solver_xpbd.py`: use the shared lifecycle and pass grouped mode to all XPBD/shared neighbor launches.
- Modify `newton/_src/solvers/xpbd/kernels.py`: select grouped particle-contact queries.
- Modify `newton/_src/solvers/xpbd/fluid_kernels.py`: select grouped lambda, delta, viscosity, and vorticity queries.
- Modify `newton/_src/solvers/sph/solver_sph.py`: use grouped builds and pass grouped mode to SPH/PBF/render/diffuse launches.
- Modify `newton/_src/solvers/sph/kernels.py`: select grouped SPH, PBF, render, smoothing, spawn, and diffuse-advection queries.
- Modify `newton/_src/viewer/viewer.py`: add optional `worlds` API parameters and preserve world IDs during active-particle compaction.
- Modify `newton/_src/viewer/viewer_gl.py`: forward world IDs, offsets, and the visible-world mask to fluid batches.
- Modify `newton/_src/viewer/gl/fluid.py`: offset/filter surface vertices on device and diffuse positions after the existing host transfer.
- Modify `newton/tests/test_solver_xpbd_fluid.py`, `newton/tests/test_solver_sph.py`, and `newton/tests/test_viewer_fluid.py`: eligibility, fallback, grouped CUDA graph, packing, and routing regressions.
- Create `asv/benchmarks/simulation/bench_fluid_multiworld.py`: end-to-end filtered/grouped CUDA-graph timings for 1/8/64/256 worlds.
- Modify the ten XPBD/SPH fluid examples listed in Task 7: pass surface and diffuse world IDs.
- Modify `docs/concepts/worlds.rst`, `docs/guide/visualization.rst`, and `CHANGELOG.md`: document solver selection and viewer behavior.

### Task 1: Establish the grouped-grid lifecycle and eligibility contract

**Files:**
- Modify: `newton/tests/test_solver_xpbd_fluid.py:20-65, before devices = get_test_devices()`
- Modify: `newton/_src/solvers/solver.py:177-199`
- Modify: `newton/_src/solvers/xpbd/solver_xpbd.py:332-335`
- Modify: `newton/_src/solvers/sph/solver_sph.py:292-333, 375-380`

- [ ] **Step 1: Verify the two prerequisite implementations are present**

Run:

```bash
rg -n "def test_world_pair|particle_world" \
  newton/_src/geometry/broad_phase_common.py \
  newton/_src/solvers/xpbd/fluid_kernels.py \
  newton/_src/solvers/sph/kernels.py
uv pip install -e /home/maximiliank/Work/warp-main-multiworld-reference
uv run --no-sync python - <<'PY'
import inspect
import warp as wp

print("reserve.grouped", "grouped" in inspect.signature(wp.HashGrid.reserve).parameters)
print("build.groups", "groups" in inspect.signature(wp.HashGrid.build).parameters)
PY
```

Expected: `test_world_pair` is defined, the kernels accept `particle_world`, and the final two lines are `reserve.grouped True` and `build.groups True`. Stop here if either prerequisite is absent.

- [ ] **Step 2: Write failing eligibility tests**

Add the following helper and test to `newton/tests/test_solver_xpbd_fluid.py`, then register the test in the module's existing `_name` tuple:

```python
def _build_particle_grouping_model(device, world_count: int, include_global: bool = False):
    world = newton.ModelBuilder(gravity=0.0)
    world.add_particle(
        pos=(0.0, 0.0, 0.0),
        vel=(0.0, 0.0, 0.0),
        mass=PARTICLE_MASS,
        radius=RADIUS,
        flags=FLUID_FLAGS,
    )
    scene = newton.ModelBuilder(gravity=0.0)
    if include_global:
        scene.add_particle(
            pos=(1.0, 0.0, 0.0),
            vel=(0.0, 0.0, 0.0),
            mass=PARTICLE_MASS,
            radius=RADIUS,
            flags=FLUID_FLAGS,
        )
    scene.replicate(world, world_count)
    return scene.finalize(device=device)


def test_particle_grid_grouping_eligibility(test, device):
    local_model = _build_particle_grouping_model(device, world_count=2)
    local_xpbd = newton.solvers.SolverXPBD(local_model, fluid_rest_distance=SPACING)
    local_sph = newton.solvers.SolverSPH(local_model, smoothing_length=2.2 * RADIUS)
    test.assertTrue(local_xpbd._particle_grid_grouped)
    test.assertTrue(local_sph._particle_grid_grouped)

    single_model = _build_particle_grouping_model(device, world_count=1)
    single_solver = newton.solvers.SolverXPBD(single_model, fluid_rest_distance=SPACING)
    test.assertFalse(single_solver._particle_grid_grouped)

    mixed_model = _build_particle_grouping_model(device, world_count=2, include_global=True)
    mixed_solver = newton.solvers.SolverXPBD(mixed_model, fluid_rest_distance=SPACING)
    test.assertFalse(mixed_solver._particle_grid_grouped)
```

- [ ] **Step 3: Run the test and observe the missing contract**

Run:

```bash
uv run --no-sync --extra dev -m newton.tests -k test_particle_grid_grouping_eligibility
```

Expected: FAIL with `AttributeError: 'SolverXPBD' object has no attribute '_particle_grid_grouped'`.

- [ ] **Step 4: Implement shared eligibility, grouped reserve, and grouped build selection**

Add these methods to `SolverBase`; the host read happens only during construction/model-change notification, never during a captured step:

```python
class SolverBase:
    def __init__(self, model: Model):
        self.model = model
        self._particle_grid_grouped = False

    def _configure_particle_grid(self) -> None:
        """Select and reserve the particle grid path for the current model."""
        model = self.model
        self._particle_grid_grouped = bool(
            model.world_count > 1
            and model.particle_count > 0
            and model.particle_world is not None
            and all(int(world) >= 0 for world in model.particle_world.numpy())
        )
        if model.particle_count > 0 and model.particle_grid is not None:
            with wp.ScopedDevice(model.device):
                model.particle_grid.reserve(model.particle_count, grouped=self._particle_grid_grouped)

    def _build_particle_grid(self, points: wp.array[wp.vec3], radius: float) -> None:
        """Build the selected particle grid without changing legacy call semantics."""
        model = self.model
        assert model.particle_grid is not None
        with wp.ScopedDevice(model.device):
            if self._particle_grid_grouped:
                model.particle_grid.build(points, radius=radius, groups=model.particle_world)
            else:
                model.particle_grid.build(points, radius=radius)
```

In `SolverXPBD.__init__`, replace the direct legacy reserve block with:

```python
        self._configure_particle_grid()
```

In `SolverSPH._ensure_particle_storage`, leave creation of a missing `wp.HashGrid` in place but remove its direct `reserve(n)`. Call `self._configure_particle_grid()` after `_ensure_particle_storage()` in `__init__`, and refresh it on particle-property notifications:

```python
        self._ensure_particle_storage()
        self._configure_particle_grid()
        self._ensure_diffuse_storage()

    @override
    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        if flags & ModelFlags.PARTICLE_PROPERTIES:
            self._ensure_particle_storage()
            self._configure_particle_grid()
            self._ensure_diffuse_storage()
```

Also call `_configure_particle_grid()` from XPBD's `PARTICLE_PROPERTIES` notification branch after `_update_fluid_settings()`.

- [ ] **Step 5: Run focused lifecycle tests**

Run:

```bash
uv run --no-sync --extra dev -m newton.tests -k test_particle_grid_grouping_eligibility
```

Expected: PASS on every configured CPU/CUDA device.

- [ ] **Step 6: Commit the lifecycle**

```bash
git add newton/_src/solvers/solver.py \
  newton/_src/solvers/xpbd/solver_xpbd.py \
  newton/_src/solvers/sph/solver_sph.py \
  newton/tests/test_solver_xpbd_fluid.py
git commit -m "Select grouped particle grids"
```

### Task 2: Route XPBD builds and queries through the grouped path

**Files:**
- Test: `newton/tests/test_particle_world_filtering.py:test_xpbd_multiworld_matches_independent_fluid_baselines, test_particle_global_world_keeps_fallback_interactions`
- Modify: `newton/_src/solvers/xpbd/solver_xpbd.py:558-580, 621-683, 903-910, 1023-1086, 1278-1321`
- Modify: `newton/_src/solvers/xpbd/kernels.py:232-295`
- Modify: `newton/_src/solvers/xpbd/fluid_kernels.py:91-173, 176-296, 299-341, 344-421`

- [ ] **Step 1: Run the correctness oracles before changing XPBD queries**

Run:

```bash
uv run --no-sync --extra dev -m newton.tests -k \
  'test_xpbd_multiworld_matches_independent_fluid_baselines or test_particle_global_world_keeps_fallback_interactions'
```

Expected: PASS. These are the pre-optimization numerical oracle and local/global fallback oracle; Task 1 separately proves that the first model is grouped-eligible and the second is not.

- [ ] **Step 2: Add a `grouped: wp.bool` kernel argument and select exact-world queries**

Place `grouped: wp.bool` immediately after the correctness prerequisite's `particle_world` argument in `solve_particle_particle_contacts`, `compute_fluid_lambdas`, `solve_fluid_deltas`, `compute_fluid_vorticity`, and `solve_fluid_velocities`. Replace each query construction with these exact blocks; leave every `test_world_pair()` rejection inside its loop:

```python
# solve_particle_particle_contacts
world_i = particle_world[i]
if grouped:
    query = wp.hash_grid_query(grid, x, radius + max_radius + k_cohesion, world_i)
else:
    query = wp.hash_grid_query(grid, x, radius + max_radius + k_cohesion)

# compute_fluid_lambdas
world_i = particle_world[i]
if grouped:
    query = wp.hash_grid_query(grid, x, h, world_i)
else:
    query = wp.hash_grid_query(grid, x, h)

# solve_fluid_deltas
world_i = particle_world[i]
if grouped:
    query = wp.hash_grid_query(grid, x, h, world_i)
else:
    query = wp.hash_grid_query(grid, x, h)

# compute_fluid_vorticity
world_i = particle_world[i]
if grouped:
    query = wp.hash_grid_query(grid, x, h, world_i)
else:
    query = wp.hash_grid_query(grid, x, h)

# solve_fluid_velocities
world_i = particle_world[i]
if grouped:
    query = wp.hash_grid_query(grid, x, h, world_i)
else:
    query = wp.hash_grid_query(grid, x, h)
```

- [ ] **Step 3: Replace every XPBD grid build and pass the mode to every launch**

Replace the builds in `step()`, `update_render_particles()`, and `_step_diffuse_particles()` with:

```python
        self._build_particle_grid(state_out.particle_q, search_radius)  # step()
        self._build_particle_grid(state.particle_q, h)                 # render
        self._build_particle_grid(state_out.particle_q, h)             # diffuse
```

Pass `self._particle_grid_grouped` immediately after `model.particle_world` to all five XPBD kernels. Task 3 adds the flag to the shared render/diffuse kernels after their signatures change. The XPBD launch ordering must be:

```python
inputs=[
    model.particle_grid.id,
    particle_q,
    # existing arrays ...
    model.particle_flags,
    model.particle_world,
    self._particle_grid_grouped,
    # existing scalar parameters ...
]
```

- [ ] **Step 4: Run the XPBD world-isolation and fluid suites**

Run:

```bash
uv run --no-sync --extra dev -m newton.tests -k \
  'test_solver_xpbd_fluid or test_xpbd_multiworld_matches_independent_fluid_baselines or test_particle_global_world_keeps_fallback_interactions'
```

Expected: PASS; colocated local worlds use grouped queries, and mixed global/local models still use ungrouped queries plus `test_world_pair()`.

- [ ] **Step 5: Commit XPBD acceleration**

```bash
git add newton/_src/solvers/xpbd/kernels.py \
  newton/_src/solvers/xpbd/fluid_kernels.py \
  newton/_src/solvers/xpbd/solver_xpbd.py \
  newton/tests/test_solver_xpbd_fluid.py
git commit -m "Group XPBD particle queries by world"
```

### Task 3: Route SPH, PBF, render, and diffuse queries through the grouped path

**Files:**
- Test: `newton/tests/test_particle_world_filtering.py:test_sph_multiworld_matches_independent_baselines, test_particle_global_world_keeps_fallback_interactions, test_sph_diffuse_neighbors_are_isolated, test_sph_diffuse_spawning_ignores_other_worlds`
- Modify: `newton/_src/solvers/sph/solver_sph.py:409-511, 716-945`
- Modify: `newton/_src/solvers/sph/kernels.py:479-706, 881-1028, 1034-1138, 1186-1238, 1253-1326, 1336-1415`

- [ ] **Step 1: Run the SPH correctness oracles before changing queries**

Run:

```bash
uv run --no-sync --extra dev -m newton.tests -k \
  'test_sph_multiworld_matches_independent_baselines or test_particle_global_world_keeps_fallback_interactions or test_sph_diffuse_neighbors_are_isolated or test_sph_diffuse_spawning_ignores_other_worlds'
```

Expected: PASS through the filtered path. Task 1's eligibility test proves the local-only model selects grouped mode once this task wires the queries.

- [ ] **Step 2: Add grouped query selection to every shared/SPH neighbor kernel**

Add `grouped: wp.bool` immediately after the relevant world array(s), then use the source particle's world for each query. Apply these exact query blocks and keep the prerequisite's defensive `test_world_pair()` check before distance math, counters, covariance, or atomics:

```python
# compute_sph_density_pressure
world_i = particle_world[i]
if grouped:
    query = wp.hash_grid_query(grid, xi, smoothing_length, world_i)
else:
    query = wp.hash_grid_query(grid, xi, smoothing_length)

# compute_sph_vorticity
world_i = particle_world[i]
if grouped:
    query = wp.hash_grid_query(grid, xi, smoothing_length, world_i)
else:
    query = wp.hash_grid_query(grid, xi, smoothing_length)

# integrate_sph_particles
world_i = particle_world[i]
if grouped:
    query = wp.hash_grid_query(grid, xi, contact_radius, world_i)
else:
    query = wp.hash_grid_query(grid, xi, contact_radius)

# compute_sph_render_particles (both center and covariance passes)
world_i = particle_world[i]
if grouped:
    query = wp.hash_grid_query(grid, xi, h, world_i)
    query_cov = wp.hash_grid_query(grid, xi, h, world_i)
else:
    query = wp.hash_grid_query(grid, xi, h)
    query_cov = wp.hash_grid_query(grid, xi, h)

# compute_pbf_lambdas
world_i = particle_world[i]
if grouped:
    query = wp.hash_grid_query(grid, xi, smoothing_length, world_i)
else:
    query = wp.hash_grid_query(grid, xi, smoothing_length)

# solve_pbf_deltas
world_i = particle_world[i]
if grouped:
    query = wp.hash_grid_query(grid, xi, smoothing_length, world_i)
else:
    query = wp.hash_grid_query(grid, xi, smoothing_length)

# smooth_sph_velocities
world_i = particle_world[i]
if grouped:
    query = wp.hash_grid_query(grid, xi, h, world_i)
else:
    query = wp.hash_grid_query(grid, xi, h)

# update_sph_diffuse_particles
world_i = diffuse_world[tid]
if grouped:
    query = wp.hash_grid_query(grid, x, smoothing_length, world_i)
else:
    query = wp.hash_grid_query(grid, x, smoothing_length)

# spawn_sph_diffuse_particles
world_i = particle_world[i]
if grouped:
    query = wp.hash_grid_query(grid, xi, smoothing_length, world_i)
else:
    query = wp.hash_grid_query(grid, xi, smoothing_length)
```

- [ ] **Step 3: Replace every SPH build with the shared helper**

Use these exact replacements:

```python
self._build_particle_grid(state_in.particle_q, particle_search_radius)       # step
self._build_particle_grid(state_out.particle_q, self.smoothing_length)       # PBF loop
self._build_particle_grid(fluid_state.particle_q, self.smoothing_length)     # velocity smoothing
self._build_particle_grid(fluid_state.particle_q, self.smoothing_length)     # render rebuild
self._build_particle_grid(fluid_state.particle_q, self.smoothing_length)     # diffuse rebuild
```

Pass `model.particle_world` and `self._particle_grid_grouped` in the same order used by each updated signature. In `update_sph_diffuse_particles`, add `grouped` immediately after the new `fluid_world` parameter and pass `model.particle_world, self._particle_grid_grouped`; keep `self.diffuse_worlds` in its existing diffuse-state position later in the argument list. In `spawn_sph_diffuse_particles`, pass `model.particle_world, self._particle_grid_grouped`.

- [ ] **Step 4: Run SPH and cross-solver isolation tests**

Run:

```bash
uv run --no-sync --extra dev -m newton.tests -k \
  'test_solver_sph or test_sph_multiworld_matches_independent_baselines or test_particle_global_world_keeps_fallback_interactions or test_sph_diffuse_neighbors_are_isolated or test_sph_diffuse_spawning_ignores_other_worlds'
```

Expected: PASS, including density, pressure, PBF, XSPH, render covariance, diffuse advection/spawn, and local/global fallback cases.

- [ ] **Step 5: Commit SPH acceleration**

```bash
git add newton/_src/solvers/sph/kernels.py \
  newton/_src/solvers/sph/solver_sph.py \
  newton/tests/test_solver_sph.py
git commit -m "Group SPH particle queries by world"
```

### Task 4: Prove grouped solver capture and benchmark scaling

**Files:**
- Modify: `newton/tests/test_solver_xpbd_fluid.py:before registration block`
- Modify: `newton/tests/test_solver_sph.py:1282-1328`
- Create: `asv/benchmarks/simulation/bench_fluid_multiworld.py`

- [ ] **Step 1: Write a grouped XPBD CUDA graph regression**

Add and register this test:

```python
def test_xpbd_grouped_cuda_graph_capture(test, device):
    if not wp.get_device(device).is_cuda:
        test.skipTest("CUDA graph capture requires a CUDA device")

    model = _build_particle_grouping_model(device, world_count=2)
    solver = newton.solvers.SolverXPBD(model, iterations=2, fluid_rest_distance=SPACING)
    test.assertTrue(solver._particle_grid_grouped)
    state_0 = model.state()
    state_1 = model.state()

    # Compile kernels; grouped capacity was already reserved by the constructor.
    solver.step(state_0, state_1, None, None, 1.0 / 240.0)
    state_0, state_1 = state_1, state_0
    with wp.ScopedCapture(device=device) as capture:
        state_0.clear_forces()
        solver.step(state_0, state_1, None, None, 1.0 / 240.0)

    wp.capture_launch(capture.graph)
    q = state_1.particle_q.numpy()
    test.assertTrue(np.isfinite(q).all())
    np.testing.assert_allclose(q[0], q[1], atol=1.0e-6, rtol=0.0)
```

- [ ] **Step 2: Convert the existing SPH graph test to a grouped multi-world model**

Build its particle grid in a template builder, replicate it twice into a scene, finalize the scene, and add these assertions around the existing capture/replay body:

```python
    test.assertEqual(model.world_count, 2)
    test.assertTrue(solver._particle_grid_grouped)
    # after replay
    q = state_1.particle_q.numpy().reshape(2, -1, 3)
    test.assertTrue(np.isfinite(q).all())
    np.testing.assert_allclose(q[0], q[1], atol=1.0e-5, rtol=0.0)
```

- [ ] **Step 3: Run both capture tests**

Run:

```bash
uv run --no-sync --extra dev -m newton.tests -k 'grouped_cuda_graph_capture or sph_cuda_graph_capture'
```

Expected: CPU variants SKIP; every CUDA variant captures and replays without allocation errors and reports PASS.

- [ ] **Step 4: Add an end-to-end ASV benchmark**

Create `asv/benchmarks/simulation/bench_fluid_multiworld.py` with two solvers, fixed 64 particles per colocated world, world counts `1, 8, 64, 256`, and modes `filtered` and `grouped`. The benchmark deliberately flips only the private selection flag in `filtered` mode so both modes use identical local-only input and the same correctness filters:

```python
import warp as wp
from asv_runner.benchmarks.mark import skip_benchmark_if

import newton


def _make_model(world_count):
    world = newton.ModelBuilder(gravity=0.0)
    world.add_particle_grid(
        pos=wp.vec3(),
        rot=wp.quat_identity(),
        vel=wp.vec3(),
        dim_x=4,
        dim_y=4,
        dim_z=4,
        cell_x=0.05,
        cell_y=0.05,
        cell_z=0.05,
        mass=0.125,
        jitter=0.0,
        radius_mean=0.025,
        flags=newton.ParticleFlags.ACTIVE | newton.ParticleFlags.FLUID,
    )
    scene = newton.ModelBuilder(gravity=0.0)
    scene.replicate(world, world_count)
    return scene.finalize(device="cuda:0")


class TimeFluidMultiworld:
    params = (["xpbd", "sph"], [1, 8, 64, 256], ["filtered", "grouped"])
    param_names = ["solver", "world_count", "query_mode"]
    repeat = 5
    number = 20

    def setup(self, solver, world_count, query_mode):
        self.model = _make_model(world_count)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        if solver == "xpbd":
            self.solver = newton.solvers.SolverXPBD(
                self.model, iterations=2, fluid_rest_distance=0.05, fluid_cohesion=0.0
            )
        else:
            self.solver = newton.solvers.SolverSPH(
                self.model, smoothing_length=0.09, rest_density=1000.0, gas_constant=50.0
            )
        if query_mode == "filtered":
            self.solver._particle_grid_grouped = False
        self.solver.step(self.state_0, self.state_1, None, None, 1.0 / 240.0)
        self.state_0, self.state_1 = self.state_1, self.state_0
        with wp.ScopedCapture(device=self.model.device) as capture:
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, None, None, 1.0 / 240.0)
        self.graph = capture.graph

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_captured_step(self, solver, world_count, query_mode):
        wp.capture_launch(self.graph)
        wp.synchronize_device(self.model.device)


if __name__ == "__main__":
    from newton.utils import run_benchmark

    run_benchmark(TimeFluidMultiworld)
```

Candidate-visit, grouped-build memory, capture-allocation, and BVH-reference measurements remain in the companion Warp benchmark; this Newton benchmark measures the end-to-end captured solver cost and must not add timing assertions to unit tests.

- [ ] **Step 5: Smoke-test and record the benchmark command**

Run:

```bash
uv run --no-sync python asv/benchmarks/simulation/bench_fluid_multiworld.py
```

Expected: all 16 parameter combinations complete; for 64 and 256 worlds, grouped captured steps are materially faster than filtered steps, with no numerical or capture error. Record raw results in the PR description, not in source.

- [ ] **Step 6: Commit capture coverage and benchmark**

```bash
git add newton/tests/test_solver_xpbd_fluid.py \
  newton/tests/test_solver_sph.py \
  asv/benchmarks/simulation/bench_fluid_multiworld.py
git commit -m "Benchmark grouped multi-world fluids"
```

### Task 5: Add world-aware surface packing to ViewerBase and ViewerGL

**Files:**
- Modify: `newton/tests/test_viewer_fluid.py:14-121`
- Modify: `newton/_src/viewer/viewer.py:2456-2518, 2551-2605`
- Modify: `newton/_src/viewer/viewer_gl.py:1247-1291`
- Modify: `newton/_src/viewer/gl/fluid.py:33-81, 913-1084`

- [ ] **Step 1: Write failing surface-pack and high-level-routing tests**

Import `_pack_fluid_vertices` and add a device-parametrized test registered with `add_function_test`:

```python
def test_fluid_surface_world_offsets_and_visibility(test, device):
    points = wp.array([(1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
    worlds = wp.array([0, 1, -1], dtype=wp.int32, device=device)
    offsets = wp.array([(-10.0, 0.0, 0.0), (10.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
    visible = wp.array([1, 0], dtype=wp.int32, device=device)
    radii = wp.array([0.1, 0.1, 0.1], dtype=wp.float32, device=device)
    dummy4 = wp.zeros(1, dtype=wp.vec4, device=device)
    packed = wp.zeros(3 * 16, dtype=wp.float32, device=device)

    wp.launch(
        _pack_fluid_vertices,
        dim=3,
        inputs=[points, radii, 1, 0.0, 1.0, dummy4, dummy4, dummy4, 0, worlds, offsets, visible, 1, packed],
        device=device,
    )
    data = packed.numpy().reshape(3, 16)
    np.testing.assert_allclose(data[:, :3], [[-9.0, 0.0, 0.0], [12.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    np.testing.assert_allclose(data[:, 3], [0.1, 0.0, 0.1])
```

Extend `_LogFluidProbe.log_fluid()` with `worlds=None`, store it, and add to the existing active-particle routing test:

```python
        self.assertIsNotNone(viewer.logged_fluid["worlds"])
        np.testing.assert_array_equal(viewer.logged_fluid["worlds"].numpy(), [-1, -1])
```

- [ ] **Step 2: Run tests to verify the API/kernel mismatch**

Run:

```bash
uv run --no-sync --extra dev -m newton.tests -k 'fluid_surface_world_offsets or show_fluid_routes'
```

Expected: FAIL because `_pack_fluid_vertices` lacks the world arguments and the high-level logger does not forward `worlds`.

- [ ] **Step 3: Extend the public logging signatures without shifting existing positional arguments**

Append `worlds: wp.array[wp.int32] | None = None` after `hidden` in both `ViewerBase.log_fluid()` and `ViewerGL.log_fluid()`. Document it as "World index per particle; local worlds receive display offsets and visibility filtering, while world `-1` remains unshifted and visible." Keep `worlds` last so existing positional callers remain source-compatible.

In `ViewerBase._log_particles()`, preserve alignment when active particles are compacted:

```python
            worlds = self.model.particle_world
            # inside active_count < n
            if worlds is not None:
                worlds_out = wp.empty(active_count, dtype=wp.int32, device=self.device)
                wp.launch(compact, dim=n, inputs=[worlds, mask, offsets, worlds_out], device=self.device)
                worlds = worlds_out

            # in the show_fluid call
            self.log_fluid(
                name="/model/fluid",
                points=points,
                radii=radii,
                color=self.fluid_color,
                ior=self.fluid_ior,
                hidden=False,
                worlds=worlds,
            )
```

- [ ] **Step 4: Offset and collapse surface particles in the existing pack kernel**

Append these arguments to `_pack_fluid_vertices`:

```python
    worlds: wp.array[wp.int32],
    world_offsets: wp.array[wp.vec3],
    visible_worlds_mask: wp.array[wp.int32],
    use_worlds: int,
```

After loading `p` and computing `r`, add:

```python
    if use_worlds != 0:
        world = worlds[tid]
        if world >= 0:
            if world_offsets and world < world_offsets.shape[0]:
                p += world_offsets[world]
            if visible_worlds_mask and (
                world >= visible_worlds_mask.shape[0] or visible_worlds_mask[world] == 0
            ):
                r = 0.0
```

In `FluidBatch`, cache zero-length dummy arrays alongside `_dummy_radii`, pass the real/dummy arrays and `use_worlds` into the CUDA kernel, and mirror the same offset/zero-radius logic in the CPU/NumPy packing branch. Never compact the CUDA surface data on the host; a hidden local-world particle is represented by radius zero.

Finally forward viewer state in `ViewerGL.log_fluid()`:

```python
        batch.update(
            points,
            radii,
            radius_scale,
            anisotropy,
            anisotropy_secondary,
            anisotropy_tertiary,
            worlds=worlds,
            world_offsets=self.world_offsets,
            visible_worlds_mask=self._visible_worlds_mask,
        )
```

- [ ] **Step 5: Run viewer surface tests**

Run:

```bash
uv run --no-sync --extra dev -m newton.tests -k test_viewer_fluid
```

Expected: PASS on CPU/CUDA; world 0 is shifted, hidden world 1 has zero radius, global world `-1` is unchanged, and active compaction preserves world alignment.

- [ ] **Step 6: Commit surface world handling**

```bash
git add newton/_src/viewer/viewer.py \
  newton/_src/viewer/viewer_gl.py \
  newton/_src/viewer/gl/fluid.py \
  newton/tests/test_viewer_fluid.py
git commit -m "Offset fluid surfaces by world"
```

### Task 6: Apply the same world handling to the existing diffuse host path

**Files:**
- Modify: `newton/tests/test_viewer_fluid.py:after surface packing tests`
- Modify: `newton/_src/viewer/viewer.py:2520-2549`
- Modify: `newton/_src/viewer/viewer_gl.py:1293-1325`
- Modify: `newton/_src/viewer/gl/fluid.py:1095-1190`

- [ ] **Step 1: Write a failing host-path diffuse test**

Import `DiffuseBatch` and add:

```python
def test_diffuse_world_offsets_and_visibility(self):
    batch = DiffuseBatch.__new__(DiffuseBatch)
    batch._host_positions = np.zeros((0, 4), dtype=np.float32)
    batch._host_velocities = np.zeros((0, 4), dtype=np.float32)
    batch._ensure_capacity = lambda count: None
    batch._upload = lambda: None

    positions = np.array(
        [[1.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0], [3.0, 0.0, 0.0, 1.0], [4.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    velocities = np.zeros_like(positions)
    worlds = np.array([0, 1, -1, 0], dtype=np.int32)
    offsets = np.array([[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)
    visible = np.array([1, 0], dtype=np.int32)

    batch.update(positions, velocities, worlds=worlds, world_offsets=offsets, visible_worlds_mask=visible)

    self.assertEqual(batch.count, 2)
    np.testing.assert_allclose(batch._host_positions[:, :3], [[-9.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
```

- [ ] **Step 2: Run the test and observe the missing arguments**

Run:

```bash
uv run --no-sync --extra dev -m newton.tests -k test_diffuse_world_offsets_and_visibility
```

Expected: FAIL with `TypeError: DiffuseBatch.update() got an unexpected keyword argument 'worlds'`.

- [ ] **Step 3: Extend diffuse APIs and filter after the existing host transfer**

Append `worlds: wp.array[wp.int32] | None = None` after `hidden` in `ViewerBase.log_fluid_diffuse()` and `ViewerGL.log_fluid_diffuse()`, document the same global/local semantics, and forward:

```python
        batch.update(
            positions,
            velocities,
            worlds=worlds,
            world_offsets=self.world_offsets,
            visible_worlds_mask=self._visible_worlds_mask,
        )
```

Implement `DiffuseBatch.update()` with the current CPU sorting data as its output:

```python
    def update(self, positions, velocities, worlds=None, world_offsets=None, visible_worlds_mask=None):
        if positions is None:
            self.count = 0
            return

        source_positions = positions.numpy() if isinstance(positions, wp.array) else np.asarray(positions)
        host_positions = np.array(source_positions, dtype=np.float32, copy=True)
        if velocities is None:
            host_velocities = np.zeros_like(host_positions)
        else:
            source_velocities = velocities.numpy() if isinstance(velocities, wp.array) else np.asarray(velocities)
            host_velocities = np.asarray(source_velocities, dtype=np.float32)

        live = host_positions[:, 3] > 0.0
        if worlds is not None:
            host_worlds = worlds.numpy() if isinstance(worlds, wp.array) else np.asarray(worlds)
            host_offsets = (
                world_offsets.numpy() if isinstance(world_offsets, wp.array) else np.asarray(world_offsets)
            ) if world_offsets is not None else None
            host_visible = (
                visible_worlds_mask.numpy()
                if isinstance(visible_worlds_mask, wp.array)
                else np.asarray(visible_worlds_mask)
            ) if visible_worlds_mask is not None else None
            local = host_worlds >= 0
            if host_offsets is not None:
                valid_offset = local & (host_worlds < len(host_offsets))
                host_positions[valid_offset, :3] += host_offsets[host_worlds[valid_offset]]
            if host_visible is not None:
                valid_world = local & (host_worlds < len(host_visible))
                live &= ~local | (valid_world & (host_visible[np.minimum(host_worlds, len(host_visible) - 1)] != 0))

        self._host_positions = np.ascontiguousarray(host_positions[live])
        self._host_velocities = np.ascontiguousarray(host_velocities[live])
        self.count = int(self._host_positions.shape[0])
        self._ensure_capacity(self.count)
        self._upload()
```

Retain `sort_for_view()` unchanged. This is intentionally the deferred CPU diffuse path; do not add device scans, GPU sort keys, mapped diffuse VBOs, or new renderer dependencies.

- [ ] **Step 4: Run diffuse and full viewer tests**

Run:

```bash
uv run --no-sync --extra dev -m newton.tests -k 'test_diffuse_world_offsets_and_visibility or test_viewer_visible_worlds or test_viewer_world_offsets'
```

Expected: PASS; hidden local diffuse particles are omitted, visible local particles are shifted, global particles remain visible/unshifted, and existing world layout behavior is unchanged.

- [ ] **Step 5: Commit diffuse world handling**

```bash
git add newton/_src/viewer/viewer.py \
  newton/_src/viewer/viewer_gl.py \
  newton/_src/viewer/gl/fluid.py \
  newton/tests/test_viewer_fluid.py
git commit -m "Filter diffuse particles by world"
```

### Task 7: Pass world IDs from every XPBD/SPH example and document behavior

**Files:**
- Modify: `newton/examples/fluid/example_fluid_sph_cup_transfer.py:381-401`
- Modify: `newton/examples/fluid/example_fluid_sph_dam_break.py:244-272`
- Modify: `newton/examples/fluid/example_fluid_sph_interactive_tank.py:903-935`
- Modify: `newton/examples/fluid/example_fluid_sph_wave_pool.py:312-344`
- Modify: `newton/examples/fluid/example_fluid_xpbd_cereal_bowl.py:440-455`
- Modify: `newton/examples/fluid/example_fluid_xpbd_cup.py:298-313`
- Modify: `newton/examples/fluid/example_fluid_xpbd_cup_transfer.py:514-536`
- Modify: `newton/examples/fluid/example_fluid_xpbd_dam_break.py:196-219`
- Modify: `newton/examples/fluid/example_fluid_xpbd_interactive_tank.py:307-329`
- Modify: `newton/examples/fluid/example_fluid_xpbd_wave_pool.py:354-376`
- Modify: `newton/tests/test_viewer_fluid.py:66-86`
- Modify: `docs/concepts/worlds.rst:209-215`
- Modify: `docs/guide/visualization.rst:616-622`
- Modify: `CHANGELOG.md:[Unreleased] Added and Changed sections`

- [ ] **Step 1: Add an example-routing assertion before changing examples**

Keep the high-level probe assertion from Task 5 and run the registered fluid examples once after the edits below. This provides a test failure if an override drops the new keyword.

- [ ] **Step 2: Pass explicit surface and diffuse world arrays in all ten examples**

Add `worlds=self.model.particle_world` to every non-`None` `log_fluid()` call listed above. Add the corresponding solver diffuse array to every non-`None` `log_fluid_diffuse()` call:

```python
# SPH examples
worlds=self.sph_solver.diffuse_worlds,
# or, where the attribute is named self.solver
worlds=self.solver.diffuse_worlds,

# XPBD examples
worlds=self.solver.diffuse_worlds,
```

Do not add `worlds` to calls whose `positions` argument is `None`; those calls only hide an existing batch. Do not change `example_mpm_viscous.py`, because the design scope names the XPBD/SPH paths.

- [ ] **Step 3: Run all affected example smoke tests**

Run:

```bash
uv run --no-sync --extra dev -m newton.tests -k 'test_basic.example_fluid_sph or test_basic.example_fluid_xpbd'
```

Expected: all registered XPBD/SPH fluid examples PASS with the null viewer; no override rejects `worlds`.

- [ ] **Step 4: Document solver fast-path/fallback semantics**

Add this note after the colocated-world tip in `docs/concepts/worlds.rst`:

```rst
.. note::
   :class:`~newton.solvers.SolverXPBD` and :class:`~newton.solvers.SolverSPH`
   isolate particle-neighbor interactions by :attr:`~newton.Model.particle_world`.
   Models with more than one world and only non-negative particle world IDs use
   grouped hash-grid queries. If any particle is global (world ``-1``), the
   solvers use the compatibility-filtered fallback so global particles can still
   interact with every local world.
```

Extend the visualization example in `docs/guide/visualization.rst`:

```rst
Fluid batches need their world array when they are logged directly so ViewerGL
can apply the same display-only offsets and visible-world filter:

.. code-block:: python

   viewer.log_fluid(
       "/model/fluid",
       solver.render_positions,
       radii=model.particle_radius,
       worlds=model.particle_world,
   )
   viewer.log_fluid_diffuse(
       "/model/fluid/diffuse",
       solver.diffuse_positions,
       solver.diffuse_velocities,
       worlds=solver.diffuse_worlds,
   )

Global particles (world ``-1``) remain unshifted and visible. The built-in
particle logger supplies ``model.particle_world`` automatically.
```

- [ ] **Step 5: Add user-facing changelog entries**

Insert at random positions in the appropriate `[Unreleased]` categories:

```markdown
### Added

- Add optional `worlds` inputs to `Viewer.log_fluid()` and `Viewer.log_fluid_diffuse()` so ViewerGL applies display offsets and visible-world filtering to fluid surfaces and diffuse particles.

### Changed

- Use grouped Warp hash-grid queries for multi-world XPBD and SPH particle neighbors when every particle is world-local; retain filtered local/global compatibility when a model contains global particles.
```

- [ ] **Step 6: Commit examples and documentation**

```bash
git add newton/examples/fluid/example_fluid_sph_*.py \
  newton/examples/fluid/example_fluid_xpbd_*.py \
  docs/concepts/worlds.rst \
  docs/guide/visualization.rst \
  CHANGELOG.md
git commit -m "Document multi-world fluid rendering"
```

### Task 8: Verify compatibility, formatting, and delivery gates

**Files:**
- Verify only; do not add production behavior in this task.

- [ ] **Step 1: Prove the fallback remains exact legacy Warp usage**

Run the single-world, local/global, and legacy fluid tests:

```bash
uv run --no-sync --extra dev -m newton.tests -k \
  'particle_grid_grouping_eligibility or local_global or test_fluid_rest_lattice or test_sph_computes_positive_density'
```

Expected: PASS, with `_particle_grid_grouped == False` for single-world and global-particle models.

- [ ] **Step 2: Run all focused solver and viewer suites**

Run:

```bash
uv run --no-sync --extra dev -m newton.tests -k \
  'test_solver_xpbd_fluid or test_solver_sph or test_viewer_fluid or test_viewer_visible_worlds or test_viewer_world_offsets'
```

Expected: PASS on all configured devices, with only documented CUDA-only skips.

- [ ] **Step 3: Run the complete Newton test suite**

Run:

```bash
uv run --no-sync --extra dev -m newton.tests
```

Expected: PASS with no new failures.

- [ ] **Step 4: Run formatting and documentation checks**

Run:

```bash
uvx pre-commit run -a
uv run --no-sync docs/generate_api.py --validate
```

Expected: both commands exit 0. `generate_api.py --validate` reports no public API documentation drift; the methods are existing public symbols with added optional parameters.

- [ ] **Step 5: Re-run the standalone benchmark and inspect scaling**

Run:

```bash
uv run --no-sync python asv/benchmarks/simulation/bench_fluid_multiworld.py
```

Expected: all cases complete. Keep performance numbers out of assertions; attach the 1/8/64/256-world filtered-versus-grouped table to the PR.

- [ ] **Step 6: Confirm the intentional deferrals**

Run:

```bash
git diff --check
git diff -- newton/_src/viewer/gl/fluid.py | rg 'array_scan|radix_sort|RegisteredGLBuffer'
```

Expected: `git diff --check` exits 0 and the second command finds no newly added GPU diffuse compaction/sort/interoperability code. Existing `RegisteredGLBuffer` lines in the surface path may appear only as unchanged context.

- [ ] **Step 7: Enforce the Warp release gate without inventing a version**

Before merging Newton independently, replace the editable Warp checkout with the first published `warp-lang` release that contains the grouped API, update `pyproject.toml`, `uv.lock`, and `asv.conf.json` to that exact released version, then repeat Steps 1-5. Until that release exists, deliver this as a stacked Newton change on the companion Warp commit; do not claim it works with Warp 1.14.

- [ ] **Step 8: Commit any formatter-only changes**

```bash
git add -u
git commit -m "Format grouped fluid integration"
```

Expected: create this commit only if formatting changed tracked files; otherwise leave the verified branch unchanged.
