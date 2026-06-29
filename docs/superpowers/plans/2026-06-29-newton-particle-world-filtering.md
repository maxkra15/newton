# Newton Particle World Filtering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every Newton XPBD/SPH particle-neighbor, render-neighbor, and diffuse-neighbor operation respect world membership while preserving global world `-1` interactions and preventing unsafe multi-world particle reordering.

**Architecture:** Keep Warp 1.14's ungrouped `HashGrid` build/query API unchanged and add a defensive world predicate immediately after every candidate is returned. Pass `Model.particle_world` through all affected kernels and compare diffuse particles' stored world IDs against fluid candidates; independently, make `SolverXPBD.reorder_particles()` return before global Morton sorting when `world_count > 1`.

**Tech Stack:** Python 3.10+, Warp 1.14.0 (`wp.HashGrid`, Warp kernels), Newton `ModelBuilder`, NumPy, repository `unittest` runner, `uv`.

---

## Scope and compatibility contract

This plan is the correctness-only first subproject from `docs/superpowers/specs/2026-06-29-fluid-multiworld-isolation-design.md`. Do not add grouped `HashGrid` reserve/build/query calls, do not change the Warp dependency, and do not add ViewerGL world-offset APIs here. The implementation must run against the existing `uv.lock` entry `warp-lang==1.14.0` and preserve the exact compatibility rule:

```python
world_a == world_b or world_a == -1 or world_b == -1
```

The predicate must execute before distance calculations, density/gradient accumulation, neighbor counts or caps, covariance accumulation, and diffuse spawn atomics. Particle-shape filtering already has this rule and stays unchanged.

## File map

- Create `newton/tests/test_particle_world_filtering.py`: focused end-to-end regressions for XPBD, SPH, render buffers, diffuse particles, global `-1`, and reorder metadata.
- Modify `newton/_src/geometry/broad_phase_common.py:131-180`: define the shared Warp-compatible world predicate and reuse it from collision world/group filtering.
- Modify `newton/_src/solvers/xpbd/kernels.py:232-291`: filter XPBD particle-particle contacts.
- Modify `newton/_src/solvers/xpbd/fluid_kernels.py:91-411`: filter all four XPBD fluid neighbor passes.
- Modify `newton/_src/solvers/xpbd/solver_xpbd.py:432-497,521-581,605-684,1019-1086,1278-1321`: guard multi-world reorder and wire world arrays to XPBD, render, and diffuse launches.
- Modify `newton/_src/solvers/sph/kernels.py:478-705,880-1030,1033-1137,1185-1237,1252-1325,1335-1451`: filter every SPH, PBF, render, and diffuse query.
- Modify `newton/_src/solvers/sph/solver_sph.py:409-488,716-823,836-945`: wire `model.particle_world` to every affected launch.
- Modify `newton/tests/test_solver_xpbd_fluid.py:387-407`: update the existing direct `compute_fluid_lambdas` launch for its new argument.
- Modify `CHANGELOG.md:114-123`: add an Unreleased Fixed entry describing user-visible isolation and `-1` compatibility.

### Task 1: Add the first failing XPBD isolation regressions

**Files:**
- Create: `newton/tests/test_particle_world_filtering.py`

- [ ] **Step 1: Create the focused test scaffold and model helpers**

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for particle-neighbor isolation between Newton worlds."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverSPH, SolverXPBD
from newton.tests.unittest_utils import add_function_test, get_test_devices

SPACING = 0.05
RADIUS = 0.025
MASS = 0.125
FLUID_FLAGS = int(newton.ParticleFlags.ACTIVE | newton.ParticleFlags.FLUID)
ACTIVE_FLAGS = int(newton.ParticleFlags.ACTIVE)


def _make_particle_builder(positions, velocities, flags=FLUID_FLAGS):
    builder = newton.ModelBuilder(gravity=0.0)
    builder.default_particle_radius = RADIUS
    count = len(positions)
    builder.add_particles(
        pos=[wp.vec3(*p) for p in positions],
        vel=[wp.vec3(*v) for v in velocities],
        mass=[MASS] * count,
        radius=[RADIUS] * count,
        flags=[flags] * count if isinstance(flags, int) else flags,
    )
    return builder


def _make_colocated_model(device, *world_builders, global_builders=()):
    builder = newton.ModelBuilder(gravity=0.0)
    for global_builder in global_builders:
        builder.add_builder(global_builder)
    for world_builder in world_builders:
        builder.add_world(world_builder, xform=wp.transform_identity())
    return builder.finalize(device=device)


def _step_xpbd(model, **kwargs):
    state_in = model.state()
    state_out = model.state()
    solver = SolverXPBD(model, fluid_rest_distance=SPACING, **kwargs)
    solver.step(state_in, state_out, control=None, contacts=None, dt=1.0 / 120.0)
    return state_out, solver


def _step_sph(model, **kwargs):
    state_in = model.state()
    state_out = model.state()
    solver = SolverSPH(model, **kwargs)
    solver.step(state_in, state_out, control=None, contacts=None, dt=1.0 / 120.0)
    return state_out, solver
```

- [ ] **Step 2: Add the minimal coincident-world numerical reproducer**

```python
def test_xpbd_coincident_fluid_worlds_are_isolated(test, device):
    particle = _make_particle_builder([(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)])
    model = _make_colocated_model(device, particle, particle)
    initial_q = model.particle_q.numpy().copy()

    state, _solver = _step_xpbd(model, iterations=1, fluid_cohesion=0.0)

    np.testing.assert_allclose(state.particle_q.numpy(), initial_q, rtol=0.0, atol=1.0e-7)
    np.testing.assert_allclose(state.particle_qd.numpy(), 0.0, rtol=0.0, atol=1.0e-7)
```

- [ ] **Step 3: Add the mixed fluid/solid cross-world contact regression**

```python
def test_xpbd_mixed_fluid_solid_contacts_are_isolated(test, device):
    fluid = _make_particle_builder([(-0.01, 0.0, 0.0)], [(0.0, 0.0, 0.0)], FLUID_FLAGS)
    solid = _make_particle_builder([(0.01, 0.0, 0.0)], [(0.0, 0.0, 0.0)], ACTIVE_FLAGS)
    model = _make_colocated_model(device, fluid, solid)
    initial_q = model.particle_q.numpy().copy()

    state, _solver = _step_xpbd(model, iterations=1, fluid_cohesion=0.0)

    np.testing.assert_allclose(state.particle_q.numpy(), initial_q, rtol=0.0, atol=1.0e-7)
    np.testing.assert_allclose(state.particle_qd.numpy(), 0.0, rtol=0.0, atol=1.0e-7)
```

- [ ] **Step 4: Register the two tests using the repository device convention**

```python
devices = get_test_devices(mode="basic")


class TestParticleWorldFiltering(unittest.TestCase):
    pass


for _name in (
    "test_xpbd_coincident_fluid_worlds_are_isolated",
    "test_xpbd_mixed_fluid_solid_contacts_are_isolated",
):
    add_function_test(
        TestParticleWorldFiltering,
        _name,
        globals()[_name],
        devices=devices,
        check_output=False,
    )


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
```

- [ ] **Step 5: Run each reproducer and verify the pre-fix failures**

Run: `uv run --extra dev -m newton.tests -k test_xpbd_coincident_fluid_worlds_are_isolated -v`

Expected: FAIL on CPU and CUDA (when available) at `assert_allclose`; the two coincident world-local particles move apart instead of matching `initial_q`.

Run: `uv run --extra dev -m newton.tests -k test_xpbd_mixed_fluid_solid_contacts_are_isolated -v`

Expected: FAIL at `assert_allclose`; the fluid and solid particles from different worlds receive XPBD contact corrections.

- [ ] **Step 6: Commit the failing regression tests**

```bash
git add newton/tests/test_particle_world_filtering.py
git commit -m "Test particle world isolation"
```

### Task 2: Add the shared compatibility predicate and XPBD core filters

**Files:**
- Modify: `newton/_src/geometry/broad_phase_common.py:131-180`
- Modify: `newton/_src/solvers/xpbd/kernels.py:4-7,232-291`
- Modify: `newton/_src/solvers/xpbd/fluid_kernels.py:13-16,91-411`
- Modify: `newton/_src/solvers/xpbd/solver_xpbd.py:1019-1086,1278-1321`
- Modify: `newton/tests/test_solver_xpbd_fluid.py:387-407`
- Test: `newton/tests/test_particle_world_filtering.py`

- [ ] **Step 1: Define one Warp predicate and reuse it in broad-phase filtering**

Insert before `test_group_pair`, then replace the open-coded world check in `test_world_and_group_pair`:

```python
@wp.func
def test_world_pair(world_a: int, world_b: int) -> bool:
    """Return whether two Newton world IDs may interact."""
    return world_a == world_b or world_a == -1 or world_b == -1


@wp.func
def test_world_and_group_pair(world_a: int, world_b: int, collision_group_a: int, collision_group_b: int) -> bool:
    if not test_world_pair(world_a, world_b):
        return False
    return test_group_pair(collision_group_a, collision_group_b)
```

- [ ] **Step 2: Filter XPBD particle contacts before reading candidate geometry**

Import `test_world_pair`, add `particle_world` immediately after `particle_flags`, cache `world_i`, and guard each candidate:

```python
from ...geometry.broad_phase_common import test_world_pair


@wp.kernel
def solve_particle_particle_contacts(
    grid: wp.uint64,
    particle_x: wp.array[wp.vec3],
    particle_v: wp.array[wp.vec3],
    particle_invmass: wp.array[float],
    particle_radius: wp.array[float],
    particle_flags: wp.array[wp.int32],
    particle_world: wp.array[wp.int32],
    k_mu: float,
    k_cohesion: float,
    max_radius: float,
    dt: float,
    relaxation: float,
    deltas: wp.array[wp.vec3],
):
    world_i = particle_world[i]
    while wp.hash_grid_query_next(query, index):
        if not test_world_pair(world_i, particle_world[index]):
            continue
        if i_fluid != 0 and (particle_flags[index] & ParticleFlags.FLUID) != 0:
            continue
        if (particle_flags[index] & ParticleFlags.ACTIVE) != 0 and index != i:
            n = x - particle_x[index]
            d = wp.length(n)
```

- [ ] **Step 3: Add world inputs and early guards to both XPBD density passes**

Import `test_world_pair`; add `particle_world: wp.array[wp.int32]` after `particle_flags` in both signatures. The candidate blocks must begin exactly as follows:

```python
# compute_fluid_lambdas
world_i = particle_world[i]
while wp.hash_grid_query_next(query, j):
    if not test_world_pair(world_i, particle_world[j]):
        continue
    if j == i:
        continue
    flags_j = particle_flags[j]

# solve_fluid_deltas
world_i = particle_world[i]
while wp.hash_grid_query_next(query, j):
    if not test_world_pair(world_i, particle_world[j]):
        continue
    if j == i:
        continue
    flags_j = particle_flags[j]
```

This placement ensures incompatible candidates do not consume `fluid_max_neighbors` capacity.

- [ ] **Step 4: Add world inputs and early guards to XPBD vorticity and viscosity**

Add `particle_world: wp.array[wp.int32]` after `particle_flags` in `compute_fluid_vorticity` and `solve_fluid_velocities`:

```python
# compute_fluid_vorticity
world_i = particle_world[i]
while wp.hash_grid_query_next(query, j):
    if not test_world_pair(world_i, particle_world[j]):
        continue
    if j == i:
        continue

# solve_fluid_velocities
world_i = particle_world[i]
while wp.hash_grid_query_next(query, j):
    if not test_world_pair(world_i, particle_world[j]):
        continue
    if j == i:
        continue
```

- [ ] **Step 5: Wire `model.particle_world` into all five XPBD core launches**

In `SolverXPBD.step()`, insert `model.particle_world` immediately after `model.particle_flags` for these launches:

```python
# solve_particle_particle_contacts
model.particle_flags,
model.particle_world,
model.particle_mu,

# compute_fluid_lambdas
model.particle_flags,
model.particle_world,
self._fluid_h,

# solve_fluid_deltas
model.particle_flags,
model.particle_world,
self._fluid_lambda,

# compute_fluid_vorticity
model.particle_flags,
model.particle_world,
self._fluid_density,

# solve_fluid_velocities
model.particle_flags,
model.particle_world,
self._fluid_density,
```

- [ ] **Step 6: Update the existing direct XPBD density-kernel test**

In `densities_with_cap()` in `test_solver_xpbd_fluid.py`, insert the new argument:

```python
model.particle_inv_mass,
model.particle_flags,
model.particle_world,
h,
```

- [ ] **Step 7: Run the XPBD regressions and existing direct-kernel test**

Run: `uv run --extra dev -m newton.tests -k test_xpbd_coincident_fluid_worlds_are_isolated -k test_xpbd_mixed_fluid_solid_contacts_are_isolated -k test_fluid_max_neighbors_truncates_density -v`

Expected: PASS on every selected device; no Warp kernel signature/compilation errors.

- [ ] **Step 8: Commit the shared predicate and XPBD core filters**

```bash
git add newton/_src/geometry/broad_phase_common.py newton/_src/solvers/xpbd/kernels.py newton/_src/solvers/xpbd/fluid_kernels.py newton/_src/solvers/xpbd/solver_xpbd.py newton/tests/test_solver_xpbd_fluid.py
git commit -m "Isolate XPBD particle neighbors by world"
```

### Task 3: Lock down XPBD baseline equivalence and multi-world reorder safety

**Files:**
- Modify: `newton/tests/test_particle_world_filtering.py` before the registration block
- Modify: `newton/_src/solvers/xpbd/solver_xpbd.py:432-497`

- [ ] **Step 1: Add a colocated multi-world versus independent XPBD oracle**

```python
def test_xpbd_multiworld_matches_independent_fluid_baselines(test, device):
    positions = [(0.0, 0.0, 0.0), (0.035, 0.0, 0.0), (0.0, 0.04, 0.0), (0.03, 0.035, 0.01)]
    velocities = [
        [(0.3, 0.0, 0.0), (0.0, 0.2, 0.0), (-0.1, 0.0, 0.1), (0.0, -0.2, 0.0)],
        [(-0.2, 0.1, 0.0), (0.1, -0.3, 0.0), (0.0, 0.2, -0.1), (0.2, 0.0, 0.0)],
    ]
    templates = [_make_particle_builder(positions, world_velocities) for world_velocities in velocities]
    multi_model = _make_colocated_model(device, *templates)
    baseline_models = [template.finalize(device=device) for template in templates]
    kwargs = dict(
        iterations=2,
        fluid_cohesion=0.0,
        fluid_viscosity=0.35,
        fluid_vorticity_confinement=0.2,
    )

    multi_state, multi_solver = _step_xpbd(multi_model, **kwargs)
    baselines = [_step_xpbd(model, **kwargs) for model in baseline_models]
    worlds = multi_model.particle_world.numpy()

    for world_id, (baseline_state, baseline_solver) in enumerate(baselines):
        mask = worlds == world_id
        np.testing.assert_allclose(multi_state.particle_q.numpy()[mask], baseline_state.particle_q.numpy(), rtol=1.0e-5, atol=1.0e-6)
        np.testing.assert_allclose(multi_state.particle_qd.numpy()[mask], baseline_state.particle_qd.numpy(), rtol=1.0e-5, atol=1.0e-6)
        np.testing.assert_allclose(multi_solver._fluid_density.numpy()[mask], baseline_solver._fluid_density.numpy(), rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_allclose(multi_solver._fluid_vorticity.numpy()[mask], baseline_solver._fluid_vorticity.numpy(), rtol=1.0e-5, atol=1.0e-5)
```

- [ ] **Step 2: Register and run the baseline oracle**

Add `"test_xpbd_multiworld_matches_independent_fluid_baselines"` to the registration tuple.

Run: `uv run --extra dev -m newton.tests -k test_xpbd_multiworld_matches_independent_fluid_baselines -v`

Expected before Task 2: FAIL because each slice includes the other world's density, lambda, viscosity, and vorticity contributions. Expected now: PASS.

- [ ] **Step 3: Add the failing multi-world reorder metadata regression**

```python
def test_xpbd_multiworld_reorder_is_noop(test, device):
    positions = [(0.08, 0.0, 0.0), (0.0, 0.08, 0.0), (0.0, 0.0, 0.08), (0.0, 0.0, 0.0)]
    velocities = [(0.0, 0.0, 0.0)] * len(positions)
    template = _make_particle_builder(positions, velocities)
    model = _make_colocated_model(device, template, template)
    state = model.state()
    solver = SolverXPBD(model, iterations=1, fluid_rest_distance=SPACING)
    before_state = (state.particle_q.numpy().copy(), state.particle_qd.numpy().copy())
    before_model = tuple(
        array.numpy().copy()
        for array in (
            model.particle_q,
            model.particle_qd,
            model.particle_colors,
            model.particle_mass,
            model.particle_inv_mass,
            model.particle_radius,
            model.particle_flags,
            model.particle_world,
            model.particle_world_start,
        )
    )

    solver.reorder_particles(state)

    np.testing.assert_array_equal(state.particle_q.numpy(), before_state[0])
    np.testing.assert_array_equal(state.particle_qd.numpy(), before_state[1])
    for array, expected in zip(
        (
            model.particle_q,
            model.particle_qd,
            model.particle_colors,
            model.particle_mass,
            model.particle_inv_mass,
            model.particle_radius,
            model.particle_flags,
            model.particle_world,
            model.particle_world_start,
        ),
        before_model,
        strict=True,
    ):
        np.testing.assert_array_equal(array.numpy(), expected)
```

- [ ] **Step 4: Register and run the reorder test to verify it fails**

Add `"test_xpbd_multiworld_reorder_is_noop"` to the registration tuple.

Run: `uv run --extra dev -m newton.tests -k test_xpbd_multiworld_reorder_is_noop -v`

Expected: FAIL because the global Morton permutation changes particle order and interleaves `particle_world` while `particle_world_start` remains unchanged.

- [ ] **Step 5: Guard and document `reorder_particles()`**

Update the docstring and early-return condition:

```python
"""Sort single-world fluid particles into spatial order.

Multi-world models are left unchanged because a global Morton permutation can
interleave worlds and invalidate :attr:`Model.particle_world_start`. A later
segmented reorder may sort independently inside each immutable world range.
"""
model = self.model
n = model.particle_count
if model.world_count > 1 or not self._all_fluid or not self._has_fluid or n <= 1:
    return
```

- [ ] **Step 6: Run both reorder contracts**

Run: `uv run --extra dev -m newton.tests -k test_xpbd_multiworld_reorder_is_noop -k test_fluid_reorder_is_pure_relabel -k test_fluid_reorder_noop_when_not_all_fluid -v`

Expected: PASS; single-world all-fluid sorting still reorders, while multi-world and non-fluid calls are bit-exact no-ops.

- [ ] **Step 7: Commit the oracle and reorder guard**

```bash
git add newton/tests/test_particle_world_filtering.py newton/_src/solvers/xpbd/solver_xpbd.py
git commit -m "Guard multi-world particle reordering"
```

### Task 4: Add failing SPH core and global-fallback regressions

**Files:**
- Modify: `newton/tests/test_particle_world_filtering.py` before the registration block

- [ ] **Step 1: Add the full SPH slice-versus-independent oracle**

```python
def test_sph_multiworld_matches_independent_baselines(test, device):
    positions = [(0.0, 0.0, 0.0), (0.04, 0.0, 0.0), (0.0, 0.05, 0.0), (0.035, 0.04, 0.015)]
    velocities = [
        [(0.4, 0.0, 0.0), (0.0, 0.3, 0.0), (-0.2, 0.0, 0.1), (0.0, -0.1, 0.0)],
        [(-0.3, 0.1, 0.0), (0.1, -0.4, 0.0), (0.0, 0.2, -0.1), (0.25, 0.0, 0.0)],
    ]
    templates = [_make_particle_builder(positions, world_velocities, ACTIVE_FLAGS) for world_velocities in velocities]
    multi_model = _make_colocated_model(device, *templates)
    baseline_models = [template.finalize(device=device) for template in templates]
    kwargs = dict(
        smoothing_length=0.12,
        rest_density=80.0,
        gas_constant=12.0,
        viscosity=0.2,
        particle_friction=0.15,
        cohesion=0.1,
        surface_tension=0.05,
        vorticity_confinement=0.1,
        pbf_iterations=2,
        xsph_strength=0.25,
        shape_collision=False,
        render_smoothing=0.0,
        render_anisotropy_scale=0.0,
    )

    multi_state, multi_solver = _step_sph(multi_model, **kwargs)
    baselines = [_step_sph(model, **kwargs) for model in baseline_models]
    worlds = multi_model.particle_world.numpy()
    for world_id, (baseline_state, baseline_solver) in enumerate(baselines):
        mask = worlds == world_id
        np.testing.assert_allclose(multi_state.particle_q.numpy()[mask], baseline_state.particle_q.numpy(), rtol=1.0e-5, atol=1.0e-6)
        np.testing.assert_allclose(multi_state.particle_qd.numpy()[mask], baseline_state.particle_qd.numpy(), rtol=1.0e-5, atol=1.0e-6)
        np.testing.assert_allclose(multi_solver.particle_density.numpy()[mask], baseline_solver.particle_density.numpy(), rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_allclose(multi_solver.particle_pressure.numpy()[mask], baseline_solver.particle_pressure.numpy(), rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_allclose(multi_solver.particle_vorticity.numpy()[mask], baseline_solver.particle_vorticity.numpy(), rtol=1.0e-5, atol=1.0e-5)
```

- [ ] **Step 2: Add the exact `-1` fallback density regression**

```python
def test_particle_global_world_keeps_fallback_interactions(test, device):
    local = _make_particle_builder([(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)], ACTIVE_FLAGS)
    global_particle = _make_particle_builder([(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)], ACTIVE_FLAGS)
    model = _make_colocated_model(device, local, local, global_builders=(global_particle,))
    _state, solver = _step_sph(
        model,
        smoothing_length=0.1,
        rest_density=1.0,
        gas_constant=0.0,
        viscosity=0.0,
        shape_collision=False,
        render_smoothing=0.0,
        render_anisotropy_scale=0.0,
    )
    worlds = model.particle_world.numpy()
    density = solver.particle_density.numpy()
    local_density = density[worlds >= 0]
    global_density = density[worlds == -1]

    test.assertEqual(local_density.shape, (2,))
    test.assertEqual(global_density.shape, (1,))
    np.testing.assert_allclose(local_density, global_density[0] * (2.0 / 3.0), rtol=1.0e-5, atol=1.0e-5)
```

- [ ] **Step 3: Register and run both SPH tests to verify failure**

Add both names to the registration tuple:

```python
"test_sph_multiworld_matches_independent_baselines",
"test_particle_global_world_keeps_fallback_interactions",
```

Run: `uv run --extra dev -m newton.tests -k test_sph_multiworld_matches_independent_baselines -k test_particle_global_world_keeps_fallback_interactions -v`

Expected: FAIL. The baseline oracle sees cross-world density/pressure/force/PBF/XSPH contributions, and each local particle has the same three-particle density as the global particle instead of the expected two-particle density.

- [ ] **Step 4: Commit the failing SPH regressions**

```bash
git add newton/tests/test_particle_world_filtering.py
git commit -m "Test SPH particle world isolation"
```

### Task 5: Filter every SPH simulation query and wire launches

**Files:**
- Modify: `newton/_src/solvers/sph/kernels.py:4-27,478-705,1033-1137,1185-1237`
- Modify: `newton/_src/solvers/sph/solver_sph.py:409-488,716-823`
- Test: `newton/tests/test_particle_world_filtering.py`

- [ ] **Step 1: Import the predicate and filter density and vorticity**

Add `particle_world: wp.array[wp.int32]` after `particle_flags` in both kernels:

```python
from ...geometry.broad_phase_common import test_world_pair

# compute_sph_density_pressure
world_i = particle_world[i]
while wp.hash_grid_query_next(query, j):
    if not test_world_pair(world_i, particle_world[j]):
        continue
    if _is_active_particle(particle_flags[j]):
        r = xi - particle_q[j]
        density += particle_mass[j] * _poly6_kernel(wp.dot(r, r), smoothing_length)

# compute_sph_vorticity
world_i = particle_world[i]
while wp.hash_grid_query_next(query, j):
    if not test_world_pair(world_i, particle_world[j]):
        continue
    if j != i and _is_active_particle(particle_flags[j]):
        r_vec = xi - particle_q[j]
        r = wp.length(r_vec)
        if r > EPS and r < smoothing_length:
            rho_j = wp.max(density[j], EPS)
            grad = _spiky_gradient(r_vec, r, smoothing_length)
            omega += particle_mass[j] / rho_j * wp.cross(particle_qd[j] - vi, grad)
```

- [ ] **Step 2: Filter `integrate_sph_particles` before contact and force work**

The kernel already receives `particle_world`; use the cached `world_idx` as the source ID:

```python
world_idx = particle_world[i]
accel += gravity[wp.max(world_idx, 0)] * buoyancy
while wp.hash_grid_query_next(query, j):
    if not test_world_pair(world_idx, particle_world[j]):
        continue
    if j != i and _is_active_particle(particle_flags[j]):
        xj = particle_q[j]
        r_vec = xi - xj
        r = wp.length(r_vec)
        if r < contact_radius and r > EPS:
            contact_count += 1
```

- [ ] **Step 3: Filter both PBF passes before density/gradient work**

Add `particle_world` after `particle_flags` in each signature:

```python
# compute_pbf_lambdas
world_i = particle_world[i]
while wp.hash_grid_query_next(query, j):
    if not test_world_pair(world_i, particle_world[j]):
        continue
    if not _is_active_particle(particle_flags[j]):
        continue

# solve_pbf_deltas
world_i = particle_world[i]
while wp.hash_grid_query_next(query, j):
    if not test_world_pair(world_i, particle_world[j]):
        continue
    if j == i or not _is_active_particle(particle_flags[j]):
        continue
```

- [ ] **Step 4: Filter post-projection XSPH smoothing**

Add `particle_world` after `particle_flags` in `smooth_sph_velocities`:

```python
world_i = particle_world[i]
while wp.hash_grid_query_next(query, j):
    if not test_world_pair(world_i, particle_world[j]):
        continue
    if not _is_active_particle(particle_flags[j]):
        continue
```

- [ ] **Step 5: Wire all six SPH core launches**

Insert `model.particle_world` immediately after `model.particle_flags` in `compute_sph_density_pressure`, `compute_sph_vorticity`, `compute_pbf_lambdas`, `solve_pbf_deltas`, and `smooth_sph_velocities`. `integrate_sph_particles` already receives it; retain that argument and rely on the new guard.

```python
model.particle_flags,
model.particle_world,
```

- [ ] **Step 6: Run SPH core and existing solver tests**

Run: `uv run --extra dev -m newton.tests -k test_sph_multiworld_matches_independent_baselines -k test_particle_global_world_keeps_fallback_interactions -k test_sph_pressure_separates_overlapping_particles -k test_sph_pbf_projection_separates_without_pressure -k test_sph_xsph_velocity_smoothing_reduces_local_velocity_noise -v`

Expected: PASS on CPU and CUDA when available.

- [ ] **Step 7: Commit SPH core filtering**

```bash
git add newton/_src/solvers/sph/kernels.py newton/_src/solvers/sph/solver_sph.py
git commit -m "Isolate SPH particle neighbors by world"
```

### Task 6: Test and filter shared render and diffuse consumers

**Files:**
- Modify: `newton/tests/test_particle_world_filtering.py` before the registration block
- Modify: `newton/_src/solvers/sph/kernels.py:880-1030,1252-1325,1335-1451`
- Modify: `newton/_src/solvers/sph/solver_sph.py:836-945`
- Modify: `newton/_src/solvers/xpbd/solver_xpbd.py:521-581,605-684`

- [ ] **Step 1: Add a failing XPBD render-neighbor regression**

```python
def test_xpbd_render_neighbors_are_isolated(test, device):
    left = _make_particle_builder([(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)])
    right = _make_particle_builder([(0.04, 0.0, 0.0)], [(0.0, 0.0, 0.0)])
    model = _make_colocated_model(device, left, right)
    state = model.state()
    solver = SolverXPBD(model, iterations=1, fluid_rest_distance=SPACING)

    solver.update_render_particles(state, smoothing=1.0, anisotropy_scale=1.0)

    np.testing.assert_allclose(solver.render_positions.numpy(), state.particle_q.numpy(), rtol=0.0, atol=1.0e-7)
```

Register it, then run: `uv run --extra dev -m newton.tests -k test_xpbd_render_neighbors_are_isolated -v`

Expected: FAIL because each world-local render point smooths toward the other world.

- [ ] **Step 2: Add deterministic SPH diffuse advection coverage for local and global IDs**

```python
def test_sph_diffuse_neighbors_are_isolated(test, device):
    positive = _make_particle_builder([(0.0, 0.0, 0.0)], [(1.0, 0.0, 0.0)], ACTIVE_FLAGS)
    negative = _make_particle_builder([(0.0, 0.0, 0.0)], [(-1.0, 0.0, 0.0)], ACTIVE_FLAGS)
    model = _make_colocated_model(device, positive, negative)
    solver = SolverSPH(
        model,
        smoothing_length=0.1,
        rest_density=1.0,
        gas_constant=0.0,
        viscosity=0.0,
        max_diffuse_particles=2,
        diffuse_threshold=1.0e9,
        diffuse_spawn_probability=0.0,
        diffuse_drag=120.0,
        diffuse_buoyancy=1.0,
        diffuse_ballistic=1,
        shape_collision=False,
        render_smoothing=0.0,
        render_anisotropy_scale=0.0,
    )
    solver.diffuse_positions.assign(np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32))
    solver.diffuse_velocities.assign(np.zeros((2, 4), dtype=np.float32))
    solver.diffuse_worlds.assign(np.array([0, -1], dtype=np.int32))
    solver.diffuse_slot_states.assign(np.ones(2, dtype=np.int32))
    state_in, state_out = model.state(), model.state()

    solver.step(state_in, state_out, control=None, contacts=None, dt=1.0 / 120.0)

    diffuse_v = solver.diffuse_velocities.numpy()[:, :3]
    np.testing.assert_allclose(diffuse_v[0], [1.0, 0.0, 0.0], rtol=1.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(diffuse_v[1], [0.0, 0.0, 0.0], rtol=0.0, atol=1.0e-5)
```

- [ ] **Step 3: Add a diffuse-spawn regression that reaches the atomic only through a foreign neighbor**

```python
def test_sph_diffuse_spawning_ignores_other_worlds(test, device):
    left = _make_particle_builder([(-0.01, 0.0, 0.0)], [(-1.0, 0.0, 0.0)], ACTIVE_FLAGS)
    right = _make_particle_builder([(0.01, 0.0, 0.0)], [(1.0, 0.0, 0.0)], ACTIVE_FLAGS)
    model = _make_colocated_model(device, left, right)
    state, solver = _step_sph(
        model,
        smoothing_length=0.1,
        rest_density=1.0,
        gas_constant=0.0,
        viscosity=0.0,
        max_diffuse_particles=8,
        diffuse_threshold=0.01,
        diffuse_spawn_probability=1.0,
        diffuse_jitter=0.0,
        shape_collision=False,
        render_smoothing=0.0,
        render_anisotropy_scale=0.0,
    )

    test.assertEqual(int(solver.diffuse_spawn_counter.numpy()[0]), 0)
    test.assertEqual(int(np.count_nonzero(solver.diffuse_positions.numpy()[:, 3] > 0.0)), 0)
```

Register both diffuse tests. Run: `uv run --extra dev -m newton.tests -k test_sph_diffuse_neighbors_are_isolated -k test_sph_diffuse_spawning_ignores_other_worlds -v`

Expected: FAIL; the local diffuse particle averages both worlds and the separating cross-world pair emits diffuse particles.

- [ ] **Step 4: Filter both render queries, including covariance**

Add `particle_world` after `particle_flags` in `compute_sph_render_particles` and apply the guard before activity/geometry work in both loops:

```python
world_i = particle_world[i]
while wp.hash_grid_query_next(query, j):
    if not test_world_pair(world_i, particle_world[j]):
        continue
    if not _is_active_particle(particle_flags[j]):
        continue

# covariance query
while wp.hash_grid_query_next(query_cov, k):
    if not test_world_pair(world_i, particle_world[k]):
        continue
    if not _is_active_particle(particle_flags[k]):
        continue
```

- [ ] **Step 5: Filter diffuse advection before weight/count accumulation**

Add `fluid_world` after `fluid_flags` in `update_sph_diffuse_particles`:

```python
world_idx = diffuse_world[tid]
g = gravity[wp.max(world_idx, 0)]
while wp.hash_grid_query_next(query, j):
    if not test_world_pair(world_idx, fluid_world[j]):
        continue
    if not _is_active_particle(fluid_flags[j]):
        continue
    r_vec = x - fluid_q[j]
    r2 = wp.dot(r_vec, r_vec)
```

- [ ] **Step 6: Filter diffuse spawning before divergence, neighbor count, and atomics**

`spawn_sph_diffuse_particles` already receives `fluid_world`; cache the source ID and guard candidates:

```python
world_i = fluid_world[i]
while wp.hash_grid_query_next(query, j):
    if not test_world_pair(world_i, fluid_world[j]):
        continue
    if j == i or not _is_active_particle(fluid_flags[j]):
        continue
```

- [ ] **Step 7: Wire render and diffuse world arrays in both solvers**

For both `SolverSPH._update_render_particles()` and `SolverXPBD.update_render_particles()`, insert:

```python
model.particle_flags,
model.particle_world,
```

For both `_step_diffuse_particles()` methods, pass the new fluid-world input to `update_sph_diffuse_particles`:

```python
model.particle_flags,
model.particle_world,
model.gravity,
```

Keep the existing `model.particle_world` argument to `spawn_sph_diffuse_particles` in both solvers.

- [ ] **Step 8: Run focused render/diffuse tests and existing single-world coverage**

Run: `uv run --extra dev -m newton.tests -k test_xpbd_render_neighbors_are_isolated -k test_sph_diffuse_neighbors_are_isolated -k test_sph_diffuse_spawning_ignores_other_worlds -k test_fluid_render_particles -k test_fluid_diffuse_particles_spawn_and_expire -k test_sph_render_buffers_smooth_and_stretch_particles -k test_sph_diffuse_particles_spawn_from_fast_fluid -v`

Expected: PASS; single-world render/diffuse behavior remains intact.

- [ ] **Step 9: Commit shared render and diffuse filtering**

```bash
git add newton/tests/test_particle_world_filtering.py newton/_src/solvers/sph/kernels.py newton/_src/solvers/sph/solver_sph.py newton/_src/solvers/xpbd/solver_xpbd.py
git commit -m "Isolate render and diffuse particle neighbors"
```

### Task 7: Audit every consumer, document the fix, and verify Warp 1.14

**Files:**
- Modify: `CHANGELOG.md:114-123`
- Verify: all files changed above

- [ ] **Step 1: Audit all 15 query loops for an early world guard**

Run:

```bash
rg -n -A6 "while wp.hash_grid_query_next" newton/_src/solvers/xpbd/kernels.py newton/_src/solvers/xpbd/fluid_kernels.py newton/_src/solvers/sph/kernels.py
```

Expected: 15 loops total: one XPBD contact, four XPBD fluid, and ten SPH/shared loops (density, vorticity, force integration, two render loops, two PBF loops, XSPH smoothing, diffuse update, diffuse spawn). Every block shows `test_world_pair(...)` before candidate position/velocity reads, counts, caps, covariance, or atomics.

- [ ] **Step 2: Confirm no post-Warp-1.14 grouped API entered this changeset**

Run:

```bash
git diff --check
git diff | rg "groups=|grouped=True|hash_grid_query\([^\n]*,[^\n]*,[^\n]*,[^\n]*\)"
```

Expected: `git diff --check` exits 0; the second command exits 1 with no matches.

- [ ] **Step 3: Add the user-facing changelog entry at a non-terminal position under Fixed**

Insert within `## [Unreleased]` → `### Fixed`:

```markdown
- Fix XPBD and SPH particle, fluid-render, and diffuse-neighbor interactions crossing between colocated worlds while preserving interactions with global world `-1` particles; keep XPBD particle reordering unchanged for multi-world models until segmented reordering is available.
```

- [ ] **Step 4: Run the complete focused regression module**

Run: `uv run --extra dev -m newton.tests -p 'test_particle_world_filtering.py' -v`

Expected: all CPU and CUDA variants PASS; no unexpected output.

- [ ] **Step 5: Run the complete existing XPBD-fluid and SPH modules**

Run: `uv run --extra dev -m newton.tests -p 'test_solver_xpbd_fluid.py' -v`

Expected: PASS (CUDA-only texture-SDF cases may SKIP on CPU).

Run: `uv run --extra dev -m newton.tests -p 'test_solver_sph.py' -v`

Expected: PASS (CUDA graph case SKIPs when CUDA is unavailable).

- [ ] **Step 6: Verify the locked released Warp and run formatting/lint checks**

Run:

```bash
uv run --extra dev python -c 'import warp as wp; assert wp.__version__ == "1.14.0", wp.__version__'
uvx pre-commit run -a
```

Expected: version assertion exits 0 and all pre-commit hooks PASS. Do not change `pyproject.toml` or `uv.lock`.

- [ ] **Step 7: Review the final diff for scope and commit**

Run: `git status --short && git diff --stat && git diff -- CHANGELOG.md`

Expected: only the nine planned source/test/changelog files are changed or created; no Warp source, dependency, Viewer API, example, or grouped-grid changes appear.

```bash
git add CHANGELOG.md newton/_src/geometry/broad_phase_common.py newton/_src/solvers/xpbd/kernels.py newton/_src/solvers/xpbd/fluid_kernels.py newton/_src/solvers/xpbd/solver_xpbd.py newton/_src/solvers/sph/kernels.py newton/_src/solvers/sph/solver_sph.py newton/tests/test_solver_xpbd_fluid.py newton/tests/test_particle_world_filtering.py
git commit -m "Fix particle isolation between worlds"
```
