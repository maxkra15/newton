# Implicit MPM Multi-World Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Make SolverImplicitMPM isolate coincident Newton worlds by default, including FEM topology, convergence reductions, colliders, projection, and grain rendering, with deterministic tests and a CUDA benchmark.

**Architecture:** Keep one Newton solver and monolithic particle/state arrays, but build Warp multi-environment FEM geometry and environment-first space partitions so the assembled system is block diagonal. Carry environment offsets into nonlinear residual reductions and Warp linear operators, and carry collider-world metadata through stable-ID indirection tables so each query visits global and matching local colliders only. Preserve single-world behavior and provide Config.separate_worlds=False as the explicit legacy coupled path.

**Tech Stack:** Python 3.11+, Newton ModelBuilder/Model/State, Warp 1.15 multi-environment FEM and sparse linear solvers, unittest/NumPy, ASV, Ruff/pre-commit.

---

## File map

- Modify newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py: public configuration and collider API, particle-world validation, multi-environment grid/PIC construction, environment-first scratchpad layout, and solver data plumbing.
- Modify newton/_src/solvers/implicit_mpm/implicit_mpm_solver_kernels.py: environment-aware active-cell lookup and per-environment sparse voxel extraction/padding.
- Modify newton/_src/solvers/implicit_mpm/solve_rheology.py: per-environment nonlinear residuals and scalar strain-DOF batch offsets for Krylov solvers.
- Modify newton/_src/solvers/implicit_mpm/implicit_mpm_model.py: collider-world inference, static-shape grouping, validation, and packed query metadata.
- Modify newton/_src/solvers/implicit_mpm/rasterized_collisions.py: stable-ID world-filtered collider lookup for rasterization and direct particle projection.
- Modify newton/_src/solvers/implicit_mpm/render_grains.py: repeat particle environment IDs for grain quadrature.
- Modify newton/tests/test_implicit_mpm.py: policy, reference-equivalence, grid/integration mode, collider, projection, and linear-solver regression coverage.
- Create asv/benchmarks/simulation/bench_implicit_mpm.py: quick one-solver-versus-independent-solvers CUDA benchmark.
- Modify docs/concepts/worlds.rst: public implicit-MPM world semantics and sparse-grid guidance.
- Modify CHANGELOG.md: Unreleased Added entry.

### Task 1: Lock the public isolation contract with failing tests

**Files:**
- Modify: newton/tests/test_implicit_mpm.py
- Modify: newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py:615-1035

- [ ] **Step 1: Add deterministic builder and stepping helpers**

Add imports for get_cuda_test_devices and define helpers directly above the existing tests. The helper creates particles locally, then uses add_world so every MPM particle has an ordinary world ID; the single-world reference deliberately remains a global-particle model.

~~~python
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


def _make_mpm_particle_builder(gravity=(0.0, -9.81, 0.0), velocity=(0.0, 0.0, 0.0)):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=gravity)
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_particle_grid(
        pos=wp.vec3(0.025, 0.025, 0.025),
        rot=wp.quat_identity(),
        vel=wp.vec3(velocity),
        dim_x=2,
        dim_y=2,
        dim_z=2,
        cell_x=0.05,
        cell_y=0.05,
        cell_z=0.05,
        mass=0.01,
        jitter=0.0,
        custom_attributes={"mpm:young_modulus": 1.0e4, "mpm:poisson_ratio": 0.2},
    )
    return builder


def _make_mpm_config(grid_type="dense", integration_scheme="pic", solver="jacobi"):
    config = SolverImplicitMPM.Config()
    config.grid_type = grid_type
    config.voxel_size = 0.1
    config.integration_scheme = integration_scheme
    config.solver = solver
    config.max_iterations = 4
    config.tolerance = 0.0
    config.warmstart_mode = "grid"
    return config


def _step_mpm(model, config, step_count=3, dt=0.01):
    solver = SolverImplicitMPM(model, config=config)
    state_0 = model.state()
    state_1 = model.state()
    for _ in range(step_count):
        solver.step(state_0, state_1, control=None, contacts=None, dt=dt)
        state_0, state_1 = state_1, state_0
    return solver, state_0
~~~

- [ ] **Step 2: Add configuration and global-particle policy tests**

Add these tests. The two-world model is formed from two local source builders; the rejection case adds a global particle before those worlds.

~~~python
def test_multiworld_isolation_config(test, device):
    config = SolverImplicitMPM.Config()
    test.assertTrue(config.separate_worlds)


def test_multiworld_global_particles_rejected(test, device):
    builder = _make_mpm_particle_builder()
    local = _make_mpm_particle_builder()
    builder.add_world(local)
    builder.add_world(local)
    model = builder.finalize(device=device)

    with test.assertRaisesRegex(ValueError, "global MPM particles"):
        SolverImplicitMPM(model, _make_mpm_config())


def test_single_world_global_particles_supported(test, device):
    model = _make_mpm_particle_builder().finalize(device=device)
    _solver, state = _step_mpm(model, _make_mpm_config(), step_count=1)
    test.assertTrue(np.isfinite(state.particle_q.numpy()).all())


def test_multiworld_shared_grid_opt_out_accepts_global_particles(test, device):
    builder = _make_mpm_particle_builder()
    local = _make_mpm_particle_builder()
    builder.add_world(local)
    builder.add_world(local)
    model = builder.finalize(device=device)
    config = _make_mpm_config()
    config.separate_worlds = False
    _solver, state = _step_mpm(model, config, step_count=1)
    test.assertTrue(np.isfinite(state.particle_q.numpy()).all())
~~~

Register the four tests on basic devices using add_function_test. Use explicit names containing multiworld so the focused command selects them.

- [ ] **Step 3: Run the policy tests and confirm the new contract is absent**

Run:

~~~bash
uv run --extra dev -m newton.tests -k 'multiworld_isolation_config|multiworld_global_particles_rejected|single_world_global_particles_supported|multiworld_shared_grid_opt_out'
~~~

Expected: the config test fails because separate_worlds is missing, and construction does not yet raise for the global-particle case.

- [ ] **Step 4: Add the configuration field and construction-time validation**

Add the field under Config grid settings and resolve the internal mode before ImplicitMPMModel is constructed.

~~~python
separate_worlds: bool = True
"""Use independent FEM environments for each world in a multi-world model.

Set to False to retain the legacy shared-grid behavior. Isolated multi-world
models require every MPM particle to belong to a local world.
"""
~~~

In __init__, use these exact internal names and validation rules:

~~~python
self._separate_worlds = bool(config.separate_worlds and model.world_count > 1)
self._environment_count = model.world_count if self._separate_worlds else 1
self._particle_environment = model.particle_world if self._separate_worlds else None

if self._separate_worlds:
    particle_world = model.particle_world.numpy()
    global_particle_ids = np.flatnonzero(particle_world < 0)
    if global_particle_ids.size:
        raise ValueError(
            "SolverImplicitMPM cannot isolate a multi-world model containing global MPM particles; "
            "replicate the particles into each world or set Config.separate_worlds=False for legacy coupled behavior."
        )
    if np.any(particle_world >= model.world_count):
        raise ValueError("MPM particle world IDs must be smaller than model.world_count.")

self._mpm_model = ImplicitMPMModel(model, config)
~~~

Expand the class and Config docstrings to say isolation is default, global particles are supported only in the effective single environment, and False restores the shared topology.

- [ ] **Step 5: Run the policy tests**

Run the Step 3 command.

Expected: all four tests pass on every available basic device.

- [ ] **Step 6: Commit the contract**

~~~bash
git add newton/tests/test_implicit_mpm.py newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py
git commit -m "Add implicit MPM world isolation contract"
~~~

### Task 2: Isolate dense and fixed grids plus PIC and GIMP binning

**Files:**
- Modify: newton/tests/test_implicit_mpm.py
- Modify: newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py:159-360,1206-1475
- Modify: newton/_src/solvers/implicit_mpm/implicit_mpm_solver_kernels.py:984-1001

- [ ] **Step 1: Add the coincident-world reference-equivalence test**

Add a helper that runs two independent global-particle references and a coincident local-world model. Set per-world gravity after finalization and slice with particle_world_start; Newton start arrays include a leading global range, so local world i occupies starts[i + 1]:starts[i + 2].

~~~python
def _run_multiworld_reference_case(device, grid_type="dense", integration_scheme="pic", solver="jacobi"):
    gravity = ((3.0, -2.0, 0.0), (-5.0, 1.0, 0.0))
    reference = []
    for world_gravity in gravity:
        model = _make_mpm_particle_builder(gravity=world_gravity).finalize(device=device)
        _solver, state = _step_mpm(
            model,
            _make_mpm_config(grid_type, integration_scheme, solver),
        )
        reference.append((state.particle_q.numpy(), state.particle_qd.numpy()))

    source = _make_mpm_particle_builder(gravity=0.0)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_world(source)
    builder.add_world(source)
    model = builder.finalize(device=device)
    for world, world_gravity in enumerate(gravity):
        model.set_gravity(world_gravity, world=world)

    _solver, state = _step_mpm(
        model,
        _make_mpm_config(grid_type, integration_scheme, solver),
    )
    starts = model.particle_world_start.numpy()
    particle_q = state.particle_q.numpy()
    particle_qd = state.particle_qd.numpy()
    for world in range(2):
        world_slice = slice(starts[world + 1], starts[world + 2])
        np.testing.assert_allclose(particle_q[world_slice], reference[world][0], rtol=2.0e-4, atol=2.0e-5)
        np.testing.assert_allclose(particle_qd[world_slice], reference[world][1], rtol=2.0e-4, atol=2.0e-5)


def test_multiworld_dense_pic_matches_independent(test, device):
    _run_multiworld_reference_case(device, grid_type="dense", integration_scheme="pic")


def test_multiworld_dense_gimp_matches_independent(test, device):
    _run_multiworld_reference_case(device, grid_type="dense", integration_scheme="gimp")
~~~

Register dense PIC and GIMP on basic devices. Later add a fixed CUDA registration using the same helper.

- [ ] **Step 2: Run the dense PIC test and observe coupled results**

Run:

~~~bash
uv run --extra dev -m newton.tests -k multiworld_dense_pic_matches_independent
~~~

Expected: FAIL because coincident particles contribute to the same FEM nodes.

- [ ] **Step 3: Make scratchpad partitions environment-major**

Add environment_first to rebuild_function_spaces and pass it through all three make_space_partition calls:

~~~python
def rebuild_function_spaces(
    self,
    pic: fem.PicQuadrature,
    velocity_basis_str: str,
    strain_basis_str: str,
    collider_basis_str: str,
    max_cell_count: int,
    environment_first: bool,
    temporary_store: fem.TemporaryStore,
):
~~~

Each partition call gains:

~~~python
environment_first=environment_first,
~~~

Store the partition arrays after construction so later tasks can consume them:

~~~python
self.velocity_environment_offsets = vel_space_partition.env_offsets if environment_first else None
self.collider_environment_offsets = collision_space_partition.env_offsets if environment_first else None
self.strain_environment_offsets = strain_space_partition.env_offsets if environment_first else None
~~~

When collision and velocity bases alias, set collider_environment_offsets from velocity_environment_offsets. Pass self._separate_worlds from _rebuild_scratchpad.

- [ ] **Step 4: Build dense/fixed multi-environment geometry and environment-aware active cells**

Pass the effective environment count to Grid3D:

~~~python
grid = fem.Grid3D(
    bounds_lo=wp.vec3(grid_min * voxel_size),
    bounds_hi=wp.vec3(grid_max * voxel_size),
    res=wp.vec3i((grid_max - grid_min).astype(int)),
    env_count=self._environment_count,
)
~~~

Extend mark_active_cells with particle_environment and use the environment overload only when the optional array is present:

~~~python
@fem.integrand
def mark_active_cells(
    s: fem.Sample,
    domain: fem.Domain,
    positions: wp.array[wp.vec3],
    particle_flags: wp.array[int],
    particle_environment: wp.array[int],
    active_cells: wp.array[int],
):
    if ~particle_flags[s.qp_index] & newton.ParticleFlags.ACTIVE:
        return

    x = positions[s.qp_index]
    if particle_environment:
        s_grid = fem.lookup(domain, x, int(particle_environment[s.qp_index]))
    else:
        s_grid = fem.lookup(domain, x)

    if s_grid.element_index != fem.NULL_ELEMENT_INDEX:
        active_cells[s_grid.element_index] = 1
~~~

Pass self._particle_environment from _create_geometry_partition.

- [ ] **Step 5: Use Warp environment-indexed PIC directly**

For non-GIMP construction, remove the manual _particle_grid_locations call and pass world positions and environment IDs to PicQuadrature:

~~~python
if self.gimp:
    particle_locations = self._particle_grid_locations_gimp(
        domain, positions, self._mpm_model.particle_radius, self._particle_environment
    )
    pic = fem.PicQuadrature(
        domain=domain,
        positions=particle_locations,
        measures=self._mpm_model.particle_volume,
        temporary_store=self.temporary_store,
        use_domain_element_indices=True,
    )
else:
    pic = fem.PicQuadrature(
        domain=domain,
        positions=positions,
        measures=self._mpm_model.particle_volume,
        env_indices=self._particle_environment,
        temporary_store=self.temporary_store,
        use_domain_element_indices=True,
    )
~~~

Delete _particle_grid_locations because no call remains.

- [ ] **Step 6: Pass environments through every GIMP corner lookup**

Add particle_environment to the method and dynamic-kernel arguments. Resolve the environment once per particle and use the matching element_partition_lookup overload:

~~~python
environment = int(particle_environment[p]) if particle_environment else 0
if particle_environment:
    sample = cell_lookup(domain_arg, corner, environment)
else:
    sample = cell_lookup(domain_arg, corner)
~~~

Apply this branch at every GIMP center/corner lookup in the kernel and add particle_environment to wp.launch inputs in signature order.

- [ ] **Step 7: Run dense PIC and GIMP tests**

Run:

~~~bash
uv run --extra dev -m newton.tests -k 'multiworld_dense_pic_matches_independent|multiworld_dense_gimp_matches_independent'
~~~

Expected: both pass on available basic devices.

- [ ] **Step 8: Register and run fixed-grid CUDA coverage**

Add:

~~~python
def test_multiworld_fixed_pic_matches_independent(test, device):
    _run_multiworld_reference_case(device, grid_type="fixed", integration_scheme="pic")
~~~

Register it only for get_cuda_test_devices(mode="basic").

Run:

~~~bash
uv run --extra dev -m newton.tests -k multiworld_fixed_pic_matches_independent
~~~

Expected: PASS on CUDA, or a normal unittest skip when no CUDA device is available.

- [ ] **Step 9: Commit dense/fixed world topology**

~~~bash
git add newton/tests/test_implicit_mpm.py newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py newton/_src/solvers/implicit_mpm/implicit_mpm_solver_kernels.py
git commit -m "Isolate implicit MPM dense world grids"
~~~

### Task 3: Build sparse multi-environment NanoVDB geometry

**Files:**
- Modify: newton/tests/test_implicit_mpm.py
- Modify: newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py:1206-1276
- Modify: newton/_src/solvers/implicit_mpm/implicit_mpm_solver_kernels.py:845-868

- [ ] **Step 1: Add sparse PIC and GIMP reference tests**

Add two wrappers around _run_multiworld_reference_case and register them only on CUDA basic devices:

~~~python
def test_multiworld_sparse_pic_matches_independent(test, device):
    _run_multiworld_reference_case(device, grid_type="sparse", integration_scheme="pic")


def test_multiworld_sparse_gimp_matches_independent(test, device):
    _run_multiworld_reference_case(device, grid_type="sparse", integration_scheme="gimp")
~~~

- [ ] **Step 2: Run sparse coverage before implementation**

Run:

~~~bash
uv run --extra dev -m newton.tests -k 'multiworld_sparse_pic_matches_independent|multiworld_sparse_gimp_matches_independent'
~~~

Expected on CUDA: FAIL because a plain Nanogrid does not expose multiple environments. Without CUDA: both are skipped.

- [ ] **Step 3: Extract and pad voxel coordinates per particle range**

Replace allocate_by_voxels with a coordinate-returning helper that retains the existing single-environment volume helper for compatibility:

~~~python
def particle_voxels(particle_q: wp.array, voxel_size: float, padding_voxels: int = 0) -> wp.array:
    volume = wp.Volume.allocate_by_voxels(voxel_points=particle_q.flatten(), voxel_size=voxel_size)
    for _ in range(padding_voxels):
        voxels = wp.empty(volume.get_voxel_count(), dtype=wp.vec3i, device=particle_q.device)
        volume.get_voxels(voxels)
        padded_voxels = wp.empty((voxels.shape[0], 3, 3, 3), dtype=wp.vec3i, device=particle_q.device)
        wp.launch(pad_voxels, dim=voxels.shape[0], inputs=[voxels, padded_voxels], device=particle_q.device)
        volume = wp.Volume.allocate_by_voxels(voxel_points=padded_voxels.flatten(), voxel_size=voxel_size)
    voxels = wp.empty(volume.get_voxel_count(), dtype=wp.vec3i, device=particle_q.device)
    volume.get_voxels(voxels)
    return voxels


def allocate_by_voxels(particle_q, voxel_size, padding_voxels: int = 0):
    voxels = particle_voxels(particle_q, voxel_size, padding_voxels)
    return wp.Volume.allocate_by_voxels(voxel_points=voxels, voxel_size=voxel_size)
~~~

Import particle_voxels into solver_implicit_mpm.py.

- [ ] **Step 4: Construct Nanogrid from environment voxel arrays**

In _allocate_grid, use the validated contiguous ranges from particle_world_start. Empty slices must produce empty wp.vec3i arrays on the model device.

~~~python
if self.grid_type == "sparse":
    if self._separate_worlds:
        starts = self.model.particle_world_start.numpy()
        cell_ijks = []
        for environment in range(self._environment_count):
            begin = int(starts[environment + 1])
            end = int(starts[environment + 2])
            if begin == end:
                cell_ijks.append(wp.empty(0, dtype=wp.vec3i, device=positions.device))
            else:
                cell_ijks.append(
                    particle_voxels(positions[begin:end], voxel_size, padding_voxels=padding_voxels)
                )
        grid = fem.Nanogrid.from_environment_voxels(
            cell_ijks,
            voxel_size=voxel_size,
            temporary_store=temporary_store,
            device=positions.device,
        )
    else:
        volume = allocate_by_voxels(positions, voxel_size, padding_voxels=padding_voxels)
        grid = fem.Nanogrid(volume, temporary_store=temporary_store)
~~~

Do not provide env_offsets or reduce Warp's default guard cells.

- [ ] **Step 5: Run sparse tests and the existing single-world sparse path**

Run:

~~~bash
uv run --extra dev -m newton.tests -k 'multiworld_sparse|test_implicit_mpm'
~~~

Expected: sparse multi-world tests pass on CUDA or skip without CUDA; existing implicit-MPM tests remain green.

- [ ] **Step 6: Commit sparse isolation**

~~~bash
git add newton/tests/test_implicit_mpm.py newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py newton/_src/solvers/implicit_mpm/implicit_mpm_solver_kernels.py
git commit -m "Build sparse implicit MPM world grids"
~~~

### Task 4: Batch rheology reductions and Krylov coefficients by world

**Files:**
- Modify: newton/tests/test_implicit_mpm.py
- Modify: newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py:2114-2165
- Modify: newton/_src/solvers/implicit_mpm/solve_rheology.py:71-117,201-245,497-557,1293-1330,1574-1675

- [ ] **Step 1: Add a finite-tolerance linear-solver reference test**

Use the same coincident-world helper with CR, a nonzero tolerance, and unequal per-world stiffness. Extend _make_mpm_particle_builder with a young_modulus argument and forward it into custom_attributes, then add:

~~~python
def test_multiworld_cr_matches_independent(test, device):
    _run_multiworld_reference_case(device, grid_type="dense", integration_scheme="pic", solver="cr")
~~~

For this wrapper set max_iterations=12 and tolerance=1.0e-5 in a local config override inside the reference helper when solver is cr. Register on basic devices.

- [ ] **Step 2: Run the CR test and inspect convergence mismatch**

Run:

~~~bash
uv run --extra dev -m newton.tests -k multiworld_cr_matches_independent
~~~

Expected: FAIL because LinearOperator currently performs one global reduction.

- [ ] **Step 3: Add strain environment offsets to RheologyData**

Add one optional node-offset field; None retains every existing one-system call site.

~~~python
@dataclass
class RheologyData:
    strain_mat: sp.BsrMatrix
    transposed_strain_mat: sp.BsrMatrix
    compliance_mat: sp.BsrMatrix
    strain_node_volume: wp.array[float]
    yield_params: wp.array[YieldParamVec]
    unilateral_strain_offset: wp.array[float]
    color_offsets: wp.array[int]
    color_blocks: wp.array2d[int]
    elastic_strain_delta: wp.array[vec6]
    plastic_strain_delta: wp.array[vec6]
    stress: wp.array[vec6]
    strain_environment_offsets: wp.array[int] | None = None
    has_viscosity: bool = False
    has_dilatancy: bool = False
    strain_velocity_node_count: int = -1
~~~

Pass scratch.strain_environment_offsets when solver_implicit_mpm.py constructs RheologyData.

- [ ] **Step 4: Convert node offsets to scalar vec6 offsets for Warp linear solvers**

Add a module constant and kernel, then allocate the scaled result in
_LinearSolver. Keeping the vec6 width named avoids repeating an unexplained
literal at operator construction sites.

~~~python
_STRESS_DOF_COUNT = 6


@wp.kernel
def scale_offsets(offsets: wp.array[int], scale: int, scaled_offsets: wp.array[int]):
    i = wp.tid()
    scaled_offsets[i] = offsets[i] * scale
~~~

In _LinearSolver.__init__:

~~~python
self._batch_offsets = None
if self.rheology.strain_environment_offsets is not None:
    self._batch_offsets = fem.borrow_temporary_like(
        self.rheology.strain_environment_offsets, temporary_store
    )
    wp.launch(
        scale_offsets,
        dim=self._batch_offsets.shape[0],
        inputs=[self.rheology.strain_environment_offsets, _STRESS_DOF_COUNT, self._batch_offsets],
    )

self.linear_operator = LinearOperator(
    shape=shape,
    dtype=dtype,
    device=device,
    matvec=self._delassus_matvec,
    batch_offsets=self._batch_offsets,
)
self.preconditioner = LinearOperator(
    shape=shape,
    dtype=dtype,
    device=device,
    matvec=self._preconditioner_matvec,
    batch_offsets=self._batch_offsets,
)
~~~

Release _batch_offsets when present in _LinearSolver.release.

- [ ] **Step 5: Make nonlinear residual reduction environment-aware**

Extend ArraySquaredNorm with optional batch_offsets and return shape (2, batch_count). Keep its current tree reduction when offsets is None. For batched input, use one cooperative block per environment and compute sum and maximum independently:

~~~python
@wp.kernel
def _batched_squared_norm_kernel(
    data: wp.array[float],
    batch_offsets: wp.array[int],
    result: wp.array2d[float],
):
    batch, lane = wp.tid()
    value_sum = float(0.0)
    value_max = float(0.0)
    for i in range(batch_offsets[batch] + lane, batch_offsets[batch + 1], wp.block_dim()):
        value = data[i]
        value_sum += value
        value_max = wp.max(value_max, value)
    wp.tile_store(result[0], wp.tile_sum(wp.tile(value_sum)), offset=batch)
    wp.tile_store(result[1], wp.tile_max(wp.tile(value_max)), offset=batch)
~~~

Use tile size 256 on CUDA and 1 on CPU. Construct ArraySquaredNorm with self.rheology.strain_environment_offsets in _RheologySolver.

Replace update_condition with a worst-environment check. A residual row contains squared L2 and squared Linf values; normalize L2 by 1 plus that environment's node count.

~~~python
@wp.kernel
def update_condition(
    residual_threshold: float,
    batch_offsets: wp.array[int],
    total_node_count: int,
    solve_granularity: int,
    max_iterations: int,
    residual: wp.array2d[float],
    iteration: wp.array[int],
    condition: wp.array[int],
):
    cur_it = iteration[0] + solve_granularity
    converged = True
    for batch in range(residual.shape[1]):
        node_count = wp.where(batch_offsets, batch_offsets[batch + 1] - batch_offsets[batch], total_node_count)
        converged = converged and residual[0, batch] < residual_threshold * float(1 + node_count)
        converged = converged and residual[1, batch] < residual_threshold
    iteration[0] = cur_it
    condition[0] = wp.where(converged or cur_it > max_iterations, 0, 1)
~~~

Pass self.size as total_node_count at the graph-loop launch. For the non-graph
host loop, compute normalized arrays from residual.numpy() and the node
offsets, then terminate only when every batch satisfies both thresholds.
Verbose output reports the maximum normalized L2 and Linf across worlds.

- [ ] **Step 6: Use a per-environment absolute tolerance scale for linear methods**

In solve_rheology, derive the scale from the largest environment node span when offsets are present; otherwise retain the existing total size:

~~~python
if rheology.strain_environment_offsets is None:
    tolerance_scale = math.sqrt(1.0 + rheology.stress.shape[0])
else:
    offsets = rheology.strain_environment_offsets.numpy()
    tolerance_scale = math.sqrt(1.0 + int(np.max(np.diff(offsets), initial=0)))
~~~

Add import numpy as np at the top of solve_rheology.py.

- [ ] **Step 7: Run nonlinear and linear reference tests**

Run:

~~~bash
uv run --extra dev -m newton.tests -k 'multiworld_dense_pic_matches_independent|multiworld_cr_matches_independent'
~~~

Expected: both Jacobi and CR reference tests pass.

- [ ] **Step 8: Commit batched convergence**

~~~bash
git add newton/tests/test_implicit_mpm.py newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py newton/_src/solvers/implicit_mpm/solve_rheology.py
git commit -m "Batch implicit MPM rheology solves by world"
~~~

### Task 5: Assign and validate collider worlds during model setup

**Files:**
- Modify: newton/tests/test_implicit_mpm.py
- Modify: newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py:984-1035
- Modify: newton/_src/solvers/implicit_mpm/implicit_mpm_model.py:384-605
- Modify: newton/_src/solvers/implicit_mpm/rasterized_collisions.py:41-70

- [ ] **Step 1: Add collider metadata policy tests**

Create a small box mesh helper and test custom-world validation, global dynamic rejection, and global kinematic acceptance:

~~~python
def _box_collider_mesh(device):
    mesh = newton.Mesh.create_box(
        0.25,
        0.05,
        0.25,
        duplicate_vertices=False,
        compute_normals=False,
        compute_uvs=False,
        compute_inertia=False,
    )
    return wp.Mesh(points=wp.array(mesh.vertices, dtype=wp.vec3, device=device), indices=wp.array(mesh.indices, dtype=int, device=device))


def test_multiworld_collider_world_validation(test, device):
    source = _make_mpm_particle_builder(gravity=0.0)
    builder = newton.ModelBuilder(gravity=0.0)
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_world(source)
    builder.add_world(source)
    model = builder.finalize(device=device)
    solver = SolverImplicitMPM(model, _make_mpm_config())
    with test.assertRaisesRegex(ValueError, "collider world ID"):
        solver.setup_collider(collider_meshes=[_box_collider_mesh(device)], collider_world_ids=[2])


def test_multiworld_global_dynamic_collider_rejected(test, device):
    source = _make_mpm_particle_builder(gravity=0.0)
    scene = newton.ModelBuilder(gravity=0.0)
    SolverImplicitMPM.register_custom_attributes(scene)
    scene.add_world(source)
    scene.add_world(source)
    body = scene.add_body(mass=1.0)
    scene.add_shape_box(body, hx=0.25, hy=0.05, hz=0.25)
    model = scene.finalize(device=device)
    with test.assertRaisesRegex(ValueError, "global dynamic collider"):
        SolverImplicitMPM(model, _make_mpm_config())
~~~

Register on basic devices.

- [ ] **Step 2: Run metadata tests before adding the API**

Run:

~~~bash
uv run --extra dev -m newton.tests -k 'multiworld_collider_world_validation|multiworld_global_dynamic_collider_rejected'
~~~

Expected: the first test errors on an unknown keyword and the second does not raise the intended error.

- [ ] **Step 3: Extend the public and internal collider signatures**

Add collider_world_ids immediately before model in both setup_collider signatures, keeping it keyword-only in the public method because all parameters after self are passed through keyword use:

~~~python
collider_world_ids: list[int] | None = None,
~~~

Forward the value unchanged and document: custom mesh defaults global; body collider infers body_world; valid values are -1 through world_count - 1.

- [ ] **Step 4: Extend Collider with stable-ID query metadata**

Add arrays to the Warp struct:

~~~python
collider_world: wp.array[int]
"""Newton world ID for each stable collider ID; -1 denotes global."""

collider_face_offset: wp.array[int]
"""First face-material entry for each stable collider ID."""

query_collider_ids: wp.array[int]
"""Stable collider IDs grouped as global, then world 0 through world N-1."""

query_world_offsets: wp.array[int]
"""Offsets into query_collider_ids for global and each local world group."""
~~~

- [ ] **Step 5: Group default static shapes by shape_world and infer all collider IDs**

In ImplicitMPMModel.setup_collider, replace the body_shapes dictionary with an aligned collider_shapes list. For default discovery:

~~~python
if collider_body_ids is None and collider_meshes is None:
    collider_body_ids = []
    collider_meshes = []
    inferred_world_ids = []
    collider_shapes = []

    shape_world = model.shape_world.numpy()
    static_shapes = _get_body_collision_shapes(model, -1)
    for world_id in sorted(set(int(shape_world[s]) for s in static_shapes)):
        world_shapes = [s for s in static_shapes if int(shape_world[s]) == world_id]
        collider_body_ids.append(-1)
        collider_meshes.append(None)
        inferred_world_ids.append(world_id)
        collider_shapes.append(world_shapes)

    body_world = model.body_world.numpy()
    for body_id in range(model.body_count):
        shapes = _get_body_collision_shapes(model, body_id)
        if shapes:
            collider_body_ids.append(body_id)
            collider_meshes.append(None)
            inferred_world_ids.append(int(body_world[body_id]))
            collider_shapes.append(shapes)
    collider_world_ids = inferred_world_ids
~~~

For explicit inputs, infer body-backed entries from body_world, assign custom meshes -1, and allow supplied collider_world_ids to override only custom/static entries. Validate list lengths and every ID. If model is not self.model in isolated mode, require model.world_count == self.model.world_count.

- [ ] **Step 6: Reject global dynamic colliders and create grouped indirection**

After body arrays are resolved, inspect body mass for each global body-backed collider. Raise:

~~~python
raise ValueError(
    "SolverImplicitMPM cannot isolate a global dynamic collider; replicate the body into each world, "
    "make it static or kinematic, or set Config.separate_worlds=False for legacy coupled behavior."
)
~~~

Build metadata without reordering collider meshes so collider IDs and collider_body_index remain stable:

~~~python
face_counts = np.array([mesh.indices.shape[0] // 3 for mesh in collider_meshes], dtype=np.int32)
face_offsets = np.concatenate(([0], np.cumsum(face_counts, dtype=np.int32)))
grouped_ids = []
query_offsets = [0]
for world_id in (-1, *range(self.model.world_count)):
    grouped_ids.extend(i for i, collider_world in enumerate(collider_world_ids) if collider_world == world_id)
    query_offsets.append(len(grouped_ids))

self.collider.collider_world = wp.array(collider_world_ids, dtype=int)
self.collider.collider_face_offset = wp.array(face_offsets[:-1], dtype=int)
self.collider.query_collider_ids = wp.array(grouped_ids, dtype=int)
self.collider.query_world_offsets = wp.array(query_offsets, dtype=int)
~~~

- [ ] **Step 7: Run metadata tests**

Run the Step 2 command.

Expected: both tests pass.

- [ ] **Step 8: Commit collider-world setup**

~~~bash
git add newton/tests/test_implicit_mpm.py newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py newton/_src/solvers/implicit_mpm/implicit_mpm_model.py newton/_src/solvers/implicit_mpm/rasterized_collisions.py
git commit -m "Assign implicit MPM colliders to worlds"
~~~

### Task 6: Filter rasterization and direct projection by world

**Files:**
- Modify: newton/tests/test_implicit_mpm.py
- Modify: newton/_src/solvers/implicit_mpm/rasterized_collisions.py:155-430,538-610
- Modify: newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py:1103-1135,1697-1725

- [ ] **Step 1: Add world-local and global collider behavior tests**

Build two coincident worlds with one particle each above two coincident kinematic platforms. Give platform 0 velocity +X and platform 1 velocity -X, step once with high friction, and assert opposite particle x velocities. Build a separate scene with one global ground plane and assert both worlds receive upward contact response. Use the same scene for project_outside by placing both particles below their matching platforms and asserting both are projected.

The core assertions are:

~~~python
starts = model.particle_world_start.numpy()
velocity = state.particle_qd.numpy()
test.assertGreater(velocity[starts[1], 0], 0.0)
test.assertLess(velocity[starts[2], 0], 0.0)

solver.project_outside(state, state, dt=0.01)
projected_y = state.particle_q.numpy()[:, 1]
test.assertTrue(np.all(projected_y >= -1.0e-5))
~~~

Name the tests test_multiworld_local_colliders_are_isolated and test_multiworld_global_static_collider_applies_to_all. Register them on CUDA basic devices because mesh query behavior is the production GPU path.

- [ ] **Step 2: Run collider behavior tests before filtering**

Run:

~~~bash
uv run --extra dev -m newton.tests -k 'multiworld_local_colliders_are_isolated|multiworld_global_static_collider_applies_to_all'
~~~

Expected on CUDA: the local-collider test fails because every node scans both platforms; without CUDA the tests skip.

- [ ] **Step 3: Query stable collider IDs by global and local ranges**

Add a helper that performs the existing mesh query for one stable ID and updates the best result, then make collision_sdf accept environment_index. For isolated environment e, iterate ranges [offsets[0], offsets[1]) and [offsets[e+1], offsets[e+2]); for the legacy sentinel -2, iterate every query_collider_ids entry. Use collider_face_offset[collider_id] + query.face for material lookup.

Extract the current per-mesh block from collision_sdf into an @wp.func named
_query_collider_id. Its arguments are the stable collider ID, query point,
Collider, rigid-body arrays, dt, and the current minimum/result tuple; it
returns the updated tuple in this order: min_sdf, sdf_grad, sdf_vel,
closest_point, closest_collider_id, material_id. The existing rigid-motion
post-processing remains in collision_sdf after range traversal. The selection
structure is:

~~~python
if environment_index == -2:
    for query_index in range(collider.query_collider_ids.shape[0]):
        collider_id = collider.query_collider_ids[query_index]
        min_sdf, sdf_grad, sdf_vel, closest_point, closest_collider_id, material_id = _query_collider_id(
            collider_id, x, collider, min_sdf, sdf_grad, sdf_vel, closest_point, closest_collider_id, material_id
        )
else:
    for query_index in range(collider.query_world_offsets[0], collider.query_world_offsets[1]):
        collider_id = collider.query_collider_ids[query_index]
        min_sdf, sdf_grad, sdf_vel, closest_point, closest_collider_id, material_id = _query_collider_id(
            collider_id, x, collider, min_sdf, sdf_grad, sdf_vel, closest_point, closest_collider_id, material_id
        )
    begin = collider.query_world_offsets[environment_index + 1]
    end = collider.query_world_offsets[environment_index + 2]
    for query_index in range(begin, end):
        collider_id = collider.query_collider_ids[query_index]
        min_sdf, sdf_grad, sdf_vel, closest_point, closest_collider_id, material_id = _query_collider_id(
            collider_id, x, collider, min_sdf, sdf_grad, sdf_vel, closest_point, closest_collider_id, material_id
        )
~~~

Implement the repeated body as an @wp.func returning the updated minimum/result tuple so the three loops call one implementation. Keep collider_id as the stable original index.

- [ ] **Step 4: Derive raster node environment from partition offsets**

Add a binary-search helper:

~~~python
@wp.func
def environment_from_offsets(index: int, offsets: wp.array[int]) -> int:
    return wp.lower_bound(offsets, 0, offsets.shape[0], index + 1) - 1
~~~

Extend rasterize_collider and its kernel with node_environment_offsets. In the kernel:

~~~python
environment = environment_from_offsets(i, node_environment_offsets) if node_environment_offsets else -2
sdf, sdf_gradient, sdf_vel, collider_id, material_id = collision_sdf(
    x, environment, collider, body_q, body_qd, body_q_prev, dt
)
~~~

Pass scratch.collider_environment_offsets from solver_implicit_mpm.py.

- [ ] **Step 5: Filter project_outside with each particle's environment**

Extend project_outside_collider with particle_environment and use:

~~~python
environment = int(particle_environment[i]) if particle_environment else -2
sdf, sdf_gradient, sdf_vel, _collider_id, material_id = collision_sdf(
    pos_adv, environment, collider, body_q, body_qd, body_q_prev, dt
)
~~~

Pass self._particle_environment from SolverImplicitMPM.project_outside.

- [ ] **Step 6: Run collider tests and existing collider velocity regression**

Run:

~~~bash
uv run --extra dev -m newton.tests -k 'multiworld_local_colliders|multiworld_global_static_collider|finite_difference_collider_velocity'
~~~

Expected: all available tests pass; CUDA-only tests skip normally without a driver.

- [ ] **Step 7: Commit query filtering**

~~~bash
git add newton/tests/test_implicit_mpm.py newton/_src/solvers/implicit_mpm/rasterized_collisions.py newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py
git commit -m "Filter implicit MPM colliders by world"
~~~

### Task 7: Preserve grain rendering across FEM environments

**Files:**
- Modify: newton/tests/test_implicit_mpm.py
- Modify: newton/_src/solvers/implicit_mpm/render_grains.py:130-205
- Modify: newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py:1140-1190

- [ ] **Step 1: Add a multi-world grain update test**

After stepping the dense two-world model, sample two grains per particle, update them, and verify finite positions and unchanged shape:

~~~python
def test_multiworld_render_grains(test, device):
    source = _make_mpm_particle_builder(gravity=0.0, velocity=(0.1, 0.0, 0.0))
    builder = newton.ModelBuilder(gravity=0.0)
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_world(source)
    builder.add_world(source)
    model = builder.finalize(device=device)
    config = _make_mpm_config()
    solver = SolverImplicitMPM(model, config)
    state_0 = model.state()
    state_1 = model.state()
    grains = solver.sample_render_grains(state_0, grains_per_particle=2)
    solver.step(state_0, state_1, control=None, contacts=None, dt=0.01)
    solver.update_render_grains(state_0, state_1, grains, dt=0.01)
    test.assertEqual(grains.shape, (model.particle_count, 2))
    test.assertTrue(np.isfinite(grains.numpy()).all())
~~~

Register on basic devices.

- [ ] **Step 2: Run the grain test before passing environment IDs**

Run:

~~~bash
uv run --extra dev -m newton.tests -k multiworld_render_grains
~~~

Expected: FAIL when PicQuadrature requires environment indices for the multi-environment grid.

- [ ] **Step 3: Repeat particle environments for grain samples**

Extend update_render_grains with optional particle_environment and pass it from SolverImplicitMPM. Add a kernel:

~~~python
@wp.kernel
def repeat_particle_environment(
    particle_environment: wp.array[int],
    grains_per_particle: int,
    grain_environment: wp.array[int],
):
    particle, grain = wp.tid()
    grain_environment[particle * grains_per_particle + grain] = particle_environment[particle]
~~~

Allocate grain_environment only when particle_environment is present, launch over grains.shape, and pass it as env_indices to the grain PicQuadrature. Release the temporary after interpolation.

- [ ] **Step 4: Run the grain test**

Run the Step 2 command.

Expected: PASS.

- [ ] **Step 5: Commit grain isolation**

~~~bash
git add newton/tests/test_implicit_mpm.py newton/_src/solvers/implicit_mpm/render_grains.py newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py
git commit -m "Preserve worlds in implicit MPM grain rendering"
~~~

### Task 8: Add the quick CUDA efficiency benchmark

**Files:**
- Create: asv/benchmarks/simulation/bench_implicit_mpm.py

- [ ] **Step 1: Create a directly runnable ASV benchmark**

Implement FastImplicitMPMMultiworld with params world_count=[1,8,32] and layout=["multiworld","independent"]. Use one deterministic 2x2x2 particle block per world, fixed grid, tolerance zero, four solver iterations, and five timed steps. setup constructs, warms once, and synchronizes. time_step performs all layout work and exactly one wp.synchronize_device at the tail. number=1, repeat=3, rounds=2.

The class contract is:

~~~python
class FastImplicitMPMMultiworld:
    params = ([1, 8, 32], ["multiworld", "independent"])
    param_names = ["world_count", "layout"]
    number = 1
    repeat = 3
    rounds = 2
    steps = 5

    def setup(self, world_count, layout):
        skip_benchmark_if(not wp.is_cuda_available())
        self.runners = self._make_multiworld(world_count) if layout == "multiworld" else [
            self._make_single_world() for _ in range(world_count)
        ]
        self._step_all()
        wp.synchronize_device()

    def time_step(self, world_count, layout):
        self._step_all()
        wp.synchronize_device()

    def track_milliseconds_per_world_step(self, world_count, layout):
        start = time.perf_counter()
        self._step_all()
        wp.synchronize_device()
        return 1000.0 * (time.perf_counter() - start) / (world_count * self.steps)
~~~

Each runner stores model, solver, two states, and swaps states after every step. Do not report a speedup inside one parameter row because ASV compares layout rows directly.

Add the standard __main__ parser used by neighboring simulation benchmarks and call newton.utils.run_benchmark.

- [ ] **Step 2: Compile/import the benchmark without a GPU**

Run:

~~~bash
uv run --extra dev python -m py_compile asv/benchmarks/simulation/bench_implicit_mpm.py
uv run --extra dev python -c 'from asv.benchmarks.simulation.bench_implicit_mpm import FastImplicitMPMMultiworld; print(FastImplicitMPMMultiworld.param_names)'
~~~

Expected: both commands exit zero and print ['world_count', 'layout'].

- [ ] **Step 3: Run the quick benchmark when CUDA is available**

Run:

~~~bash
uv run --extra dev python asv/benchmarks/simulation/bench_implicit_mpm.py --bench FastImplicitMPMMultiworld
~~~

Expected with CUDA: both layouts run for requested world counts and report time plus milliseconds per world-step. Without CUDA: report the benchmark skip and retain the compile/import evidence.

- [ ] **Step 4: Commit the benchmark**

~~~bash
git add asv/benchmarks/simulation/bench_implicit_mpm.py
git commit -m "Benchmark implicit MPM multi-world batching"
~~~

### Task 9: Document behavior and complete verification

**Files:**
- Modify: docs/concepts/worlds.rst
- Modify: CHANGELOG.md
- Verify all modified Python and RST files

- [ ] **Step 1: Add public world semantics**

In docs/concepts/worlds.rst after the colocation guidance, add:

~~~rst
Implicit MPM worlds
~~~~~~~~~~~~~~~~~~~

:class:`newton.solvers.SolverImplicitMPM` uses independent Warp FEM
environments for models with multiple worlds. Worlds can occupy identical
physics-space coordinates without sharing grid mass, momentum, stress, or
collider response. This isolation is enabled by default; set
``SolverImplicitMPM.Config.separate_worlds = False`` only when legacy
shared-grid coupling is intentional.

MPM particles in an isolated multi-world model must belong to a local world.
Global particles are ambiguous because one particle state cannot be updated
independently by several FEM environments. Global static or kinematic
colliders remain valid and affect every world; dynamic collider bodies must be
replicated into their worlds.

Dense and fixed grids use common physical bounds for every world. Prefer the
sparse grid for heterogeneous or physically separated particle bounds so each
environment allocates only its active voxels. Sparse NanoVDB reconstruction
remains outside CUDA graph capture.
~~~

- [ ] **Step 2: Add the changelog entry**

Under Unreleased / Added add:

~~~markdown
- Add independent multi-world grids, rheology solves, collider filtering, and grain interpolation to SolverImplicitMPM.
~~~

- [ ] **Step 3: Run focused CPU tests from a clean process**

Run:

~~~bash
uv run --extra dev -m newton.tests -k 'test_implicit_mpm|multiworld'
~~~

Expected: all CPU/basic implicit-MPM tests pass; CUDA-only cases are explicitly skipped on a machine without a CUDA driver.

- [ ] **Step 4: Run the complete implicit-MPM module**

Run:

~~~bash
uv run --extra dev -m newton.tests -k test_implicit_mpm
~~~

Expected: zero failures.

- [ ] **Step 5: Run formatting, lint, and static file checks**

Run:

~~~bash
uvx pre-commit run --files newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py newton/_src/solvers/implicit_mpm/implicit_mpm_solver_kernels.py newton/_src/solvers/implicit_mpm/solve_rheology.py newton/_src/solvers/implicit_mpm/implicit_mpm_model.py newton/_src/solvers/implicit_mpm/rasterized_collisions.py newton/_src/solvers/implicit_mpm/render_grains.py newton/tests/test_implicit_mpm.py asv/benchmarks/simulation/bench_implicit_mpm.py docs/concepts/worlds.rst CHANGELOG.md
~~~

Expected: every hook passes. Apply formatter-generated changes, rerun affected tests, and rerun this command until clean.

- [ ] **Step 6: Verify scope and dependency invariants**

Run:

~~~bash
git status --short
git diff --check HEAD
git diff HEAD -- uv.lock pyproject.toml
git -C /home/maximiliank/Work/warp-main-multiworld-reference status --short
~~~

Expected: only intended Newton files and the ignored design/plan records are present; diff check is empty; dependency files have no diff; Warp reference checkout is clean.

- [ ] **Step 7: Request two-stage code review and address findings**

Use superpowers:requesting-code-review. First ask a reviewer to compare the implementation with the approved design and acceptance criteria; after any corrections and reruns, ask a second reviewer to inspect code quality, API compatibility, world indexing, graph behavior, and test rigor.

- [ ] **Step 8: Commit documentation and final corrections**

~~~bash
git add docs/concepts/worlds.rst CHANGELOG.md
git add -u
git commit -m "Document implicit MPM multi-world isolation"
~~~

- [ ] **Step 9: Perform final verification immediately before handoff**

Use superpowers:verification-before-completion and rerun the complete implicit-MPM test command, the exact pre-commit command, git diff --check HEAD, and git status --short. Record which CUDA cases were executed and which were skipped because no driver is available. Do not push any branch or tag.
