# IsaacLab Franka Pour Multi-World MPM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a clean `max/franka-pour-multiworld-mpm` branch that consumes the paired Warp/Newton stack and runs Franka Pour with isolated rebuildable sparse MPM across RL environments.

**Architecture:** Start from committed coupled-manager work, port only the Franka Pour task and its focused tests, delegate graph/reset behavior to Newton's public solver contracts, and resolve sparse capacity after the final RL environment count is known. The stable Torch `DirectRLEnv` loop remains outside the captured physics graph.

**Tech Stack:** IsaacLab, Isaac Sim, Newton, Warp, MJWarp, PyTorch, `pytest`, CUDA graphs.

---

### Task 1: Create the clean IsaacLab worktree

**Files:**
- Source: `/home/maximiliank/Work/IsaacLab`
- Read-only source material: `/home/maximiliank/Work/IsaacLab-coupling`
- Worktree: `/home/maximiliank/.config/superpowers/worktrees/IsaacLab/max-franka-pour-multiworld-mpm`

- [ ] **Step 1: Verify the committed base and preserve dirty work**

```bash
test "$(git -C /home/maximiliank/Work/IsaacLab rev-parse max/newton-coupling-manager)" = 80d2b8b42bc793c82ffb060cef394b14f5953cb3
git -C /home/maximiliank/Work/IsaacLab-coupling status --short > /tmp/isaaclab-coupling-preserved-status.txt
```

Expected: exact base SHA; the dirty status is recorded but never modified.

- [ ] **Step 2: Create the isolated branch**

```bash
git -C /home/maximiliank/Work/IsaacLab worktree add \
  /home/maximiliank/.config/superpowers/worktrees/IsaacLab/max-franka-pour-multiworld-mpm \
  -b max/franka-pour-multiworld-mpm \
  80d2b8b42bc793c82ffb060cef394b14f5953cb3
```

Expected: clean worktree on the requested branch.

### Task 2: Expose Newton's solver-owned graph lifecycle

**Files:**
- Modify: `source/isaaclab_newton/isaaclab_newton/physics/newton_manager.py`
- Modify: `source/isaaclab_newton/isaaclab_newton/physics/mpm_manager.py`
- Modify: `source/isaaclab_newton/isaaclab_newton/physics/coupled_manager.py`
- Test: `source/isaaclab_newton/test/physics/test_newton_manager_abstraction.py`
- Modify: `docs/source/overview/core-concepts/physical-backends/newton/newton-manager-abstraction.rst`

- [ ] **Step 1: Add failing capability/preparation tests**

Use fake solvers with public properties and a preparation counter:

```python
class GraphAwareSolver:
    def __init__(self, supported):
        self.supports_cuda_graph_capture = supported
        self.prepared = []

    def prepare_cuda_graph_capture(self, contacts=None):
        self.prepared.append(contacts)
```

Assert `_supports_cuda_graph_capture()` mirrors the solver, unsupported solvers stay eager, and supported solvers receive exactly one `prepare_cuda_graph_capture(_contacts)` call before `wp.ScopedCapture` begins.

- [ ] **Step 2: Run RED**

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab_newton/test/physics/test_newton_manager_abstraction.py \
  -k 'solver_owned_graph or prepare_cuda_graph_capture' -vv
```

Expected: FAIL because the base manager returns `True` unconditionally and never calls the public preparation hook.

- [ ] **Step 3: Delegate capability and preparation**

Implement:

```python
@classmethod
def _supports_cuda_graph_capture(cls) -> bool:
    return bool(cls._solver is not None and cls._solver.supports_cuda_graph_capture)
```

Before either immediate or deferred capture, call `cls._solver.prepare_cuda_graph_capture(cls._contacts)` only after capability succeeds and before entering a capture window. Remove `NewtonMPMManager`'s fixed-only private heuristic and remove coupled manager's private direct `prepare_contacts` shortcut; Newton recursively prepares coupled entries.

- [ ] **Step 4: Run GREEN and commit**

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab_newton/test/physics/test_newton_manager_abstraction.py \
  -k 'cuda_graph or graph_capture' -vv
git add source/isaaclab_newton/isaaclab_newton/physics/newton_manager.py \
  source/isaaclab_newton/isaaclab_newton/physics/mpm_manager.py \
  source/isaaclab_newton/isaaclab_newton/physics/coupled_manager.py \
  source/isaaclab_newton/test/physics/test_newton_manager_abstraction.py \
  docs/source/overview/core-concepts/physical-backends/newton/newton-manager-abstraction.rst
git commit -m "Delegate Newton graph capture lifecycle"
```

Expected: selected tests PASS.

### Task 3: Add a public masked solver reset seam

**Files:**
- Modify: `source/isaaclab_newton/isaaclab_newton/physics/newton_manager.py`
- Test: `source/isaaclab_newton/test/physics/test_newton_manager_abstraction.py`

- [ ] **Step 1: Add RED reset tests**

Test the proposed API:

```python
NewtonManager.reset_solver_state(state=state, world_mask=world_mask, flags=flags)
```

Assert exact forwarding to `solver.reset`, omitted `state` uses current state 0, mask identity is preserved, and calling before solver initialization raises `RuntimeError("Newton solver is not initialized")`.

- [ ] **Step 2: Run RED**

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab_newton/test/physics/test_newton_manager_abstraction.py \
  -k reset_solver_state -vv
```

Expected: FAIL because the public manager method does not exist.

- [ ] **Step 3: Implement and run GREEN**

```python
@classmethod
def reset_solver_state(cls, state=None, world_mask=None, flags=None) -> None:
    if cls._solver is None:
        raise RuntimeError("Newton solver is not initialized")
    if state is None:
        state = cls._state_0
    cls._solver.reset(state, world_mask=world_mask, flags=flags)
```

Run the RED command again; expected PASS.

- [ ] **Step 4: Commit**

```bash
git add source/isaaclab_newton/isaaclab_newton/physics/newton_manager.py \
  source/isaaclab_newton/test/physics/test_newton_manager_abstraction.py
git commit -m "Expose masked Newton solver reset"
```

### Task 4: Port coupled implicit-MPM correctness fixes

**Files:**
- Modify: `source/isaaclab_newton/isaaclab_newton/physics/coupled_manager.py`
- Modify: `source/isaaclab_newton/test/physics/test_newton_manager_abstraction.py`
- Modify: `source/isaaclab_newton/changelog.d/coupled-pr2848-refresh.minor.rst`

- [ ] **Step 1: Add RED tests from the dirty source material**

Port only the tests that assert: kinematic coupled MPM bodies become massless before finalization; an MPM entry enables FK-before-step; `project_outside_colliders=True` invokes projection per substep; `False` does not; teardown clears selected entry names.

- [ ] **Step 2: Run RED**

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab_newton/test/physics/test_newton_manager_abstraction.py \
  -k 'coupled and (kinematic or project_outside or fk)' -vv
```

Expected: the new coupled MPM tests FAIL against the committed base.

- [ ] **Step 3: Port only the required implementation**

Apply the functional diff from the read-only `IsaacLab-coupling` `coupled_manager.py`: clear mass/inertia for kinematic bodies, set `_needs_fk_before_step` for MPM entries, collect MPM entries that request projection, invoke `project_outside` after each coupled substep, and clear class state on teardown. Do not port render, video, Scoop, demo, or scratch changes.

- [ ] **Step 4: Run GREEN and commit**

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab_newton/test/physics/test_newton_manager_abstraction.py \
  -k 'coupled and (kinematic or project_outside or fk)' -vv
git add source/isaaclab_newton/isaaclab_newton/physics/coupled_manager.py \
  source/isaaclab_newton/test/physics/test_newton_manager_abstraction.py \
  source/isaaclab_newton/changelog.d/coupled-pr2848-refresh.minor.rst
git commit -m "Fix coupled implicit MPM colliders"
```

Expected: selected tests PASS.

### Task 5: Forward the isolated-world MPM configuration

**Files:**
- Modify: `source/isaaclab_newton/isaaclab_newton/physics/mpm_manager_cfg.py`
- Test: `source/isaaclab_newton/test/physics/test_newton_manager_abstraction.py`

- [ ] **Step 1: Add RED configuration test**

```python
cfg = MPMSolverCfg(separate_worlds=False, grid_type="sparse", max_active_cell_count=1234)
solver_cfg = cfg.to_solver_config()
assert solver_cfg.separate_worlds is False
assert solver_cfg.max_active_cell_count == 1234
```

- [ ] **Step 2: Run RED**

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab_newton/test/physics/test_newton_manager_abstraction.py \
  -k mpm_solver_cfg -vv
```

Expected: FAIL because `MPMSolverCfg` does not expose `separate_worlds`.

- [ ] **Step 3: Implement and run GREEN**

Add `"separate_worlds"` to `_SOLVER_CONFIG_FIELDS` and:

```python
separate_worlds: bool = True
"""Use independent FEM environments for Newton worlds; disable only for legacy shared-grid behavior."""
```

Update `max_active_cell_count` docs to cover dense, fixed, and rebuildable sparse total capacity. Rerun the RED command; expected PASS.

- [ ] **Step 4: Commit**

```bash
git add source/isaaclab_newton/isaaclab_newton/physics/mpm_manager_cfg.py \
  source/isaaclab_newton/test/physics/test_newton_manager_abstraction.py
git commit -m "Expose isolated MPM world configuration"
```

### Task 6: Port Franka Pour helpers before the task

**Files:**
- Create tests: `source/isaaclab_tasks/test/contrib/test_franka_pour_cube_bowl_mesh.py`
- Create tests: `source/isaaclab_tasks/test/contrib/test_franka_pour_media_fill.py`
- Create: `source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour/__init__.py`
- Create: `source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour/cube_bowl_mesh.py`
- Create: `source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour/media_fill.py`
- Create: `source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour/cup_media.py`

- [ ] **Step 1: Port the two tests with `apply_patch` only**

Use the exact files from the read-only coupling checkout. Do not copy `can_mesh.py`, `data/can_i01_open.npz`, caches, or scratch files.

- [ ] **Step 2: Run RED**

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab_tasks/test/contrib/test_franka_pour_cube_bowl_mesh.py \
  source/isaaclab_tasks/test/contrib/test_franka_pour_media_fill.py -vv
```

Expected: collection FAILS because the `franka_pour` helper modules do not exist.

- [ ] **Step 3: Port the four helper files with `apply_patch`**

Use the corresponding source-material files verbatim, except for formatting demanded by current hooks.

- [ ] **Step 4: Run GREEN and commit**

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab_tasks/test/contrib/test_franka_pour_cube_bowl_mesh.py \
  source/isaaclab_tasks/test/contrib/test_franka_pour_media_fill.py -vv
git add source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour \
  source/isaaclab_tasks/test/contrib/test_franka_pour_cube_bowl_mesh.py \
  source/isaaclab_tasks/test/contrib/test_franka_pour_media_fill.py
git commit -m "Add Franka Pour media geometry"
```

Expected: both helper suites PASS.

### Task 7: Port task configuration, MDP, registration, and capacity policy

**Files:**
- Create: `source/isaaclab_tasks/test/contrib/test_franka_pour_mdp.py`
- Create: `source/isaaclab_tasks/test/contrib/test_franka_pour_env_cfg.py`
- Create: `source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour/config/__init__.py`
- Create: `source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour/config/franka/__init__.py`
- Create: `source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour/config/franka/agents/__init__.py`
- Create: `source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour/config/franka/agents/rsl_rl_ppo_cfg.py`
- Create: `source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour/mdp/__init__.py`
- Create: `source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour/mdp/events.py`
- Create: `source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour/mdp/observations.py`
- Create: `source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour/mdp/rewards.py`
- Create: `source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour/mdp/terminations.py`
- Create: `source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour/pour_env_cfg.py`

- [ ] **Step 1: Port tests first and add capacity cases**

Add expected sparse capacities: 1 env → 16,000; 4 → 16,000; 8 → 32,000; 64 → 120,000; play floor → 24,000; explicit override unchanged; fixed grid retains configured ceiling. Assert MPM entry `separate_worlds=True`, positive capacity, requested CUDA graph, unique body ownership, and correct proxy labels.

- [ ] **Step 2: Run RED**

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab_tasks/test/contrib/test_franka_pour_mdp.py \
  source/isaaclab_tasks/test/contrib/test_franka_pour_env_cfg.py -vv
```

Expected: collection FAILS because task configuration and MDP modules are absent.

- [ ] **Step 3: Port source packages and add a pure resolver**

Port the listed source-material files. Add:

```python
def resolve_mpm_active_cell_count(
    *,
    grid_type: str,
    num_envs: int,
    cells_per_env: int,
    minimum: int,
    maximum: int,
    override: int | None,
) -> int:
    if override is not None:
        return override
    if grid_type != "sparse":
        return maximum
    return min(maximum, max(minimum, cells_per_env * num_envs))
```

Set `separate_worlds=True`, `grid_padding=0`, and `solver="jacobi"` for the graph-captured sparse MPM entry. Keep the resolver pure in this task; the environment applies it after Hydra/CLI selects the final `scene.num_envs` in Task 8.

- [ ] **Step 4: Run GREEN and commit**

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab_tasks/test/contrib/test_franka_pour_mdp.py \
  source/isaaclab_tasks/test/contrib/test_franka_pour_env_cfg.py -vv
git add source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour \
  source/isaaclab_tasks/test/contrib/test_franka_pour_mdp.py \
  source/isaaclab_tasks/test/contrib/test_franka_pour_env_cfg.py
git commit -m "Configure multi-world Franka Pour"
```

Expected: both suites PASS.

### Task 8: Port the environment with public selective reset

**Files:**
- Create: `source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour/pour_env.py`
- Create: `source/isaaclab_tasks/changelog.d/franka-pour.minor.rst`
- Test: `source/isaaclab_tasks/test/contrib/test_franka_pour_env_cfg.py`
- Test: `source/isaaclab_newton/test/assets/test_mpm_object.py`

- [ ] **Step 1: Add RED reset regression**

Create two worlds with different particle history, reset world 0, and assert world 0 returns to identity/zero/one MPM rest fields while world 1 parent and entry-local arrays are bitwise unchanged. Assert task source contains no `_solver._entries` access.

- [ ] **Step 2: Run RED**

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab_newton/test/assets/test_mpm_object.py \
  source/isaaclab_tasks/test/contrib/test_franka_pour_env_cfg.py \
  -k 'partial_reset or private_solver_access' -vv
```

Expected: FAIL because `pour_env.py` is absent and no task-level public reset path exists.

- [ ] **Step 3: Port and adapt `pour_env.py`**

Port the source-material environment. Preserve per-world builder hooks, particle slices, cup proxy mapping, containment, rewards, and robot/cup reset. Replace `_mpm_entry_states()` with:

```python
world_mask_torch = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
world_mask_torch[env_ids] = True
world_mask = wp.from_torch(world_mask_torch, dtype=wp.bool)
for state in (NewtonManager.get_state_0(), NewtonManager.get_state_1()):
    NewtonManager.reset_solver_state(state=state, world_mask=world_mask)
```

The task writes reset positions/velocities first, then delegates MPM internal history and coupled entry synchronization to the public solver reset. Remove all `_solver._entries` traversal.

At the beginning of `_prepare_newton_extras()`, call `resolve_mpm_active_cell_count()` with the final `self.num_envs` and assign the result only to the `media` entry's `MPMSolverCfg.max_active_cell_count` before Newton constructs the solver.

- [ ] **Step 4: Run GREEN and commit**

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab_newton/test/assets/test_mpm_object.py \
  source/isaaclab_tasks/test/contrib/test_franka_pour_env_cfg.py \
  -k 'partial_reset or private_solver_access' -vv
git add source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pour/pour_env.py \
  source/isaaclab_tasks/changelog.d/franka-pour.minor.rst \
  source/isaaclab_tasks/test/contrib/test_franka_pour_env_cfg.py \
  source/isaaclab_newton/test/assets/test_mpm_object.py
git commit -m "Add isolated Franka Pour environment"
```

Expected: selected tests PASS.

### Task 9: Lock and validate the local Newton/Warp pair

**Files:**
- Modify: `isaaclab.sh`
- Modify: `source/isaaclab/pyproject.toml`
- Modify: `source/isaaclab_newton/pyproject.toml`
- Modify: `source/isaaclab_physx/pyproject.toml`
- Modify: `source/isaaclab_visualizers/pyproject.toml`
- Modify: `tools/wheel_builder/res/python_packages.toml`

- [ ] **Step 1: Capture final immutable revisions**

```bash
NEWTON_SHA=$(git -C /home/maximiliank/.config/superpowers/worktrees/newton-mpm-multiworld/max-implicit-mpm-multiworld-sparse-capture rev-parse HEAD)
WARP_SHA=$(git -C /home/maximiliank/.config/superpowers/worktrees/warp-main-multiworld-reference/max-warp-max rev-parse HEAD)
printf '%s\n%s\n' "$NEWTON_SHA" "$WARP_SHA"
```

Expected: two full immutable SHAs.

- [ ] **Step 2: Add a symmetric local Warp source override**

Extend `isaaclab.sh` so `WARP_SOURCE_DIR`, when set, prepends a validated Warp source tree exactly as `NEWTON_SOURCE_DIR` does. The source tree must contain the native library built in Task 4 of the Warp plan. No absolute path is committed.

- [ ] **Step 3: Update active dependency declarations together**

Replace all seven active old Newton commit pins with `NEWTON_SHA`. Replace the three active `warp-lang==1.13.0` constraints with `warp-lang==1.16.0.dev0`, matching the built branch wheel. Do not edit historical changelog text. Local tests use `NEWTON_SOURCE_DIR` and `WARP_SOURCE_DIR`; distribution installs require the corresponding immutable Git commit and built Warp wheel artifact once published.

- [ ] **Step 4: Validate import provenance**

```bash
NEWTON_SOURCE_DIR=/home/maximiliank/.config/superpowers/worktrees/newton-mpm-multiworld/max-implicit-mpm-multiworld-sparse-capture \
WARP_SOURCE_DIR=/home/maximiliank/.config/superpowers/worktrees/warp-main-multiworld-reference/max-warp-max \
./isaaclab.sh -p -c \
  'import newton, warp as wp; wp.init(); print(newton.__file__); print(wp.__version__, wp.__file__); print(wp.context.runtime.core._name)'
```

Expected: Newton and Warp resolve to the isolated worktrees, Warp reports `1.16.0.dev0`, and the native library comes from `max-warp-max`.

- [ ] **Step 5: Commit**

```bash
git add isaaclab.sh source/isaaclab/pyproject.toml source/isaaclab_newton/pyproject.toml \
  source/isaaclab_physx/pyproject.toml source/isaaclab_visualizers/pyproject.toml \
  tools/wheel_builder/res/python_packages.toml
git commit -m "Pin the multi-world sparse MPM stack"
```

### Task 10: Run the multi-world task smoke and quality gates

**Files:**
- Test: `source/isaaclab_tasks/test/contrib/test_contrib_environments_smoke.py`
- Verification only otherwise

- [ ] **Step 1: Run focused unit suites**

```bash
./isaaclab.sh -p -m pytest source/isaaclab_newton/test/physics/test_newton_manager_abstraction.py -vv
./isaaclab.sh -p -m pytest \
  source/isaaclab_tasks/test/contrib/test_franka_pour_cube_bowl_mesh.py \
  source/isaaclab_tasks/test/contrib/test_franka_pour_media_fill.py \
  source/isaaclab_tasks/test/contrib/test_franka_pour_mdp.py \
  source/isaaclab_tasks/test/contrib/test_franka_pour_env_cfg.py -vv
```

Expected: all focused tests PASS.

- [ ] **Step 2: Run headless multi-environment smoke**

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab_tasks/test/contrib/test_contrib_environments_smoke.py \
  -k 'Isaac-Pour-Franka-v0' -vv
```

Expected: the task initializes multiple worlds, captures physics, steps, resets a subset, replays, and remains finite. A runtime capability skip must name the unavailable Isaac Sim/CUDA capability.

- [ ] **Step 3: Run repository gates**

```bash
./isaaclab.sh -d
./isaaclab.sh -f
git diff --check 80d2b8b42bc793c82ffb060cef394b14f5953cb3..HEAD
git status --short
```

Expected: documentation and formatting gates pass and the worktree is clean.
