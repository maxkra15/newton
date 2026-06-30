# Pure Newton Waterhose Insertion Demo Implementation Plan

> **For Codex:** Execute this plan sequentially with test-driven development.
> Keep all work on `max/pr2848-waterhose-demo`; do not open a pull request.

**Goal:** Add one compact, standalone Newton example in which RBY1 grasps a
VBD waterhose connector, inserts it into a refrigerator socket, releases it,
and backs away using PR 2848 proxy coupling.

**Architecture:** Build one Newton simulation model and split it into MuJoCo
robot and VBD hose entries through `SolverCoupledProxy`. Build a second,
robot-only model for analytic IK. A single Warp state-machine kernel produces
live end-effector and gripper targets, and one CUDA graph captures target
generation, IK, control writes, collision, and coupled substeps.

**Stack:** Python 3.12, Newton, Warp, MuJoCo-Warp, USD, `unittest`, CUDA graphs.

---

## Task 1: Register the missing example as a failing test

**Files:**

- Modify: `newton/tests/test_examples.py`

1. Add a CUDA-only example registration for
   `multiphysics.example_waterhose_insert`, using the null viewer, USD-required
   marker, one world, and enough frames for the complete scripted sequence.
2. Run:

   ```bash
   uv run --extra dev -m newton.tests \
     -k test_examples.TestMultiphysicsExamples.test_multiphysics_example_waterhose_insert
   ```

3. Confirm RED because the example module does not exist. Do not weaken or skip
   the test.
4. Commit the failing test independently.

## Task 2: Add the minimal offline asset bundle

**Files:**

- Add: `newton/examples/assets/waterhose/fridge/fridge.usda`
- Add: `newton/examples/assets/waterhose/fridge/fridge_waterhose.usda`
- Add: `newton/examples/assets/waterhose/fridge/texture/*`
- Add: `newton/examples/assets/waterhose/fridge/cable/cable001.usda`
- Add: `newton/examples/assets/waterhose/fridge/cable/plug.usda`
- Add: `newton/examples/assets/waterhose/rby1df/rby1df.usda`
- Add: `newton/examples/assets/waterhose/rby1df/rby1df_waterhose.usda`

1. Copy only the files transitively referenced by the current IsaacLab
   WaterhoseDemo RBY1, fridge, cable, connector, and textures. Exclude the HDR,
   presentation ground, and unrelated assets.
2. Rewrite only relative asset references that change under the Newton asset
   directory.
3. Use USD inspection to ensure every reference resolves locally:

   ```bash
   uv run python - <<'PY'
   from pathlib import Path
   from pxr import Usd
   root = Path("newton/examples/assets/waterhose")
   for path in root.rglob("*.usd*"):
       assert Usd.Stage.Open(str(path)), path
   PY
   ```

4. Commit the asset bundle separately so later code diffs remain reviewable.

## Task 3: Add geometry tests before the example implementation

**Files:**

- Add: `newton/tests/test_waterhose_demo.py`

1. Write `unittest` cases for a small public-to-the-example geometry helper:
   connector-tip position, axial depth, radial error, and axis alignment.
2. Include an exact seated pose, a radial miss, and a reversed-axis case.
3. Import the not-yet-created helper from
   `newton.examples.multiphysics.example_waterhose_insert`.
4. Run:

   ```bash
   uv run --extra dev -m newton.tests -k test_waterhose_demo
   ```

5. Confirm RED on the missing module/helper.

## Task 4: Build the compact scene and expose geometry helpers

**Files:**

- Add: `newton/examples/multiphysics/example_waterhose_insert.py`

1. Add SPDX header, standalone example documentation, constants, the nested
   fourteen-phase enum, label resolution helpers, and the tested connector
   metric helper.
2. Import RBY1 and refrigerator USD into one `ModelBuilder`.
3. Parse the authored `BasisCurves` centerline with `pxr.UsdGeom`, call
   `ModelBuilder.add_rod`, merge the connector shape onto segment zero, and add
   a collision-free fixed tail anchor.
4. Restrict robot collisions to the two right fingers and retain only the
   required refrigerator housing/socket colliders.
5. Finalize the model and initialize both state buffers with `eval_fk`.
6. Run the geometry tests and a one-frame construction smoke command. Fix only
   scene/import failures at this checkpoint.

## Task 5: Add modern coupled solver setup

**Files:**

- Modify: `newton/examples/multiphysics/example_waterhose_insert.py`

1. Create `mjc` and `vbd` `SolverCoupled.Entry` definitions from resolved body,
   joint, and shape lists.
2. Configure `SolverMuJoCo` with Newton contacts and the current RBY1 drive and
   gravity-compensation settings.
3. Configure `SolverVBD` with the current hard-contact/history/soft-joint
   settings.
4. Add a `SolverCoupledProxy.Config` containing only the two right-finger body
   proxies in `staggered` mode.
5. Build the proxy destination collision pipeline with explicit pairs: preserve
   normal VBD pairs and keep finger pairs only against the connector.
6. Preallocate the global collision pipeline and contact buffers, then run a
   short uncaptured simulation smoke test that asserts finite state.

## Task 6: Implement GPU state machine and analytic IK

**Files:**

- Modify: `newton/examples/multiphysics/example_waterhose_insert.py`

1. Add Warp quaternion/transform helpers and one kernel that implements:

   ```text
   REST -> APPROACH -> ENGAGE -> GRASP -> HOLD_GRASP -> RETRACT ->
   SETTLE -> CARRY -> ALIGN -> INSERT -> HOLD_INSERTED -> RELEASE ->
   BACKOFF -> DONE
   ```

2. Store phase, elapsed time, entry poses, frozen tip offset, IK targets, and
   gripper target in preallocated device arrays.
3. Build a robot-only IK model and analytic LM solver with right-EE, left-hold,
   torso-hold, and joint-limit objectives.
4. Write IK and explicit three-joint right-gripper targets directly into the
   coupled model control array on device.
5. Implement `simulate()` with ten coupled substeps and no per-frame allocation
   or host reads.
6. Capture state-machine update, IK, control update, and simulation in one CUDA
   graph. Keep `--no-graph-capture` for diagnostics.
7. Run the example uncaptured first and inspect phase progression; then enable
   graph capture and verify equivalent final metrics.

## Task 7: Make the end-to-end test pass and tune minimally

**Files:**

- Modify: `newton/examples/multiphysics/example_waterhose_insert.py`
- Modify: `newton/tests/test_examples.py` only if the frame budget needs a
  justified increase

1. Implement `test_post_step()` to check finite simulation state in test mode.
2. Implement `test_final()` to assert `DONE`, seated connector depth/radius,
   axis alignment, and gripper backoff.
3. Run the registered example test. Preserve the current IsaacLab trajectory
   constants; tune only physics parameters demonstrably needed by the pure
   Newton port.
4. Compare graph-captured and uncaptured success, and run at least three
   consecutive captured executions to detect nondeterministic contact failure.
5. Record the final default substeps, VBD iterations, phase duration, and total
   runtime in code comments only where the rationale is non-obvious.

## Task 8: Register and document the finished example

**Files:**

- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Add: `docs/images/examples/example_waterhose_insert.jpg`

1. Add the standard multi-physics example tile and command:

   ```bash
   python -m newton.examples waterhose_insert
   ```

2. Add an `Added` changelog entry at a random position in the unreleased
   section.
3. Capture a representative 320x320 screenshot showing RBY1 holding or seating
   the connector. Keep image size reasonable.

## Task 9: Verification and fork handoff

1. Run focused verification:

   ```bash
   uv run --extra dev -m newton.tests -k test_waterhose_demo
   uv run --extra dev -m newton.tests \
     -k test_examples.TestMultiphysicsExamples.test_multiphysics_example_waterhose_insert
   uv run --extra dev -m newton.tests \
     -k test_coupled_solver.TestSolverCoupledBasic.test_step
   uvx pre-commit run --files \
     newton/examples/multiphysics/example_waterhose_insert.py \
     newton/tests/test_waterhose_demo.py newton/tests/test_examples.py \
     README.md CHANGELOG.md \
     docs/plans/2026-06-30-waterhose-insertion-demo-design.md \
     docs/plans/2026-06-30-waterhose-insertion-demo-plan.md
   ```

2. Review `git diff --check`, the full branch diff from
   `origin/pr-2848-head`, and asset sizes/references.
3. Commit logical checkpoints with imperative subjects.
4. Push `max/pr2848-waterhose-demo` only to `maxkra15`; do not push upstream
   and do not open a pull request.
