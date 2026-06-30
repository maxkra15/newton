# Waterhose Robot–Fridge Collision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the RBY1's authored collision geometry against the refrigerator while keeping hose coupling scoped to the two right fingers.

**Architecture:** The MuJoCo source view will retain every collision shape authored by the robot USD and continue to use Newton contacts against the static refrigerator housing. Robot self-collision remains disabled, the socket remains excluded from the robot view, and only the two fingertip bodies are mirrored into VBD.

**Tech Stack:** Python, Newton `ModelBuilder`, `SolverMuJoCo`, `SolverVBD`, Warp, `unittest`

---

### Task 1: Reproduce the stripped robot collision set

**Files:**
- Modify: `newton/tests/test_waterhose_demo.py`
- Test: `newton/tests/test_waterhose_demo.py`

- [ ] **Step 1: Write the failing regression test**

Add imports for `newton`, `Example`, `SolverMuJoCo`, and `SolverVBD`, then build the RBY1 through the example's scene helper:

```python
def test_robot_keeps_authored_collision_links(self):
    builder = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(builder)
    SolverVBD.register_custom_attributes(builder, dahl_defaults_enabled=False)

    example = waterhose.Example.__new__(waterhose.Example)
    _, _, robot_shapes, finger_bodies = example._add_robot(builder, load_visual_shapes=False)
    collision_bit = int(newton.ShapeFlags.COLLIDE_SHAPES)
    collision_bodies = {
        builder.shape_body[shape]
        for shape in robot_shapes
        if builder.shape_flags[shape] & collision_bit
    }
    collision_labels = {str(builder.body_label[body]).rsplit("/", 1)[-1] for body in collision_bodies}

    self.assertEqual(len(finger_bodies), 2)
    self.assertGreater(len(collision_bodies), len(finger_bodies))
    self.assertIn("right_arm_wrist_pitch", collision_labels)
    self.assertIn("right_gripper_base", collision_labels)
```

- [ ] **Step 2: Run the test and verify the current behavior fails**

Run:

```bash
uv run --extra dev -m newton.tests -k test_waterhose_demo
```

Expected: `test_robot_keeps_authored_collision_links` fails because only the two finger bodies retain `COLLIDE_SHAPES`.

### Task 2: Preserve authored robot colliders

**Files:**
- Modify: `newton/examples/multiphysics/example_waterhose_insert.py:790`
- Test: `newton/tests/test_waterhose_demo.py`

- [ ] **Step 1: Replace collision-flag clearing with material configuration**

In `Example._add_robot`, leave imported shape flags unchanged and configure only shapes already authored for collision:

```python
collision_bit = int(newton.ShapeFlags.COLLIDE_SHAPES)
finger_set = set(right_finger_bodies)
for shape in robot_shapes:
    if not builder.shape_flags[shape] & collision_bit:
        continue
    builder.shape_material_ke[shape] = CONTACT_KE
    builder.shape_material_kd[shape] = CONTACT_KD
    builder.shape_material_mu[shape] = 5.0 if builder.shape_body[shape] in finger_set else CONTACT_MU
    builder.shape_margin[shape] = 0.001
```

Do not change `enable_self_collisions=False`, `_configure_mujoco_view`, or the two-body proxy list.

- [ ] **Step 2: Run the focused tests**

Run:

```bash
uv run --extra dev -m newton.tests -k test_waterhose_demo
```

Expected: all waterhose geometry and collision tests pass.

- [ ] **Step 3: Run a coupled scene smoke test**

Run:

```bash
uv run -m newton.examples waterhose_insert --viewer null --device cuda:0 \
  --num-frames 1 --substeps 2 --vbd-iterations 4 --mujoco-iterations 20 --quiet
```

Expected: exit code 0 with no model-view ownership or collision-pipeline errors.

- [ ] **Step 4: Run the captured trajectory**

Run:

```bash
uv run -m newton.examples waterhose_insert --viewer null --device cuda:0 \
  --num-frames 3200 --quiet
```

Expected: the trajectory completes without NaNs or coupled-solver errors. Final insertion assertions are intentionally not enabled for this diagnostic run.

- [ ] **Step 5: Commit the collision change**

```bash
git add newton/examples/multiphysics/example_waterhose_insert.py \
  newton/tests/test_waterhose_demo.py
git commit -m "Improve waterhose robot collision"
```

### Task 3: Check whether a solid backstop is necessary

**Files:**
- Inspect: `newton/examples/multiphysics/example_waterhose_insert.py`

- [ ] **Step 1: Run the Newton viewer**

Run:

```bash
uv run -m newton.examples waterhose_insert --viewer gl --device cuda:0
```

Expected: the right arm, wrist, and gripper respect the refrigerator housing while the connector can still enter the socket.

- [ ] **Step 2: Keep the implementation minimal when the authored collision succeeds**

If the full robot collision geometry prevents visible housing penetration, do not add a box proxy. If repeatable penetration remains below the socket, capture the phase and location and create a separate test-first change for one MuJoCo-only analytic backstop; do not restore the 245 convex fragments.

