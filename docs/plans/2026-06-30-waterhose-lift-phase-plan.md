# Waterhose Lift Phase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route the held connector vertically to socket height before the state machine approaches the refrigerator.

**Architecture:** Add an explicit `LIFT` phase between `SETTLE` and `CARRY`. The existing Warp state-machine kernel will hold phase-entry `x/y`, move to the live pre-insertion `z`, and rotate toward the socket before `CARRY` performs the horizontal approach.

**Tech Stack:** Python, Warp kernels, Newton coupled solvers, NumPy, `unittest`

---

### Task 1: Add the explicit phase to the state sequence

**Files:**
- Modify: `newton/tests/test_waterhose_demo.py`
- Modify: `newton/examples/multiphysics/example_waterhose_insert.py:70-100`
- Modify: `newton/examples/multiphysics/example_waterhose_insert.py:370-390`
- Modify: `newton/examples/multiphysics/example_waterhose_insert.py:600-625`

- [ ] **Step 1: Write the failing phase-order test**

Add this test to `TestWaterhoseGeometry`:

```python
def test_lift_phase_precedes_carry(self):
    self.assertEqual(getattr(waterhose, "LIFT", None), 7)
    self.assertEqual(waterhose.Example.Phase.LIFT + 1, waterhose.Example.Phase.CARRY)
    self.assertEqual(waterhose.DONE, 14)
```

- [ ] **Step 2: Run the focused suite and verify the test fails**

Run:

```bash
uv run --extra dev -m newton.tests -k test_waterhose_demo
```

Expected: `test_lift_phase_precedes_carry` fails because `LIFT` is absent.

- [ ] **Step 3: Insert `LIFT` and shift later phase values**

Update the module constants and nested `Example.Phase` values to:

```python
REST = 0
APPROACH = 1
ENGAGE = 2
GRASP = 3
HOLD_GRASP = 4
RETRACT = 5
SETTLE = 6
LIFT = 7
CARRY = 8
ALIGN = 9
INSERT = 10
HOLD_INSERTED = 11
RELEASE = 12
BACKOFF = 13
DONE = 14
```

Update the phase durations after `SETTLE` from one five-second `CARRY` entry to a three-second `LIFT` and two-second `CARRY`:

```python
            0.3,
            3.0,
            2.0,
            2.0,
            4.0,
```

The first `2.0` after `3.0` is `CARRY`; the second is the existing `ALIGN` duration.

- [ ] **Step 4: Run the focused suite**

Run:

```bash
uv run --extra dev -m newton.tests -k test_waterhose_demo
```

Expected: all six tests pass; the kernel still holds position during `LIFT` until Task 2 adds its target.

### Task 2: Make `LIFT` a vertical waypoint

**Files:**
- Modify: `newton/tests/test_waterhose_demo.py`
- Modify: `newton/examples/multiphysics/example_waterhose_insert.py:160-230`

- [ ] **Step 1: Write a kernel-level failing waypoint test**

Add a test that launches `_update_state_machine` on CPU with a phase-entry end-effector position of `(1, 2, 3)`, an identity connector grasp, and a socket at `(4, 5, 6)`:

```python
def test_lift_target_preserves_xy_and_reaches_preinsert_height(self):
    phase_count = waterhose.DONE + 1
    start = waterhose.wp.transform(waterhose.wp.vec3(1.0, 2.0, 3.0), waterhose.wp.quat_identity())
    identity = waterhose.wp.transform()

    phase = waterhose.wp.array([waterhose.LIFT], dtype=int, device="cpu")
    target_position = waterhose.wp.zeros(1, dtype=waterhose.wp.vec3, device="cpu")
    target_rotation = waterhose.wp.zeros(1, dtype=waterhose.wp.vec4, device="cpu")
    gripper_blend = waterhose.wp.zeros(1, dtype=float, device="cpu")

    waterhose.wp.launch(
        waterhose._update_state_machine,
        dim=1,
        inputs=[
            waterhose.wp.array([start, start], dtype=waterhose.wp.transform, device="cpu"),
            phase,
            waterhose.wp.array([1.0], dtype=float, device="cpu"),
            waterhose.wp.ones(phase_count, dtype=float, device="cpu"),
            waterhose.wp.array([start], dtype=waterhose.wp.transform, device="cpu"),
            waterhose.wp.array([start], dtype=waterhose.wp.transform, device="cpu"),
            waterhose.wp.zeros(1, dtype=waterhose.wp.vec3, device="cpu"),
            waterhose.wp.zeros(phase_count, dtype=waterhose.wp.vec3, device="cpu"),
            waterhose.wp.zeros(phase_count, dtype=waterhose.wp.vec3, device="cpu"),
            waterhose.wp.zeros(phase_count, dtype=waterhose.wp.vec3, device="cpu"),
            waterhose.wp.zeros(phase_count, dtype=waterhose.wp.vec3, device="cpu"),
            waterhose.wp.zeros(phase_count, dtype=float, device="cpu"),
            waterhose.wp.zeros(phase_count, dtype=int, device="cpu"),
            waterhose.wp.zeros(phase_count, dtype=waterhose.wp.vec3, device="cpu"),
            waterhose.wp.zeros(phase_count, dtype=waterhose.wp.vec3, device="cpu"),
            waterhose.wp.zeros(phase_count, dtype=waterhose.wp.vec3, device="cpu"),
            waterhose.wp.zeros(phase_count, dtype=waterhose.wp.vec3, device="cpu"),
            0,
            1,
            identity,
            identity,
            waterhose.wp.vec3(4.0, 5.0, 6.0),
            waterhose.wp.quat_identity(),
            waterhose.wp.vec3(),
            waterhose.CONNECTOR_TIP_LENGTH,
            waterhose.wp.quat_identity(),
            waterhose.wp.vec3(),
            0.01,
            target_position,
            target_rotation,
            gripper_blend,
        ],
        device="cpu",
    )

    expected_z = 6.0 - 0.018 - waterhose.CONNECTOR_TIP_LENGTH
    np.testing.assert_allclose(target_position.numpy()[0], [1.0, 2.0, expected_z], atol=1.0e-6)
    self.assertEqual(int(phase.numpy()[0]), waterhose.CARRY)
```

- [ ] **Step 2: Run the test and verify the held-pose failure**

Run:

```bash
uv run --extra dev -m newton.tests -k test_waterhose_demo
```

Expected: the new test reports target `z=3.0` instead of the pre-insertion height because `LIFT` has no target branch.

- [ ] **Step 3: Implement the vertical target**

Include `LIFT` in live tip-offset tracking:

```python
    if p == LIFT or p == CARRY or p == ALIGN:
        tip_offset = live_tip_offset
```

After computing `preinsert_position`, construct the lift waypoint:

```python
    lift_position = wp.vec3(start_position[0], start_position[1], preinsert_position[2])
```

Add the phase branch immediately before `CARRY`:

```python
    elif p == LIFT:
        target_pos = lift_position
        target_rot = socket_grasp_rotation
        grip = 1.0
    elif p == CARRY:
        target_pos = preinsert_position
        target_rot = socket_grasp_rotation
        grip = 1.0
```

- [ ] **Step 4: Run the focused suite**

Run:

```bash
uv run --extra dev -m newton.tests -k test_waterhose_demo
```

Expected: all seven tests pass, including exact lift `x/y/z` and transition to `CARRY`.

### Task 3: Verify the captured demo

**Files:**
- Modify only if formatting requires it: `newton/examples/multiphysics/example_waterhose_insert.py`
- Modify only if formatting requires it: `newton/tests/test_waterhose_demo.py`

- [ ] **Step 1: Run pre-commit on changed files**

```bash
uvx pre-commit run --files \
  newton/examples/multiphysics/example_waterhose_insert.py \
  newton/tests/test_waterhose_demo.py
```

Expected: all hooks pass.

- [ ] **Step 2: Run a one-frame coupled smoke test**

```bash
uv run -m newton.examples waterhose_insert --viewer null --device cuda:0 \
  --num-frames 1 --substeps 2 --vbd-iterations 4 --mujoco-iterations 20 --quiet
```

Expected: exit code 0.

- [ ] **Step 3: Run the captured transfer trajectory**

```bash
uv run -m newton.examples waterhose_insert --viewer null --device cuda:0 \
  --num-frames 2000 --quiet
```

Expected: exit code 0 without NaNs or coupled-solver errors; 2,000 frames cover `LIFT` entry and the following `CARRY` approach.

- [ ] **Step 4: Commit and push to the fork**

```bash
git add newton/examples/multiphysics/example_waterhose_insert.py \
  newton/tests/test_waterhose_demo.py
git commit -m "Route waterhose transfer above fridge"
git push maxkra15 max/pr2848-waterhose-demo
```

