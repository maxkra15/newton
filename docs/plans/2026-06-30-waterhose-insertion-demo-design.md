# Pure Newton Waterhose Insertion Demo Design

## Context

The current IsaacLab waterhose demo uses an RBY1 robot to grasp a flexible hose
connector and insert it into a refrigerator socket. The simulation couples a
MuJoCo-Warp articulated robot to a VBD cable through Newton body proxies. An
older standalone Newton prototype demonstrates related behavior, but it keeps
separate models, implements coupling manually, contains an obsolete 19-phase
insert/pull sequence, and is more than 3,000 lines long.

This change will create a focused standalone Newton example on top of pull
request 2848 at commit `0c8408d8aa249fa073ed27ea28c823580a6c089f`.
It will use the current public `SolverCoupledProxy` API and treat the current
IsaacLab scripted insertion demo as the behavioral reference.

## Goals

- Run one complete RBY1 waterhose insertion scene using only Newton and Warp.
- Reproduce the current IsaacLab demo's grasp, insertion, release, and backoff
  behavior.
- Use the latest proxy coupling implementation from pull request 2848.
- Keep steady-state control and simulation GPU-resident and CUDA-graph-safe.
- Keep the example small enough to serve as a best-practice reference.
- Provide deterministic final-state validation through the Newton example test
  interface.

## Non-goals

- IsaacLab environment, manager, training, teleoperation, or video integration.
- Batched or replicated worlds.
- The older regrasp, connector snap, and pull-test sequence.
- Custom coupling kernels, duplicate simulation models, interactive tuning
  panels, or profiling infrastructure.
- External Nucleus or network asset dependencies at runtime.

## Files and assets

The implementation will add one executable example:

`newton/examples/multiphysics/example_waterhose_insert.py`

Required offline scene assets will live under
`newton/examples/assets/waterhose/`. Only the RBY1 model, refrigerator and
socket, hose curve, connector, and referenced textures will be included. The
IsaacLab sky HDR, ground presentation asset, and unrelated second hose will be
omitted. The example will use Newton's normal viewer and a simple ground plane.

The example will be registered in the root `README.md`, accompanied by the
standard 320x320 screenshot under `docs/images/examples/`, and registered in
`newton/tests/test_examples.py` as a CUDA/USD example.

## Scene construction

A single `ModelBuilder` will assemble the full simulation scene. It will import
the fixed-base RBY1, import the refrigerator visual and collision geometry, and
create the hose rod from the authored cable centerline. The connector collision
shape will be attached to the first rod body, matching the current IsaacLab
model and avoiding a compliant connector-to-hose weld. The last cable segment
will be fixed to a collision-free kinematic anchor at its authored location.

Only the two right-gripper finger bodies will retain robot collision. Static
refrigerator shapes will be divided by purpose: the robot entry sees the outer
housing but not the socket bore, while the VBD entry sees the hose/connector
contacts with the housing and socket. The destination proxy collision pipeline
will retain finger-to-connector rigid pairs and remove finger-to-hose particle
and finger-to-refrigerator pairs.

An additional RBY1-only model will be used by Newton IK, as in the PR 2848
Franka cable reference. Its joint coordinate layout will match the robot prefix
of the simulation model, allowing a direct device-to-device copy of IK results
into the simulation control targets.

## Solvers and coupling

`SolverCoupledProxy` will own two entries:

- `mjc`: `SolverMuJoCo` for the RBY1 articulation, using Newton contacts,
  `implicitfast`, an elliptic cone, and the current robot gains and gravity
  compensation.
- `vbd`: `SolverVBD` for the hose and connector, using hard rigid contacts,
  contact history, soft rod joints, and the current IsaacLab material and AVBD
  parameters.

The two right-finger bodies will be proxies from `mjc` to `vbd`. Coupling will
use `staggered` mode, one relaxation iteration, unit effective-mass scale, and a
collision refresh every substep. The default frame rate will be 100 Hz with ten
coupled substeps and twenty VBD iterations. Essential solver knobs may remain as
command-line arguments for debugging, but there will be no alternative custom
coupling path.

## State machine and IK

The scripted behavior will contain these fourteen phases:

`REST`, `APPROACH`, `ENGAGE`, `GRASP`, `HOLD_GRASP`, `RETRACT`, `SETTLE`,
`CARRY`, `ALIGN`, `INSERT`, `HOLD_INSERTED`, `RELEASE`, `BACKOFF`, and `DONE`.

A compact Warp kernel will hold the phase, elapsed time, phase-entry snapshots,
frozen connector-tip offset, and current IK/gripper targets. Targets will use
smoothstep position interpolation and shortest-path normalized quaternion
interpolation. Transitions will respect minimum durations and end-effector
convergence, with bounded hard timeouts. Alignment and insertion will use the
live connector-tip pose, socket axis, pre-insertion standoff, and insertion
depth from the current IsaacLab implementation.

Newton analytic IK will solve right-end-effector position and rotation plus
left-gripper and torso hold objectives, followed by a joint-limit objective.
The solved arm, torso, and gripper coordinates will be written directly into
MuJoCo position targets on the device.

## Frame execution and performance

Initialization may perform CPU asset parsing and label/index resolution. The
steady-state frame will not allocate arrays, copy simulation state to NumPy, or
synchronize with the host. One CUDA graph will capture:

1. state-machine target update from live body state;
2. analytic Newton IK;
3. device-side control target update;
4. ten collision and coupled-solver substeps; and
5. kinematic state reconstruction between substeps.

Explicit collision-pair lists will avoid broad-phase work for irrelevant robot,
refrigerator, and hose pairs. All contacts and temporary buffers will be
preallocated before capture. `--no-graph-capture` will remain available only as
a standard debugging escape hatch.

## Validation

`test_post_step()` will reject non-finite body or joint state. `test_final()`
will require:

- the state machine to reach `DONE`;
- connector-tip radial error to be inside the socket tolerance;
- connector axial depth to be inside the seated interval;
- connector/socket axes to satisfy the alignment threshold; and
- the released end effector to have backed away from the socket.

The example test will run headlessly on CUDA with enough frames to finish the
script. Focused unit tests will cover pure state-machine transition/target math
where it can be isolated cheaply. Implementation will follow test-first steps:
the new example registration/test must fail before the example exists, then pass
after the smallest implementation is added.

## Error handling

Startup will fail with descriptive errors when required assets, RBY1 body or
joint labels, connector/socket shapes, or expected contact pairs cannot be
resolved. CUDA graph capture will be enabled only on CUDA devices; the example
will retain an uncaptured execution path for diagnostics. No behavior will
silently fall back to the old custom coupling implementation.
