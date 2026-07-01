# Waterhose Lift Phase Design

## Context

After grasping and retracting the connector, the current state machine sends the
end effector directly to the socket standoff during `CARRY`. That straight-line
interpolation crosses the refrigerator's front plane below the socket opening.
The newly restored robot collision shapes correctly resist this invalid path.

The current IsaacLab waterhose branch contains the same direct `CARRY` target,
so the standalone demo copied it faithfully. This change implements the safer
lift-then-approach behavior expected from the scripted demo.

## State Sequence

Insert an explicit `LIFT` phase between `SETTLE` and `CARRY`:

```text
REST -> APPROACH -> ENGAGE -> GRASP -> HOLD_GRASP -> RETRACT -> SETTLE
     -> LIFT -> CARRY -> ALIGN -> INSERT -> HOLD_INSERTED -> RELEASE
     -> BACKOFF -> DONE
```

`LIFT` keeps the phase-entry end-effector `x` and `y` coordinates, which leaves
the gripper on the robot side of the refrigerator. It raises `z` to the live
pre-insertion end-effector height and rotates toward the socket grasp
orientation while keeping the gripper closed.

`CARRY` then moves from that lifted pose to the pre-insertion pose. Because both
poses share the same target height, this leg approaches the socket opening
horizontally instead of cutting diagonally through the refrigerator.

## Timing and Control

Split the existing five-second transfer budget into a three-second `LIFT` and a
two-second `CARRY`. Keep the existing smoothstep interpolation, convergence
tolerances, and two-times-duration hard timeout. Include `LIFT` in the phases
that use the live connector-tip offset so the lift height accounts for the
actual grasp rather than the idealized static offset.

All later phase indices advance by one. `DONE` becomes phase 14, and diagnostic
arrays continue to use `DONE + 1`, so they grow automatically.

## Scope

- Do not change collision materials, solver settings, IK objectives, or the
  coupled proxy configuration.
- Do not add a general motion planner or collision-based replanning.
- Keep CUDA graph capture and the one-environment pure-Newton architecture.

## Verification

- Add a focused test for the lift waypoint: output `x` and `y` must equal the
  phase-entry pose, while output `z` must equal pre-insertion height.
- Verify `LIFT` occurs immediately before `CARRY` in the public phase enum.
- Run all focused waterhose tests and pre-commit hooks.
- Run the captured trajectory and inspect phase diagnostics to confirm that the
  `CARRY` entry is already at pre-insertion height.
- Hand off the Newton viewer command for visual confirmation that the robot
  lifts clear before approaching the socket.

