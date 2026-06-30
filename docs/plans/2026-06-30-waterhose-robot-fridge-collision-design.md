# Waterhose Robot–Fridge Collision Design

## Context

The standalone waterhose demo imports 32 authored RBY1 collision shapes, but
currently clears collision flags from all except the two right fingertips. The
source solver therefore cannot prevent the arm, wrist, gripper base, or other
robot links from intersecting the refrigerator. The remaining fingertip contact
uses the refrigerator's single static concave housing mesh.

## Requirements

- Preserve the one-environment pure-Newton coupled demo and CUDA graph path.
- Restore normal robot–refrigerator collision without proxying the full robot
  into VBD.
- Keep the connector socket available to the VBD hose while preventing the
  robot from fighting the socket bore.
- Keep robot self-collision disabled and avoid the refrigerator's legacy set of
  245 convex fragments.
- Avoid unrelated solver or state-machine tuning.

## Design

Keep every collision shape that the RBY1 USD authors as collidable. Apply the
demo's contact material and margin to those shapes, but do not turn visual-only
shapes into colliders. `enable_self_collisions=False` remains on USD import, so
restoring the authored flags does not introduce robot self-contact.

The MuJoCo source entry continues to own all robot bodies and use Newton
contacts. Its view sees the refrigerator housing but disables the socket and
ground shapes. Thus the robot contacts the static housing through its authored
collision geometry, while the VBD entry retains detailed hose–housing and
connector–socket contact.

Proxy coupling remains unchanged: only the two right-finger bodies are mirrored
into VBD. Restoring source-side collision on the rest of the robot does not add
coupling work or allow non-finger links to contact the hose.

Do not initially add an analytic refrigerator proxy. First run the complete
trajectory with all authored robot colliders. If the gripper still penetrates
the thin housing mesh, add one invisible MuJoCo-only solid backstop around the
below-socket cavity as a separate, evidence-driven change; do not restore the
245-fragment hull set.

## Verification

- Add a regression test that builds the robot and verifies more than the two
  fingertip bodies retain authored shape-collision flags, including the right
  wrist and gripper base.
- Verify the proxy body list still contains exactly the two right fingers.
- Run the focused waterhose tests.
- Run a coupled one-frame smoke test, followed by the complete captured demo and
  visual inspection in the Newton viewer.

