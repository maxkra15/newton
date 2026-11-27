# Coupled scene authoring in Newton

Both clips were captured from [Newton commit `dc455d23`](https://github.com/maxkra15/newton/commit/dc455d23a33e4d62675fce1c2fe33880299919ae) on an RTX PRO 6000 Blackwell with CUDA. The recorder samples the native ViewerGL workspace and two 1/60-second simulation steps per 30-fps video frame during playback.

- In **MuJoCo Warp + MPM**, drop a dynamic sphere into a granular bed, pause, lift it with the viewport gizmo, and resume.
- In **MuJoCo Warp + VBD**, drop the sphere into a corner-pinned cloth hammock, then lift and release it again.

The Scene Editor API stages the supports and deformable starting scene. The recorder drives the native palette drag, viewport gizmo, and Play/Pause controls through ImGui. Each clip verifies that both native solver entries are active and that the paused lift preserves the Model, solver, and live particle positions. The GIFs are reduced-size previews of the MP4s.
