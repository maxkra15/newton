# Newton live scene authoring demo

Captured from Newton feature commit `66b2995` (`maxkra15/viewer-live-authoring`).

The demo uses the unmodified native viewer at 1360 × 900 with native CPU MuJoCo. Input is automated through the real Pyglet/ImGui event handlers: create a scene, drag two boxes and a sphere from the palette, raise the sphere, start physics, then move an existing box with its viewport handle.

`scene-authoring-demo.mp4` is an actual X11 capture of the dedicated viewer window at 30 fps, approximately 29 seconds long. The GIF is a reduced-size preview of the same recording. Playback was not sped up. This is a workflow demonstration, not a performance benchmark.

Screenshots show the scene-creation dialog, object placement, simulation, and viewport manipulation. Capture action timestamps are provided in `scene-authoring/demo-metadata.json`.

The glasses-library demo shows the cached SimReady browser, large asset preview, native GUI drag-and-drop import into the viewport, and playback. It uses the same source commit and CPU physics configuration. The cached package is used directly; the recording does not simulate an Internet download. Action timestamps and asset provenance are in `scene-authoring/glasses-demo-metadata.json`.

This branch contains demonstration media only. It is separate from the source-code pull request.

## Asset attribution

The asset-browser screenshot and glasses-library demo render `sm_accessory_eyeglasses_a01_01.usd`, published by NVIDIA in [nvidia/simready-assets](https://huggingface.co/datasets/nvidia/simready-assets/tree/653d91e182011f357cf1644013cc94ddf194d349/accessory_eyeglasses_a01), pinned at revision `653d91e182011f357cf1644013cc94ddf194d349`. The source asset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). It is shown as a rendered illustration within the Newton editor; this media branch does not redistribute the source USD package.
