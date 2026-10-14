# Beauty renders (MuJoCo pose -> Blender Cycles)

Studio "product shots" of the simulated hand: glossy white shells, black ribbed TPU pads, satin metal, a seamless
grey sweep, soft-box lighting, 85 mm lens with subtle depth of field. Output:
`five_finger/renders/beauty/*.jpg` (2000 x 1500).

| file | runs in | does |
|---|---|---|
| `export_pose.py` | project venv | loads a scene, sets a hand pose + wrist pose, settles it, writes `<name>.glb` (one node per visible geom, MuJoCo world, metres) and `<name>.json` (per node: material class, rgba, group, digit; bounding boxes, camera target) |
| `blender_render.py` | Blender | imports the GLB, assigns materials per class, smooth-by-angle shading, cyclorama, light rig, camera framing, Cycles (Metal GPU) + OpenImageDenoise |
| `render_beauty.py` | project venv | the shot list (`SHOTS`): runs both for every shot |

Needs Blender 4.2 or newer (developed with 5.2.2 at `/Applications/Blender.app`; other paths: `--blender` or a
`blender` on `PATH`). The first Cycles run on a machine compiles the Metal kernels (about 2.5 min, once).

## Run

From the repository root:

```sh
.venv/bin/python Demo/AHSimulation/render/render_beauty.py               # all shots, final quality
.venv/bin/python Demo/AHSimulation/render/render_beauty.py --preview     # 800 x 600, 48 spp -> beauty/preview/
.venv/bin/python Demo/AHSimulation/render/render_beauty.py --only hero,ok --sheet   # subset + contact sheet
```

`--model` picks another model under `AHSimulation/` (default `AH_Right_5F_3S_Wrist`, `mjcf/scene_shells.xml`);
`--work DIR` keeps the GLB/JSON exports (default: a deleted temp dir). One shot by hand:

```sh
.venv/bin/python Demo/AHSimulation/render/export_pose.py --pose Peace --wrist -15 5 --out /tmp/peace.glb
/Applications/Blender.app/Contents/MacOS/Blender -b -P Demo/AHSimulation/render/blender_render.py -- \
    --glb /tmp/peace.glb --out /tmp/peace.jpg --az 160 --el -6 --preview
```

`export_pose.py --list-poses` lists the named poses (`ultimate_finger/data/hand_poses.json` over `DEMO_POSES_5F`);
`--digit thumb=-12,12` overrides one digit's (m1, m2) command in degrees.

## Time per shot (Apple M5 Pro, 20-core GPU)

| | pose + export | Blender (import, set-up, render) |
|---|---|---|
| final, 2000 x 1500, 384 spp adaptive + denoise | 2-3 s | 29-33 s |
| preview, 800 x 600, 48 spp | 2-3 s | 3-4 s |

All six final shots: 198 s. About 2 GB RAM per Blender process; one shot at a time.

## Add or change a shot

Add an entry to `SHOTS` in `render_beauty.py`, then iterate with `--preview --only <name>`:

```python
"wave": dict(
    pose="OpenHand", digits={"thumb": [-12, 12]}, wrist=(-20, 10),          # (flexion > 0 palm-ward, deviation > 0 radial)
    render=dict(az=150, el=-8, frame="hand", fill=0.72, offset=[0.0, 0.05])),
```

`render` holds `blender_render.py` options (`--key-name` -> `key_name`):

- camera: `az` / `el` in MuJoCo's free-camera convention (az 180 looks at the palm, 0 at the back, 270 from the
  thumb side; el < 0 looks down), `roll`, `lens` (mm), `fstop` (0 = no depth of field), `focus_near`;
- framing: `frame` (`hand`, `fingers`, `arm`, `all`), optionally narrowed by `frame_classes` and `frame_digits`;
  `fill` = fraction of the frame the framed part spans, `offset` = where its centre sits (fractions of the frame);
- staging: `rotate` = [rx, ry, rz] deg turns the whole model about the hand centre (e.g. `[90, 45, 0]` lays the
  forearm horizontal, pronated 45 deg); `floor_gap` (m below the lowest point; 0 = resting on the sweep);
  `wall`, `backdrop`, `backdrop_edge`, `glow` for the sweep;
- look: `rig` = JSON overrides of the soft boxes (`{"key": {"az": -35, "E": 5}}`, see `RIG`), `mat` = JSON
  overrides of the Principled inputs per class (see `MATERIALS`), `exposure`, `view`;
- `save_blend` writes the set-up scene for look-dev in the Blender GUI.

Material classes come from `CLASS_RULES` in `export_pose.py` (mesh/geom/material names: `pad` -> `black_tpu`;
`shell`, `cover`, `forearm` -> `white_plastic`; servos, frames, plates -> `dark_plastic`; pins, screws, links,
bearings, cores -> `metal`; else `white_plastic` if the rgba is near white, `other`). Groups and digits come from
the body tree and the finger actuators, so a re-built model (new covers) renders without changes.

## Limitations

- The palm shell's large surfaces show low-frequency waviness and the edge around the palm pad is slightly
  jagged at full resolution: that is the shell mesh itself, the glossy material just reveals it.
- Poses are servo commands, settled in the simulation; nothing checks for interpenetration in the frame (check the
  preview).
- Lights, sweep and camera follow the camera direction, so a strongly rotated model (`rotate`) may need a `rig`
  tweak.
