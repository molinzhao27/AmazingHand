"""Studio "product shot" render of a posed hand GLB (from export_pose.py) with Cycles. Runs inside Blender:

  Blender -b -P blender_render.py -- --glb pose.glb --out shot.jpg [--preview] [camera / light options]

Scene: the GLB (MuJoCo world, metres, Z up) under one empty, Principled materials per material class (sidecar
JSON), smooth shading by angle (30 deg) and a bevel shader for rounded edge highlights; a seamless cyclorama
(floor, cove, wall) square to the camera, its albedo pooling light behind the subject; a soft-box rig (key, kick
strip, fill, two strip rims, top; optional backdrop wash) placed relative to the camera, so every shot is lit the
same way; an 85 mm camera that frames a chosen part of the model (hand, fingers of some digits, ...) with subtle
depth of field; Cycles on the GPU (Metal on Apple silicon) with OpenImageDenoise; Khronos PBR Neutral view.

Camera direction uses MuJoCo's free-camera convention (render_showcase.py): azimuth in degrees about +Z, elevation
< 0 looks down. Light directions are relative to the camera: azimuth 0 = from the camera, +90 = from camera right,
180 = from behind the subject; elevation above the horizon.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

# --------------------------------------------------------------------------------------------- look settings
MATERIALS = {
    # class: Principled BSDF inputs (colours are linear). Base colour None = from the geom rgba (see make_material).
    # white: glossy injection-moulded ABS/PC, a thin clear coat for crisp soft-box highlights, a touch of subsurface
    "white_plastic": {"Base Color": (0.80, 0.80, 0.785), "Roughness": 0.22, "Specular IOR Level": 0.5,
                      "Coat Weight": 0.5, "Coat Roughness": 0.04, "Coat IOR": 1.5,
                      "Subsurface Weight": 0.06, "Subsurface Radius": (1.0, 0.85, 0.7), "Subsurface Scale": 0.0015},
    # ribbed TPU grip pads: black, matte
    "black_tpu": {"Base Color": (0.010, 0.010, 0.011), "Roughness": 0.6, "Specular IOR Level": 0.35},
    # satin aluminium / steel (light parts), black anodised (dark parts); tint from the rgba
    "metal": {"Base Color": None, "Metallic": 1.0, "Roughness": 0.30},
    # servos, finger frames, plates: dark satin plastic
    "dark_plastic": {"Base Color": None, "Roughness": 0.42, "Specular IOR Level": 0.5},
    "other": {"Base Color": None, "Roughness": 0.45},
}
BEVEL_RADIUS = {"white_plastic": 0.0003, "metal": 0.00025, "dark_plastic": 0.0003, "black_tpu": 0.0003, "other": 0.0003}
SMOOTH_ANGLE = 30.0  # deg: edges sharper than this stay crisp

# Light rig (soft boxes). az / el: direction from the subject, deg, camera-relative (az 0 = from the camera,
# +90 = camera right, 180 = from behind); dist m; size (w, h) m; E = irradiance at the subject, W/m^2 (the power is
# derived from it, so moving a light does not change the exposure); rgb tint. "bg" is aimed at the backdrop wall
# behind the subject instead (a soft glow that fades to the frame edges). Override per shot with --rig JSON.
RIG = {
    "key": dict(az=-50, el=40, dist=1.3, size=(0.35, 0.9), E=4.0, rgb=(1.0, 0.98, 0.95)),   # tall soft box, left
    "kick": dict(az=30, el=15, dist=1.0, size=(0.1, 0.9), E=3.0, rgb=(1.0, 1.0, 1.0)),      # thin strip: streaks
    "fill": dict(az=65, el=10, dist=1.6, size=(1.4, 1.4), E=0.55, rgb=(0.95, 0.97, 1.0)),
    "rim_l": dict(az=-145, el=25, dist=1.0, size=(0.2, 1.2), E=6.0, rgb=(1.0, 1.0, 1.0)),
    "rim_r": dict(az=140, el=30, dist=1.0, size=(0.2, 1.2), E=4.5, rgb=(1.0, 1.0, 1.0)),
    "top": dict(az=0, el=80, dist=1.4, size=(1.2, 1.2), E=0.8, rgb=(1.0, 1.0, 1.0)),
    "bg": dict(E=0.0, size=(1.0, 1.0), rgb=(1.0, 1.0, 1.0)),                                  # off by default
}


# --------------------------------------------------------------------------------------------- helpers
def log(*a):
    print("[blender_render]", *a, flush=True)


def srgb_to_lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(prog="blender_render.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glb", type=Path, required=True)
    ap.add_argument("--json", type=Path, help="sidecar (default: next to the GLB)")
    ap.add_argument("--out", type=Path, required=True, help=".png or .jpg")
    ap.add_argument("--preview", action="store_true", help="fast: small size, few samples")
    ap.add_argument("--res", type=int, nargs=2, metavar=("W", "H"))
    ap.add_argument("--samples", type=int)
    # camera
    ap.add_argument("--az", type=float, default=150.0, help="camera azimuth, deg (MuJoCo convention)")
    ap.add_argument("--el", type=float, default=-10.0, help="camera elevation, deg (< 0 looks down)")
    ap.add_argument("--roll", type=float, default=0.0, help="camera roll, deg")
    ap.add_argument("--lens", type=float, default=85.0, help="focal length, mm (36 mm sensor)")
    ap.add_argument("--frame", default="hand", help="what to frame: hand | fingers | arm | all (comma list ok)")
    ap.add_argument("--frame-classes", default="", help="only these material classes count for framing")
    ap.add_argument("--frame-digits", default="", help="only these digits count for framing (index,middle,...)")
    ap.add_argument("--fill", type=float, default=0.75, help="fraction of the frame the framed part spans")
    ap.add_argument("--offset", type=float, nargs=2, default=(0.0, 0.0), metavar=("X", "Y"),
                    help="framed part's centre in the frame, fractions of width/height from the centre (+ right/up)")
    ap.add_argument("--fstop", type=float, default=11.0)
    ap.add_argument("--focus", default=None, help="group to focus on (default: --frame); or x,y,z in metres")
    ap.add_argument("--focus-near", type=float, default=0.35,
                    help="focus depth as a quantile of the focus group's depths (0 = nearest point)")
    ap.add_argument("--rotate", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("RX", "RY", "RZ"),
                    help="turn the whole model (XYZ Euler, deg) about the hand centre before framing")
    # studio
    ap.add_argument("--floor-gap", type=float, default=0.0, help="m between the model's lowest point and the floor")
    ap.add_argument("--wall", type=float, default=0.9, help="m from the hand centre to the backdrop wall")
    ap.add_argument("--backdrop", type=float, nargs=3, default=(0.42, 0.42, 0.43),
                    help="backdrop albedo behind the subject (linear)")
    ap.add_argument("--backdrop-edge", type=float, nargs=3, default=(0.12, 0.12, 0.125),
                    help="backdrop albedo far from the subject (linear)")
    ap.add_argument("--glow", type=float, default=0.4, help="radius (sigma, m) of the backdrop's bright pool")
    ap.add_argument("--light-scale", type=float, default=1.0)
    ap.add_argument("--mat", default="{}",
                    help='JSON overrides of material inputs, e.g. {"white_plastic": {"Roughness": 0.2}}')
    ap.add_argument("--rig", default="{}", help='JSON overrides of the light rig, e.g. {"key": {"az": -35, "E": 4}}')
    ap.add_argument("--exposure", type=float, default=-0.3)
    ap.add_argument("--view", default="Khronos PBR Neutral",
                    help="view transform (Khronos PBR Neutral, AgX, Filmic, Standard)")
    ap.add_argument("--look", default="AgX - Medium High Contrast", help="look, for --view AgX")
    ap.add_argument("--world", default="gradient", choices=["gradient", "studio"],
                    help="what the shiny parts reflect besides the soft boxes")
    ap.add_argument("--no-bevel", action="store_true", help="skip the bevel shader (rounded edge highlights)")
    ap.add_argument("--save-blend", type=Path, help="also save the set-up scene (look-dev in the GUI)")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)
    a.json = a.json or a.glb.with_suffix(".json")
    if a.res is None:
        a.res = (800, 600) if a.preview else (2000, 1500)
    if a.samples is None:
        a.samples = 48 if a.preview else 384
    return a


# --------------------------------------------------------------------------------------------- scene
def import_model(glb, side):
    bpy.ops.import_scene.gltf(filepath=str(glb))
    info = {n["name"]: n for n in side["nodes"]}
    by_prefix = {n["name"].split("_", 1)[0]: n for n in side["nodes"]}
    root = bpy.data.objects.new("MODEL", None)
    bpy.context.scene.collection.objects.link(root)
    objs = []
    for ob in list(bpy.data.objects):
        if ob.type != "MESH":
            if ob is not root and ob.type == "EMPTY":
                bpy.data.objects.remove(ob)
            continue
        n = info.get(ob.name) or by_prefix.get(ob.name.split("_", 1)[0])
        if n is None:
            log("warning: no sidecar entry for", ob.name)
            n = {"cls": "other", "group": "arm", "rgba": [0.5, 0.5, 0.5, 1]}
        mw = ob.matrix_world.copy()
        ob.parent = root
        ob.matrix_world = mw
        ob["cls"], ob["group"], ob["rgba"] = n["cls"], n["group"], n["rgba"]
        ob["digit"] = n.get("digit") or ""
        objs.append(ob)
    return root, objs


def shade_smooth(objs, angle_deg):
    done = set()
    for ob in objs:
        me = ob.data
        if me.name in done:
            continue
        done.add(me.name)
        if "sharp_face" in me.attributes:
            me.attributes.remove(me.attributes["sharp_face"])
        me.set_sharp_from_angle(angle=math.radians(angle_deg))
        me.update()


def make_material(cls, rgba, bevel):
    p = dict(MATERIALS.get(cls, MATERIALS["other"]))
    mat = bpy.data.materials.new(f"{cls}")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    col = p.pop("Base Color")
    if col is None:
        lin = srgb_to_lin(rgba[:3])
        lum = float(np.dot(lin, [0.2126, 0.7152, 0.0722]))
        if cls == "metal":
            # satin aluminium / steel; dark parts read as black anodising; tints (brass balls) toned down
            grey = np.full(3, lum)
            lin = np.clip(grey + 0.35 * (lin - grey), 0.0, 1.0)
            lin = np.clip(lin * (0.95 / max(lin.max(), 1e-3)) if lum > 0.25 else np.maximum(lin, 0.05), 0, 0.95)
            p["Roughness"] = 0.28 if lum > 0.25 else 0.36
        elif cls == "dark_plastic":
            v = 0.018 + 0.04 * float(np.dot(rgba[:3], [0.2126, 0.7152, 0.0722]))
            lin = np.array([v, v, v * 1.06])
        col = tuple(float(x) for x in lin)
    bsdf.inputs["Base Color"].default_value = (*col, 1.0)
    for k, v in p.items():
        bsdf.inputs[k].default_value = v
    if bevel and BEVEL_RADIUS.get(cls):
        bev = nt.nodes.new("ShaderNodeBevel")
        bev.samples = 6
        bev.inputs["Radius"].default_value = BEVEL_RADIUS[cls]
        nt.links.new(bev.outputs["Normal"], bsdf.inputs["Normal"])
        if p.get("Coat Weight"):
            nt.links.new(bev.outputs["Normal"], bsdf.inputs["Coat Normal"])
    return mat


def assign_materials(objs, bevel):
    cache = {}
    for ob in objs:
        cls, rgba = ob["cls"], tuple(round(x, 3) for x in ob["rgba"])
        key = (cls, rgba) if MATERIALS.get(cls, {}).get("Base Color", 1) is None else (cls,)
        if key not in cache:
            cache[key] = make_material(cls, np.array(rgba), bevel)
        ob.data.materials.clear()
        ob.data.materials.append(cache[key])
        ob.material_slots[0].link = "OBJECT"   # meshes are shared; the material goes on the object
        ob.material_slots[0].material = cache[key]


def world_points(objs, max_pts=400000):
    """World-space vertices of the objects (a random subset of max_pts; None = all)."""
    arrs = []
    for ob in objs:
        me = ob.data
        v = np.empty(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get("co", v)
        v = v.reshape(-1, 3).astype(np.float64)
        M = np.array(ob.matrix_world)
        arrs.append(v @ M[:3, :3].T + M[:3, 3])
    P = np.concatenate(arrs) if arrs else np.zeros((1, 3))
    if max_pts and len(P) > max_pts:
        P = P[np.random.default_rng(0).choice(len(P), max_pts, replace=False)]
    return P


def select(objs, spec, classes="", digits=""):
    groups = {"hand": {"hand", "fingers"}, "fingers": {"fingers"}, "arm": {"arm"}, "all": {"hand", "fingers", "arm"}}
    want = set().union(*(groups[s.strip()] for s in spec.split(",")))
    cl = {c.strip() for c in classes.split(",") if c.strip()}
    dg = {c.strip() for c in digits.split(",") if c.strip()}
    sel = [o for o in objs if o["group"] in want and (not cl or o["cls"] in cl) and (not dg or o.get("digit") in dg)]
    if not sel:
        raise SystemExit(f"nothing to frame for --frame {spec} classes {classes!r} digits {digits!r}")
    return sel


def cam_basis(az, el, roll):
    a, e = math.radians(az), math.radians(el)
    f = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    if roll:
        c, s = math.cos(math.radians(roll)), math.sin(math.radians(roll))
        r, u = c * r + s * u, -s * r + c * u
    return f, r, u


def fit_camera(P, f, r, u, tan_w, tan_h, fill, offset, start):
    """Camera position on the line of sight so the points span `fill` of the frame, centred at `offset`."""
    c = start - 1.0 * f
    for _ in range(40):
        q = P - c
        z = q @ f
        x, y = (q @ r) / z, (q @ u) / z
        zc = float(np.median(z))
        scale = max((x.max() - x.min()) / (2 * tan_w * fill), (y.max() - y.min()) / (2 * tan_h * fill))
        dx = (x.max() + x.min()) / 2 - offset[0] * 2 * tan_w
        dy = (y.max() + y.min()) / 2 - offset[1] * 2 * tan_h
        c = c + f * zc * (1 - scale) * 0.7 + (r * dx + u * dy) * zc
        if abs(scale - 1) < 1e-4 and abs(dx) < 1e-5 and abs(dy) < 1e-5:
            break
    return c


def look_rot(f, r, u):
    """Blender camera: looks down -Z, +Y up, +X right."""
    R = np.column_stack([r, u, -f])
    return Matrix(R.tolist()).to_quaternion()


def add_camera(a, P_frame, P_focus):
    f, r, u = cam_basis(a.az, a.el, a.roll)
    cam_data = bpy.data.cameras.new("CAM")
    cam_data.lens = a.lens
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = 36.0
    W, H = a.res
    tan_w = 18.0 / a.lens
    tan_h = tan_w * H / W
    target = P_frame.mean(0)
    c = fit_camera(P_frame, f, r, u, tan_w, tan_h, a.fill, a.offset, target)
    cam = bpy.data.objects.new("CAM", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = Vector(c.tolist())
    cam.rotation_mode = "QUATERNION"
    cam.rotation_quaternion = look_rot(f, r, u)
    depth = (P_focus - c) @ f
    fd = float(np.quantile(depth, a.focus_near))
    cam_data.dof.use_dof = a.fstop > 0
    cam_data.dof.focus_distance = fd
    cam_data.dof.aperture_fstop = max(a.fstop, 0.5)
    cam_data.clip_start, cam_data.clip_end = 0.01, 50.0
    bpy.context.scene.camera = cam
    log(f"camera {np.round(c, 3).tolist()} dist {float(np.linalg.norm(c - target)):.3f} m, focus {fd:.3f} m, "
        f"f/{a.fstop:g}, {a.lens:g} mm")
    return c, f, r, u, target


def add_cyclorama(target, f, z_floor, wall_dist, albedo, edge, glow_radius, radius=0.35, width=8.0, height=5.0,
                  front=6.0):
    """Floor + quarter-round cove + wall, wall square to the camera's horizontal view direction."""
    fh = np.array([f[0], f[1], 0.0])
    fh /= np.linalg.norm(fh)
    rh = np.cross(fh, [0, 0, 1.0])
    prof = [(-front, 0.0)]
    s_w = wall_dist
    n = 48
    for i in range(n + 1):
        t = -math.pi / 2 + (math.pi / 2) * i / n
        prof.append((s_w - radius + radius * math.cos(t), radius + radius * math.sin(t)))
    prof.append((s_w, height))
    verts, faces = [], []
    cols = 3
    for i, (s, h) in enumerate(prof):
        for j in range(cols):
            w = -width / 2 + width * j / (cols - 1)
            p = np.array([target[0], target[1], z_floor]) + fh * s + rh * w + np.array([0, 0, h])
            verts.append(tuple(p))
    for i in range(len(prof) - 1):
        for j in range(cols - 1):
            a0, a1 = i * cols + j, i * cols + j + 1
            b0, b1 = (i + 1) * cols + j, (i + 1) * cols + j + 1
            faces.append((a0, b0, b1, a1))
    me = bpy.data.meshes.new("cyc")
    me.from_pydata(verts, [], faces)
    if "sharp_face" in me.attributes:
        me.attributes.remove(me.attributes["sharp_face"])
    me.update()
    ob = bpy.data.objects.new("CYCLORAMA", me)
    bpy.context.scene.collection.objects.link(ob)
    # albedo falls off from `albedo` behind the subject (where the line of sight meets the wall) to `edge`: the
    # soft pool of light of a studio sweep, independent of the light rig
    t = wall_dist / max(float(np.dot(f, fh)), 0.2)
    centre = bpy.data.objects.new("BACKDROP_CENTRE", None)
    bpy.context.scene.collection.objects.link(centre)
    centre.location = Vector((target + f * t).tolist())
    mat = bpy.data.materials.new("backdrop")
    mat.use_nodes = True
    nt = mat.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Roughness"].default_value = 0.85
    b.inputs["Specular IOR Level"].default_value = 0.15
    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.object = centre
    ln = nt.nodes.new("ShaderNodeVectorMath")
    ln.operation = "LENGTH"
    nt.links.new(tc.outputs["Object"], ln.inputs[0])
    m1 = nt.nodes.new("ShaderNodeMath")
    m1.operation, m1.inputs[1].default_value = "DIVIDE", glow_radius
    nt.links.new(ln.outputs["Value"], m1.inputs[0])
    m2 = nt.nodes.new("ShaderNodeMath")
    m2.operation, m2.inputs[1].default_value = "POWER", 2.0
    nt.links.new(m1.outputs[0], m2.inputs[0])
    m3 = nt.nodes.new("ShaderNodeMath")
    m3.operation, m3.inputs[1].default_value = "MULTIPLY", -0.5
    nt.links.new(m2.outputs[0], m3.inputs[0])
    m4 = nt.nodes.new("ShaderNodeMath")
    m4.operation = "EXPONENT"
    nt.links.new(m3.outputs[0], m4.inputs[0])
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs["A"].default_value = (*edge, 1.0)
    mix.inputs["B"].default_value = (*albedo, 1.0)
    nt.links.new(m4.outputs[0], mix.inputs["Factor"])
    nt.links.new(mix.outputs["Result"], b.inputs["Base Color"])
    me.materials.append(mat)
    # flip so normals face the subject
    if np.dot(np.array(me.polygons[len(me.polygons) // 2].normal), fh) > 0:
        me.flip_normals()
    return ob


def _area_light(name, pos, aim, size, power, rgb, up_hint):
    ld = bpy.data.lights.new(name, "AREA")
    ld.shape = "RECTANGLE"
    ld.size, ld.size_y = size
    ld.energy = power
    ld.color = rgb
    ob = bpy.data.objects.new(name, ld)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = Vector(pos.tolist())
    # the light shines along its local -Z; strips stand upright (long side along local Y ~ world up)
    zl = pos - aim
    zl /= np.linalg.norm(zl)
    up = np.array([0, 0, 1.0]) if abs(zl[2]) < 0.95 else up_hint
    x = np.cross(up, zl)
    x /= np.linalg.norm(x)
    y = np.cross(zl, x)
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = Matrix(np.column_stack([x, y, zl]).tolist()).to_quaternion()
    ob.visible_camera = False
    return ob


def add_lights(target, f, scale, rig, wall):
    fh = np.array([f[0], f[1], 0.0])
    fh /= np.linalg.norm(fh)
    rh = np.cross(fh, [0, 0, 1.0])
    Z = np.array([0, 0, 1.0])
    for name, L in rig.items():
        if L.get("E", 0) <= 0:
            continue
        if name == "bg":
            # from low between the subject and the wall, up onto the wall behind the subject
            aim = target + fh * wall + Z * L.get("aim_up", 0.08)
            pos = target + fh * wall * L.get("frac", 0.45) - Z * L.get("below", 0.3)
        else:
            a, e = math.radians(L["az"]), math.radians(L["el"])
            d = math.cos(e) * (math.cos(a) * -fh + math.sin(a) * rh) + math.sin(e) * Z
            pos, aim = target + L["dist"] * d, target
        dist = float(np.linalg.norm(pos - aim))
        # an area light of power P gives about P / (pi d^2) W/m^2 on axis at distance d
        _area_light(name, pos, aim, L["size"], L["E"] * scale * math.pi * dist ** 2, L.get("rgb", (1, 1, 1)), fh)


def setup_world(kind, f):
    w = bpy.data.worlds.new("studio")
    bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    bg = nt.nodes["Background"]
    if kind == "studio":
        env = nt.nodes.new("ShaderNodeTexEnvironment")
        path = Path(bpy.utils.system_resource("DATAFILES")) / "studiolights" / "world" / "studio.exr"
        env.image = bpy.data.images.load(str(path))
        mp = nt.nodes.new("ShaderNodeMapping")
        tc = nt.nodes.new("ShaderNodeTexCoord")
        mp.inputs["Rotation"].default_value[2] = math.atan2(f[1], f[0])
        nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"])
        nt.links.new(mp.outputs["Vector"], env.inputs["Vector"])
        nt.links.new(env.outputs["Color"], bg.inputs["Color"])
        bg.inputs["Strength"].default_value = 0.25
    else:
        # soft grey dome, brighter overhead: clean reflections on the glossy shells
        tc = nt.nodes.new("ShaderNodeTexCoord")
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        nt.links.new(tc.outputs["Generated"], sep.inputs["Vector"])
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = -0.2, 1.0
        nt.links.new(sep.outputs["Z"], mr.inputs["Value"])
        nt.links.new(mr.outputs["Result"], ramp.inputs["Fac"])
        ramp.color_ramp.elements[0].color = (0.05, 0.05, 0.055, 1)
        ramp.color_ramp.elements[1].color = (0.6, 0.6, 0.62, 1)
        nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
        bg.inputs["Strength"].default_value = 0.35


def setup_render(a):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    cy = sc.cycles
    if not a.cpu:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for dev_type in ("METAL", "OPTIX", "CUDA", "HIP", "ONEAPI"):
            try:
                prefs.compute_device_type = dev_type
                prefs.get_devices()
                gpus = [d for d in prefs.devices if d.type == dev_type]
                if gpus:
                    for d in prefs.devices:
                        d.use = d.type == dev_type
                    cy.device = "GPU"
                    log("device", dev_type, [d.name for d in gpus])
                    break
            except TypeError:
                continue
    cy.samples = a.samples
    cy.use_adaptive_sampling = True
    cy.adaptive_threshold = 0.02 if a.preview else 0.008
    cy.use_denoising = True
    cy.denoiser = "OPENIMAGEDENOISE"
    cy.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    cy.denoising_prefilter = "ACCURATE"
    try:
        cy.denoising_use_gpu = True
    except AttributeError:
        pass
    cy.max_bounces = 10
    cy.diffuse_bounces, cy.glossy_bounces, cy.transmission_bounces = 4, 6, 4
    cy.transparent_max_bounces = 4
    cy.caustics_reflective = cy.caustics_refractive = False
    cy.blur_glossy = 0.5
    cy.sample_clamp_indirect = 8.0
    sc.render.resolution_x, sc.render.resolution_y = a.res
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = False
    vs = sc.view_settings
    vs.view_transform = a.view
    try:
        vs.look = a.look if a.view == "AgX" else "None"
    except TypeError:
        vs.look = "None"
    vs.exposure = a.exposure
    sc.display_settings.display_device = "sRGB"
    s = sc.render.image_settings
    if a.out.suffix.lower() in (".jpg", ".jpeg"):
        s.file_format, s.quality = "JPEG", 93
        s.color_mode = "RGB"
    else:
        s.file_format, s.color_mode, s.color_depth = "PNG", "RGB", "8"
    sc.render.filepath = str(a.out)


def main():
    a = parse_args()
    t0 = time.time()
    side = json.loads(a.json.read_text())
    bpy.ops.wm.read_factory_settings(use_empty=True)
    root, objs = import_model(a.glb, side)
    shade_smooth(objs, SMOOTH_ANGLE)
    for k, v in json.loads(a.mat).items():
        MATERIALS.setdefault(k, {}).update({kk: tuple(vv) if isinstance(vv, list) else vv for kk, vv in v.items()})
    assign_materials(objs, not a.no_bevel)

    # turn the model about the hand centre (e.g. forearm horizontal); the floor goes under its lowest point
    hc = np.array(side["bbox"]["hand"]["center"])
    if any(a.rotate):
        R = (Matrix.Translation(Vector(hc.tolist())) @ Euler([math.radians(x) for x in a.rotate], "XYZ")
             .to_matrix().to_4x4() @ Matrix.Translation(Vector((-hc).tolist())))
        root.matrix_world = R @ root.matrix_world
    bpy.context.view_layer.update()

    P_all = world_points(objs, max_pts=None)
    P_frame = world_points(select(objs, a.frame, a.frame_classes, a.frame_digits))
    if a.focus and "," in a.focus and all(_is_float(x) for x in a.focus.split(",")):
        P_focus = np.array([[float(x) for x in a.focus.split(",")]])
    else:
        P_focus = world_points(select(objs, a.focus, "", "") if a.focus else
                               select(objs, a.frame, a.frame_classes, a.frame_digits))
    c, f, r, u, target = add_camera(a, P_frame, P_focus)
    z_floor = float(P_all[:, 2].min()) - a.floor_gap
    add_cyclorama(target, f, z_floor, a.wall, a.backdrop, a.backdrop_edge, a.glow)
    rig = {k: dict(v) for k, v in RIG.items()}
    for k, v in json.loads(a.rig).items():
        rig.setdefault(k, {}).update(v)
    add_lights(target, f, a.light_scale, rig, a.wall)
    setup_world(a.world, f)
    setup_render(a)
    if a.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(a.save_blend))
    t1 = time.time()
    log(f"scene ready in {t1 - t0:.1f} s ({len(objs)} objects); rendering {a.res[0]}x{a.res[1]} @ {a.samples} spp")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.render.render(write_still=True)
    log(f"wrote {a.out}  render {time.time() - t1:.1f} s, total {time.time() - t0:.1f} s")


def _is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    main()
