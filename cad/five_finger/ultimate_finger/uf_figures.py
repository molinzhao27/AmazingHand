"""Figures of the ultimate finger CAD (make_ultimate_finger.py): layer profiles, assembly and exploded renders
(offscreen MuJoCo), cross-sections with widths, side poses over the servo travel, width profile vs stock/slim."""

import numpy as np
import trimesh
from PIL import Image, ImageDraw

import uf_layout as U

HERE = U.HERE
STL = HERE / "stl"
FIG = HERE / "figures"
SFX = U.SUFFIX
C = dict(core=[0.82, 0.83, 0.85, 1], core2=[0.68, 0.71, 0.75, 1], link=[0.95, 0.45, 0.20, 1], l2=[0.20, 0.55, 0.95, 1],
         shell=[0.97, 0.97, 0.96, 1], pad=[0.10, 0.10, 0.11, 1], pin=[0.55, 0.57, 0.62, 1], gimbal=[0.74, 0.62, 0.82, 1],
         hw=[0.6, 0.62, 0.66, 1])
PARTS = [  # (stl stem, frame, colour key)
    ("UF_Link", "link", "link"), ("UF_P1_Proximal", "P1", "core"), ("UF_P1_Shell", "P1", "shell"), ("UF_P1_Pad", "P1", "pad"),
    ("pin_2x16_MCP", "P1", "pin"), ("pin_2x12_PIP", "P1", "pin"), ("pin_2x12_A", "P1", "pin"),
    ("UF_P2_Middle", "P2", "core2"), ("UF_P2_Shell", "P2", "shell"), ("UF_P2_Pad", "P2", "pad"), ("pin_2x8_P4", "P2", "pin"),
    ("pin_2x12_DIP", "P2", "pin"), ("UF_P3_Distal", "P3", "core"), ("UF_P3_Shell", "P3", "shell"), ("UF_P3_Pad", "P3", "pad"),
    ("pin_2x8_B", "P3", "pin"), ("UF_L2_Coupler", "L2", "l2")]


def load(stem):
    return trimesh.load(STL / f"{stem}{SFX}.stl")


def stock(name, dx=0.0):
    m = trimesh.load(U.STOCK_STL / f"Amazing Hand Parts - {name}.stl")
    m.apply_translation([dx, 0, 0])
    return m


def T4(T2):
    M = np.eye(4)
    M[1:3, 1:3], M[1:3, 3] = T2[:2, :2], T2[:2, 2]
    return M


def k_of_f(f):
    psi = np.radians(np.interp(f, U.drive["f"], U.drive["psi"]))
    return int(np.argmin(np.abs(U.PSI - psi)))


def scene_items(k, explode=None, with_gimbal=True):
    items = []
    for stem, frame, col in PARTS:
        m = load(stem).copy()
        m.apply_transform(T4(U.pose(frame, k)))
        if explode is not None:
            m.apply_translation(explode.get(stem, (0, 0, 0)))
        items.append((m, C[col]))
    if with_gimbal:
        items.append((stock("Gimbal", -U.LINK_DX), C["gimbal"]))
    return items


def render(items, views, size=(640, 420), lookat=(0, 45, 0), dist=170.0):
    import mujoco
    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth, spec.visual.global_.offheight = size
    spec.visual.headlight.ambient = [0.42] * 3
    spec.visual.headlight.diffuse = [0.6] * 3
    spec.visual.headlight.specular = [0.05] * 3
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.93, 0.94, 0.96], rgb2=[0.62, 0.66, 0.72], width=256, height=1536)
    tmp = FIG / f"_tmp{SFX}"
    tmp.mkdir(exist_ok=True)
    import uuid
    tag = uuid.uuid4().hex[:8]   # unique file names: MuJoCo caches mesh assets by path
    for i, (m, rgba) in enumerate(items):
        f = tmp / f"m{i}_{tag}.stl"
        m.export(f)
        spec.add_mesh(name=f"m{i}", file=str(f), scale=[1e-3] * 3)
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"m{i}", rgba=rgba)
    model = spec.compile()
    for f in tmp.glob("*.stl"):
        f.unlink()
    tmp.rmdir()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    r = mujoco.Renderer(model, size[1], size[0])
    cam = mujoco.MjvCamera()
    cam.lookat[:] = np.array(lookat) * 1e-3
    cam.distance = dist * 1e-3
    tiles = []
    for az, el, title in views:
        cam.azimuth, cam.elevation = az, el
        r.update_scene(data, cam)
        im = Image.fromarray(r.render())
        d = ImageDraw.Draw(im)
        d.rectangle([4, 3, 10 + 6 * len(title), 17], fill=(25, 25, 25))
        d.text((7, 4), title, fill=(255, 255, 255))
        tiles.append(np.asarray(im))
    r.close()
    return tiles


def grid(tiles, ncol):
    rows = [np.concatenate(tiles[i:i + ncol], 1) for i in range(0, len(tiles), ncol)]
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    return np.concatenate(rows, 0)


def assembly(log):
    k = k_of_f(np.degrees(0) + U.np.interp(np.degrees(U.D.psi_s), U.drive["psi"], U.drive["f"]))
    tiles = render(scene_items(k), [(0, 0, "side, straight finger"), (90, -89, "back (dorsal)"),
                                    (90, 89, "palm side"), (55, -30, "3/4 back"), (125, 32, "3/4 palm"),
                                    (270, -12, "from the fingertip")], dist=135.0)
    Image.fromarray(grid(tiles, 3)).save(FIG / f"assembly{SFX}.png")
    # exploded: shells and pads off their cores, cores apart along y, links to the side
    # shells up, pads down, links to one side, pins to the other; the phalanges pulled apart along the finger
    ex = {"UF_P1_Shell": (0, 0, 20), "UF_P1_Pad": (0, 0, -20), "UF_P2_Shell": (0, 8, 20), "UF_P2_Pad": (0, 8, -20),
          "UF_P2_Middle": (0, 8, 0), "pin_2x8_P4": (16, 8, 0), "pin_2x12_DIP": (16, 8, 0), "UF_P3_Shell": (0, 16, 20),
          "UF_P3_Pad": (0, 16, -20), "UF_P3_Distal": (0, 16, 0), "pin_2x8_B": (16, 16, 0), "UF_L2_Coupler": (-18, 8, 0),
          "UF_Link": (-18, 0, -4), "pin_2x16_MCP": (16, 0, 0), "pin_2x12_PIP": (16, 0, 0), "pin_2x12_A": (16, 0, 0)}
    tiles = render(scene_items(k, ex), [(35, -38, "exploded, from the back (shells up, pads down, Link and L2 left, "
                                                  "pins right)"), (145, 32, "exploded, from the palm side")],
                   size=(960, 640), lookat=(0, 52, 0), dist=175.0)
    Image.fromarray(grid(tiles, 2)).save(FIG / f"exploded{SFX}.png")
    log(f"figures: assembly{SFX}.png, exploded{SFX}.png")


def side_poses(log):
    fs = float(np.interp(np.degrees(U.D.psi_s), U.drive["psi"], U.drive["f"]))
    tiles = []
    for f, title in ((-60.0, "f = -60 (clamp)"), (fs, f"straight f = {fs:.0f}"), (0.0, "f = 0"), (45.0, "f = +45"),
                     (90.0, "full travel f = +90")):
        k = k_of_f(f)
        s = U.SOL
        t = (f"{title}: MCP {np.degrees(s['mcp'][k]):.0f}, PIP {np.degrees(s['pip'][k]):.0f}, "
             f"DIP {np.degrees(s['dip'][k]):.0f}")
        tiles += render(scene_items(k), [(0, 0, t)], size=(560, 420), lookat=(0, 35, -20), dist=175.0)
    Image.fromarray(grid(tiles, 3)).save(FIG / f"side_poses{SFX}.png")
    log(f"figures: side_poses{SFX}.png")


def sections(W, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath
    from shapely.geometry import Polygon
    k = k_of_f(float(np.interp(np.degrees(U.D.psi_s), U.drive["psi"], U.drive["f"])))
    meshes = {}
    for stem, frame, col in PARTS:
        m = load(stem).copy()
        m.apply_transform(T4(U.pose(frame, k)))
        meshes[stem] = (m, col)
    yp, yd = U.PIP[0], U.DIP[0]
    ys = [20.0, yp - 12.0, yp, yp + 9.0, yp + 15.0, yd, yd + 9.0, U.TIP_END - 4.0]
    fig, axes = plt.subplots(2, 4, figsize=(17, 9))
    for ax, y in zip(axes.ravel(), ys):
        xs = []
        for stem, (m, col) in meshes.items():
            s = m.section(plane_origin=[0, y, 0], plane_normal=[0, 1, 0])
            if s is None:
                continue
            poly = Polygon()
            for loop in s.discrete:   # even-odd: a shell section is a ring
                if len(loop) >= 4:
                    poly = poly.symmetric_difference(Polygon(loop[:, [0, 2]]).buffer(0))
            for pg in (poly.geoms if hasattr(poly, "geoms") else [poly]):
                if pg.is_empty or not hasattr(pg, "exterior"):
                    continue
                verts, codes = [], []
                for ring in [pg.exterior, *pg.interiors]:
                    c = np.asarray(ring.coords)
                    verts += c.tolist()
                    codes += [MPath.MOVETO] + [MPath.LINETO] * (len(c) - 2) + [MPath.CLOSEPOLY]
                ax.add_patch(PathPatch(MPath(verts, codes), fc=C[col][:3], ec="k", lw=0.4,
                                       alpha=0.9 if col != "shell" else 0.75, zorder=1 if col == "shell" else 2))
            xs.append(s.vertices[:, 0])
        if xs:
            x = np.concatenate(xs)
            ax.annotate("", (x.min(), -11.5), (x.max(), -11.5), arrowprops=dict(arrowstyle="<->", lw=0.8))
            ax.text(0, -12.8, f"{x.max() - x.min():.2f} mm", ha="center", va="top", fontsize=9)
        ax.set_title(f"y = {y:.1f} mm" + (" (PIP knuckle)" if abs(y - yp) < 0.1 else " (DIP knuckle)" if abs(y - yd) < 0.1
                                            else ""), fontsize=10)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-15, 13)
        ax.set_aspect("equal")
        ax.grid(True, lw=0.3)
        ax.set_xlabel("x (mm)")
    fig.suptitle("Cross-sections of the straight finger (palm down, -z), widths including shells, pads and pins: "
                 "grey cores, orange Link, blue L2, white TPU, dark pads, steel pins", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / f"sections{SFX}.png", dpi=110)
    plt.close(fig)
    # width profile vs the stock and slim fingers
    fig, ax = plt.subplots(figsize=(12, 4.6))
    for k_, (lab, col) in (("proximal", ("proximal", "#2a78d6")), ("middle", ("middle", "#1baf7a")),
                           ("distal", ("distal", "#eda100"))):
        pr = np.array(W[k_]["profile"])
        ax.plot(pr[:, 0], pr[:, 1], color=col, lw=2, label=f"{lab} (this design)")
    for (y0, y1, w, lab) in ((8, 51.5, 23.76, "stock proximal shell 23.8"), (53, 93.4, 21.51, "stock distal shell 21.5"),
                             (8, 51.5, 19.41, "slim proximal 19.4"), (58.8, 93, 14.86, "slim distal 14.9")):
        ax.plot([y0, y1], [w, w], lw=1, ls=(0, (4, 2)), color="#888" if "stock" in lab else "#bbb")
        ax.text(y0, w + 0.2, lab, fontsize=7.5, color="#666")
    for (y0, y1, w, lab) in ((10, yp, 17.0, "limit <= 17"), (yp, yd, 16.0, "limit <= 16"), (yd, U.TIP_END + 2, 15.0, "limit <= 15")):
        ax.plot([y0, y1], [w, w], lw=1.2, color="#e05050")
        ax.text(y1, w, lab, fontsize=7.5, color="#e05050", ha="right", va="bottom")
    ax.set_xlabel("y (mm from the MCP, straight finger)")
    ax.set_ylabel("width across the pins (mm)")
    ax.set_title("Finger width along its length (cores + shells + pads + pins)")
    ax.set_ylim(4, 25)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, lw=0.3)
    fig.tight_layout()
    fig.savefig(FIG / f"width_profile{SFX}.png", dpi=110)
    plt.close(fig)
    log(f"figures: sections{SFX}.png, width_profile{SFX}.png")


def layers(P, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    bands = [(-1.5, 1.5, "C |x|<1.5"), (1.5, 1.75, "g1 1.5..1.75"), (1.75, 4.0, "M 1.75..4.0"), (4.0, 4.25, "g2 4.0..4.25"),
             (4.25, 6.0, "O 4.25..6.0")]
    fig, axes = plt.subplots(4, 5, figsize=(24, 12))
    for r, name in enumerate(("P1", "P2", "P3", "L2")):
        for c, (x0, x1, lab) in enumerate(bands):
            ax = axes[r, c]
            geoms = [g for a, b, g in P[name].slabs if a < x1 - 1e-6 and b > x0 + 1e-6 and b > 0]
            g = unary_union(geoms) if geoms else Polygon()
            for poly in (g.geoms if hasattr(g, "geoms") else [g]):
                if poly.is_empty:
                    continue
                ax.fill(*poly.exterior.xy, fc="#9ab8de", ec="k", lw=0.5)
                for rr in poly.interiors:
                    ax.fill(*rr.xy, fc="white", ec="k", lw=0.5)
            if name == "L2":
                for gg in [s[2] for s in P["link_arm"].slabs]:
                    ax.fill(*gg.exterior.xy, fc="#f0a070", ec="k", lw=0.5, alpha=0.6 if c == 0 else 0.0)
            for kk, p in U.POINTS.items():
                ax.plot(*p, "k+", ms=5)
                ax.annotate(kk, p, fontsize=6, xytext=(2, 2), textcoords="offset points")
            ax.set_aspect("equal")
            ax.set_xlim(-6, U.TIP_END + 2)
            ax.set_ylim(-12, 12)
            ax.set_title(f"{name if name != 'L2' else 'L2 (blue) and Link arm (orange, Link frame)'}: {lab}", fontsize=8)
    fig.suptitle("Layer profiles (y, z) of the cores, one column per lateral band (+x side; the -x side is carved "
                 "separately because the gimbal is 0.35 mm off-centre)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / f"layers{SFX}.png", dpi=80)
    plt.close(fig)
    log(f"figures: layers{SFX}.png")


def all_figures(P, B, W, log):
    FIG.mkdir(exist_ok=True)
    layers(P, log)
    sections(W, log)
    assembly(log)
    side_poses(log)
