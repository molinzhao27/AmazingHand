#!/usr/bin/env python3
"""What sets the finger width, stock vs slim: widths of every contributor and cross-section figures.

Reads the stock parts (cad/stl, the screws/pins from the MuJoCo assets) and the slim parts written by
make_slim_fingers.py (cad/five_finger/slim_fingers/stl); all in the finger part-studio frame
(x lateral = along the pins, y along the finger, z dorsal).

Writes width_breakdown.png (cross-sections through each phalanx and the PIP knuckle, stock | slim,
with the lateral stack-up dimensioned), silhouettes.png (top and side views of the straight finger)
and width_table.md.

Usage: .venv/bin/python cad/five_finger/slim_fingers/width_analysis.py
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import shapely
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.patches import Polygon as MPoly  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_slim_fingers as msf  # noqa: E402

C = {"core": "#7f9fd4", "shell": "#f2b36b", "pad": "#55585e", "screw": "#3b3b3b", "pin": "#9aa6b2",
     "link": "#b08a6a", "gimbal": "#b48ccf", "distal core": "#86c79a"}


def stock_screws():
    out = []
    for x, y, z in ((11.8, 30, 4), (-11.6, 30, 4), (9.6, 74.442, 1.139), (-9.6, 74.442, 1.139)):
        out.append(msf.hw_mesh(msf.SCREW25, (x, y, z), (-np.sign(x), 0, 0)))
    return out


def slim_screws(design):
    out = []
    for hx, (y, z) in ((design["prox_screw_head_x"], msf.PROX_SCREW), (design["dist_screw_head_x"], msf.DIST_SCREW)):
        for s in (-1, 1):
            out.append(msf.screw_mesh((s * hx, y, z), (-s, 0, 0)))
    return out


def polys(mesh, y):
    try:
        p = msf.section_poly(mesh, y)
    except Exception:
        return []
    if p.is_empty:
        return []
    return list(p.geoms) if hasattr(p, "geoms") else [p]


def draw(ax, mesh, y, color, alpha=0.9, lw=0.8, z=1):
    """Filled section with real holes (compound path), so parts inside a hollow shell stay visible."""
    for p in polys(mesh, y):
        verts, codes = [], []
        for ring in [p.exterior] + list(p.interiors):
            r = np.asarray(ring.coords)
            verts += r.tolist()
            codes += [MPath.MOVETO] + [MPath.LINETO] * (len(r) - 2) + [MPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MPath(verts, codes), fc=color, ec="k", lw=lw, alpha=alpha, zorder=z))


def xext(meshes, y):
    xs = [np.asarray(p.exterior.coords)[:, 0] for m in meshes for p in polys(m, y)]
    return (min(x.min() for x in xs), max(x.max() for x in xs)) if xs else (np.nan, np.nan)


def silhouette(mesh, plane):
    """Projection of a mesh on the (y, x) plane (top view) or (y, z) plane (side view)."""
    idx = [1, 0] if plane == "top" else [1, 2]
    tri = mesh.vertices[mesh.faces][:, :, idx]
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    good = np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]) > 1e-9
    sil = shapely.union_all(shapely.polygons(tri[good])).buffer(0)
    return list(getattr(sil, "geoms", [sil]))


def dim(ax, x0, x1, z, text, color="k", above=True):
    ax.annotate("", (x0, z), (x1, z), arrowprops=dict(arrowstyle="<->", color=color, lw=0.9, shrinkA=0, shrinkB=0))
    ax.text((x0 + x1) / 2, z + (0.35 if above else -0.9), text, ha="center", fontsize=7.5, color=color)


def main():
    design = json.loads((HERE / "design.json").read_text())
    st = {n: msf.load_stl(n) for n in ("Proximal", "Proximal_Shell", "Distal", "Distal_Shell", "Link", "Gimbal")}
    sl = {n: trimesh.load(HERE / "stl" / f"{n}.stl") for n in ("Proximal_Slim", "Proximal_Shell_Slim", "Distal_Slim",
                                                                "Distal_Shell_Slim", "Distal_Pad_Slim")}
    pins = [msf.hw_mesh(msf.PIN16, (0, 52, 0), (-1, 0, 0)), msf.hw_mesh(msf.PIN10, (0, 49.216, 5.315), (-1, 0, 0))]
    scr0, scr1 = stock_screws(), slim_screws(design)

    stations = [("proximal phalanx, through the shell screw (y = 30)", 30.0, "prox"),
                ("PIP knuckle, through the 16 mm pin (y = 52)", 52.0, "pip"),
                ("distal phalanx (y = 66)", 66.0, "dist"),
                ("distal phalanx, through the shell screw (y = 74.4)", 74.442, "dist")]
    fig, axs = plt.subplots(len(stations), 2, figsize=(12, 5.0 * len(stations)))
    rows = []
    for (title, y, kind), (a0, a1) in zip(stations, axs):
        for ax, tag in ((a0, "stock"), (a1, "slim")):
            if kind == "prox":
                core, shell = (st["Proximal"], st["Proximal_Shell"]) if tag == "stock" else (sl["Proximal_Slim"], sl["Proximal_Shell_Slim"])
                draw(ax, core, y, C["core"])
                draw(ax, shell, y, C["shell"])
                draw(ax, st["Link"], y, C["link"])
            elif kind == "pip":
                draw(ax, st["Proximal"] if tag == "stock" else sl["Proximal_Slim"], y, C["core"])
                draw(ax, st["Distal"] if tag == "stock" else sl["Distal_Slim"], y, C["distal core"])
                draw(ax, st["Link"], y, C["link"])
                for p in pins[:1]:
                    draw(ax, p, y, C["pin"], z=3)
                for m in ((st["Proximal_Shell"], st["Distal_Shell"]) if tag == "stock" else (sl["Proximal_Shell_Slim"], sl["Distal_Shell_Slim"])):
                    draw(ax, m, y, C["shell"], alpha=0.5)
            else:
                core, shell = (st["Distal"], st["Distal_Shell"]) if tag == "stock" else (sl["Distal_Slim"], sl["Distal_Shell_Slim"])
                draw(ax, core, y, C["distal core"])
                draw(ax, shell, y, C["shell"])
                if tag == "slim":
                    draw(ax, sl["Distal_Pad_Slim"], y, C["pad"])
            for s in (scr0 if tag == "stock" else scr1):
                draw(ax, s, y, C["screw"], z=4)
            # dimensions
            if kind == "prox":
                co = st["Proximal"] if tag == "stock" else sl["Proximal_Slim"]
                sh = st["Proximal_Shell"] if tag == "stock" else sl["Proximal_Shell_Slim"]
            elif kind == "dist":
                co = st["Distal"] if tag == "stock" else sl["Distal_Slim"]
                sh = st["Distal_Shell"] if tag == "stock" else sl["Distal_Shell_Slim"]
            else:
                co, sh = (st["Proximal"] if tag == "stock" else sl["Proximal_Slim"]), None
            c0, c1 = xext([co], y)
            ztop = 16.5
            dim(ax, c0, c1, -13.5, f"PLA core {c1 - c0:.1f}", above=False)
            if sh is not None:
                s0, s1 = xext([sh], y)
                dim(ax, s0, s1, ztop, f"{tag} {s1 - s0:.1f} mm wide")
                rows.append((title, tag, s1 - s0, c1 - c0))
            else:
                dim(ax, -8.0, 8.0, ztop, "16 mm pin (ears 2.85 + 10.5 slot)")
            ax.set_xlim(-14, 14)
            ax.set_ylim(-15.5, 19)
            ax.set_aspect("equal")
            ax.grid(alpha=0.25)
            ax.set_title(f"{tag}: {title}", fontsize=9)
            ax.set_xlabel("x (lateral, along the pins) mm", fontsize=8)
            ax.set_ylabel("z (dorsal) mm", fontsize=8)
    handles = [MPoly([[0, 0]], fc=c, ec="k", label=k) for k, c in C.items()]
    fig.legend(handles=handles, loc="lower center", ncol=len(C), fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(HERE / "width_breakdown.png", dpi=110)

    # silhouettes: top view (x-y) and side view (y-z), straight finger
    fig, axs = plt.subplots(2, 1, figsize=(13, 9))
    for ax, plane in zip(axs, ("top", "side")):
        for tag, ms, col, ls in (("stock", [st["Proximal_Shell"], st["Distal_Shell"]], "tab:red", "--"),
                                 ("slim", [sl["Proximal_Shell_Slim"], sl["Distal_Shell_Slim"]], "tab:blue", "-")):
            for m in ms:
                for p in silhouette(m, plane):
                    xy = np.asarray(p.exterior.coords)
                    ax.plot(xy[:, 0], xy[:, 1], ls, color=col, lw=1.4, label=tag if m is ms[0] else None)
        for m, col in ((st["Proximal"], C["core"]), (st["Distal"], C["distal core"]), (st["Gimbal"], C["gimbal"]), (st["Link"], C["link"])):
            for p in silhouette(m, plane):
                ax.fill(*np.asarray(p.exterior.coords).T, color=col, alpha=0.35, lw=0)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        ax.set_title(("top view (x vs y): finger width" if plane == "top" else "side view (z vs y): finger depth") +
                     " - stock shells dashed red, slim shells blue, cores/gimbal/link filled", fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / "silhouettes.png", dpi=100)

    # table
    md = ["| section | stock shell width | slim shell width | reduction | core width |", "|---|---|---|---|---|"]
    by = {}
    for title, tag, w, c in rows:
        by.setdefault(title, {})[tag] = (w, c)
    for title, d in by.items():
        (w0, c0), (w1, _) = d["stock"], d["slim"]
        md.append(f"| {title} | {w0:.2f} | {w1:.2f} | -{w0 - w1:.2f} mm (-{100 * (w0 - w1) / w0:.1f} %) | {c0:.2f} |")
    (HERE / "width_table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
