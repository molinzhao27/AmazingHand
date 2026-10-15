"""FreeCAD macro: open the assembled hand with every part as its own object.

Imports AH_Right_5F_3S_Wrist_assembly.step (the B-rep parts, in sub-assemblies palm, index, middle, ring, pinky, thumb,
wrist) and adds every part of AH_Right_5F_3S_Wrist_assembly.3mf that the STEP does not have (the mesh-only parts:
palm shells and pad, finger and thumb covers and pads, fingertip cores, the hand's servos, horns, screws, ball joints,
bushings, rods and pins) as a named, coloured mesh object in its digit's group. FreeCAD's own 3MF import merges a file
into one mesh, so this is the way to get them as separate parts (to hide, section or measure one by one).

GUI: Macro > Macros... > add this folder (or copy the file into the macro folder), select load_in_freecad.py, Execute.
Takes about a minute; then save the document if you want an .FCStd.
Headless: freecadcmd cad/five_finger/assembly/load_in_freecad.py   (builds the document and prints a summary)
"""

import os
import time
import xml.etree.ElementTree as ET
import zipfile

import FreeCAD
import Import
import Mesh
import numpy as np

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:   # run some way that does not set __file__: look in the working directory
    HERE = os.getcwd()
NAME = "AH_Right_5F_3S_Wrist_assembly"
STEP, TMF = os.path.join(HERE, NAME + ".step"), os.path.join(HERE, NAME + ".3mf")
GROUPS = ("palm", "index", "middle", "ring", "pinky", "thumb", "wrist")
NS = "{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}"


def transform(attr):
    """3MF transform "m00 m01 m02 m10 ... m32" (row-vector convention: p' = p . M[:3] + M[3])."""
    if not attr:
        return np.vstack([np.eye(3), np.zeros(3)])
    return np.array(attr.split(), float).reshape(4, 3)


def read_3mf(path):
    """Colours, mesh objects and part objects (components) of a 3MF core-spec file, and the build items."""
    colors, meshes, parts, items, cur, oid = [], {}, {}, [], None, None
    with zipfile.ZipFile(path) as z, z.open("3D/3dmodel.model") as f:
        for ev, el in ET.iterparse(f, events=("start", "end")):
            tag = el.tag.replace(NS, "")
            if ev == "start":
                if tag == "object":
                    oid = el.get("id")
                    cur = dict(name=el.get("name"), pindex=el.get("pindex"), v=[], t=[], comps=[])
                continue
            if tag == "vertex":
                cur["v"].append((float(el.get("x")), float(el.get("y")), float(el.get("z"))))
            elif tag == "triangle":
                cur["t"].append((int(el.get("v1")), int(el.get("v2")), int(el.get("v3"))))
            elif tag == "component":
                cur["comps"].append((el.get("objectid"), transform(el.get("transform"))))
            elif tag == "base":
                c = el.get("displaycolor").lstrip("#")
                colors.append(tuple(int(c[i:i + 2], 16) / 255 for i in (0, 2, 4)))
            elif tag == "object":
                (parts if cur["comps"] else meshes)[oid] = cur
            elif tag == "item":
                items.append((el.get("objectid"), transform(el.get("transform"))))
            if tag in ("vertex", "triangle"):
                el.clear()
    return colors, meshes, parts, items


def part_mesh(meshes, comps, M_item):
    pts, tris, off, pindex = [], [], 0, None
    for cid, M in comps:
        m = meshes[cid]
        V = np.asarray(m["v"]) @ M[:3] + M[3]
        pts.append(V @ M_item[:3] + M_item[3])
        tris.append(np.asarray(m["t"]) + off)
        off += len(V)
        pindex = m["pindex"] if pindex is None else pindex
    V, F = np.vstack(pts), np.vstack(tris)
    mesh = Mesh.Mesh()
    mesh.addFacets(([FreeCAD.Vector(*p) for p in V.tolist()], [tuple(t) for t in F.tolist()]))
    return mesh, pindex


def main():
    t0 = time.time()
    doc = FreeCAD.newDocument("AH_Right_5F_3S_Wrist")
    Import.insert(STEP, doc.Name)
    in_step = {o.Label for o in doc.Objects}
    group = {}
    for g in GROUPS:
        found = [o for o in doc.getObjectsByLabel(g) if o.TypeId == "App::Part"]
        group[g] = found[0] if found else None
    colors, meshes, parts, items = read_3mf(TMF)
    added = 0
    for oid, M_item in items:
        p = parts.get(oid) or dict(meshes[oid], comps=[(oid, transform(None))])
        if p["name"] in in_step:
            continue
        mesh, pindex = part_mesh(meshes, p["comps"], M_item)
        obj = doc.addObject("Mesh::Feature", "Mesh")
        obj.Label, obj.Mesh = p["name"], mesh
        prefix = p["name"].split("_")[0]
        grp = group.get(prefix if prefix in GROUPS else "palm")
        if grp is not None:
            grp.addObject(obj)
        if FreeCAD.GuiUp and pindex is not None:
            try:
                obj.ViewObject.ShapeColor = colors[int(pindex)]
            except Exception:   # the colour is cosmetic
                pass
        added += 1
    doc.recompute()
    FreeCAD.Console.PrintMessage(f"{NAME}: STEP imported, {added} mesh parts added from the 3MF, "
                                 f"{time.time() - t0:.0f} s\n")
    if FreeCAD.GuiUp:
        import FreeCADGui
        FreeCADGui.SendMsgToActiveView("ViewFit")
        FreeCADGui.activeDocument().activeView().viewIsometric()
    return doc


DOC = main()
