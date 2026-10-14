# The "ultimate" finger: thin, styled, 3 phalanges, stock drive

A finger for the 5-digit AmazingHand ([FIVE_FINGER.md](../../../FIVE_FINGER.md)) that merges three earlier studies into one buildable, simulated design:

| From | What it contributes |
|---|---|
| [three_segment](../three_segment/README.md) | 3 phalanges on the unchanged 2-servo drive: a re-dimensioned crossed four-bar (Link → middle phalanx P2) plus a DIP coupler L2. **The linkage is kept as is**: it is verified, well conditioned, and a re-optimisation would not move PIP past ~78° (section 2). |
| [slim_fingers](../slim_fingers/README.md) | TPU shells hugging the cores (0.25 mm clearance, ≥ 1.2 mm wall), shell ends chosen from a joint-range sweep (≥ 0.5 mm), exposed knuckles. The thumb reuses the slim parts. |
| [fingers](../fingers/) (styled shells) | TPU grip-pad inserts in a raised land with a 0.1 mm glue gap and a rounded top edge. Restyled after the user's reference hand: black pads with fine transverse ribs over the palm face of every phalanx, and a nail-like fingertip cap that the pad wraps (section 4). |

New here: the **lateral stacking** of the cores (it sets the width), shells as sleeves and caps lofted around everything that moves inside them, the strength analysis of the thinner parts with a torque-limit recommendation, the thumb's matching pad set, a **shorter pinky** variant, and the 5-digit hand `AH_Right_5F_3S`.

![open hand](../../../Demo/AHSimulation/ultimate_finger/renders/hand_OpenHand_shells.jpg)

![finger close-up](../../../Demo/AHSimulation/ultimate_finger/renders/finger_closeup.jpg)

*AH_Right_5F_3S on the palm shells of `cad/five_finger/stl`: thin 3-phalanx fingers with white TPU sleeves, black ribbed grip pads and exposed knuckles; nail-like fingertip caps that the pads wrap; slim thumb with the same pads; shorter pinky.*


**In short**

- **Look:** white rounded-rectangle sleeves with 3–3.6 mm corners, and black ribbed pads (1 mm pitch) over the palm face of every phalanx. The fingertip cap is a little fuller than its neck, with a domed, nail-like back that rolls over the end and the ribbed pad wrapping around it. Covers and pads only; the mechanism is unchanged.
- **Width:** proximal 15.8 mm, middle 15.0 mm, distal 13.5 mm including covers and pads (targets ≤ 17 / 15 / 13.5); inner knuckles 12.0 mm. The MCP knuckle stays 16.2 mm (stock gimbal and Proximal ears).
- **Range:** MCP 89° / PIP 78° / DIP 54° (stock 75° / 69° / –), identical to the 3-phalanx concept. A six-bar stretch study shows PIP 100° / DIP 70° is reachable inside the stock drive, at the cost of 2 parts, 3 pins and 2–3 link layers per finger (section 2): not built.
- **Strength:** the Link arm (a bending member) and the PIP pin in P1's fork govern. With Link, L2, P1, P2 in PA-CF the parts hold 7.5 N at the fingertip when straight (the servos can push 3.7 N there) and 26 N from f = +30 on. **Set `max_torque_limit` = 800 (80 %)**; it protects contact up to f = +45. Beyond that, limit the load in software (section 5).
- **Simulation:** single finger PASS (kinematics, stability, loop closures, no self-contact in the working range).
  - The 5-digit hand is stable.
  - Finger-to-finger contacts are about half those of stock fingers on the same 30 mm parallel layout.
  - The fingers do not touch the final palm plate or the palm shells.
  - Peace sign, middle finger, pad-to-pad pinch and OK sign work (section 7).
- **Not done:** printing, fit and load tests; contact/grasp simulation; the left hand; the thumb in the new style. On the powered wrist the hand is `AH_Right_5F_3S_Wrist` ([FIVE_FINGER.md](../../../FIVE_FINGER.md#styled-3-phalanx-fingers-final)).

## 1. What changed, in one table

| | Stock 2-phalanx | Slim (2-phalanx) | 3-phalanx concept | **This finger** | Target |
|---|---|---|---|---|---|
| Phalanges | 2 | 2 | 3 | **3** | 3 |
| Proximal width (shell) | 23.8 | 19.4 | 20.0 | **15.8** | ≤ 17 |
| Middle width | – | – | 10.0 (pad only, no cover) | **15.0** | ≤ 15 |
| Distal width | 21.5 | 14.9 | 18.0 | **13.5** (13.46) | ≤ 13.5 |
| Inner knuckles (PIP, DIP) | 16.2 | 16.2 | 16.2 | **12.0** | |
| MCP knuckle (stock gimbal + Proximal ears) | 16.2 | 16.2 | 16.8 | 16.2 | stock |
| MCP / PIP / DIP at full travel | 75 / 69 / – | 75 / 69 / – | 89 / 78 / 54 | **89 / 78 / 54** | (90 / 100 / 70 human) |
| Straight finger at f = (m1 − m2)/2 | −37.5° | −37.5° | −56° | **−56°** (clamp f ≥ −60°) | |
| Moving mass beyond the gimbal | 18.4 g | ~18 g | 23.5 g | **21.0 g** | |
| Pins per finger | 2×16, 2×10 | same | 5×16, 2×10 | **2×16, 2×10 (stock) + 3×12, 2×8** | |

All widths are across the pins (x), measured on the exported meshes including shells, pads and pins ([report_cad.txt](report_cad.txt)).

![sections](figures/sections.png)

![width profile](figures/width_profile.png)

## 2. Kinematics

The planar linkage of the [3-phalanx study](../three_segment/README.md#3-the-design-option-a) is used unchanged ([design.json](design.json) copies it): segments 47 / 26 / 13.85 mm (MCP→PIP→DIP→tip site), loop 1 = crossed four-bar gimbal–Link–P2–P1 with P4 = (43.20, 3.95), loop 2 = crossed four-bar P2–P1–L2–P3 with A = (45.65, −5.85), B = (65.45, 3.65), the new Link arm rotated 14.25° so the finger is straight at f = −56°. In the sim ([verify_finger.txt](../../../Demo/AHSimulation/ultimate_finger/reports/verify_finger.txt)):

| f | MCP | PIP | DIP | total curl | stock MCP / PIP |
|---|---|---|---|---|---|
| −60 (clamp) | −3.2 | −4.0 | −2.9 | −10 | −18.8 / −19.3 |
| −56 (straight) | 0 | 0 | 0 | 0 | −15 / −15 |
| 0 | 45.2 | 49.3 | 34.6 | 129 | 31.1 / 31.1 |
| +45 | 73.8 | 72.4 | 50.7 | 197 | 60.4 / 58.3 |
| +90 | **88.8** | **78.1** | **54.4** | **221** | 75.4 / 68.8 |

Sim vs planar design ≤ 0.02° MCP, 0.20° PIP, 0.37° DIP; the drive (Link angle, abduction) matches the stock finger to 0.22° / 0.01°.

**Why not re-optimise for more PIP?** With the Link as the only input and the stock gimbal pivots, a crossed four-bar tops out at PIP ≈ 78° for MCP ≈ 89° ([three_segment §3.3](../three_segment/README.md#33-why-pip-stops-at-about-78)). The **stretch study** ([sixbar/](../../../Demo/AHSimulation/ultimate_finger/sixbar/sixbar_report.txt)) checked six-bars inside the existing drive:

| Layout (planar, transmission ≥ 40° in flexion / ≥ 25° over the travel) | MCP | PIP | DIP | extra parts / pins | centre layers | PIP play (0.05 mm/pin) |
|---|---|---|---|---|---|---|
| current crossed four-bar | 88.8 | 78.1 | 54.4 | – | 1 | ±2.5° |
| (a) Watt six-bar: Link → rocker R on P1 → coupler C → P2 | 88.0 | **100.0** | 69.6 | +2 / +3 | 3 | ±5.0° |
| (b) Stephenson six-bar: Link → K → P1, Link → D → P2 | 90.0 | **100.0** | 69.7 | +2 / +3 | 2 | ±6.2° |
| (c) P2 hung on two links ("virtual P4") | 90.0 | 76.7 | 57.0 | +2 / +3 | 3 | – |

So **yes, PIP 100° / DIP 70° is reachable within the stock drive**, with a Watt or Stephenson six-bar. It is **not built here** because it works against the other goals: it needs 2–3 link layers in the centre of the finger instead of one (≈ +3–6 mm width at the knuckles), 3 more pins per finger with about twice the PIP play, and at f = +90 the fingertip curls back to y ≈ −25 mm, under its own knuckle, so the palm shell and plate would have to be re-checked for it. It is the right next step for a "fist-closing" finger that may be wider.

![side poses](figures/side_poses.png)

## 3. Thin layout

Every part is a stack of slabs across the pin axis x ([uf_layout.py](uf_layout.py), figure [layers.png](figures/layers.png)):

| Band | \|x\| (mm) | At the PIP | At the DIP | Elsewhere |
|---|---|---|---|---|
| C (centre) | 0–1.5 | Link arm eye (P4), L2 eye (A) | L2 eye (B) | Link arm in P1's channel; L2 inside P2 |
| g1 | 1.5–1.75 | 0.25 mm running gap | gap | filled only to join a part's own bands |
| M | 1.75–4.0 | **P2** (inside P1's fork) | **P3** (inside P2's fork) | P1 walls, P2 walls, P3 tip body |
| g2 | 4.0–4.25 | gap | gap | filled where a part spans M and O |
| O | 4.25–6.0 | **P1 fork** | **P2 fork** (walls 5.85, pin bosses 6.0) | P1 walls; P3 tip body only to 4.75 |

- The **nesting flips between the knuckles**: P2 is the inner part at the PIP and the outer fork at the DIP. P3 is therefore an inner part at the DIP and its fingertip body can be 9.5 mm wide (13.5 mm with the cap) without passing P2's fork.
- **Pins**: hardened Ø2 dowels. 2×12 at the PIP, A and DIP, flush with the 12.0 mm forks; 2×8 at P4 and B, flush with the 8.0 mm inner parts and held in place by the outer fork 0.25 mm away. MCP (2×16) and Link pivot (2×10) stay stock.
- **Links**: the Link arm and L2 are 3.0 mm thick (stock arm 3.8). The Link arm is 8 mm tall at its root, tapering to 4.5 mm at the neck, because its bending moment falls linearly towards P4 (section 5). The stock Link base (pivot and ball-stud bosses) is kept, so the rods, horns and calibration are unchanged.
- **Carving.** Each slab is its blank minus the swept envelope (whole servo travel, 0.25° Link steps) of every slab of every other part that overlaps it laterally (with the 0.25 mm gap), grown by 0.6 mm in-plane. Knuckle "essentials" (pin bosses, forks) are laid out first, then the fillers, so the bosses are never eaten. Slivers under 0.9 mm are removed. Clearance therefore holds by construction in the flexion plane; the sim re-checks it in 3D.
- **What limits the width.** At the knuckles: 3 links-and-walls layers × 2 + 2 running gaps: 2 × (1.5 + 0.25 + 2.25 + 0.25 + 1.75) = 12.0 mm. One more mm would need either thinner fork walls than 1.75 mm (pin bearing, section 5) or a smaller running gap than 0.25 mm (FDM tolerance). On the covers: 12.0 mm (P1) or 11.7 mm (P2) core + 2 × (0.25 clearance + 0.08 rounding margin + 1.27 TPU) = 15.4 / 15.0 mm; the proximal sleeve takes 0.2 mm more per side for its rounder corners (15.8 mm). The MCP knuckle stays 16.2 mm because the stock gimbal and the stock Proximal ears around it are kept (the brief keeps the MCP interface stock).

## 4. Styling: covers, pads, fingertip

The covers follow the user's reference hand: slender segments with rounded-rectangle sections, black ribbed TPU pads over the palm face of every phalanx, and a nail-like fingertip cap with the ribbed pad wrapping its end. Covers are white, pads black. Only covers and pads changed; cores, pins, links and kinematics did not. The one exception is the P3 core's tip body (no joint in it), which is trimmed to the new cap's cavity, as the tip already was before.

![assembly](figures/assembly.png)

- **Covers (white TPU 82A).** P1 and P2 get open sleeves, P3 a closed cap.
  - **Cavity.** As before, it hugs the core (0.25 mm) and everything that moves inside it (0.6 mm: the Link arm in P1, L2's bar in P2), as a B-spline loft through sections every 2 mm.
  - **Outside.** It is now its own section: a rounded rectangle around the walled cavity, with the largest corner radius the widths allow.
    - Proximal sleeve: 3.4–3.6 mm corners. It may grow 0.25 mm per side and stays under the 17 mm target.
    - Middle sleeve: 3.0–3.2 mm corners. It gets no sideways growth because 15.0 mm is its minimum.
    - Both may grow up to 0.6 mm in height.
  - **Walls.** Where the outside rounds beyond the cavity, the TPU is solid, so there are no voids under the skin. The wall is at least 1.27 mm nominal (measured in section 7).
- **Cover ends.** These are planes chosen from the joint sweeps, exactly like the slim fingers but in pairs per knuckle.
  - Every part that moves past a cover end stays at least 0.5 mm from the cover material: both skins and the wall between them.
  - The PIP and DIP fork bosses stay uncovered, so the knuckles are exposed.
  - The knuckle gap is minimised.
  - The middle sleeve's ends are slanted, by 0.6–0.7 mm per mm. On the back, L2's B-end rises out of P2 when the DIP flexes; on the palm side, P1's A-lug swings in when the PIP flexes.
  - The outer edge of an end is rounded where the B-rep fillet succeeds (0.25–0.4 mm). The report lists which ends got it; the others keep a sharp outer edge.
  - P1 sleeve: y = 13.87 − 0.25 z … 43.40 + 0.20 z.
  - P2 sleeve: y = 54.46 − 0.70 z … 66.38 − 0.60 z.
  - P3 cap: from y = 76.58 − 0.20 z.
- **Fingertip cap.** Up to 4.5 mm beyond the DIP it is the rounded walled core, as on the sleeves. Beyond that the tip is drawn on purpose:
  - **Width.** It grows from the 12.7 mm neck to 13.4 mm at its fullest, so it is slightly bulbous. The target is ≤ 13.5 mm; the middle sleeve stays the widest at 15.0 because of the DIP fork.
  - **Nail side.** It rises 0.35 mm, with 3.6 mm corners.
  - **Pad side.** It bulges 0.6 mm (a pulp), with 2.4 mm corners.
  - **End.** A 6 mm rounded end closes to a point 35 % of the height up from the pad side. Seen from the side it is a soft "D": a flat-ish pad side, a domed nail side that rolls over the end, and a rounded end.
  - **Nail plate.** There is no raised nail plate any more; the dome is the nail.
- **Grip pads (black TPU 82A).** There is one per phalanx, covering most of the palm face.
  - **Size.** The half width is the cover half width − 1.5 mm, so the pads wrap a little onto the rounded palm-side corners. They stop 1.8 mm from the cover ends, measured over the pad's whole height, and have rounded corners. The fingertip pad starts 2.6 mm from the cap's base, so its land clears the middle sleeve's end as the DIP flexes.
  - **Seat.** Each sits in a 0.45 mm land (0.8 mm rim) with a pocket and a 0.1 mm glue gap.
  - **Thickness and ribs.** The pads are 0.8 mm thick (0.35 mm proud). Fine transverse ribs run at a 1.0 mm pitch, with grooves 0.45 mm wide and 0.3 mm deep (0.55 mm ribs).
  - **Fingertip pad.** It is bounded in the side view instead of the top view. It covers the palm face up to 0.6 mm above its lowest line, then rises over the second half of the rounded end and runs level 3.2 mm above the end point, so it wraps the lower half of the end. Its ribs follow the end as radial grooves.
  - **Edges.** The fingertip pad's edges cross the cap surface steeply, so there are no slivers. It reaches the sides only where the end has already narrowed by more than its thickness.
  - **Width.** The pads never widen the finger (checked per pad).
- **Retention.**
  - The sleeves have no screws: an M2.5 head cannot be recessed in a 12 mm core without hitting the links.
  - The P1 sleeve slides on from the PIP end before P2 is pinned, and is trapped between the MCP ears and P2.
  - The P2 sleeve slides on from the PIP end; the cap slides over the fingertip.
  - All three are glued with flexible CA (e.g. Loctite 4850, or 406 + 770 primer for TPU). The pads are glued into their pockets the same way.
- **Thumb (2 phalanges).** It keeps the slim-finger cores and shells ([slim_fingers](../slim_fingers/README.md)), with black ribbed pads of the same kind on the proximal and distal shells and no nail plate. Its shells keep the slim shape and were not restyled (see risks). Its kinematics and screws (M2.5 × 5) are stock.

![exploded](figures/exploded.png)

## 5. Strength, materials and the servo torque limit

Source: section 6 of [verify_finger.txt](../../../Demo/AHSimulation/ultimate_finger/reports/verify_finger.txt), figure [strength.png](figures/strength.png). Method: the finger is held at a flexion command f with frictionless pins and the fingertip pad is pushed with 5 N along the pad normal; every joint force is read from MuJoCo (body interaction forces incl. the loop-closure forces), then turned into stresses with the actual thin geometry. Everything scales linearly with the pad force (checked: 0 / 5 / 10 N give 0 / 68 / 137 N at P4).

**Loads per 5 N at the fingertip** (the cranks are short, so the pins carry 8–15× the pad force; 1.4–1.9× the stock finger):

| f | motor torque | P4 | PIP | A | DIP | B | MCP¹ | Link-arm moment at its root |
|---|---|---|---|---|---|---|---|---|
| −56 (straight) | 0.31 N·m | 68 N | 77 N | 11 N | 14 N | 11 N | ~68 N | 690 N·mm |
| 0 | 0.16 | 41 | 54 | 8 | 13 | 8 | ~46 | 300 |
| +30 | 0.065 | 42 | 53 | 9 | 14 | 9 | ~46 | 146 |
| +60 | 0.015 | 44 | 54 | 9 | 14 | 9 | ~47 | 49 |

¹ The MCP hinge is two site-pair closures in the model; the pin load is their sum (≈ twice the first pair, which the 3-phalanx study reported).

**New finding: the Link arm is a bending member.** It carries the whole Link torque (≈ 0.9 N·m per 5 N in the straight pose) as a force at P4 almost perpendicular to the arm, so its root sees ~690 N·mm per 5 N. That is why the 3.0 mm arm is 7.7 mm tall at the root, tapering to 4.5 mm at the neck: 26 MPa per 5 N in the straight pose, falling to 7 MPa at f = +30 because the Link torque per newton drops as the finger curls.

**Stresses per 5 N** (bearing = F/(d·t·n); pins as beams between the wall centres; hardened Ø2 dowels):

| Element | straight | f = 0 | f ≥ +30 | Limit used (PLA / PETG / PA-CF / steel) |
|---|---|---|---|---|
| Link arm, root to mid (bending + axial) | 26.6 | 12.3 | 7.1–3.1 | 25 / 22 / 40 |
| PIP pin in the P1 fork (1.75 mm walls), bearing | 11.1 | 7.7 | 7.7 | 25 / 20 / 40 |
| P4 pin in the 3.0 mm Link eye, bearing | 11.3 | 6.8 | 7.0–7.4 | 25 / 20 / 40 |
| PIP pin in P2 (2.25 mm walls), bearing | 8.6 | 6.0 | 6.0 | 25 / 20 / 40 |
| PIP pin bending / P4 pin bending | 111 / 92 | 77 / 56 | 77 / 60 | 600 (hardened ISO 8734) |
| DIP, A, B joints (any part) | ≤ 2.1 | ≤ 1.5 | ≤ 2.2 | |
| L2 axial | 1.0 | 0.8 | 0.8 | buckling 564 N (PLA) ≫ 11 N |
| P1 / P2 / P3 bending at their thinnest sections | 0.1 / 1.1 / 0.4 | | | |

The limits are typical FDM datasheet strengths in the layer plane, halved (static, dry): PLA 50 MPa, PETG 45, PA-CF (PA6/PA12-CF) 80. Pin-hole bearing is in the layer plane when the parts print with the pin axes vertical (all cores print on a side face).

**Allowable fingertip force** = the lowest element limit. With the recommended materials it is **7.5 N in the straight pose** (Link arm), 16 N at f = 0 and **26 N from f = +30 on** (PIP pin in the P1 fork). Compare with what the servos can hold at stall: 3.7 N straight, 7.1 N at f = 0, 17 N at +30, 30 N at +45, 68 N at +60, 220 N at +75. So the parts are stronger than the servos up to about f = +45; beyond that the linkage's leverage grows (the horns approach toggle) and the servos could break the PIP fork.

**Servo torque limit** (SCS0009 EEPROM register `max_torque_limit`, address 16, 0–1000 = 0–100 % of stall; rustypot `write_raw_max_torque_limit(id, value)`). A fixed limit that keeps every part safe for contact anywhere from straight up to f_max, and the fingertip force it then still allows:

| Materials | f_max = +30 | +45 | +60 | +75 |
|---|---|---|---|---|
| all PLA | 94 % (3.4 / 6.8 / 16 / 16 N) | 51 % (1.9 / 3.7 / 8.8 / 16 N) | 22 % | 6 % |
| all PETG | 75 % | 41 % | 17 % | 5 % |
| **Link, L2, P1, P2 in PA-CF; P3 PLA** | 100 % | **81 % (3.0 / 5.9 / 14 / 26 N)** | 35 % (1.3 / 2.5 / 6.0 / 11 N) | 10 % |

(forces in brackets: straight / f = 0 / +30 / +45)

**Recommendation.**
- **Materials:** Link, L2, P1 and P2 in PA-CF (PA6-CF or PA12-CF, dried, hardened nozzle); PETG if PA-CF is not available (then use the PETG row). P3 in PLA or PETG. Covers white TPU 82A (Filaflex 82A as stock), pads black TPU 82A. Pins: hardened Ø2 m6 dowels (ISO 8734 / DIN 6325), 3 × 2×12 and 2 × 2×8 per finger; ISO 2338 unhardened pins are marginal in bending at 26 N.
- **Torque limit: `max_torque_limit` = 800 (80 %)** on the eight servos of the four 3-phalanx fingers (right hand IDs 1–6 and 9–10; the thumb, 7/8, is unchanged). It protects every part for contact up to f = +45 and costs little force in the open half (3.0 instead of 3.7 N straight).
- **Beyond f = +45 the limit alone is not enough**, as on the stock finger (the leverage near full travel is huge). For grasps that close further, limit in software: stop advancing the flexion command when `present_load` reaches the pose-dependent value (35 % at +60, 10 % at +75 of stall), or cap the command at contact + 5°. Free motion to f = +90 is fine (no load).
- **Hyperextension clamp:** f ≥ −60° (m1 − m2 ≥ −120°): P1 meets the finger frame from f ≈ −65°, as the stock Proximal does from f ≈ −42°.

## 6. Single-finger model and verification

[build_models.py](../../../Demo/AHSimulation/ultimate_finger/build_models.py) cuts the index finger out of `AH_Right_5F` (palm plate, its two SCS0009, finger frames, both motor chains, the 5 stock loop closures) and converts it, like the 3-phalanx prototype: the Link keeps its body and gets the new mesh; new bodies `p2` (on the Link at P4), `p1` (on P2 at the PIP), `l2` (on P1 at A), `p3` (on P2 at the DIP); one more closure L2 ↔ P3 at B; cores, shells, pads and **pins are all mesh geoms** of the bodies that carry them, inertias from the meshes. PIP and DIP have stops just outside their kinematic range (section 10, loop-2 branch point). [verify_finger.py](../../../Demo/AHSimulation/ultimate_finger/verify_finger.py) reuses the checks of the 3-phalanx study unchanged; report [verify_finger.txt](../../../Demo/AHSimulation/ultimate_finger/reports/verify_finger.txt). **Verdict: PASS.**

| Check | Ultimate finger | Stock finger |
|---|---|---|
| Loads, unique names, 2 actuators ±90°, 6 loop closures | yes | yes (5) |
| Sim vs planar design (settled flexion sweep) | ≤ 0.02° MCP, 0.20° PIP, 0.38° DIP | |
| Drive unchanged: Link angle, abduction vs stock | ≤ 0.22°, ≤ 0.01° | |
| Stability: each actuator alone, range corners, 60 abrupt random full-range steps | no NaN, no warnings | same |
| Loop-closure error settled / ramped / abrupt | 0.006 / 0.031 / 0.52 mm | 0.002 / 0.014 / 0.56 mm |
| Motor tracking, settled, 25 × 25 grid | ≤ 0.24° (nothing binds) | ≤ 0.19° |
| Joint ranges over the whole (m1, m2) square | MCP −14.6…88.8, PIP −20.5…78.1, DIP −16.8…54.4, abduction ±38.5 | MCP −30.3…75.4, PIP −32.3…68.8 |
| Loop 2 stays on its branch, 60 abrupt steps | yes: DIP −33…56° transient (soft stops at −22 / +62°; the other branch lies below −47°) | – |
| Self-collision, 13 × 13 grid + flexion line, exact mesh booleans, 579 part pairs | **0 new contacts in the working range** | 3 stock pairs overlap in its working range |
| Moving mass beyond the gimbal | 21.0 g (covers and pads 1.9 g more than before the restyle) | 18.4 g |

Self-collision details: all new parts (cores, shells, pads, pins) were checked against every other part of the finger; pins are excused only against the two parts they join, so the A pin is checked in P2's slot. The remaining overlaps are the stock rod-end mesh artefact (in every pose, also in the stock model) and the hyperextension stop of P1 against the finger frame and a bushing below MCP −3° (f < −60°, beyond the clamp; the stock Proximal does the same from f ≈ −42°). Closest approaches in the working range: 0.25 mm between parts in neighbouring layers (the designed running gap); covers and pads ≥ 0.51 mm from anything moving past them over the whole travel (CAD sweep: P1 sleeve to the Link 0.51, P2 sleeve to P1 0.54 and to P3 0.56, cap to P2 0.51; pinky 0.50 / 0.54 / 0.62 / 0.51); covers 0.17–0.19 mm from their own cores.

![finger](../../../Demo/AHSimulation/ultimate_finger/renders/finger_compare_side.jpg)

## 7. The 5-digit hand `AH_Right_5F_3S`

[build_models.py](../../../Demo/AHSimulation/ultimate_finger/build_models.py) generates [AH_Right_5F_3S/mjcf](../../../Demo/AHSimulation/AHSimulation/AH_Right_5F_3S/mjcf/) from `AH_Right_5F_Slim` (= `AH_Right_5F` with the slim finger parts, on the current **parallel 30 mm finger layout** of [layout.py](../layout.py)). Nothing is hand-edited; rerun it whenever `AH_Right_5F`/`_Slim` are regenerated.

- **Index, middle, ring**: the ultimate finger; **pinky**: the same finger with its phalanges scaled ×0.88 (`--pinky-scale`, meshes `ufpk_*`, CAD `*_S088`); **thumb**: the slim-finger thumb with black ribbed pads. White covers (`uf_shell`), black pads (`uf_pad`, rgba 0.07 0.07 0.08).
- Same layout as `AH_Right_5F/mjcf`: `robot.xml` (root body and its `Part …` comments untouched, actuator names and order `finger1_motor1 … finger5_motor2` kept, the stock joint names `passive4 (N)`/`passive_5 (N)` reused for P2/P1, plus `coupler_a (N)` and `dip (N)`), `keyframes.xml` regenerated by settling at ctrl = 0, `scene.xml` with the finger targets moved onto the tips at the keyframe, `scene_shells.xml` with the palm/back shells and palm pad of `cad/five_finger/stl` (white shells, black pad; `--shells DIR` to use others), and a **self-contained `assets/`** (every mesh copied, all finger geoms are meshes, so `make_5f_shells.py --model AH_Right_5F_3S` and `build_wrist_model.py --hand AH_Right_5F_3S` can use it as is).
- 55 bodies, 48 joints, nq 93, 10 actuators, 29 loop closures.

### Verification ([verify_hand.txt](../../../Demo/AHSimulation/ultimate_finger/reports/verify_hand.txt))

[verify_hand.py](../../../Demo/AHSimulation/ultimate_finger/verify_hand.py) imports `five_finger/verify_5f_model.py` unchanged.

- **Structure**: names unique, every ctrlrange ±90°, keyframe loop error 0.0001 mm, mocap targets on the tips.
- **Every ultimate finger of the hand moves like the standalone model**: tip within 0.075 mm at 6 commands.
- **Shorter pinky**: the linkage closes over the whole travel (planar check and sim, loop error ≤ 0.0015 mm settled). At f = +90: MCP 87.8, PIP 79.7, DIP 54.7° (ring 88.8 / 78.1 / 54.4); straight at the same f = −56°. The fingertip is 10.3 mm shorter (83.2 vs 93.5 mm from its finger frame).
- **Stability**: each actuator alone, all range corners, the demo poses twice, 60 abrupt random full-range steps: no NaN, no warnings; loop error settled ≤ 0.006 mm, transient ≤ 0.93 mm (the thumb's stock closure, as in the stock model); motors settle within 0.25°.
- **Neighbouring fingers**, 9 × 9 grid per digit, all 6561 pose pairs, touching = sampled distance < 0.225 mm. The finger layout changed under this work (fanned → parallel at 30 mm), so the stock-finger baseline was re-measured on the same layout ([baseline_AH_Right_5F.txt](../../../Demo/AHSimulation/ultimate_finger/reports/baseline_AH_Right_5F.txt)):

| Pair | Stock fingers, old fanned layout | Stock fingers, parallel 30 mm layout | **Ultimate fingers, parallel 30 mm** |
|---|---|---|---|
| ring / pinky | 887 | 1751 | **952** |
| middle / ring | 887 | 1706 | **869** |
| index / middle | 973 | 1656 | **808** |
| thumb / ring, thumb / pinky | > 5 mm | > 5 mm | > 5 mm |

  The thin fingers halve the contacts on the tighter layout. The restyle (rounder, 0.4 mm wider proximal sleeves, wider pads) adds about 10 % over the unstyled thin finger (880 / 784 / 734). Ring/pinky contact needs the two fingers abducted towards each other (2/456 pairs at c = −22.5°, 11/489 at c = 0, 73/375 at c = +45°; c = a_pinky − a_ring); it is mostly proximal sleeve against proximal sleeve, and the ring's Link ball-stud screw against the pinky's sleeve.
- **Digits vs the final palm plate** ([cb16aff](../layout.py)). The moving parts of every digit were checked against the plate and the wrist interface on the 9 × 9 grid (section 5b of the report):
  - No contact in any of the 81 poses, for every digit.
  - Closest approach 0.40 mm: the gimbal screw's washer at full hyperextension, the same feature as on the stock thumb.
  - On the same plate, stock fingers touch it in 7–9 of 81 poses and slim fingers in 3–4 (`five_finger/reports/verify_AH_Right_5F[_Slim].txt`).
- **Palm and back shells** (`cad/five_finger/stl`: Top_Shell, Palm_Shell, Palm_Pad_Palm). The coordinator fitted them to the merged sweeps of `AH_Right_5F` and this hand ([verify_hand_shells.txt](../../../Demo/AHSimulation/ultimate_finger/reports/verify_hand_shells.txt), 13 × 13 grid per digit):
  - No moving part touches them.
  - Closest: pinky fingertip pad 1.50 mm, other fingers ≥ 2.65 mm, thumb 2.39 mm, palm pad ≥ 3.65 mm.
  - The plate and wrist interface sit in the shell's lap joint by design.

### Poses

The demo poses of `verify_5f_model.py` are re-mapped for the new fingers:
- Flexion is rescaled from [stock straight −37.5°, 90°] to [−56°, 90°].
- Hyperextension is kept as an offset, clamped at f ≥ −60°.
- Abduction is kept, within |a| + |f| ≤ 90.

The thumb commands of CloseHand, Perfect and Pinched were then searched against the 3-phalanx index ([thumb_refine.py](../../../Demo/AHSimulation/ultimate_finger/thumb_refine.py), [thumb_poses.txt](../../../Demo/AHSimulation/ultimate_finger/reports/thumb_poses.txt)):
- A coarse 7.5° grid, then a 1.5° grid.
- Every candidate is checked with exact mesh booleans, so "touch" means touching with no interpenetration.

Commands are (m1, m2) in degrees for the right hand ([hand_poses.json](../../../Demo/AHSimulation/ultimate_finger/data/hand_poses.json)); f = (m1 − m2)/2, a = (m1 + m2)/2.

| Pose | index | middle | ring | pinky | thumb | every digit pair (exact booleans) |
|---|---|---|---|---|---|---|
| OpenHand | (−53.1, 53.1) | (−53.1, 53.1) | (−53.1, 53.1) | (−53.1, 53.1) | (−35, 35) | ≥ 5.0 mm |
| CloseHand | (90, −90) | (90, −90) | (90, −90) | (90, −90) | **(24, −43.5)** | thumb rests on the index's middle sleeve, 0.09 mm, no overlap; others ≥ 5 mm |
| SpreadHand | (−30, 90) | (−49.7, 49.7) | (−85, 35) | (−90, 30) | (−90, −4) | ≥ 5.8 mm |
| ClenchHand | (−52.4, 42.4) | (−48.1, 58.1) | (−38.1, 68.1) | (−45.1, 61.1) | (−4, 90) | ≥ 4.0 mm |
| Index_Pointing | (−58.5, 58.5) | (90, −90) | (90, −90) | (90, −90) | (90, −90) | ≥ 6 mm |
| Perfect ("OK") | **(30, −30)** | (−13, 13) | (−35.9, 35.9) | (−35.9, 35.9) | **(22.5, −52.5)** | fingertips touch, 0.16 mm, no overlap (pads 0.87 mm apart); others ≥ 2.3 mm |
| Victory | (−33.5, 83.5) | (−83.5, 33.5) | (90, −90) | (90, −90) | (90, −90) | ≥ 6.3 mm |
| **Pinched** | **(20, 0)** | (90, −90) | (90, −90) | (90, −90) | **(84, 30)** | **pad on pad**: index and thumb fingertip pads 0.21 mm apart, no overlap; others ≥ 5.4 mm |
| **Peace** | (−26, 86) | (−86, 26) | (90, −90) | (90, −90) | (90, −90) | ≥ 6.1 mm |
| **MiddleFinger** (= Fuck) | (90, −90) | (−56, 56) | (90, −90) | (90, −90) | (22.5, −45) | thumb 0.36 mm above the index's middle sleeve (no touch); others ≥ 7.1 mm |

- **Peace sign.** Index and middle are straight and spread into a V: a = ±30°, as in `five_finger/hand_poses.py` (the abduction room at f = −56° is |a| ≤ 34°). Ring and pinky are fully curled.
  - The thumb **cannot fold over the ring and pinky.** Searching its whole 2-motor grid never brings its pad within 19.6 mm of the ring's middle phalanx. The thumb module sits on the index side, and its two joints swing it towards the index, not across the palm.
  - The pose therefore uses the tucked thumb (90, −90), folded into the palm on the index side.
  - Every pair is ≥ 6.1 mm apart, and the tips are 58 mm apart across the V.
- **Middle finger.** The middle finger is straight at f = −56° with no abduction; the other three are fully curled.
  - The thumb folds across the curled index and rests 0.36 mm above its middle sleeve.
  - Nothing touches.
- **Pinch.** With the bigger, wrapping fingertip pads, the pads now meet face to face: index f = +10, a = +10, thumb (84, 30).
  - This is the "pinch fix": the re-mapped stock pinch (index curled, thumb (0, −75)) does not reach the index tip on this layout.
  - With a = 0 the index pad meets the thumb at an angle: coarse candidates at a = 0 reached 0.02 mm, but the overlap-free fine candidate chosen there was 0.12 mm.
  - The search covers index f = 0 / 10 / 20 and a = −10 / 0 / +10.
- **OK sign (Perfect).** The fingertips touch, cap on cap, 0.87 mm short of pad on pad.
  - A more curled index (f = 45–60) runs its middle sleeve into the thumb first.
- **CloseHand.** In the stock command the thumb folds into the palm, where the 3-phalanx index tip now curls; it passed 11 mm through the index.
  - The searched command lays the thumb over the curled index's middle phalanx (a fist with the thumb outside).

![peace](../../../Demo/AHSimulation/ultimate_finger/renders/hand_Peace.jpg)

![peace with shells](../../../Demo/AHSimulation/ultimate_finger/renders/hand_Peace_shells.jpg)

![middle finger](../../../Demo/AHSimulation/ultimate_finger/renders/hand_MiddleFinger_shells.jpg)

![pinch](../../../Demo/AHSimulation/ultimate_finger/renders/hand_Pinched.jpg)

![open hand with shells](../../../Demo/AHSimulation/ultimate_finger/renders/hand_OpenHand_shells.jpg)

### Notes for the palm shells and the wrist

- The fingers **curl much further into the palm** than the stock ones: MCP 89° instead of 75°, and the fingertip comes to ~51 mm from its MCP (stock 69 mm), 14 mm further under the knuckle line. The shells in `cad/five_finger/stl` are fitted to the merged sweeps of `AH_Right_5F` and this hand (coordinator). If the finger geometry changes again, refit them: `make_5f_shells.py` takes `--model AH_Right_5F_3S`, and its keep-out sweep uses every finger geom, all of which are meshes.
- The pinky is shorter (×0.88 phalanges), so its tip sweep is smaller.
- The sleeves are 15.8 mm wide (proximal) instead of 19.4 (slim) and 23.8 (stock): the openings between the fingers in the palm shell can follow the thinner fingers.
- Moving mass per finger 21.0 g (stock 18.4 g, slim ≈ 18 g).
- Software for this hand: straight at f = −56° (not −37.5°), clamp f ≥ −60° (m1 − m2 ≥ −120°), abduction room |a| ≤ 90 − |f|, torque limit as in section 5.

## 8. BOM delta per finger (index, middle, ring, pinky), vs the stock finger

| Stock part | Ultimate finger | Material |
|---|---|---|
| Link | `UF_Link` (stock base + 3.0 mm tapered arm); re-use the 2 ball studs and 2 M2×10 screws | PA-CF |
| Proximal | `UF_P1_Proximal` (stock MCP end + new 12 mm body) | PA-CF |
| Distal | `UF_P2_Middle` | PA-CF |
| – | `UF_P3_Distal`, `UF_L2_Coupler` | PLA / PA-CF |
| Proximal_Shell, Distal_Shell | `UF_P1_Shell`, `UF_P2_Shell`, `UF_P3_Shell` (fingertip cap) | white TPU 82A |
| – | `UF_P1_Pad`, `UF_P2_Pad`, `UF_P3_Pad` (ribbed; the fingertip pad wraps the end) | black TPU 82A |
| 2× Ø2×16, 2× Ø2×10 pins | MCP Ø2×16 and Link pivot Ø2×10 kept; **3× Ø2×12 + 2× Ø2×8** hardened dowels (PIP, A, DIP; P4, B) | steel |
| 4× M2.5×6 shell screws | none (shells glued, flexible CA) | |

Net per finger: +2 printed rigid parts, +1 TPU shell, +3 pads, −1 Ø2×16, −1 Ø2×10, +3 Ø2×12, +2 Ø2×8, −4 screws. About 9.4 cm³ of rigid filament, 5.3 cm³ of white TPU (covers) and 0.4 cm³ of black TPU (pads) per finger (pinky 8.2 / 4.3 / 0.3) (stock finger set: 12.1 + 7.0). Unchanged: servos, horns, rods, ball joints, gimbal, finger frames, bushings, plate, electronics, IDs. Pinky: the same parts with the `_S088` suffix. Thumb: the slim-finger cores and screws, `Thumb_Proximal_Shell`, `Thumb_Distal_Shell`, `Thumb_Proximal_Pad`, `Thumb_Distal_Pad`.

## 9. Print and assembly notes

- **Rigid parts:** print on a side face (pin axes vertical) so pin bearing loads lie in the layer plane; 100 % infill for the Link, L2 and P2 (they are thin); 0.12–0.16 mm layers. Ream the holes with a Ø2 drill: Ø1.95 press (P1 at PIP and A, P2 at P4 and DIP, P3 at B), Ø2.15 running (P2 at PIP, P3 at DIP, Link at P4, L2 at A and B). Remove the elephant's foot on every face that runs against another part: the running gaps are 0.25 mm.
- **TPU covers (white):** stand them on their flat end face; no supports. The cap stands on its base face, nose up. 3 perimeters make the 1.2–1.27 mm wall. Where the outside rounds beyond the cavity, the wall is locally solid up to about 2 mm; infill fills it.
- **Pads (black TPU):** the ribs are 0.55 mm wide and 0.3 mm high at a 1.0 mm pitch. Use a 0.4 mm nozzle and 0.1 mm layers. The proximal and middle pads print ribs-up. The fingertip pad, curved around the end, prints on its flat inner side or as a two-material job together with its cap (they share a frame). Glue into the land pocket with flexible CA; the 0.1 mm gap takes the glue.
- **Assembly order** (the P4 and B holes are only reachable at certain poses): P1 on the gimbal (MCP pin) → Link (pivot pin) with its arm in P1's channel → slide the P1 sleeve on from the PIP end → slide the P2 sleeve onto P2 → P2 into P1's fork with the PIP pin → flex P2 fully so its P4 hole clears P1's fork, press the P4 pin through P2 and the Link eye → L2 onto the A pin (through P1's fork, L2's eye and P2's slot) → P3 into P2's fork with the DIP pin → flex the DIP fully so B rises above P2's walls, press the B pin → glue the cap on P3.
- **Hard stops (recommended addition, not in the CAD):** the model has PIP/DIP stops at −25° / −22°; add small stop faces on P2 against P1 and on P3 against P2 there. See risks.

## 10. Risks and open points

- **PIP 78° / DIP 54°**, not the human 100° / 70°. The six-bar study shows how to get there (section 2) at the cost of width, parts and play.
- **Strength in deep flexion**: above f ≈ +45 the servos can out-push the parts; the torque limit alone does not cover the last 45° (section 5). Not load-tested.
- **Loop-2 branch point.** The DIP coupler's four-bar has a second assembly branch, reached at PIP ≈ −27° (the design never goes below −21°). In the simulation, abrupt full-range command steps let the soft closures overshoot and flip loop 2 (DIP ≈ −72°) until PIP/DIP stops were added to the model. A rigid printed linkage cannot reach −27°, but play and a hard knock could; add the stop faces above.
- **Print tolerance**: 0.25 mm running gaps, 1.95/2.15 holes, 1.5 mm boss walls and a 1.75 mm press-fit length per fork wall. Expect one or two fit iterations; first print a P1/P2/pin set.
- **TPU fit and retention**: sleeves and cap are glued, not screwed; their min clearance to the core is 0.17–0.19 mm in places (0.25 nominal) because of the loft between sections. Not printed.
- **Covers and pads.**
  - The covers' end edges are rounded only where the B-rep fillet succeeded (the report says which); elsewhere the outer edge is sharp.
  - The 0.55 mm ribs and 0.45 mm grooves are at the limit of a 0.4 mm nozzle in TPU.
  - The fingertip pad is a curved part; printing it flat-side down needs supports, or a two-material print with its cap.
  - The pinky's middle-sleeve and fingertip covers clear each other by exactly the 0.5 mm design margin at full DIP flexion; any print oversize there rubs.
- **Thumb not restyled**: it keeps the slim shells with the new black ribbed pads, so it looks squarer than the fingers. A thumb in the same style (rounded sleeves, wrapped fingertip pad) is the next styling step.
- **Fingertip core trim**: the P3 core's tip body is trimmed to the cap cavity (0.25–0.27 mm clearance) from just before the designed tip. The OCC booleans for this trim are fragile; the build tries several variants and logs which one worked, and the cap-vs-core check (0 mm³ overlap) confirms it.
- **Pin play**: 7 finger pins + 2 rods; ±2.5° PIP and ±3° DIP play with 0.05 mm radial play per pin (six-bar study).
- **Not verified**: anything physical (printing, fit, friction, wear, strength), contact/grasp simulation (all geoms are visual), the left hand, the wrist on this model (coordinator).

## 11. Shorter pinky (×0.88)

The finger is parametric in its phalanx lengths: `make_ultimate_finger.py --scale s` scales P1, P2 and P3 and keeps everything at the knuckles: the MCP interface, the drive, the Link base and the crank offsets of both four-bars (P4, A, B relative to the PIP and DIP). The Link arm and L2 shorten with the phalanges. The whole CAD, shell planes and checks are regenerated ([report_cad_S088.txt](report_cad_S088.txt), [design_S088.json](design_S088.json)).

| Scale | P1 / P2 / P3 (mm) | Assembles over the whole travel | Straight at | MCP / PIP / DIP at f = +90 | Min transmission (flexion / whole travel) |
|---|---|---|---|---|---|
| 1.00 | 47.0 / 26.0 / 13.8 | yes | −56° | 88.8 / 78.1 / 54.4 | 47° / 25° |
| 0.90 | 42.3 / 23.4 / 12.5 | yes | −56° | 88.0 / 79.4 / 54.6 | 47° / 25° |
| **0.88 (built)** | 41.4 / 22.9 / 12.2 | yes | −56° | 87.8 / 79.7 / 54.7 (sim) | 47° / 25° |
| 0.85 | 40.0 / 22.1 / 11.8 | yes | −56° | 87.5 / 80.2 / 54.8 | 47° / 26° |

(planar, from the same drive map; 0.88 confirmed in the hand model, loop error ≤ 0.0015 mm settled). The linkage closes and keeps its range: shorter cranks-to-length ratios even add 1–2° of PIP. The pinky's fingertip ends 10.3 mm closer to its knuckle (83.2 vs 93.5 mm). Widths are unchanged (15.8 / 15.0 / 13.5 mm), so the pinky looks like a smaller copy. Same straight command and calibration as the other fingers.

## 12. Reproduce

Run from the repo root, **one at a time** (the CAD build peaks at about 6 GB). `AH_Right_5F_3S` is rebuilt by the first three lines:

```bash
.venv/bin/python cad/five_finger/ultimate_finger/make_ultimate_finger.py                          # ~8 min: STEP/STL, checks, figures
.venv/bin/python cad/five_finger/ultimate_finger/make_ultimate_finger.py --scale 0.88 --no-figures # ~6 min: the pinky
.venv/bin/python Demo/AHSimulation/ultimate_finger/build_models.py                                 # model/finger_uf.xml + AH_Right_5F_3S (--shells DIR: palm shells, default cad/five_finger/stl)
cd Demo/AHSimulation/ultimate_finger
../../../.venv/bin/python verify_finger.py          # ~3 min, reports/verify_finger.txt, figures/strength.png
../../../.venv/bin/python verify_hand.py --keep-pinch   # ~50 min, reports/verify_hand.txt (incl. the plate), data/hand_poses.json
../../../.venv/bin/python thumb_refine.py           # ~35 min, reports/thumb_poses.txt (thumb commands of CloseHand, Perfect, Pinched)
../../../.venv/bin/python verify_hand.py --only-plate    # ~1 min, reports/verify_hand_plate.txt
../../../.venv/bin/python verify_hand.py --only-shells   # ~5 min, reports/verify_hand_shells.txt (cad/five_finger/stl, or --shells DIR)
../../../.venv/bin/python baseline_neighbours.py --model AH_Right_5F   # stock-finger baseline on the same layout
../../../.venv/bin/python render.py --shells        # renders/ (hand_<pose>_shells.jpg with the palm shells)
../../../.venv/bin/python sixbar/sixbar_study.py    # the six-bar stretch study (planar)
```

View the hand:

```bash
.venv/bin/python -m mujoco.viewer --mjcf=Demo/AHSimulation/AHSimulation/AH_Right_5F_3S/mjcf/scene_shells.xml   # or scene.xml without the shells
```

## 13. Files

| Path | Contents |
|---|---|
| [make_ultimate_finger.py](make_ultimate_finger.py) | CAD build: cores, pins, lofted covers (styled outside, fingertip cap), lands/pockets/ribbed pads (mesh texture), thumb pads; exports, checks, figures |
| [uf_layout.py](uf_layout.py) | 2D layer layout and carving of the cores (bands, blanks, priorities), the motion |
| [uf_checks.py](uf_checks.py), [uf_figures.py](uf_figures.py) | CAD checks (validity, watertight, pin-hole walls, thin features, widths, TPU walls, joint sweeps) and figures |
| [design.json](design.json), [design_S088.json](design_S088.json) | linkage (copied from the 3-phalanx study), layers, pins, shell planes, parts, materials, which sim body carries which mesh |
| [step/](step/), [stl/](stl/) | `UF_Link`, `UF_P1_Proximal`, `UF_P2_Middle`, `UF_P3_Distal`, `UF_L2_Coupler` (rigid); `UF_P1/P2/P3_Shell` (TPU); `UF_P1/P2/P3_Pad` (black TPU; STEP untextured, STL ribbed); pins; `Thumb_*`; `*_S088` = pinky. Every part one solid, every STL watertight |
| [report_cad.txt](report_cad.txt), [report_cad_S088.txt](report_cad_S088.txt) | CAD check reports |
| [figures/](figures/) | `assembly`, `exploded`, `sections`, `width_profile`, `layers`, `side_poses`, `strength` |
| [Demo/AHSimulation/ultimate_finger/](../../../Demo/AHSimulation/ultimate_finger/) | `build_models.py`, `verify_finger.py`, `verify_hand.py`, `thumb_refine.py`, `baseline_neighbours.py`, `render.py`, `model/`, `reports/`, `renders/`, `data/hand_poses.json`, `sixbar/` |
| [AH_Right_5F_3S/mjcf/](../../../Demo/AHSimulation/AHSimulation/AH_Right_5F_3S/mjcf/) | the generated 5-digit hand |

## 14. What is verified and what is not

**Verified (CAD, MuJoCo 3.15)**, after the restyle, for the index-size finger and the ×0.88 pinky:
- Every part is one valid solid and a watertight STL.
- Pin-hole walls ≥ 1.52 mm.
- TPU walls ≥ 1.22 mm in sections (1.27 nominal); 3-D 1st percentile ≥ 1.22 mm.
- Covers 0.17–0.19 mm from their cores, never overlapping them.
- Pads never widen the finger; widths are 15.8 / 15.0 / 13.5 mm.
- Covers and pads clear everything moving past them by ≥ 0.50 mm over the whole servo travel.
- The sim matches the planar linkage within 0.4°, and the drive is unchanged.
- Stability and loop closures hold over the full range; loop 2 stays on its branch.
- No new part touches another in the working range (exact mesh booleans).
- The pin and link loads and the resulting stresses were computed.
- For the 5-digit hand: structure, stability, neighbour clearances, no contact with the final palm plate or the palm shells, and the poses above.

**Not verified:**
- Printing and fit: 0.25 mm gaps, press fits, glued covers, 0.55 mm TPU ribs.
- Friction, backlash and wear.
- Strength by test.
- Grasping with contact.
- The left hand.
- The wrist on this hand.
