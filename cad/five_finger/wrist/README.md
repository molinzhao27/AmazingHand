# Powered wrist and forearm base for the 5-digit AmazingHand

![overview](../../../Demo/AHSimulation/wrist/renders/wrist_overview.jpg)

This adds a 2-DOF wrist (flexion/extension and radial/ulnar deviation) and a forearm base to the 5-digit hand. The hand bolts on **unchanged**: no hand part is modified. The design is in the spirit of the Shadow hand: a slim gimbal at the wrist, the motors in the forearm, and a tapered forearm on an elbow flange.

The wrist is driven like an AmazingHand finger. Two servos in the forearm push two rods that end behind the gimbal:

- **Flexion** comes from moving the two rods together: ≈ (m1 − m2)/2.
- **Deviation** comes from moving them against each other: ≈ (m1 + m2)/2.

Everything was designed in CadQuery, assembled in MuJoCo and checked there. **Nothing has been printed or built yet** (see [Limits](#limits-and-open-questions)).

![mechanism](../../../Demo/AHSimulation/wrist/renders/wrist_mechanism.jpg)

## Files

| What | Where |
|---|---|
| Design parameters + linkage kinematics (IK/FK, Jacobian), shared by CAD, sim and control | [wrist_design.py](wrist_design.py) |
| CadQuery source: STEP + STL of every printed part, bought-part envelopes, printability / interference checks | [make_wrist.py](make_wrist.py), [step/](step/), [stl/](stl/), [stl/ref/](stl/ref/), [wrist_parts.json](wrist_parts.json), [previews/](previews/) |
| MuJoCo model (generated, do not edit) | [AH_Right_5F_Wrist/mjcf/](../../../Demo/AHSimulation/AHSimulation/AH_Right_5F_Wrist/mjcf/) |
| Generator, mass model, checks, renders | [build_wrist_model.py](../../../Demo/AHSimulation/wrist/build_wrist_model.py), [hand_mass_model.py](../../../Demo/AHSimulation/wrist/hand_mass_model.py), [verify_wrist_model.py](../../../Demo/AHSimulation/wrist/verify_wrist_model.py), [check_thumb_rule.py](../../../Demo/AHSimulation/wrist/check_thumb_rule.py), [render_wrist_model.py](../../../Demo/AHSimulation/wrist/render_wrist_model.py), [wrist_sim.py](../../../Demo/AHSimulation/wrist/wrist_sim.py) |
| Reports and renders | [reports/](../../../Demo/AHSimulation/wrist/reports/), [renders/](../../../Demo/AHSimulation/wrist/renders/) |
| Control code: AHControl patch + TOML, Python demo | [control/](../../../Demo/AHSimulation/wrist/control/) |

The STEP/STL files are in mm, in the hand's CAD part-studio frame (x from index to pinky, y along the fingers, z towards the back), at the neutral wrist pose. Each part prints as exported, in the orientation listed under [Printing](#printing). The `L-*` files are the left-hand versions: the `R-*` parts mirrored about x = −50 mm, the same rule that maps the stock R parts onto the L parts. The ring, horn arm and elbow flange are their own mirror images. There is no left-hand sim model.

## View it

```bash
.venv/bin/python -m mujoco.viewer --mjcf=Demo/AHSimulation/AHSimulation/AH_Right_5F_Wrist/mjcf/scene_shells.xml
.venv/bin/python -m mujoco.viewer --mjcf=Demo/AHSimulation/AHSimulation/AH_Right_5F_Wrist/mjcf/scene.xml
```

In the viewer, `wrist_motor1` / `wrist_motor2` (the last two sliders) are servo angles, not wrist angles. To pose the wrist in wrist angles, use `wrist_design.servo_ik(phi, psi)`.

Regenerate everything (from the repo root, after the 5F steps in [FIVE_FINGER.md](../../../FIVE_FINGER.md)):

```bash
.venv/bin/python cad/five_finger/wrist/make_wrist.py                     # CAD + checks  (~20 s)
.venv/bin/python Demo/AHSimulation/wrist/build_wrist_model.py --keyframe   # MJCF          (~15 s)
.venv/bin/python Demo/AHSimulation/wrist/verify_wrist_model.py             # checks        (~10 min)
.venv/bin/python Demo/AHSimulation/wrist/check_thumb_rule.py               # thumb map     (~1 min)
.venv/bin/python Demo/AHSimulation/wrist/render_wrist_model.py             # renders       (~2 min)
```

## Architecture and why

```
 hand (unchanged) ── R-Wrist_Interface (stock) ══ 4 screws ══ R-Wrist_Hand_Adapter
                                                                 │  deviation axis (z), F683ZZ + d3 pins
                                                          Wrist_Gimbal_Ring (d11 cable bore)
                                                                 │  flexion axis (x), F683ZZ + d3 pins
 forearm:  R-Wrist_Fork (cap + fork) ── R-Forearm_Front / R-Forearm_Back ── Elbow_Flange ── stand
              2 x STS3215 in R-Wrist_Servo_Frame ── horn arms (r 26) ── M3 push rods ── lever bar on the adapter
```

### Gimbal

The gimbal is a ring Cardan joint. Its two axes cross at the wrist centre C, 25 mm below the wrist interface: CAD (24.5, −127.2, −11.5) mm, on the interface axis, and root (0, 0, −0.025) m.

- **Flexion axis (proximal).** The axis is along x. The fork on the forearm carries it. Flexion is the large motion (±60°), and putting its fork at the sides lets the hand swing without meeting a fork arm.
- **Deviation axis (distal).** The axis is along z. The hand adapter's front and back lugs carry it.
- **Cable path.** The ring has an 11 mm bore. The hand's bus cable runs through the interface's 34 mm bore, the adapter, the ring and the fork cap, then down the back of the forearm. It crosses the wrist on both axes, so it barely changes length.
- **Bearings.** Each of the four pins is a 3 mm dowel pressed into the ring. It turns in an F683ZZ flanged bearing in the outer lug.

The gimbal is deliberately compact: everything on the flexion axis stays inside |x| < 16 mm. At the palm-ward + deviation corners, a push rod (at |x| ≈ 19–21 mm) comes close to the flexion axis: 2.5 mm on the full ±60 × ±25° rectangle and 5.4 mm on the operating ROM. At the stop corners it crosses the axis. My first, larger gimbal (Ø34 ring, F684 bearings, lugs at |x| 17.5–24.5) was hit there by the rods at 11 of 143 ROM poses. `verify_wrist_model.py` reports this rod-to-axis distance and the clearance to the compact fork lugs: 3.6 mm on the ROM, 1.7 mm at the stops. See [Verification](#verification).

### Drive

The drive uses two parallel push rods. Each Feetech STS3215 in the forearm turns a 26 mm horn arm in a vertical plane: the servo axis is along x and the horn faces outwards. The arm pushes an M3 rod with a ball link at each end. The rods end on ball studs at the ends of a lever bar behind the gimbal, on the hand adapter:

- the bar's stud balls sit at (x, y, z) = (±21, 0, 21.25) mm from C;
- 21.25 mm is the flexion lever and 21 mm the deviation lever;
- the rod lengths between ball centres are 66.0 mm (motor1) and 96.0 mm (motor2).

Every ball stud points along x. All the arm's out-of-plane rod tilt is therefore about the stud axes. The bottom ball links see ≤ 1.7° of misalignment and the top ones ≤ 26.7°, which comes from the deviation itself.

**Parallel, not serial.** I chose a parallel two-servo drive over a serial gimbal with one servo on each axis because:

1. **Both servos share every load.** Per N·m at the joint, each servo needs 0.61 N·m (flexion) or 0.62 N·m (deviation) at the neutral pose. That rises to at most 1.02 (flexion) or 1.18 (deviation) at the corners of the operating ROM. In a serial gimbal one servo always carries the whole axis, and the distal servo also rides on the proximal one.
2. **Looks and inertia.** The 55 g servos sit in the forearm. Only the ≈ 32 g adapter + ring and the 11 g of rods move with the hand, and the wrist stays a slim Ø54 neck, as on the Shadow hand.
3. **Same principle as the AmazingHand finger.** The finger is also a two-servo parallel drive with ball-jointed rods, and the motor sign convention is the finger's: motor1 = ulnar (pinky-side) servo, motor2 = radial (index-side) servo; palm-ward flexion is (+, −) like CloseHand, and radial deviation is (+, +) like abducting toward the index.
4. **Cost.** The control is coupled, so it needs the IK in [wrist_design.py](wrist_design.py) (closed form, about 20 lines). The leverage also drops at the combined corners.

**Linearisation at the neutral pose.**

- flexion ≈ 1.224 · (m1 − m2)/2
- radial deviation ≈ 1.238 · (m1 + m2)/2

The wrist moves about 1.2× faster than the servos.

**Servo packaging.** The servo bodies lie lengthwise, front to back (gear end towards the back), and are stacked 30 mm apart (motor1 upper at y = −66, motor2 lower at y = −96 mm from C). The arm studs point inwards. The two arms then work in separate planes on opposite sides, and neither rod passes the other servo. One printed frame holds both servos.

### Range of motion

The operating ROM is coupled, like a human wrist. `wrist_design.clamp_wrist()` enforces it, and every command should go through it.

| | |
|---|---|
| flexion / extension | −60 … +60° |
| radial / ulnar deviation | ±25° up to ±40° of flexion, tapering linearly to ±15° at ±60° |
| servo angles needed | ≤ 54.9° (ctrlrange ±75°) |
| hard stops | flexion ≈ ±66°: the adapter flange lands on two pads on the fork cap. Deviation ≈ ±31°: two pads under the adapter flange land on the fork's bearing bosses. The sim joint limits are ±65 / ±30°. |

The linkage alone could do the full ±60 × ±25° rectangle. Towards its last corners, though, a push rod dips onto its own servo's gear-end envelope: at |flexion| ≥ 50° with |deviation| = 25°, and within 0.1 mm at (±60°, ±20°). Hence the taper, which keeps ≥ 1.4 mm.

### Forearm pronation / supination (3rd DOF)

I recommend against it for this build:

- **It would be heavy.** It would need a roll stage under the elbow flange (a third STS3215 with a crossed-roller or lazy-susan bearing) that carries the whole forearm + hand (≈ 0.95 kg) off-axis.
- **Cabling gets harder.** It needs either a slip ring or a ±180° cable loop for the 10 A hand supply and the bus.
- **Little gain on a fixed stand.** Most robot arms that would carry this (e.g. SO-100, Reachy) already have a wrist roll.

If needed, the elbow flange (4 × M5 on a 100 mm square, Ø20 centre hole) is the interface where a roll module would go. It is not designed here.

## Servo choice and torque budget

**Mass model** ([hand_mass_model.py](../../../Demo/AHSimulation/wrist/hand_mass_model.py)). Every hand mesh is integrated (trimesh) with a material density:

- PLA 1.15 g/cm³;
- TPU 1.10 g/cm³;
- steel 7.85 g/cm³;
- igus bushings 1.46 g/cm³;
- SCS0009 13.2 g each;
- +30 g of unmodelled screws and cables.

The PLA value is **calibrated**: the same model gives the stock hand (AH_Right + stock shells) 397 g, against the 400 g in the README. With 1.00 g/cm³ it would be 375 g.

Results:

- **5-digit hand with shells and pads:** 472 g, CoM 92 mm from the wrist centre.
- **Everything the wrist moves:** the adapter and ring (with pins, bearings, studs and screws) add 31.5 g, for 503 g with the CoM 86.9 mm from C. The rods weigh 4.8 / 6.5 g.
- **Payload case:** 200 g placed at the OpenHand fingertip centroid of index…pinky, 201 mm from C.
- **Inertia about the axes:** 0.0054 / 0.0058 kg·m² (flexion / deviation), and 0.0132 / 0.0139 kg·m² with the payload.

**Static torque per servo**, over the whole operating ROM, for six ways of mounting the forearm. The sim (servo outer loop settled) matches the analytic Jacobian model to within 0.08 N·m.

| load | worst pose / mounting | worst per-servo torque | neutral wrist, worst mounting |
|---|---|---|---|
| hand only | (−60°, ±15°), forearm horizontal, thumb up/down | **0.49 N·m** | 0.25 N·m |
| hand + 200 g at the fingertips | (−60°, ±15°), forearm horizontal, thumb up/down | **0.98 N·m** | 0.50 N·m |

**The same worst case as a share of each candidate servo:**

| servo | stall | hand only | + 200 g |
|---|---|---|---|
| STS3215 C001 (7.4 V version) on the hand's 6 V rail | 1.62 N·m | 30 % | 60 % |
| STS3215 C001 at 7.4 V | 1.91 N·m | 25 % | 51 % |
| **STS3215 C018 (12 V version)**, chosen | **2.94 N·m** | **17 %** | **33 %** |
| STS3250 at 12 V | 4.9 N·m | 10 % | 20 % |

**Dynamic** (sim, minimum-jerk moves, C018 servo model, forearm horizontal palm-down / vertical hand-up):

- **±60° flexion sweep in 2.0 s:** ≤ 1.30 N·m peak (44 % of stall).
- **Same sweep in 1.2 s:** ≤ 1.96 N·m (67 %).
- **Same sweep in 0.8 s:** reaches the torque limit for 160 ms with the payload; tracking error 13.5°.
- **±25° deviation sweep in 0.6 s:** ≤ 2.06 N·m (70 %).

Most of the dynamic torque is the motor's own back-EMF / gear damping at speed. Inertia × acceleration is small: 0.26 N·m at 20 rad/s² even with the payload. So the realistic speed limit is about one full flexion sweep per ~1.2 s.

**Decision: 2 × Feetech STS3215 C018 (12 V, 30 kg·cm, 1:345, magnetic encoder, 55 g).** With the 200 g payload, in the worst pose and mounting, it runs at ≈ 33 % of stall, i.e. its rated torque. That leaves a 3× margin to stall for dynamics. The 7.4 V C001 on the hand's 6 V rail would carry the hand alone everywhere (≤ 30 %), but not the payload at the extremes (≈ 60 % of stall continuously).

Same size, same 25T horn, same protocol and registers: a C001 drops in for hand-only use with no CAD change. The sim uses the `sts3215_345` values that Pollen identified for the Open Duck Mini (kp 17.11 N·m/rad, damping 0.56, armature 0.027, frictionloss 0.068), with the force limit set to the datasheet stall of 2.94 N·m.

**Position sag.** A Feetech servo is a P controller. With the hand up and the wrist at 60°, its compliance plus the linkage gain let the wrist sag 1.5–2.7° from the command. The controller should therefore run a small outer loop on the servo angles (read `present_position`, nudge `goal_position` by 0.1 × error at 20–50 Hz). In the sim this brings every ROM pose to within 0.06° of the command. Do not run that loop faster than the servo's own ~3 Hz position loop: a gain of 0.4 at 50 Hz oscillated in the sim.

## Mounting the hand: the wrist interface pattern

I read the bolt pattern from `R-Wrist_Interface.step` with CadQuery/OCP. Angles are measured as atan2(z, x) about the interface axis at CAD (x, z) = (24.5, −11.5): 0° = pinky side, 90° = back.

- **Spigot.** A Ø49 / Ø34 ring spigot from y = −95 to −102.2 mm. Its end face has three protruding pads (sectors 31–89°, 151–209°, 271–329°, at y = −102.2) and three pads recessed by 1.2 mm between them (at y = −101).
- **Screw holes.** 6 × Ø2.4 blind holes, 9 mm deep, on a 42 mm circle, in pairs at 0° ± 20°, 120° ± 20° and 240° ± 20°, in the recessed pads. Ø2.4 is the pilot for a 3 mm thermoplastic (PT / thread-forming) screw.
- **Locating holes.** 3 × Ø2.5 × 4 at 0°, 120° and 240°, on the same circle.

`R-Wrist_Hand_Adapter`'s flange (Ø54) mates the recessed pads. It has 1.4 mm pockets for the protruding pads, which key it rotationally. It takes **4 screws (3.0×12 PT or M3×12 thread-forming) at ±20° and ±140°**, heads from below in counterbores. The ±100° pair lies under the adapter's front and back lugs and is left empty. The Ø22 centre bore passes the hand's bus cable.

## Forearm base

- **Shape.** A tapered shell, lofted with smooth transitions. It is 71 × 76.5 mm at the servos, narrows at the palm side towards the wrist, and flares to 82 × 89 mm at the elbow. It is split front/back at the servos' back face (z = −10 mm from C).
- **Servo mounting.** `R-Forearm_Front` holds the printed servo frame: 2 countersunk M3 through the palm side and 2 through the side walls into heat-set inserts. `R-Forearm_Back` is a cover. Its two ribs keep the servos seated in the frame.
- **Fork.** `R-Wrist_Fork` (cap + fork) carries the flexion bearings, the rod slots (computed from the rod sweep over the full stop rectangle + 4.5 mm), the flexion-stop pads and the Ø12 cable hole. It screws to both shells with 4 × M3×10.
- **Elbow flange.** `Elbow_Flange` (120 × 120 × 6 mm) has 4 × Ø5.5 counterbored holes on a **100 mm square** for M5 into a plate, table or aluminium extrusion frame, plus a Ø20 cable hole. The shells screw onto it from below (4 × M3×10).
- **Electronics bay.** Below the servos there is an 85 mm tall bay with room for the bus adapter and power distribution. A Ø12 cable notch at the back of the elbow end leads out.
- **Cable channel.** The hand cable comes down the back centre, between the rods and the cap screws.

## Printing

Results from `make_wrist.py`:

- Every part is one valid, watertight solid.
- At the 1st percentile, local wall thickness is ≥ 1.25 mm on every part, and < 0.4 % of the surface samples are under 1.2 mm (edges and chamfers).
- Overhangs > 45° total ≤ 100 mm² per part (chamfers, the bottom of round bosses). The exception is the servo frame, at 239 mm²: the ~47° undersides of its two side blocks. Horizontal ceilings are short bridges: the adapter's pad pockets (≤ 25 mm) and the counterbores. Each part's report is in [previews/cad_checks.txt](previews/cad_checks.txt).

![print orientation](previews/parts_print_orientation.jpg)

| Part | g (1.0 g/cm³) | Print | Material |
|---|---|---|---|
| R-Wrist_Hand_Adapter | 19.9 | flange face down, lugs up; the three 1.4 mm pad pockets bridge, gussets carry the lever bar | PETG (load path) |
| Wrist_Gimbal_Ring | 1.6 | flat | PETG |
| R-Wrist_Fork | 24.9 | cap face down, fork up; teardrop bearing seats | PETG |
| R-Forearm_Front | 60.5 | standing on the elbow end | PLA |
| R-Forearm_Back | 72.8 | standing on the elbow end | PLA |
| R-Wrist_Servo_Frame | 30.7 | base down | PLA |
| Wrist_Horn_Arm ×2 | 2 × 2.4 | flat, stud face down | PETG |
| Elbow_Flange | 80.9 | flat | PLA |
| **Total** | **≈ 296 g** | | |

PLA works for every part. I suggest PETG for the four gimbal and linkage parts because they carry sustained loads (creep).

## BOM delta (right hand; the left is the same with L- parts)

| Item | Qty | Notes |
|---|---|---|
| Feetech STS3215 **C018** (12 V, 30 kg·cm) + its 25T disc horn and horn screw | 2 | IDs 21/22. A C001 (7.4 V) fits for hand-only use |
| F683ZZ flanged ball bearing 3×7×3 | 4 | pressed into the four lugs |
| Dowel pin Ø3×10, ISO 8734 | 4 | pressed into the ring; push out from the bore to remove |
| Shim washer DIN 988 3×6×0.5 | 4 | optional, takes up axial play at the ring |
| M3 ball stud, Ø5.8 ball, ball centre 6.5 mm above the seat | 2 | lever bar (heat-set inserts) |
| M3 ball stud, Ø5.8 ball, ball centre 6.0 mm above the seat | 2 | horn arms (M3 nut in the arm) |
| M3 ball link for Ø5.8 balls, **swivel ≥ ±30°** | 4 | the top links see up to 26.7° |
| M3 threaded rod, stainless | 2 | adjust to 66.0 and 96.0 mm between ball centres; 4 × M3 jam nuts |
| M3 heat-set insert (Ø4.0 hole) | 14 | adapter 2, shells 8, servo frame 4 |
| 3.0×12 thermoplastic screw (or M3×12 thread-forming) | 4 | adapter → wrist interface |
| M3×10 socket head | 8 | fork cap 4, elbow flange 4 |
| M3×8 countersunk | 4 | servo frame (2 through the palm side, 2 through the side walls) |
| M2.5×6 | 8 | horn arm → disc horn (holes drawn on a 14 mm circle at 45°: **check your horn**, `make_horn_arm()`) |
| M3 nut | 2 | arm studs |
| M5 screws + T-nuts / nuts | 4 | elbow flange → stand |
| 3-wire servo cable, ~0.35 m | 1 | hand bus cable through the wrist (crimp its connector after threading the Ø11 ring bore) |
| 12 V supply ≥ 6 A + buck 12→6 V ≥ 10 A (or keep the hand's 6 V supply) | 1 | see [Power](#power) |
| Optional: 2nd USB-TTL bus adapter for the wrist | 1 | see [Control](#control-code-changes) |

Added mass: printed parts ≈ 296 g, servos 110 g, hardware ≈ 35 g, for ≈ 0.45 kg of wrist + forearm. Of that, only ≈ 32 g (adapter + ring) plus the 11 g of rods move with the hand.

**Assembly order:**

1. **Wrist bearings.** Press the bearings into the fork and adapter lugs, flanges inwards.
2. **Hand adapter.** Screw the adapter to the hand's wrist interface.
3. **Gimbal.** Put the ring between the fork lugs and press two pins in from outside through the Ø4 holes. Then put the adapter over the ring and press the other two pins in. The back one goes through the access hole in the lever block.
4. **Servos.** Centre each servo (goal 0 rad = raw 2048) and fit its horn arm pointing backwards (+z), square to the rod. Slide the servos into the frame from the back.
5. **Frame and shells.** Screw the frame in and fit the rods. Then close the back cover, cap screws first, then the flange.

## Servo IDs

| | right hand | left hand |
|---|---|---|
| wrist motor1 (ulnar / pinky side, upper servo) | **21** | **31** |
| wrist motor2 (radial / index side, lower servo) | **22** | **32** |

This keeps the existing rules: the hand uses 1–10 on the right and 11–20 on the left, left = right + 10, and motor1 is the odd ID. It does not clash with 1–20.

New STS servos ship as ID 1. Set the IDs one servo at a time, alone on the bus. `change_id` in AHControl only speaks SCS0009, and the STS EEPROM is lock-protected, so use the Feetech FD tool or rustypot's bus helper, which opens the lock:

```python
import rustypot
with rustypot.Bus("/dev/ttyACM0", 1_000_000, 0.1, {}) as bus:
    bus.change_id(rustypot.Sts3215PyController.definition(), 1, 21)   # then 1 -> 22 for the second servo
```

## Power

| branch | servos | voltage | moving / holding | stall (all) |
|---|---|---|---|---|
| hand (unchanged, see FIVE_FINGER.md) | 10 × SCS0009 | 5–6 V (max 7.4) | ~1.5–3 A | 10 A |
| wrist | 2 × STS3215 C018 | **12 V** | 0.18 A no-load each; ≈ 0.3–0.5 A each holding the hand | 2 × 2.7 A |

**The wrist servos cannot share the hand's 6 V rail at their rated torque,** and the SCS0009s must never see 12 V. Keep the two rails separate and share only GND and the data line. The Feetech TTL signal is referenced to GND, and its levels (high > 2 V, low < 0.45 V) do not depend on the servo supply.

Two ways to wire it:

- **(a) One adapter** feeds the hand branch as today. The wrist branch joins the data and GND lines through a splitter whose V+ pin is *not* connected, and is powered from 12 V. A 3-pin connector with the red wire cut works, labelled clearly.
- **(b) A second USB-TTL adapter** for the wrist on its own 12 V supply. Wiring is simplest, and the software then simply opens two ports.

A single 12 V ≥ 6 A supply can feed the wrist directly and the hand through a 12→6 V buck converter rated ≥ 10 A (≥ 5 A for light use). The wrist servos' own protection cuts torque at 70 °C or after 2 s at > 80 % of stall.

## Control code changes

**Library support.** rustypot supports the STS3215:

- **Python.** `rustypot` 1.12.0 (from PyPI; I installed it into `.venv` to check the API) has `Sts3215PyController` and the mixed-model `Bus`.
- **Rust.** Crate 1.12.0 has `servo::feetech::sts3215::Sts3215Controller`, with the same builder as the SCS0009 controller. AHControl's `rustypot = "1.1.0"` resolves to the latest 1.x because no Cargo.lock is committed; use `cargo update -p rustypot` if an older one is cached.

**Register layouts differ.** SCS0009 is big-endian, 1024 steps over 300°, 511 = 0 rad. STS3215 is little-endian, 4096 steps per turn, 2048 = 0 rad. So one controller cannot address both families. An STS ID in the SCS0009 sync write would get byte-swapped goals.

**Torque limit.** Set the wrist servos' `torque_limit` to 60–70 % for normal use. The ROM never needs more than 33 %, and the limit keeps the M3 ball studs well below yield if something blocks the wrist.

**Directions and zero.** Positive servo angle = increasing position = clockwise seen from the horn (datasheet). The sim and `wrist_design` use this convention. Before the first run:

1. Run `wrist_hw_demo.py --check-direction`, which moves each servo +10°. If a rod moves the wrong way, set that servo's `invert`.
2. Measure the zero offsets: with the hand held straight, read the present positions.

**Python** ([control/wrist_hw_demo.py](../../../Demo/AHSimulation/wrist/control/wrist_hw_demo.py), new):

- It drives the hand (IDs 1–10) and the wrist (21/22) on one port through `rustypot.Bus`, or the wrist on a 2nd adapter with `--wrist-port`.
- It runs IK → `clamp_wrist` → goals + outer loop.
- **Not run on hardware.**

To add the wrist to `PythonExample/AmazingHand_Demo.py` with a second adapter (option b):

```python
import sys; sys.path.insert(0, "../cad/five_finger/wrist")      # path to wrist_design.py
import wrist_design as wd
from rustypot import Sts3215PyController
w = Sts3215PyController(serial_port="COM12", baudrate=1000000, timeout=0.5)
WRIST_IDS, WRIST_ZERO, WRIST_INV = [21, 22], [0.0, 0.0], [False, False]   # your calibration (rad)

def Move_Wrist(Flex_deg, Dev_deg):            # flexion > 0 palm-ward, deviation > 0 radial
    th = wd.servo_ik(*wd.clamp_wrist(np.deg2rad(Flex_deg), np.deg2rad(Dev_deg)))
    goal = [(-1 if inv else 1) * (t + z) for t, z, inv in zip(th, WRIST_ZERO, WRIST_INV)]
    w.sync_write_goal_position(WRIST_IDS, goal)
```

…and in `main()`: `w.sync_write_torque_enable(WRIST_IDS, [True, True])`. With a single adapter (option a), the port can only be opened once: switch the whole script to `rustypot.Bus` as in `wrist_hw_demo.py`.

**AHControl (Rust/dora).**

- [control/ahcontrol_wrist.patch](../../../Demo/AHSimulation/wrist/control/ahcontrol_wrist.patch) removes the "Only SCS0009 motors are supported" rejection and accepts `SCS0009` or `STS3215` per motor.
  - It opens an `Sts3215Controller` on a `try_clone()` of the same serial port (one thread, so packets never interleave).
  - It routes each motor's goal to the controller of its family.
  - STS torque-enable takes `bool`.
  - Apply it with `git apply Demo/AHSimulation/wrist/control/ahcontrol_wrist.patch`. It is **checked to apply cleanly but not compiled**: cargo would have had to download the crates.
- [control/r_hand_5f_wrist.toml](../../../Demo/AHSimulation/wrist/control/r_hand_5f_wrist.toml) is `r_hand_5f.toml` + a `r_wrist` entry (IDs 21/22, `model = "STS3215"`). Point the dataflow's `--config` at it.
- `get_zeros` / `set_zeros` / `goto` / `change_id` still speak SCS0009 only. Use the Python snippets for the wrist servos.

**AHSimulation / dora sim node** (`mj_mink_right.py`). These changes are not made, because tracked files are left untouched here:

1. Load `AH_Right_5F_Wrist/mjcf/scene.xml`. The hand root still sits at the world origin at the neutral wrist, so the finger targets keep their positions.
2. Keep mink's finger IK off the wrist joints, `qpos[0:2]`, e.g. with a high-cost `mink.PostureTask` on those DOFs. Alternatively, solve the fingers on AH_Right_5F and copy the result.
3. Express the fingertip targets in the `r_wrist_interface` frame once the wrist moves.
4. Command the wrist from a wrist target (φ, ψ):
   - set `data.ctrl[10:12] = wrist_design.servo_ik(*wrist_design.clamp_wrist(phi, psi))`;
   - send `motor_pos = np.zeros(12)` with the wrist in `[10, 11]`;
   - add `self.metadata["r_wrist"] = [10, 11]`.
5. Hand tracking can give (φ, ψ) from the hand's orientation relative to the forearm.

## Sim model (`AH_Right_5F_Wrist`)

- **Structure.** `world → wrist_forearm` (fixed) carries the forearm, fork, frame, flange and both servos, then:
  - `wrist_gimbal_ring`: `wrist_flexion` hinge, axis root +y, > 0 palm-ward, range ±65°;
  - `wrist_hand_adapter`: `wrist_deviation` hinge, axis root −x, > 0 radial, ±30°;
  - `r_wrist_interface`: the AH_Right_5F hand subtree, copied unchanged and re-rooted.
- **Linkage bodies.** The horn bodies `wrist_horn1/2` carry the hinges `wrist_motor1/2` (axes along the servo shafts, > 0 = clockwise from the horn). The rod bodies `wrist_rod1/2` have ball joints. Each rod's upper ball is a `connect` equality to a site on the adapter.
- **Counts.** 54 bodies, 46 joints, nq = 97, nv = 80, 12 actuators, 27 equalities, 1 keyframe.
  - `qpos[0:2]` = wrist, `qpos[2:87]` = the AH_Right_5F hand in its own order, `qpos[87:97]` = linkage.
  - `ctrl[0:10]` = the finger actuators, unchanged in name and order; `ctrl[10:12]` = `wrist_motor1/2` (position, class `sts3215_12v`, ctrlrange ±75°).
- **Mass.** The hand gets a lumped inertial: the mass model (472 g with shells) minus the 110.6 g the finger bodies already carry, i.e. 361.3 g at root (−10.3, −11.8, 55.7) mm with a full inertia tensor. The wrist parts' inertials come from their STLs (PLA 1.0 g/cm³ + bought-part masses).
- **Closures.** The two rod closures use `solimp="0.9999 0.9999 0.001"`. With the hand's default (0.99 0.999) they stretched 0.35 mm under load (1–4° of wrist error); now ≤ 0.07 mm.
- **Keyframe.** Settled at ctrl = 0. The wrist sags 0.07° / 0.15° under gravity, and the hand's qpos matches AH_Right_5F's key to 7e-5.
- **Scenes.** `scene.xml` moves the floor under the elbow flange (z = −0.246 m). `scene_shells.xml` includes `robot_shells.xml`, which carries the 5F palm/back shells and grip pads on the moving palm.
- **No contacts.** All geoms are visual (contype 0), as in the hand model. Collisions are checked geometrically instead; see below.

## Verification

Full log: [reports/verify_AH_Right_5F_Wrist.txt](../../../Demo/AHSimulation/wrist/reports/verify_AH_Right_5F_Wrist.txt) (+ `.json`). CAD checks: [previews/cad_checks.txt](previews/cad_checks.txt).

**Loads.** Names are unique, and the actuator order is AH_Right_5F's + `wrist_motor1/2`. The keyframe matches.

**Stability.** No NaN and no MuJoCo warnings anywhere. Sweeps run with the fingers in OpenHand and in CloseHand:

| sweep | max \|qvel\| (rad/s) | push-rod closure error, max (mm) | finger closures, max (mm) |
|---|---|---|---|
| each wrist servo alone 0 → +75° → −75° → 0 | 1.1 | 0.023 | ≤ 0.009 |
| both servos around their ±75° box (drives the wrist onto its stops) | 4.9 | 0.062 | ≤ 0.009 |
| IK around the coupled-ROM boundary with the outer loop | 1.6 | 0.004 | ≤ 0.009 |
| 60 abrupt random steps of all 12 actuators | 91 (fingers) | 0.068 | 0.865 (same as AH_Right_5F / stock) |

**ROM.**

- The whole operating ROM is reachable with ≤ 54.9° of servo travel.
- Achieved vs commanded wrist angle: ≤ 2.7° open loop (P sag) and **≤ 0.06° with the outer loop**.
- Without gravity the kinematics match the IK to within the servo friction band (≈ 0.4°).

**Torque.** See the tables above. Static, sim vs analytic: within 0.08 N·m.

**Gimbal vs rods (analytic).** The push rods come within 5.37 mm of the flexion axis on the ROM and cross it (−0.55 mm) at the stop corners. They clear the compact fork lugs by 3.63 mm on the ROM and 1.72 mm at the stops.

**CAD at the neutral pose.**

- No intersection between any two printed parts or bought-part envelopes, apart from the intended press fits (pins in the ring) and the balls in their sockets.
- The cap's rod slots are cut from the rods' sweep over the whole stop rectangle, grown by 4.5 mm around the rod centreline. The clearance figures are in the table below.

**Clearance over the ROM** (surface samples every 1 mm, refined to 0.3 mm; wrist poses on the coupled ROM × keyframe / OpenHand / CloseHand / SpreadHand, plus 8 poses at the hard stops):

| pair | min clearance on the operating ROM | at the hard stops (outside the ROM) |
|---|---|---|
| hand (palm/back shells, pads, all digits) vs forearm, fork, ring, rods | **> 5 mm** | 1.98 mm (forearm, at −65 / −30°) |
| thumb over a 7×7 grid of its motors vs forearm | only the over-extended thumb ((m1, m2) = (−90, +90)) touches, at wrist flexion 60°. With thumb flexion (m1 − m2)/2 ≥ −70° it stays ≥ 3.3 mm everywhere ([reports/thumb_vs_forearm.txt](../../../Demo/AHSimulation/wrist/reports/thumb_vs_forearm.txt)), hence `wrist_design.thumb_flexion_min()` | – |
| hand adapter vs fork / forearm | 1.06 mm | 0.06 mm: the flange reaching its flexion-stop pad, as designed |
| adapter ball studs vs forearm | 3.70 mm | 1.43 mm |
| hand adapter vs ring, ring vs fork (gimbal gaps) | 0.73 mm | 0.73 mm |
| push rods vs fork, shells, frame | 1.25 mm | 0.08–0.12 mm at the (±65, ±30°) corners |
| push rod vs its own servo | 1.38 mm | 0.03–0.05 mm at the (±65, ±30°) corners |
| push rod (top link) vs hand adapter | 0.68 mm (at deviation ±25°) | 0.13 mm |
| rods vs the other servo / each other / other arm; arms vs each other | > 5 mm | > 5 mm |
| horn arms vs forearm | 0.78 mm | 0.78 mm |

The bought parts are modelled as envelopes:

- each servo is a 45.2 × 24.7 × 35 mm box plus a Ø24 × 3 mm disc horn;
- each ball link is a Ø9.5 × 7 mm socket with a Ø6.5 neck;
- the studs and rods are cylinders.

The smallest ROM clearances (0.68–0.78 mm) are therefore against conservative shapes. At the combined (±65°, ±30°) stop corners, which only an out-of-ROM command can reach, the rods would brush their servo's gear end.

## Limits and open questions

- **Not built.** Nothing has been printed or assembled. Print the adapter, ring and fork first to check:
  - the press fits (F683ZZ in Ø7.0, Ø3 pins in Ø2.9);
  - the 1.4 mm interface pockets;
  - the teardrop seats on your printer.
- **Assumed bought-part details:**
  - the STS3215 connector location: the frame leaves a window on the bottom face near the rear end;
  - the 25T disc horn's screw pattern (4 × M2.5 on a 14 mm circle, `make_horn_arm()`);
  - the ball-stud heights (6.0 / 6.5 mm to the ball centre);
  - the ball links' swivel: the top links need ≥ ±27°.
  
  Check these against your parts and edit the parameters in `wrist_design.py` / `make_wrist.py`.
- **Stud loads.** At servo stall a rod can push ~110 N (2.94 N·m / 26 mm). That bends the M3 ball stud close to yield. Normal loads are ≤ 40 N. Keep the STS3215 `torque_limit` at ≤ 60–70 % (the demo starts at 50 %); the worst static need is 33 %.
- **The sim is not a collision sim.** The wrist parts are visual geoms. Interference was checked geometrically on sampled poses (CAD booleans at neutral, surface-distance sweeps over the ROM).
- **Servo model.** It is borrowed from Pollen's Open Duck Mini identification of the STS3215, not identified on this wrist, so the P-sag numbers (1.5–2.7°) and dynamic margins are estimates.
- **The ROM is a software contract.** `wrist_design.clamp_wrist()` (coupled ROM) and `thumb_flexion_min()` (thumb interlock) must run in the controller.
  - The hard stops (≈ ±66° flexion, ≈ ±31° deviation) catch bad single-axis commands.
  - A combined out-of-ROM command near the stop corners can still press a rod against its servo's gear end.
  - The servos' own angle limits cannot express the coupled region.
- **Control code is untested.**
  - The AHControl patch applies cleanly but was not compiled.
  - `wrist_hw_demo.py` has not run on hardware.
  - The rotation sign (clockwise from the horn = positive) comes from the datasheet. Run `--check-direction` before anything else.
- **Mixed power.** Wiring the 12 V wrist branch onto the hand's 6 V bus is the main risk to the hand. Use a splitter with V+ cut, or a separate adapter.
- **Strength by rough hand calculation only (no FEA).** The fork stems and lugs see ≈ 10–15 MPa bending if both servos stall against a stop. The Ø3 pins and F683ZZ bearings carry ≈ 110 N with a wide margin. PETG is suggested for the four load-path parts (creep).
- **Left hand.** The L-* CAD parts are mirror images and were not checked separately. There is no left-hand MJCF. Its wrist servo directions are mirrored, so check them with `--check-direction`.
- **Rod length.** A 1 mm error in a rod's ball-to-ball length tilts the neutral wrist by ~2.7°. Calibrate the zero offsets with the hand held straight.
- **Cable.** The hand's 3-wire bus cable bends through the gimbal and is not strain-relieved in the design. Use flexible silicone wire, and crimp the connector after threading the Ø11 ring bore.
