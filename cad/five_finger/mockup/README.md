# All-printed mock-up of the 5-digit hand

A version of the final hand (`AH_Right_5F_3S_Wrist`: 3-phalanx fingers, slim thumb mount, sculpted shells, powered wrist and forearm) with no bought parts. It is for checking assembly and shape on a real print before buying servos and hardware.

- **Real printed parts:** every one is used unchanged.
- **Bought parts:** each is replaced by a printed part that fills the same holes, so you assemble the real hand.
- **Posable:**
  - The dummy servos have a plain output stub, and the printed horns sit on it with a friction fit.
  - The push rods are printed in one piece, with snap-on ball cups at the exact ball-to-ball length from the sim.
  - You can bend each finger and the wrist by hand, and they hold the pose.

| File | What it is |
|---|---|
| [make_mockup.py](make_mockup.py) | The stand-ins, with the fits as parameters at the top. It puts each one where the sim has the bought part and checks it against the real parts. Writes [stl/](stl/), [stl/test/](stl/test/) and [report.txt](report.txt). |
| [make_plates.py](make_plates.py) | Print orientation, plate packing and slicing (Anycubic Kobra X, Anycubic Slicer Next). Writes [plates/](plates/), one 3MF project per plate. |
| [plates/](plates/) | The plates: one 3MF project per group, already sliced, with its plates in it. Open one in Anycubic Slicer Next, check it, print plate by plate. The 3MFs are not in git (184 MB with their G-code): `make_plates.py` rebuilds them in about 10 minutes. [summary.json](plates/summary.json) lists every plate's time and filament. |

## Print order

1. **`00_test_fit_PLA.3mf`** (PLA, about 1 h 20 min). The real hole sizes, plus every stand-in fit in three sizes. Work through the table below and note the size that fits.
2. **`00_test_switch_PLA_TPU.3mf`** (PLA + TPU 95A, about 9 min). A PLA base with a TPU ribbed pad on top, with exactly one filament change. It checks that your Kobra X feeds TPU 95A and swaps PLA → TPU cleanly, and how well the two bond.
   - **`00_test_balls_PLA.3mf`** (PLA, a few minutes) is just the two ball posts. It is for re-testing the cups when your first fit test still had the old one-piece posts, whose Ø2.2 necks snap off.
3. Set the fits you picked at the top of `make_mockup.py`, then regenerate (see [Regenerate](#regenerate)).
4. Print the projects `01_…` to `08_…`.

## The fit test

The hole gauge has a notch at one corner. Its holes, from the notch: **1.50, 1.90, 1.95, 2.15, 2.20, 2.30, 2.70, 2.80, 2.90, 3.20, 4.00, 7.00, 8.15** mm. These are the holes of the real parts, printed the way the cores print (vertical).

Every other piece comes in three sizes: the current parameter and ±0.1 mm (the cups' lips ±0.2 mm). The pieces carry no size marks, and [fit_test_pieces.png](fit_test_pieces.png) shows each shape.

Sort each group of three by size before testing. Use calipers (measure across the round, not the flat), or try all three in the same hole or on the same stub:
- The loosest pin is the smallest.
- The loosest ring has the largest hole.
- The tallest cup has the highest lip.

The 12 mm pins are the `Fit_Pin` and `Fit_Axle` sets together: six pins of 1.9, 2.0, 2.0, 2.1, 2.1 and 2.2 mm.

| Pieces | Try them in | Right when | Parameter |
|---|---|---|---|
| `Fit_Pin_D*` | 1.95 and 2.15 | It presses into 1.95 (the 3-phalanx fingers' press holes) and turns freely in 2.15 (their running holes) | `PIN_D` |
| `Fit_Axle_D*` | 2.20 | It slides through (the finger Link's ball-stud hole) | `AXLE_D` |
| `Fit_Axle_D*` | `Fit_Ball_Hole_D*` rings | It presses in and holds | `BALL_HOLE` |
| `Fit_Peg_D*` | 2.80, then 1.90 | It passes 2.80 and presses into 1.90 (the 2.5 mm self-tapping screws' pilot holes) | `PEG_D` |
| `Fit_Horn_Bore_D*` | `Fit_Servo_Stub` | It turns by hand but holds its angle | `HORN_BORE` |
| `Fit_Cup_Hand_Lip*` | the hand ball post: press `Fit_Ball_Hand_Pin` through `Fit_Ball_Hand_Post` (flush underneath), then `Fit_Ball_Hand_Ball` flush onto its top | It snaps on with a click, swivels freely and does not fall off. A higher lip snaps tighter | `CUP_LIP` |
| `Fit_Cup_Wrist_Lip*` | the wrist ball post: `Fit_Ball_Wrist_Pin` (shank first, collar down) into `Fit_Ball_Wrist_Post`, `Fit_Ball_Wrist_Ball` flush onto its neck | Same | `WCUP_LIP` |
| `Fit_Pin3_D*` | 2.90 | It presses in (the wrist's gimbal ring) | `PIN3_D` |
| `Fit_F683_OD*` | 7.00 | It presses in (the wrist's bearing seats) | `F683_OD` |
| `Fit_Insert_OD*` | 4.00 | It presses in (the heat-set insert holes) | `INSERT_OD` |
| `Fit_PegM3_D*` | the insert you picked | It presses in and holds | `M3_PEG_D` |
| `Fit_PegM25_D*` | 2.70, then 2.30 | It passes 2.70 (horn arm) and presses into 2.30 (disc horn) | `M25_PEG_D` |
| `Fit_Disc_Bore_D*` | `Fit_STS_Stub` | It turns by hand but holds | `DISC_BORE` |
| `Fit_Arm_Shank_D*` | 3.20 | It presses in (the ball stud through the wrist's horn arm) | `STUD_ARM_SHANK` |

Also try a piece of 1.75 mm filament in 1.95, 2.15 and 2.20. It is the fallback for any Ø2 pin, and it is how the dummy servos are pinned to the finger frames (below).

**Checked on a Kobra X in PLA (2026-10-17):** every fit works at the value now in `make_mockup.py`, and so do the cups on the new ball posts.

## Stand-ins

No ball is printed on its stud. A thin neck printed upright breaks across its layers as soon as a cup is levered on it, as the first fit test's ball posts did. So every ball is a separate part, pressed onto a pin that prints lying down, so its layers run along it. The ball's press fit is the same as the fit test's ball rings on the axle pins.

[report.txt](report.txt) has every stand-in with its quantity and its deepest overlap with a real printed part when placed where the sim has the bought part. Only the press fits should show there.

| Stand-in | Qty | Replaces |
|---|---|---|
| `Dummy_SCS0009` | 10 | Feetech SCS0009. The sim's servo mesh, its spline a plain Ø4 stub. Its ears keep their Ø2.0 holes. |
| `Mock_Horn` | 10 | The reworked cross horn, with a nut-sized boss under its arm where the stud's nut sits. It slides onto the stub. |
| `Stud_Pin_D2.1_L8` + `Ball_D5` | 10 + 10 | The horn's M2 ball stud. The pin presses up through the boss and the arm, flush underneath. A ball presses onto its top, flush with the pin end. |
| `Rod_<digit>_M1/M2` | 10 | Two M2 ball joints, the M2 × 18 rod and the printed spacer. One piece, 33.0 mm between ball centres. The cup with its mouth up goes on the horn ball. |
| `Axle_D2.1_L20` + `Ball_D5` | 5 + 10 | The two ball studs on each finger's Link. The axle goes through the Link's Ø2.2 hole, with a ball pressed flush on each end. |
| `Pin_D2_L8/10/12/16` | 8 / 6 / 12 / 6 | The Ø2 dowels. |
| `Bushing_0608` | 10 | igus bushing 0608-04 (same shape). |
| `Washer_M2.5` | 5 | M2.5 large washer (same shape). |
| `Peg_2.5x8` | 22 | 2.5 × 8 self-tapping screws: gimbal with washer (5), finger frames → plate (3 per digit), back shell → hand adapter (2). |
| `Peg_2.5x16` | 3 | Back shell → palm plate ribs. |
| `Peg_2.5x6` | 4 | Hand adapter → palm plate legs. |
| `Dummy_STS3215` + `Disc_Horn` | 2 + 2 | Feetech STS3215 and its 25T disc horn (friction on a Ø6 stub). |
| `Peg_M2.5x6` | 8 | Horn arm → disc horn. |
| `Bushing_F683` | 4 | F683ZZ flanged bearings (running on the pins). |
| `Pin_D3_L10` | 4 | Ø3 × 10 gimbal pins. |
| `Ball_Stud_Top` / `Ball_Stud_Arm` + `Ball_D5.8` | 2 / 2 + 4 | M3 ball studs. The top ones press straight into the lever bar's insert holes, the arm ones through the horn arm (no nut). A ball presses flush onto each neck. |
| `Rod_Wrist_M1/M2` | 1 / 1 | Two M3 ball links and the M3 rod. One piece, 66.0 / 96.0 mm between ball centres. |
| `Insert_M3` | 12 | M3 heat-set inserts (press sleeves). |
| `Peg_M3x10` / `Peg_M3x8_CSK` | 8 / 4 | M3 screws: fork cap 4, elbow flange 4 / servo frame 4. |

**Not replaced:**

- **The servo screws** (four per finger, 2 × 7 mm self-tappers into Ø1.5 pilot holes): too small for a printed pin. Push a 1.75 mm filament piece through each dummy servo's ear hole into the finger frame. Ream the frame's Ø1.5 hole to 1.8 mm, or heat the filament tip. Or leave them out: the servo pair is trapped between the two finger frames and the plate.
- **The stock Spacer:** not printed, because the one-piece rods replace rod + spacer.
- **Cables, electronics and the M5 stand screws:** not part of the mock-up.

## Multi-material on the Kobra X

- **One swap per print needs a flat boundary.** The Kobra X changes filament on every layer that has both materials. A part prints with a single swap only if its two materials meet on one flat layer, as in the switch test.
  - The palm pad sits in a recess on a curved palm, so it would cost a swap on every layer it spans. It prints on its own.
- **Finger covers print as separate halves**, a white PLA back and a black TPU glove. A fused one-piece cover cannot be fitted to its core:
  - On the proximal and middle phalanges and the thumb's proximal, sliding the core in from either end runs into 160–480 mm³ of cover. Those are also the covers whose seam is flat enough for one swap.
  - The fingertip and thumb-tip covers could go on in one piece, but their glove wraps over the tip, so they need a swap on dozens of layers.
  - The halves go on from above and below the pinned core. The covers were designed in TPU, so a PLA back won't flex to snap on: check one finger's set first, and glue them if they don't hold.
- **TPU through the colour unit:**
  - Anycubic lists TPU 95A for the Kobra X only as a single material, and multi-material only with TPU 68D. Reviewers report 95A works through the colour unit, which is what the switch test checks.
  - The design's Filaflex 82A is not on Anycubic's list for this printer.
- **No prime tower:** the Kobra X flushes at its own purge station at every swap, so the presets don't purge into a tower.

## Plates

Every real part is white PLA except the finger gloves and the palm pad, which are black TPU 95A. The stand-ins are black PLA.

| Settings | Value |
|---|---|
| Layer height | 0.12 mm for the small precise parts and the cover backs, 0.20 mm for the big ones, 0.16 mm for TPU |
| Walls | 3 |
| Infill | 20 % gyroid |
| PLA temperature | 200 °C (210 °C first layer) |
| Avoid crossing walls | on |
| Supports | tree supports from the bed only, at the presets' threshold (20° at 0.12 mm, 30° at 0.16 and 0.20 mm). The palm plate, the shells and the pad also get them on their own surfaces. |
| Brims | only on parts that need one (tall and slender, a small base, or a line contact), on their own `_brim` plates |

Parts lie in the orientation from the build notes where there is one: finger cores with their pin axes vertical, so the pin holes print round, and wrist parts as `make_wrist.PRINT_UP`. Other parts take the direction with the least overhang that still sits on a flat face and doesn't stand tall and thin.

The big curved parts and the covers were oriented by slicing every axis plus the best few candidates from a support-volume estimate. Each keeps the least support among the orientations that are safe to print: low on the bed, or standing on a real base.

| Part | Orientation | Support |
|---|---|---|
| Palm plate | Back face down | 11 g (every other side 40 g or more) |
| Back shell | Roof down, 13 mm tall | 17 g, 3 h 30 min (open side down: 28 g, 4 h 25 min). The outside of the back sits on the supports, so expect marks there. |
| Palm shell | Rim up, with a brim | 31 g, 5 h 30 min (rim down: 46 g, 6 h 30 min). The palm face sits on the supports; the pad covers most of it. |
| Pad | Ribs up | 9 g. Standing on edge would need 1 g, but an 85 mm-tall TPU sheet wobbles. |
| Cover backs (PLA) | On their seam, dome up | 0.1–0.3 g each, inside where it doesn't show |
| Gloves (TPU) | On their outside, seam up | 0.1–0.4 g each: a few tips that peel off |

Standing either shell on its wrist end would need less support (7 g and 22 g), but it would be about 135 mm tall on a point (back shell, 0.1 mm² first layer) or on a 9 mm² sliver (palm shell). The back shell can't print "open side down with no supports", as FIVE_FINGER.md says: that leaves a 130 mm-wide roof with nothing under it.

Support in all: 136 g out of 716 g, down from 191 g out of 779 g. Most of it is under the palm shell, the back shell, the forearm halves and the palm plate.

| Project | Plates | Material | Time per plate | Filament | Contents |
|---|---|---|---|---|---|
| `00_test_fit_PLA.3mf` | 1 | PLA | 1 h 24 min | 13 g | Fit test: hole gauge + every stand-in fit in three sizes |
| `00_test_balls_PLA.3mf` | 1 | PLA | 8 min | 1 g | Just the two ball posts of the fit test (block, stud pin, ball), to re-test the cups |
| `00_test_switch_PLA_TPU.3mf` | 1 | PLA + TPU 95A | 8 min | 3 g | PLA / TPU 95A switch test (one filament change) |
| `01_hand_cores_PLA_white.3mf` | 2 | PLA white | 7 h 48 min, 46 min | 76 g | Finger cores (3-phalanx and thumb), finger frames, gimbals |
| `01_hand_cores_PLA_white_brim.3mf` | 1 | PLA white | 1 h 5 min | 10 g | The finger Links (UF_Link, stock thumb Link) |
| `02_palm_PLA_white.3mf` | 2 | PLA white | 3 h 30 min, 4 h 34 min | 133 g | Back shell (roof down); palm plate and hand adapter |
| `02_palm_PLA_white_brim.3mf` | 1 | PLA white | 5 h 33 min | 75 g | Palm shell (rim up) |
| `03_wrist_PLA_white.3mf` | 1 | PLA white | 3 h 53 min | 90 g | Gimbal ring, wrist fork, servo frame, horn arms, elbow flange |
| `04_forearm_PLA_white_brim.3mf` | 1 | PLA white | 5 h 6 min | 172 g | Forearm front and back (standing, 183 mm) |
| `05_standins_PLA_black.3mf` | 4 | PLA black | 6 h 2 min, 1 h 30 min, 42 min, 50 min | 99 g | Stand-ins: both kinds of dummy servo, horns, pins, pegs, bushings, washers, balls, Link axles and horn stud pins, wrist ball studs, inserts, wrist rods and disc horns |
| `05_standins_PLA_black_brim.3mf` | 1 | PLA black | 1 h 25 min | 10 g | Stand-ins: the 10 finger push rods |
| `06_covers_PLA_white.3mf` | 1 | PLA white | 1 h 9 min | 10 g | Cover backs of the proximal and middle phalanges and the thumb |
| `06_covers_PLA_white_brim.3mf` | 1 | PLA white | 19 min | 2 g | Fingertip cover backs |
| `07_TPU95A_black_pad_brim.3mf` | 1 | TPU 95A black | 2 h 34 min | 21 g | Palm grip pad (ribs up) |
| `08_TPU95A_black_gloves_brim.3mf` | 1 | TPU 95A black | 2 h 16 min | 20 g | Finger and thumb gloves |

In all, without the test plates: 12 projects and 17 plates, about 49 h of printing. That is 567 g of white PLA, 109 g of black PLA and 40 g of TPU 95A.

## Things to watch while assembling

- **Dummy servo vs finger frame 2.** The CAD servo's ears sit 0.6 mm inside Finger_Frame-2 in the stock model too. The dummy is trimmed to the frames with 0.15 mm clearance, so it goes in. A real servo's ears must be thinner than the model.
- **Thumb at home.** In the sim's home pose the thumb's slim distal core sits 0.7 mm inside its proximal core, and so does the end of their 2 × 10 pin. The sim never collides parts, so this is most likely the pose being past the thumb's extension stop. Check that the thumb's distal swings freely near full extension.
- **Covers.** These are the hardest prints: thin 1.3 mm skins. They lie low now, so they no longer stand on a rim. They were designed in TPU, so a PLA back is stiffer than intended. Print one finger's set first and check that it goes on.

## Regenerate

```bash
.venv/bin/python cad/five_finger/mockup/make_mockup.py    # stand-ins, test pieces, report.txt (~4 min, ~5 GB)
.venv/bin/python cad/five_finger/mockup/make_plates.py    # orientation, packing, slicing -> plates/*.3mf
.venv/bin/python cad/five_finger/mockup/make_plates.py --only 00_test   # just the two test plates
```

`make_plates.py` needs Anycubic Slicer Next in `/Applications`. It runs the slicer's command line with the Kobra X 0.4 mm presets that ship with it.
