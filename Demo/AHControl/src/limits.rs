//! Limits of the final 5-digit right hand, AH_Right_5F_3S (also AH_Right_5F_3S_Wrist): 3-phalanx index, middle, ring
//! and pinky, and the slim-mount thumb (FIVE_FINGER.md). Turned on by `limits = "AH_Right_5F_3S"` at the top of the
//! config. Ported unchanged from PythonExample/AmazingHand_Demo_5F.py (thumb_command_ok, clamp_thumb, converge_limit,
//! clamp_hand, check_hand, thumb_path, trajectory), which explains every number; `cargo test` checks the port against
//! it.
//!
//! A command is the right-hand motor command (m1, m2) of each digit, in deg, before the offset is added (the sim sends
//! it in rad). Fingers: f = (m1 - m2)/2, a = (m1 + m2)/2 (> 0 towards the index).

use rustypot::servo::feetech::scs0009::Scs0009Controller;
use std::{
    error::Error,
    thread,
    time::{Duration, Instant},
};

/// The value of the config's `limits` key for this hand.
pub const HAND: &str = "AH_Right_5F_3S";

/// The digits in the order of a `Cmd`: the fingers from the index side to the pinky side, then the thumb.
pub const DIGITS: [&str; 5] = ["index", "middle", "ring", "pinky", "thumb"];
/// The config's finger_name of each digit (FIVE_FINGER.md, "Servo IDs": index 1/2, middle 3/4, ring 5/6, thumb 7/8,
/// pinky 9/10).
pub const FINGER_NAMES: [&str; 5] = ["r_finger1", "r_finger2", "r_finger3", "r_finger5", "r_finger4"];
const FINGERS: usize = 4;
const THUMB: usize = 4;

/// (m1, m2) per digit, deg.
pub type Cmd = [[f64; 2]; 5];

// 3-phalanx finger limits (deg)
const F_MIN: f64 = -60.0;
const F_MAX: f64 = 90.0;
const M_MAX: f64 = 90.0;
const A_INNER: f64 = 30.0;
const CONVERGE_MAX: f64 = 15.0;
const CONVERGE_CURLED: f64 = 30.0; // when the more curled finger of the pair is at f >= CURL_FULL
const CURL_FROM: f64 = 60.0; // the limit rises linearly from CONVERGE_MAX to CONVERGE_CURLED between these
const CURL_FULL: f64 = 90.0;
const TOL: f64 = 1e-6;

const STREAM_HZ: f64 = 50.0;
const THUMB_PATH_STEP: f64 = 0.25; // deg, sampling of the thumb's clamped path
const HOME_SPEED: f64 = 1.5; // rad/s

// Thumb command envelope of the slim mount, right hand (cad/five_finger/layout.py, THUMB_ENVELOPE["slim"]):
// m2_min(m1) <= m2 <= m2_max(m1), linear between the tabulated m1, none outside them.
const ENV_M1: [f64; 22] = [
    -67.5, -60.0, -52.5, -45.0, -37.5, -30.0, -22.5, -15.0, -7.5, 0.0, 7.5, 15.0, 22.5, 30.0, 37.5, 45.0, 52.5, 60.0,
    67.5, 75.0, 82.5, 90.0,
];
const ENV_M2_MIN: [f64; 22] = [
    30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 22.5, 22.5, 22.5, 22.5, 22.5, 30.0, 30.0, 37.5, 45.0, 52.5, 60.0,
    67.5, 75.0, 82.5,
];
const ENV_M2_MAX: [f64; 22] = [
    30.0, 37.5, 45.0, 52.5, 67.5, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0,
    90.0, 90.0, 90.0,
];

/// Home: fingers at (0, 0), the thumb at clamp_thumb(0, 0) = (0, 22.5) ((0, 0) is inside the palm shell).
pub const HOME: Cmd = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 22.5]];

/// np.interp(x, xp, fp) for increasing xp, with numpy's arithmetic (its arm64 builds fuse the last line into one
/// multiply-add).
fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n = xp.len();
    if n == 1 || x <= xp[0] {
        return fp[0];
    }
    if x >= xp[n - 1] {
        return fp[n - 1];
    }
    let j = xp.partition_point(|&v| v <= x) - 1; // xp[j] <= x < xp[j + 1]
    if xp[j] == x {
        return fp[j];
    }
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    slope.mul_add(x - xp[j], fp[j])
}

/// Whether the thumb command (m1, m2) (deg) is inside the envelope.
pub fn thumb_command_ok(m1: f64, m2: f64) -> bool {
    let lo = interp(m1, &ENV_M1, &ENV_M2_MIN);
    let hi = interp(m1, &ENV_M1, &ENV_M2_MAX);
    lo - 1e-9 <= m2 && m2 <= hi + 1e-9 && ENV_M1[0] - 1e-9 <= m1 && m1 <= ENV_M1[21] + 1e-9
}

/// The nearest command inside the envelope, keeping m1 (deg).
pub fn clamp_thumb(m1: f64, m2: f64) -> [f64; 2] {
    let m1 = m1.max(ENV_M1[0]).min(ENV_M1[21]);
    let m2 = m2.max(interp(m1, &ENV_M1, &ENV_M2_MIN)).min(interp(m1, &ENV_M1, &ENV_M2_MAX));
    [m1, m2]
}

fn finger_fa(m: [f64; 2]) -> (f64, f64) {
    ((m[0] - m[1]) / 2.0, (m[0] + m[1]) / 2.0)
}

/// (lo, hi) of finger k's abduction at flexion f (deg): within 90 - |f|, and within A_INNER towards a neighbour.
fn abduction_range(k: usize, f: f64) -> (f64, f64) {
    let room = (M_MAX - f.abs()).max(0.0);
    let (mut lo, mut hi) = (-room.min(A_INNER), room.min(A_INNER));
    if k == 0 {
        hi = room; // the index spreading outward, towards the thumb
    }
    if k == FINGERS - 1 {
        lo = -room; // the pinky
    }
    (lo, hi)
}

/// How far a neighbour pair may converge (a_right - a_left, deg): CONVERGE_MAX, rising linearly to CONVERGE_CURLED as
/// the more curled of the two goes from f = CURL_FROM to CURL_FULL.
fn converge_limit(f_left: f64, f_right: f64) -> f64 {
    let s = ((f_left.max(f_right) - CURL_FROM) / (CURL_FULL - CURL_FROM)).max(0.0).min(1.0);
    CONVERGE_MAX + (CONVERGE_CURLED - CONVERGE_MAX) * s
}

/// A whole-hand command moved inside every limit; a command already inside comes back unchanged. Fingers: f clipped
/// to [F_MIN, F_MAX], then a to abduction_range(); a neighbour pair converging by more than converge_limit() is pulled
/// apart by turning back whichever of the two is swung towards the other, in proportion to how far.
pub fn clamp_hand(cmd: &Cmd) -> Result<Cmd, String> {
    for (d, c) in DIGITS.iter().zip(cmd) {
        if !(c[0].is_finite() && c[1].is_finite()) {
            return Err(format!("{d}: non-finite command {c:?}"));
        }
    }
    let eps = 1e-9; // what is inside by this much is left alone
    let mut fa = [[0.0; 2]; FINGERS];
    let mut moved = [false; FINGERS];
    for k in 0..FINGERS {
        let (f, a) = finger_fa(cmd[k]);
        let f2 = f.max(F_MIN).min(F_MAX);
        let (lo, hi) = abduction_range(k, f2);
        let a2 = a.max(lo).min(hi);
        fa[k] = [f2, a2];
        moved[k] = (f2 - f).abs() > eps || (a2 - a).abs() > eps;
    }
    for _ in 0..2 * FINGERS {
        let mut changed = false;
        for l in 0..FINGERS - 1 {
            let r = l + 1;
            let (a0, a1) = (fa[l][1], fa[r][1]);
            let lim = converge_limit(fa[l][0], fa[r][0]);
            if a1 - a0 <= lim + eps {
                continue;
            }
            let (p, n) = (a1.max(0.0), (-a0).max(0.0)); // how far each is swung towards the other
            let rest = a1.min(0.0) - a0.max(0.0); // <= 0
            let k = (lim - rest) / (p + n); // 0 <= k < 1
            if a0 < 0.0 {
                fa[l][1] = -n * k;
                moved[l] = true;
            }
            if a1 > 0.0 {
                fa[r][1] = p * k;
                moved[r] = true;
            }
            changed = true;
        }
        if !changed {
            break;
        }
    }
    let mut out = *cmd;
    for k in 0..FINGERS {
        if moved[k] {
            let [f, a] = fa[k];
            out[k] = [(a + f).max(-M_MAX).min(M_MAX), (a - f).max(-M_MAX).min(M_MAX)];
        }
    }
    out[THUMB] = clamp_thumb(cmd[THUMB][0], cmd[THUMB][1]);
    Ok(out)
}

/// Every limit of this hand, checked on a whole-hand command. Returns the violations.
pub fn check_hand(cmd: &Cmd) -> Vec<String> {
    let mut bad = vec![];
    for (d, &[m1, m2]) in DIGITS.iter().zip(cmd) {
        if !(m1.is_finite() && m2.is_finite()) {
            bad.push(format!("{d}: non-finite ({m1}, {m2})"));
            return bad;
        }
        if m1.abs() > M_MAX + TOL || m2.abs() > M_MAX + TOL {
            bad.push(format!("{d}: ({m1:.2}, {m2:.2}) beyond +-{M_MAX}"));
        }
    }
    let (mut fs, mut a) = ([0.0; FINGERS], [0.0; FINGERS]);
    for k in 0..FINGERS {
        let f;
        (f, a[k]) = finger_fa(cmd[k]);
        fs[k] = f;
        let (lo, hi) = abduction_range(k, f);
        if f < F_MIN - TOL || f > F_MAX + TOL {
            bad.push(format!("{}: flexion f = {f:.2} outside [{F_MIN}, {F_MAX}]", DIGITS[k]));
        }
        if a[k].abs() > M_MAX - f.abs() + TOL || !(lo - TOL <= a[k] && a[k] <= hi + TOL) {
            bad.push(format!("{}: abduction a = {:.2} outside [{lo:.2}, {hi:.2}] at f = {f:.2}", DIGITS[k], a[k]));
        }
    }
    for l in 0..FINGERS - 1 {
        let lim = converge_limit(fs[l], fs[l + 1]);
        if a[l + 1] - a[l] > lim + TOL {
            let d = a[l + 1] - a[l];
            bad.push(format!("{}/{}: converging by {d:.2} > {lim:.2}", DIGITS[l], DIGITS[l + 1]));
        }
    }
    let [m1, m2] = cmd[THUMB];
    if !thumb_command_ok(m1, m2) {
        bad.push(format!("thumb: ({m1:.2}, {m2:.2}) outside the slim-mount envelope"));
    }
    bad
}

/// The thumb's way from t0 to t1: the straight line, clamped into the envelope point by point, as a polyline
/// (xs, ys, cumulative max-norm length).
fn thumb_path(t0: [f64; 2], t1: [f64; 2]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let d = (t1[0] - t0[0]).abs().max((t1[1] - t0[1]).abs());
    let n = ((d / THUMB_PATH_STEP).ceil() as usize).max(1);
    let (mut xs, mut ys) = (vec![t0[0]], vec![t0[1]]);
    for k in 0..=n {
        let [x, y] = clamp_thumb(
            t0[0] + (t1[0] - t0[0]) * k as f64 / n as f64,
            t0[1] + (t1[1] - t0[1]) * k as f64 / n as f64,
        );
        if (x - xs[xs.len() - 1]).abs().max((y - ys[ys.len() - 1]).abs()) > 1e-9 {
            xs.push(x);
            ys.push(y);
        }
    }
    let mut cum = vec![0.0];
    for i in 1..xs.len() {
        cum.push(cum[i - 1] + (xs[i] - xs[i - 1]).abs().max((ys[i] - ys[i - 1]).abs()));
    }
    (xs, ys, cum)
}

/// Frames from start to goal at speed (rad/s), STREAM_HZ apart, every one clamped; the fingers go straight, the
/// thumb along thumb_path(). Where the clamps move a finger faster than the straight line does (a curled neighbour
/// uncurling shrinks converge_limit under it), the move gets more frames (up to 4x), so no motor steps more than
/// speed / STREAM_HZ between frames.
pub fn trajectory(start: &Cmd, goal: &Cmd, speed: f64) -> Result<Vec<Cmd>, String> {
    let v = speed.to_degrees();
    let (xs, ys, cum) = thumb_path(start[THUMB], goal[THUMB]);
    let len = cum[cum.len() - 1];
    let mut d_fingers: f64 = 0.0;
    for k in 0..FINGERS {
        for j in 0..2 {
            d_fingers = d_fingers.max((goal[k][j] - start[k][j]).abs());
        }
    }
    let n0 = ((d_fingers.max(len) / v * STREAM_HZ).ceil() as usize).max(1);
    let mut n = n0;
    let reference = clamp_hand(start)?; // start may be outside the limits (where the hand is at power-on)
    loop {
        let mut frames = vec![];
        for i in 1..=n {
            let s = i as f64 / n as f64;
            let mut cmd = *start;
            for k in 0..FINGERS {
                for j in 0..2 {
                    cmd[k][j] = start[k][j] + (goal[k][j] - start[k][j]) * s;
                }
            }
            cmd[THUMB] = [interp(s * len, &cum, &xs), interp(s * len, &cum, &ys)];
            frames.push(clamp_hand(&cmd)?);
        }
        frames[n - 1] = *goal;
        let mut step: f64 = 0.0;
        let mut prev = &reference;
        for frame in &frames {
            for k in 0..5 {
                for j in 0..2 {
                    step = step.max((frame[k][j] - prev[k][j]).abs());
                }
            }
            prev = frame;
        }
        if step <= v / STREAM_HZ * (1.0 + 1e-9) || n >= 4 * n0 {
            return Ok(frames);
        }
        n = (4 * n0).min((n + 1).max((n as f64 * step / (v / STREAM_HZ)).ceil() as usize));
    }
}

/// One servo of the config.
pub struct Servo {
    pub id: u8,
    pub offset: f64,
    pub invert: bool,
}

impl Servo {
    /// Goal position (rad) of command m (deg), as AHControl writes it.
    fn goal(&self, m: f64) -> f64 {
        let g = m.to_radians() + self.offset;
        if self.invert {
            -g
        } else {
            g
        }
    }

    /// Command (deg) of a present position (rad).
    fn command(&self, pos: f64) -> f64 {
        ((if self.invert { -pos } else { pos }) - self.offset).to_degrees()
    }
}

/// The hand a config asks for, if any. `limits` is the config's top-level `limits` key, `entries` its [[motors]]
/// (finger_name, [(servo, model); 2]); see Hand::new.
pub fn from_config(
    toml_str: &str,
    limits: Option<&str>,
    entries: Vec<(String, [(Servo, String); 2])>,
) -> Result<Option<Hand>, String> {
    // facet_toml drops keys it does not know, so a `limits` below [Fingers] or in a [[motors]] would leave the hand
    // without limits
    let table: toml::Table = toml_str.parse().map_err(|e| format!("{e}"))?;
    let nested = table.get("Fingers").and_then(|t| t.get("limits")).is_some()
        || table.get("motors").and_then(|m| m.as_array()).is_some_and(|m| m.iter().any(|e| e.get("limits").is_some()));
    if nested {
        return Err("`limits` must be a top-level key: put it at the top of the file, above [Fingers]".into());
    }
    limits.map(|l| Hand::new(l, entries)).transpose()
}

/// The hand's digits as configured (a digit left out of the config is not powered).
pub struct Hand {
    digits: [Option<[Servo; 2]>; 5],
}

impl Hand {
    /// The hand of a config whose `limits` is `limits`. The config must hold the index, middle, ring and pinky, each
    /// once, on SCS0009 servos; the thumb (r_finger4) may be left out. Any other entry must be all STS3215 (the wrist,
    /// Demo/AHSimulation/wrist/control/ahcontrol_wrist.patch), which these limits do not cover.
    pub fn new(limits: &str, entries: Vec<(String, [(Servo, String); 2])>) -> Result<Hand, String> {
        if limits != HAND {
            return Err(format!("limits = {limits:?}: unknown hand, the only one is {HAND:?}"));
        }
        let mut digits: [Option<[Servo; 2]>; 5] = Default::default();
        for (name, [(s1, model1), (s2, model2)]) in entries {
            match FINGER_NAMES.iter().position(|n| *n == name) {
                Some(k) => {
                    if model1 != "SCS0009" || model2 != "SCS0009" {
                        return Err(format!("{name}: the {} must be on SCS0009 servos", DIGITS[k]));
                    }
                    if digits[k].is_some() {
                        return Err(format!("{name} is in the config twice"));
                    }
                    digits[k] = Some([s1, s2]);
                }
                None if model1 == "STS3215" && model2 == "STS3215" => {}
                None => {
                    return Err(format!(
                        "{name}: not a digit of {HAND} ({}), and not the STS3215 wrist",
                        FINGER_NAMES.join(", ")
                    ))
                }
            }
        }
        for k in 0..FINGERS {
            if digits[k].is_none() {
                return Err(format!("{} ({}) is missing: the limits need all four fingers", FINGER_NAMES[k], DIGITS[k]));
            }
        }
        Ok(Hand { digits })
    }

    fn configured(&self) -> impl Iterator<Item = (usize, &[Servo; 2])> {
        self.digits.iter().enumerate().filter_map(|(k, d)| d.as_ref().map(|s| (k, s)))
    }

    /// The start-up goal (rad) of servo `id`: its home, not command 0. None if it is not a digit's.
    pub fn home_goal(&self, id: u8) -> Option<f64> {
        for (k, servos) in self.configured() {
            for (j, s) in servos.iter().enumerate() {
                if s.id == id {
                    return Some(s.goal(HOME[k][j]));
                }
            }
        }
        None
    }

    /// The sim's joint buffer (rad) with every digit's command clamped (the rest unchanged). `indices` gives a
    /// finger_name's two indices from the event's metadata. Refused (Err) when an event lacks a configured digit or
    /// holds a non-finite command; the caller then writes nothing.
    pub fn clamp_buffer(&self, buffer: &[f64], indices: impl Fn(&str) -> Option<Vec<i64>>) -> Result<Vec<f64>, String> {
        let mut cmd = HOME;
        let mut at = [[0usize; 2]; 5];
        for (k, _) in self.configured() {
            let name = FINGER_NAMES[k];
            let idx = indices(name).ok_or_else(|| format!("no {name} in the event"))?;
            for j in 0..2 {
                let i = *idx.get(j).ok_or_else(|| format!("{name}: indices {idx:?}"))?;
                if i < 0 || i as usize >= buffer.len() {
                    return Err(format!("{name}: index {i} outside the buffer ({} values)", buffer.len()));
                }
                at[k][j] = i as usize;
                cmd[k][j] = buffer[i as usize].to_degrees();
            }
        }
        let out = clamp_hand(&cmd)?;
        let bad = check_hand(&out);
        if !bad.is_empty() {
            return Err(bad.join("; "));
        }
        let mut clamped = buffer.to_vec();
        for (k, _) in self.configured() {
            for j in 0..2 {
                if out[k][j] != cmd[k][j] {
                    clamped[at[k][j]] = out[k][j].to_radians();
                }
            }
        }
        Ok(clamped)
    }

    /// From where the hand is to home, streamed at 50 Hz at HOME_SPEED, every frame clamped (the thumb follows the
    /// edge of its envelope instead of cutting through the palm shell). The torque goes on at the first frame.
    pub fn go_home(&self, c: &mut Scs0009Controller) -> Result<(), Box<dyn Error>> {
        let mut here = HOME;
        let mut ids = vec![];
        for (k, servos) in self.configured() {
            for (j, s) in servos.iter().enumerate() {
                let pos = c.read_present_position(s.id)?;
                thread::sleep(Duration::from_millis(10));
                here[k][j] = s.command(pos[0]);
                ids.push(s.id);
            }
        }
        let frames = trajectory(&here, &HOME, HOME_SPEED)?;
        println!("limits {HAND}: homing in {:.2} s ({} frames)", frames.len() as f64 / STREAM_HZ, frames.len());
        let t0 = Instant::now();
        for (i, frame) in frames.iter().enumerate() {
            let bad = check_hand(frame);
            if !bad.is_empty() {
                return Err(format!("refused: {}", bad.join("; ")).into());
            }
            let goals: Vec<f64> = self
                .configured()
                .flat_map(|(k, servos)| [servos[0].goal(frame[k][0]), servos[1].goal(frame[k][1])])
                .collect();
            c.sync_write_goal_position(&ids, &goals)?;
            if i == 0 {
                c.sync_write_torque_enable(&ids, &vec![1u8; ids.len()])?;
            }
            let due = t0 + Duration::from_secs_f64((i + 1) as f64 / STREAM_HZ);
            thread::sleep(due.saturating_duration_since(Instant::now()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// xorshift64*, for reproducible sweeps without a crate
    struct Rng(u64);
    impl Rng {
        fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            let u = (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64 / (1u64 << 53) as f64;
            lo + (hi - lo) * u
        }
    }

    #[test]
    fn converge_limits() {
        for (fl, fr, want) in [(-56.0, -56.0, 15.0), (60.0, -56.0, 15.0), (-56.0, 75.0, 22.5), (90.0, -60.0, 30.0)] {
            assert!((converge_limit(fl, fr) - want).abs() < 1e-12 && converge_limit(fr, fl) == converge_limit(fl, fr));
        }
    }

    #[test]
    fn home_is_inside() {
        assert_eq!(clamp_thumb(0.0, 0.0), HOME[THUMB]);
        assert_eq!(clamp_hand(&HOME).unwrap(), HOME);
        assert!(check_hand(&HOME).is_empty());
    }

    #[test]
    fn clamps_land_inside_and_stay() {
        let mut rng = Rng(0x5eed);
        let mut inside = 0;
        for i in 0..20000 {
            let r = if i % 2 == 0 { 150.0 } else { 90.0 };
            let mut cmd = HOME;
            for d in cmd.iter_mut() {
                *d = [rng.uniform(-r, r), rng.uniform(-r, r)];
            }
            let out = clamp_hand(&cmd).unwrap();
            assert!(check_hand(&out).is_empty(), "{cmd:?} -> {out:?}: {:?}", check_hand(&out));
            assert_eq!(clamp_hand(&out).unwrap(), out, "not idempotent at {out:?}");
            if check_hand(&cmd).is_empty() {
                assert_eq!(out, cmd);
                inside += 1;
            }
        }
        assert!(inside > 0);
        let mut cmd = HOME;
        cmd[2][0] = f64::NAN;
        assert!(clamp_hand(&cmd).is_err());
    }

    #[test]
    fn trajectory_stays_inside() {
        let mut rng = Rng(7);
        for _ in 0..200 {
            let mut ends = [HOME; 2];
            for e in ends.iter_mut() {
                for d in e.iter_mut() {
                    *d = [rng.uniform(-120.0, 120.0), rng.uniform(-120.0, 120.0)];
                }
                *e = clamp_hand(e).unwrap();
            }
            let frames = trajectory(&ends[0], &ends[1], 1.5).unwrap();
            assert!(frames.iter().all(|f| check_hand(f).is_empty()));
            assert_eq!(*frames.last().unwrap(), ends[1]);
        }
    }

    fn entries(names: &[&str], model: &str) -> Vec<(String, [(Servo, String); 2])> {
        let servo = |id| (Servo { id, offset: 0.1, invert: false }, model.to_string());
        let id = |name: &str| 2 * name[name.len() - 1..].parse::<u8>().unwrap();
        names.iter().map(|n| (n.to_string(), [servo(id(n) - 1), servo(id(n))])).collect()
    }

    #[test]
    fn config() {
        let hand = ["r_finger1", "r_finger2", "r_finger3", "r_finger4", "r_finger5"];
        let top = "limits = \"AH_Right_5F_3S\"\n[Fingers]\n[[motors]]\nfinger_name = \"r_finger1\"\n";
        assert!(from_config(top, None, entries(&hand, "SCS0009")).unwrap().is_none());
        assert!(from_config(top, Some(HAND), entries(&hand, "SCS0009")).unwrap().is_some());
        assert!(from_config(top, Some(HAND), entries(&hand[..3], "SCS0009")).is_err()); // no pinky
        assert!(from_config(top, Some(HAND), entries(&hand[..4], "SCS0009")).is_err()); // no pinky
        assert!(from_config(top, Some(HAND), entries(&[hand[0], hand[1], hand[2], hand[4]], "SCS0009")).is_ok());
        assert!(from_config(top, Some("AH_Right_5F"), entries(&hand, "SCS0009")).is_err());
        assert!(from_config(top, Some(HAND), entries(&hand, "STS3215")).is_err());
        let mut e = entries(&hand, "SCS0009");
        e.extend(entries(&["l_finger1"], "SCS0009"));
        assert!(from_config(top, Some(HAND), e).is_err());
        let mut e = entries(&hand, "SCS0009");
        e.extend(entries(&["r_wrist1"], "STS3215"));
        assert!(from_config(top, Some(HAND), e).is_ok());
        let mut e = entries(&hand, "SCS0009");
        e.extend(entries(&["r_finger2"], "SCS0009"));
        assert!(from_config(top, Some(HAND), e).is_err());
        // below [Fingers] or in a [[motors]], facet_toml would drop it
        let nested = "[Fingers]\nlimits = \"AH_Right_5F_3S\"\n[[motors]]\nfinger_name = \"r_finger1\"\n";
        assert!(from_config(nested, None, entries(&hand, "SCS0009")).is_err());
        let nested = "[Fingers]\n[[motors]]\nfinger_name = \"r_finger1\"\nlimits = \"AH_Right_5F_3S\"\n";
        assert!(from_config(nested, None, entries(&hand, "SCS0009")).is_err());
    }

    #[test]
    fn buffer() {
        let names = ["r_finger1", "r_finger2", "r_finger3", "r_finger4", "r_finger5"];
        let hand = Hand::new(HAND, entries(&names, "SCS0009")).unwrap();
        assert_eq!(hand.home_goal(8), Some(22.5f64.to_radians() + 0.1));
        assert_eq!(hand.home_goal(1), Some(0.1));
        assert_eq!(hand.home_goal(21), None);
        // Victory, as the sim sends it (rad, in r_finger1..5 order), plus a value that is no digit's
        let victory = [-33.5, 83.5, -83.5, 33.5, 90.0, -90.0, 48.0, 41.0, 90.0, -90.0, 12.0];
        let buffer: Vec<f64> = victory.iter().map(|v: &f64| v.to_radians()).collect();
        let at = |name: &str| names.iter().position(|n| *n == name).map(|k| vec![2 * k as i64, 2 * k as i64 + 1]);
        // the middle converges onto the fully curled ring by 25 deg: allowed (converge_limit 30)
        assert_eq!(hand.clamp_buffer(&buffer, at).unwrap(), buffer);
        // with the ring at f = 0 the limit is 15: the middle's abduction is cut from -25 to -15
        let mut ring_mid = buffer.clone();
        ring_mid[4..6].copy_from_slice(&[0.0, 0.0]);
        let out = hand.clamp_buffer(&ring_mid, at).unwrap();
        for i in [0, 1, 4, 5, 6, 7, 8, 9, 10] {
            assert_eq!(out[i], ring_mid[i], "{i}");
        }
        assert!((out[2].to_degrees() + 73.5).abs() < 1e-9 && (out[3].to_degrees() - 43.5).abs() < 1e-9);
        // refused: a digit missing from the event, a non-finite command, an index outside the buffer
        assert!(hand.clamp_buffer(&buffer, |n| if n == "r_finger4" { None } else { at(n) }).is_err());
        let mut nan = buffer.clone();
        nan[6] = f64::NAN;
        assert!(hand.clamp_buffer(&nan, at).is_err());
        assert!(hand.clamp_buffer(&buffer[..9], at).is_err());
        // a thumb inside the palm shell is moved out
        let mut thumb_in = buffer.clone();
        thumb_in[6..8].copy_from_slice(&[0.0, 0.0]);
        let out = hand.clamp_buffer(&thumb_in, at).unwrap();
        assert_eq!((out[6], out[7]), (0.0, 22.5f64.to_radians()));
    }

    /// Outputs of PythonExample/AmazingHand_Demo_5F.py's clamp_hand on the hand_poses.json poses and a few random
    /// commands (exact: the port does the same float operations).
    #[test]
    fn same_as_python() {
        for (cmd, want) in PYTHON_CASES {
            assert_eq!(clamp_hand(&cmd).unwrap(), want, "clamp_hand({cmd:?})");
        }
    }

    #[rustfmt::skip]
    const PYTHON_CASES: [(Cmd, Cmd); 36] = [
        (
            [[-53.1, 53.1], [-53.1, 53.1], [-53.1, 53.1], [-53.1, 53.1], [-52.5, 45.0]],
            [[-53.1, 53.1], [-53.1, 53.1], [-53.1, 53.1], [-53.1, 53.1], [-52.5, 45.0]],
        ),
        (
            [[90.0, -90.0], [90.0, -90.0], [90.0, -90.0], [90.0, -90.0], [-19.5, 45.0]],
            [[90.0, -90.0], [90.0, -90.0], [90.0, -90.0], [90.0, -90.0], [-19.5, 45.0]],
        ),
        (
            [[-30.0, 90.0], [-49.7, 49.7], [-85.0, 35.0], [-90.0, 30.0], [-36.0, 71.0]],
            [[-30.0, 90.0], [-49.7, 49.7], [-85.0, 35.0], [-90.0, 30.0], [-36.0, 71.0]],
        ),
        (
            [[-52.4, 42.4], [-48.1, 58.1], [-38.1, 68.1], [-45.1, 61.1], [-50.0, 37.0]],
            [[-52.4, 42.4], [-48.1, 58.1], [-38.1, 68.1], [-45.1, 61.1], [-50.0, 37.0]],
        ),
        (
            [[-58.5, 58.5], [90.0, -90.0], [90.0, -90.0], [90.0, -90.0], [-1.0, 23.5]],
            [[-58.5, 58.5], [90.0, -90.0], [90.0, -90.0], [90.0, -90.0], [-1.0, 23.5]],
        ),
        (
            [[35.0, -55.0], [-13.0, 13.0], [-35.9, 35.9], [-35.9, 35.9], [-6.0, 67.5]],
            [[35.0, -55.0], [-13.0, 13.0], [-35.9, 35.9], [-35.9, 35.9], [-6.0, 67.5]],
        ),
        (
            [[-33.5, 83.5], [-83.5, 33.5], [90.0, -90.0], [90.0, -90.0], [48.0, 41.0]],
            [[-33.5, 83.5], [-83.5, 33.5], [90.0, -90.0], [90.0, -90.0], [48.0, 41.0]],
        ),
        (
            [[0.0, 0.0], [90.0, -90.0], [90.0, -90.0], [90.0, -90.0], [-7.5, 42.0]],
            [[0.0, 0.0], [90.0, -90.0], [90.0, -90.0], [90.0, -90.0], [-7.5, 42.0]],
        ),
        (
            [[90.0, -90.0], [-56.0, 56.0], [90.0, -90.0], [90.0, -90.0], [-19.5, 45.0]],
            [[90.0, -90.0], [-56.0, 56.0], [90.0, -90.0], [90.0, -90.0], [-19.5, 45.0]],
        ),
        (
            [[-26.0, 86.0], [-86.0, 26.0], [90.0, -90.0], [90.0, -90.0], [48.0, 41.0]],
            [[-26.0, 86.0], [-86.0, 26.0], [90.0, -90.0], [90.0, -90.0], [48.0, 41.0]],
        ),
        (
            [[90.0, -90.0], [-56.0, 56.0], [90.0, -90.0], [90.0, -90.0], [-19.5, 45.0]],
            [[90.0, -90.0], [-56.0, 56.0], [90.0, -90.0], [90.0, -90.0], [-19.5, 45.0]],
        ),
        (
            [[-96.31955589736914, 41.973949714546364], [-9.819479656954467, -38.84984186758559], [-43.524799707104634, 87.1554737559795], [121.54315100315218, -96.79404245308541], [45.835440805539584, -60.50916979332922]],
            [[-82.90942424571875, 37.090575754281254], [6.605756859596816, -22.424605351034305], [-52.909424245718746, 67.09057575428126], [90.0, -90.0], [45.835440805539584, 38.335440805539584]],
        ),
        (
            [[84.05319602923029, 75.57302889671007], [24.456744746037245, 45.49177747423559], [2.727665338023315, 58.661146507602595], [-9.291501827981426, -29.013758403522864], [-39.97814120997137, -49.26004976818738]],
            [[84.05319602923029, 75.57302889671007], [19.482483635900827, 40.51751636409917], [2.0332594152103596, 57.96674058478964], [-9.291501827981426, -29.013758403522864], [-39.97814120997137, 30.0]],
        ),
        (
            [[75.0, 7.5], [-82.5, -15.0], [-45.0, 30.0], [67.5, -97.5], [-7.5, -7.5]],
            [[75.0, 7.5], [-56.25, 11.25], [-45.0, 30.0], [75.0, -90.0], [-7.5, 22.5]],
        ),
        (
            [[-40.44575966028259, -91.3807200385338], [28.459761301815846, -19.40605321480663], [-60.00254887645505, -87.17516362913413], [112.38721715765385, 89.2386946564298], [32.0129030054937, -46.469826399812355]],
            [[12.434146789187398, -38.50081358906381], [25.899573858373035, -21.966240658249443], [6.0863073763395406, -21.08630737633954], [19.074261250612025, -4.074261250612025], [32.0129030054937, 30.0]],
        ),
        (
            [[80.42756237633671, 11.407927410458198], [-12.102706868408632, 72.08092782075997], [-32.51847529766477, 35.27907662408101], [-33.51234109277338, -42.920451663876], [36.15136011015398, -48.97944379667562]],
            [[80.42756237633671, 11.407927410458198], [-12.102706868408632, 72.08092782075997], [-32.51847529766477, 35.27907662408101], [-33.51234109277338, -42.920451663876], [36.15136011015398, 30.0]],
        ),
        (
            [[7.5, 0.0], [-7.5, 15.0], [22.5, -60.0], [-67.5, 45.0], [97.5, 7.5]],
            [[7.5, 0.0], [-7.5, 15.0], [22.5, -60.0], [-67.5, 45.0], [90.0, 82.5]],
        ),
        (
            [[36.4509778877813, -38.35670496095103], [-23.94202202380985, -1.5510966248831304], [-9.008726278149048, 52.69176685488692], [23.15325134885976, -25.11880710226011], [-149.45929391795428, 88.20931823265127]],
            [[36.4509778877813, -38.35670496095103], [-16.72333218531097, 5.667593213615753], [-21.37811605236559, 40.32237708067038], [23.15325134885976, -25.11880710226011], [-67.5, 30.0]],
        ),
        (
            [[3.489159805980404, -31.2219605139714], [-0.008095965985090992, -73.17973197096875], [72.84673305960351, 88.15251016521061], [-79.42766084719507, -25.518380653551347], [41.41162400743503, -33.43713638078112]],
            [[3.489159805980404, -31.2219605139714], [29.085818002491827, -44.08581800249183], [-0.15288855280355307, 15.152888552803553], [-79.42766084719507, -25.518380653551347], [41.41162400743503, 33.91162400743503]],
        ),
        (
            [[-22.5, 15.0], [22.5, -15.0], [-60.0, 52.5], [7.5, 90.0], [90.0, 75.0]],
            [[-22.5, 15.0], [22.5, -15.0], [-57.916666666666664, 54.583333333333336], [-27.916666666666668, 54.58333333333333], [90.0, 82.5]],
        ),
        (
            [[36.28063956001485, -101.91770619957906], [134.11862499349883, -142.9102713984672], [-60.687163073741566, -65.51595206982125], [51.54779451681141, -3.791080178246564], [-122.13343537276583, -146.1426336765209]],
            [[48.198345759593906, -90.0], [90.0, -90.0], [-5.937753259289286, -10.766542255368973], [34.317289590199856, -21.02158510485812], [-67.5, 30.0]],
        ),
        (
            [[18.776834419254815, -1.5330001786332992], [18.248781841272205, 11.524295495796208], [70.2826394939424, 75.31143441670741], [-56.113711001574295, 79.58175650469616], [51.79079273810104, 25.02472540013395]],
            [[18.776834419254815, -1.5330001786332992], [18.248781841272205, 11.524295495796208], [27.3721412071517, 32.400936129916715], [-48.26597724843907, 71.73402275156093], [51.79079273810104, 44.29079273810104]],
        ),
        (
            [[-82.5, 30.0], [-90.0, 7.5], [60.0, 82.5], [97.5, -52.5], [52.5, 22.5]],
            [[-78.75, 33.75], [-56.25, 41.25], [-3.75, 18.75], [90.0, -60.0], [52.5, 45.0]],
        ),
        (
            [[94.27291856149516, -109.57156451366102], [34.6524823076601, -29.14995734112782], [79.81090243988486, -129.68569181383782], [32.044314310261015, 107.12360699599405], [38.50435870892718, -53.59869136394377]],
            [[90.0, -90.0], [34.6524823076601, -29.14995734112782], [90.0, -90.0], [-7.539646342866519, 67.53964634286652], [38.50435870892718, 31.004358708927178]],
        ),
        (
            [[28.04329006594005, -29.132488307518063], [30.34670191381621, -66.89800774043037], [-39.60148604473932, -84.11069192455524], [-74.89630998671565, 9.858531566971173], [-48.2943201493847, 2.894865774791733]],
            [[28.04329006594005, -29.132488307518063], [30.34670191381621, -66.89800774043037], [-7.7453970600920385, -52.25460293990796], [-74.89630998671565, 9.858531566971173], [-48.2943201493847, 30.0]],
        ),
        (
            [[97.5, 30.0], [97.5, 75.0], [52.5, -30.0], [52.5, -37.5], [75.0, -37.5]],
            [[90.0, 22.5], [41.25, 18.75], [52.5, -30.0], [52.5, -37.5], [75.0, 67.5]],
        ),
        (
            [[-90.44154693165059, 27.715508112703215], [32.30394021865138, 28.14800902440676], [-91.46274331379306, 29.66557772734506], [94.21316791568356, -89.06084610866631], [-14.981766256866125, 91.53105107144549]],
            [[-66.5785275221769, 51.578527522176906], [9.577965597122308, 5.422034402877692], [-90.0, 30.0], [90.0, -90.0], [-14.981766256866125, 90.0]],
        ),
        (
            [[-89.81815292143521, 32.0217469654301], [-86.46713847814767, 32.66448859524043], [104.46289116500392, -45.41914619565558], [86.72730462459933, 87.14703721076074], [57.83220184310201, 34.702080386828754]],
            [[-88.89820297800256, 31.10179702199744], [-73.97196371105059, 45.159663362337504], [83.00537784613807, -66.87665951452142], [29.790133706919292, 30.209866293080708], [57.83220184310201, 50.33220184310201]],
        ),
        (
            [[-88.12041789247685, 28.909160210938694], [-88.42252286190175, 35.34543989385901], [-30.617536856223822, 92.90135521883532], [-89.66151286108536, 25.041419123955556], [91.5363943047539, 26.170681021985164]],
            [[-80.55561663253286, 36.47396147088269], [-67.04082758082508, 52.95917241917492], [-52.04082758082508, 67.95917241917492], [-89.66151286108536, 25.041419123955556], [90.0, 82.5]],
        ),
        (
            [[-45.14305654922439, 66.30197988003485], [-24.232644577083192, -24.863728015717854], [-83.48335811533362, 33.971108428562886], [88.6344524900644, -34.821439573770235], [75.79206494213555, 1.4508813424466616]],
            [[-45.14305654922439, 66.30197988003485], [-22.28628742877563, -22.91737086741029], [-66.32906242004121, 51.12540412385529], [69.99008989978302, -53.465802164051624], [75.79206494213555, 68.29206494213555]],
        ),
        (
            [[-93.07495317952139, 29.40219823554389], [14.139943899689646, 16.606232657500357], [88.14983833104259, -28.934510723280045], [-87.04419600272658, -86.18797729141772], [-35.077356486725506, 94.2393443044423]],
            [[-69.9177732235672, 50.082226776432805], [3.849082397527451, 6.315371155338162], [78.62440130359411, -38.45994775072851], [-87.04419600272658, -86.18797729141772], [-35.077356486725506, 74.76793053982348]],
        ),
        (
            [[-83.40214337327481, 33.55364434473111], [20.67399031649225, -103.24347619614116], [-29.58251512659083, 90.72001562828225], [90.01241563622507, -63.731292017582156], [17.21580065338309, 4.387767711625959]],
            [[-82.17731394488611, 34.77847377311981], [54.23867979859191, -69.67878671404151], [-51.74068682956644, 68.25931317043356], [90.0, -63.743707653807235], [17.21580065338309, 22.5]],
        ),
        (
            [[-21.265077697615062, 86.89190945870538], [-85.13599940963623, -85.27893089909087], [-88.40895280932047, -89.09805626288416], [-23.610304554932256, 90.99463918559496], [-43.88737947854386, 79.98008045643289]],
            [[-21.265077697615062, 86.89190945870538], [-22.42853425527268, -22.57146574472732], [-7.1554482732181555, -7.8445517267818445], [-49.80247187026361, 64.80247187026362], [-43.88737947854386, 54.72524104291227]],
        ),
        (
            [[73.04230442074729, -43.32041920231262], [108.4217997107701, -38.57263375404139], [43.77278569261524, -105.30214011058712], [93.92581600876689, -32.40774064369978], [48.79348512137201, 7.321506351965035]],
            [[73.04230442074729, -43.32041920231262], [90.0, -56.99443346481149], [66.39643183399242, -82.67849396920992], [77.29447870942516, -49.039077943041505], [48.79348512137201, 41.29348512137201]],
        ),
        (
            [[26.84985498900415, -95.80044147948504], [-89.14161547451755, 24.445740094296546], [-27.514717979995947, 85.3939133368049], [-3.0599856952783946, -7.140024418962998], [-59.10441268831934, 79.33846474971796]],
            [[38.02763906962908, -84.6226573988601], [-64.42861283190027, 49.15874273691384], [-49.08925070589364, 63.81938061090721], [-3.0599856952783946, -7.140024418962998], [-59.10441268831934, 38.39558731168066]],
        ),
        (
            [[-34.13846794038005, -29.72997071757346], [90.53542849728224, -97.51182133371746], [84.74458486880599, -92.8512289990118], [-25.374875262725695, 89.71546720011841], [-12.173662668281578, 56.89081306254842]],
            [[-32.204248611403294, -27.795751388596706], [90.0, -90.0], [87.66528169501531, -89.93053217280247], [-29.278843003361185, 85.81149945948292], [-12.173662668281578, 56.89081306254842]],
        ),
    ];
}
