"""Planar kinematics of the 3-phalanx finger (flexion plane of the finger design frame, (y, z) in mm).

Loop 1 (kept from the stock finger, re-dimensioned): crossed four-bar
    ground = gimbal (MCP pivot O_M, Link pivot O_L, 6 mm apart, stock),
    input  = Link (stock base with the two rod balls; arm to P4),
    coupler= P2 (middle phalanx: pivots P4 and PIP),
    output = P1 (proximal: MCP -> PIP).
Loop 2 (new): crossed four-bar on P2
    ground = P2 (PIP -> DIP), input = P1 (crank PIP -> A), output = P3 (crank DIP -> B),
    coupler L2 = A -> B (two-force member).

All points are given in the straight pose (MCP = PIP = DIP = 0), where every part's CAD frame is
the finger frame. psi is the Link-base flexion relative to the gimbal measured exactly like in the
stock model (0 = stock straight pose); psi_s is the base angle at which the new finger is straight
(the arm of the new Link is drawn rotated accordingly).
Flexion (positive) rotates towards -z: R(t) = [[cos t, sin t], [-sin t, cos t]] acting on (y, z).
"""

import numpy as np

O_M = np.array([0.0, 0.0])
O_L = np.array([0.0, -6.0])


def R(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, s], [-s, c]])


def ang(v):
    return np.arctan2(v[..., 1], v[..., 0])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rot(v, t):
    """Flexion-rotate vectors v (..., 2) by angles t (...)."""
    c, s = np.cos(t), np.sin(t)
    return np.stack([c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1]], -1)


def circ_int(c1, r1, c2, r2, sign):
    """Intersection of circle(c1, r1) and circle(c2, r2) on the side `sign` of c1->c2 (vectorised).
    Returns points and a validity mask (False where the circles do not meet)."""
    d = c2 - c1
    dist = np.linalg.norm(d, axis=-1)
    a = (r1 ** 2 - r2 ** 2 + dist ** 2) / (2 * dist)
    h2 = r1 ** 2 - a ** 2
    ok = h2 >= 0
    h = np.sqrt(np.clip(h2, 0, None))
    u = d / dist[..., None]
    perp = np.stack([-u[..., 1], u[..., 0]], -1)
    return c1 + a[..., None] * u + sign * h[..., None] * perp, ok


def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def seg_dist(p, a, b):
    """Distance from points p to segments a-b (vectorised)."""
    ab = b - a
    t = np.clip(np.sum((p - a) * ab, -1) / np.maximum(np.sum(ab * ab, -1), 1e-12), 0, 1)
    return np.linalg.norm(a + t[..., None] * ab - p, axis=-1)


def vec_angle(u, v):
    """Unsigned angle between vectors (deg, 0..180)."""
    c = np.sum(u * v, -1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


class Design:
    def __init__(self, L1, L2, L3t, P4_0, A0, B0, psi_s, tip_z=-4.436):
        self.L1, self.L2, self.L3t = L1, L2, L3t
        self.PIP0 = np.array([L1, 0.0])
        self.DIP0 = np.array([L1 + L2, 0.0])
        self.TIP0 = np.array([L1 + L2 + L3t, tip_z])
        self.P4_0, self.A0, self.B0 = np.asarray(P4_0, float), np.asarray(A0, float), np.asarray(B0, float)
        self.psi_s = psi_s
        self.c = np.linalg.norm(self.P4_0 - self.PIP0)
        self.rA = np.linalg.norm(self.A0 - self.PIP0)
        self.rB = np.linalg.norm(self.B0 - self.DIP0)
        self.dAB = np.linalg.norm(self.B0 - self.A0)
        self.L_link = np.linalg.norm(self.P4_0 - O_L)
        # branch signs of the straight pose. loop 1: PIP = intersection of circle(O_M, L1) and circle(P4, c); side relative to O_M -> P4
        self.sg1 = np.sign(cross(self.P4_0 - O_M, self.PIP0 - O_M))
        self.sg2 = np.sign(cross(self.A0 - self.DIP0, self.B0 - self.DIP0))

    @classmethod
    def from_json(cls, d):
        g = d["geometry"]
        return cls(g["L1"], g["L2"], g["L3_tip"], g["P4"], g["A"], g["B"], np.radians(d["drive"]["psi_straight_deg"]),
                   g["tip"][1])

    def solve(self, psi):
        """Vectorised solution for Link-base angles psi (rad). Returns a dict of arrays (angles in rad,
        points (..., 2) in mm, each in the frame stated) and 'ok' (assembles on the straight-pose branch)."""
        psi = np.atleast_1d(np.asarray(psi, float))
        P4 = O_L + rot(np.broadcast_to(self.P4_0 - O_L, psi.shape + (2,)), psi - self.psi_s)
        PIP, ok1 = circ_int(np.broadcast_to(O_M, P4.shape), self.L1, P4, self.c, self.sg1)
        th1 = -wrap(ang(PIP - O_M) - ang(self.PIP0 - O_M))
        th12 = -wrap(ang(P4 - PIP) - ang(self.P4_0 - self.PIP0))
        th2 = wrap(th12 - th1)
        # loop 2 in the P2 frame (straight-pose coordinates)
        A = self.PIP0 + rot(np.broadcast_to(self.A0 - self.PIP0, psi.shape + (2,)), -th2)
        B, ok2 = circ_int(np.broadcast_to(self.DIP0, A.shape), self.rB, A, self.dAB, self.sg2)
        th3 = -wrap(ang(B - self.DIP0) - ang(self.B0 - self.DIP0))
        phi_l2_p2 = -wrap(ang(B - A) - ang(self.B0 - self.A0))
        def to_g(p_p2):  # P2-frame straight-pose coords -> gimbal (finger) frame
            return PIP + rot(p_p2 - self.PIP0, th12)

        out = dict(psi=psi, mcp=th1, pip=th2, dip=th3, l2=wrap(th2 + phi_l2_p2), psi_rel=psi - self.psi_s,
                   P4=P4, PIP=PIP, A_p2=A, B_p2=B, A=to_g(A), B=to_g(B),
                   DIP=to_g(np.broadcast_to(self.DIP0, A.shape)))
        out["TIP"] = out["DIP"] + rot(np.broadcast_to(self.TIP0 - self.DIP0, psi.shape + (2,)), th12 + th3)
        # transmission angles (deg; 90 is ideal, 0/180 is a dead point)
        out["mu_out1"] = vec_angle(PIP - O_M, P4 - PIP)      # P1 vs P2 crank (output side of loop 1)
        out["mu_in1"] = vec_angle(P4 - O_L, PIP - P4)        # Link vs P2 crank (input side of loop 1)
        out["mu_out2"] = vec_angle(B - A, B - self.DIP0)     # L2 vs P3 crank
        out["mu_in2"] = vec_angle(A - self.PIP0, B - A)      # P1 crank vs L2
        out["ok"] = ok1 & ok2
        return out

    def solve_psi(self, psi):
        s = self.solve(np.array([psi]))
        if not s["ok"][0]:
            return None
        return {k: (v[0] if np.ndim(v) else v) for k, v in s.items()}

    def link_arm_in_link_frame(self):
        """P4 in the new Link's own CAD frame (= stock Link frame): the straight-pose P4 rotated back
        by -psi_s about the Link pivot."""
        return O_L + rot(self.P4_0 - O_L, -self.psi_s)


def check_branch(dz, sol):
    """True where the solution lies on the same assembly branch as the straight pose."""
    s1 = np.sign(cross(sol["P4"] - O_M, sol["PIP"] - O_M)) == dz.sg1
    s2 = np.sign(cross(sol["A_p2"] - dz.DIP0, sol["B_p2"] - dz.DIP0)) == dz.sg2
    return s1 & s2
