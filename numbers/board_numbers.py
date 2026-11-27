"""
Numbers quoted in the paper that do not need the 3D simulation.

  - critical aspect ratio of a half-ellipsoid with a disk platform (Sec. 2.4, Eq. critical_rho)
  - Table 1: signed stiffness (q0 - h0)/b and natural frequency or divergence rate
  - edge-contact angles and the cut needed for touchdown near 20 deg (Sec. 2.5)
  - the laboratory boards in the light-base limit (Sec. 2.4, "The Laboratory Boards")
  - boards S and U of the two-board comparison (Sec. 3.3)
  - the loaded-board estimate (Sec. 4.1)

Solid bases are homogeneous half-ellipsoids (cut at the centre); the platform is a uniform disk
of diameter w and thickness t resting on the cut face, mass ratio mu = m_p / m_b (Eq. platform).
"""
import numpy as np
from scipy.optimize import brentq

G = 9.81
W, T = 0.45, 0.02            # reference platform: diameter, thickness (m)

def solid_board(a, b, mu, w=W, t=T):
    """h0 and (j^2 + h0^2) for a solid half-ellipsoid (semi-axes a, a, b) with a disk platform."""
    h0b, jb = 0.625 * b, a * a / 5 + 19 * b * b / 320
    zp, jp = b + t / 2, w * w / 16 + t * t / 12
    h0 = (h0b + mu * zp) / (1 + mu)
    j = (jb + (h0b - h0) ** 2 + mu * ((zp - h0) ** 2 + jp)) / (1 + mu)
    return h0, j + h0 * h0

def mu_same_material(a1, a2, b, w=W, t=T):
    return (np.pi * (w / 2) ** 2 * t) / (2 * np.pi / 3 * a1 * a2 * b)

def rate(s, ell):
    w2 = G * s / ell
    return ("%.2f Hz" % (np.sqrt(w2) / 2 / np.pi)) if w2 > 0 else ("%.2f 1/s" % np.sqrt(-w2))

def theta_max(a, b, k, w=W):
    f = lambda th: np.sqrt(a * a * np.sin(th) ** 2 + b * b * np.cos(th) ** 2) - (b - k) * np.cos(th) - w / 2 * np.sin(th)
    return np.degrees(brentq(f, 1e-3, 1.4))

b = 0.15
print("== critical aspect ratio, half-ellipsoid, b = 150 mm, platform 45 cm x 2 cm")
for mu in (0, 0.1, 0.25, 0.5):
    print("  mu = %.2f: rho_c = %.3f" % (mu, np.sqrt(b / solid_board(b, b, mu)[0])))
print("  platform mass that makes rho = 1.25 neutral: mu = %.3f" % (0.015 / (1 + T / 2 / b - 0.64)))

print("\n== Table 1: (q0 - h0)/b ; f0 or sigma")
RHO = (0.75, 1.0, 1.25, 1.78)
own = [mu_same_material(b / r, b / r, b) for r in RHO]
for label, mus in [("mu = 0", [0] * 4), ("mu = 0.25", [0.25] * 4), ("mu = 0.50", [0.5] * 4),
                   ("mu = 0.70", [0.7] * 4), ("estimated mu", own)]:
    cells = []
    for r, mu in zip(RHO, mus):
        a = b / r; h0, ell = solid_board(a, b, mu); s = a * a / b - h0
        cells.append("%+.3f; %s" % (s / b, rate(s, ell)))
    print("  %-13s" % label, " | ".join(cells))
print("  estimated mu:", ", ".join("%.2f" % m for m in own))

print("\n== edge contact with the 45-cm platform, half-bases b = 150 mm")
for r in (0.75, 1.0, 1.25):
    print("  rho = %.2f: theta_max = %.1f deg" % (r, theta_max(b / r, b, b)))
k20 = brentq(lambda k: theta_max(b, b, k) - 20.0, 0.02, 0.149)
print("  sphere R = 150 mm touches down at 20 deg for k = %.1f mm (lambda = %.2f)" % (k20 * 1e3, k20 / b - 1))

print("\n== laboratory boards: b = 140 mm, k = 75 mm, platform 45 cm x 2.5 cm, light-base limit")
bl, kl, tl = 0.140, 0.075, 0.025
zp, jp = kl + tl / 2, W * W / 16 + tl * tl / 12
print("  z_p = %.1f mm, rho_c -> %.3f, lambda = %.2f" % (zp * 1e3, np.sqrt(bl / zp), kl / bl - 1))
for r in (0.75, 1.0, 1.25):
    q0 = bl / r ** 2; s = q0 - zp
    print("  rho = %.2f: q0 - z_p = %+.1f mm, %s, theta_max = %.1f deg" %
          (r, s * 1e3, rate(s, jp + zp * zp), theta_max(bl / r, bl, kl)))
for nu in (0.1, 0.2):
    h0 = zp - nu * (zp - kl / 2)
    print("  shell with %.0f%% of the board mass, centroid at mid-height: margin of rho = 1.25 is %+.1f mm" %
          (100 * nu, (bl / 1.25 ** 2 - h0) * 1e3))

print("\n== boards S (hemisphere R = 130 mm) and U (a = 90, b = 160 mm)")
res = {}
for name, a, bb in (("S", 0.130, 0.130), ("U", 0.090, 0.160)):
    for tag, mu in (("base only", 0.0), ("with platform", mu_same_material(a, a, bb))):
        h0, ell = solid_board(a, bb, mu); s = a * a / bb - h0; res[name, tag] = (s, ell)
        print("  %s %-14s mu = %.2f: q0 - h0 = %+.1f mm, %s" % (name, tag, mu, s * 1e3, rate(s, ell)))
for tag in ("base only", "with platform"):
    dk = res["U", tag][0] - res["S", tag][0]; dl = res["U", tag][1] - res["S", tag][1]
    print("  %s: stiffness differential %+.1f mm/rad, inertia differential %+.0f mm^2" % (tag, dk * 1e3, dl * 1e6))
    if tag == "base only":
        for f in (0.3, 0.8):
            print("    at %.1f Hz: %+.1f mm/rad" % (f, (dk - dl * (2 * np.pi * f) ** 2 / G) * 1e3))
        lever = abs(dk) * np.radians(7)
        print("    at 7 deg: lever %.1f mm; for M = 3 kg: %.2f N m; CoP shift under 70 kg: %.2f mm" %
              (lever * 1e3, 3 * G * lever, 3 * lever / 70 * 1e3))

print("\n== loaded board: 70 kg rider, centre of mass 1.1 m above the platform, 3 kg board, b = 150 mm")
print("  combined centroid height: %.2f m; largest q0 of the bases: %.2f m" %
      ((70 * (b + 1.1) + 3 * 0.1) / 73, b / 0.75 ** 2))
