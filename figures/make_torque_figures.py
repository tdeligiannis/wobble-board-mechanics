"""
Figs. 3 and 4 of main_v8: holding-torque field and torque along a circular sway, for the
unloaded board WITH its platform (uniform disk, Eq. platform), exact support function.
Torques are per unit board weight and divided by b: tau/(M g b).  Fig. 4b also shows the
base alone (mu = 0) as thin dashed lines.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm
from boardgeom import board, grad_U, mu_est

os.makedirs("output", exist_ok=True)
TMAX, A, KAPPA = np.deg2rad(20.0), np.deg2rad(12.0), 0.35
FIELD = [("oblate, $\\rho=0.75$", 0.75, 0.75), ("prolate, $\\rho=1.78$", 1.78, 1.78),
         ("prolate, $\\rho=1.25$", 1.25, 1.25),
         ("triaxial, $\\rho_{AP}{=}1.78,\\ \\rho_{ML}{=}0.75$", 1.78, 0.75)]
TRAJ = [("oblate $\\rho=0.75$", 0.75, 0.75, "#1f6fb4"), ("sphere $\\rho=1$", 1.0, 1.0, "#8a8a8a"),
        ("prolate $\\rho=1.25$", 1.25, 1.25, "#e07b39"), ("prolate $\\rho=1.78$", 1.78, 1.78, "#a32020"),
        ("triaxial $1.78/0.75$", 1.78, 0.75, "#3fa66a")]
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})

# ------------------------------------------------------------------ Fig. 3
q = np.linspace(-TMAX, TMAX, 15)
QX, QY = np.meshgrid(q, q)
mask = np.hypot(QX, QY) <= TMAX + 1e-9
P = [(r1, r2, mu_est(r1, r2)) for _, r1, r2 in FIELD]
H0 = [board(*p)[0] for p in P]
fields = [grad_U(QX, QY, r1, r2, h) for (r1, r2, _), h in zip(P, H0)]
def radial(gx, gy):
    th = np.hypot(QX, QY)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(th > 0, (gx * QX + gy * QY) / th, 0.0)
rads = [radial(*f) for f in fields]
norm = TwoSlopeNorm(vmin=min(np.nanmin(r[mask]) for r in rads), vcenter=0.0,
                    vmax=max(np.nanmax(r[mask]) for r in rads))
mags = [np.nanmax(np.hypot(*f)[mask]) for f in fields]
BASE = max(mags) * 14                 # one arrow scale for (a), (b), (d)
ZOOM = [1, 1, 4, 1]                   # (c) drawn four times longer
KEY = [0.20, 0.20, 0.05, 0.20]

fig = plt.figure(figsize=(12.5, 11))
gs = gridspec.GridSpec(2, 2, wspace=0.22, hspace=0.28)
for k, (lab, r1, r2) in enumerate(FIELD):
    ax = fig.add_subplot(gs[k // 2, k % 2])
    gx, gy = fields[k]
    qv = ax.quiver(np.rad2deg(QX)[mask], np.rad2deg(QY)[mask], gx[mask], gy[mask], rads[k][mask],
                   cmap="RdBu", norm=norm, angles="uv", scale=BASE / ZOOM[k], width=0.006, pivot="tail")
    ax.quiverkey(qv, 0.83, 0.05, KEY[k], r"$%.2f\,Mgb$" % KEY[k], labelpos="N",
                 coordinates="axes", color="0.25", fontproperties={"size": 8.5})
    if r1 != r2:
        R1, R2 = 1 / r1 ** 2, 1 / r2 ** 2
        phc = np.arccos(np.sqrt((H0[k] - R2) / (R1 - R2)))
        for base in (0.0, np.pi):
            for sgn in (1, -1):
                ang = base + sgn * phc
                ax.plot([0, 20 * np.cos(ang)], [0, 20 * np.sin(ang)], ls=":", lw=1.0, color="0.35")
        print("triaxial neutral directions (small tilt): +-%.1f deg from AP" % np.rad2deg(phc))
    ax.add_patch(plt.Circle((0, 0), 20, fill=False, ls="--", color="0.6", lw=1))
    ax.set_xlim(-23, 23); ax.set_ylim(-23, 23); ax.set_aspect("equal")
    ax.text(0.0, 1.02, "(%s)" % "abcd"[k], transform=ax.transAxes, fontsize=12, fontweight="bold")
    ax.text(0.09, 1.02, lab, transform=ax.transAxes, fontsize=10.5)
    ax.set_xlabel("tilt toward AP (deg)"); ax.set_ylabel("tilt toward ML (deg)")
    print("(%s) mu=%.3f h0=%.4f stiffness AP %+.3f ML %+.3f max|tau| %.3f" %
          ("abcd"[k], P[k][2], H0[k], 1 / r1 ** 2 - H0[k], 1 / r2 ** 2 - H0[k], mags[k]))
cax = fig.add_axes([0.33, 0.045, 0.36, 0.015])
cb = fig.colorbar(qv, cax=cax, orientation="horizontal")
cb.set_label(r"radial holding torque $\tau_r/(Mgb)$   (blue: outward;  red: inward)")
fig.savefig("output/torque_field.png", dpi=200, bbox_inches="tight")
fig.savefig("output/torque_field.pdf", bbox_inches="tight")

# ------------------------------------------------------------------ Fig. 4
ph = np.linspace(0, 2 * np.pi, 721)
tx, ty = A * np.cos(ph), A * np.sin(ph)
def dyn(r1, r2, mu, sx, sy):
    h0, l1, l2 = board(r1, r2, mu)
    gx, gy = grad_U(sx, sy, r1, r2, h0)
    return gx - KAPPA * l1 * sx, gy - KAPPA * l2 * sy

fig2 = plt.figure(figsize=(13.5, 5.8))
gs2 = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.2], wspace=0.28)
axA = fig2.add_subplot(gs2[0])
axA.plot(np.rad2deg(tx), np.rad2deg(ty), color="0.6", lw=1.2)
smp = np.linspace(0, 2 * np.pi, 13)[:-1]
for lab, r, col in [("oblate $\\rho=0.75$ (stable)", 0.75, "#1f6fb4"),
                    ("prolate $\\rho=1.78$ (unstable)", 1.78, "#a32020")]:
    sx, sy = A * np.cos(smp), A * np.sin(smp)
    ux, uy = dyn(r, r, mu_est(r, r), sx, sy)
    axA.quiver(np.rad2deg(sx), np.rad2deg(sy), ux, uy, color=col, angles="uv", scale=1.6,
               width=0.008, pivot="tail", label=lab, zorder=3)
axA.set_aspect("equal"); axA.set_xlim(-21, 21); axA.set_ylim(-21, 21)
axA.set_xlabel("tilt toward AP (deg)"); axA.set_ylabel("tilt toward ML (deg)")
axA.text(0.0, 1.02, "(a)", transform=axA.transAxes, fontsize=12, fontweight="bold")
axA.legend(loc="upper right", fontsize=8.5, frameon=False)

axB = fig2.add_subplot(gs2[1])
for lab, r1, r2, col in TRAJ:
    for mu, style in ((mu_est(r1, r2), dict(lw=2.2, label=lab)), (0.0, dict(lw=1.0, ls="--", alpha=0.8))):
        ux, uy = dyn(r1, r2, mu, tx, ty)
        rad = (ux * tx + uy * ty) / A
        axB.plot(np.rad2deg(ph), rad, color=col, **style)
        h0, l1, l2 = board(r1, r2, mu)
        print("%-22s mu=%.2f radial min %+.3f max %+.3f | per unit tilt AP %+.3f ML %+.3f" %
              (lab, mu, rad.min(), rad.max(), (1/r1**2 - h0) - KAPPA*l1, (1/r2**2 - h0) - KAPPA*l2))
axB.plot([], [], color="0.3", lw=2.2, label="with platform")
axB.plot([], [], color="0.3", lw=1.0, ls="--", label="base only")
axB.axhline(0, color="0.3", lw=1.0)
for x in (90, 180, 270):
    axB.axvline(x, color="0.88", lw=0.7)
axB.set_xlim(0, 360); axB.set_xticks([0, 90, 180, 270, 360])
axB.set_xticklabels(["0\nAP", "90\nML", "180\nAP", "270\nML", "360"])
axB.set_xlabel("sway phase (deg): direction of tilt around the circle")
axB.set_ylabel(r"radial torque $\tau_r/(Mgb)$  (+ outward, $-$ inward)")
axB.text(0.0, 1.02, "(b)", transform=axB.transAxes, fontsize=12, fontweight="bold")
axB.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=8.5, frameon=False)
fig2.savefig("output/torque_trajectory.png", dpi=200, bbox_inches="tight")
fig2.savefig("output/torque_trajectory.pdf", bbox_inches="tight")
