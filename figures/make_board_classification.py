"""
Fig. 1 of main_v8: kinematic classification of wobble boards (schematic, drawn to the rolling
kinematics).  (a) rocker board on a cylindrical base, side view; (b) omnidirectional board on a
spherical base, side view and tilt directions; (c) path dependence on a spherical base: the
axis is tilted forward, swung through 90 degrees at constant tilt, and brought back to level.
The board ends level again, but displaced on the floor and turned about the vertical.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Arc

os.makedirs("output", exist_ok=True)
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
C_BASE, C_PLAT, C_ACC = "#8fb4d9", "#c9a36b", "#2e8b57"

def side_view(ax, theta, R=1.0, k=0.55, w=2.6, t=0.12):
    """Board on a circular-arc base of radius R, cut at clearance k, tilted by theta (rolls to +x)."""
    cx, cy = R * theta, R                                   # centre after rolling without slip
    rot = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    half = np.arccos((R - k) / R)
    s = np.linspace(-half, half, 60)
    arc = np.stack([R * np.sin(s), -R * np.cos(s)])          # body frame, centre at origin
    plat = np.array([[-w/2, w/2, w/2, -w/2], [k - R, k - R, k - R + t, k - R + t]])
    for pts, col in ((arc, C_BASE), (plat, C_PLAT)):
        p = rot @ pts
        ax.fill(p[0] + cx, p[1] + cy, color=col, ec="0.25", lw=1.0, zorder=3)
    ax.plot([-1.9, 2.6], [0, 0], color="0.2", lw=1.5)
    ax.plot([cx], [0], "o", color="k", ms=4, zorder=5)
    ax.plot([cx, cx], [0, cy], ls=":", color="0.4", lw=1)
    axis = rot @ np.array([[0, 0], [k - R, k - R + 0.9]])
    ax.plot(axis[0] + cx, axis[1] + cy, color="0.3", lw=1, ls="--")
    ax.plot([cx, cx], [cy + (k - R), cy + (k - R) + 0.9], color="0.6", lw=1, ls="--")
    ax.add_patch(Arc((cx, cy + k - R), 1.1, 1.1, angle=90, theta1=-np.degrees(theta), theta2=0, color="0.3"))
    ax.text(cx + 0.30, cy + k - R + 0.68, r"$\theta$", fontsize=11)
    ax.set_aspect("equal"); ax.set_xlim(-1.9, 2.6); ax.set_ylim(-0.55, 2.25); ax.axis("off")
    return cx

fig = plt.figure(figsize=(13.5, 4.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.08)
th = np.deg2rad(16)

ax = fig.add_subplot(gs[0])
cx = side_view(ax, th)
ax.annotate("", xy=(cx, -0.22), xytext=(0, -0.22), arrowprops=dict(arrowstyle="<->", color=C_ACC, lw=1.6))
ax.text(cx / 2, -0.47, r"$x = R\,\theta$", ha="center", color=C_ACC, fontsize=11)
ax.text(-1.85, 2.08, "(a)", fontsize=13, fontweight="bold")
ax.text(-1.3, 2.08, "cylindrical base: one tilt axis", fontsize=10)
ax.text(-1.3, 1.84, "holonomic: position follows from tilt", fontsize=9, color="0.3")

ax = fig.add_subplot(gs[1])
side_view(ax, th)
ax.text(-1.85, 2.08, "(b)", fontsize=13, fontweight="bold")
ax.text(-1.3, 2.08, "spherical base: tilts in every direction", fontsize=10)
ax.text(-1.3, 1.84, "non-holonomic: position depends on the path", fontsize=9, color="0.3")
ins = ax.inset_axes([0.0, 0.0, 0.26, 0.34]); ins.set_aspect("equal"); ins.axis("off")
ins.add_patch(plt.Circle((0, 0), 1, fc=C_PLAT, ec="0.25", lw=1))
ins.add_patch(plt.Circle((0, 0), 0.45, fc="none", ec="0.35", ls="--", lw=1))
for a in np.arange(0, 360, 45):
    d = np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])
    ins.add_patch(FancyArrowPatch(0.12 * d, 0.9 * d, arrowstyle="-|>", mutation_scale=9, color=C_ACC, lw=1.2))
ins.set_xlim(-1.15, 1.15); ins.set_ylim(-1.15, 1.3); ins.text(0, 1.1, "top view", ha="center", fontsize=8)

# ---- (c) path dependence, computed from the rolling constraint of a sphere with no axial spin
ax = fig.add_subplot(gs[2])
T0 = np.deg2rad(40)                                           # exaggerated tilt, for visibility
s = np.linspace(0, 1, 200)
# leg 1: tilt toward +x to T0;  leg 2: swing tilt direction 0 -> 90 deg at tilt T0;  leg 3: back to level
x1, y1 = T0 * s, 0 * s                                        # centre path / R
ph = s * np.pi / 2
x2 = T0 - np.sin(T0) * np.cos(T0) * (1 - np.cos(ph)); y2 = np.sin(T0) * np.cos(T0) * np.sin(ph)
x3 = x2[-1] + 0 * s; y3 = y2[-1] - T0 * s
for (x, y), lab in (((x1, y1), "1  tilt forward"), ((x2, y2), "2  swing 90$^\\circ$"), ((x3, y3), "3  return to level")):
    ax.plot(x, y, color=C_ACC, lw=2)
    m = len(x) // 2
    ax.annotate("", xy=(x[m + 3], y[m + 3]), xytext=(x[m], y[m]), arrowprops=dict(arrowstyle="-|>", color=C_ACC, lw=2))
ax.text(0.48, 0.03, "1  tilt forward", fontsize=9, color=C_ACC, ha="center", va="bottom")
ax.text(0.60, 0.33, "2  swing the tilt 90$^\\circ$", fontsize=9, color=C_ACC, ha="left")
ax.text(0.17, 0.30, "3  return\n    to level", fontsize=9, color=C_ACC, ha="right")
yaw = (np.pi / 2) * (1 - np.cos(T0))
for (px, py, ang, a_) in ((0, 0, 0, 0.35), (x3[-1], y3[-1], yaw, 0.9)):
    ax.add_patch(plt.Circle((px, py), 0.055, fc=C_PLAT, ec="0.25", lw=1, alpha=a_, zorder=4))
    ax.plot([px, px + 0.11 * np.cos(ang)], [py, py + 0.11 * np.sin(ang)], color="k", lw=1.6, zorder=5)
ax.text(-0.07, 0.0, "start,\nlevel", ha="right", va="center", fontsize=9)
ax.text(x3[-1] + 0.14, y3[-1] - 0.06, "end, level again:\ndisplaced and turned", ha="left", va="center", fontsize=9)
ax.set_aspect("equal"); ax.set_xlim(-0.45, 1.05); ax.set_ylim(-0.40, 0.80); ax.axis("off")
ax.text(-0.44, 0.70, "(c)", fontsize=13, fontweight="bold")
ax.text(-0.27, 0.70, "floor track of the base centre, top view", fontsize=10)
print("panel (c): tilt %.0f deg, net displacement (%.3f, %.3f) R, yaw %.1f deg" %
      (np.degrees(T0), x3[-1], y3[-1], np.degrees(yaw)))
fig.savefig("output/board_classification.png", dpi=200, bbox_inches="tight")
fig.savefig("output/board_classification.pdf", bbox_inches="tight")
