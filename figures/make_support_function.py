"""
Fig. 2 of main_v8: the half-ellipsoidal bases and their support functions h_K/b.
Top row: the bases (curved side down, cut face up).  Bottom row: h_K/b over the tilt disk.
The support function is a property of the base surface alone; the platform does not change it.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import cm, gridspec
from matplotlib.colors import Normalize
from boardgeom import h_hat

os.makedirs("output", exist_ok=True)
C_OB, C_SP, C_PR, C_TRI = "#1f6fb4", "#8a8a8a", "#e07b39", "#3fa66a"

def draw_base(ax, r1, r2, x0=0.0, color=C_OB, alpha=0.55, label=None):
    a1, a2 = 1.0 / r1, 1.0 / r2
    Uu, V = np.meshgrid(np.linspace(np.pi / 2, np.pi, 40), np.linspace(0, 2 * np.pi, 70))
    ax.plot_surface(a1 * np.sin(Uu) * np.cos(V) + x0, a2 * np.sin(Uu) * np.sin(V), np.cos(Uu) + 1.0,
                    color=color, alpha=alpha, linewidth=0, antialiased=True)
    rr, tt = np.meshgrid(np.linspace(0, 1, 12), np.linspace(0, 2 * np.pi, 70))
    ax.plot_surface(a1 * rr * np.cos(tt) + x0, a2 * rr * np.sin(tt), np.ones_like(rr),
                    color=color, alpha=min(1, alpha + 0.15), linewidth=0)
    if label:
        ax.text(x0, 0, 1.32, label, ha="center", fontsize=9)

TH, PH = np.meshgrid(np.deg2rad(np.linspace(0, 45, 70)), np.linspace(0, 2 * np.pi, 160))
X, Y = np.rad2deg(TH) * np.cos(PH), np.rad2deg(TH) * np.sin(PH)
ring = np.linspace(0, 2 * np.pi, 200)
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
fig = plt.figure(figsize=(14, 12.4))
gs = gridspec.GridSpec(2, 2, height_ratios=[0.82, 1.0], hspace=0.06, wspace=0.14)

def panel(ax, letter):
    ax.text2D(0.02, 0.95, "(%s)" % letter, transform=ax.transAxes, fontsize=13, fontweight="bold")

ax = fig.add_subplot(gs[0, 0], projection="3d")
draw_base(ax, 0.75, 0.75, x0=-2.7, color=C_OB, label=r"oblate $\rho=0.75$")
draw_base(ax, 1.0, 1.0, x0=0.0, color=C_SP, label=r"sphere $\rho=1$")
draw_base(ax, 1.25, 1.25, x0=2.2, color=C_PR, label=r"prolate $\rho=1.25$")
panel(ax, "a")
ax.set_xlim(-4.3, 3.1); ax.set_ylim(-2.2, 2.2); ax.set_zlim(0, 1.4)
ax.set_box_aspect((7.4, 4.4, 1.9)); ax.view_init(elev=18, azim=-62)
ax.set_zticks([0, 0.5, 1.0]); ax.set_xlabel("AP"); ax.set_ylabel("ML")

ax = fig.add_subplot(gs[0, 1], projection="3d")
draw_base(ax, 1.78, 0.75, color=C_TRI, alpha=0.6)
ax.text(1.15, 0, 1.05, "AP\n(narrow)", ha="center", fontsize=8.5, color="#c0504d")
ax.text(0, 1.75, 1.05, "ML\n(wide)", ha="center", fontsize=8.5, color="#1f6fb4")
panel(ax, "b")
ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.7, 1.7); ax.set_zlim(0, 1.4)
ax.set_box_aspect((3.2, 3.4, 1.9)); ax.view_init(elev=18, azim=-62)
ax.set_zticks([0, 0.5, 1.0]); ax.set_xlabel("AP"); ax.set_ylabel("ML")

ax = fig.add_subplot(gs[1, 0], projection="3d")
ax.plot_surface(X, Y, h_hat(TH, PH, 0.75, 0.75), color=C_OB, alpha=0.55, linewidth=0)
ax.plot_surface(X, Y, np.ones_like(TH), color=C_SP, alpha=0.30, linewidth=0)
ax.plot_surface(X, Y, h_hat(TH, PH, 1.25, 1.25), color=C_PR, alpha=0.55, linewidth=0)
ax.plot(20 * np.cos(ring), 20 * np.sin(ring), np.ones_like(ring), "k--", lw=1.2)
panel(ax, "c")
ax.set_zlim(0.85, 1.25); ax.view_init(elev=24, azim=-58); ax.set_box_aspect((1, 1, 0.55))
ax.set_xlabel("tilt toward AP (deg)"); ax.set_ylabel("tilt toward ML (deg)"); ax.set_zlabel(r"$h_K/b$")
ax.legend(handles=[Patch(facecolor=C_OB, alpha=.55, label="oblate: base centre rises with tilt"),
                   Patch(facecolor=C_SP, alpha=.45, label="sphere: base centre stays level"),
                   Patch(facecolor=C_PR, alpha=.55, label="prolate: base centre falls with tilt")],
          loc="upper center", bbox_to_anchor=(0.5, -0.03), fontsize=8.5, frameon=False)

ax = fig.add_subplot(gs[1, 1], projection="3d")
Z = h_hat(TH, PH, 1.78, 0.75)
norm = Normalize(Z.min(), Z.max())
fc = cm.viridis(norm(Z)); fc[..., 3] = 0.80
ax.plot_surface(X, Y, Z, facecolors=fc, linewidth=0, antialiased=True, shade=False)
ax.plot_surface(X, Y, np.ones_like(TH), color=C_SP, alpha=0.18, linewidth=0)
ax.plot(20 * np.cos(ring), 20 * np.sin(ring), np.ones_like(ring), "k--", lw=1.2)
ax.text(58, 0, 0.83, "AP: falls", fontsize=9, color="#c0504d", ha="center", zorder=10)
ax.text(0, 58, 1.20, "ML: rises", fontsize=9, color="#1f6fb4", ha="center", zorder=10)
panel(ax, "d")
ax.set_zlim(0.85, 1.25); ax.view_init(elev=24, azim=-58); ax.set_box_aspect((1, 1, 0.55))
ax.set_xlabel("tilt toward AP (deg)"); ax.set_ylabel("tilt toward ML (deg)"); ax.set_zlabel("")
m = cm.ScalarMappable(norm=norm, cmap="viridis"); m.set_array([])
fig.colorbar(m, ax=ax, shrink=0.5, pad=0.08, label=r"$h_K/b$")
fig.savefig("output/support_function_full.png", dpi=200, bbox_inches="tight")
fig.savefig("output/support_function_full.pdf", bbox_inches="tight")
