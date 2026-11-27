"""Axial spin of a driven spherical board (Sec. 2.5, Eq. (yaw_phase)).

A spherical base (hemisphere, R = b = 150 mm, M = 3 kg) starts at a tilt of 10 deg with
omega_3 = 0, at rest or circling, and is driven for 6 s by a pure couple.  A couple along the
horizontal axis about which the board is tilted has no component about the board's axis or
about the vertical and leaves omega_3 at zero.  A couple normal to the board's axis that has a
vertical component generates axial spin.  The last column checks that the change in total
energy equals the work done by the couple.  Run from this folder: python torque_spin.py"""
import numpy as np
from scipy.integrate import solve_ivp
from board3d import Board, qrot, initial, G_ACC

Z = np.array([0, 0, 1.0])
def frame(y):
    q = y[0:4]/np.linalg.norm(y[0:4]); R = qrot(q); return R, R[:, 2]

# --- torque laws (world frame). T0 in N m, Om in rad/s
def tau_tiltaxis(T0, Om):       # along k = z x e3: axis-normal AND horizontal (no vertical part)
    def f(t, y, bd):
        R, e3 = frame(y); k = np.cross(Z, e3); n = np.linalg.norm(k)
        return T0*np.sin(Om*t)*k/n if n > 1e-9 else np.zeros(3)
    return f
def tau_inplane_azim(T0, Om):   # along e3 x k: axis-normal, in the vertical plane of tilt -> has vertical part
    def f(t, y, bd):
        R, e3 = frame(y); k = np.cross(Z, e3); n = np.linalg.norm(k)
        if n < 1e-9: return np.zeros(3)
        return T0*np.sin(Om*t)*np.cross(e3, k/n)
    return f
def tau_bodyrot(T0, Om):        # in the platform plane, direction rotating in the body frame
    def f(t, y, bd):
        R, e3 = frame(y); return R @ (T0*np.array([np.cos(Om*t), np.sin(Om*t), 0.0]))
    return f
def tau_bodyfixed(T0, Om):      # in the platform plane, fixed body direction, oscillating
    def f(t, y, bd):
        R, e3 = frame(y); return R @ (T0*np.sin(Om*t)*np.array([1.0, 0, 0]))
    return f
def tau_vert_proj(T0, Om):      # vertical couple with its axial part removed
    def f(t, y, bd):
        R, e3 = frame(y); return T0*np.sin(Om*t)*(Z - (Z@e3)*e3)
    return f
def tau_worldrot(T0, Om):       # horizontal in the world, rotating: HAS an axial component when tilted
    def f(t, y, bd):
        return T0*np.array([np.cos(Om*t), np.sin(Om*t), 0.0])
    return f

def run(bd, y0, tf, T, name):
    # augment state with W = int tau.omega dt and S = int omega_3 dt
    def rhs(t, y):
        d = bd.rhs(t, y[:9], T)
        R, e3 = frame(y); w = y[4:7]
        tau = T(t, y, bd) if T else np.zeros(3)
        return np.concatenate([d, [tau@w, w@e3]])
    s = solve_ivp(rhs, (0, tf), np.concatenate([y0, [0, 0]]), method="DOP853",
                  rtol=1e-10, atol=1e-12, max_step=2e-3, dense_output=True)
    t = np.linspace(0, tf, 6001); Y = s.sol(t)
    w3 = np.array([Y[4:7, i] @ frame(Y[:, i])[1] for i in range(len(t))])
    th = np.degrees([bd.tilt(Y[0:4, i]/np.linalg.norm(Y[0:4, i])) for i in range(len(t))])
    # max axial and vertical torque components actually applied
    ax = vt = tm = 0.0
    if T:
        for i in range(0, len(t), 10):
            tau = T(t[i], Y[:, i], bd); e3 = frame(Y[:, i])[1]
            ax = max(ax, abs(tau@e3)); vt = max(vt, abs(tau[2])); tm = max(tm, np.linalg.norm(tau))
    E0 = bd.energy(Y[:9, 0]); E1 = bd.energy(Y[:9, -1])
    pw = abs((E1-E0) - Y[9, -1])
    print(f"{name:34s} |tau|max={tm:5.3f} axial={ax:8.1e} vert={vt:8.1e} | "
          f"tilt max={th.max():5.1f} deg | max|w3|={np.abs(w3).max():9.2e} rad/s  "
          f"int w3 dt={np.degrees(Y[10,-1]):9.3e} deg | power-balance err={pw:8.1e} J")
    return np.abs(w3).max()

bd = Board(1.0, 1.0)             # sphere, R = b = 0.15, M = 3
f0 = np.sqrt(G_ACC*(bd.b - bd.h0)/(bd.b**2/5 + 19*bd.b**2/320 + bd.h0**2))/(2*np.pi)
print(f"sphere: natural frequency {f0:.3f} Hz, holding torque at 10 deg "
      f"{bd.M*G_ACC*(bd.b-bd.h0)*np.sin(np.radians(10)):.3f} N m\n")
tf = 6.0
circ = initial(np.radians(10), 0.0, (0, 0, 0))
# circling release: tilt 10 deg, precession of the axis at 1 rad/s about vertical with omega_3 = 0
def circling(theta, rate):
    y = initial(theta, 0.0); R, e3 = frame(y)
    w = rate*Z; w = w - (w@e3)*e3      # remove axial part so that omega_3 = 0
    y[4:7] = w; return y
yc = circling(np.radians(10), 1.0)

for T0, Om in [(0.10, 2*np.pi*0.5), (0.10, 2*np.pi*1.3), (0.25, 2*np.pi*0.5)]:
    print(f"--- T0 = {T0} N m, drive {Om/2/np.pi:.1f} Hz")
    print(" start: at rest, tilted 10 deg")
    run(bd, circ, tf, None,                    "  no torque")
    run(bd, circ, tf, tau_tiltaxis(T0, Om),    "  tilt axis (normal, horizontal)")
    run(bd, circ, tf, tau_inplane_azim(T0, Om),"  e3 x k (normal, has vertical)")
    run(bd, circ, tf, tau_bodyfixed(T0, Om),   "  platform plane, body-fixed dir")
    run(bd, circ, tf, tau_bodyrot(T0, Om),     "  platform plane, rotating dir")
    run(bd, circ, tf, tau_vert_proj(T0, Om),   "  vertical minus axial part")
    run(bd, circ, tf, tau_worldrot(T0, Om),    "  world-horizontal rotating")
    print(" start: circling, tilt 10 deg, omega_3 = 0")
    run(bd, yc, tf, None,                      "  no torque")
    run(bd, yc, tf, tau_tiltaxis(T0, Om),      "  tilt axis (normal, horizontal)")
    run(bd, yc, tf, tau_inplane_azim(T0, Om),  "  e3 x k (normal, has vertical)")
    run(bd, yc, tf, tau_bodyrot(T0, Om),       "  platform plane, rotating dir")
    run(bd, yc, tf, tau_vert_proj(T0, Om),     "  vertical minus axial part")
