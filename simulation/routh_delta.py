"""Routh's third integral for a spherical board (Sec. 2.5, Eq. (yaw_phase)).

For a body of revolution whose surface of contact is spherical, rolling on a rough plane,
Routh (Advanced Rigid Dynamics, 6th ed., Art. 243, Eq. (delta)) gives the integral

    omega_3^2 { C + a^2 sin^2(theta) + (h - a cos(theta))^2 C/A } = D,

with a the radius of the sphere, h the distance from its centre to the centre of gravity, and
A, C the transverse and axial moments of inertia about the centre of gravity per unit mass.
A zero axial spin omega_3 therefore stays zero, and a non-zero one varies with the tilt theta.

The script releases a hemispherical board (R = b = 150 mm) while circling, with omega_3 = 0 and
with omega_3 = 1 rad/s (the runs of spin_check.py), and reports the range of omega_3, the
relative variation of D, and the accumulated axial spin.
Expected: omega_3 = 0 stays zero; with 1 rad/s, omega_3 between 1.000 and 1.003 rad/s,
D constant to about 1e-10, accumulated spin 2295.5 deg against 2291.8 deg at a constant rate.
Run from this folder: python routh_delta.py"""
import numpy as np
from board3d import Board, qrot, initial, simulate

trapz = getattr(np, "trapezoid", None) or np.trapz

B = Board(1.0, 1.0, b=0.15)
a = B.b                      # radius of the sphere
h = 3*a/8                    # centre of sphere to centre of gravity
A = a*a/5 + 19*a*a/320       # transverse moment of inertia about G, per unit mass
C = 2*a*a/5                  # axial moment of inertia, per unit mass

for spin in (0.0, 1.0):
    y0 = initial(np.radians(20), 0.0); n = qrot(y0[:4])[:, 2]
    w = np.array([1.0, 0, 0]); w = w - (w @ n)*n; w = 0.25*w/np.linalg.norm(w)
    y0[4:7] = w + spin*n
    T = 40.0; s = simulate(B, y0, T, max_step=5e-3)
    ts = np.linspace(0, T, int(T*2000) + 1); Y = s.sol(ts)
    th = np.array([B.tilt(Y[:4, i]) for i in range(len(ts))])
    w3 = np.array([Y[4:7, i] @ qrot(Y[:4, i]/np.linalg.norm(Y[:4, i]))[:, 2] for i in range(len(ts))])
    D = w3**2 * (C + a*a*np.sin(th)**2 + (h - a*np.cos(th))**2 * C/A)
    print(f"initial axial spin {spin:.1f} rad/s: tilt {np.degrees(th.min()):.2f} to {np.degrees(th.max()):.2f} deg")
    if spin == 0.0:
        print(f"   max |omega_3| = {np.abs(w3).max():.2e} rad/s (stays zero)")
    else:
        print(f"   omega_3 from {w3.min():.5f} to {w3.max():.5f} rad/s")
        print(f"   relative variation of Routh's integral D = {(D.max() - D.min())/D.max():.2e}")
        print(f"   accumulated axial spin = {np.degrees(trapz(w3, ts)):.2f} deg "
              f"(constant rate: {np.degrees(spin*T):.2f} deg)")
