"""
Shared geometry for the figures of main_v8: solid half-ellipsoidal base (cut at its centre)
carrying a uniform disk platform.  All lengths are in units of the vertical semi-axis b.

Platform: diameter w, thickness t, resting on the cut face (centroid at 1 + T/2 above the lowest
point), mass ratio mu = m_p/m_b.  Paper: Eq. (platform).
"""
import numpy as np

B_M, W_M, T_M = 0.15, 0.45, 0.02          # b, platform diameter, platform thickness (m)
W, T = W_M / B_M, T_M / B_M
H0B = 0.625                               # centroid height of the bare half-ellipsoid / b

def mu_est(r1, r2):
    """Mass ratio for a platform and a solid base of the same material."""
    v_plat = np.pi * (W_M / 2) ** 2 * T_M
    v_base = (2 * np.pi / 3) * (B_M / r1) * (B_M / r2) * B_M
    return v_plat / v_base

def board(r1, r2, mu):
    """Return h0/b and the inertia per unit mass about the contact point, /b^2, for AP and ML."""
    zp, jp = 1 + T / 2, W ** 2 / 16 + T ** 2 / 12
    h0 = (H0B + mu * zp) / (1 + mu)
    def ell(r):
        jb = (1 / r ** 2) / 5 + 19 / 320
        j = (jb + (H0B - h0) ** 2 + mu * ((zp - h0) ** 2 + jp)) / (1 + mu)
        return j + h0 ** 2
    return h0, ell(r1), ell(r2)

def h_hat(theta, phi, r1, r2):
    """Support function h_K / b."""
    return np.sqrt(np.sin(theta) ** 2 * (np.cos(phi) ** 2 / r1 ** 2 + np.sin(phi) ** 2 / r2 ** 2)
                   + np.cos(theta) ** 2)

def U(tx, ty, r1, r2, h0):
    """Centroid height / b over the tilt plane: potential per unit weight, /b."""
    th = np.hypot(tx, ty); ph = np.arctan2(ty, tx)
    return h_hat(th, ph, r1, r2) - (1 - h0) * np.cos(th)

def grad_U(tx, ty, r1, r2, h0, e=1e-5):
    return ((U(tx + e, ty, r1, r2, h0) - U(tx - e, ty, r1, r2, h0)) / (2 * e),
            (U(tx, ty + e, r1, r2, h0) - U(tx, ty - e, r1, r2, h0)) / (2 * e))
