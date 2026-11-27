"""
Full 3D dynamics of a wobble board: rigid solid half-ellipsoid (semi-axes a1, a2, b, cut at its
centre) rolling without slip on a horizontal floor.  Non-holonomic, all three rotations free.

Body frame: origin at the ellipsoid centre O, z along the symmetry axis (platform face at z = 0,
curved base below).  Centre of mass G = (0, 0, -3b/8).
World frame: z up, floor z = 0.

Unknowns eliminated: the contact force F.  With rho = r_contact - r_G (world):
   no slip:       v_G = -omega x rho
   translation:   M a_G = F - M g z
   rotation:      d(I omega)/dt = rho x F + tau_h
=> [I + M(|rho|^2 1 - rho rho^T)] alpha = -omega x I omega - M rho x (omega x rho_dot)
                                          + M g rho x z + tau_h
rho_dot is evaluated by a centred finite difference of rho over a small rotation exp(omega dt).

Yaw: swing-twist decomposition q = q_swing (x) q_twist about the body z axis,
psi = 2 atan2(q_z, q_w).  At zero tilt the board's orientation is a pure rotation by psi
about the vertical, so psi is exactly the yaw when the board is level.
"""
import numpy as np
from scipy.integrate import solve_ivp

G_ACC = 9.81

def qmul(p, q):
    w1, x1, y1, z1 = p; w2, x2, y2, z2 = q
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])

def qrot(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
                     [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]])

def qaxis(axis, ang):
    axis = np.asarray(axis, float); axis = axis / np.linalg.norm(axis)
    return np.concatenate([[np.cos(ang/2)], np.sin(ang/2) * axis])

def expmat(w):
    th = np.linalg.norm(w)
    if th < 1e-14: return np.eye(3)
    k = w / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K

class Board:
    def __init__(self, rho_ap, rho_ml, b=0.15, M=3.0):
        self.b, self.M = b, M
        self.a1, self.a2 = b / rho_ap, b / rho_ml
        self.D2 = np.diag([self.a1**2, self.a2**2, b**2])
        self.gb = np.array([0.0, 0.0, -3*b/8])
        j1 = self.a1**2/5 + 19*b**2/320      # about body y (rocking in x-z, AP)
        j2 = self.a2**2/5 + 19*b**2/320      # about body x (rocking in y-z, ML)
        self.Ib = M * np.diag([j2, j1, (self.a1**2 + self.a2**2)/5])
        self.h0 = 5*b/8

    def rho(self, R):
        n = R.T @ np.array([0, 0, 1.0])                  # world up, in body coordinates
        xs = -self.D2 @ n / np.sqrt(n @ self.D2 @ n)      # lowest body point (contact)
        return R @ (xs - self.gb)

    def rhs(self, t, y, tau_fun=None):
        q = y[0:4] / np.linalg.norm(y[0:4]); w = y[4:7]
        R = qrot(q); r = self.rho(R); M = self.M
        e = 1e-6
        rdot = (self.rho(expmat(w*e) @ R) - self.rho(expmat(-w*e) @ R)) / (2*e)
        Iw = R @ self.Ib @ R.T
        A = Iw + M * (r @ r * np.eye(3) - np.outer(r, r))
        tau = tau_fun(t, y, self) if tau_fun else np.zeros(3)
        rhs = -np.cross(w, Iw @ w) - M * np.cross(r, np.cross(w, rdot)) \
              + M * G_ACC * np.cross(r, [0, 0, 1.0]) + tau
        alpha = np.linalg.solve(A, rhs)
        qd = 0.5 * qmul(np.concatenate([[0], w]), q)
        vG = -np.cross(w, r)
        return np.concatenate([qd, alpha, vG[:2]])

    # ---- observables ----
    def tilt(self, q):
        return np.arccos(np.clip(qrot(q)[2, 2], -1, 1))          # angle of body z from vertical
    def tilt_dir(self, q):
        z = qrot(q)[:, 2]; return np.arctan2(z[1], z[0])
    def yaw(self, q):
        return 2*np.arctan2(q[3], q[0])
    def energy(self, y):
        q = y[0:4] / np.linalg.norm(y[0:4]); w = y[4:7]
        R = qrot(q); r = self.rho(R); vG = -np.cross(w, r)
        return 0.5*self.M*vG @ vG + 0.5*w @ (R @ self.Ib @ R.T) @ w + self.M*G_ACC*(-r[2])

def initial(theta0, phi0, omega0=(0, 0, 0)):
    k = np.array([-np.sin(phi0), np.cos(phi0), 0.0])          # tilt the axis toward (cos phi, sin phi)
    q = qaxis(k, theta0)
    return np.concatenate([q, np.asarray(omega0, float), [0.0, 0.0]])

def simulate(board, y0, t_end, tau_fun=None, max_step=2e-3):
    return solve_ivp(lambda t, y: board.rhs(t, y, tau_fun), (0, t_end), y0, method="DOP853",
                     rtol=1e-10, atol=1e-12, max_step=max_step, dense_output=True)
