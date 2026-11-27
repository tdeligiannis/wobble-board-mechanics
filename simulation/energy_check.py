"""Energy conservation check quoted in Methods: triaxial base (rho_AP=0.75, rho_ML=1.25),
released at 15 deg, tilt direction 0.6 rad, initial angular velocity (0.3, -0.2, 0) rad/s, 5 s, no applied torque."""
import numpy as np
from board3d import *
B = Board(0.75, 1.25)
s = simulate(B, initial(np.radians(15), 0.6, (0.3, -0.2, 0.0)), 5.0)
E = [B.energy(s.sol(t)) for t in np.linspace(0, 5, 200)]
print("relative energy drift: %.1e" % ((max(E) - min(E)) / abs(E[0])))
