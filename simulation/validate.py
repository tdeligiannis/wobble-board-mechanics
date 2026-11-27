"""Planar release of a sphere against the exact 2D equation, and energy conservation for three releases."""
import numpy as np
from scipy.integrate import solve_ivp
from board3d import *
# 1) planar sphere release from 20 deg vs the exact 2D equation
b=0.15; B=Board(1.0,1.0,b=b)
sol=simulate(B, initial(np.radians(20),0.0), 3.0)
q=b; d=b-5*b/8; j2=b*b/5+19*b*b/320
def f2d(t,s):
    th,om=s
    I=j2+q*q+d*d-2*q*d*np.cos(th)
    return [om,(-q*d*np.sin(th)*om**2 - G_ACC*d*np.sin(th))/I]
s2=solve_ivp(f2d,(0,3),[np.radians(20),0],rtol=1e-11,atol=1e-12,dense_output=True)
ts=np.linspace(0,3,301)
th3=np.array([B.tilt(sol.sol(t)[:4]) * np.sign(np.cos(B.tilt_dir(sol.sol(t)[:4]))) for t in ts])
th2=s2.sol(ts)[0]
print("sphere planar: max |theta_3D - theta_2D| = %.2e deg" % np.degrees(np.max(np.abs(th3-th2))))
# 2) energy conservation, hardest case: triaxial released off-axis with spin
for name,(ra,rm),y0 in [("sphere planar",(1,1),initial(np.radians(20),0)),
                        ("triaxial 1.2/0.75 off-axis",(1.2,0.75),initial(np.radians(20),np.radians(45))),
                        ("oblate swirling",(0.75,0.75),initial(np.radians(20),0,(0,0,0.0)) )]:
    Bb=Board(ra,rm,b=b)
    if name=="oblate swirling":
        y0=initial(np.radians(20),0); y0[4:7]=[1.5,0,0.0]   # tangential tilt rate about world x
    s=simulate(Bb,y0,6.0)
    E=[Bb.energy(s.sol(t)) for t in np.linspace(0,6,121)]
    print(f"{name:28s} energy drift = {(max(E)-min(E))/abs(E[0]-Bb.M*G_ACC*Bb.h0+1e-30):.2e} of the initial excess energy")
