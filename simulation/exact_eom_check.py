"""3D simulation against the exact planar equation of motion, Eq. (eom_exact), for in-plane releases."""
import numpy as np
from scipy.integrate import solve_ivp
from board3d import *
b=0.15
for name,rho in [("oblate",0.75),("prolate",1.25),("board U-like",1.6)]:
    a=b/rho; B=Board(rho,rho,b=b); h0=5*b/8; d=b-h0; j2=a*a/5+19*b*b/320  # rocking about body y uses a1
    hk=lambda t: np.sqrt(a*a*np.sin(t)**2+b*b*np.cos(t)**2)
    H=lambda t: hk(t)-d*np.cos(t)
    def dH(t,e=1e-6): return (H(t+e)-H(t-e))/(2*e)
    I=lambda t: j2+H(t)**2+dH(t)**2
    def dI(t,e=1e-6): return (I(t+e)-I(t-e))/(2*e)
    f=lambda t,s: [s[1], (-0.5*dI(s[0])*s[1]**2 - 9.81*dH(s[0]))/I(s[0])]
    th0=np.radians(20) if rho<1.3 else np.radians(5)
    T=1.5 if rho<1.3 else 0.6
    s2=solve_ivp(f,(0,T),[th0,0],rtol=1e-11,atol=1e-12,dense_output=True)
    s3=simulate(B,initial(th0,0.0),T)
    ts=np.linspace(0,T,301)
    th3=np.array([B.tilt(s3.sol(t)[:4])*np.sign(np.cos(B.tilt_dir(s3.sol(t)[:4]))) for t in ts])
    print(f"{name:12s}: max |theta_exact_planar - theta_3D| = {np.degrees(np.max(np.abs(s2.sol(ts)[0]-th3))):.1e} deg over {T} s")
