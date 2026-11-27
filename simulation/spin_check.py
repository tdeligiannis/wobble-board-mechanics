"""Yaw of a spherical board released while circling, with and without spin about its own axis."""
import numpy as np
from board3d import *
B=Board(1.0,1.0,b=0.15)
for spin in (0.0,1.0):
    y0=initial(np.radians(20),0.0); R=qrot(y0[:4]); n=R[:,2]
    w=np.array([1.0,0,0]); w=w-(w@n)*n; w=0.25*w/np.linalg.norm(w)
    y0[4:7]=w+spin*n
    T=40.0; s=simulate(B,y0,T,max_step=5e-3)
    ts=np.linspace(0,T,int(T*2000)+1); qs=[s.sol(t)[:4] for t in ts]
    th=np.array([B.tilt(q) for q in qs]); ph=np.unwrap([B.tilt_dir(q) for q in qs]); yw=np.unwrap([B.yaw(q) for q in qs])
    geo=np.concatenate([[0],np.cumsum(0.5*((1-np.cos(th[1:]))+(1-np.cos(th[:-1])))*np.diff(ph))])
    print(f"axial spin {spin:.1f} rad/s: turns {abs(ph[-1]-ph[0])/(2*np.pi):.1f}, yaw {np.degrees(yw[-1]):+.2f} deg, solid angle {np.degrees(geo[-1]):+.2f} deg")
