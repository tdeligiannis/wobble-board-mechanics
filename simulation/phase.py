"""Yaw against the solid angle swept by the board axis, Eq. (yaw_phase)."""
import numpy as np, json
from board3d import *
b=0.15; out={}
print("Test: yaw change = solid angle enclosed by the board axis, in the sense of circling, i.e. integral of (1 - cos theta) d(phi)")
for name,ra in [("oblate 0.75",0.75),("sphere 1.0",1.0),("prolate 1.25",1.25),("triaxial 1.2/0.75",None)]:
    Bd=Board(ra,ra,b=b) if ra else Board(1.2,0.75,b=b)
    for vt in (0.25,1.0):
        y0=initial(np.radians(20),0.0); n=qrot(y0[:4])[:,2]
        w=np.array([1.0,0,0]); w=w-(w@n)*n; y0[4:7]=vt*w/np.linalg.norm(w)
        T=12.0
        s=simulate(Bd,y0,T,max_step=5e-3); ts=np.linspace(0,T,int(T*2000)+1); qs=[s.sol(t)[:4] for t in ts]
        th=np.array([Bd.tilt(q) for q in qs]); ph=np.unwrap([Bd.tilt_dir(q) for q in qs]); yw=np.unwrap([Bd.yaw(q) for q in qs])
        geo=+np.concatenate([[0],np.cumsum(0.5*((1-np.cos(th[1:]))+(1-np.cos(th[:-1])))*np.diff(ph))])
        err=np.max(np.abs(yw-geo))
        out[f"{name} vt={vt}"]=dict(yaw_end=np.degrees(yw[-1]),geo_end=np.degrees(geo[-1]),max_err_deg=np.degrees(err))
        print(f"  {name:18s} v_t={vt:4.2f}: yaw after {T:.0f}s = {np.degrees(yw[-1]):+8.3f} deg, solid angle = {np.degrees(geo[-1]):+8.3f} deg, max mismatch over run = {np.degrees(err):.1e} deg")
json.dump(out,open("phase.json","w"),indent=1)
print("\nCircular sway of amplitude A, yaw per cycle = 2*pi*(1-cos A):")
for A in (5,8,12,15,20): print(f"  A={A:2d} deg: {np.degrees(2*np.pi*(1-np.cos(np.radians(A)))):6.2f} deg per cycle")
