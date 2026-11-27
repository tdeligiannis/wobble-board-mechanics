"""Axisymmetric bases released while circling, with no spin about the board axis: yaw accumulation."""
import numpy as np, json
from board3d import *
b=0.15; res={}
for name,ra in [("oblate 0.75",0.75),("sphere 1.0",1.0),("prolate 1.25",1.25)]:
    Bd=Board(ra,ra,b=b)
    for vt in (0.25,0.5,1.0):
        y0=initial(np.radians(20),0.0)
        n=qrot(y0[:4])[:,2]                         # board axis
        w=np.array([1.0,0,0]); w=w-(w@n)*n; w=vt*w/np.linalg.norm(w)
        y0[4:7]=w                                   # tangential tilt rate, zero spin about the board axis
        T=20.0 if ra<1.2 else 40.0
        s=simulate(Bd,y0,T,max_step=5e-3)
        ts=np.linspace(0,T,int(T*400)+1); qs=[s.sol(t)[:4] for t in ts]
        til=np.degrees([Bd.tilt(q) for q in qs]); yw=np.unwrap([Bd.yaw(q) for q in qs]); dr=np.unwrap([Bd.tilt_dir(q) for q in qs])
        spin0=float(s.sol(0)[4:7]@n)
        revs=abs(dr[-1]-dr[0])/(2*np.pi); per=np.degrees(yw[-1]-yw[0])/revs if revs>0.5 else float('nan')
        res[f"{name} vt={vt}"]=dict(tilt_min=til.min(),tilt_max=til.max(),revs=revs,yaw_total=np.degrees(yw[-1]),yaw_per_rev=per,T=T,spin0=spin0)
        print(f"{name:12s} v_t={vt:4.2f} rad/s (spin0={spin0:+.1e}): tilt {til.min():5.2f}-{til.max():5.2f} deg, {revs:5.1f} rev/{T:.0f}s, yaw total {np.degrees(yw[-1]):+7.2f} deg, {per:+6.3f} deg/rev")
json.dump(res,open("caseC2.json","w"),indent=1)
