"""In-plane releases from 20 deg on axisymmetric bases: time to level, yaw, distance rolled."""
import numpy as np, json
from scipy.optimize import minimize_scalar
from board3d import *
b=0.15
def first_level(Bd, y0, t_end=20.0):
    s=simulate(Bd,y0,t_end,max_step=5e-3)
    ts=np.linspace(0,t_end,8001); til=np.array([Bd.tilt(s.sol(t)[:4]) for t in ts])
    # first local minimum of tilt
    i=next(i for i in range(1,len(ts)-1) if til[i]<=til[i-1] and til[i]<=til[i+1])
    r=minimize_scalar(lambda t: Bd.tilt(s.sol(t)[:4]), bounds=(ts[i-1],ts[i+1]), method="bounded", options={"xatol":1e-9})
    t1=r.x; y1=s.sol(t1)
    yaws=np.array([Bd.yaw(s.sol(t)[:4]) for t in ts])
    return dict(t_level=t1, tilt_min_deg=np.degrees(Bd.tilt(y1[:4])), yaw_at_level_deg=np.degrees(Bd.yaw(y1[:4])),
                max_abs_yaw_deg_20s=np.degrees(np.max(np.abs(yaws))), rolled_mm=1000*np.hypot(y1[7],y1[8]))
out={}
for name,ra,rm in [("oblate 0.75",0.75,0.75),("sphere 1.0",1.0,1.0),("prolate 1.25",1.25,1.25)]:
    for phi in (0,37):
        r=first_level(Board(ra,rm,b=b), initial(np.radians(20),np.radians(phi)))
        out[f"{name}, phi={phi}"]=r
        print(f"{name:13s} phi={phi:2d}: level at t={r['t_level']:.3f}s  min tilt={r['tilt_min_deg']:.1e} deg  yaw at level={r['yaw_at_level_deg']:+.1e} deg  max|yaw| 20s={r['max_abs_yaw_deg_20s']:.1e} deg  rolled={r['rolled_mm']:.1f} mm")
json.dump(out,open("caseA.json","w"),indent=1)
