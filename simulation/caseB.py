"""Stable triaxial bases released off their principal axes: closest approach to level and yaw."""
import numpy as np, json
from scipy.optimize import minimize_scalar
from board3d import *
b=0.15
res={}
for ra,rm in [(1.2,0.75),(1.1,0.75)]:
    Bd=Board(ra,rm,b=b)
    print(f"triaxial rho_AP={ra}, rho_ML={rm}:  AP margin (q0-h0)/b={1/ra**2-0.625:+.3f}, ML margin={1/rm**2-0.625:+.3f}")
    for phi in (0,15,30,45,60,75,90):
        s=simulate(Bd,initial(np.radians(20),np.radians(phi)),20.0,max_step=5e-3)
        ts=np.linspace(0,20,16001); til=np.array([Bd.tilt(s.sol(t)[:4]) for t in ts]); yw=np.array([Bd.yaw(s.sol(t)[:4]) for t in ts])
        i=next(i for i in range(1,len(ts)-1) if til[i]<=til[i-1] and til[i]<=til[i+1])
        r=minimize_scalar(lambda t: Bd.tilt(s.sol(t)[:4]), bounds=(ts[i-1],ts[i+1]), method="bounded", options={"xatol":1e-9})
        y1=s.sol(r.x)
        res[f"{ra}/{rm} phi={phi}"]=dict(t1=r.x, tilt_min=np.degrees(Bd.tilt(y1[:4])), yaw1=np.degrees(Bd.yaw(y1[:4])),
                                       yaw_max=np.degrees(np.max(np.abs(yw))), yaw_end=np.degrees(yw[-1]), tilt_min_20s=np.degrees(til.min()))
        d=res[f"{ra}/{rm} phi={phi}"]
        print(f"  phi={phi:2d}: first closest approach t={d['t1']:.3f}s tilt={d['tilt_min']:.2f} deg yaw={d['yaw1']:+.3f} deg | over 20 s: max|yaw|={d['yaw_max']:.2f} deg, closest tilt={d['tilt_min_20s']:.2f} deg")
json.dump(res,open("caseB.json","w"),indent=1)
