import numpy as np, math
P,NS,M,RHO=0.9,300,0.02,0.01
def geninv(s,q):
    n=len(s); idx=min(max(int(math.ceil(q*n))-1,0),n-1); return float(s[idx])
def omega(a,d,p=P):
    s=np.sort(np.asarray(a).ravel()); return max(geninv(s,min(p+d,1))-geninv(s,p), geninv(s,p)-geninv(s,max(p-d,0)))
def env(a,d): return M+omega(a,d)+RHO
for k in ["highd","ngsim","waymo"]:
    a=np.load(k+".npy").ravel(); print(f"{k} @0.055: env={env(a,0.055):.4f}  (manuscript: highd/ngsim 0.056, waymo 0.088)")
import experiments as E
from data import RegimeGenerator
rm=np.random.default_rng(7); gen=RegimeGenerator(seed=int(rm.integers(1e9))); post=E.assign_post(20,True)
R,Bs,st,tc=E.build_fleet(20,E.T_DEFAULT,"freeflow",post,gen,E.T0,E.STAGGER)
ctrl=R[:,tc:].ravel(); print(f"controlled @0.055: env={env(ctrl,0.055):.4f} (manuscript 0.075)  range=[{ctrl.min():.3f},{ctrl.max():.3f}]")
import closed_loop as CL
r=np.random.default_rng(500); Bpre=CL.bstar("mild",np.random.default_rng(1)); Bpost=CL.bstar("aggr",np.random.default_rng(2))
leads,shift=CL.build_leads(20,3000,1000,750,r)
m=CL.simulate("P5",leads,shift,r,Bpre,Bpost,np.full(20,Bpre),lat_ln=(1.1,0.8),ge=(0.15,0.35),ret_scores=True)
sc=np.asarray(m["scores_post"]); print(f"closed scores range=[{sc.min():.3f},{sc.max():.3f}] n={sc.size} under_max={m['under_max']:.4f}")
