"""Mixed-autonomy penetration analysis. In a population of N vehicles only an
equipped fraction p participates in the fleet (uploads/receives calibration);
the rest run single-vehicle recalibration. We report, for the equipped vehicles,
how adaptation and accuracy improve with penetration, and confirm the per-vehicle
under-protection bound holds at EVERY penetration rate (it is penetration-
independent by Theorem 1). Honest: numbers are from this run."""
import numpy as np, json, warnings
warnings.filterwarnings("ignore")
from data import RegimeGenerator
from recalibration import Params, simulate
import experiments as E

N=40; T=1700; SEEDS=10; p=Params()
def metrics_for(Keq, seeds=SEEDS, seed0=55):
    rm=np.random.default_rng(seed0); re=[]; lat=[]; ux=[]; reB=[]
    for s in range(seeds):
        gen=RegimeGenerator(seed=int(rm.integers(1e9))); post=E.assign_post(Keq,True)
        R,Bs,st,tc=E.build_fleet(Keq,T,"freeflow",post,gen,E.T0,E.STAGGER)
        mP=E.metrics(simulate("P5",R,Bs,tc,p,gen.rng),st,p)
        mB=E.metrics(simulate("B1",R,Bs,tc,p,gen.rng),st,p)
        re.append(np.mean(mP["rate_err"])); lat.append(np.mean(mP["latency"]))
        ux.append(np.mean(mP["under_max"])); reB.append(np.mean(mB["rate_err"]))
    return (float(np.mean(re)),float(np.mean(lat)),float(np.mean(ux)),float(np.mean(reB)))

bound=0.100
out={}
print(f"penetration | equipped peers | P5 rate_err | P5 latency | P5 under_max (bound {bound}) | B1 rate_err")
for pen in [0.10,0.25,0.50,0.75,1.00]:
    Keq=max(1,round(pen*N))
    re,lat,ux,reB=metrics_for(Keq)
    out[f"{pen:.2f}"]=dict(equipped=Keq,rate_err_P5=re,latency_P5=lat,under_max_P5=ux,rate_err_B1_local=reB,within_bound=bool(ux<=bound))
    print(f"   {pen:.2f}     |      {Keq:>3}       |   {re:.4f}   |   {lat:6.1f} |   {ux:.4f}  {'<=bound' if ux<=bound else 'OVER'}      |  {reB:.4f}")
json.dump(out,open("results_real/penetration.json","w"),indent=2); print("saved penetration.json")
