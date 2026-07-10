"""Experiment K: cross-domain heterogeneous fleet. Vehicles are drawn from
DIFFERENT real corpora (highD / NGSIM / Waymo) in one fleet -- the worst case
for collaboration (different cities/countries/vehicle types, post-shift matching
boundaries 0.31 / 0.38 / 0.85). Tests whether shift-aware weighting keeps
collaboration graceful and whether the safety bound holds under extreme
heterogeneity. Honest: numbers are from this run."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, json
import real_backend as RB
import experiments as E
from recalibration import Params, simulate

RB.NAME_MAP = dict(RB.NAME_MAP); RB.NAME_MAP["calm"]="highd"   # a post-domain that stays in highD
E.RegimeGenerator = RB.DatasetRegimes; E._BSTAR = {}
p=Params(); K=21; T=E.T_DEFAULT; SEEDS=15; rm=np.random.default_rng(7)

def run(post, label):
    out={}
    for cond in ["B1","P5"]:
        re_,umax,bnd=[],[],[]
        for s in range(SEEDS):
            gen=RB.DatasetRegimes(seed=int(rm.integers(1e9)))
            R,Bs,st,tc=E.build_fleet(K,T,"freeflow",post,gen,E.T0,E.STAGGER)
            if cond=="B1": bnd.append(E.theoretical_bound(R,tc,p))
            m=E.metrics(simulate(cond,R,Bs,tc,p,gen.rng),st,p)
            re_.append(float(np.mean(m["rate_err"]))); umax.append(float(np.max(m["under_max"])))
        out[cond]={"rate_err":E.boot_ci(re_),"under_max":E.boot_ci(umax)}
    out["bound"]=E.boot_ci(bnd)
    print(f"[{label}] bound={out['bound'][0]:.4f}")
    for c in ["B1","P5"]:
        print(f"   {c}: rate_err={out[c]['rate_err'][0]:.4f} {out[c]['rate_err'][1:]}  under_max={out[c]['under_max'][0]:.4f} {out[c]['under_max'][1:]}")
    return out

res={}
# homogeneous reference: all post -> ngsim
res["homogeneous"]=run(["congested"]*K, "homogeneous (all NGSIM)")
# cross-domain heterogeneous: mix highD-stable / Waymo / NGSIM
mix=[["congested","moderate","calm"][i%3] for i in range(K)]   # ngsim, waymo, highd
res["cross_domain"]=run(mix, "cross-domain (NGSIM+Waymo+highD)")
json.dump(res, open("results_real/experiment_K.json","w"), indent=2)
print("saved results_real/experiment_K.json")
