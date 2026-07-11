#!/usr/bin/env python3
"""Real-pool clamp ablation + Waymo held-out generalization (open-loop resampled).
Uses the REAL highD/NGSIM/Waymo risk pools via real_backend.patch().
- Ablation isolates WHICH component secures safety: local-only, naive fleet-clamp,
  ACI-no-clamp, robust-agg-no-clamp, fleet+clamp (unweighted/robust/shift-aware).
- Waymo is HELD OUT: parameters (Ns,m,alpha) fixed from the highD/NGSIM development
  corpora; Waymo is used only to test whether the under-protection bound generalizes.
All numbers from this run."""
import numpy as np, json, os
import real_backend, experiments as E
E = real_backend.patch()                      # patch RegimeGenerator -> real pools; RES=results_real
from extra_experiments import Params, simulate_agg, agg_wquantile, agg_trimmed, agg_mean
OUT=os.path.dirname(os.path.abspath(__file__))

p=Params(); K=20; T=E.T_DEFAULT
ADV=np.zeros(K,bool); ADV[:int(0.30*K)]=True   # 30% adversarial uploads (U=1.0 maximal)

# variant -> (cond, use_adv, aggregator).  cond semantics in simulate_agg:
#   B1=local clamp only; ACI=no clamp; FEDNOCLAMP=clamp reads fleet (no local clamp);
#   AGG/P5=fleet assimilated + LOCAL clamp (agg picks weighting).
VARIANTS={
 "B1_local_only":     ("B1",         False, agg_wquantile),  # no fleet, local clamp
 "B2_naive_fleetclamp":("FEDNOCLAMP", True,  agg_mean),       # clamp reads unweighted fleet -> exposed
 "ACI_no_clamp":      ("ACI",        False, agg_wquantile),  # online conformal, no clamp
 "Robust_no_clamp":   ("FEDNOCLAMP", True,  agg_trimmed),    # robust agg but NO local clamp
 "FedAvg_plus_clamp": ("AGG",        True,  agg_mean),       # fleet + local clamp, unweighted agg
 "Robust_plus_clamp": ("AGG",        True,  agg_trimmed),    # fleet + local clamp, robust agg
 "P5_full":           ("P5",         True,  agg_wquantile),  # fleet + local clamp, shift-aware (full)
}

def run(pre, post, variants, seeds, seed0):
    rm=np.random.default_rng(seed0)
    out={v:{"rate_err":[],"under_max":[],"missed":[]} for v in variants}; bnd=[]
    for s in range(seeds):
        gen=E.RegimeGenerator(seed=int(rm.integers(1e9)))
        post_regimes=[post]*K
        R,Bs,st,tc=E.build_fleet(K,T,pre,post_regimes,gen,E.T0,E.STAGGER)
        bnd.append(E.theoretical_bound(R,tc,p))
        for v in variants:
            cond,use_adv,agg=VARIANTS[v]
            mask=ADV if use_adv else None
            log=simulate_agg(cond,R,Bs,tc,p,gen.rng,agg=agg,adv_mask=mask)
            m=E.metrics(log,st,p)
            out[v]["rate_err"].append(float(np.mean(m["rate_err"])))
            out[v]["under_max"].append(float(np.max(m["under_max"])))
            out[v]["missed"].append(float(np.mean(m["missed"])))
    res={v:{k:E.boot_ci(val)+[float(np.std(val,ddof=1))] for k,val in out[v].items()} for v in variants}  # [mean,lo,hi,std]
    res["bound"]=E.boot_ci(bnd)
    res["Bstar_pre"]=float(E.RegimeGenerator(seed=1).true_matching_boundary(pre,p.tau))
    res["Bstar_post"]=float(E.RegimeGenerator(seed=1).true_matching_boundary(post,p.tau))
    return res

if __name__=="__main__":
    import sys
    what=sys.argv[1] if len(sys.argv)>1 else "ablation"
    seeds=int(sys.argv[2]) if len(sys.argv)>2 else 15
    if what=="ablation":
        # DEVELOPMENT shift: highD (freeflow) -> NGSIM (congested), the severe stress domain
        res=run("freeflow","congested",list(VARIANTS),seeds,seed0=777)
        json.dump(res,open(os.path.join(OUT,"results_real","reviewer_ablation.json"),"w"),indent=2)
        print(f"ABLATION highD->NGSIM (n={seeds}) bound={res['bound'][0]:.3f} "
              f"B*post={res['Bstar_post']:.3f}")
        for v in VARIANTS:
            print(f"  {v:<22} under_max={res[v]['under_max'][0]:.3f}  rate_err={res[v]['rate_err'][0]:.3f}")
    elif what=="holdout":
        # HELD-OUT generalization: params fixed from dev; test on Waymo (moderate), never tuned
        res=run("freeflow","moderate",["B1_local_only","B2_naive_fleetclamp","P5_full"],seeds,seed0=888)
        json.dump(res,open(os.path.join(OUT,"results_real","reviewer_waymo_holdout.json"),"w"),indent=2)
        print(f"WAYMO HELD-OUT highD->Waymo (n={seeds}) bound={res['bound'][0]:.3f} B*post={res['Bstar_post']:.3f}")
        for v in ["B1_local_only","B2_naive_fleetclamp","P5_full"]:
            print(f"  {v:<22} under_max={res[v]['under_max'][0]:.3f}  rate_err={res[v]['rate_err'][0]:.3f}")
