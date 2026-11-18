#!/usr/bin/env python3
"""Resumable driver for the reviewer-strengthening experiments.
Runs in <45s chunks. Paired seeds across conditions (seed=100+i) so conditions
share the same random draws -> enables paired significance tests.
Usage: python3 run_block.py <block_id> <seeds_this_call>
Accumulates raw per-seed values in results_real/_reviewer_accum.json (resumable)."""
import sys, json, os, time
import numpy as np
import deploy_experiments as DX

OUT=os.path.dirname(os.path.abspath(__file__))
ACC=os.path.join(OUT,"results_real","_reviewer_accum.json")
_BS=os.path.join(OUT,"results_real","_bstar.json" if os.environ.get("CL_SCORE","8")=="8" else "_bstar_4feat.json")
if os.path.exists(_BS):
    BS=json.load(open(_BS)); Bpre,Bpost=BS["Bpre"],BS["Bpost"]
else:  # fresh clone: compute the closed-loop matching boundaries (same seeds as run_reviewer_experiments.cache_bstar)
    import closed_loop as _CL
    Bpre=_CL.bstar("mild",np.random.default_rng(1)); Bpost=_CL.bstar("aggr",np.random.default_rng(2))
METRICS=["coll","conflict","rate_err","under_max","speed","unnec_interv",
         "missed_highrisk","hardbrake_freq","speed_disturb","thw_post","thw_after_interv"]

def units_for(block):
    """Return list of (unit_name, kwargs) where kwargs has 'cond','K' and physics params."""
    abl=["AEB","B0","B1","B2","RobNoClamp","FedAvg","RobFCP","P5"]
    if block=="abl0":   return [(c,dict(cond=c,adv=0.0)) for c in abl]
    if block=="abl25":  return [(f"{c}_adv25",dict(cond=c,adv=0.25)) for c in abl]
    if block=="v2x":
        u=[]
        for ms in (100,300,500,1000):
            u.append((f"P5_delay{ms}",dict(cond="P5",latency=ms//100)))
        for pl in (5,10,20,40):
            u.append((f"P5_loss{pl}",dict(cond="P5",loss=pl/100)))
        u.append(("P5_stale",dict(cond="P5",stale=30)))                      # reuse ~3s-old estimate
        u.append(("P5_byz20",dict(cond="P5",adv=0.20)))                      # 20% malicious uploads
        u.append(("P5_mixed",dict(cond="P5",latency=3,loss=0.20,adv=0.20)))  # delay+loss+malicious
        u.append(("B2_mixed",dict(cond="B2",latency=3,loss=0.20,adv=0.20)))  # naive clamp, same faults
        return u
    if block=="sens_adv":  return [(f"P5_adv{int(a*100)}",dict(cond="P5",adv=a)) for a in (0.0,0.10,0.25,0.50,0.70)] \
                                + [(f"B2_adv{int(a*100)}",dict(cond="B2",adv=a)) for a in (0.0,0.10,0.25,0.50,0.70)]
    if block=="sens_K":    return [(f"P5_K{K}",dict(cond="P5",K=K)) for K in (5,10,25,50,100)] \
                                + [(f"B1_K{K}",dict(cond="B1",K=K)) for K in (5,10,25,50,100)]
    if block=="sens_Ns":   return [(f"P5_Ns{n}",dict(cond="P5",Ns=n)) for n in (150,300,600)]
    if block=="sens_m":    return [(f"P5_m{int(m*1000)}",dict(cond="P5",m=m)) for m in (0.01,0.02,0.05)]
    if block=="sens_tau":  return [(f"P5_tau{int(t*100)}",dict(cond="P5",tau=t)) for t in (0.05,0.10,0.15)]
    if block=="sens_P":    return [(f"P5_P{p}",dict(cond="P5",Pagg=p)) for p in (10,20,50)]
    if block=="deploy":  # item 8: 30% delayed + 20% malicious during the mild->aggressive transition
        kw=dict(latency=3,loss=0.30,adv=0.20)  # ~300ms delay, 30% dropped, 20% malicious
        return [("deploy_AEB",dict(cond="AEB",**kw)),("deploy_B0",dict(cond="B0",**kw)),
                ("deploy_B1",dict(cond="B1",**kw)),("deploy_B2",dict(cond="B2",**kw)),
                ("deploy_P5",dict(cond="P5",**kw))]
    raise SystemExit("unknown block "+block)

def main():
    block=sys.argv[1]; add=int(sys.argv[2])
    acc=json.load(open(ACC)) if os.path.exists(ACC) else {}
    units=units_for(block); t0=time.time()
    for name,kw in units:
        cond=kw.pop("cond"); K=kw.pop("K",20)
        rec=acc.setdefault(name,{m:[] for m in METRICS})
        done=len(rec["coll"])
        for i in range(done,done+add):
            r=np.random.default_rng(100+i)
            leads,shift=DX.build_leads(K,3000,1000,750,r)
            m=DX.simulate(cond,leads,shift,r,Bpre,Bpost,np.full(K,Bpre),**kw)
            for mt in METRICS: rec[mt].append(m[mt])
        kw["cond"]=cond  # restore (not needed, local)
    json.dump(acc,open(ACC,"w"))
    n=len(acc[units[0][0]]["coll"])
    print(f"block={block} now n={n} seeds | {len(units)} units | {time.time()-t0:.1f}s this call")

if __name__=="__main__": main()
