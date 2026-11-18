#!/usr/bin/env python3
"""Closed-loop transportation microsimulation (IDM car-following + V2X) for the
P5 fleet-recalibration paper. K independent aggressive follower-leader pairs with
staggered (heterogeneous) distribution shift from a mild to an aggressive leader
regime. Each follower runs the safety supervisor; the fleet recalibrates over a
V2X channel with latency and packet loss. External baseline: fixed-TTC AEB.
Reports transportation safety (collisions, conflict rate), mobility (mean speed),
calibration (rate error) and the under-protection bound. Honest: numbers are from
this run."""
import numpy as np, json, argparse
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

DT=0.1
A_MAX,B_COMF,V0,T_HW,S0,DELTA,LEN = 1.8,2.5,33.0,0.45,1.0,4.0,5.0
TAU,GAMMA,NS,M,P,H = 0.10,0.05,300,0.02,20,0.03
TTC_CRIT=1.5; SUP_DECEL=5.5
REG={"mild":(0.022,11.0),"aggr":(0.065,21.0)}
V_INIT,GAP_INIT,LEAD_BASE=26.0,22.0,27.0

def risk4(gap,v,dv):
    """Reduced four-feature score (inverse TTC, inverse THW, DRAC, speed; weights 0.32/0.26/0.26/0.16).
    Used only for the four-feature ablation (CL_SCORE=4)."""
    gap=np.maximum(gap,0.1); appr=np.maximum(dv,0.0)
    with np.errstate(divide="ignore",invalid="ignore"):
        ttc=np.where(appr>1e-3,gap/np.where(appr>1e-3,appr,1.0),1e9)
        thw=np.where(v>1e-2,gap/np.where(v>1e-2,v,1.0),1e9)
    drac=appr**2/(2*gap)
    f1=8.0/np.maximum(ttc,1e-3); f1=f1/(1+f1)
    f2=3.0/np.maximum(thw,1e-3); f2=f2/(1+f2)
    f3=drac/4.0; f3=f3/(1+f3)
    f4=v/30.0; f4=f4/(1+f4)
    return np.clip(0.32*f1+0.26*f2+0.26*f3+0.16*f4,0,1)

# ---- risk score used in closed loop: the eight-feature score of Eq. (1) (default) or the
# ---- reduced four-feature ablation (set environment variable CL_SCORE=4).
import os as _os
SCORE = _os.environ.get("CL_SCORE", "8")
_JS = {"v1": None, "a1": None}
def reset_risk():
    """Call at the start of every simulated episode (resets the jerk estimator)."""
    _JS["v1"] = None; _JS["a1"] = None
def risk8(gap,v,dv):
    """Eight-feature score of Eq. (1) (risk_score.py); acceleration and jerk from finite differences of speed at DT."""
    from risk_score import risk_score
    va = np.asarray(v, float)
    if _JS["v1"] is None or np.shape(_JS["v1"]) != np.shape(va):
        a = np.zeros_like(va); j = np.zeros_like(va)
    else:
        a = (va - _JS["v1"]) / DT
        j = np.zeros_like(va) if _JS["a1"] is None else (a - _JS["a1"]) / DT
    _JS["v1"] = va.copy(); _JS["a1"] = a
    return risk_score(gap, va, dv, jerk=j)
def risk(gap,v,dv):
    return risk8(gap,v,dv) if SCORE == "8" else risk4(gap,v,dv)

def idm(v,gap,dv):
    s_star=S0+np.maximum(0.0,v*T_HW+v*dv/(2*np.sqrt(A_MAX*B_COMF)))
    return A_MAX*(1-(v/V0)**DELTA-(s_star/np.maximum(gap,0.5))**2)

def lead1(T,rng,regime):
    rate,depth=REG[regime]; v=np.full(T,LEAD_BASE); t=0
    while t<T:
        t+=1+int(rng.exponential(1.0/rate))
        if t<T:
            d=int(rng.integers(6,16)); v[t:t+d]=np.maximum(2.0,LEAD_BASE-depth)
    return np.clip(v,1.0,30.0)

def build_leads(K,T,t0,stagger,rng):
    shift=(t0+rng.integers(0,stagger+1,K)).astype(int)
    leads=np.zeros((K,T))
    for k in range(K):
        s=shift[k]; leads[k,:s]=lead1(s,rng,"mild"); leads[k,s:]=lead1(T-s,rng,"aggr")
    return leads,shift

def bstar(regime,rng,n=50000):
    reset_risk()
    lead=lead1(n,rng,regime); v=V_INIT; gap=GAP_INIT; Rs=np.empty(n)
    for t in range(n):
        dv=v-lead[t]; Rs[t]=risk(gap,v,dv)
        a=np.clip(idm(v,gap,dv),-B_COMF*2,A_MAX); v=max(0,v+a*DT); gap=max(0.5,gap+(lead[t]-v)*DT)
        if gap<=0.6: gap=GAP_INIT; v=lead[t]
    return float(np.quantile(Rs,1-TAU))

def simulate(cond,leads,shift,rng,Bpre,Bpost,B0c,latency=0,loss=0.0,lat_ln=None,ge=None,ret_scores=False):
    reset_risk()
    K,T=leads.shape
    v=np.full(K,V_INIT); gap=np.full(K,GAP_INIT)
    win=np.full((K,NS),Bpre); B=np.full(K,Bpre); ptr=0; pend=[]; ge_state=0
    U=np.quantile(win,1-TAU,axis=1)
    interv=np.zeros((K,T)); coll=0; under=np.zeros(T); spd=np.zeros(T); conf=np.zeros(T)
    Rlog=np.zeros((K,T)) if ret_scores else None
    for t in range(T):
        ld=leads[:,t]; dv=v-ld; R=risk(gap,v,dv)
        if ret_scores: Rlog[:,t]=R
        with np.errstate(divide="ignore",invalid="ignore"):
            ttc=np.where(dv>0.1,gap/np.where(dv>0.1,dv,1.0),1e9)
        Bstar_t=np.where(t>=shift,Bpost,Bpre)
        if t%5==0: U=np.quantile(win,1-TAU,axis=1)
        if cond=="AEB":
            engage=ttc<TTC_CRIT
        elif cond=="B0":
            Beff=B0c; engage=R>Beff; under[t]=np.max(np.clip(Beff-Bstar_t,0,None))
        else:
            Beff=np.minimum(B,U+M); engage=R>Beff; under[t]=np.max(np.clip(Beff-Bstar_t,0,None))
        a=np.where(engage,-SUP_DECEL,np.clip(idm(v,gap,dv),-B_COMF*2,A_MAX))
        v=np.clip(v+a*DT,0,V0); gap=gap+(ld-v)*DT
        hit=gap<=0.4; coll+=int(hit.sum()); gap=np.where(hit,GAP_INIT,gap); v=np.where(hit,ld,v); gap=np.maximum(gap,0.5)
        conf[t]=np.mean((ttc>0)&(ttc<TTC_CRIT)); interv[:,t]=engage; spd[t]=np.mean(v)
        if cond in ("B1","P5","FedAvg","RobFCP"): B=np.clip(B+GAMMA*(engage.astype(float)-TAU),0,1)
        if cond=="P5" and t>0 and t%P==0:
            if ge is not None:
                ge_state = (1 if rng.random()<ge[0] else 0) if ge_state==0 else (0 if rng.random()<ge[1] else 1)
                drop = (ge_state==1)
            else:
                drop = (rng.random()<loss)
            if not drop:
                lat = latency if lat_ln is None else max(0,int(round(rng.lognormal(lat_ln[0],lat_ln[1]))))
                ws=np.exp(-((U-np.median(U))**2)/(2*H**2)); ws/=ws.sum(); pend.append((t+lat,float(np.sum(ws*U))))
            ready=[uf for(dd,uf)in pend if dd<=t]; pend=[(dd,uf)for(dd,uf)in pend if dd>t]
            if ready:
                Uf=ready[-1]; lam=0.5*np.exp(-((Uf-U)**2)/(2*H**2)); B=np.clip((1-lam)*B+lam*Uf,0,1)
        if cond=="FedAvg" and t>0 and t%P==0:
            Uf=float(np.mean(U)); lam=0.5; B=np.clip((1-lam)*B+lam*Uf,0,1)  # naive unweighted FedAvg, fixed trust, no shift gate
        if cond=="RobFCP" and t>0 and t%P==0:
            u=np.sort(U); kk=int(0.15*len(u)); Uf=float(np.mean(u[kk:len(u)-kk])) if len(u)>2*kk else float(np.mean(u))  # Rob-FCP: trimmed (Byzantine-robust) aggregate
            lam=0.5; B=np.clip((1-lam)*B+lam*Uf,0,1)
        win[:,ptr]=R; ptr=(ptr+1)%NS
    post=slice(int(shift.max())+200,T)
    rate=np.array([np.mean(interv[k,post]) for k in range(K)])
    veh_steps=K*(T-int(shift.max()))/1000.0
    out=dict(coll=coll/max(veh_steps,1e-9), conflict=float(np.mean(conf[post])),
                rate_err=float(np.mean(np.abs(rate-TAU))),
                under_max=float(np.max(under[post])) if cond!="AEB" else np.nan,
                speed=float(np.mean(spd[post])))
    if ret_scores: out["scores_post"]=Rlog[:,post].ravel()
    return out

def ci(vals):
    v=np.array([x for x in vals if not(isinstance(x,float)and np.isnan(x))])
    if len(v)<2: return [float(v.mean()) if len(v) else np.nan]*3
    rng=np.random.default_rng(0); bs=[v[rng.integers(0,len(v),len(v))].mean() for _ in range(2000)]
    return [float(v.mean()),float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--seeds",type=int,default=12)
    ap.add_argument("--K",type=int,default=20); ap.add_argument("--T",type=int,default=3000); a=ap.parse_args()
    Bpre=bstar("mild",np.random.default_rng(1)); Bpost=bstar("aggr",np.random.default_rng(2))
    B0c=np.full(a.K,Bpre)
    print(f"B*_pre(mild)={Bpre:.3f} B*_post(aggr)={Bpost:.3f}")
    res={"Bpre":Bpre,"Bpost":Bpost,"conditions":{},"v2x":{}}
    for cond in ["AEB","B0","B1","FedAvg","RobFCP","P5"]:
        acc={m:[] for m in["coll","conflict","rate_err","under_max","speed"]}
        for s in range(a.seeds):
            r=np.random.default_rng(100+s); leads,shift=build_leads(a.K,a.T,a.T//3,a.T//4,r)
            m=simulate(cond,leads,shift,r,Bpre,Bpost,B0c)
            for k in acc: acc[k].append(m[k])
        res["conditions"][cond]={k:ci(v) for k,v in acc.items()}
        c=res["conditions"][cond]
        print(f"{cond:>4}: coll/1k={c['coll'][0]:.3f} conflict={c['conflict'][0]:.4f} rate_err={c['rate_err'][0]:.4f} under={c['under_max'][0]} speed={c['speed'][0]:.2f}")
    for(lat,loss)in[(0,0.0),(3,0.2),(10,0.5)]:
        acc={m:[] for m in["coll","rate_err","under_max"]}
        for s in range(a.seeds):
            r=np.random.default_rng(200+s); leads,shift=build_leads(a.K,a.T,a.T//3,a.T//4,r)
            m=simulate("P5",leads,shift,r,Bpre,Bpost,B0c,latency=lat,loss=loss)
            for k in acc: acc[k].append(m[k])
        key=f"lat{int(lat*DT*1000)}ms_loss{int(loss*100)}"; res["v2x"][key]={k:ci(v) for k,v in acc.items()}
        c=res["v2x"][key]; print(f"V2X {key}: coll/1k={c['coll'][0]:.3f} rate_err={c['rate_err'][0]:.4f} under={c['under_max'][0]:.4f}")
    # realistic V2X: log-normal latency (median ~0.3s, heavy tail to ~1.5s) + Gilbert-Elliott bursty loss (~30%)
    acc={m:[] for m in ["coll","rate_err","under_max"]}
    for s in range(a.seeds):
        r=np.random.default_rng(500+s); leads,shift=build_leads(a.K,a.T,a.T//3,a.T//4,r)
        m=simulate("P5",leads,shift,r,Bpre,Bpost,B0c,lat_ln=(1.1,0.8),ge=(0.15,0.35))
        for k in acc: acc[k].append(m[k])
    res["v2x"]["realistic_lognormal+GE"]={k:ci(v) for k,v in acc.items()}
    c=res["v2x"]["realistic_lognormal+GE"]; print(f"V2X realistic(lognormal+GE bursty): coll/1k={c['coll'][0]:.3f} rate_err={c['rate_err'][0]:.4f} under={c['under_max'][0]:.4f}")
    res["seeds"]=a.seeds
    res["score"]="eight-feature" if SCORE=="8" else "four-feature"
    out="results_real/closed_loop.json" if SCORE=="8" else "results_real/closed_loop_4feat.json"
    json.dump(res,open(out,"w"),indent=2); print("saved",out)
