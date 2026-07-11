#!/usr/bin/env python3
"""Extended closed-loop experiments for TR-C reviewer strengthening.
Reuses the EXACT primitives of closed_loop.py (risk, idm, build_leads, bstar,
module constants) so numbers are consistent with the paper's closed loop.
Adds: (a) ablation variants incl. B2 (clamp reads fleet, unsafe), robust-no-clamp,
clamp-only; (b) adversarial upload injection into the closed loop; (c) named V2X
fault grid (delay/loss/stale/Byzantine/mixed); (d) full traffic-performance
metrics; (e) sensitivity sweeps; (f) 30-seed statistics; (g) a combined
deployment scenario. All numbers are produced by this run - no hardcoding."""
import numpy as np, json, argparse, os
import closed_loop as CL

DT=CL.DT; TAU=CL.TAU; GAMMA=CL.GAMMA; NS=CL.NS; M=CL.M; P=CL.P; H=CL.H
SUP_DECEL=CL.SUP_DECEL; TTC_CRIT=CL.TTC_CRIT; V0=CL.V0; B_COMF=CL.B_COMF; A_MAX=CL.A_MAX
risk=CL.risk; idm=CL.idm; build_leads=CL.build_leads; bstar=CL.bstar
OUT=os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(OUT,"results_real"),exist_ok=True)

def simulate(cond, leads, shift, rng, Bpre, Bpost, B0c,
             latency=0, loss=0.0, lat_ln=None, ge=None,
             adv=0.0, stale=0, Ns=NS, m=M, tau=TAU, gamma=GAMMA, Pagg=P):
    """Generalized closed-loop simulate with adversarial injection + extra metrics.
    cond in: AEB, B0, B1, FedAvg, RobFCP, P5, B2 (clamp reads fleet, unsafe),
             RobNoClamp (robust agg as boundary, no clamp), ClampOnly (=FedAvg w/ clamp).
    adv: fraction of vehicles that upload a maliciously inflated boundary (=> under-protect).
    stale: if >0, fleet estimate is deliberately reused with this many extra steps of delay.
    Returns transportation + calibration + safety metrics (all from this run)."""
    K,T=leads.shape
    v=np.full(K,CL.V_INIT); gap=np.full(K,CL.GAP_INIT)
    win=np.full((K,Ns),Bpre); B=np.full(K,Bpre); ptr=0; pend=[]; ge_state=0
    U=np.quantile(win,1-tau,axis=1)
    n_adv=int(round(adv*K)); adv_mask=np.zeros(K,bool); adv_mask[:n_adv]=True
    # metric accumulators (per step, post-shift window filtered later)
    interv=np.zeros((K,T)); coll=0
    under=np.zeros(T); spd=np.zeros(T); conf=np.zeros(T)
    unnec=np.zeros(T); missed=np.zeros(T); hardb=np.zeros(T); thw_step=np.zeros(T)
    recent_engage=np.zeros(K)  # steps-since-last-intervention counter (for headway-after)
    thw_after_vals=[]
    for t in range(T):
        ld=leads[:,t]; dv=v-ld; R=risk(gap,v,dv)
        with np.errstate(divide="ignore",invalid="ignore"):
            ttc=np.where(dv>0.1,gap/np.where(dv>0.1,dv,1.0),1e9)
            thw=np.where(v>1e-2,gap/np.where(v>1e-2,v,1.0),1e9)
        Bstar_t=np.where(t>=shift,Bpost,Bpre)
        if t%5==0: U=np.quantile(win,1-tau,axis=1)
        # fleet estimate available to conditions that read it in the clamp
        Uf_now=float(np.median(U))
        if cond=="AEB":
            engage=ttc<TTC_CRIT; Beff=None
        elif cond=="B0":
            Beff=B0c; engage=R>Beff; under[t]=np.max(np.clip(Beff-Bstar_t,0,None))
        elif cond=="B2":   # naive: clamp reads the (adversarial) fleet estimate -> unsafe
            Uf_c=_fleet_clamp(U,adv_mask,H); Beff=np.minimum(B,Uf_c+m); engage=R>Beff
            under[t]=np.max(np.clip(Beff-Bstar_t,0,None))
        elif cond=="RobNoClamp":  # robust trimmed fleet boundary used directly, NO clamp
            Beff=np.full(K,_trimmed(U)); engage=R>Beff; under[t]=np.max(np.clip(Beff-Bstar_t,0,None))
        else:              # B1, FedAvg, RobFCP, P5, ClampOnly -> local clamp
            Beff=np.minimum(B,U+m); engage=R>Beff; under[t]=np.max(np.clip(Beff-Bstar_t,0,None))
        a=np.where(engage,-SUP_DECEL,np.clip(idm(v,gap,dv),-B_COMF*2,A_MAX))
        v=np.clip(v+a*DT,0,V0); gap=gap+(ld-v)*DT
        hit=gap<=0.4; coll+=int(hit.sum()); gap=np.where(hit,CL.GAP_INIT,gap); v=np.where(hit,ld,v); gap=np.maximum(gap,0.5)
        # --- metrics ---
        interv[:,t]=engage; spd[t]=np.mean(v); conf[t]=np.mean((ttc>0)&(ttc<TTC_CRIT))
        truly_high=R>Bstar_t
        unnec[t]=np.mean(engage & ~truly_high)     # intervened but not truly high-risk
        missed[t]=np.mean(truly_high & ~engage)    # truly high-risk but did not intervene
        hardb[t]=np.mean(a<=-3.0)                   # hard-braking events (a <= -3 m/s^2)
        thw_step[t]=float(np.median(np.clip(thw,0,10)))
        # headway a few steps after an intervention onset
        newly=engage & (recent_engage==0)
        recent_engage=np.where(engage,8.0,np.maximum(0,recent_engage-1))
        aft=recent_engage>0
        if aft.any(): thw_after_vals.append(float(np.mean(np.clip(thw[aft],0,10))))
        # fast update
        if cond in ("B1","P5","FedAvg","RobFCP","B2"): B=np.clip(B+gamma*(engage.astype(float)-tau),0,1)
        # fleet assimilation into fast update (accuracy channel)
        if cond=="P5" and t>0 and t%Pagg==0:
            drop=_drop(rng,loss,ge)
            if not drop:
                lat=_latency(rng,latency,lat_ln)+stale
                Uf=_fleet_agg(U,adv_mask,H,shift_aware=True)
                pend.append((t+lat,Uf))
            ready=[uf for(dd,uf)in pend if dd<=t]; pend=[(dd,uf)for(dd,uf)in pend if dd>t]
            if ready:
                Uf=ready[-1]; lam=0.5*np.exp(-((Uf-U)**2)/(2*H**2)); B=np.clip((1-lam)*B+lam*Uf,0,1)  # per-vehicle trust, matches closed_loop.py
        if cond in ("FedAvg","B2") and t>0 and t%Pagg==0:  # both assimilate unweighted fleet; they differ ONLY in the clamp
            Uv=_inject(U,adv_mask); Uf=float(np.mean(Uv)); B=np.clip(0.5*B+0.5*Uf,0,1)
        if cond=="RobFCP" and t>0 and t%Pagg==0:
            Uf=_trimmed(_inject(U,adv_mask)); B=np.clip(0.5*B+0.5*Uf,0,1)
        win[:,ptr]=R; ptr=(ptr+1)%Ns
    post=slice(int(shift.max())+200,T)
    rate=np.array([np.mean(interv[k,post]) for k in range(K)])
    veh_steps=K*(T-int(shift.max()))/1000.0
    out=dict(coll=coll/max(veh_steps,1e-9), conflict=float(np.mean(conf[post])),
             rate_err=float(np.mean(np.abs(rate-tau))),
             under_max=float(np.max(under[post])) if cond!="AEB" else np.nan,
             speed=float(np.mean(spd[post])),
             unnec_interv=float(np.mean(unnec[post])),
             missed_highrisk=float(np.mean(missed[post])),
             hardbrake_freq=float(np.mean(hardb[post])),
             speed_disturb=float(np.std(spd[post])),
             thw_post=float(np.mean(thw_step[post])),
             thw_after_interv=float(np.mean(thw_after_vals)) if thw_after_vals else np.nan)
    return out

def _inject(U, adv_mask):
    """Malicious vehicles report an artificially inflated boundary (claim low risk),
    which - if trusted in the clamp - would raise the effective boundary and under-protect."""
    Uv=U.copy()
    if adv_mask.any(): Uv[adv_mask]=min(1.0, float(np.max(U))+0.30)
    return Uv

def _fleet_agg(U, adv_mask, h, shift_aware=True):
    Uv=_inject(U,adv_mask)
    if shift_aware:
        ws=np.exp(-((Uv-np.median(U))**2)/(2*h**2)); ws=ws/ws.sum(); return float(np.sum(ws*Uv))
    return float(np.mean(Uv))

def _fleet_clamp(U, adv_mask, h):
    """Naive B2: UNWEIGHTED fleet estimate used inside the clamp -> exposed to attack
    (no shift-aware defense, matching the paper's B2 'naive fleet-clamp')."""
    return float(np.mean(_inject(U, adv_mask)))

def _trimmed(U, frac=0.15):
    u=np.sort(U); kk=int(frac*len(u))
    return float(np.mean(u[kk:len(u)-kk])) if len(u)>2*kk else float(np.mean(u))

def _drop(rng, loss, ge):
    if ge is not None:
        # Gilbert-Elliott handled by caller state; approximate with bad-state prob
        return rng.random()<ge[0]
    return rng.random()<loss

def _latency(rng, latency, lat_ln):
    return latency if lat_ln is None else max(0,int(round(rng.lognormal(lat_ln[0],lat_ln[1]))))

def ci(vals):
    v=np.array([x for x in vals if not(isinstance(x,float) and np.isnan(x))])
    if len(v)<2: return dict(mean=float(v.mean()) if len(v) else float("nan"),std=0.0,lo=float("nan"),hi=float("nan"),n=len(v))
    rng=np.random.default_rng(0); bs=[v[rng.integers(0,len(v),len(v))].mean() for _ in range(3000)]
    return dict(mean=float(v.mean()),std=float(v.std(ddof=1)),
                lo=float(np.percentile(bs,2.5)),hi=float(np.percentile(bs,97.5)),n=int(len(v)))

def run_conditions(conds, seeds, K=20, T=3000, seed0=100, **kw):
    Bpre=bstar("mild",np.random.default_rng(1)); Bpost=bstar("aggr",np.random.default_rng(2))
    B0c=np.full(K,Bpre); res={}
    metrics=["coll","conflict","rate_err","under_max","speed","unnec_interv",
             "missed_highrisk","hardbrake_freq","speed_disturb","thw_post","thw_after_interv"]
    raw={c:{m:[] for m in metrics} for c in conds}
    for c in conds:
        for s in range(seeds):
            r=np.random.default_rng(seed0+s); leads,shift=build_leads(K,T,T//3,T//4,r)
            mres=simulate(c,leads,shift,r,Bpre,Bpost,B0c,**kw)
   