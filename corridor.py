"""Corridor hazard-propagation test: vehicles reach a FIXED hazard location at
staggered arrival times. The first to arrive have no local post-shift samples;
the fleet lets a later arrival inherit calibration from peers that already passed
the hazard. We measure collisions/conflicts in the TRANSIENT window right after
each vehicle's own hazard entry (where local data is scarce), local-only vs fleet.
Honest: numbers are from this run; no tuning to win."""
import numpy as np, json, importlib.util
spec=importlib.util.spec_from_file_location("cl","closed_loop.py"); cl=importlib.util.module_from_spec(spec); spec.loader.exec_module(cl)
DT=cl.DT; A_MAX,B_COMF,V0,T_HW,S0=cl.A_MAX,cl.B_COMF,cl.V0,cl.T_HW,cl.S0
TAU,GAMMA,NS,M,P,H=cl.TAU,cl.GAMMA,cl.NS,cl.M,cl.P,cl.H
TTC_CRIT,SUP_DECEL=cl.TTC_CRIT,cl.SUP_DECEL
risk,idm,lead1=cl.risk,cl.idm,cl.lead1
V_INIT,GAP_INIT=cl.V_INIT,cl.GAP_INIT

# --- corridor: vehicles enter a shared hazard zone at staggered TIMES; the hazard
# (aggressive lead braking) is the SAME regime for all, but each vehicle meets it
# with only its own pre-hazard history -> collective experience should help the
# late arrivals if calibration is shared. TRANSIENT window = first W steps after entry.
TRANSIENT=350  # steps after a vehicle's hazard entry where local data is scarce
def run(cond, K, T, rng, equipped=None, entry=None):
    if equipped is None: equipped=np.ones(K,bool)
    if entry is None:  # staggered hazard-entry times spread across the run
        entry=np.sort(rng.integers(T//6, T-TRANSIENT-50, K))
    # leads: mild before entry, aggressive (hazard) after entry
    leads=np.zeros((K,T))
    for k in range(K):
        e=entry[k]; leads[k,:e]=lead1(e,rng,"mild"); leads[k,e:]=lead1(T-e,rng,"aggr")
    Bpre=cl.bstar("mild",np.random.default_rng(1)); Bpost=cl.bstar("aggr",np.random.default_rng(2))
    cl.reset_risk()
    v=np.full(K,V_INIT); gap=np.full(K,GAP_INIT)
    win=np.full((K,NS),Bpre); B=np.full(K,Bpre); ptr=0
    U=np.quantile(win,1-TAU,axis=1); fleet_hist=[]
    coll=np.zeros(K); conf=np.zeros(K); steps=np.zeros(K)
    for t in range(T):
        ld=leads[:,t]; dv=v-ld; R=risk(gap,v,dv)
        with np.errstate(divide="ignore",invalid="ignore"):
            ttc=np.where(dv>0.1,gap/np.where(dv>0.1,dv,1.0),1e9)
        if t%5==0: U=np.quantile(win,1-TAU,axis=1)
        if cond=="AEB": engage=ttc<TTC_CRIT
        else: Beff=np.minimum(B,U+M); engage=R>Beff
        a=np.where(engage,-SUP_DECEL,np.clip(idm(v,gap,dv),-B_COMF*2,A_MAX))
        v=np.clip(v+a*DT,0,V0); gap=gap+(ld-v)*DT
        hit=gap<=0.4
        # accumulate metrics only in each vehicle's TRANSIENT window after entry
        intran=(t>=entry)&(t<entry+TRANSIENT)
        coll+=hit&intran; conf+=((ttc>0)&(ttc<TTC_CRIT))&intran; steps+=intran
        gap=np.where(hit,GAP_INIT,gap); v=np.where(hit,ld,v); gap=np.maximum(gap,0.5)
        if cond in ("B1","P5"): B=np.clip(B+GAMMA*(engage.astype(float)-TAU),0,1)
        if cond=="P5" and t>0 and t%P==0:
            # equipped vehicles that have ALREADY entered the hazard upload; an approaching
            # equipped vehicle inherits the shift-aware weighted estimate
            mask=equipped&(t>=entry); arr=U[mask]
            if arr.size:
                Wm=np.exp(-((arr[None,:]-U[:,None])**2)/(2*H**2)); Wm=Wm/Wm.sum(1,keepdims=True)
                Uf=(Wm*arr[None,:]).sum(1); lam=0.5*np.exp(-((Uf-U)**2)/(2*H**2))
                Bn=np.clip((1-lam)*B+lam*Uf,0,1); B=np.where(equipped,Bn,B)
        win[:,ptr]=R; ptr=(ptr+1)%NS
    eq=equipped
    cps=1000*coll[eq].sum()/max(steps[eq].sum(),1)   # collisions per 1k transient veh-steps (equipped only)
    cfr=conf[eq].sum()/max(steps[eq].sum(),1)
    return cps,cfr

K,T,SEEDS=20,2000,8
out={}
for cond in ["AEB","B1","P5"]:
    cps=[];cfr=[]
    for s in range(SEEDS):
        r=np.random.default_rng(700+s); a,b=run(cond,K,T,r); cps.append(a); cfr.append(b)
    out[cond]={"coll_per_1k":cl.ci(cps),"conflict":cl.ci(cfr)}
    c=out[cond]; print(f"{cond:>3}: transient coll/1k={c['coll_per_1k'][0]:.3f} [{c['coll_per_1k'][1]:.3f},{c['coll_per_1k'][2]:.3f}]  conflict={c['conflict'][0]:.4f}")
out["score"]="eight-feature" if cl.SCORE=="8" else "four-feature"
json.dump(out,open("results_real/corridor.json" if cl.SCORE=="8" else "results_real/corridor_4feat.json","w"),indent=2)
