"""Transportation scenarios for the closed loop: stop-and-go congestion waves and
sudden cut-ins. Compares the fixed-TTC AEB baseline against the calibrated
supervisor (local B1 and fleet P5). Honest: numbers are from this run."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, json
import closed_loop as cl

def stopgo(T,rng):
    t=np.arange(T); v=15+9*np.sin(2*np.pi*t/180.0+rng.uniform(0,6))
    return np.clip(v,3.0,30.0)
def cutin(T,rng):
    v=np.full(T,24.0); t=0
    while t<T:
        t+=int(rng.exponential(55))
        if t<T: v[t:t+5]=np.maximum(3.0,24.0-18.0)   # sudden deep speed drop (cut-in proxy)
    return np.clip(v,2.0,30.0)


def merge(T,rng):
    # a slower vehicle merges in: periodic moderate speed drops to ~15 m/s
    v=np.full(T,26.0); t=0
    while t<T:
        t+=int(rng.exponential(40))
        if t<T: v[t:t+int(rng.integers(8,18))]=np.maximum(8.0,26.0-11.0)
    return np.clip(v,3.0,30.0)
def emergency(T,rng):
    # abrupt hard emergency braking to near-stop
    v=np.full(T,26.0); t=0
    while t<T:
        t+=int(rng.exponential(70))
        if t<T: v[t:t+int(rng.integers(5,12))]=np.maximum(1.0,26.0-23.0)
    return np.clip(v,1.0,30.0)

def bstar_lead(fn,rng,n=15000):
    lead=fn(n,rng); v=cl.V_INIT; gap=cl.GAP_INIT; Rs=np.empty(n)
    for t in range(n):
        dv=v-lead[t]; Rs[t]=cl.risk(gap,v,dv)
        a=np.clip(cl.idm(v,gap,dv),-cl.B_COMF*2,cl.A_MAX); v=max(0,v+a*cl.DT); gap=max(0.5,gap+(lead[t]-v)*cl.DT)
        if gap<=0.6: gap=cl.GAP_INIT; v=lead[t]
    return float(np.quantile(Rs,1-cl.TAU))

K,T,SEEDS=20,2000,8
res={}
for name,fn in [("cutin",cutin),("merge",merge),("emergency",emergency),("stopgo",stopgo)]:
    Bp=bstar_lead(fn,np.random.default_rng(3)); B0c=np.full(K,Bp); res[name]={}
    for cond in ["AEB","B1","P5"]:
        coll=[]
        for s in range(SEEDS):
            r=np.random.default_rng(300+s)
            leads=np.stack([fn(T,np.random.default_rng(9000+s*30+k)) for k in range(K)])
            shift=np.zeros(K,int)
            coll.append(cl.simulate(cond,leads,shift,r,Bp,Bp,B0c)["coll"])
        res[name][cond]=cl.ci(coll)
        print(f"{name:>7} {cond:>4}: coll/1k={res[name][cond][0]:.3f} {[round(x,3) for x in res[name][cond][1:]]}")
json.dump(res,open("results_real/scenarios.json","w"),indent=2); print("saved")
