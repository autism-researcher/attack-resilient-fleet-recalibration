"""M-4: risk-score agnosticism. Same kinematics, three different risk scores
(composite 8-feature; PCM-style criticality; MPrISM-style MPC safety). For each,
recalibration must hold the target rate and keep under-protection within bound."""
import numpy as np, json
from data import RegimeGenerator, REGIMES
from risk_score import risk_score
from recalibration import Params, simulate

TAU=0.10; K=20; T=1500; T0=400; STAGGER=450; SEEDS=8
HETERO=["congested","rain","dense","night","moderate"]; PRE="freeflow"
ADV_FRAC=0.30; ADV_OFF=0.25

def _sat(x): return x/(1.0+x)
# alternative scores over kinematics -> [0,1], monotone in danger
def score_composite(gap,v,dv,jerk): return risk_score(gap,v,dv,jerk=jerk)
def score_pcm(gap,v,dv,jerk):
    gap=np.maximum(gap,0.1); appr=np.maximum(dv,0.0)
    a_req=appr**2/(2*gap)              # required decel (DRAC); PEGASUS-style criticality
    return np.clip(_sat(a_req/6.0),0,1)
def score_mprism(gap,v,dv,jerk):
    gap=np.maximum(gap,0.1); appr=np.maximum(dv,0.0); d_safe=2.0
    b=appr**2/(2*np.maximum(gap-d_safe,0.3))   # MPC braking to avoid collision over short horizon
    return np.clip(_sat(b/8.0),0,1)
SCORES={"composite":score_composite,"PCM":score_pcm,"MPrISM":score_mprism}

def bstar(score,regime,rng,n=200000):
    g=RegimeGenerator(seed=rng); gap,v,dv,jk=g.sample_kinematics(regime,n)
    return float(np.quantile(score(gap,v,dv,jk),1-TAU))

def build(score,gen,rng):
    shift=(T0+rng.integers(0,STAGGER+1,K)).astype(int)
    post=[HETERO[i%len(HETERO)] for i in range(K)]
    bpre=bstar(score,PRE,int(rng.integers(1e9)))
    bp={r:bstar(score,r,int(rng.integers(1e9))) for r in set(post)}
    R=np.zeros((K,T)); Bs=np.zeros((K,T))
    for k in range(K):
        st=shift[k]; pr=post[k]
        gap,v,dv,jk=gen.sample_kinematics(PRE,st); R[k,:st]=score(gap,v,dv,jk)
        gap,v,dv,jk=gen.sample_kinematics(pr,T-st); R[k,st:]=score(gap,v,dv,jk)
        Bs[k,:st]=bpre; Bs[k,st:]=bp[pr]
    return R,Bs,shift,int(shift.min())

def under_max(log,shift):
    beff=log["beff"]; Bs=log["Bstar"]; K,T=beff.shape; ux=[]
    for k in range(K):
        u=np.maximum(beff[k,int(shift[k]):]-Bs[k,int(shift[k]):],0.0); ux.append(u.max())
    return float(np.mean(ux))
def rate_err(log,shift,p):
    inter=log["intervened"]; K,T=inter.shape; res=[]
    for k in range(K):
        st=int(shift[k]); steady=slice(min(st+350,T-1),T)
        w=100; c=np.cumsum(inter[k].astype(float)); 
        rate=np.array([ (c[t]-(c[t-w] if t>=w else 0))/min(t+1,w) for t in range(T)])
        res.append(np.mean(np.abs(rate[steady]-p.tau)))
    return float(np.mean(res))

p=Params()
out={}
adv=np.zeros(K,bool); adv[:int(ADV_FRAC*K)]=True
for name,score in SCORES.items():
    re_b1,re_p5,um_h,um_a=[],[],[],[]
    for s in range(SEEDS):
        rng=np.random.default_rng(1000+s); gen=RegimeGenerator(seed=int(rng.integers(1e9)))
        R,Bs,shift,tc=build(score,gen,rng)
        re_b1.append(rate_err(simulate("B1",R,Bs,tc,p,gen.rng),shift,p))
        lp=simulate("P5",R,Bs,tc,p,gen.rng); re_p5.append(rate_err(lp,shift,p)); um_h.append(under_max(lp,shift))
        la=simulate("P5",R,Bs,tc,p,gen.rng,adv_mask=adv,adv_offset=ADV_OFF); um_a.append(under_max(la,shift))
    # corrected per-step certificate (same as experiments.theoretical_bound)
    import experiments as _E
    bound=_E.theoretical_bound(R,tc,p)
    out[name]=dict(rate_err_B1=float(np.mean(re_b1)),rate_err_P5=float(np.mean(re_p5)),
                   under_honest=float(np.mean(um_h)),under_adv=float(np.mean(um_a)),bound=float(bound))
    o=out[name]
    print(f"{name:>9}: rate_err B1={o['rate_err_B1']:.4f} P5={o['rate_err_P5']:.4f} | under_max honest={o['under_honest']:.4f} adv(30%)={o['under_adv']:.4f} | bound={o['bound']:.4f}")
json.dump(out,open("results_real/agnostic.json","w"),indent=2); print("saved results_real/agnostic.json")
