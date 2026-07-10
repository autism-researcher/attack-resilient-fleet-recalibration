"""M-5 parameter sensitivity (gamma, N_s, m) + M-6 bootstrap hypothesis tests."""
import numpy as np, json
from data import RegimeGenerator
from recalibration import Params, simulate
import experiments as E

TAU=0.10; K=20; T=1500; SEEDS=10
def run(p, seeds=SEEDS, seed0=7, adv=False):
    rm=np.random.default_rng(seed0); reB1=[]; reP5=[]; uxP5=[]
    am=np.zeros(K,bool); am[:int(0.30*K)]=True
    for s in range(seeds):
        gen=RegimeGenerator(seed=int(rm.integers(1e9))); post=E.assign_post(K,True)
        R,Bs,st,tc=E.build_fleet(K,T,"freeflow",post,gen,E.T0,E.STAGGER)
        reB1.append(np.mean(E.metrics(simulate("B1",R,Bs,tc,p,gen.rng),st,p)["rate_err"]))
        mp=E.metrics(simulate("P5",R,Bs,tc,p,gen.rng,adv_mask=am if adv else None,adv_offset=0.25 if adv else 0.0),st,p)
        reP5.append(np.mean(mp["rate_err"])); uxP5.append(np.mean(mp["under_max"]))
    return np.array(reB1),np.array(reP5),np.array(uxP5)

# ---- M-5 sensitivity: vary one param at a time around defaults (gamma .05, N_s 300, m .02)
base=dict(gamma=0.05,N_s=300,m=0.02)
grid={"gamma":[0.02,0.05,0.10],"N_s":[150,300,600],"m":[0.01,0.02,0.05]}
sens={}
for name,vals in grid.items():
    sens[name]={}
    for val in vals:
        kw=dict(base); kw[name]=val; p=Params(tau=TAU,**kw)
        _,reP5,uxP5=run(p)
        sens[name][str(val)]=dict(rate_err_P5=float(reP5.mean()),under_max_P5=float(uxP5.mean()))
        print(f"{name}={val}: rate_err_P5={reP5.mean():.4f}  under_max_P5={uxP5.mean():.4f}")
json.dump(sens,open("results_real/sensitivity.json","w"),indent=2)

# ---- M-6 hypothesis tests (paired bootstrap) on headline contrasts at default params
p=Params(tau=TAU,**base)
reB1,reP5,uxP5=run(p)                      # P5 vs B1 rate error (paired)
_,reP5a,uxP5a=run(p,adv=True)             # P5 under-protection: honest vs 30% adversarial (paired)
def paired_boot_p(a,b,nb=20000,seed=0):
    # H0: mean(a-b)=0; two-sided bootstrap p on the paired difference
    d=np.asarray(a)-np.asarray(b); rng=np.random.default_rng(seed); n=len(d)
    dm=d.mean(); samp=np.array([d[rng.integers(0,n,n)].mean() for _ in range(nb)])
    # center under H0 and count |centered| >= |observed|
    c=samp-dm; pval=float((np.abs(c)>=abs(dm)).mean()); 
    return dm, max(pval,1.0/nb)
d1,p1=paired_boot_p(reB1,reP5)            # is P5 rate-err different from B1?
d2,p2=paired_boot_p(uxP5a,uxP5)           # is adversarial under-prot != honest? (expect ~0 diff -> NOT sig)
tests={"P5_vs_B1_rate_err":{"mean_diff_B1_minus_P5":float((reB1-reP5).mean()),"p":p1},
       "P5_under_adv_minus_honest":{"mean_diff":float(d2),"p":p2}}
json.dump(tests,open("results_real/htests.json","w"),indent=2)
print(f"\n[M6] P5 vs B1 rate-err: mean(B1-P5)={(reB1-reP5).mean():+.4f}, paired-bootstrap p={p1:.4f}")
print(f"[M6] P5 under-prot adv - honest: mean diff={d2:+.4f}, p={p2:.4f} (expect non-sig: safety invariant to adversary)")
