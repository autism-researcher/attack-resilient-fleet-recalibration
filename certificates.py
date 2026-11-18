#!/usr/bin/env python3
"""
Authoritative finite-sample certificate constants for the fleet-recalibration paper.
Resolves reviewer objections (1) binding LOWER-tail variance and (2) invalid
deterministic score-grid argument.  See PROOF notes below.  Every number is
computed from committed pools / generators -- nothing is hand-set.

delta (explicit, conservative; certified for the conditionally-i.i.d. window regime, Bernstein):
    delta = sqrt(2 v log(2/alpha)/Ns) + (2/3) log(2/alpha)/Ns ,
    v = v(delta) = max{(p-delta)(1-(p-delta)),(p+delta)(1-(p+delta))}
                 = (p-delta)(1-(p-delta))   for p>=1/2   (BINDING LOWER TAIL)
solved as a fixed point.  Per-step alpha=0.05; simultaneous alpha->alpha/(K T).

Adapted (closed-loop) regime: NOT certified.  A Freedman argument would evaluate the
empirical CDF at the bracketing levels x=Fbar^{-1}(p+-delta), but under closed-loop
feedback these levels depend on the terminal window law, which is not predictable
(see Supplementary S8, Remark).  The "closed" row below is therefore a reference
number only; the manuscript reports closed-loop under-protection as empirical.
Simultaneous certificates use K=20, T=1700 (T_DEFAULT) and zeta=0.05, i.e.
alpha=zeta/(K T) ~ 1.5e-6, for the real and controlled laws.

omega_F computed EXACTLY from each realized law via the generalized inverse.
Envelope = m + omega_F(delta) + rho, m=0.02, rho=0.01.
"""
import numpy as np, json, math

P, NS, M, RHO, ALPHA = 0.9, 300, 0.02, 0.01, 0.05

def v_binding(delta, p=P):
    return max((p-delta)*(1-(p-delta)), (p+delta)*(1-(p+delta)))

def delta_fp(alpha, Ns=NS, p=P, it=300):
    L = math.log(2.0/alpha); d = 0.05
    for _ in range(it):
        d2 = math.sqrt(2*v_binding(d,p)*L/Ns) + (2.0/3.0)*L/Ns
        if abs(d2-d) < 1e-13: return d2
        d = d2
    return d

def geninv(s, q):
    n=len(s); idx=min(max(int(math.ceil(q*n))-1,0),n-1); return float(s[idx])

def omega(a, delta, p=P):
    s=np.sort(np.asarray(a).ravel()); lo=max(p-delta,0.0); hi=min(p+delta,1.0)
    return max(geninv(s,hi)-geninv(s,p), geninv(s,p)-geninv(s,lo))

def env(a, delta): return M + omega(a, delta) + RHO

# ---------- reconstruct the four realized laws from committed code ----------
laws, KT = {}, {}
for f in ["highd","ngsim","waymo"]:
    laws[f]=np.load(f+".npy").ravel(); KT[f]=20*1700   # real cross-dataset: K=20, T=T_DEFAULT

# controlled generator: POST-shift faced streams pooled over a fixed 30-seed set
import experiments as E
from data import RegimeGenerator
KctrL=20; ctrl=[]
for sd in range(7,37):
    rm=np.random.default_rng(sd)
    gen=RegimeGenerator(seed=int(rm.integers(1e9))); post=E.assign_post(KctrL, True)
    R,Bs,st,tc=E.build_fleet(KctrL, E.T_DEFAULT, "freeflow", post, gen, E.T0, E.STAGGER)
    ctrl.append(R[:, tc:].ravel())
laws["controlled"]=np.concatenate(ctrl); KT["controlled"]=KctrL*E.T_DEFAULT

# closed loop: post-shift censored score law pooled over a fixed 30-seed set
import closed_loop as CL
Bpre=CL.bstar("mild",np.random.default_rng(1)); Bpost=CL.bstar("aggr",np.random.default_rng(2))
cl=[]
for sd in range(500,530):
    r=np.random.default_rng(sd)
    leads,shift=CL.build_leads(20,3000,3000//3,3000//4,r)
    m=CL.simulate("P5",leads,shift,r,Bpre,Bpost,np.full(20,Bpre),lat_ln=(1.1,0.8),ge=(0.15,0.35),ret_scores=True)
    cl.append(np.asarray(m["scores_post"]).ravel())
laws["closed"]=np.concatenate(cl); KT["closed"]=20*3000

d_step=delta_fp(ALPHA)
print(f"delta_step (alpha=0.05, binding v={v_binding(d_step):.4f}) = {d_step:.4f}   [old wrong=0.055, v=0.09]")
print(f"{'law':>11} {'n':>7} {'d_sim':>7} | {'omega_step':>10} {'ENV_step':>9} | {'omega_sim':>9} {'ENV_sim':>8}")
res={"delta_step":d_step,"v_binding_step":v_binding(d_step),"laws":{}}
for k,a in laws.items():
    d_sim=delta_fp(ALPHA/KT[k])
    e_s, e_m = env(a,d_step), env(a,d_sim)
    print(f"{k:>11} {a.size:>7} {d_sim:>7.4f} | {omega(a,d_step):>10.4f} {e_s:>9.4f} | {omega(a,d_sim):>9.4f} {e_m:>8.4f}")
    res["laws"][k]=dict(n=int(a.size),KT=int(KT[k]),delta_sim=d_sim,
                        env_step=round(e_s,4),env_sim=round(e_m,4),
                        omega_step=round(omega(a,d_step),4),omega_sim=round(omega(a,d_sim),4))
json.dump(res,open("results_real/certificates.json","w"),indent=2)
print("\nsaved results_real/certificates.json")
