#!/usr/bin/env python3
"""Corrected per-step certificates for Experiment O (rare-event, low tau),
using the binding-tail-variance delta and exact omega_F on the controlled law.
Observed under_max values are the real committed lowtau.json runs (unchanged)."""
import numpy as np, math, json
from data import RegimeGenerator
import experiments as E

M,RHO,ALPHA=0.02,0.01,0.05
def v_binding(delta,p):
    return max((p-delta)*(1-p+delta),(p+delta)*(1-p-delta)) if p>=0.5 else max((p-delta)*(1-p+delta),(p+delta)*(1-p-delta))
def delta_fp(alpha,Ns,p,it=300):
    L=math.log(2/alpha); d=0.05
    for _ in range(it):
        d2=math.sqrt(2*max(v_binding(d,p),1e-9)*L/Ns)+(2/3)*L/Ns
        if abs(d2-d)<1e-13: return d2
        d=d2
    return d
def geninv(s,q):
    n=len(s); i=min(max(int(math.ceil(q*n))-1,0),n-1); return float(s[i])
def omega(a,d,p):
    s=np.sort(a); return max(geninv(s,min(p+d,1))-geninv(s,p), geninv(s,p)-geninv(s,max(p-d,0)))

# pooled controlled post-shift law (fixed seed set)
ctrl=[]
rm=np.random.default_rng(71)
for _ in range(30):
    gen=RegimeGenerator(seed=int(rm.integers(1e9))); post=E.assign_post(20,True)
    R,Bs,st,tc=E.build_fleet(20,1700,"freeflow",post,gen,E.T0,E.STAGGER)
    ctrl.append(R[:,tc:].ravel())
law=np.concatenate(ctrl)

obs={(0.10,300):0.032,(0.05,300):0.073,(0.02,300):0.123,(0.01,300):0.161,(0.01,1200):0.126,(0.005,2400):0.127}
print(f"{'tau':>6}{'Ns':>6}{'p':>7}{'delta':>8}{'omega':>8}{'cert':>8}{'obs':>7}  within")
res={}
for (tau,Ns),o in obs.items():
    p=1-tau; d=delta_fp(ALPHA,Ns,p); w=omega(law,d,p); cert=M+w+RHO
    print(f"{tau:>6}{Ns:>6}{p:>7.3f}{d:>8.4f}{w:>8.4f}{cert:>8.4f}{o:>7.3f}  {o<=cert}")
    res[f"tau{tau}_Ns{Ns}"]=dict(tau=tau,Ns=Ns,delta=round(d,4),cert=round(cert,4),obs=o,within=bool(o<=cert))
json.dump(res,open("results_real/exp_o_certificates.json","w"),indent=2)
print("saved results_real/exp_o_certificates.json")
