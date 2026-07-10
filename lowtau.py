"""Low-tau (rare-event) demonstration. A deployed last-line guard targets a much
lower intervention rate than tau=0.10. As tau falls the operating quantile moves
into the tail where the density f_min shrinks, so the finite-sample term of the
bound grows; a larger window N_s compensates. We quantify this tradeoff honestly.
Honest: numbers are from this run."""
import numpy as np, json, warnings
warnings.filterwarnings("ignore")
from data import RegimeGenerator
from recalibration import Params, simulate
import experiments as E

K=20; T=1700; SEEDS=10
def run(tau, Ns, seeds=SEEDS, seed0=71):
    p=Params(tau=tau, N_s=Ns); rm=np.random.default_rng(seed0)
    re=[]; ux=[]; bnd=[]
    for s in range(seeds):
        gen=RegimeGenerator(seed=int(rm.integers(1e9))); post=E.assign_post(K,True)
        R,Bs,st,tc=E.build_fleet(K,T,"freeflow",post,gen,E.T0,E.STAGGER)
        m=E.metrics(simulate("P5",R,Bs,tc,p,gen.rng),st,p)
        re.append(np.mean(m["rate_err"])); ux.append(np.mean(m["under_max"]))
        bnd.append(E.theoretical_bound(R,tc,p))
    return float(np.mean(re)), float(np.mean(ux)), float(np.mean(bnd))

print("tau    Ns    rate_err(P5)   under_max(P5)   proven_bound   within?")
out={}
# (a) tau sweep at fixed Ns=300; (b) low tau with enlarged Ns to recover tightness
configs=[(0.10,300),(0.05,300),(0.02,300),(0.01,300),(0.01,1200),(0.005,2400)]
for tau,Ns in configs:
    re,ux,bnd=run(tau,Ns)
    ok = ux<=bnd
    out[f"tau{tau}_Ns{Ns}"]=dict(tau=tau,Ns=Ns,rate_err=re,under_max=ux,bound=bnd,within=bool(ok))
    print(f"{tau:<6} {Ns:<5} {re:.4f}        {ux:.4f}         {bnd:.4f}      {'yes' if ok else 'NO'}")
json.dump(out,open("results_real/lowtau.json","w"),indent=2); print("saved lowtau.json")
