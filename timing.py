"""M-4 reviewer ask: measured wall-clock runtime of the per-vehicle supervisor
and the cloud aggregation, to back the O(1)/O(log Ns)/O(K) analysis."""
import numpy as np, time, json
rng=np.random.default_rng(0)
TAU,GAMMA,NS,M,H=0.10,0.05,300,0.02,0.03
def timeit(fn,reps):
    fn(); t=time.perf_counter()
    for _ in range(reps): fn()
    return (time.perf_counter()-t)/reps

# --- per-vehicle per-step supervisor cost (fast update + local clamp), single vehicle ---
B=np.array([0.5]); U=np.array([0.5]); R=np.array([0.4])
def step_single():
    Beff=np.minimum(B,U+M); eng=R>Beff
    B[:]=np.clip(B+GAMMA*(eng.astype(float)-TAU),0,1)
t_step=timeit(step_single,200000)

# --- slow quantile recompute over the window (per recompute) ---
win=rng.random((1,NS))
def quant(): return np.quantile(win,1-TAU,axis=1)
t_quant=timeit(quant,50000)

# --- cloud aggregation: shift-aware weighted quantile, vs fleet size K ---
def agg_cost(K,reps=3000):
    Ui=rng.random(K); neff=np.full(K,float(NS)); Uq=Ui.copy()
    def agg():
        W=neff[None,:]*np.exp(-((Ui[None,:]-Uq[:,None])**2)/(2*H**2)); W=W/W.sum(1,keepdims=True)
        return (W*Ui[None,:]).sum(1)
    return timeit(agg,reps)
agg={K:agg_cost(K) for K in [10,100,1000]}

res={"per_vehicle_step_us":t_step*1e6,
     "slow_quantile_recompute_us":t_quant*1e6,
     "aggregation_us":{str(k):v*1e6 for k,v in agg.items()}}
print(f"per-vehicle supervisor step (fast update + local clamp): {t_step*1e6:.3f} us")
print(f"slow (1-tau)-quantile recompute over Ns={NS}: {t_quant*1e6:.2f} us  (amortized /5 steps = {t_quant*1e6/5:.2f} us/step)")
for k,v in agg.items(): print(f"cloud aggregation K={k:>4}: {v*1e6:.2f} us")
amort=t_step*1e6 + t_quant*1e6/5
print(f"==> total per-vehicle per-step ~ {amort:.2f} us, i.e. ~{amort/1000:.4f} ms (control cycle 10-100 ms)")
json.dump(res,open("results_real/timing.json","w"),indent=2)
