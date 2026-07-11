"""Experiments A-D for Paper P5: bootstrap CIs, safety outcomes, figures."""
import os, json, argparse, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import RegimeGenerator
from recalibration import Params, simulate

OUT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
RES = os.path.join(OUT, "results"); os.makedirs(RES, exist_ok=True)

params_tau = 0.10
T_DEFAULT = 1700; T0 = 450; STAGGER = 500
CONDS = ["B0", "B1", "B2", "P5"]
FLEETS_B = [1, 5, 20, 100]
FLEETS_D = [1, 2, 3, 5, 10, 20, 50, 100]
HETERO_POST = ["congested", "rain", "dense", "night", "moderate"]
ADV_FRAC = 0.30; ADV_OFFSET = 0.25

_BSTAR = {}
def bstar_of(regime):
    if regime not in _BSTAR:
        _BSTAR[regime] = RegimeGenerator(seed=12345).true_matching_boundary(regime, params_tau, n=300000)
    return _BSTAR[regime]

def assign_post(K, hetero):
    return [HETERO_POST[i % len(HETERO_POST)] for i in range(K)] if hetero else ["congested"] * K

def build_fleet(K, T, pre, post_regimes, gen, t0, stagger):
    shift = (t0 + gen.rng.integers(0, stagger + 1, size=K)).astype(int)
    bpre = bstar_of(pre); bpost = {r: bstar_of(r) for r in set(post_regimes)}
    R = np.zeros((K, T)); Bstar = np.zeros((K, T))
    for k in range(K):
        st = shift[k]; pr = post_regimes[k]
        R[k, :st] = gen.stream(pre, st); R[k, st:] = gen.stream(pr, T - st)
        Bstar[k, :st] = bpre; Bstar[k, st:] = bpost[pr]
    return R, Bstar, shift, int(shift.min())

def moving_rate(intervened, w=100):
    K, T = intervened.shape
    c = np.cumsum(intervened.astype(float), axis=1)
    cprev = np.zeros_like(c); cprev[:, w:] = c[:, :-w]
    counts = np.minimum(np.arange(1, T + 1), w)
    return (c - cprev) / counts

def metrics(log, shift, p):
    inter = log["intervened"]; beff = log["beff"]; Bstar = log["Bstar"]; R = log["R"]
    K, T = inter.shape; rate = moving_rate(inter, 100)
    re = np.zeros(K); lat = np.zeros(K); um = np.zeros(K); ux = np.zeros(K); ms = np.zeros(K)
    for k in range(K):
        st = int(shift[k]); post = slice(st, T); steady = slice(min(st + 350, T - 1), T)
        re[k] = np.mean(np.abs(rate[k, steady] - p.tau))
        L = T - st
        for t in range(st, T):
            if np.all(np.abs(rate[k, t:min(t + 50, T)] - p.tau) < p.eps): L = t - st; break
        lat[k] = L
        u = np.maximum(beff[k, post] - Bstar[k, post], 0.0); um[k] = u.mean(); ux[k] = u.max()
        hi = R[k, post] > Bstar[k, post]; ms[k] = (hi & (~inter[k, post])).sum() / max(hi.sum(), 1)
    return dict(rate_err=re, latency=lat, under_mean=um, under_max=ux, missed=ms)

def boot_ci(x, nboot=2000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x, float)
    bs = [rng.choice(x, len(x), replace=True).mean() for _ in range(nboot)]
    return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

def theoretical_bound(R, t_cal, p, alpha=0.05, rho=0.01):
    # Corrected per-step certificate: m + omega_F(delta) + rho.
    # delta = binding-tail two-sided Bernstein fixed point (lower tail q=1-tau-delta
    # is binding for 1-tau>=1/2, so variance = (P-d)(1-P+d), NOT P(1-P)).
    # omega_F = exact quantile modulus of the (post-shift) faced law at level P=1-tau.
    import math
    P = 1 - p.tau; L = math.log(2.0 / alpha); Ns = p.N_s
    d = 0.05
    for _ in range(300):
        v = max((P - d) * (1 - P + d), (P + d) * (1 - P - d))
        d2 = math.sqrt(2 * v * L / Ns) + (2.0 / 3.0) * L / Ns
        if abs(d2 - d) < 1e-13:
            d = d2; break
        d = d2
    law = np.sort(R[:, t_cal:].ravel())  # post-shift faced law
    n = len(law)
    def gi(qq): return law[min(max(int(math.ceil(qq * n)) - 1, 0), n - 1)]
    omega = max(gi(min(P + d, 1.0)) - gi(P), gi(P) - gi(max(P - d, 0.0)))
    return float(p.m + omega + rho)

def _save(n, o):
    with open(os.path.join(RES, f"{n}.json"), "w") as f: json.dump(o, f, indent=2)

def expA(seeds, T=T_DEFAULT, seed0=7):
    p = Params(); rm = np.random.default_rng(seed0); K = 20
    acc = {c: {m: [] for m in ["rate_err","missed","under_mean","under_max"]} for c in CONDS}; bnd = []
    for s in range(seeds):
        gen = RegimeGenerator(seed=int(rm.integers(1e9))); post = assign_post(K, True)
        R, Bs, st, tc = build_fleet(K, T, "freeflow", post, gen, T0, STAGGER); bnd.append(theoretical_bound(R, tc, p))
        for c in CONDS:
            mt = metrics(simulate(c, R, Bs, tc, p, gen.rng), st, p)
            for key in acc[c]: acc[c][key].append(np.mean(mt[key]))
    out = {c: {k: boot_ci(v) for k, v in acc[c].items()} for c in CONDS}; out["theoretical_bound"] = boot_ci(bnd)
    _save("A", out); _figA(out); return out

def expB(seeds, T=T_DEFAULT, seed0=17):
    p = Params(); rm = np.random.default_rng(seed0); lat = {"B1": {}, "P5": {}}
    for K in FLEETS_B:
        for c in ["B1", "P5"]:
            v = []
            for s in range(seeds):
                gen = RegimeGenerator(seed=int(rm.integers(1e9))); post = assign_post(K, True)
                R, Bs, st, tc = build_fleet(K, T, "freeflow", post, gen, T0, STAGGER)
                v.append(np.mean(metrics(simulate(c, R, Bs, tc, p, gen.rng), st, p)["latency"]))
            lat[c][str(K)] = boot_ci(v)
    _save("B", lat); _figB(lat); return lat

def expC(seeds, T=T_DEFAULT, seed0=27):
    p = Params(); rm = np.random.default_rng(seed0); K = 20
    res = {c: {"honest": [], "adv": []} for c in ["B1","B2","P5"]}; bnd = []
    adv = np.zeros(K, bool); adv[:int(ADV_FRAC * K)] = True
    for s in range(seeds):
        gen = RegimeGenerator(seed=int(rm.integers(1e9))); post = assign_post(K, True)
        R, Bs, st, tc = build_fleet(K, T, "freeflow", post, gen, T0, STAGGER); bnd.append(theoretical_bound(R, tc, p))
        for c in ["B1","B2","P5"]:
            res[c]["honest"].append(np.mean(metrics(simulate(c, R, Bs, tc, p, gen.rng), st, p)["under_max"]))
            res[c]["adv"].append(np.mean(metrics(simulate(c, R, Bs, tc, p, gen.rng, adv_mask=adv, adv_offset=ADV_OFFSET), st, p)["under_max"]))
    out = {c: {k: boot_ci(v) for k, v in res[c].items()} for c in ["B1","B2","P5"]}; out["theoretical_bound"] = boot_ci(bnd)
    _save("C", out); _figC(out); return out

def expD(seeds, T=T_DEFAULT, seed0=37):
    p = Params(); rm = np.random.default_rng(seed0); err = {"B1": {}, "P5": {}}
    for K in FLEETS_D:
        for c in ["B1", "P5"]:
            v = []
            for s in range(seeds):
                gen = RegimeGenerator(seed=int(rm.integers(1e9))); post = assign_post(K, True)
                R, Bs, st, tc = build_fleet(K, T, "freeflow", post, gen, T0, STAGGER)
                v.append(np.mean(metrics(simulate(c, R, Bs, tc, p, gen.rng), st, p)["rate_err"]))
            err[c][str(K)] = boot_ci(v)
    err["Kstar"] = _figD(err); _save("D", err); return err

def merge():
    s = {}
    for n in ["A","B","C","D"]:
        with open(os.path.join(RES, f"{n}.json")) as f: s[n] = json.load(f)
    _save("summary", s); _print(s); return s

def _figA(A):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4)); lab = {"B0":"B0 frozen","B1":"B1 local","B2":"B2 naive","P5":"P5 fleet"}; x = np.arange(len(CONDS))
    for j, key, ttl in [(0,"rate_err","Intervention-rate error |rate-tau|"),(1,"missed","Missed high-risk events (rate)")]:
        m=[A[c][key][0] for c in CONDS]; lo=[A[c][key][0]-A[c][key][1] for c in CONDS]; hi=[A[c][key][2]-A[c][key][0] for c in CONDS]
        ax[j].bar(x, m, yerr=[lo,hi], capsize=4, color=["#b0b0b0","#6fa8dc","#e69138","#38761d"])
        ax[j].set_xticks(x); ax[j].set_xticklabels([lab[c] for c in CONDS], rotation=15); ax[j].set_title(ttl); ax[j].grid(axis="y", alpha=0.3)
    fig.suptitle("Experiment A - fleet=20, heterogeneous shift"); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "expA_accuracy_safety.png"), dpi=140); plt.close(fig)

def _figB(latB):
    fig, ax = plt.subplots(figsize=(6,4))
    for c,col in [("B1","#6fa8dc"),("P5","#38761d")]:
        m=[latB[c][str(k)][0] for k in FLEETS_B]; lo=[latB[c][str(k)][1] for k in FLEETS_B]; hi=[latB[c][str(k)][2] for k in FLEETS_B]
        ax.plot(FLEETS_B,m,"o-",color=col,label={"B1":"B1 local-only","P5":"P5 fleet"}[c]); ax.fill_between(FLEETS_B,lo,hi,color=col,alpha=0.2)
    ax.set_xscale("log"); ax.set_xlabel("fleet size K"); ax.set_ylabel("adaptation latency (steps)")
    ax.set_title("Experiment B - adaptation latency vs fleet size"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "expB_latency.png"), dpi=140); plt.close(fig)

def _figC(C):
    conds=["B1","B2","P5"]; x=np.arange(3); w=0.36; fig,ax=plt.subplots(figsize=(6.8,4.2))
    for off,key,lab,col,hatch in [(-w/2,"honest","honest fleet","#6fa8dc",None),(w/2,"adv","30% adversarial uploads","#e69138","//")]:
        m=[C[c][key][0] for c in conds]; lo=[C[c][key][0]-C[c][key][1] for c in conds]; hi=[C[c][key][2]-C[c][key][0] for c in conds]
        ax.bar(x+off,m,w,yerr=[lo,hi],capsize=3,label=lab,color=col,hatch=hatch,edgecolor="black",linewidth=.4)
    b=C["theoretical_bound"][0]; ax.axhline(b,ls="--",color="#b8860b",label=f"certified bound = {b:.3f}")
    ax.set_xticks(x); ax.set_xticklabels(["B1 local","B2 naive","P5 fleet"]); ax.set_ylabel("max under-protection (B_eff-B*)+")
    ax.legend(fontsize=8); ax.set_title("Experiment C - safety under adversarial fleet uploads"); ax.grid(axis="y",alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "expC_underprotection.png"), dpi=140); plt.close(fig)

def _figD(errD):
    fig, ax = plt.subplots(figsize=(6,4))
    for c,col in [("B1","#6fa8dc"),("P5","#38761d")]:
        m=[errD[c][str(k)][0] for k in FLEETS_D]; lo=[errD[c][str(k)][1] for k in FLEETS_D]; hi=[errD[c][str(k)][2] for k in FLEETS_D]
        ax.plot(FLEETS_D,m,"o-",color=col,label={"B1":"B1 local-only","P5":"P5 fleet"}[c]); ax.fill_between(FLEETS_D,lo,hi,color=col,alpha=0.2)
    Kstar=None
    for k in FLEETS_D:
        if errD["P5"][str(k)][0] <= 0.9*errD["B1"][str(k)][0]: Kstar=k; break
    if Kstar: ax.axvline(Kstar,ls=":",color="black",label=f"K* = {Kstar}")
    ax.set_xscale("log"); ax.set_xlabel("fleet size K"); ax.set_ylabel("intervention-rate error")
    ax.set_title("Experiment D - accuracy vs fleet size"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "expD_threshold.png"), dpi=140); plt.close(fig)
    return Kstar

def _print(s):
    def f(ci): return f"{ci[0]:.4f} [{ci[1]:.4f},{ci[2]:.4f}]"
    print("\n========== P5 FLEET EXPERIMENT RESULTS ==========")
    print("[A] fleet=20, heterogeneous shift")
    for c in CONDS:
        print(f"  {c}: rate_err={f(s['A'][c]['rate_err'])} missed={f(s['A'][c]['missed'])} under_max={f(s['A'][c]['under_max'])}")
    print(f"  proven bound: {f(s['A']['theoretical_bound'])}")
    print("[B] adaptation latency (steps)")
    for k in FLEETS_B: print(f"  K={k:>3}: B1={f(s['B']['B1'][str(k)])} P5={f(s['B']['P5'][str(k)])}")
    print("[C] max under-protection: honest | adversarial uploads")
    for c in ["B1","B2","P5"]: print(f"  {c}: honest={f(s['C'][c]['honest'])} adv={f(s['C'][c]['adv'])}")
    print(f"  proven bound: {f(s['C']['theoretical_bound'])}")
    print("[D] intervention-rate error vs K")
    for k in FLEETS_D: print(f"  K={k:>3}: B1={f(s['D']['B1'][str(k)])} P5={f(s['D']['P5'][str(k)])}")
    print(f"  benefit threshold K* = {s['D']['Kstar']}")
    print("=================================================\n")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
 