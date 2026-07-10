"""
Extra experiments requested in TR-C forensic revision, run for real (no hardcoded
numbers):

  Experiment H  : five aggregators under a 30% Byzantine fraction -> bias, rate
                  error, worst-case under-protection. Clamp is LOCAL for all, so
                  safety must be invariant to the aggregator; only accuracy differs.
  Experiment I  : local differential privacy. Laplace noise on each uploaded
                  quantile (smooth-sensitivity model), sweep eps0 in {0.5,1,2}
                  and fleet sizes; report rate-error deviation from noise-free and
                  that under-protection stays within the bound.
  Baselines     : ACI-only (online update, NO clamp) and FedNoClamp (boundary =
                  fleet quantile directly, NO local clamp) vs B1 and P5, to show
                  both unclamped designs breach the bound.

All conditions reuse the exact risk score, fleet build, metrics and bound of the
main study (imported from experiments.py / recalibration.py).
"""
import warnings; warnings.filterwarnings("ignore")
import os, json
import numpy as np
from recalibration import Params, _local_quantile, fleet_estimate
import experiments as E

OUT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(OUT, "results_real"); os.makedirs(RES, exist_ok=True)

# ----------------------------- aggregators -----------------------------
def agg_mean(U, n, p, Uq):        return np.full(len(U), float(np.mean(U)))
def agg_wquantile(U, n, p, Uq):   return fleet_estimate(U, n, p, weighted=True, U_query=Uq)
def agg_median(U, n, p, Uq):      return np.full(len(U), float(np.median(U)))
def agg_trimmed(U, n, p, Uq):
    k = max(1, int(0.10 * len(U)))
    s = np.sort(U)[k:len(U) - k] if len(U) > 2 * k else U
    return np.full(len(U), float(np.mean(s)))
def agg_krum(U, n, p, Uq):
    # Byzantine-robust selection: pick the point closest to its n-f-2 neighbours
    K = len(U); f = max(1, int(0.30 * K)); mm = max(1, K - f - 2)
    D = np.abs(U[:, None] - U[None, :]) ** 2
    scores = np.array([np.sort(D[i])[1:mm + 1].sum() for i in range(K)])
    return np.full(K, float(U[int(np.argmin(scores))]))

AGGS = {"Mean": agg_mean, "Weighted quantile": agg_wquantile,
        "Median": agg_median, "Trimmed mean": agg_trimmed, "Krum": agg_krum}

# --------------------- generalized simulate loop -----------------------
def simulate_agg(cond, R, Bstar, t_cal, p, rng, agg=agg_wquantile,
                 adv_mask=None, ldp_eps=None, track_bias=False, bstar_post=None):
    """cond in {B1,P5,ACI,FEDNOCLAMP} or 'AGG' (P5-like with chosen aggregator)."""
    K, T = R.shape
    pre = R[:, :t_cal]
    B0 = np.quantile(pre, 1.0 - p.tau, axis=1)
    B = B0.copy()
    window = pre[:, -p.N_s:].copy() if pre.shape[1] >= p.N_s else np.repeat(pre[:, -1:], p.N_s, axis=1).copy()
    n_eff = np.full(K, float(p.N_s))
    if adv_mask is None: adv_mask = np.zeros(K, bool)
    beff_log = np.zeros((K, T)); inter_log = np.zeros((K, T), dtype=bool)
    U_local = _local_quantile(window, p.tau)
    biases = []
    # LDP smooth-sensitivity of the empirical (1-tau)-quantile: gap ~ 1/(N_s * f)
    fmin = 0.3
    if ldp_eps is not None:
        q = np.quantile(pre.ravel(), 1 - p.tau); band = 0.03
        fmin = max(((np.abs(pre.ravel() - q) < band).mean()) / (2 * band), 0.3)
        sens = 1.0 / (p.N_s * fmin)
    for t in range(T):
        Rt = R[:, t]
        if t % 5 == 0:
            U_local = _local_quantile(window, p.tau)
        if cond == "ACI":
            Beff = B                                   # NO clamp
        elif cond == "FEDNOCLAMP":
            Beff = np.minimum.reduce([B])              # placeholder; set below
        if cond in ("B1",):
            Beff = np.minimum(B, U_local + p.m)
        elif cond in ("P5", "AGG", "FEDNOCLAMP"):
            if t % p.P == 0:
                U_report = U_local.copy()
                U_report[adv_mask] = 1.0               # maximal corrupted upload
                if ldp_eps is not None:
                    U_report = np.clip(U_report + rng.laplace(0.0, sens / ldp_eps, K), 0, 1)
                U_fleet = agg(U_report, n_eff, p, U_local)
                if track_bias and bstar_post is not None and t >= t_cal:
                    biases.append(float(np.mean(np.abs(U_fleet - bstar_post))))
                B_fleet = U_fleet
            if cond == "FEDNOCLAMP":
                Beff = B_fleet                         # clamp reads the fleet directly (NO local clamp)
            else:
                Beff = np.minimum(B, U_local + p.m)    # LOCAL clamp
        elif cond == "ACI":
            Beff = B
        intervened = Rt > Beff
        inter_log[:, t] = intervened; beff_log[:, t] = Beff
        # fast rate-tracking update (all adaptive conditions)
        B = np.clip(B + p.gamma * (intervened.astype(float) - p.tau), 0.0, 1.0)
        if cond in ("P5", "AGG", "FEDNOCLAMP") and t > 0 and t % p.P == 0:
            lam = p.lam
            if agg is agg_wquantile:
                gate = np.exp(-((B_fleet - U_local) ** 2) / (2 * p.h ** 2)); lam = p.lam * gate
            B = np.clip((1 - lam) * B + lam * B_fleet, 0.0, 1.0)
        window[:, t % p.N_s] = Rt
    log = dict(intervened=inter_log, beff=beff_log, Bstar=Bstar, R=R, params=p)
    return (log, biases) if track_bias else log


def run_expH(seeds=12, seed0=41):
    """Aggregators under 30% Byzantine, shared low->high shift (B* ~0.31 -> ~0.85)."""
    p = Params(); rm = np.random.default_rng(seed0); K = 20; T = E.T_DEFAULT
    adv = np.zeros(K, bool); adv[:int(0.30 * K)] = True
    out = {}
    for name, agg in AGGS.items():
        bias_l, re_l, ux_l, bnd_l = [], [], [], []
        for s in range(seeds):
            gen = E.RegimeGenerator(seed=int(rm.integers(1e9)))
            R, Bs, st, tc = E.build_fleet(K, T, "freeflow", ["congested"] * K, gen, E.T0, E.STAGGER)
            bstar_post = E.bstar_of("congested")
            log, biases = simulate_agg("AGG", R, Bs, tc, p, gen.rng, agg=agg,
                                       adv_mask=adv, track_bias=True, bstar_post=bstar_post)
            m = E.metrics(log, st, p)
            bias_l.append(np.mean(biases) if biases else np.nan)
            re_l.append(float(np.mean(m["rate_err"]))); ux_l.append(float(np.max(m["under_max"])))
            bnd_l.append(E.theoretical_bound(R, tc, p))
        out[name] = dict(bias=E.boot_ci(bias_l), rate_err=E.boot_ci(re_l),
                         under_max=E.boot_ci(ux_l))
        print(f"[H] {name:18s} bias={out[name]['bias'][0]:.3f}  rate_err={out[name]['rate_err'][0]:.3f}  under_max={out[name]['under_max'][0]:.3f}")
    out["bound"] = E.boot_ci(bnd_l)
    json.dump(out, open(os.path.join(RES, "expH.json"), "w"), indent=2)
    print("  bound=%.3f  -> saved results_real/expH.json" % out["bound"][0])
    return out


def run_expI(seeds=12, seed0=53):
    """LDP: rate-error deviation from noise-free across eps0 and fleet size."""
    p = Params(); K_list = [5, 20, 100]; eps_list = [None, 2.0, 1.0, 0.5]; T = E.T_DEFAULT
    out = {}
    for K in K_list:
        rm = np.random.default_rng(seed0 + K)
        base = {}
        for eps in eps_list:
            re_l, ux_l, bnd_l = [], [], []
            for s in range(seeds):
                gen = E.RegimeGenerator(seed=int(rm.integers(1e9))); post = E.assign_post(K, True)
                R, Bs, st, tc = E.build_fleet(K, T, "freeflow", post, gen, E.T0, E.STAGGER)
                log = simulate_agg("P5", R, Bs, tc, p, gen.rng, agg=agg_wquantile, ldp_eps=eps)
                m = E.metrics(log, st, p)
                re_l.append(float(np.mean(m["rate_err"]))); ux_l.append(float(np.max(m["under_max"])))
                bnd_l.append(E.theoretical_bound(R, tc, p))
            key = "noisefree" if eps is None else f"eps{eps}"
            base[key] = dict(rate_err=float(np.mean(re_l)), under_max=float(np.mean(ux_l)),
                             bound=float(np.mean(bnd_l)))
        nf = base["noisefree"]["rate_err"]
        for k in base:
            base[k]["rate_err_dev_vs_noisefree"] = abs(base[k]["rate_err"] - nf)
        out[f"K{K}"] = base
        worst = max(base[k]["rate_err_dev_vs_noisefree"] for k in base)
        print(f"[I] K={K}: max rate-err deviation vs noise-free = {worst:.4f}; under<=bound: "
              + str(all(base[k]['under_max'] <= base[k]['bound'] for k in base)))
    json.dump(out, open(os.path.join(RES, "expI_ldp.json"), "w"), indent=2)
    print("  -> saved results_real/expI_ldp.json")
    return out


def run_baselines(seeds=12, seed0=61):
    """ACI-only and FedNoClamp vs B1, P5 (heterogeneous shift, 30% adversarial)."""
    p = Params(); rm = np.random.default_rng(seed0); K = 20; T = E.T_DEFAULT
    adv = np.zeros(K, bool); adv[:int(0.30 * K)] = True
    conds = ["B1", "ACI", "FEDNOCLAMP", "P5"]
    out = {c: {"rate_err": [], "under_max": []} for c in conds}; bnd = []
    for s in range(seeds):
        gen = E.RegimeGenerator(seed=int(rm.integers(1e9))); post = E.assign_post(K, True)
        R, Bs, st, tc = E.build_fleet(K, T, "freeflow", post, gen, E.T0, E.STAGGER)
        bnd.append(E.theoretical_bound(R, tc, p))
        for c in conds:
            am = adv if c in ("P5", "FEDNOCLAMP") else None
            log = simulate_agg(c, R, Bs, tc, p, gen.rng, agg=agg_wquantile, adv_mask=am)
            m = E.metrics(log, st, p)
            out[c]["rate_err"].append(float(np.mean(m["rate_err"])))
            out[c]["under_max"].append(float(np.max(m["under_max"])))
    res = {c: {k: E.boot_ci(v) for k, v in out[c].items()} for c in conds}
    res["bound"] = E.boot_ci(bnd)
    for c in conds:
        print(f"[base] {c:10s} rate_err={res[c]['rate_err'][0]:.3f}  under_max={res[c]['under_max'][0]:.3f}  (bound {res['bound'][0]:.3f})")
    json.dump(res, open(os.path.join(RES, "baselines_extra.json"), "w"), indent=2)
    print("  -> saved results_real/baselines_extra.json")
    return res


if __name__ == "__main__":
    print("=== Experiment H (aggregators) ==="); run_expH()
    print("\n=== Experiment I (LDP) ===");        run_expI()
    print("\n=== Extra baselines ===");           run_baselines()
