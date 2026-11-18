"""Additional forensic checks (release v1.0.4).

  split     : disjoint reference/stream halves of each real pool (B* from one half, streams from the other)
  split70   : 70% of trajectories supply the streams, B* from the held-out 30% only
  closed    : paired seed-level differences for the main closed loop (Table 5 seeds 100-111)
  downward  : downward (over-intervention) fleet attack on the real replay
  coverage  : empirical coverage of the per-step and simultaneous certificates across independent runs
  online    : conservative online (DKW) upper bound on the quantile modulus, vs the retrospective value
  robustclamp: clamp reads a robust fleet aggregate (trimmed mean, median, Krum), 0-50% adversarial
  online_sim: simultaneous (all K*T vehicle-steps) rigorous online envelope vs window length
  deploy_paired: paired differences for the combined-fault deployment test (30 seeds)
  closed4   : paired closed loop with the reduced four-feature score (ablation)
  rho       : envelopes for operator-set drift bounds rho = 0, 0.01, 0.02, 0.05
  hetero    : latency / rate error vs post-shift heterogeneity and fleet size (local-only vs proposed, paired)
  mixture   : quantile modulus / envelope of mixed pre/post-shift windows (transient) and burn-in k*
  transient_split: observed under-protection in the transient (mixed-window) vs certified (post-turnover) interval
  holm      : Holm-adjusted confirmatory family and the unadjusted secondary paired tests
  nsweep    : certified envelopes (four terms) vs observed under-protection for N_s = 100-5000, with the alpha budget

Usage:  python forensic_checks.py all          (or one of: split closed downward coverage)
Outputs: results_real/forensic_{split,closed_paired,downward,coverage,online_envelope}.json
"""
import os, sys, json, warnings
import numpy as np
warnings.filterwarnings("ignore")
import experiments as E
from recalibration import Params, simulate

OUT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(OUT, "results_real")
K = 20
NAME_MAP = {"freeflow": "highd", "moderate": "waymo", "night": "waymo",
            "congested": "ngsim", "dense": "ngsim", "rain": "ngsim"}
POOLS = {ds: np.load(os.path.join(OUT, "real", f"{ds}.npy")) for ds in ["highd", "ngsim", "waymo"]}


def save(name, obj):
    with open(os.path.join(RES, name), "w") as f: json.dump(obj, f, indent=2)
    print("saved", name)


def signflip(d, nperm=100000, seed=0):
    d = np.asarray(d, float); n = len(d); obs = abs(d.mean())
    if 2 ** n <= nperm: signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
    else: signs = np.random.default_rng(seed).choice([-1, 1], size=(nperm, n))
    null = np.abs((signs * d).mean(axis=1))
    return float(max((null >= obs - 1e-15).mean(), 1.0 / len(null)))


def paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return {"mean_diff": E.boot_ci(d), "p_signflip": signflip(d), "n": int(len(d))}


def make_backend(ref, stream):
    class Split:
        def __init__(self, seed=0): self.rng = np.random.default_rng(seed)
        def stream(self, regime, n): return self.rng.choice(stream[NAME_MAP.get(regime, "waymo")], size=n, replace=True)
        def true_matching_boundary(self, regime, tau, n=0): return float(np.quantile(ref[NAME_MAP.get(regime, "waymo")], 1 - tau))
    return Split


def use_backend(cls):
    E.RegimeGenerator = cls; E._BSTAR = {}


def full_backend():
    return make_backend(POOLS, POOLS)


# ---------------------------------------------------------------- 1. split
def run_split(n_splits=5, seeds=15, frac_ref=0.5, outname="forensic_split.json"):
    """Disjoint trajectory-level split of each real pool (one value = one trajectory).
    The first `frac_ref` share supplies the vehicles' streams; B* is computed only from the disjoint rest."""
    p = Params(); out = {"n_splits": n_splits, "seeds_per_split": seeds, "splits": []}
    conds = ["B1", "B2", "P5"]; adv = np.zeros(K, bool); adv[:int(0.3 * K)] = True
    pooled = {c: {"rate_err": [], "under_max": [], "under_max_adv30": []} for c in conds}
    for sp in range(n_splits):
        rs = np.random.default_rng(1000 + sp); ref, stm = {}, {}
        for ds, a in POOLS.items():
            perm = rs.permutation(len(a)); cut = int(round(frac_ref * len(a))); stm[ds] = a[perm[:cut]]; ref[ds] = a[perm[cut:]]
        use_backend(make_backend(ref, stm))
        bst = {ds: float(np.quantile(ref[ds], 0.9)) for ds in ref}
        bst_stream = {ds: float(np.quantile(stm[ds], 0.9)) for ds in stm}
        rm = np.random.default_rng(7 + sp)
        for s in range(seeds):
            gen = E.RegimeGenerator(seed=int(rm.integers(1e9))); post = E.assign_post(K, True)
            R, Bs, st, tc = E.build_fleet(K, E.T_DEFAULT, "freeflow", post, gen, E.T0, E.STAGGER)
            for c in conds:
                m = E.metrics(simulate(c, R, Bs, tc, p, gen.rng), st, p)
                ma = E.metrics(simulate(c, R, Bs, tc, p, gen.rng, adv_mask=adv, adv_offset=E.ADV_OFFSET), st, p)
                pooled[c]["rate_err"].append(float(np.mean(m["rate_err"])))
                pooled[c]["under_max"].append(float(np.mean(m["under_max"])))
                pooled[c]["under_max_adv30"].append(float(np.mean(ma["under_max"])))
        out["splits"].append({"Bstar_ref_half": bst, "Bstar_stream_half": bst_stream})
    out["results"] = {c: {k: E.boot_ci(v) for k, v in pooled[c].items()} for c in conds}
    out["paired_proposed_minus_naive_under_max"] = paired(pooled["P5"]["under_max"], pooled["B2"]["under_max"])
    out["paired_proposed_adv30_minus_honest"] = paired(pooled["P5"]["under_max_adv30"], pooled["P5"]["under_max"])
    use_backend(full_backend())
    out["frac_stream"] = frac_ref; out["frac_heldout_Bstar"] = 1 - frac_ref
    save(outname, out); return out


# ---------------------------------------------------------------- 2. closed loop paired
def run_closed(seeds=12, K_=20, T=3000, score="8", outname="forensic_closed_paired.json"):
    import closed_loop as CL
    CL.SCORE = score
    Bpre = CL.bstar("mild", np.random.default_rng(1)); Bpost = CL.bstar("aggr", np.random.default_rng(2))
    B0c = np.full(K_, Bpre); conds = ["AEB", "B0", "B1", "FedAvg", "RobFCP", "P5"]
    per = {c: {"coll": [], "rate_err": [], "under_max": []} for c in conds}
    for c in conds:
        for s in range(seeds):
            r = np.random.default_rng(100 + s); leads, shift = CL.build_leads(K_, T, T // 3, T // 4, r)
            m = CL.simulate(c, leads, shift, r, Bpre, Bpost, B0c)
            for k in per[c]: per[c][k].append(float(m[k]))
    out = {"seeds": seeds, "per_seed": per, "paired_proposed_minus": {}}
    for c in ["AEB", "B0", "B1", "FedAvg", "RobFCP"]:
        out["paired_proposed_minus"][c] = {k: paired(per["P5"][k], per[c][k]) for k in ["coll", "rate_err", "under_max"]
                                          if not np.any(np.isnan(per[c][k]))}
    out["score"] = "eight-feature" if score == "8" else "four-feature"
    CL.SCORE = "8"
    save(outname, out); return out


# ---------------------------------------------------------------- 3. downward attack
def run_downward(seeds=15, offsets=(-0.02, -0.05, -0.1, -0.25, -0.5)):
    use_backend(full_backend()); p = Params(); conds = ["B1", "B2", "P5"]
    adv = np.zeros(K, bool); adv[:int(0.3 * K)] = True
    acc = {c: {f"{o}": {"rate_err": [], "signed_rate_dev": [], "unnecessary": [], "over_prot": []}
               for o in (0.0,) + tuple(offsets)} for c in conds}
    rm = np.random.default_rng(77)
    for s in range(seeds):
        gen = E.RegimeGenerator(seed=int(rm.integers(1e9))); post = E.assign_post(K, True)
        R, Bs, st, tc = E.build_fleet(K, E.T_DEFAULT, "freeflow", post, gen, E.T0, E.STAGGER)
        for c in conds:
            for o in (0.0,) + tuple(offsets):
                log = simulate(c, R, Bs, tc, p, gen.rng, adv_mask=adv if o else None, adv_offset=o)
                m = E.metrics(log, st, p); rate = E.moving_rate(log["intervened"], 100)
                sd, un, op = [], [], []
                for k in range(K):
                    stk = int(st[k]); steady = slice(min(stk + 350, E.T_DEFAULT - 1), E.T_DEFAULT); post_ = slice(stk, E.T_DEFAULT)
                    sd.append(np.mean(rate[k, steady] - p.tau))
                    un.append(np.mean(log["intervened"][k, post_] & (R[k, post_] <= Bs[k, post_])))
                    op.append(np.mean(np.maximum(Bs[k, post_] - log["beff"][k, post_], 0)))
                a = acc[c][f"{o}"]
                a["rate_err"].append(float(np.mean(m["rate_err"]))); a["signed_rate_dev"].append(float(np.mean(sd)))
                a["unnecessary"].append(float(np.mean(un))); a["over_prot"].append(float(np.mean(op)))
    out = {"seeds": seeds, "adv_fraction": 0.3, "offsets": list(offsets),
           "definitions": {"signed_rate_dev": "mean(moving intervention rate - tau) over the steady window",
                           "unnecessary": "fraction of post-shift steps with an intervention while R <= B*",
                           "over_prot": "mean (B* - B_eff)_+ over post-shift steps"},
           "results": {c: {o: {k: E.boot_ci(v) for k, v in d.items()} for o, d in acc[c].items()} for c in conds}}
    save("forensic_downward.json", out); return out


# ---------------------------------------------------------------- 4. coverage audit
def run_coverage(seeds=30):
    cert = json.load(open(os.path.join(RES, "certificates.json")))["laws"]
    p = Params(); adv = np.zeros(K, bool); adv[:int(0.3 * K)] = True; out = {"seeds": seeds, "alpha_step": 0.05, "zeta": 0.05}
    for setting in ["real", "controlled"]:
        if setting == "real": use_backend(full_backend())
        else:
            from data import RegimeGenerator; E.RegimeGenerator = RegimeGenerator; E._BSTAR = {}
        rm = np.random.default_rng(4242); res = {}
        for c, am in [("P5", None), ("P5_adv30", adv), ("B2_adv30", adv)]:
            cond = c.split("_")[0]; viol_frac, run_ok, run_max = [], [], []
            rm = np.random.default_rng(4242)
            for s in range(seeds):
                gen = E.RegimeGenerator(seed=int(rm.integers(1e9))); post = E.assign_post(K, True)
                R, Bs, st, tc = E.build_fleet(K, E.T_DEFAULT, "freeflow", post, gen, E.T0, E.STAGGER)
                log = simulate(cond, R, Bs, tc, p, gen.rng, adv_mask=am, adv_offset=E.ADV_OFFSET if am is not None else 0.0)
                nv = tot = 0; ok = True; rmax = 0.0
                for k in range(K):
                    law = NAME_MAP[post[k]] if setting == "real" else "controlled"
                    es, em = cert[law]["env_step"], cert[law]["env_sim"]
                    u = np.maximum(log["beff"][k, int(st[k]):] - Bs[k, int(st[k]):], 0)
                    nv += int((u > es).sum()); tot += u.size; rmax = max(rmax, float(u.max()))
                    ok = ok and bool(u.max() <= em)
                viol_frac.append(nv / tot); run_ok.append(ok); run_max.append(rmax)
            res[c] = {"per_step_violation_fraction": E.boot_ci(viol_frac),
                      "max_per_step_violation_fraction_any_run": float(np.max(viol_frac)),
                      "simultaneous_event_held_in_runs": f"{int(np.sum(run_ok))}/{seeds}",
                      "true_run_max": E.boot_ci(run_max), "true_run_max_worst": float(np.max(run_max))}
        out[setting] = res
    use_backend(full_backend())
    out["note"] = ("Envelopes are retrospective: omega is computed from the realized post-shift law (certificates.json), "
                   "which is not observable online. Per-step event: (B_eff - B*)_+ <= env_step at a single (i,t), nominal "
                   "coverage 1 - alpha = 0.95. Simultaneous event: max over all (i,t) <= env_sim, nominal 1 - zeta = 0.95.")
    save("forensic_coverage.json", out); return out


# ---------------------------------------------------------------- 5. valid online modulus bound (DKW)
def _delta_fp(alpha, Ns, p=0.9):
    import math
    L = math.log(2.0 / alpha); d = 0.05
    for _ in range(300):
        v = max((p - d) * (1 - p + d), (p + d) * (1 - p - d))
        d2 = math.sqrt(2 * v * L / Ns) + (2.0 / 3.0) * L / Ns
        if abs(d2 - d) < 1e-13: return d2
        d = d2
    return d


def _controlled_law():
    from data import RegimeGenerator; E.RegimeGenerator = RegimeGenerator; E._BSTAR = {}
    ctrl = []
    for sd in range(7, 37):
        rm = np.random.default_rng(sd); gen = RegimeGenerator(seed=int(rm.integers(1e9))); post = E.assign_post(K, True)
        R, Bs, st, tc = E.build_fleet(K, E.T_DEFAULT, "freeflow", post, gen, E.T0, E.STAGGER); ctrl.append(R[:, tc:].ravel())
    use_backend(full_backend())
    return np.concatenate(ctrl)


def run_online(n_windows=500, alpha_total=0.05, windows=(300, 600, 1000, 2000)):
    """Conservative, online-computable upper bound on the quantile modulus omega_Fbar(delta).
    For a conditionally i.i.d. window of Ns scores in [0,1], DKW-Massart gives sup_x |Fhat - Fbar| <= eps,
    eps = sqrt(log(2/alpha2)/(2 Ns)), with prob. >= 1 - alpha2. On that event
        Fhat^{-1}(q - eps) <= Fbar^{-1}(q) <= Fhat^{-1}(q + eps),
    where an upper level q + eps > 1 must be replaced by the known upper support 1 (NOT the sample maximum,
    which is not an upper confidence bound) and a lower level q - eps <= 0 by the lower support 0. Hence
        omega_Fbar(delta) <= max{ Qup(p+delta) - Qlo(p),  Qup(p) - Qlo(p-delta) },
        Qup(q) = Qhat(q+eps) if q+eps <= 1 else 1,   Qlo(q) = Qhat(q-eps) if q-eps > 0 else 0.
    The envelope m + omega_DKW + rho holds with prob. >= 1 - alpha_q - alpha_dkw; we split alpha_q = alpha_dkw = alpha_total/2
    so the per-step online envelope has the same 95% level as the retrospective one. rho is an assumed drift bound."""
    import math
    p = 0.9; Mm, RHO = 0.02, 0.01
    alpha = alpha2 = alpha_total / 2          # quantile event + DKW event share the total per-step budget
    laws = {ds: POOLS[ds] for ds in ["ngsim", "waymo", "highd"]}; laws["controlled"] = _controlled_law()
    out = {"alpha_total": alpha_total, "alpha_q": alpha, "alpha_dkw": alpha2, "n_windows": n_windows, "support": [0.0, 1.0], "by_window": {}}
    for Ns in windows:
        d = _delta_fp(alpha, Ns, p); eps = math.sqrt(math.log(2 / alpha2) / (2 * Ns))
        def qhat(w, x): return float(w[min(max(math.ceil(x * Ns) - 1, 0), Ns - 1)])
        def Qup(w, x): return 1.0 if x + eps > 1 else qhat(w, x + eps)
        def Qlo(w, x): return 0.0 if x - eps <= 0 else qhat(w, x - eps)
        d_r = _delta_fp(alpha_total, Ns, p)     # retrospective envelope keeps the single-event budget alpha_total
        blk = {"delta_online": d, "delta_retrospective": d_r, "eps": eps, "upper_level_exceeds_1": bool(p + d + eps > 1), "laws": {}}
        for k, a in laws.items():
            s_all = np.sort(a); n = len(s_all)
            def qa(x): return float(s_all[min(max(math.ceil(min(max(x, 0.0), 1.0) * n) - 1, 0), n - 1)])
            om_retro = max(qa(p + d_r) - qa(p), qa(p) - qa(p - d_r))
            om_true = max(qa(p + d) - qa(p), qa(p) - qa(p - d))   # true modulus at the online delta (coverage check)
            rng = np.random.default_rng(0); om, cover = [], []
            for _ in range(n_windows):
                w = np.sort(rng.choice(a, Ns, replace=True))
                o = max(Qup(w, p + d) - Qlo(w, p), Qup(w, p) - Qlo(w, p - d))
                om.append(o); cover.append(o >= om_true)
            om = np.array(om)
            blk["laws"][k] = {"env_step_retrospective": round(Mm + om_retro + RHO, 4),
                              "env_step_online_dkw_median": round(float(Mm + np.median(om) + RHO), 4),
                              "env_step_online_dkw_p05_p95": [round(float(Mm + np.percentile(om, 5) + RHO), 4),
                                                              round(float(Mm + np.percentile(om, 95) + RHO), 4)],
                              "fraction_windows_bound_covers_true_omega": float(np.mean(cover))}
        out["by_window"][str(Ns)] = blk
    save("forensic_online_envelope.json", out); return out


# ---------------------------------------------------------------- 5b. realized drift between the window law and the current target
def run_drift():
    """For a vehicle whose window holds k post-shift and Ns-k pre-shift samples, the window-averaged law is the mixture
    Fbar_k = (1-k/Ns) F_pre + (k/Ns) F_post and its (1-tau)-quantile Bbar_k is deterministic given k. The drift term
    of Assumption A3 is |Bbar_k - B*_post|; the one-sided bound only needs (Bbar_k - B*_post)_+ .
    Reports, for each pre->post pair used in the studies, the two-sided and one-sided drift as a function of k and
    the burn-in k after which the two-sided drift stays <= 0.01."""
    p = 0.9; Ns = 300
    ctrl = {}
    from data import RegimeGenerator
    g = RegimeGenerator(seed=12345)
    for r in ["freeflow", "congested", "rain", "dense", "night", "moderate"]: ctrl[r] = np.sort(g.stream(r, 200000))
    use_backend(full_backend())
    real = {k: np.sort(v) for k, v in POOLS.items()}
    pairs = [("real", "highd", "ngsim", real), ("real", "highd", "waymo", real)] + \
            [("controlled", "freeflow", r, ctrl) for r in ["congested", "rain", "dense", "night", "moderate"]]
    def mix_q(a, b, wb):
        grid = np.linspace(0, 1, 4001); F = (1 - wb) * np.searchsorted(a, grid, "right") / len(a) + wb * np.searchsorted(b, grid, "right") / len(b)
        return float(grid[np.argmax(F >= p)])
    out = {"Ns": Ns, "tau": 0.1, "pairs": {}}
    for kind, pre, post, L in pairs:
        a, b = L[pre], L[post]; bstar_post = float(np.quantile(b, p)); bstar_pre = float(np.quantile(a, p))
        ks = np.arange(0, Ns + 1); two = []; one = []
        for k in ks:
            bb = mix_q(a, b, k / Ns); two.append(abs(bb - bstar_post)); one.append(max(bb - bstar_post, 0.0))
        two = np.array(two); one = np.array(one)
        burn = int(next((k for k in ks if np.all(two[k:] <= 0.01)), Ns))
        out["pairs"][f"{pre}->{post}"] = {"kind": kind, "Bstar_pre": round(bstar_pre, 4), "Bstar_post": round(bstar_post, 4),
            "two_sided_max": round(float(two.max()), 4), "one_sided_max": round(float(one.max()), 4),
            "two_sided_at_k": {str(k): round(float(two[k]), 4) for k in [0, 50, 100, 150, 200, 250, 270, 300]},
            "burn_in_k_for_rho_0.01": burn}
    save("forensic_drift.json", out); return out


# ---------------------------------------------------------------- 6. sensitivity of the envelopes to the assumed drift bound rho
def run_rho(rhos=(0.0, 0.01, 0.02, 0.05)):
    """Envelopes are m + omega + rho, so they shift one-for-one with the operator-set drift bound rho.
    Retrospective values from certificates.json (computed at rho = 0.01), online DKW values from
    forensic_online_envelope.json; the algorithm itself does not use rho, so observed values are unchanged."""
    cert = json.load(open(os.path.join(RES, "certificates.json")))["laws"]
    onl = json.load(open(os.path.join(RES, "forensic_online_envelope.json")))["by_window"]["300"]["laws"]
    out = {"rhos": list(rhos), "laws": {}}
    for k in ["controlled", "ngsim", "waymo"]:
        out["laws"][k] = {str(r): {"env_step_retrospective": round(cert[k]["env_step"] - 0.01 + r, 4),
                                   "env_sim_retrospective": round(cert[k]["env_sim"] - 0.01 + r, 4),
                                   "env_step_online_dkw": round(onl[k]["env_step_online_dkw_median"] - 0.01 + r, 4)} for r in rhos}
    save("forensic_rho.json", out); return out


# ---------------------------------------------------------------- 7. robust fleet-clamp baselines, adversarial sweep
def run_robustclamp(seeds=30, fracs=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5)):
    """Clamp reads a ROBUST fleet aggregate (trimmed mean 10%, median, Krum) instead of the local quantile,
    on the real highD->NGSIM shift of Table 3 (maximal U=1.0 adversarial uploads, run-maximum metric)."""
    from extra_experiments import simulate_agg, agg_trimmed, agg_median, agg_krum, agg_wquantile
    use_backend(full_backend()); p = Params()
    variants = {"Trimmed-mean fleet-clamp": ("FEDNOCLAMP", agg_trimmed), "Median fleet-clamp": ("FEDNOCLAMP", agg_median),
                "Krum fleet-clamp": ("FEDNOCLAMP", agg_krum), "Proposed": ("P5", agg_wquantile)}
    acc = {v: {str(fr): {"rate_err": [], "under_max": []} for fr in fracs} for v in variants}
    rm = np.random.default_rng(777)
    for s in range(seeds):
        gen = E.RegimeGenerator(seed=int(rm.integers(1e9)))
        R, Bs, st, tc = E.build_fleet(K, E.T_DEFAULT, "freeflow", ["congested"] * K, gen, E.T0, E.STAGGER)
        for fr in fracs:
            adv = np.zeros(K, bool); adv[:int(round(fr * K))] = True
            for v, (cond, agg) in variants.items():
                m = E.metrics(simulate_agg(cond, R, Bs, tc, p, gen.rng, agg=agg, adv_mask=adv if fr > 0 else None), st, p)
                acc[v][str(fr)]["rate_err"].append(float(np.mean(m["rate_err"])))
                acc[v][str(fr)]["under_max"].append(float(np.max(m["under_max"])))
    out = {"seeds": seeds, "fracs": list(fracs), "shift": "highD->NGSIM (Table 3 setting)", "adversary": "U = 1.0 uploads",
           "metric": "run maximum over vehicles (as Table 3)",
           "results": {v: {fr: {k: E.boot_ci(x) for k, x in d.items()} for fr, d in acc[v].items()} for v in acc}}
    save("forensic_robustclamp.json", out); return out


def run_online_sim(n_windows=300, zeta=0.05, windows=(300, 600, 1000, 2000, 5000)):
    """Simultaneous (all K*T vehicle-steps) rigorous online envelope. The two error budgets are allocated jointly:
    alpha_q + alpha_dkw = zeta/(K*T), with alpha_q = alpha_dkw = zeta/(2*K*T), so a union bound over the K*T
    vehicle-steps gives simultaneous coverage >= 1 - zeta (K = 20, T = 1700). Known support [0,1] replaces out-of-range levels."""
    import math
    p = 0.9; KT = K * E.T_DEFAULT; Mm, RHO = 0.02, 0.01
    laws = {ds: POOLS[ds] for ds in ["ngsim", "waymo", "highd"]}; laws["controlled"] = _controlled_law()
    res = {}
    for Ns in windows:
        aq = ad = zeta / (2 * KT)
        d = _delta_fp(aq, Ns, p); eps = math.sqrt(math.log(2 / ad) / (2 * Ns))
        def qh(w, x): return float(w[min(max(math.ceil(x * Ns) - 1, 0), Ns - 1)])
        def Qup(w, x): return 1.0 if x + eps > 1 else qh(w, x + eps)
        def Qlo(w, x): return 0.0 if x - eps <= 0 else qh(w, x - eps)
        row = {}
        for k, a in laws.items():
            rng = np.random.default_rng(0); om = []
            for _ in range(n_windows):
                w = np.sort(rng.choice(a, Ns)); om.append(max(Qup(w, p + d) - Qlo(w, p), Qup(w, p) - Qlo(w, p - d)))
            row[k] = round(Mm + float(np.median(om)) + RHO, 3)
        res[str(Ns)] = {"delta_sim": round(d, 4), "eps_sim": round(eps, 4), "env_sim_online": row}
    save("forensic_online_simultaneous.json", {"KT": KT, "zeta": zeta, "alpha_q": zeta / (2 * KT), "alpha_dkw": zeta / (2 * KT), "by_window": res}); return res


def run_deploy_paired(seeds=30):
    """Paired seed-level differences for the combined-fault deployment test (Table S17 / Sec. 7.5)."""
    import deploy_experiments as DX
    from run_block import units_for
    import closed_loop as CL
    Bpre = CL.bstar("mild", np.random.default_rng(1)); Bpost = CL.bstar("aggr", np.random.default_rng(2))
    units = dict(units_for("deploy")); per = {}
    for name in ["deploy_B2", "deploy_P5", "deploy_B1"]:
        kw = dict(units[name]); cond = kw.pop("cond"); K_ = kw.pop("K", 20); acc = {"coll": [], "rate_err": [], "under_max": []}
        for i in range(seeds):
            r = np.random.default_rng(100 + i); leads, shift = DX.build_leads(K_, 3000, 1000, 750, r)
            m = DX.simulate(cond, leads, shift, r, Bpre, Bpost, np.full(K_, Bpre), **kw)
            for k in acc: acc[k].append(float(m[k]))
        per[name] = acc
    out = {"score": "eight-feature" if CL.SCORE == "8" else "four-feature", "seeds": seeds, "per_seed": per,
           "P5_minus_B2": {k: paired(per["deploy_P5"][k], per["deploy_B2"][k]) for k in ["coll", "rate_err", "under_max"]},
           "P5_minus_B1": {k: paired(per["deploy_P5"][k], per["deploy_B1"][k]) for k in ["coll", "rate_err", "under_max"]}}
    save("forensic_deploy_paired.json", out); return out


# ---------------------------------------------------------------- 8. heterogeneity x fleet-size sweep
def _post_assign(K_, frac_alt, alts, base):
    """First round(frac_alt*K) vehicles take the alternative regimes (cycled), the rest the base regime.
    Shift times are drawn independently in build_fleet, so vehicle order carries no information."""
    n_alt = int(round(frac_alt * K_))
    return [alts[i % len(alts)] for i in range(n_alt)] + [base] * (K_ - n_alt)


def run_hetero(seeds=30, Ks=(5, 10, 20, 50, 100), seed0=91):
    """Latency / rate error / under-protection vs post-shift heterogeneity and fleet size (paired B1 vs P5).
    Heterogeneity = fraction of the fleet whose post-shift regime differs from the base regime; it is also reported
    as the realized cross-vehicle SD of the post-shift target B*. Real: base NGSIM, alternative Waymo (0-50%).
    Controlled: base 'congested', alternatives rain/dense/night/moderate cycled (0-80%; 80% = Table 2 fleet)."""
    from data import RegimeGenerator
    setups = {"real": (full_backend(), "congested", ["moderate"], (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)),
              "controlled": (RegimeGenerator, "congested", ["rain", "dense", "night", "moderate"], (0.0, 0.2, 0.4, 0.6, 0.8))}
    p = Params(); out = {"seeds": seeds, "Ks": list(Ks), "metrics": "latency and rate error: mean over vehicles; "
           "under_max: mean over vehicles of the per-vehicle post-shift maximum (Table 2 metric)", "setups": {}}
    for name, (backend, base, alts, fracs) in setups.items():
        use_backend(backend); blk = {"base": base, "alternatives": alts, "fracs": list(fracs), "cells": {}}
        for K_ in Ks:
            for fr in fracs:
                post = _post_assign(K_, fr, alts, base)
                acc = {c: {"latency": [], "rate_err": [], "under_max": []} for c in ["B1", "P5"]}; sd_b = []
                rm = np.random.default_rng(seed0)
                for s in range(seeds):
                    gen = E.RegimeGenerator(seed=int(rm.integers(1e9)))
                    R, Bs, st, tc = E.build_fleet(K_, E.T_DEFAULT, "freeflow", post, gen, E.T0, E.STAGGER)
                    sd_b.append(float(np.std(Bs[:, -1])))
                    for c in ["B1", "P5"]:
                        m = E.metrics(simulate(c, R, Bs, tc, p, gen.rng), st, p)
                        for k in acc[c]: acc[c][k].append(float(np.mean(m[k])))
                cell = {"frac_alt_realized": round(sum(r != base for r in post) / K_, 3), "sd_Bstar_post": round(float(np.mean(sd_b)), 4),
                        "B1": {k: E.boot_ci(v) for k, v in acc["B1"].items()}, "P5": {k: E.boot_ci(v) for k, v in acc["P5"].items()},
                        "P5_minus_B1": {k: paired(acc["P5"][k], acc["B1"][k]) for k in ["latency", "rate_err", "under_max"]}}
                blk["cells"][f"K{K_}_f{fr}"] = cell
                print(name, K_, fr, "lat", round(cell["P5_minus_B1"]["latency"]["mean_diff"][0], 1),
                      "re", round(cell["P5_minus_B1"]["rate_err"]["mean_diff"][0], 4))
        out["setups"][name] = blk
    use_backend(full_backend()); save("forensic_hetero.json", out); return out


# ---------------------------------------------------------------- 9. certified envelopes vs observed under-protection over N_s (+ alpha budget)
def _alpha_budget(Ns, KT, zeta=0.05, p=0.9):
    import math
    a_s = zeta; a_q = a_d = zeta / 2; a_S = zeta / KT; a_qS = a_dS = zeta / (2 * KT)
    return {"retro_step": {"alpha_q": a_s, "alpha_dkw": 0.0, "delta": _delta_fp(a_s, Ns, p), "eps": 0.0, "failure": a_s},
            "online_step": {"alpha_q": a_q, "alpha_dkw": a_d, "delta": _delta_fp(a_q, Ns, p),
                            "eps": math.sqrt(math.log(2 / a_d) / (2 * Ns)), "failure": a_q + a_d},
            "retro_sim": {"alpha_q": a_S, "alpha_dkw": 0.0, "delta": _delta_fp(a_S, Ns, p), "eps": 0.0, "failure": KT * a_S},
            "online_sim": {"alpha_q": a_qS, "alpha_dkw": a_dS, "delta": _delta_fp(a_qS, Ns, p),
                           "eps": math.sqrt(math.log(2 / a_dS) / (2 * Ns)), "failure": KT * (a_qS + a_dS)}}


def run_nsweep(windows=(100, 200, 300, 600, 1000, 2000, 5000), seeds=30, n_windows=300, zeta=0.05, seed0=7):
    """For each window length N_s: the four envelope terms (retrospective / online DKW, per-step / simultaneous; m = 0.02,
    rho = 0.01) and the OBSERVED under-protection of the proposed scheme (P5) run at that N_s.
    Observed runs use T0 = max(450, N_s + 150), so the initial window holds pre-shift scores, and a post-shift horizon
    T - T0 = max(1250, 500 + N_s + 450), so every vehicle's window fully turns over (N_s = 300 reproduces T = 1700).
    The simultaneous budget uses the matching horizon K*T(N_s). Observed: mean per-vehicle maximum and run maximum (over all vehicles)."""
    import math, dataclasses
    from data import RegimeGenerator
    P_, Mm, RHO = 0.9, 0.02, 0.01; KT = K * E.T_DEFAULT
    laws = {ds: POOLS[ds] for ds in ["ngsim", "waymo"]}; laws["controlled"] = _controlled_law()
    runs = {"controlled": (RegimeGenerator, None), "ngsim": (full_backend(), "congested"), "waymo": (full_backend(), "moderate")}
    out = {"KT_note": "per N_s, see alpha_budget[N_s][KT]", "zeta": zeta, "m": Mm, "rho": RHO, "seeds": seeds, "n_windows": n_windows, "windows": list(windows),
           "alpha_budget": {}, "laws": {k: {} for k in laws}}
    for Ns in windows:
        T_ns = max(E.T0, Ns + 150) + max(E.T_DEFAULT - E.T0, E.STAGGER + Ns + 450); KT_ns = K * T_ns
        bud = _alpha_budget(Ns, KT_ns, zeta, P_); bud["KT"] = KT_ns; out["alpha_budget"][str(Ns)] = bud
        def qh(w, x): return float(w[min(max(math.ceil(x * Ns) - 1, 0), Ns - 1)])
        for k, a in laws.items():
            s_all = np.sort(a); n = len(s_all)
            def qa(x): return float(s_all[min(max(math.ceil(min(max(x, 0.0), 1.0) * n) - 1, 0), n - 1)])
            def om_true(d): return max(qa(P_ + d) - qa(P_), qa(P_) - qa(P_ - d))
            row = {"retro_step": round(Mm + om_true(bud["retro_step"]["delta"]) + RHO, 4),
                   "retro_sim": round(Mm + om_true(bud["retro_sim"]["delta"]) + RHO, 4)}
            for tag in ["online_step", "online_sim"]:
                d, eps = bud[tag]["delta"], bud[tag]["eps"]
                def Qup(w, x): return 1.0 if x + eps > 1 else qh(w, x + eps)
                def Qlo(w, x): return 0.0 if x - eps <= 0 else qh(w, x - eps)
                rng = np.random.default_rng(0); om = []
                for _ in range(n_windows):
                    w = np.sort(rng.choice(a, Ns, replace=True)); om.append(max(Qup(w, P_ + d) - Qlo(w, P_), Qup(w, P_) - Qlo(w, P_ - d)))
                row[tag] = round(Mm + float(np.median(om)) + RHO, 4)
            # observed under-protection of the proposed scheme at this N_s
            backend, post_reg = runs[k]; use_backend(backend)
            T0 = max(E.T0, Ns + 150); T = T0 + max(E.T_DEFAULT - E.T0, E.STAGGER + Ns + 450); p = dataclasses.replace(Params(), N_s=Ns)
            pv, rmax, rerr = [], [], []; rm = np.random.default_rng(seed0)
            for s in range(seeds):
                gen = E.RegimeGenerator(seed=int(rm.integers(1e9)))
                post = E.assign_post(K, True) if post_reg is None else [post_reg] * K
                R, Bs, st, tc = E.build_fleet(K, T, "freeflow", post, gen, T0, E.STAGGER)
                mt = E.metrics(simulate("P5", R, Bs, tc, p, gen.rng), st, p)
                pv.append(float(np.mean(mt["under_max"]))); rmax.append(float(np.max(mt["under_max"]))); rerr.append(float(np.mean(mt["rate_err"])))
            row["observed_per_vehicle_max"] = E.boot_ci(pv); row["observed_run_max"] = E.boot_ci(rmax)
            row["observed_run_max_worst_seed"] = round(float(np.max(rmax)), 4); row["observed_rate_err"] = E.boot_ci(rerr)
            row["T0"], row["T"] = int(T0), int(T)
            out["laws"][k][str(Ns)] = row
            print(Ns, k, row["retro_step"], row["online_step"], row["retro_sim"], row["online_sim"],
                  round(row["observed_per_vehicle_max"][0], 4), round(row["observed_run_max"][0], 4))
    use_backend(full_backend()); save("forensic_nsweep.json", out); return out


# ---------------------------------------------------------------- 10. quantile modulus of MIXED (transient) windows
def run_mixture(Ns=300, tol=0.005):
    """During the first Ns post-shift steps the window law is Fbar_k = (1-k/Ns) F_pre + (k/Ns) F_post (k post-shift samples).
    Retrospective envelope m + omega_{Fbar_k}(delta) + (Bbar_k - B*_post)_+ + rho for every k, per regime pair, per-step
    (alpha = 0.05) and simultaneous (alpha = zeta/(K T) = 0.05/34000). k_star = first k after which every later mixed-window
    envelope is within tol of the pure post-shift envelope."""
    from data import RegimeGenerator
    P_, Mm, RHO = 0.9, 0.02, 0.01; grid = np.linspace(0, 1, 20001)
    pools = {k: np.sort(v) for k, v in POOLS.items()}
    g = RegimeGenerator(seed=12345); ctrl = {r: np.sort(g.stream(r, 200000)) for r in ["freeflow", "congested", "rain", "dense", "night", "moderate"]}
    use_backend(full_backend())
    def cdf(x): return np.searchsorted(x, grid, "right") / len(x)
    def qinv(F, q): q = min(max(q, 0.0), 1.0); return float(grid[np.argmax(F >= q - 1e-12)]) if q > 0 else 0.0
    def om(F, d): return max(qinv(F, P_ + d) - qinv(F, P_), qinv(F, P_) - qinv(F, P_ - d))
    d_s, d_S = _delta_fp(0.05, Ns, P_), _delta_fp(0.05 / (K * E.T_DEFAULT), Ns, P_)
    pairs = [("highd", "ngsim", pools), ("highd", "waymo", pools)] + [("freeflow", r, ctrl) for r in ["congested", "rain", "dense", "night", "moderate"]]
    res = {}
    for a, b, L in pairs:
        Fa, Fb = cdf(L[a]), cdf(L[b]); bs = qinv(Fb, P_); ks = np.arange(0, Ns + 1); st, sm = [], []
        for k in ks:
            F = (1 - k / Ns) * Fa + (k / Ns) * Fb; dr = max(qinv(F, P_) - bs, 0.0)
            st.append(Mm + om(F, d_s) + dr + RHO); sm.append(Mm + om(F, d_S) + dr + RHO)
        st, sm = np.array(st), np.array(sm)
        kst = lambda arr: int(next(k for k in ks if np.all(arr[k:] <= arr[-1] + tol)))
        res[f"{a}->{b}"] = {"post_env_step": round(float(st[-1]), 4), "transient_env_step_max": round(float(st.max()), 4),
                            "k_at_max_step": int(st.argmax()), "k_star_step": kst(st), "post_env_sim": round(float(sm[-1]), 4),
                            "transient_env_sim_max": round(float(sm.max()), 4), "k_star_sim": kst(sm)}
        print(a, b, res[f"{a}->{b}"])
    save("forensic_mixture_modulus.json", {"Ns": Ns, "m": Mm, "rho": RHO, "tolerance": tol, "delta_step": d_s, "delta_sim": d_S,
         "note": run_mixture.__doc__, "pairs": res}); return res


# ---------------------------------------------------------------- 11. observed under-protection split: transient vs certified interval
def run_transient_split(seeds=30, seed0=7):
    """Proposed scheme (N_s = 300, T = 1700). Post-shift steps split into the transient interval [shift, shift + N_s), where the
    local window mixes pre- and post-shift scores (reported empirically), and the certified interval [shift + N_s, T), where the
    window holds only post-shift scores, i.i.d. given the source law. Reports the mean per-vehicle maximum and the run maximum in each."""
    from data import RegimeGenerator
    cert = json.load(open(os.path.join(RES, "certificates.json")))["laws"]; p = Params(); Ns = p.N_s
    runs = {"controlled": (RegimeGenerator, None), "ngsim": (full_backend(), "congested"), "waymo": (full_backend(), "moderate")}
    out = {"Ns": Ns, "seeds": seeds, "laws": {}}
    for law, (backend, post_reg) in runs.items():
        use_backend(backend); rm = np.random.default_rng(seed0); acc = {k: [] for k in ["tr_pv", "tr_run", "ce_pv", "ce_run"]}
        for s in range(seeds):
            gen = E.RegimeGenerator(seed=int(rm.integers(1e9))); post = E.assign_post(K, True) if post_reg is None else [post_reg] * K
            R, Bs, st, tc = E.build_fleet(K, E.T_DEFAULT, "freeflow", post, gen, E.T0, E.STAGGER)
            log = simulate("P5", R, Bs, tc, p, gen.rng); tr, ce = [], []
            for k in range(K):
                u = np.maximum(log["beff"][k] - Bs[k], 0.0); a = int(st[k])
                tr.append(float(u[a:a + Ns].max())); ce.append(float(u[a + Ns:].max()))
            acc["tr_pv"].append(np.mean(tr)); acc["tr_run"].append(np.max(tr)); acc["ce_pv"].append(np.mean(ce)); acc["ce_run"].append(np.max(ce))
        out["laws"][law] = {"env_step_certified_interval": cert[law]["env_step"], "env_sim_certified_interval": cert[law]["env_sim"],
                            "transient": {"per_vehicle_max": E.boot_ci(acc["tr_pv"]), "run_max": E.boot_ci(acc["tr_run"]), "worst_run": float(np.max(acc["tr_run"]))},
                            "certified_interval": {"per_vehicle_max": E.boot_ci(acc["ce_pv"]), "run_max": E.boot_ci(acc["ce_run"]), "worst_run": float(np.max(acc["ce_run"]))}}
        print(law, {k: round(v[0] if isinstance(v, list) else v, 4) for kk in ["transient", "certified_interval"] for k, v in [(kk + "_pv", out["laws"][law][kk]["per_vehicle_max"]), (kk + "_run", out["laws"][law][kk]["run_max"]), (kk + "_worst", out["laws"][law][kk]["worst_run"])]})
    use_backend(full_backend()); save("forensic_transient_split.json", out); return out


# ---------------------------------------------------------------- 12. inferential roles: Holm-adjusted confirmatory family
def run_holm():
    """Holm-Bonferroni adjustment of the confirmatory family: combined-fault deployment test, proposed - naive fleet-clamp,
    on collisions, rate error and under-protection (paired sign-flip p from forensic_deploy_paired.json; p is floored at
    1/100000, the permutation resolution). Other paired tests are collected with their unadjusted p for the inference table."""
    dp = json.load(open(os.path.join(RES, "forensic_deploy_paired.json")))
    fam = {k: dp["P5_minus_B2"][k] for k in ["coll", "rate_err", "under_max"]}
    ks = sorted(fam, key=lambda k: fam[k]["p_signflip"]); m = len(ks); adj = {}; run_max = 0.0
    for i, k in enumerate(ks):
        run_max = max(run_max, min(1.0, (m - i) * fam[k]["p_signflip"])); adj[k] = run_max
    cl = json.load(open(os.path.join(RES, "forensic_closed_paired.json")))
    out = {"confirmatory_family": "combined-fault deployment, proposed - naive fleet-clamp (collisions, rate error, under-protection)",
           "confirmatory": {k: {"mean_diff": fam[k]["mean_diff"], "p_raw": fam[k]["p_signflip"], "p_holm": adj[k]} for k in fam},
           "secondary_unadjusted": {"deploy_proposed_minus_local_only": dp["P5_minus_B1"], "closed_loop_paired": cl}}
    save("forensic_inference_roles.json", out); print(json.dumps(out["confirmatory"], indent=1)); return out


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    fns = {"split": run_split, "split70": lambda: run_split(frac_ref=0.7, outname="forensic_split70.json"), "closed": run_closed, "closed4": lambda: run_closed(score="4", outname="forensic_closed_paired_4feat.json"), "downward": run_downward, "coverage": run_coverage, "online": run_online, "drift": run_drift, "rho": run_rho, "robustclamp": run_robustclamp, "online_sim": run_online_sim, "deploy_paired": run_deploy_paired, "hetero": run_hetero, "nsweep": run_nsweep, "mixture": run_mixture, "transient_split": run_transient_split, "holm": run_holm}
    for k in (fns if what == "all" else [what]): fns[k]()
