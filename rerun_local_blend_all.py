"""Paper 4: Matched Local Self-Blending control (code label B1S) added to every
fleet-vs-local comparison in the manuscript. Frozen code is imported unchanged.

  python rerun_local_blend_all.py controlled   # controlled generator, Exp. B (latency) and D (rate error)
  python rerun_local_blend_all.py realBD       # same on the real pools (seed stream restarted per K)
  python rerun_local_blend_all.py realA        # real Exp. A (rate error, missed high-risk, under-protection)
  python rerun_local_blend_all.py closed       # closed-loop IDM microsimulation (12 seeds, 100-111)
  python rerun_local_blend_all.py hetero       # heterogeneity x fleet size (forensic_checks hetero + B1S); slow
Outputs: results_real/localblend_<name>.json
"""
import sys, json, os, inspect
import numpy as np
import experiments as E
from recalibration import Params, simulate
from rerun_local_blend import simulate_b1s

OUT = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(OUT, "results_real")


def sim(c, R, Bs, tc, p, rng):
    return simulate_b1s(R, Bs, tc, p) if c == "B1S" else simulate(c, R, Bs, tc, p, rng)


def paired(a, b):
    import forensic_checks as F
    return F.paired(a, b)


def save(name, obj):
    with open(os.path.join(RES, f"localblend_{name}.json"), "w") as f: json.dump(obj, f, indent=1)
    print("saved", f"results_real/localblend_{name}.json")


def run_controlled(seeds=30, real=False):
    """Controlled generator (no --real). B: latency vs K (seed0 17); D: rate error vs K (seed0 37). Paired.
    rerun_local_blend.py patches the real backend on import, so the controlled generator is restored here."""
    if real:
        import real_backend; real_backend.patch()
    else:
        import data; E.RegimeGenerator = data.RegimeGenerator; E._BSTAR = {}
    p = Params(); out = {"seeds": seeds, "backend": "real pools" if real else "controlled NGSIM-range-informed generator"}
    for exp, Ks, seed0, metric in [("B", E.FLEETS_B, 17, "latency"), ("D", E.FLEETS_D, 37, "rate_err")]:
        out[exp] = {}
        for K in Ks:
            rm = np.random.default_rng(seed0); acc = {c: [] for c in ["B1", "B1S", "P5"]}
            for s in range(seeds):
                gen = E.RegimeGenerator(seed=int(rm.integers(1e9))); post = E.assign_post(K, True)
                R, Bs, st, tc = E.build_fleet(K, E.T_DEFAULT, "freeflow", post, gen, E.T0, E.STAGGER)
                for c in acc: acc[c].append(float(np.mean(E.metrics(sim(c, R, Bs, tc, p, gen.rng), st, p)[metric])))
            cell = {c: E.boot_ci(v) for c, v in acc.items()}
            cell["blend_effect"] = paired(acc["B1"], acc["B1S"]); cell["collab_value"] = paired(acc["B1S"], acc["P5"])
            out[exp][str(K)] = cell
            print(exp, metric, K, {c: round(cell[c][0], 4) for c in acc},
                  "collab", [round(x, 4) for x in cell["collab_value"]["mean_diff"]], "p", cell["collab_value"]["p_signflip"], flush=True)
    save("real_BD_perK_seeds" if real else "controlled", out)


def run_realA(seeds=30):
    """Real Exp. A setting (K=20, heterogeneous post, seed0 7). B1 and P5 reproduce results_real/A.json at 15 seeds."""
    import real_backend; real_backend.patch()
    p = Params(); rm = np.random.default_rng(7); K = 20
    keys = ["rate_err", "missed", "under_mean", "under_max", "latency"]
    acc = {c: {k: [] for k in keys} for c in ["B1", "B1S", "P5"]}
    for s in range(seeds):
        gen = E.RegimeGenerator(seed=int(rm.integers(1e9))); post = E.assign_post(K, True)
        R, Bs, st, tc = E.build_fleet(K, E.T_DEFAULT, "freeflow", post, gen, E.T0, E.STAGGER)
        for c in acc:
            m = E.metrics(sim(c, R, Bs, tc, p, gen.rng), st, p)
            for k in keys: acc[c][k].append(float(np.mean(m[k])))
    out = {"seeds": seeds}
    for c in acc: out[c] = {k: E.boot_ci(v) for k, v in acc[c].items()}
    out["first15_check"] = {c: {k: float(np.mean(acc[c][k][:15])) for k in ["rate_err", "missed"]} for c in ["B1", "P5"]}
    out["blend_effect"] = {k: paired(acc["B1"][k], acc["B1S"][k]) for k in keys}
    out["collab_value"] = {k: paired(acc["B1S"][k], acc["P5"][k]) for k in keys}
    for c in acc: print(c, {k: round(out[c][k][0], 4) for k in keys})
    print("collab (Local-Blend - Fleet):", {k: [round(x, 4) for x in out["collab_value"][k]["mean_diff"]] for k in keys})
    print("first-15 check vs A.json:", out["first15_check"])
    save("realA", out)


def _closed_with_b1s():
    """closed_loop.simulate with one added condition B1S: every P steps B <- 0.5 B + 0.5 U (own local quantile)."""
    import closed_loop as CL
    src = inspect.getsource(CL.simulate)
    src = src.replace('if cond in ("B1","P5","FedAvg","RobFCP"):', 'if cond in ("B1","B1S","P5","FedAvg","RobFCP"):')
    anchor = '        if cond=="FedAvg" and t>0 and t%P==0:'
    assert anchor in src and 'if cond in ("B1","B1S"' in src
    src = src.replace(anchor, '        if cond=="B1S" and t>0 and t%P==0:\n            B=np.clip(0.5*B+0.5*U,0,1)  # matched local self-blending (gate 1, own U)\n' + anchor)
    ns = vars(CL); exec(compile(src, "closed_loop_b1s", "exec"), ns)
    return CL, ns["simulate"]


def run_closed(seeds=12, K=20, T=3000):
    CL, simulate_cl = _closed_with_b1s()
    Bpre = CL.bstar("mild", np.random.default_rng(1)); Bpost = CL.bstar("aggr", np.random.default_rng(2)); B0c = np.full(K, Bpre)
    conds = ["B1", "B1S", "P5"]; per = {c: {k: [] for k in ["coll", "rate_err", "under_max", "speed"]} for c in conds}
    for c in conds:
        for s in range(seeds):
            r = np.random.default_rng(100 + s); leads, shift = CL.build_leads(K, T, T // 3, T // 4, r)
            m = simulate_cl(c, leads, shift, r, Bpre, Bpost, B0c)
            for k in per[c]: per[c][k].append(float(m[k]))
    out = {"seeds": seeds, "per_seed": per, "summary": {c: {k: CL.ci(v) for k, v in per[c].items()} for c in conds},
           "blend_effect": {k: paired(per["B1"][k], per["B1S"][k]) for k in ["coll", "rate_err", "under_max"]},
           "collab_value": {k: paired(per["B1S"][k], per["P5"][k]) for k in ["coll", "rate_err", "under_max"]}}
    for c in conds: print(c, {k: round(out["summary"][c][k][0], 4) for k in per[c]})
    print("Local-Only - Local-Blend:", {k: [round(x, 4) for x in v["mean_diff"]] + [v["p_signflip"]] for k, v in out["blend_effect"].items()})
    print("Local-Blend - Fleet:     ", {k: [round(x, 4) for x in v["mean_diff"]] + [v["p_signflip"]] for k, v in out["collab_value"].items()})
    save("closed", out)


def run_hetero(seeds=30, Ks=(5, 10, 20, 50, 100), seed0=91):
    import forensic_checks as F
    from data import RegimeGenerator
    setups = {"real": (F.full_backend(), "congested", ["moderate"], (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)),
              "controlled": (RegimeGenerator, "congested", ["rain", "dense", "night", "moderate"], (0.0, 0.2, 0.4, 0.6, 0.8))}
    p = Params(); out = {"seeds": seeds, "Ks": list(Ks), "setups": {}}
    for name, (backend, base, alts, fracs) in setups.items():
        F.use_backend(backend); blk = {"fracs": list(fracs), "cells": {}}
        for K_ in Ks:
            for fr in fracs:
                post = F._post_assign(K_, fr, alts, base)
                acc = {c: {"latency": [], "rate_err": [], "under_max": []} for c in ["B1", "B1S", "P5"]}
                rm = np.random.default_rng(seed0)
                for s in range(seeds):
                    gen = E.RegimeGenerator(seed=int(rm.integers(1e9)))
                    R, Bs, st, tc = E.build_fleet(K_, E.T_DEFAULT, "freeflow", post, gen, E.T0, E.STAGGER)
                    for c in acc:
                        m = E.metrics(sim(c, R, Bs, tc, p, gen.rng), st, p)
                        for k in acc[c]: acc[c][k].append(float(np.mean(m[k])))
                cell = {c: {k: E.boot_ci(v) for k, v in acc[c].items()} for c in acc}
                cell["P5_minus_B1"] = {k: paired(acc["P5"][k], acc["B1"][k]) for k in ["latency", "rate_err"]}
                cell["P5_minus_B1S"] = {k: paired(acc["P5"][k], acc["B1S"][k]) for k in ["latency", "rate_err"]}
                blk["cells"][f"K{K_}_f{fr}"] = cell
                print(name, K_, fr, "fleet-vs-local lat", round(cell["P5_minus_B1"]["latency"]["mean_diff"][0], 1),
                      "fleet-vs-localblend lat", round(cell["P5_minus_B1S"]["latency"]["mean_diff"][0], 1),
                      "re", round(cell["P5_minus_B1S"]["rate_err"]["mean_diff"][0], 5), flush=True)
        out["setups"][name] = blk
    save("hetero", out)


if __name__ == "__main__":
    {"controlled": run_controlled, "realBD": lambda: run_controlled(real=True), "realA": run_realA, "closed": run_closed, "hetero": run_hetero}[sys.argv[1]]()
