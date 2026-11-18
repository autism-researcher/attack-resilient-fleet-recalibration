"""Paper 4: rate-error claims (Exp. D, Exp. G) rerun with the matched Local-Blend
control (code label B1S). Frozen code imported unchanged; all three conditions
run on the same fleet per replicate (paired).

  python rerun_local_blend_rate.py --seeds 30   -> results_real/DG_localblend.json
"""
import argparse, json, os
import numpy as np
import real_backend
E = real_backend.patch()
from recalibration import Params, simulate
from rerun_local_blend import simulate_b1s

CONDS = ["B1", "B1S", "P5"]

def run(K, het, seeds, seed0):
    p = Params(); rm = np.random.default_rng(seed0)
    rows = {c: {"rate_err": [], "latency": []} for c in CONDS}
    for s in range(seeds):
        gen = E.RegimeGenerator(seed=int(rm.integers(1e9))); post = E.assign_post(K, het)
        R, Bs, st, tc = E.build_fleet(K, E.T_DEFAULT, "freeflow", post, gen, E.T0, E.STAGGER)
        logs = {"B1": simulate("B1", R, Bs, tc, p, gen.rng), "P5": simulate("P5", R, Bs, tc, p, gen.rng),
                "B1S": simulate_b1s(R, Bs, tc, p)}
        for c in CONDS:
            m = E.metrics(logs[c], st, p)
            for k in rows[c]: rows[c][k].append(float(np.mean(m[k])))
    out = {c: {k: E.boot_ci(v) for k, v in rows[c].items()} for c in CONDS}
    for k in ["rate_err", "latency"]:
        a = {c: np.array(rows[c][k]) for c in CONDS}
        out[f"{k}_blend_effect"] = E.boot_ci(a["B1"] - a["B1S"])
        out[f"{k}_collab_value"] = E.boot_ci(a["B1S"] - a["P5"])
    out["raw"] = rows
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=30); a = ap.parse_args()
    f = lambda c, s=1e3: f"{c[0]*s:6.2f} [{c[1]*s:6.2f},{c[2]*s:6.2f}]"
    res = {"D": {}, "G": {}, "seeds": a.seeds, "units": "rate_err printed x1e-3"}
    print("Exp D (heterogeneous post-shift), rate error x1e-3")
    for K in E.FLEETS_D:
        o = run(K, True, a.seeds, 37); res["D"][str(K)] = o
        print(f"K={K:3d} Local-Only {f(o['B1']['rate_err'])} Local-Blend {f(o['B1S']['rate_err'])} Fleet-Gated {f(o['P5']['rate_err'])}"
              f" | blend effect {f(o['rate_err_blend_effect'])} collaboration value {f(o['rate_err_collab_value'])}", flush=True)
    print("Exp G (K=20), rate error x1e-3")
    for name, het in [("heterogeneous", True), ("homogeneous", False)]:
        o = run(20, het, a.seeds, 67); res["G"][name] = o
        print(f"{name:13s} Local-Only {f(o['B1']['rate_err'])} Local-Blend {f(o['B1S']['rate_err'])} Fleet-Gated {f(o['P5']['rate_err'])}"
              f" | blend effect {f(o['rate_err_blend_effect'])} collaboration value {f(o['rate_err_collab_value'])}", flush=True)
    os.makedirs("results_real", exist_ok=True)
    json.dump(res, open(os.path.join("results_real", "DG_localblend.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
