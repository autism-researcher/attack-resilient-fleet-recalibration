"""Paper 4: rerun of the adaptation-latency comparison with the matched
Local-Blend control (Matched Local Self-Blending Recalibration, code label B1S).

B1S uses exactly the fleet blend rule of the proposed scheme (P5), but the blend
target is the vehicle's own local quantile (gate = 1), i.e. no communication.
Frozen code (recalibration.py, experiments.py) is imported unchanged; only the
B1S condition is added here.

Same seeds and fleet construction as Experiment B (--real), plus B1S, so all three
conditions are paired on identical streams.

  python rerun_local_blend.py --seeds 15            -> results_real/B_localblend.json
"""
import argparse, json, os
import numpy as np
import real_backend
E = real_backend.patch()
from recalibration import Params, simulate, _local_quantile


def simulate_b1s(R, Bstar, t_cal, p):
    """Matched Local Self-Blending: identical to P5 with U_fleet := U_local."""
    K, T = R.shape
    pre = R[:, :t_cal]
    B = np.quantile(pre, 1.0 - p.tau, axis=1)
    window = pre[:, -p.N_s:].copy() if pre.shape[1] >= p.N_s else np.repeat(pre[:, -1:], p.N_s, axis=1).copy()
    beff_log = np.zeros((K, T)); inter_log = np.zeros((K, T), bool)
    Q = 5; ptr = 0; U_local = _local_quantile(window, p.tau)
    for t in range(T):
        Rt = R[:, t]
        if t % Q == 0:
            U_local = _local_quantile(window, p.tau)
        Beff = np.minimum(B, U_local + p.m)
        inter = Rt > Beff; inter_log[:, t] = inter; beff_log[:, t] = Beff
        B = np.clip(B + p.gamma * (inter.astype(float) - p.tau), 0.0, 1.0)
        if t > 0 and t % p.P == 0:
            B = np.clip((1 - p.lam) * B + p.lam * U_local, 0.0, 1.0)
        window[:, ptr] = Rt; ptr = (ptr + 1) % p.N_s
    return dict(intervened=inter_log, beff=beff_log, Bstar=Bstar, R=R, params=p)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--seed0", type=int, default=17)   # same as Experiment B
    a = ap.parse_args()
    p = Params(); rm = np.random.default_rng(a.seed0)
    out = {}
    for K in E.FLEETS_B:
        rows = {"B1": [], "B1S": [], "P5": []}
        # Experiment B draws the generator seed separately for each condition; here we draw
        # ONE seed per replicate and run all three conditions on the same fleet (paired).
        for s in range(a.seeds):
            seed = int(rm.integers(1e9))
            gen = E.RegimeGenerator(seed=seed); post = E.assign_post(K, True)
            R, Bs, st, tc = E.build_fleet(K, E.T_DEFAULT, "freeflow", post, gen, E.T0, E.STAGGER)
            rows["B1"].append(np.mean(E.metrics(simulate("B1", R, Bs, tc, p, gen.rng), st, p)["latency"]))
            rows["P5"].append(np.mean(E.metrics(simulate("P5", R, Bs, tc, p, gen.rng), st, p)["latency"]))
            rows["B1S"].append(np.mean(E.metrics(simulate_b1s(R, Bs, tc, p), st, p)["latency"]))
        d_blend = np.array(rows["B1"]) - np.array(rows["B1S"])      # local self-blending effect
        d_fleet = np.array(rows["B1S"]) - np.array(rows["P5"])      # collaboration value
        out[str(K)] = {c: E.boot_ci(v) for c, v in rows.items()}
        out[str(K)]["dL_blend"] = E.boot_ci(d_blend); out[str(K)]["dL_fleet"] = E.boot_ci(d_fleet)
        out[str(K)]["raw"] = {k: [float(x) for x in v] for k, v in rows.items()}
        f = lambda c: f"{c[0]:7.1f} [{c[1]:6.1f},{c[2]:6.1f}]"
        print(f"K={K:3d}  Local-Only {f(out[str(K)]['B1'])}  Local-Blend {f(out[str(K)]['B1S'])}  Fleet-Gated {f(out[str(K)]['P5'])}"
              f"  | blend effect {f(out[str(K)]['dL_blend'])}  collaboration value {f(out[str(K)]['dL_fleet'])}", flush=True)
    os.makedirs("results_real", exist_ok=True)
    json.dump(out, open(os.path.join("results_real", "B_localblend.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
