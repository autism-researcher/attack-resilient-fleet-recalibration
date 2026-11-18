#!/usr/bin/env python3
"""
One-command runner for the reviewer-strengthening experiments.
Reproduces every new table (component ablation, deployment stress test,
Waymo held-out, V2X fault grid, traffic metrics, sensitivity) from the
committed code and the real risk pools. All numbers are produced by this run.

USAGE (from the P5_fleet_experiment/ folder):

    python run_reviewer_experiments.py --seeds 30

  optionally re-derive the NGSIM boundary from your raw CSV:
    python run_reviewer_experiments.py --seeds 30 \
        --ngsim-csv "D:/New Paper3/paper3_pipeline/data/ngsim/ngsim_trajectories.csv"

Outputs: results_real/reviewer_ablation.json, reviewer_waymo_holdout.json,
         results_real/reviewer_closedloop.json  (+ printed tables).

Runtime: ~10-20 min at --seeds 30 on a desktop CPU (closed loop dominates).
Note: this machine has no per-call time limit, so it runs straight through;
the chunked driver run_block.py is only needed inside the sandbox.
"""
import argparse, json, os, subprocess, sys, time
import numpy as np
import closed_loop as CL
import deploy_experiments as DX
from run_block import units_for, METRICS

OUT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(OUT, "results_real"); os.makedirs(RES, exist_ok=True)

def cache_bstar():
    Bpre = CL.bstar("mild", np.random.default_rng(1))
    Bpost = CL.bstar("aggr", np.random.default_rng(2))
    json.dump({"Bpre": Bpre, "Bpost": Bpost, "score": CL.SCORE}, open(os.path.join(RES, "_bstar.json" if CL.SCORE == "8" else "_bstar_4feat.json"), "w"))
    print(f"[bstar] B*_pre(mild)={Bpre:.3f}  B*_post(aggr)={Bpost:.3f}")
    return Bpre, Bpost

def ci(vals):
    v = np.array([x for x in vals if x == x])
    if len(v) < 2: return dict(mean=float(v.mean()) if len(v) else float("nan"), std=0.0, lo=float("nan"), hi=float("nan"), n=len(v))
    rng = np.random.default_rng(0); bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(3000)]
    return dict(mean=float(v.mean()), std=float(v.std(ddof=1)), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), n=int(len(v)))

def run_closed_loop(seeds, Bpre, Bpost):
    """Run every closed-loop block (paired seeds) and save one JSON."""
    blocks = ["abl0", "abl25", "deploy", "v2x",
              "sens_adv", "sens_K", "sens_Ns", "sens_m", "sens_tau", "sens_P"]
    acc = {}
    for blk in blocks:
        for name, kw in units_for(blk):
            cond = kw.pop("cond"); K = kw.pop("K", 20)
            rec = {m: [] for m in METRICS}
            for i in range(seeds):
                r = np.random.default_rng(100 + i)
                leads, shift = DX.build_leads(K, 3000, 1000, 750, r)
                m = DX.simulate(cond, leads, shift, r, Bpre, Bpost, np.full(K, Bpre), **kw)
                for mt in METRICS: rec[mt].append(m[mt])
            acc[name] = rec
        print(f"[closed-loop] {blk} done ({seeds} seeds)")
    summ = {u: {m: ci(acc[u][m]) for m in METRICS} for u in acc}
    json.dump(summ, open(os.path.join(RES, "reviewer_closedloop.json" if CL.SCORE == "8" else "reviewer_closedloop_4feat.json"), "w"), indent=2)
    return summ

def run_resampled(seeds):
    for what in ("ablation", "holdout"):
        subprocess.run([sys.executable, os.path.join(OUT, "resampled_ablation.py"), what, str(seeds)], check=True)

def maybe_ngsim(csv):
    if not csv: return
    print(f"\n[NGSIM] re-deriving boundary from raw CSV: {csv}")
    subprocess.run([sys.executable, os.path.join(OUT, "run_ngsim.py"), "--csv", csv, "--seeds", "15"], check=True)

def report(summ):
    def M(u, k): return summ[u][k]["mean"]
    print("\n================ DEPLOYMENT (30% delayed + 20% malicious) ================")
    for lbl, u in [("AEB", "deploy_AEB"), ("frozen", "deploy_B0"), ("local-only", "deploy_B1"), ("naive B2", "deploy_B2"), ("P5", "deploy_P5")]:
        print(f"  {lbl:<12} coll={M(u,'coll'):.3f}  under={M(u,'under_max'):.3f}  rate_err={M(u,'rate_err'):.4f}")
    print("\n================ V2X FAULT GRID (P5) ================")
    for u in ["P5_delay100","P5_delay300","P5_delay500","P5_delay1000","P5_loss5","P5_loss10","P5_loss20","P5_loss40","P5_stale","P5_byz20","P5_mixed"]:
        print(f"  {u[3:]:<12} coll={M(u,'coll'):.3f}  under={M(u,'under_max'):.3f}  rate_err={M(u,'rate_err'):.4f}")
    print("\nSee results_real/reviewer_ablation.json + reviewer_waymo_holdout.json for the real-pool ablation and held-out test.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--ngsim-csv", default=None, help="raw NGSIM CSV path to re-derive B* (optional)")
    a = ap.parse_args()
    t0 = time.time()
    Bpre, Bpost = cache_bstar()
    run_resampled(a.seeds)              # real-pool component ablation + Waymo held-out
    summ = run_closed_loop(a.seeds, Bpre, Bpost)  # deployment, V2X grid, traffic metrics, sensitivity
    maybe_ngsim(a.ngsim_csv)
    report(summ)
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min. All numbers from this run (no hardcoding).")
