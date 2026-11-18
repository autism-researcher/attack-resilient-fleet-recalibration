"""Run Experiments A-D on streams re-derived from a raw NGSIM CSV (for Experiments E-G use experiments_efg.py).
Usage:  python run_ngsim.py --csv path/to/NGSIM*.csv --seeds 15
Outputs land in results_real/ and figures_real/."""
import os, glob, argparse, warnings, numpy as np
warnings.filterwarnings("ignore")
import ngsim_real as N
import experiments as E

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True, help="NGSIM CSV path(s) or glob")
    ap.add_argument("--seeds", type=int, default=15)
    a = ap.parse_args()
    streams = []
    for pat in a.csv:
        for fp in sorted(glob.glob(pat)):
            print("loading", fp, "..."); streams += N.load_scored_streams(fp)
    print(f"{len(streams)} real follower streams loaded")
    pools = N.build_regime_pools(streams); N.NGSIMRegimes.POOLS = pools

    print("\n--- risk-score validation on REAL NGSIM (tau=0.10) ---")
    allR = np.concatenate(list(pools.values()))
    print(f"overall R: mean={allR.mean():.3f}  median={np.median(allR):.3f}  90th pct={np.quantile(allR,0.9):.3f}")
    for k in ["low","mid","high"]:
        p = pools[k]; q = np.quantile(p, 0.9)
        dens = ((np.abs(p-q) < 0.03).mean())/0.06
        print(f"  {k:>4} density regime (n={len(p):>8}): mean R={p.mean():.3f}  B*={q:.3f}  f(B*)~{dens:.2f}")

    # route experiments through the real-data backend
    E.RegimeGenerator = N.NGSIMRegimes; E._BSTAR = {}
    base = os.path.dirname(os.path.abspath(__file__))
    E.FIG = os.path.join(base, "figures_real"); E.RES = os.path.join(base, "results_real")
    os.makedirs(E.FIG, exist_ok=True); os.makedirs(E.RES, exist_ok=True)

    print("\nrunning A-D on real data ...")
    E.expA(a.seeds); E.expB(a.seeds); E.expC(a.seeds); E.expD(a.seeds)
    E.merge()
    print("\nDONE. Real-data results in results_real/, figures in figures_real/.")

if __name__ == "__main__": main()
