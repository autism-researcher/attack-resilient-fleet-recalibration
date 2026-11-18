"""
Real-NGSIM backend for the P5 study. Turns the public NGSIM vehicle-trajectory
CSV into per-regime risk-score pools with REAL empirical matching boundaries,
exposing the same interface as the NGSIM-range-informed generator so experiments
A-G run unchanged on real data.

Regimes are defined directly from the data by binning each follower stream's
median time-headway into terciles (low / mid / high traffic density). A
distribution shift (e.g. freeflow -> congested) is then a real low-density ->
high-density change, and each regime's ground-truth matching boundary B* is the
empirical (1-tau)-quantile of that regime's real risk scores.
"""
import numpy as np, pandas as pd
from risk_score import risk_score

def load_scored_streams(csv_path, dt=0.1):
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    def col(*names):
        for n in names:
            if n.lower() in cols: return cols[n.lower()]
        raise KeyError(f"need one of {names}; got {list(df.columns)[:12]}")
    vid_c = col("Vehicle_ID","Vehicle_Id","ID")
    sh_c  = col("Space_Headway","SpaceHeadway","spacing")
    v_c   = col("v_Vel","v_vel","Velocity","Speed")
    sh = df[sh_c].to_numpy(float)
    if np.nanmedian(sh) > 30: sh = sh * 0.3048      # feet -> m
    v = df[v_c].to_numpy(float)
    if np.nanmedian(v) > 40: v = v * 0.3048         # ft/s -> m/s
    streams = []
    for _, g in df.groupby(df[vid_c]):
        idx = g.index.to_numpy()
        gp = np.clip(sh[idx], 1.0, None); vv = np.clip(v[idx], 0.0, None)
        if len(gp) < 50: continue
        dv = np.gradient(-gp, dt)                   # gap-closing rate proxy
        R = risk_score(gp, vv, dv)
        thw = np.median(gp) / max(np.median(vv), 0.5)
        streams.append((R, float(thw)))
    return streams

def build_regime_pools(streams):
    if not streams: raise ValueError("no usable follower streams found")
    thw = np.array([s[1] for s in streams])
    q1, q2 = np.quantile(thw, [1/3, 2/3])
    buckets = {"high": [], "mid": [], "low": []}     # small headway = high density
    for R, h in streams:
        k = "high" if h <= q1 else ("mid" if h <= q2 else "low")
        buckets[k].append(R)
    return {k: (np.concatenate(v) if v else np.array([0.05])) for k, v in buckets.items()}

NAME_MAP = {"freeflow":"low","night":"low","moderate":"mid",
            "congested":"high","dense":"high","rain":"high"}

class NGSIMRegimes:
    """Drop-in for RegimeGenerator, backed by real NGSIM score pools."""
    POOLS = None
    def __init__(self, seed=0): self.rng = np.random.default_rng(seed)
    def _pool(self, regime): return NGSIMRegimes.POOLS[NAME_MAP.get(regime, "mid")]
    def stream(self, regime, n): return self.rng.choice(self._pool(regime), size=n, replace=True)
    def true_matching_boundary(self, regime, tau, n=0): return float(np.quantile(self._pool(regime), 1.0 - tau))
