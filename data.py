"""
Data layer. Two interchangeable sources emitting per-step kinematics
(gap, v, dv) that risk_score() turns into R in [0,1]:
  (1) load_ngsim(csv): reads the public NGSIM trajectory CSV and reconstructs
      follower streams (REAL data path).
  (2) RegimeGenerator: NGSIM-range-informed stochastic generator supporting
      controlled distribution shift and a known ground-truth matching boundary.
"""
import numpy as np
import pandas as pd
from risk_score import risk_score

REGIMES = {
    "freeflow": (26.0, 3.0, 45.0, 14.0, -0.2, 1.2),
    "moderate": (18.0, 3.5, 28.0, 10.0,  0.2, 1.6),
    "congested":(9.0,  3.0, 12.0,  5.0,  0.6, 1.8),
    "rain":     (15.0, 3.5, 20.0,  8.0,  0.7, 2.0),
    "dense":    (7.0,  2.5,  8.5,  3.5,  0.9, 2.0),
    "night":    (22.0, 4.0, 34.0, 12.0,  0.3, 1.8),
}
REGIME_LIST = list(REGIMES.keys())

class RegimeGenerator:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
    def sample_kinematics(self, regime, n):
        v_m, v_s, g_m, g_s, dv_m, dv_s = REGIMES[regime]
        v = np.clip(self.rng.normal(v_m, v_s, n), 0.5, 40)
        gap = np.clip(self.rng.normal(g_m, g_s, n), 1.0, 120)
        dv = self.rng.normal(dv_m, dv_s, n)
        jerk = self.rng.normal(0, 1.0, n)
        return gap, v, dv, jerk
    def stream(self, regime, n):
        gap, v, dv, jerk = self.sample_kinematics(regime, n)
        return risk_score(gap, v, dv, jerk=jerk)
    def true_matching_boundary(self, regime, tau, n=300000):
        R = self.stream(regime, n)
        return float(np.quantile(R, 1.0 - tau))

def load_ngsim(csv_path, dt=0.1):
    df = pd.read_csv(csv_path)
    sh = df["Space_Headway"].to_numpy(float)
    if np.nanmedian(sh) > 30: sh = sh * 0.3048
    v = df["v_Vel"].to_numpy(float)
    if np.nanmedian(v) > 40: v = v * 0.3048
    streams = []
    for vid, g in df.groupby("Vehicle_ID"):
        idx = g.index.to_numpy()
        gp = np.clip(sh[idx], 1.0, None); vv = v[idx]
        dv = np.gradient(-gp, dt)
        R = risk_score(gp, vv, dv)
        if len(R) > 50: streams.append(R)
    return {"ngsim_empirical": streams}
