"""Backend driving the P5 study from REAL per-dataset risk pools (highD/NGSIM/Waymo)."""
import os, numpy as np
import experiments as E
OUT=os.path.dirname(os.path.abspath(__file__))
POOLS={ds:np.load(os.path.join(OUT,"real",f"{ds}.npy")) for ds in ["highd","ngsim","waymo"]}
NAME_MAP={"freeflow":"highd","moderate":"waymo","night":"waymo","congested":"ngsim","dense":"ngsim","rain":"ngsim"}
class DatasetRegimes:
    def __init__(self, seed=0): self.rng=np.random.default_rng(seed)
    def _pool(self, regime): return POOLS[NAME_MAP.get(regime,"waymo")]
    def stream(self, regime, n): return self.rng.choice(self._pool(regime), size=n, replace=True)
    def true_matching_boundary(self, regime, tau, n=0): return float(np.quantile(self._pool(regime), 1.0-tau))
def patch():
    E.RegimeGenerator=DatasetRegimes; E._BSTAR={}
    E.FIG=os.path.join(OUT,"figures_real"); E.RES=os.path.join(OUT,"results_real")
    os.makedirs(E.FIG,exist_ok=True); os.makedirs(E.RES,exist_ok=True); return E
