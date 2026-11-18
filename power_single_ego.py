"""Spread of the per-vehicle collaboration value (one ego) -> sessions needed on the car."""
import numpy as np, real_backend
E = real_backend.patch()
from recalibration import Params, simulate
from rerun_local_blend import simulate_b1s
p = Params()
for K in (5, 20):
    for het in (True, False):
        rm = np.random.default_rng(99); d = []; dfull = []
        for s in range(30):
            gen = E.RegimeGenerator(seed=int(rm.integers(1e9))); post = E.assign_post(K + 1, het)
            R, Bs, st, tc = E.build_fleet(K + 1, E.T_DEFAULT, "freeflow", post, gen, E.T0, E.STAGGER)
            lb = E.metrics(simulate_b1s(R, Bs, tc, p), st, p)["latency"]
            fl = E.metrics(simulate("P5", R, Bs, tc, p, gen.rng), st, p)["latency"]
            d += list(lb - fl)
        d = np.array(d); m, sd = d.mean(), d.std(ddof=1)
        n = int(np.ceil(((1.96 + 0.84) * sd / m) ** 2)) if m > 0 else None
        print(f"K={K:2d} partners {'hetero' if het else 'homog '}: per-ego dL_fleet mean {m:6.1f}  SD {sd:6.1f}  "
              f"-> sessions for 80% power: {n}")
