"""Experiments E-G (restored drivers, release v1.0.4).

  E : adversarial-upload fraction sweep (0-50%), naive fleet-clamp vs proposed -> max under-protection
  F : aggregation period P in {10, 20, 50, 100}, proposed scheme -> rate error, latency, max under-protection
  G : heterogeneous vs homogeneous fleet shift, local-only vs proposed -> rate error, latency

Same machinery as experiments.py (build_fleet, metrics, boot_ci, theoretical_bound); K = 20.
"Max under-protection" is the mean over vehicles of each vehicle's post-shift maximum (experiments.metrics).
Conditions share the same risk streams within a seed (paired design).

Usage:
  python experiments_efg.py --real --seeds 15            # highD/NGSIM/Waymo pools -> results_real/{E,F,G}.json
  python experiments_efg.py --seeds 15                   # controlled generator     -> results/{E,F,G}.json
  python experiments_efg.py --real --only G --seeds 15
"""
import argparse, dataclasses, warnings
import numpy as np

FRACS = [0.0, 0.1, 0.2, 0.3, 0.5]
PERIODS = [10, 20, 50, 100]
K = 20


def expE(E, seeds, seed0=47):
    from recalibration import Params, simulate
    p = Params(); rm = np.random.default_rng(seed0)
    acc = {c: {str(f): [] for f in FRACS} for c in ["B2", "P5"]}; bnd = []
    for s in range(seeds):
        gen = E.RegimeGenerator(seed=int(rm.integers(1e9))); post = E.assign_post(K, True)
        R, Bs, st, tc = E.build_fleet(K, E.T_DEFAULT, "freeflow", post, gen, E.T0, E.STAGGER)
        bnd.append(E.theoretical_bound(R, tc, p))
        for f in FRACS:
            adv = np.zeros(K, bool); adv[:int(round(f * K))] = True
            for c in ["B2", "P5"]:
                m = E.metrics(simulate(c, R, Bs, tc, p, gen.rng, adv_mask=adv, adv_offset=E.ADV_OFFSET), st, p)
                acc[c][str(f)].append(float(np.mean(m["under_max"])))
    out = {c: {k: E.boot_ci(v) for k, v in acc[c].items()} for c in acc}
    out["bound"] = E.boot_ci(bnd); out["fracs"] = FRACS; out["seeds"] = seeds
    # proposed scheme, honest (0%) vs adversarial (30%, 50%): paired seed-level sign-flip permutation test
    out["tests"] = {f"P5_adv{f}_minus_honest": signflip(acc["P5"][str(f)], acc["P5"]["0.0"]) for f in (0.3, 0.5)}
    out["per_seed"] = acc
    E._save("E", out); return out


def signflip(a, b, nperm=100000, seed=0):
    """Two-sided paired sign-flip permutation test on d = a - b (exact if 2^n <= nperm)."""
    d = np.asarray(a, float) - np.asarray(b, float); n = len(d); obs = abs(d.mean())
    if 2 ** n <= nperm:
        signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
    else:
        signs = np.random.default_rng(seed).choice([-1, 1], size=(nperm, n))
    null = np.abs((signs * d).mean(axis=1))
    return {"mean_diff": float(d.mean()), "p": float(max((null >= obs - 1e-15).mean(), 1.0 / len(null))),
            "n_seeds": n, "test": "paired sign-flip permutation, two-sided"}


def expF(E, seeds, seed0=57):
    from recalibration import Params, simulate
    rm = np.random.default_rng(seed0)
    acc = {k: {str(P): [] for P in PERIODS} for k in ["rate_err", "latency", "under_max"]}
    for s in range(seeds):
        gen = E.RegimeGenerator(seed=int(rm.integers(1e9))); post = E.assign_post(K, True)
        R, Bs, st, tc = E.build_fleet(K, E.T_DEFAULT, "freeflow", post, gen, E.T0, E.STAGGER)
        for P in PERIODS:
            p = dataclasses.replace(Params(), P=P)
            m = E.metrics(simulate("P5", R, Bs, tc, p, gen.rng), st, p)
            for k in acc: acc[k][str(P)].append(float(np.mean(m[k])))
    out = {k: {P: E.boot_ci(v) for P, v in acc[k].items()} for k in acc}
    out["periods"] = PERIODS; out["seeds"] = seeds
    E._save("F", out); return out


def expG(E, seeds, seed0=67):
    from recalibration import Params, simulate
    p = Params(); rm = np.random.default_rng(seed0)
    regimes = {"heterogeneous": True, "homogeneous": False}
    acc = {r: {c: {"rate_err": [], "latency": []} for c in ["B1", "P5"]} for r in regimes}
    for s in range(seeds):
        seed_s = int(rm.integers(1e9))
        for r, het in regimes.items():
            gen = E.RegimeGenerator(seed=seed_s); post = E.assign_post(K, het)
            R, Bs, st, tc = E.build_fleet(K, E.T_DEFAULT, "freeflow", post, gen, E.T0, E.STAGGER)
            for c in ["B1", "P5"]:
                m = E.metrics(simulate(c, R, Bs, tc, p, gen.rng), st, p)
                for k in acc[r][c]: acc[r][c][k].append(float(np.mean(m[k])))
    out = {r: {c: {k: E.boot_ci(v) for k, v in acc[r][c].items()} for c in acc[r]} for r in acc}
    # paired seed-level differences (proposed - local-only), with bootstrap CI
    out["paired_diff"] = {r: {k: E.boot_ci(np.array(acc[r]["P5"][k]) - np.array(acc[r]["B1"][k]))
                              for k in ["rate_err", "latency"]} for r in acc}
    out["seeds"] = seeds
    E._save("G", out); return out


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--only", default="all", help="E, F, G or all")
    ap.add_argument("--real", action="store_true", help="use the real highD/NGSIM/Waymo risk pools (results_real/)")
    a = ap.parse_args()
    E = __import__("real_backend").patch() if a.real else __import__("experiments")
    fns = {"E": expE, "F": expF, "G": expG}
    for w in (["E", "F", "G"] if a.only == "all" else [a.only.upper()]):
        fns[w](E, a.seeds); print("Experiment", w, "done")
