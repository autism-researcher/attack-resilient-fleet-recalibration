"""
Recalibration engine: single-vehicle clamp (Paper P4) and the fleet-wide
collaborative scheme (Paper P5), plus baselines B0/B1/B2.

  B0 : frozen boundary (offline-calibrated on pre-shift regime; no update).
  B1 : local-only online recalibration -> fast update + LOCAL clamp.
  B2 : naive fleet averaging -> clamp uses an UNWEIGHTED fleet estimate
       (fleet enters the clamp). Demonstrates the failure the design avoids.
  P5 : proposed -> fleet estimate enters only the FAST update via a shift-gated
       blend; clamp stays strictly LOCAL.

Adversarial/stale contributors: a subset of vehicles may UPLOAD a corrupted
quantile estimate (U + adv_offset). This stresses Theorem 1: B2 (clamp reads
fleet) loses its safety bound; P5 (clamp local) does not.
"""
import numpy as np
from dataclasses import dataclass

@dataclass
class Params:
    tau: float = 0.10
    gamma: float = 0.05
    N_s: int = 300
    m: float = 0.02
    P: int = 20
    h: float = 0.03
    lam: float = 0.5
    eps: float = 0.02

def _local_quantile(window, tau):
    return np.quantile(window, 1.0 - tau, axis=1)

def fleet_estimate(U_report, n_eff, params, weighted=True, U_query=None):
    K = len(U_report)
    if not weighted:
        return np.full(K, np.average(U_report, weights=n_eff))
    Uq = (U_query if U_query is not None else U_report)[:, None]
    Ui = U_report[None, :]
    W = n_eff[None, :] * np.exp(-((Ui - Uq) ** 2) / (2 * params.h ** 2))
    W = W / W.sum(axis=1, keepdims=True)
    return (W * U_report[None, :]).sum(axis=1)

def simulate(condition, R, Bstar, t_cal, params, rng, adv_mask=None, adv_offset=0.0):
    K, T = R.shape
    p = params
    pre = R[:, :t_cal]
    B0 = np.quantile(pre, 1.0 - p.tau, axis=1)
    B = B0.copy()
    window = pre[:, -p.N_s:].copy() if pre.shape[1] >= p.N_s else np.repeat(pre[:, -1:], p.N_s, axis=1).copy()
    n_eff = np.full(K, float(p.N_s))
    if adv_mask is None: adv_mask = np.zeros(K, bool)
    beff_log = np.zeros((K, T)); inter_log = np.zeros((K, T), dtype=bool)
    Q = 5; ptr = 0
    U_local = _local_quantile(window, p.tau)
    B2_clamp = U_local.copy()
    for t in range(T):
        Rt = R[:, t]
        if t % Q == 0:
            U_local = _local_quantile(window, p.tau)
        if condition == "B0":
            Beff = B0
        elif condition == "B1":
            Beff = np.minimum(B, U_local + p.m)
        elif condition == "B2":
            if t % p.P == 0:
                U_report = U_local.copy(); U_report[adv_mask] = np.clip(U_local[adv_mask] + adv_offset, 0, 1)
                B2_clamp = fleet_estimate(U_report, n_eff, p, weighted=False)
            Beff = np.minimum(B, B2_clamp + p.m)
        elif condition == "P5":
            Beff = np.minimum(B, U_local + p.m)          # clamp STRICTLY local
        else:
            raise ValueError(condition)
        intervened = Rt > Beff
        inter_log[:, t] = intervened; beff_log[:, t] = Beff
        if condition in ("B1", "B2", "P5"):
            B = np.clip(B + p.gamma * (intervened.astype(float) - p.tau), 0.0, 1.0)
        if condition == "P5" and t > 0 and t % p.P == 0:
            U_report = U_local.copy(); U_report[adv_mask] = np.clip(U_local[adv_mask] + adv_offset, 0, 1)
            U_fleet = fleet_estimate(U_report, n_eff, p, weighted=True, U_query=U_local)
            gate = np.exp(-((U_fleet - U_local) ** 2) / (2 * p.h ** 2))
            lam = p.lam * gate
            B = np.clip((1 - lam) * B + lam * U_fleet, 0.0, 1.0)
        window[:, ptr] = Rt; ptr = (ptr + 1) % p.N_s
    return dict(intervened=inter_log, beff=beff_log, Bstar=Bstar, R=R, params=p)
