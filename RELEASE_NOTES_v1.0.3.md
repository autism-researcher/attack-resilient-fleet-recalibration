# v1.0.3 — Corrected finite-sample certificates (TR-C fourth round)

All numbers recomputed from committed pools/generators by `certificates.py`; nothing hand-set.

**Concentration analysis corrected**
- Binding **lower-tail** variance `v(delta) = (p-delta)(1-p+delta)` for `p = 1-tau >= 1/2`. For the 0.9 quantile the lower tail `q = p-delta` moves toward 1/2, so `q(1-q) > p(1-p)` — the previous `p(1-p)` understated it. Concentration constant `delta`: **0.055 -> 0.067** (`v = 0.139`).
- Adapted (martingale) case now evaluates the empirical CDF at only the **two fixed bracketing quantile levels** `Fbar^{-1}(p ± delta)` — **no score grid** (grid width does not control CDF mass). Predictable variance bounded by Jensen concavity, `sum F_s(1-F_s) <= Ns·Fbar(1-Fbar) <= Ns·v`.

**Reporting corrected**
- Run-level maxima are compared to the **simultaneous certificate** (`alpha -> alpha/(KT)`), not the per-step envelope.
- Certificates are stated as conservative worst-case guarantees, not tight operational margins; observed under-protection is several-fold smaller.
- "Local differential privacy" -> "Laplace-noise robustness" (no formal LDP mechanism is claimed).

**Corrected envelopes (per-step / simultaneous)**

| Setting | Per-step | Simultaneous |
|---|---|---|
| Controlled | 0.100 | 0.422 |
| highD | 0.070 | 0.313 |
| NGSIM | 0.065 | 0.119 |
| Waymo | 0.110 | 0.303 |
| Closed-loop | 0.041 | 0.268 |

**Code**
- New: `certificates.py` (authoritative certificate computation), `exp_o_cert.py` (rare-event/low-tau tail certificates).
- `experiments.theoretical_bound()` updated to the corrected `m + omega_F(delta) + rho` form; `agnostic.py`, `penetration.py`, and figure scripts updated to match.
- Two prior unverified numbers corrected against committed data: controlled naive-clamp under-protection `0.080 -> 0.055`; supplement Experiment E narrative (B2 is flat ~0.053 across 0–50% contamination, not rising to 0.103).
