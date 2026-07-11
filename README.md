# Attack-Resilient Collaborative Recalibration of Safety Boundaries for CAVs

Reproducible code for the paper *"Attack-Resilient Collaborative Recalibration of
Safety Boundaries for Connected Automated Vehicles"* (M. B. Hossain, M. A. S. Kamal,
K. Yamada). It implements the information-flow-isolation scheme: fleet information is
admitted only into a fast, rate-tracking boundary, while the runtime safety gate is
clamped by a strictly local window-quantile boundary, so corrupted collaboration
cannot weaken the per-vehicle under-protection bound.

## Install
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Quick demo (no licensed datasets required)
The controlled study runs entirely on the built-in NGSIM-calibrated *generator* (synthetic streams with a known ground-truth boundary), so readers without licensed data can reproduce the core safety result end-to-end:
```bash
pip install -r requirements.txt
python experiments.py --seeds 5 --only C      # Experiment C: naive fleet-clamp's under-protection is larger; local clamp holds
python make_figures_bw.py                      # rebuilds the figures from committed JSON
```
No highD/NGSIM/Waymo download is needed for this path; the real-corpus studies additionally use the derived `real/*.npy` pools already included.

## Reproduce the results
```bash
# 1) Controlled + real experiments -> results/*.json and results_real/*.json
python experiments.py --seeds 15 --only all      # Experiments A-D
python extra_experiments.py                       # clamp ablations, Byzantine (H), LDP (I)
python closed_loop.py --seeds 12                  # closed-loop microsimulation
python penetration.py ; python sens_tests.py ; python lowtau.py

# 2) Regenerate all paper figures (black-and-white safe) into figures_bw/
python make_figures_bw.py
```
All headline numbers are written to committed JSON files, so figures/tables can be
rebuilt without re-running the (stochastic) experiments.

## Finite-sample certificates (`certificates.py`)
`certificates.py` is the authoritative computation of the per-vehicle under-protection
certificates. It recomputes, from the committed pools/generators, the corrected
concentration constant and the per-step and run-level (simultaneous) envelopes reported
in the paper:
```bash
python certificates.py        # -> results_real/certificates.json
```
The concentration analysis was corrected in the TR-C revision: (i) the two-sided
Bernstein/Freedman deviation uses the **binding lower-tail variance**
`v(delta) = (p-delta)(1-p+delta)` for `p = 1-tau >= 1/2` (the lower tail is nearer 1/2,
so `q(1-q) > p(1-p)`), giving `delta = 0.067` (was 0.055); and (ii) the adapted
(martingale) case evaluates the empirical CDF at the two fixed bracketing quantile
levels only — **no score grid** — with predictable variance bounded by Jensen concavity.
The quantile modulus `omega_F` is computed exactly from each realized law. Run-level
maxima are compared to the simultaneous certificate (`alpha -> alpha/(KT)`), not the
per-step envelope. These certificates are conservative worst-case guarantees, not tight
operational margins; the observed under-protection is several-fold smaller.
Corrected per-step / simultaneous envelopes: controlled 0.100 / 0.422; highD 0.070 / 0.313;
NGSIM 0.065 / 0.119; Waymo 0.110 / 0.303; closed-loop 0.041 / 0.268.
`exp_o_cert.py` computes the rare-event (low-tau) tail certificates of Experiment O.

## Reviewer-strengthening experiments (TR-C revision)
One command reproduces every new table — component ablation (which part secures safety),
combined-fault deployment stress test, Waymo held-out generalization, named V2X fault
grid, traffic-performance metrics, and parameter sensitivity:
```bash
python run_reviewer_experiments.py --seeds 30
# optional: re-derive the NGSIM boundary end-to-end from your raw CSV
python run_reviewer_experiments.py --seeds 30 \
    --ngsim-csv "D:/New Paper3/paper3_pipeline/data/ngsim/ngsim_trajectories.csv"
```
New scripts:
- `deploy_experiments.py` — extended closed-loop harness: ablation variants (naive
  fleet-clamp, robust-no-clamp, robust/unweighted/shift-aware + clamp), adversarial
  upload injection, named V2X faults, and full traffic metrics. Reproduces the main
  closed loop **exactly** for AEB/B1.
- `resampled_ablation.py` — real-pool component ablation (highD→NGSIM) + Waymo held-out
  generalization (parameters fixed on highD/NGSIM, applied unchanged to Waymo).
- `run_block.py` — resumable chunked driver (only needed on time-limited runners).

Outputs: `results_real/reviewer_ablation.json`, `reviewer_waymo_holdout.json`,
`reviewer_closedloop.json`.

Key real findings: the **local clamp** secures safety (removing it — ACI — raises
under-protection to ~3× the clamp-based scheme and outside the guarantee's scope; every
clamp variant holds at 0.037–0.038 regardless of aggregator); under-protection is statistically **flat across 0–70% adversarial
uploads**; P5 holds the bound under delay to 1 s, loss to 40%, stale, Byzantine, and
mixed faults.

### Real-corpus provenance
- **highD** v1.0 — 60 recordings, 6 German highway sites, 25 Hz, 16.7 h, 110,516 vehicles (89,139 cars).
- **NGSIM** — US-101 (2,847 vehicles) + I-80 (3,001), freeway, 10 Hz.
- **Waymo** Open Dataset v1.2 — 45 segments.
Each corpus is reduced to a pool of 5,000 per-trajectory maximum-risk (R_max) values
(`real/{highd,ngsim,waymo}.npy`); B* = 0.313 / 0.375 / 0.849 (empirical (1−τ)-quantile, τ=0.10).

## Paper <-> code map
| Paper item | Script | Output |
|---|---|---|
| Risk score R(x) (8 features) | `risk_score.py` | — |
| Local clamp + fleet aggregator + B0/B1/B2 baselines | `recalibration.py` | — |
| Experiments A–D (accuracy, adaptation, safety, fleet size) | `experiments.py` | `results/*.json`, `figures/exp{A,B,C,D}*.png` |
| Clamp ablation, Byzantine aggregators (H), LDP (I) | `extra_experiments.py` | `results_real/{baselines_extra,expH,expI_ldp}.json` |
| Closed-loop microsimulation + V2X impairment | `closed_loop.py` | `results_real/closed_loop.json` |
| Corridor hazard (M), scenarios, penetration (N) | `corridor.py`, `scenarios.py`, `penetration.py` | `results_real/*.json` |
| Hyperparameter sensitivity (L), low-tau (O) | `sens_tests.py`, `lowtau.py` | `results_real/{sensitivity,lowtau}.json` |
| Black-and-white-safe figures for the paper | `make_figures_bw.py` | `figures_bw/*.png` |

## Data
The naturalistic corpora are public but under their own licenses and are **not**
redistributed here:
- highD: https://www.highd-dataset.com
- NGSIM (US-101/I-80): https://data.transportation.gov
- Waymo Open Dataset: https://waymo.com/open

`real/{highd,ngsim,waymo}.npy` contain only **derived** per-trajectory maximum-risk
values (5000 scalars each) used to reconstruct the empirical risk distributions; they
carry no raw trajectories. `data.py` reconstructs follower streams from a raw NGSIM CSV
if you supply one (`--ngsim path/to/NGSIM.csv`).

## Citation
If you use this code, please cite the paper (see `CITATION` / the article once
published).

## License
MIT (see `LICENSE`). Dataset licenses remain with their original providers.
