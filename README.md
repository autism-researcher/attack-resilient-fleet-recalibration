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
