# Reproducibility and audit map (release v1.0.6)

Every number in the manuscript and supplement comes from a JSON file written by a script in this release. The table maps each item to its command, seeds and output.

## Environment used for the reported results
- Python 3.11.15
- Pinned packages: `requirements-lock.txt` (numpy 2.4.4, pandas 3.0.2, matplotlib 3.10.9, pillow 12.2.0)
- CPU only; no GPU and no network access needed.

Install with `pip install -r requirements-lock.txt`. The looser `requirements.txt` also works, but numbers can differ in the last digit under other numpy versions.

## Data
`real/{highd,ngsim,waymo}.npy` hold 5000 derived per-trajectory maximum-risk values per corpus. No raw trajectories are included. The controlled generator (`data.RegimeGenerator`) is hand-parameterized within published US-101/I-80 ranges; it is not fitted.

## Item → command → output

| Manuscript item | Command | Seeds (base) | Output |
|---|---|---|---|
| Table 2, Table S4, Figs. S1–S2 (controlled A–D) | `python experiments.py --seeds 15 --only all` | 15 (rng 7/17/27/37) | `results/{A,B,C,D}.json` |
| Table 4 and real A–D | `python experiments.py --seeds 15 --only all --real` | 15 | `results_real/{A,B,C,D}.json` |
| Fig. 2 (E), Fig. S3 (F), Fig. S4 (G) | `python experiments_efg.py --seeds 15 --real` | 15 (rng 47/57/67) | `results_real/{E,F,G}.json` |
| Table 3 (component ablation), Waymo holdout | `python resampled_ablation.py ablation 30` / `holdout 30` | 30 (777/888) | `results_real/reviewer_ablation.json`, `reviewer_waymo_holdout.json` |
| Table 5, closed-loop figure (eight-feature score) | `python closed_loop.py --seeds 12` | 12 (100–111; V2X 200+, 500+) | `results_real/closed_loop.json` |
| Table S21 four-feature ablation | `CL_SCORE=4 python closed_loop.py --seeds 12` (Windows: `set CL_SCORE=4`) | same | `results_real/closed_loop_4feat.json` |
| Tables S15–S17 (V2X grid, traffic metrics, combined fault) | `python run_reviewer_experiments.py --seeds 30` | 30 | `results_real/reviewer_closedloop.json` |
| Table S6 (I), Table S7 (H), clamp-free baselines | `python extra_experiments.py` | 12 (41/53/61) | `results_real/{expI_ldp,expH,baselines_extra}.json` |
| Table S8 (L), paired tests | `python sens_tests.py` | 10 (7) | `results_real/{sensitivity,htests}.json` |
| Table S9 (O) | `python lowtau.py ; python exp_o_cert.py` | 10 (71) | `results_real/{lowtau,exp_o_certificates}.json` |
| Table S10 (K) | `python agnostic.py` | 8 | `results_real/agnostic.json` |
| Table S11, Table S12 (M), Table S13 (N) | `python scenarios.py ; python corridor.py ; python penetration.py` | 8 / 8 / 10 | `results_real/{scenarios,corridor,penetration}.json` |
| Cross-domain fleet | `python experiment_K.py` | 15 (7) | `results_real/experiment_K.json` |
| Table S18 (envelopes, K = 20, T = 1700, ζ = 0.05) | `python certificates.py` | fixed pools / seeds 7–36 | `results_real/certificates.json` |
| Table S19 and the online (DKW) column of Table S18 | `python forensic_checks.py all` (incl. `split70`: B* from held-out 30% of trajectories) | see script | `results_real/forensic_*.json` |
| Table 3 bottom rows (clamp reads a robust fleet aggregate) | `python forensic_checks.py robustclamp` | 30 (777) | `results_real/forensic_robustclamp.json` |
| Table S20 (A) envelopes vs. window length N_s, retrospective and rigorous online (DKW with known support [0,1]); online column of Table S18 | `python forensic_checks.py online` | 500 windows (rng 0) | `results_real/forensic_online_envelope.json` |
| Table S20 (B) realized drift between window law and target (two-sided, one-sided, burn-in) | `python forensic_checks.py drift` | — | `results_real/forensic_drift.json` |
| Simultaneous rigorous online envelope vs N_s (Sec. 8, Table S20 caption) | `python forensic_checks.py online_sim` | 300 windows (rng 0) | `results_real/forensic_online_simultaneous.json` |
| Paired closed-loop tests (Sec. 7.5, Table S19 B; Table S21) | `python forensic_checks.py closed` / `closed4` | 12 (100–111) | `results_real/forensic_closed_paired{,_4feat}.json` |
| Paired combined-fault tests (Sec. 7.5, Table S17) | `python forensic_checks.py deploy_paired` | 30 (100–129) | `results_real/forensic_deploy_paired.json` |
| Envelopes for ρ = 0–0.05 (one-for-one shift) | `python forensic_checks.py rho` | — | `results_real/forensic_rho.json` |
| Timing (Secs. 4.5, 7.6) | `python timing.py` | — | `results_real/timing.json` (hardware-dependent) |
| Fleet benefit vs post-shift heterogeneity and fleet size K = 5–100 (paired local-only vs proposed; `hetero_fleetsize.png`) | `python forensic_checks.py hetero` | 30 (rng 91) | `results_real/forensic_hetero.json` |
| Four envelope terms vs observed under-protection, N_s = 100–5000, and the α budget (`nsweep_envelopes.png`) | `python forensic_checks.py nsweep` | 30 (rng 7); 300 windows (rng 0) | `results_real/forensic_nsweep.json` |
| Envelope of mixed pre/post-shift (transient) windows and burn-in k* per regime pair | `python forensic_checks.py mixture` | — | `results_real/forensic_mixture_modulus.json` |
| Table S20 (C): observed under-protection in the transient interval (first N_s post-shift steps) vs the certified interval | `python forensic_checks.py transient_split` | 30 (rng 7) | `results_real/forensic_transient_split.json` |
| Table S23 (inferential roles; Holm-adjusted confirmatory family) | `python forensic_checks.py holm` (after `deploy_paired` and `closed`) | — | `results_real/forensic_inference_roles.json` |
| Matched Local Self-Blending: latency vs K (real) | `python rerun_local_blend.py --seeds 30` | 30 (rng 17) | `results_real/B_localblend_30seeds.json` |
| Matched Local Self-Blending: rate error vs K; heterogeneous vs homogeneous (real) | `python rerun_local_blend_rate.py --seeds 30` | 30 (rng 37/67) | `results_real/DG_localblend.json` |
| Matched Local Self-Blending: controlled generator (latency, rate error) | `python rerun_local_blend_all.py controlled` | 30 (rng 17/37) | `results_real/localblend_controlled.json` |
| Matched Local Self-Blending: real, seed stream restarted per K | `python rerun_local_blend_all.py realBD` | 30 | `results_real/localblend_real_BD_perK_seeds.json` |
| Matched Local Self-Blending: real K = 20 (rate error, missed, under-protection, latency) | `python rerun_local_blend_all.py realA` | 30 (rng 7) | `results_real/localblend_realA.json` |
| Matched Local Self-Blending: closed loop | `python rerun_local_blend_all.py closed` | 12 (100–111) | `results_real/localblend_closed.json` |
| Matched Local Self-Blending: heterogeneity x fleet size; collaboration-value figure | `python rerun_local_blend_all.py hetero ; python make_collab_fig.py` | 30 (rng 91) | `results_real/localblend_hetero.json`, `figures/collab_value_hetero.png` |
| All figures | `python make_figures_ress.py` | — | `figures_ress/*.png` |

## Closed-loop risk score
All closed-loop scripts (`closed_loop.py`, `deploy_experiments.py`, `scenarios.py`, `corridor.py`) use the eight-feature score of Eq. (1) (`risk_score.py`), with acceleration and jerk from finite differences of speed. Setting the environment variable `CL_SCORE=4` switches to the reduced four-feature ablation; outputs then go to `*_4feat.json`.

## Definitions used in the code
- **Under-protection:** (B_eff − B*)₊ over post-shift steps. Tables 2 and 4, Fig. 2 and the controlled sweeps report the mean over vehicles of each vehicle's maximum. Table 3, the holdout, H, I, the baselines and the closed loop report the maximum over all vehicles.
- **Adaptation latency:** first step after a vehicle's shift where the 100-step moving intervention rate stays within ±0.02 of τ for 50 steps (the remaining horizon if it never does).
- **Rate error:** mean |rate − τ| from 350 steps after the shift to the end.
- **Certified envelopes:** m + ω + ρ with m = 0.02, ρ = 0.01 (assumed), and δ = 0.067 (Bernstein, binding-tail variance). The numerical envelopes are *retrospective*, with ω taken from the realized law. The online DKW bound (`forensic_checks.py online`) replaces any upper level above 1 by the known score support 1 (the sample maximum is not an upper confidence bound); its per-step budget is split α_q = α_DKW = 0.025 (95% overall), and the simultaneous version (`online_sim`) uses α_q = α_DKW = ζ/(2KT) with ζ = 0.05, K = 20, T = 1700. At N_s ≤ 600 the online bound is conservative. Only the one-sided drift (B̄ − B*)₊ enters the one-sided bound; its realized value is reported by `forensic_checks.py drift`.
- **Certified interval:** the finite-sample certificate needs a window that is i.i.d. given the source law, so it applies from N_s steps after each vehicle's shift. Windows in the first N_s post-shift steps mix pre- and post-shift scores; that interval is reported empirically (`transient_split`), and `mixture` gives the larger quantile modulus of those mixed windows.

## Legacy files
`results_real/legacy/` holds earlier committed results kept for comparison; the manuscript uses none of them.
