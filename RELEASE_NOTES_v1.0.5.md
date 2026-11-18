# v1.0.5 — RESS submission: certified-interval scope, heterogeneity and window-length studies

Supersedes v1.0.4. No existing result changes; every v1.0.4 JSON is unchanged.

**Scope correction (certificate)**
- The finite-sample certificate assumes a local window that is i.i.d. given the source law. After a shift this holds only once the window contains post-shift scores alone, i.e. from N_s steps after the vehicle's shift. The manuscript now states the certified interval that way and reports the first N_s post-shift steps empirically.
- `forensic_checks.py mixture` → `results_real/forensic_mixture_modulus.json`: retrospective envelope of the mixed pre/post-shift window law for every studied shift (per-step up to 0.40 for highD→NGSIM, 0.11 for highD→Waymo, 0.08–0.14 for the controlled regimes).
- `forensic_checks.py transient_split` → `results_real/forensic_transient_split.json`: observed under-protection of the proposed scheme in the transient vs the certified interval (mean per-vehicle maximum 0.018–0.019 vs 0.028–0.036).

**New studies**
- `forensic_checks.py hetero` → `results_real/forensic_hetero.json`, `figures_ress/hetero_fleetsize.png`: latency and rate error vs post-shift heterogeneity and fleet size (K = 5–100), paired local-only vs proposed, 30 seeds.
- `forensic_checks.py nsweep` → `results_real/forensic_nsweep.json`, `figures_ress/nsweep_envelopes.png`: the four envelope terms and the α budget vs observed under-protection for N_s = 100–5000.

- `forensic_checks.py holm` → `results_real/forensic_inference_roles.json`: Holm–Bonferroni adjustment of the confirmatory family (combined-fault deployment, proposed − naive fleet-clamp on collisions, rate error and under-protection) and the unadjusted secondary paired tests.

**Housekeeping**
- Removed the stale `results_real/forensic_closed8.json` (superseded by `forensic_closed_paired.json`).
