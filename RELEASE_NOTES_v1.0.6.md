# v1.0.6 — Matched Local Self-Blending control

Supersedes v1.0.5. No existing script or result file is changed; every v1.0.5 JSON is unchanged.

**Why.** The fleet blend pulls the fast threshold toward a quantile estimate every P steps. A vehicle can do the same with its own quantile, with no communication. Comparing the fleet scheme only with local-only recalibration therefore mixes two effects. The Matched Local Self-Blending control (code label `B1S`) uses the same rule, weight and period as the fleet blend, with the vehicle's own quantile in place of the fleet estimate, so Local-Blend minus Fleet-Gated isolates the value of collaboration.

**New scripts** (frozen code imported unchanged)
- `rerun_local_blend.py` → `results_real/B_localblend_{15,30}seeds.json` (latency vs fleet size, real pools)
- `rerun_local_blend_rate.py` → `results_real/DG_localblend.json` (rate error vs fleet size; heterogeneous vs homogeneous)
- `rerun_local_blend_all.py controlled | realBD | realA | closed | hetero` → `results_real/localblend_{controlled,real_BD_perK_seeds,realA,closed,hetero}.json`
- `make_collab_fig.py` → `figures/collab_value_hetero.png`
- `power_single_ego.py` (single-vehicle spread of the collaboration value; used to plan the hardware study)

**Main findings**
- Controlled generator: the collaboration value is not significant at any fleet size; the gain over local-only is self-blending.
- Real highD/NGSIM/Waymo replay: small, significant collaboration value for K >= 20 (latency 5–10 steps, rate error about 1–1.5%).
- Closed loop: Fleet-Gated and Local-Blend have the same collision rate; both are above local-only, so the gap is a blending effect.
- In the real replay, heterogeneous fleets of K <= 3 are effectively homogeneous (the regimes congested, rain and dense all map to NGSIM).
