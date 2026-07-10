# Running the P5 study on REAL NGSIM data

This reproduces the full P5 evaluation (Experiments A–G) on real vehicle
trajectories instead of the NGSIM-calibrated generator. You download the public
data once; the code does the rest.

## Step 1 — Download NGSIM (public, free)
Dataset: **"Next Generation Simulation (NGSIM) Vehicle Trajectories and
Supporting Data"** (U.S. DOT / FHWA).
- Portal: the U.S. DOT ITS DataHub at **data.transportation.gov** (search "NGSIM
  vehicle trajectories"); it is also listed on catalog.data.gov.
- Download a trajectory CSV for one location (e.g., **I-80** or **US-101**). A
  single location file is enough; the full combined file is large (~1 GB).

The CSV must contain (at least) these columns, which the loader uses:
`Vehicle_ID`, `Space_Headway` (feet), `v_Vel` (ft/s). Units are auto-detected
and converted to metres / metres-per-second.

## Step 2 — Place the file
Put the downloaded CSV in this folder (`P5_fleet_experiment/`), e.g.
`NGSIM_I80.csv`.

## Step 3 — Run
```
pip install numpy pandas matplotlib scipy
python run_ngsim.py --csv "NGSIM_I80.csv" --seeds 15
```
(You may pass several files or a glob: `--csv "NGSIM_*.csv"`.)

## Outputs
- `results_real/*.json` — Experiments A–G with 95% bootstrap CIs.
- `figures_real/*.png` — the same figures as the paper, on real data.
- Console — a **risk-score validation report**: per-regime empirical matching
  boundary B*, mean risk, and density at the boundary f(B*).

## How "real data" is used
- The 8-feature risk score (identical to the paper) is computed on every real
  follower stream; closing speed is approximated from the gap-closing rate.
- Operating **regimes** are defined directly from the data by binning each
  stream's median time-headway into terciles (low / mid / high density), so a
  shift (freeflow to congested) is a real low-density to high-density change.
- Each regime's **ground-truth boundary B\*** is the empirical (1-tau)-quantile
  of that regime's real risk scores — a real, data-defined boundary.

## Scope note
This validates the risk model and the recalibration method on real trajectory
distributions and real distribution shifts. It is not a connected-fleet field
trial (no real V2X timing or correlation); that is the next step, discussed
under "Threats to validity" in the manuscript. The central guarantee (Theorem 1)
is distribution- and aggregation-agnostic, so it transfers unchanged.
