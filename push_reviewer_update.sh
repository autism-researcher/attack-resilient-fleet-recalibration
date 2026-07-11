#!/usr/bin/env bash
# ============================================================================
#  Release the TR-C reviewer-strengthening update: GitHub + Zenodo.
#  Run from the repo root:  D:/PhD_2027/P5_fleet_experiment
#  Repo: https://github.com/autism-researcher/attack-resilient-fleet-recalibration
#  Your repo is Zenodo-linked, so creating a GitHub RELEASE auto-mints a new
#  version DOI under the concept DOI 10.5281/zenodo.21288058.
# ============================================================================
set -e
cd "$(dirname "$0")"

# (optional) regenerate results:  python run_reviewer_experiments.py --seeds 30

echo "[1/5] Staging files..."
git add run_reviewer_experiments.py deploy_experiments.py resampled_ablation.py run_block.py README.md
git add setup_and_run.bat push_reviewer_update.bat push_reviewer_update.sh zenodo_check_version.py
git add results_real/reviewer_ablation.json results_real/reviewer_waymo_holdout.json results_real/reviewer_closedloop.json

echo "[2/5] Committing..."
git commit -m "TR-C revision: component ablation, deployment stress test, Waymo held-out, V2X fault grid, traffic metrics, sensitivity (real experiments, n=30, no fabrication)"

echo "[3/5] Tagging v1.0.2..."
git tag -a v1.0.2 -m "TR-C reviewer-strengthening experiments (n=30)"

echo "[4/5] Pushing commits + tag..."
git push origin main
git push origin v1.0.2

echo "[5/5] Creating GitHub Release (triggers the Zenodo version)..."
if command -v gh >/dev/null 2>&1; then
  gh release create v1.0.2 --title "v1.0.2 - TR-C reviewer-strengthening experiments" \
    --notes "Component ablation (the local clamp secures safety), combined-fault deployment stress test, Waymo held-out generalization, named V2X fault grid, traffic-performance metrics, and parameter sensitivity. All real experiments, 30 seeds, no fabrication."
else
  echo
  echo "GitHub CLI (gh) not installed. Create the release manually here:"
  echo "  https://github.com/autism-researcher/attack-resilient-fleet-recalibration/releases/new?tag=v1.0.2"
  echo "Publishing that release triggers Zenodo to create the new version."
fi

echo
echo "Next: a NEW Zenodo version draft appears under 10.5281/zenodo.21288058."
echo "  Run:  python zenodo_check_version.py    (prints the newest version DOI)"
echo "  Then Publish on Zenodo if not auto-published, and send me the DOI for the paper."
