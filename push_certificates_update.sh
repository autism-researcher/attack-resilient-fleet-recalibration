#!/usr/bin/env bash
# ============================================================================
#  Release v1.0.3: corrected finite-sample certificates (TR-C fourth round).
#  Run from the repo root:  D:/PhD_2027/P5_fleet_experiment
#  Repo: https://github.com/autism-researcher/attack-resilient-fleet-recalibration
#  Zenodo-linked, so creating a GitHub RELEASE auto-mints a new version DOI
#  under the concept DOI 10.5281/zenodo.21288058.
# ============================================================================
set -e
cd "$(dirname "$0")"

NOTES="Corrected finite-sample certificates (no fabrication; all numbers recomputed from committed data by certificates.py).
- Binding lower-tail variance v=(p-d)(1-p+d) for p=1-tau>=1/2 (q(1-q)>p(1-p) on the lower tail); concentration constant delta 0.055 -> 0.067.
- Adapted (martingale) case: two fixed bracketing quantile levels only, no score grid; predictable variance bounded by Jensen concavity.
- Run-level maxima compared to the simultaneous certificate (alpha->alpha/(KT)), not the per-step envelope.
- Certificates stated as conservative worst-case guarantees, not tight operational margins.
- Corrected per-step/simultaneous envelopes: controlled 0.100/0.422, highD 0.070/0.313, NGSIM 0.065/0.119, Waymo 0.110/0.303, closed-loop 0.041/0.268.
- New: certificates.py (authoritative), exp_o_cert.py (rare-event tail); theoretical_bound() updated; two prior unverified numbers corrected (controlled naive-clamp 0.080->0.055; Exp E narrative)."

echo "[1/5] Staging files..."
git add certificates.py exp_o_cert.py experiments.py agnostic.py closed_loop.py penetration.py \
        make_figures_bw.py make_evidence_fig.py README.md
git add results_real/certificates.json results_real/exp_o_certificates.json 2>/dev/null || true

echo "[2/5] Committing..."
git commit -m "TR-C fourth round: corrected finite-sample certificates (binding-tail variance, two-bracket martingale, run-level vs per-step); certificates.py authoritative; no fabrication"

echo "[3/5] Tagging v1.0.3..."
git tag -a v1.0.3 -m "Corrected finite-sample certificates (TR-C fourth round)"

echo "[4/5] Pushing commits + tag..."
git push origin main
git push origin v1.0.3

echo "[5/5] Creating GitHub Release (triggers the Zenodo version)..."
if command -v gh >/dev/null 2>&1; then
  gh release create v1.0.3 --title "v1.0.3 - Corrected finite-sample certificates" --notes "$NOTES"
else
  echo
  echo "GitHub CLI (gh) not installed. Create the release manually here:"
  echo "  https://github.com/autism-researcher/attack-resilient-fleet-recalibration/releases/new?tag=v1.0.3"
  echo "Paste the notes above. Publishing the release triggers Zenodo to mint the new version DOI."
fi

echo
echo "Next: a NEW Zenodo version draft appears under 10.5281/zenodo.21288058."
echo "  Run:  python zenodo_check_version.py    (prints the newest version DOI)"
echo "  Then Publish on Zenodo if not auto-published, and put that DOI in the manuscript's Data-availability line."
