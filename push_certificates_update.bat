@echo off
REM ==========================================================================
REM  Release v1.0.3: corrected finite-sample certificates (TR-C fourth round).
REM  Run from the repo root:  D:\PhD_2027\P5_fleet_experiment
REM  Zenodo-linked: a GitHub Release auto-mints a new version DOI under the
REM  concept DOI 10.5281/zenodo.21288058.
REM ==========================================================================
cd /d "%~dp0"

echo [1/5] Staging files...
git add certificates.py exp_o_cert.py experiments.py agnostic.py closed_loop.py penetration.py make_figures_bw.py make_evidence_fig.py README.md
git add results_real\certificates.json results_real\exp_o_certificates.json

echo [2/5] Committing...
git commit -m "TR-C fourth round: corrected finite-sample certificates (binding-tail variance, two-bracket martingale, run-level vs per-step); certificates.py authoritative; no fabrication"

echo [3/5] Tagging v1.0.3...
git tag -a v1.0.3 -m "Corrected finite-sample certificates (TR-C fourth round)"

echo [4/5] Pushing commits + tag...
git push origin main
git push origin v1.0.3

echo [5/5] Create the GitHub Release (this triggers the Zenodo version DOI):
where gh >nul 2>nul
if %ERRORLEVEL%==0 (
  gh release create v1.0.3 --title "v1.0.3 - Corrected finite-sample certificates" --notes-file RELEASE_NOTES_v1.0.3.md
) else (
  echo   GitHub CLI ^(gh^) not installed. Create the release manually here:
  echo   https://github.com/autism-researcher/attack-resilient-fleet-recalibration/releases/new?tag=v1.0.3
  echo   Use RELEASE_NOTES_v1.0.3.md for the notes body.
)

echo.
echo Next: run  python zenodo_check_version.py  to get the new version DOI, then
echo publish on Zenodo if needed and put that DOI in the manuscript Data-availability line.
