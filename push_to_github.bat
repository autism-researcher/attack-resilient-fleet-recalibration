@echo off
REM One-time upload to GitHub. Run from inside D:\PhD_2027\P5_fleet_experiment
setlocal
set REPO_URL=https://github.com/autism-researcher/attack-resilient-fleet-recalibration.git

git init
git add .
git commit -m "Initial public artifact (TR-C submission)"
git branch -M main
git remote add origin %REPO_URL%
git push -u origin main
git tag -a v1.0.0 -m "TR-C submission artifact v1.0.0"
git push origin v1.0.0

echo.
echo Done. Repo: %REPO_URL%  (tag v1.0.0)
pause
