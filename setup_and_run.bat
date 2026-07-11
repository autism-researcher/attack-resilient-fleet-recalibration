@echo off
REM ============================================================================
REM  One-click: install dependencies, then run the reviewer experiments.
REM  Double-click this file, or run it from a terminal in this folder.
REM ============================================================================
cd /d "%~dp0"

echo [1/3] Ensuring pip is available...
python -m ensurepip --upgrade

echo [2/3] Installing dependencies (numpy, pandas, matplotlib, pillow)...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo Install failed. If you have several Python installs, try the launcher:
  echo    py -3 -m pip install -r requirements.txt
  echo    py -3 run_reviewer_experiments.py --seeds 30
  pause
  exit /b 1
)

echo [3/3] Running the reviewer-strengthening experiments (seeds=30)...
python run_reviewer_experiments.py --seeds 30

echo.
echo Done. To also re-derive the NGSIM boundary from your raw CSV, run:
echo    python run_reviewer_experiments.py --seeds 30 --ngsim-csv "D:/New Paper3/paper3_pipeline/data/ngsim/ngsim_trajectories.csv"
pause
