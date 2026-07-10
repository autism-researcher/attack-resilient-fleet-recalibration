#!/usr/bin/env bash
# One-time upload of this folder to GitHub.
# Prereqs: git installed; run from inside the P5_fleet_experiment folder.
set -e
REPO_URL="https://github.com/autism-researcher/attack-resilient-fleet-recalibration.git"

git init
git add .
git commit -m "Initial public artifact: attack-resilient collaborative recalibration (TR-C submission)"
git branch -M main
git remote add origin "$REPO_URL"
git push -u origin main

# tag a citable release
git tag -a v1.0.0 -m "TR-C submission artifact v1.0.0"
git push origin v1.0.0
echo "Done. Repo: $REPO_URL  (tag v1.0.0)"
