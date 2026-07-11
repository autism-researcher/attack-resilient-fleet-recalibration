#!/usr/bin/env python3
"""
Read-only helper: fetch the newest Zenodo version DOI for this project's linked
record, so you can cite it in the manuscript's Data-availability line.

Usage:
    python zenodo_check_version.py

No Zenodo token needed (public record). Run it a minute or two AFTER you publish
the GitHub v1.0.2 release, so the Zenodo webhook has time to create the version.
"""
import json, sys, urllib.request

CONCEPT = "21288058"   # concept record id  (DOI 10.5281/zenodo.21288058)
URL = f"https://zenodo.org/api/records/{CONCEPT}/versions?sort=-mostrecent&size=10"

def main():
    try:
        req = urllib.request.Request(URL, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=25) as r:
            data = json.load(r)
    except Exception as e:
        print("Could not reach the Zenodo API:", e)
        print("Check manually (Versions panel):  https://doi.org/10.5281/zenodo.21288058")
        sys.exit(1)

    hits = data.get("hits", {}).get("hits", [])
    if not hits:
        print("No versions found yet.")
        print("If you just published the GitHub release, wait ~1 min and re-run.")
        sys.exit(0)

    print("Zenodo versions (newest first):")
    for h in hits:
        ver = h.get("metadata", {}).get("version", "?")
        print(f"  version {ver:<10} DOI {h.get('doi','?'):<28} {h.get('created','')[:10]}")
    newest = hits[0].get("doi", "?")
    print("\nNewest version DOI:")
    print(f"  {newest}")
    print("\nGive me this DOI and I'll place it in the manuscript's Data-availability line,")
    print("keeping the concept DOI 10.5281/zenodo.21288058 as the all-versions reference.")

if __name__ == "__main__":
    main()
