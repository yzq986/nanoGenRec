#!/usr/bin/env python
"""Regenerate EXP-004 reports from existing JSON results."""
import glob
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from gr_demo.eval.hyperparam import generate_report

dirs = sorted(glob.glob("experiments/hyperparam/*exp004*"))
if not dirs:
    print("No exp004 result dirs found!")
    sys.exit(1)

all_results = []
for d in dirs:
    json_path = f"{d}/results.json"
    try:
        with open(json_path) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} results from {json_path}")
        all_results.extend(results)
    except FileNotFoundError:
        print(f"  {json_path} not found, skipping")

if all_results:
    out = "experiments/hyperparam/exp004-combined-report.md"
    generate_report(all_results, out)
    print(f"\nDone! {len(all_results)} total results -> {out}")
