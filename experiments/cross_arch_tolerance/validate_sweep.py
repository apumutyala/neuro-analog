#!/usr/bin/env python3
"""Validate sweep results for physical correctness.

Checks that every JSON result file satisfies physically-grounded invariants:
- energy_saving_vs_digital ∈ [0, 1] (0% to 100% saving)
- speedup_vs_digital > 0
- per_trial MSE ≥ 0 (if present)
- No NaN or Inf in numeric fields

Usage:
    python experiments/cross_arch_tolerance/validate_sweep.py
    python experiments/cross_arch_tolerance/validate_sweep.py --results-dir ./results
"""

import json
import math
import argparse
import sys
from pathlib import Path
from typing import Any


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _load_jsons(results_dir: Path) -> list[tuple[Path, dict]]:
    files = []
    for p in results_dir.glob("*.json"):
        # Skip acceleration / convergence / profile files — they have different schemas
        if any(x in p.name for x in ("_acceleration", "_convergence", "_profile")):
            continue
        try:
            with open(p, "r") as f:
                data = json.load(f)
            files.append((p, data))
        except Exception as e:
            files.append((p, {"__parse_error__": str(e)}))
    return files


def _check_numerics(data: dict, path: str, issues: list[str]) -> None:
    """Recursively check for NaN/Inf in numeric fields."""
    for k, v in data.items():
        cur = f"{path}.{k}" if path else k
        if isinstance(v, dict):
            _check_numerics(v, cur, issues)
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, (dict, list)):
                    _check_numerics(item if isinstance(item, dict) else {}, f"{cur}[{i}]", issues)
                elif _is_number(item) and (math.isnan(item) or math.isinf(item)):
                    issues.append(f"{cur}[{i}] = {item} (NaN/Inf)")
        elif _is_number(v) and (math.isnan(v) or math.isinf(v)):
            issues.append(f"{cur} = {v} (NaN/Inf)")


def _validate_file(p: Path, data: dict) -> list[str]:
    issues: list[str] = []

    if "__parse_error__" in data:
        issues.append(f"JSON parse error: {data['__parse_error__']}")
        return issues

    _check_numerics(data, "", issues)

    # Energy saving
    es = data.get("energy_saving_vs_digital")
    if es is not None:
        if not (0.0 <= es <= 1.0):
            issues.append(f"energy_saving_vs_digital = {es} (must be in [0, 1])")

    # Speedup
    sp = data.get("speedup_vs_digital")
    if sp is not None:
        if not (sp > 0):
            issues.append(f"speedup_vs_digital = {sp} (must be > 0)")

    # per_trial MSE sanity
    pt = data.get("per_trial")
    metric_name = data.get("metric_name", "")
    if pt is not None and isinstance(pt, list):
        # If metric_name contains "mse", every value should be >= 0
        if "mse" in metric_name.lower():
            for i, row in enumerate(pt):
                for j, val in enumerate(row):
                    if val < 0:
                        issues.append(f"per_trial[{i}][{j}] = {val} (negative MSE)")
        # Check any per_trial value is not NaN/Inf
        for i, row in enumerate(pt):
            for j, val in enumerate(row):
                if math.isnan(val) or math.isinf(val):
                    issues.append(f"per_trial[{i}][{j}] = {val} (NaN/Inf)")

    return issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any issue found")
    args = parser.parse_args()

    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        # Default relative to this script
        results_dir = Path(__file__).parent / "results"

    files = _load_jsons(results_dir)
    total_issues = 0

    print(f"Validating {len(files)} result files in {results_dir}\n")

    for p, data in sorted(files, key=lambda x: x[0].name):
        issues = _validate_file(p, data)
        if issues:
            total_issues += len(issues)
            print(f"FAIL  {p.name}")
            for issue in issues:
                print(f"      - {issue}")
        else:
            # Brief summary for passing files
            es = data.get("energy_saving_vs_digital")
            sp = data.get("speedup_vs_digital")
            extra = ""
            if es is not None and sp is not None:
                extra = f" (energy={es*100:.1f}%, speedup={sp:.1f}x)"
            print(f"PASS  {p.name}{extra}")

    print(f"\n{'='*60}")
    print(f"Total files: {len(files)}  |  Issues: {total_issues}")
    if total_issues == 0:
        print("All checks passed.")
    else:
        print(f"{total_issues} issue(s) found.")
        if args.strict:
            sys.exit(1)


if __name__ == "__main__":
    main()
