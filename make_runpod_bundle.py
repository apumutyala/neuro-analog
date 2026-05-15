#!/usr/bin/env python3
"""Create a lean zip bundle for RunPod upload.

Includes only code, checkpoints, and config needed to train, sweep,
and plot on RunPod.  Excludes old results, generated figures, caches,
logs, git history, and other bloat.
"""
import os
import zipfile
from pathlib import Path

# Anchor at repo root
ROOT = Path(__file__).resolve().parent
ZIP_PATH = ROOT / "neuro-analog-runpod-bundle.zip"

# Base inclusions from repo root (patterns relative to ROOT)
INCLUDE_GLOBS = [
    "pyproject.toml",
    "README.md",
    "configs/**/*.yaml",
    "neuro_analog/**/*.py",
    "neuro_analog/simulator/**/*.py",
    "neuro_analog/ir/**/*.py",
    "experiments/cross_arch_tolerance/*.py",
    "experiments/cross_arch_tolerance/*.sh",
    "experiments/cross_arch_tolerance/train_hwa.py",
    "experiments/cross_arch_tolerance/models/*.py",
    "experiments/cross_arch_tolerance/checkpoints/*.pt",
]

# Explicitly create these empty output directories so scripts can write immediately
EMPTY_DIRS = [
    "experiments/cross_arch_tolerance/results",
    "experiments/cross_arch_tolerance/figures",
]

# Exclusion patterns (applied to relative path strings)
EXCLUDE_PATTERNS = [
    "__pycache__",
    ".pyc",
    ".pyo",
    ".git",
    ".pytest_cache",
    ".claude",
    "old_figs_results_checkpoints",
    "*.log",
    "*.txt",          # out.txt, out_fixed.txt, pilot_run.log — exclude to keep lean
    "*.png",
    "*.pdf",
    "*.jpg",
    "*.jpeg",
    "*.json",         # old results JSONs
    "*.ipynb",
    "*.egg-info",
    "dist/",
    "outputs/",
    "ark_exports/",
    "examples/",
    "tests/",
    "notes/",
    "notebooks/",
]


def should_exclude(rel_path: str) -> bool:
    """Return True if rel_path matches any exclusion."""
    for pat in EXCLUDE_PATTERNS:
        if pat in rel_path:
            return True
    return False


def gather_files():
    files = []
    for glob in INCLUDE_GLOBS:
        for p in ROOT.glob(glob):
            rel = p.relative_to(ROOT).as_posix()
            if should_exclude(rel):
                continue
            if p.is_file():
                files.append(rel)
    # dedupe + sort for deterministic archive
    return sorted(set(files))


def main():
    files = gather_files()
    print(f"Bundling {len(files)} files into {ZIP_PATH.name} …")
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in files:
            zf.write(ROOT / rel, rel)
            print(f"  + {rel}")
        # add empty placeholder dirs
        for d in EMPTY_DIRS:
            zf.writestr(f"{d}/", "")
            print(f"  + {d}/ (empty dir)")
    size_mb = ZIP_PATH.stat().st_size / 1e6
    print(f"\nDone: {ZIP_PATH} ({size_mb:.2f} MB)")
    print("Upload this to RunPod, unzip, then run:")
    print("  pip install -e .")
    print("  cd experiments/cross_arch_tolerance")
    print("  bash runpod_pilot.sh")


if __name__ == "__main__":
    main()
