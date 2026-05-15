#!/usr/bin/env python3
"""Validate and package the cross_arch_tolerance experiment for RunPod upload.

Checks:
  1. All modified Python files compile without syntax errors.
  2. Required model files have standard interfaces (create, train, load, evaluate).
  3. Checkpoints exist (or print reminder to train first).
  4. Plot functions exist in plot_results.py.
  5. sweep_all.py can import without runtime errors (except torch GPU).

Usage:
    python experiments/cross_arch_tolerance/validate_for_runpod.py
"""

import ast
import importlib.util
import sys
from pathlib import Path

_EXP = Path(__file__).parent
_PROJ = _EXP.parent.parent


def _py_compile(path: Path) -> bool:
    try:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
        return True
    except SyntaxError as e:
        print(f"  [FAIL] Syntax error in {path.name}: {e}")
        return False


def _check_module_interface(path: Path, required_funcs: list[str]) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return False
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            found.add(node.name)
    missing = set(required_funcs) - found
    if missing:
        print(f"  [WARN] {path.name} missing functions: {missing}")
        return False
    return True


def main():
    ok = True
    print("=" * 60)
    print("RunPod Upload Validation")
    print("=" * 60)

    # 1. Syntax check all modified files
    print("\n[1] Syntax checking modified files...")
    files = [
        _EXP / "models" / "deq.py",
        _EXP / "models" / "ssm.py",
        _EXP / "models" / "flow.py",
        _EXP / "models" / "neural_ode.py",
        _EXP / "models" / "transformer.py",
        _EXP / "models" / "diffusion.py",
        _EXP / "models" / "ebm.py",
        _EXP / "sweep_all.py",
        _EXP / "plot_results.py",
        _EXP / "transient_stability.py",
        _PROJ / "neuro_analog" / "simulator" / "substrates.py",
        _PROJ / "neuro_analog" / "simulator" / "analog_acceleration.py",
        _PROJ / "neuro_analog" / "ir" / "energy_model.py",
        _PROJ / "neuro_analog" / "ir" / "graph.py",
        _PROJ / "neuro_analog" / "ir" / "types.py",
    ]
    for f in files:
        if f.exists():
            status = "OK" if _py_compile(f) else "FAIL"
            print(f"  {status}: {f.name}")
            if status == "FAIL":
                ok = False
        else:
            print(f"  MISSING: {f}")
            ok = False

    # 2. Model interfaces
    print("\n[2] Checking model interfaces...")
    required = ["create_model", "train_model", "load_model", "evaluate", "get_family_name"]
    for name in ["deq", "ssm", "flow", "neural_ode", "transformer", "diffusion", "ebm"]:
        path = _EXP / "models" / f"{name}.py"
        if path.exists():
            good = _check_module_interface(path, required)
            extra = []
            if name == "deq":
                extra = ["evaluate_convergence_stats", "evaluate_output_mse"]
            elif name in ["ssm", "flow"]:
                extra = ["dynamics_metrics", "evaluate_output_mse"]
            elif name in ["neural_ode", "diffusion", "transformer", "ebm"]:
                extra = ["evaluate_output_mse"]
            if extra:
                good &= _check_module_interface(path, extra)
            print(f"  {'OK' if good else 'WARN'}: {name}.py")
        else:
            print(f"  MISSING: {name}.py")

    # 3. Checkpoints
    print("\n[3] Checking checkpoints...")
    ckpt_dir = _EXP / "checkpoints"
    for name in ["deq", "ssm", "flow", "neural_ode", "transformer", "diffusion", "ebm"]:
        ckpt = ckpt_dir / f"{name}.pt"
        print(f"  {'OK' if ckpt.exists() else 'MISSING'}: {ckpt.name}")

    # 4. Plot functions
    print("\n[4] Checking plot functions...")
    plot_file = _EXP / "plot_results.py"
    tree = ast.parse(plot_file.read_text(encoding="utf-8"))
    funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    needed = [f"plot_figure{i}" for i in [1,2,3,4,5,6,7,10,11,12,13]]
    for nf in needed:
        present = nf in funcs
        print(f"  {'OK' if present else 'MISSING'}: {nf}()")
        if not present:
            ok = False

    # 5. Import sweep_all (best-effort without torch)
    print("\n[5] Import sweep_all structure...")
    spec = importlib.util.spec_from_file_location("sweep_all", _EXP / "sweep_all.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        print("  OK: sweep_all.py imports (torch may fail at runtime on CPU-only)")
    except Exception as e:
        print(f"  WARN: import issue (expected if torch unavailable): {type(e).__name__}: {e}")

    # 6. Summary
    print("\n" + "=" * 60)
    if ok:
        print("VALIDATION PASSED — ready for RunPod upload.")
        print("Next steps:")
        print("  1. Upload experiments/ directory to RunPod")
        print("  2. Run: python experiments/cross_arch_tolerance/sweep_all.py --analog-domain both --physical-substrate pcm")
        print("  3. Run: python experiments/cross_arch_tolerance/sweep_all.py --compute-acceleration")
        print("  4. Run: python experiments/cross_arch_tolerance/plot_results.py")
        print("  5. Download results/ and figures/ directories")
    else:
        print("VALIDATION FAILED — fix syntax errors before upload.")
    print("=" * 60)


if __name__ == "__main__":
    main()
