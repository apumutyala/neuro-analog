"""
Post-processing script to compute amenability scores from sweep results.

This script loads sweep result JSON files, extracts sigma_10pct values,
combines them with IR analysis, and computes amenability scores with
design heuristics classification.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Dict, Optional
import sys

# Add project root to path for imports
_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from neuro_analog.ir.types import AnalogAmenabilityProfile, ArchitectureFamily
from neuro_analog.ir.energy_model import HardwareProfile, compute_amenability_score
from neuro_analog.analysis.design_heuristics import classify_failure_mode, print_heuristic_report, get_design_recommendations


def load_sweep_results(results_dir: Path) -> Dict[str, dict]:
    """Load all sweep result JSON files from a directory.
    
    Args:
        results_dir: Directory containing sweep result JSON files.
        
    Returns:
        Dictionary mapping architecture names to sweep result data.
    """
    results = {}
    
    for json_file in results_dir.glob("*_mismatch.json"):
        # Extract architecture name from filename
        arch_name = json_file.stem.replace("_mismatch", "")
        
        with open(json_file, "r") as f:
            data = json.load(f)
        
        results[arch_name] = data
    
    return results


def extract_sigma_10pct(sweep_result: dict) -> float:
    """Extract sigma at 10% degradation from sweep result.
    
    Args:
        sweep_result: Sweep result dictionary from JSON.
        
    Returns:
        Sigma threshold at 10% degradation (0.0 if not found).
    """
    return sweep_result.get("degradation_threshold_10pct", 0.0)


def compute_amenability_for_architecture(
    arch_name: str,
    sweep_result: dict,
    profile_path: Optional[Path] = None,
    hardware_profile_config: Optional[Path] = None,
) -> dict:
    """Compute amenability score for a single architecture.
    
    Args:
        arch_name: Architecture name.
        sweep_result: Sweep result dictionary.
        profile_path: Path to IR profile JSON (optional).
        hardware_profile_config: Path to hardware profile config (optional).
        
    Returns:
        Dictionary with amenability metrics.
    """
    # Load or create hardware profile
    if hardware_profile_config is not None and hardware_profile_config.exists():
        hardware_profile = HardwareProfile.from_config(hardware_profile_config)
    else:
        hardware_profile = HardwareProfile()  # Use defaults
    
    # Extract sigma_10pct from sweep results
    sigma_10pct = extract_sigma_10pct(sweep_result)
    
    # Load IR profile if available
    ir_profile = None
    if profile_path is not None and profile_path.exists():
        with open(profile_path, "r") as f:
            ir_data = json.load(f)
        # Would need full deserialization - for now use sweep data only
    
    # Create minimal profile with available data
    # In practice, this would be loaded from IR analysis
    profile = AnalogAmenabilityProfile(
        architecture=ArchitectureFamily.TRANSFORMER,  # Placeholder
        model_name=arch_name,
        model_params=0,  # Would come from IR
        analog_flop_fraction=0.5,  # Placeholder - would come from IR
        digital_flop_fraction=0.5,
        hybrid_flop_fraction=0.0,
        da_boundary_count=10,  # Placeholder - would come from IR
        layer_count=10,  # Placeholder - would come from IR
        sigma_10pct=sigma_10pct,
        min_weight_precision_bits=8,  # Placeholder
    )
    
    # Compute amenability score
    profile.amenability_score = compute_amenability_score(profile)
    
    # Classify failure mode
    failure_mode = classify_failure_mode(profile)
    
    # Get design recommendations
    recommendations = get_design_recommendations(profile)
    
    return {
        "architecture": arch_name,
        "sigma_10pct": sigma_10pct,
        "amenability_score": profile.amenability_score,
        "failure_mode": failure_mode,
        "recommendations": recommendations,
        "sweep_data": {
            "degradation_threshold_10pct": sigma_10pct,
            "normalized_mean": sweep_result.get("normalized_mean", []),
        },
    }


def compute_amenability_for_all(
    results_dir: Path,
    profile_dir: Optional[Path] = None,
    hardware_profile_config: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, dict]:
    """Compute amenability scores for all architectures in sweep results.
    
    Args:
        results_dir: Directory containing sweep result JSON files.
        profile_dir: Directory containing IR profile JSON files (optional).
        hardware_profile_config: Path to hardware profile config (optional).
        output_dir: Directory to save amenability results (optional).
        
    Returns:
        Dictionary mapping architecture names to amenability data.
    """
    results = load_sweep_results(results_dir)
    amenability_data = {}
    
    for arch_name, sweep_result in results.items():
        # Look for corresponding IR profile
        profile_path = None
        if profile_dir is not None:
            profile_path = profile_dir / f"{arch_name}_*.json"
            # Find matching profile if exists
            matching = list(profile_dir.glob(f"{arch_name}_*.json"))
            if matching:
                profile_path = matching[0]
        
        # Compute amenability
        amenability_data[arch_name] = compute_amenability_for_architecture(
            arch_name, sweep_result, profile_path, hardware_profile_config
        )
    
    # Save results if output directory specified
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save combined results
        combined_path = output_dir / "amenability_summary.json"
        with open(combined_path, "w") as f:
            json.dump(amenability_data, f, indent=2)
        print(f"Amenability summary saved to: {combined_path}")
        
        # Save per-architecture results
        for arch_name, data in amenability_data.items():
            arch_path = output_dir / f"{arch_name}_amenability.json"
            with open(arch_path, "w") as f:
                json.dump(data, f, indent=2)
    
    return amenability_data


def print_amenability_table(amenability_data: Dict[str, dict]) -> None:
    """Print a formatted table of amenability scores.
    
    Args:
        amenability_data: Dictionary of amenability data by architecture.
    """
    print("\n" + "=" * 100)
    print("Analog Amenability Score Summary")
    print("=" * 100)
    print(f"{'Architecture':<20} {'Sigma@10%':<12} {'Amenability':<12} {'Failure Mode':<25}")
    print("-" * 100)
    
    for arch_name, data in sorted(amenability_data.items()):
        sigma = data["sigma_10pct"]
        score = data["amenability_score"]
        mode = data["failure_mode"]
        
        sigma_str = f"{sigma:.1%}" if sigma > 0 else "N/A"
        score_str = f"{score:.3f}"
        
        print(f"{arch_name:<20} {sigma_str:<12} {score_str:<12} {mode:<25}")
    
    print("=" * 100 + "\n")


def main():
    """Main entry point for amenability computation."""
    parser = argparse.ArgumentParser(
        description="Compute analog amenability scores from sweep results."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing sweep result JSON files."
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default=None,
        help="Directory containing IR profile JSON files (optional)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to hardware profile YAML config (optional)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save amenability results (optional)."
    )
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        help="Specific architecture to analyze (optional)."
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    profile_dir = Path(args.profile_dir) if args.profile_dir else None
    hardware_profile_config = Path(args.config) if args.config else None
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    print(f"Loading sweep results from: {results_dir}")
    
    if args.arch:
        # Compute for single architecture
        results = load_sweep_results(results_dir)
        if args.arch not in results:
            print(f"Error: Architecture '{args.arch}' not found in results")
            return
        
        sweep_result = results[args.arch]
        amenability_data = {
            args.arch: compute_amenability_for_architecture(
                args.arch, sweep_result, None, hardware_profile_config
            )
        }
    else:
        # Compute for all architectures
        amenability_data = compute_amenability_for_all(
            results_dir, profile_dir, hardware_profile_config, output_dir
        )
    
    # Print summary table
    print_amenability_table(amenability_data)
    
    # Print detailed reports for each architecture
    for arch_name, data in amenability_data.items():
        print(f"\nDetailed report for {arch_name}:")
        print(f"  Sigma @ 10%: {data['sigma_10pct']:.1%}" if data['sigma_10pct'] > 0 else "  Sigma @ 10%: N/A")
        print(f"  Amenability score: {data['amenability_score']:.3f}")
        print(f"  Failure mode: {data['failure_mode']}")
        print(f"  Recommendation: {data['recommendations']['analog_deployment']}")


if __name__ == "__main__":
    main()
