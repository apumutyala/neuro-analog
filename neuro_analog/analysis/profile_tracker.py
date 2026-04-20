"""
Centralized tracking and analysis of AnalogAmenabilityProfile results.

Provides:
- Loading profiles from JSON logs
- Aggregating statistics across runs
- Historical comparison
- Per-architecture reporting
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime

from neuro_analog.ir.types import AnalogAmenabilityProfile, ArchitectureFamily


@dataclass
class ProfileSummary:
    """Summary statistics for profiles of one architecture."""
    architecture: str
    count: int
    avg_analog_flop_fraction: float
    avg_digital_flop_fraction: float
    avg_hybrid_flop_fraction: float
    avg_da_boundaries: float
    avg_overall_score: float
    min_overall_score: float
    max_overall_score: float
    model_names: List[str] = field(default_factory=list)


class ProfileTracker:
    """Track and analyze AnalogAmenabilityProfile results across runs."""

    def __init__(self, log_dir: str | Path = "outputs/profiles"):
        self.log_dir = Path(log_dir)
        self.profiles: Dict[str, AnalogAmenabilityProfile] = {}

    def load_profiles(self) -> int:
        """Load all JSON profiles from log directory.

        Returns:
            Number of profiles loaded.
        """
        if not self.log_dir.exists():
            return 0

        count = 0
        for json_file in self.log_dir.glob("*.json"):
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                # Store raw data for now - would need full deserialization for full profile
                self.profiles[json_file.stem] = data
                count += 1
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")
        
        return count

    def get_architecture_summary(self, architecture: str) -> Optional[ProfileSummary]:
        """Get summary statistics for a specific architecture.

        Args:
            architecture: Architecture family name (e.g., "neural_ode", "transformer")

        Returns:
            ProfileSummary with statistics, or None if no profiles found.
        """
        arch_profiles = [
            p for p in self.profiles.values()
            if p.get("architecture") == architecture
        ]

        if not arch_profiles:
            return None

        count = len(arch_profiles)
        analog_fracs = [p["analog_flop_fraction"] for p in arch_profiles]
        digital_fracs = [p["digital_flop_fraction"] for p in arch_profiles]
        hybrid_fracs = [p["hybrid_flop_fraction"] for p in arch_profiles]
        boundaries = [p["da_boundary_count"] for p in arch_profiles]
        scores = [p["overall_score"] for p in arch_profiles]
        model_names = [p["model_name"] for p in arch_profiles]

        return ProfileSummary(
            architecture=architecture,
            count=count,
            avg_analog_flop_fraction=sum(analog_fracs) / count,
            avg_digital_flop_fraction=sum(digital_fracs) / count,
            avg_hybrid_flop_fraction=sum(hybrid_fracs) / count,
            avg_da_boundaries=sum(boundaries) / count,
            avg_overall_score=sum(scores) / count,
            min_overall_score=min(scores),
            max_overall_score=max(scores),
            model_names=model_names,
        )

    def get_all_summaries(self) -> Dict[str, ProfileSummary]:
        """Get summary statistics for all architectures with profiles.

        Returns:
            Dict mapping architecture name to ProfileSummary.
        """
        summaries = {}
        for arch in ArchitectureFamily:
            summary = self.get_architecture_summary(arch.value)
            if summary is not None:
                summaries[arch.value] = summary
        return summaries

    def print_summary_table(self):
        """Print a formatted summary table of all architectures."""
        summaries = self.get_all_summaries()

        if not summaries:
            print("No profiles found in log directory.")
            return

        print("\n" + "=" * 100)
        print("FLOP Analysis Summary by Architecture")
        print("=" * 100)
        print(f"{'Architecture':<15} {'Count':<6} {'Analog FLOP%':<13} {'Digital FLOP%':<14} "
              f"{'D/A Bnd':<8} {'Avg Score':<10} {'Score Range':<15}")
        print("-" * 100)

        for arch_name, summary in sorted(summaries.items()):
            score_range = f"{summary.min_overall_score:.3f} - {summary.max_overall_score:.3f}"
            print(f"{arch_name:<15} {summary.count:<6} "
                  f"{summary.avg_analog_flop_fraction:>6.1%}{'':<6} "
                  f"{summary.avg_digital_flop_fraction:>6.1%}{'':<6} "
                  f"{summary.avg_da_boundaries:>6.1f}{'':<2} "
                  f"{summary.avg_overall_score:>6.3f}{'':<4} "
                  f"{score_range:<15}")

        print("=" * 100)
        print(f"Total profiles: {len(self.profiles)}")
        print(f"Log directory: {self.log_dir}")
        print("=" * 100 + "\n")

    def export_summary_csv(self, output_path: str | Path):
        """Export summary statistics to CSV.

        Args:
            output_path: Path to save CSV file.
        """
        import csv
        summaries = self.get_all_summaries()

        if not summaries:
            print("No profiles to export.")
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "architecture", "count", "avg_analog_flop_fraction",
                "avg_digital_flop_fraction", "avg_hybrid_flop_fraction",
                "avg_da_boundaries", "avg_overall_score",
                "min_overall_score", "max_overall_score"
            ])

            for arch_name, summary in sorted(summaries.items()):
                writer.writerow([
                    arch_name, summary.count,
                    summary.avg_analog_flop_fraction,
                    summary.avg_digital_flop_fraction,
                    summary.avg_hybrid_flop_fraction,
                    summary.avg_da_boundaries,
                    summary.avg_overall_score,
                    summary.min_overall_score,
                    summary.max_overall_score,
                ])

        print(f"Summary exported to: {output_path}")

    def compare_with_baseline(self, architecture: str, baseline_model_name: str | None = None) -> Dict:
        """Compare current profiles against a baseline.

        Args:
            architecture: Architecture family to compare.
            baseline_model_name: If specified, use this model as baseline.
                If None, uses the oldest profile as baseline.

        Returns:
            Dict with comparison metrics.
        """
        arch_profiles = [
            p for p in self.profiles.values()
            if p.get("architecture") == architecture
        ]

        if len(arch_profiles) < 2:
            print(f"Need at least 2 profiles for comparison, found {len(arch_profiles)}")
            return {}

        # Sort by timestamp (embedded in filename)
        sorted_profiles = sorted(arch_profiles, key=lambda x: x.get("model_name", ""))

        if baseline_model_name:
            baseline = next((p for p in sorted_profiles if p.get("model_name") == baseline_model_name), None)
            if not baseline:
                print(f"Baseline model '{baseline_model_name}' not found, using oldest profile")
                baseline = sorted_profiles[0]
        else:
            baseline = sorted_profiles[0]

        latest = sorted_profiles[-1]

        comparison = {
            "baseline_model": baseline.get("model_name"),
            "latest_model": latest.get("model_name"),
            "architecture": architecture,
            "analog_flop_fraction_delta": latest.get("analog_flop_fraction", 0) - baseline.get("analog_flop_fraction", 0),
            "da_boundary_count_delta": latest.get("da_boundary_count", 0) - baseline.get("da_boundary_count", 0),
            "overall_score_delta": latest.get("overall_score", 0) - baseline.get("overall_score", 0),
        }

        return comparison

    def print_comparison(self, architecture: str, baseline_model_name: str | None = None):
        """Print a comparison between baseline and latest profile.

        Args:
            architecture: Architecture family to compare.
            baseline_model_name: Optional specific baseline model name.
        """
        comparison = self.compare_with_baseline(architecture, baseline_model_name)

        if not comparison:
            return

        print("\n" + "=" * 80)
        print(f"Historical Comparison: {architecture}")
        print("=" * 80)
        print(f"Baseline:  {comparison['baseline_model']}")
        print(f"Latest:    {comparison['latest_model']}")
        print("-" * 80)

        analog_delta = comparison["analog_flop_fraction_delta"]
        analog_sign = "+" if analog_delta >= 0 else ""
        print(f"Analog FLOP fraction:  {analog_sign}{analog_delta:+.1%}")

        boundary_delta = comparison["da_boundary_count_delta"]
        boundary_sign = "+" if boundary_delta >= 0 else ""
        print(f"D/A boundaries:          {boundary_sign}{boundary_delta:+d}")

        score_delta = comparison["overall_score_delta"]
        score_sign = "+" if score_delta >= 0 else ""
        print(f"Overall score:          {score_sign}{score_delta:+.3f}")

        print("=" * 80 + "\n")


def analyze_profiles(log_dir: str | Path = "outputs/profiles", export_csv: bool = True, compare: bool = False):
    """Main entry point for profile analysis.

    Args:
        log_dir: Directory containing profile JSON files.
        export_csv: If True, export summary to CSV.
        compare: If True, show historical comparison for architectures with multiple profiles.
    """
    tracker = ProfileTracker(log_dir)
    count = tracker.load_profiles()
    print(f"Loaded {count} profiles from {log_dir}")

    if count > 0:
        tracker.print_summary_table()
        if export_csv:
            csv_path = Path(log_dir) / "summary.csv"
            tracker.export_summary_csv(csv_path)
        if compare:
            # Show comparison for architectures with multiple profiles
            summaries = tracker.get_all_summaries()
            for arch_name, summary in summaries.items():
                if summary.count >= 2:
                    tracker.print_comparison(arch_name)
    else:
        print("No profiles found to analyze.")
