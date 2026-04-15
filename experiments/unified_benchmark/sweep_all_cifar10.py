"""
Run analog mismatch sweeps on all trained CIFAR-10 models.

Generates degradation curves for each architecture family.
"""

import sys
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neuro_analog.simulator import analogize, mismatch_sweep
from models import (
    NeuralODENet, S4DNet, DEQNet, DiffusionClassifier,
    FlowClassifier, EBMClassifier, ViTClassifier
)


def load_model(arch_name, checkpoint_path, device):
    """Load trained model from checkpoint."""
    if arch_name == 'neural_ode':
        model = NeuralODENet(num_classes=10, hidden_dim=256)
    elif arch_name == 's4d':
        model = S4DNet(num_classes=10, d_model=256, n_layers=4)
    elif arch_name == 'deq':
        model = DEQNet(num_classes=10, hidden_dim=256, max_iter=30)
    elif arch_name == 'diffusion':
        model = DiffusionClassifier(num_classes=10, hidden_dim=256)
    elif arch_name == 'flow':
        model = FlowClassifier(num_classes=10, n_flows=8)
    elif arch_name == 'ebm':
        model = EBMClassifier(num_classes=10, hidden_dim=256)
    elif arch_name == 'transformer':
        model = ViTClassifier(num_classes=10, d_model=256, n_heads=8, n_layers=6)
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


def get_test_loader():
    """Load CIFAR-10 test set."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                           (0.2023, 0.1994, 0.2010))
    ])
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=4
    )
    
    return test_loader


def evaluate_accuracy(model):
    """Evaluation function for mismatch_sweep."""
    test_loader = get_test_loader()
    device = next(model.parameters()).device
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return 100.0 * correct / total


def run_sweep(arch_name, checkpoint_dir, output_dir,
              sigma_values, n_trials, n_adc_bits, device):
    """Run mismatch sweep for one architecture."""
    print(f"\n{'='*70}")
    print(f"Running sweep: {arch_name}")
    print(f"{'='*70}")

    checkpoint_path = checkpoint_dir / f"{arch_name}_cifar10_best.pt"

    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return None

    model = load_model(arch_name, checkpoint_path, device)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Sigma values: {sigma_values}")
    print(f"Trials per sigma: {n_trials}")
    print(f"ADC bits: {n_adc_bits}")

    # Get calibration data (sample batch from test set)
    test_loader = get_test_loader()
    calibration_batch = next(iter(test_loader))[0][:32].to(device)

    result = mismatch_sweep(
        model,
        eval_fn=evaluate_accuracy,
        sigma_values=sigma_values,
        n_trials=n_trials,
        n_adc_bits=n_adc_bits,
        calibration_data=calibration_batch,  # Critical: prevents ADC clipping
        analog_domain='conservative'
    )
    
    output_path = output_dir / f"{arch_name}_cifar10_sweep.json"
    result.save(str(output_path))
    
    threshold = result.degradation_threshold(0.10)
    print(f"\nResults:")
    print(f"  Baseline accuracy: {result.digital_baseline:.2f}%")
    print(f"  Threshold @ 10% loss: σ = {threshold:.3f}")
    print(f"  Saved to: {output_path}")
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-dir', type=str, 
                       default='checkpoints/cifar10',
                       help='Directory containing trained checkpoints')
    parser.add_argument('--output-dir', type=str,
                       default='results/cifar10',
                       help='Directory to save sweep results')
    parser.add_argument('--sigma-values', type=str,
                       default='0.0,0.03,0.05,0.07,0.10,0.12,0.15',
                       help='Comma-separated sigma values')
    parser.add_argument('--n-trials', type=int, default=50,
                       help='Number of Monte Carlo trials per sigma')
    parser.add_argument('--n-adc-bits', type=int, default=8,
                       help='ADC bit width')
    parser.add_argument('--arch', type=str, default=None,
                       help='Run specific architecture only (default: all)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    args = parser.parse_args()
    
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sigma_values = [float(s) for s in args.sigma_values.split(',')]
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {device}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Output directory: {output_dir}")
    
    architectures = [
        'neural_ode', 's4d', 'deq', 'diffusion',
        'flow', 'ebm', 'transformer'
    ]
    
    if args.arch:
        if args.arch not in architectures:
            print(f"ERROR: Unknown architecture '{args.arch}'")
            print(f"Available: {', '.join(architectures)}")
            return
        architectures = [args.arch]
    
    results = {}
    for arch in architectures:
        result = run_sweep(
            arch, checkpoint_dir, output_dir,
            sigma_values, args.n_trials, args.n_adc_bits, device
        )
        if result:
            results[arch] = result
    
    print(f"\n{'='*70}")
    print("Summary of all sweeps:")
    print(f"{'='*70}")
    print(f"{'Architecture':<15} {'Baseline':>10} {'σ @ 10%':>10}")
    print(f"{'-'*70}")
    
    for arch, result in results.items():
        baseline = result.digital_baseline
        threshold = result.degradation_threshold(0.10)
        print(f"{arch:<15} {baseline:>9.2f}% {threshold:>9.3f}")
    
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
