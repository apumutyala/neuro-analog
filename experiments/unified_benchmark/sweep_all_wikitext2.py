"""Analog mismatch sweep for WikiText-2 language models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import json
import numpy as np
import logging
from tqdm import tqdm
import math

# Import neuro_analog simulator
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from neuro_analog.simulator import analogize, mismatch_sweep

def load_model_and_data(arch, checkpoint_dir, device):
    """Load trained model and test data."""
    checkpoint_path = Path(checkpoint_dir) / f"{arch}_lm_best.pt"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vocab_size = checkpoint['vocab_size']
    config = checkpoint['config']
    
    # Load model architecture
    if arch == "transformer":
        from models.transformer_lm import TransformerLM
        model = TransformerLM(
            vocab_size=vocab_size,
            hidden_dim=config['hidden_dim'],
            n_layers=config['n_layers'],
            n_heads=config['n_heads'],
            dropout=config['dropout']
        )
    elif arch == "s4d":
        from models.s4d_lm import S4DLM
        model = S4DLM(
            vocab_size=vocab_size,
            hidden_dim=config['hidden_dim'],
            n_layers=config['n_layers'],
            dropout=config['dropout']
        )
    elif arch == "neural_ode":
        from models.neural_ode_lm import NeuralODELM
        model = NeuralODELM(
            vocab_size=vocab_size,
            hidden_dim=config['hidden_dim'],
            n_layers=config['n_layers'],
            dropout=config['dropout']
        )
    elif arch == "deq":
        from models.deq_lm import DEQLM
        model = DEQLM(
            vocab_size=vocab_size,
            hidden_dim=config['hidden_dim'],
            n_heads=config['n_heads'],
            dropout=config['dropout']
        )
    elif arch == "flow":
        from models.flow_lm import FlowLM
        model = FlowLM(
            vocab_size=vocab_size,
            hidden_dim=config['hidden_dim'],
            n_flows=8,
            dropout=config['dropout']
        )
    elif arch == "ebm":
        from models.ebm_lm import EBMLM
        model = EBMLM(
            vocab_size=vocab_size,
            hidden_dim=config['hidden_dim'],
            dropout=config['dropout']
        )
    elif arch == "diffusion":
        from models.diffusion_lm import DiffusionLM
        model = DiffusionLM(
            vocab_size=vocab_size,
            hidden_dim=config['hidden_dim'],
            n_steps=20,
            dropout=config['dropout']
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load test data
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast
    from train_wikitext2 import WikiTextDataset
    
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=False, padding=False,
                        return_attention_mask=False)
    
    tokenized = dataset.map(tokenize_function, batched=True, 
                           remove_columns=["text"])
    test_dataset = WikiTextDataset(tokenized["test"], config['seq_len'])
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                            num_workers=2)
    
    return model, test_loader, checkpoint['val_perplexity']

def compute_perplexity(model, dataloader, device):
    """Compute perplexity on dataset."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            input_ids = batch[:, :-1]
            target_ids = batch[:, 1:]
            
            logits = model(input_ids)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1)
            )
            
            total_loss += loss.item()
            total_tokens += target_ids.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 20))
    return perplexity

def sweep_single_model(arch, checkpoint_dir, output_dir, sigma_values,
                      n_trials, device):
    """Run mismatch sweep for a single model."""
    logging.info(f"Starting sweep for {arch}")

    # Load model and data
    model, test_loader, baseline_ppl = load_model_and_data(
        arch, checkpoint_dir, device
    )

    logging.info(f"Baseline perplexity: {baseline_ppl:.2f}")

    # Get calibration data (sample batch from test set)
    calibration_batch = next(iter(test_loader))[0][:32].to(device)

    # Sweep configuration
    sweep_config = {
        'sigma_values': sigma_values,
        'n_trials': n_trials,
        'adc_bits': 8,
        'adc_mode': 'conservative'
    }
    
    results = {
        'arch': arch,
        'digital_baseline': float(baseline_ppl),
        'sigma_values': sigma_values,
        'perplexity_mean': [],
        'perplexity_std': [],
        'normalized_mean': [],
        'normalized_std': []
    }
    
    # Run sweep for each sigma
    for sigma in tqdm(sigma_values, desc=f"Sweeping {arch}"):
        trial_perplexities = []

        for trial in range(n_trials):
            # Analogize model with correct kwarg names (sigma_mismatch, n_adc_bits)
            analog_model = analogize(
                model,
                sigma_mismatch=sigma,
                n_adc_bits=sweep_config['adc_bits'],
            )

            # Calibrate V_ref from calibration data (critical: prevents ADC clipping)
            from neuro_analog.simulator import calibrate_analog_model
            calibrate_analog_model(analog_model, calibration_batch)
            
            # Apply ADC profile and disable thermal (pure mismatch sweep)
            from neuro_analog.simulator import configure_analog_profile, set_all_noise
            configure_analog_profile(analog_model, sweep_config['adc_mode'])
            set_all_noise(analog_model, mismatch=True, thermal=False, quantization=True)
            
            # Evaluate
            ppl = compute_perplexity(analog_model, test_loader, device)
            trial_perplexities.append(ppl)
        
        mean_ppl = np.mean(trial_perplexities)
        std_ppl = np.std(trial_perplexities)
        
        results['perplexity_mean'].append(float(mean_ppl))
        results['perplexity_std'].append(float(std_ppl))
        results['normalized_mean'].append(float(baseline_ppl / mean_ppl))
        results['normalized_std'].append(float(std_ppl / baseline_ppl))
        
        logging.info(f"σ={sigma:.3f}: PPL={mean_ppl:.2f}±{std_ppl:.2f} "
                    f"(normalized={baseline_ppl/mean_ppl:.3f})")
    
    # Compute degradation threshold (10% increase in perplexity = 10% decrease in normalized)
    normalized = np.array(results['normalized_mean'])
    threshold_idx = np.where(normalized < 0.90)[0]
    
    if len(threshold_idx) > 0:
        # Interpolate threshold
        idx = threshold_idx[0]
        if idx > 0:
            x1, x2 = sigma_values[idx-1], sigma_values[idx]
            y1, y2 = normalized[idx-1], normalized[idx]
            threshold = x1 + (0.90 - y1) * (x2 - x1) / (y2 - y1)
        else:
            threshold = sigma_values[0]
    else:
        threshold = sigma_values[-1]
    
    results['degradation_threshold_10pct'] = float(threshold)
    
    logging.info(f"Threshold @ 10% degradation: σ={threshold:.3f}")
    
    # Save results
    output_path = Path(output_dir) / f"{arch}_lm_sweep.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Results saved to {output_path}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Sweep WikiText-2 models")
    parser.add_argument("--arch", type=str, default=None,
                       choices=["neural_ode", "s4d", "deq", "diffusion",
                               "flow", "ebm", "transformer"],
                       help="Architecture to sweep (if None, sweep all)")
    parser.add_argument("--checkpoint-dir", type=str,
                       default="checkpoints/wikitext2")
    parser.add_argument("--output-dir", type=str,
                       default="results/wikitext2")
    parser.add_argument("--sigma-values", type=str,
                       default="0.0,0.03,0.05,0.07,0.10,0.12,0.15",
                       help="Comma-separated sigma values")
    parser.add_argument("--n-trials", type=int, default=50,
                       help="Monte Carlo trials per sigma")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    # Parse sigma values
    sigma_values = [float(s) for s in args.sigma_values.split(',')]
    
    # Setup directories and logging
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(Path(args.output_dir) / "sweep.log"),
            logging.StreamHandler()
        ]
    )
    
    logging.info("="*60)
    logging.info("WikiText-2 Analog Mismatch Sweep")
    logging.info("="*60)
    logging.info(f"Sigma values: {sigma_values}")
    logging.info(f"Trials per sigma: {args.n_trials}")
    logging.info(f"Device: {args.device}")
    
    # Determine which architectures to sweep
    if args.arch:
        architectures = [args.arch]
    else:
        architectures = ["neural_ode", "s4d", "deq", "diffusion",
                        "flow", "ebm", "transformer"]
    
    # Run sweeps
    all_results = {}
    for arch in architectures:
        try:
            results = sweep_single_model(
                arch, args.checkpoint_dir, args.output_dir,
                sigma_values, args.n_trials, args.device
            )
            all_results[arch] = results
        except Exception as e:
            logging.error(f"Failed to sweep {arch}: {e}")
            continue
    
    # Summary
    logging.info("\n" + "="*60)
    logging.info("SWEEP SUMMARY")
    logging.info("="*60)
    
    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1]['degradation_threshold_10pct'],
        reverse=True
    )
    
    for arch, res in sorted_results:
        logging.info(
            f"{arch:15s} Baseline: {res['digital_baseline']:6.2f}  "
            f"Threshold: σ={res['degradation_threshold_10pct']:.3f}"
        )
    
    logging.info("="*60)

if __name__ == "__main__":
    main()
