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
from neuro_analog.ir.energy_model import HardwareProfile

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

    # Hardware profile for energy/latency estimation
    profile = HardwareProfile()

    # Use mismatch_sweep for consistency with CIFAR-10 and to get energy/latency metrics
    result = mismatch_sweep(
        model,
        eval_fn=lambda m: compute_perplexity(m, test_loader, device),
        sigma_values=sigma_values,
        n_trials=n_trials,
        n_adc_bits=8,
        calibration_data=calibration_batch,
        analog_domain='conservative',
        hardware_profile=profile,
    )

    # Save results using SweepResult's save method (includes energy/latency)
    output_path = Path(output_dir) / f"{arch}_lm_sweep.json"
    result.save(str(output_path))

    threshold = result.degradation_threshold(0.10)
    logging.info(f"Threshold @ 10% degradation: σ={threshold:.3f}")
    if result.energy_saving_vs_digital is not None:
        logging.info(f"Energy saving vs digital: {result.energy_saving_vs_digital*100:.1f}%")
    if result.speedup_vs_digital is not None:
        logging.info(f"Speedup vs digital: {result.speedup_vs_digital:.1f}x")
    logging.info(f"Results saved to {output_path}")

    return result

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
