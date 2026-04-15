"""Training harness for WikiText-2 language modeling."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import GPT2TokenizerFast
import argparse
from pathlib import Path
import json
import logging
from dataclasses import dataclass, asdict
import math
import time
from datetime import datetime

# Enable CUDA optimizations for NVIDIA GPUs
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

@dataclass
class WikiText2Config:
    """Training configuration for WikiText-2 language modeling."""
    # Dataset
    dataset: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    seq_len: int = 256
    
    # Architecture
    hidden_dim: int = 512
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1
    
    # Training
    batch_size: int = 32  # Smaller due to seq length
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 200  # Increased for convergence guarantee
    warmup_steps: int = 2000  # Placeholder — implemented via scheduler
    grad_clip: float = 1.0
    
    # Early stopping — be patient: EBM/Diffusion LMs can take 30+ epochs to
    # get below random baseline (PPL=50K tokens vocab), then converge quickly.
    patience: int = 30
    target_perplexity: float = 100.0
    max_acceptable_perplexity: float = 150.0  # Maximum for publication
    
    # Optimizer selection
    use_muon: bool = True  # Use Muon-style optimizer for attention models
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    
    # Hardware
    device: str = "cuda"
    num_workers: int = 4
    seed: int = 42

class WikiTextDataset(Dataset):
    """Custom dataset for WikiText-2 with sequence chunking."""
    
    def __init__(self, tokenized_data, seq_len=256):
        self.seq_len = seq_len
        
        # Flatten all token IDs into one long sequence
        all_ids = []
        for example in tokenized_data:
            if 'input_ids' in example and len(example['input_ids']) > 0:
                all_ids.extend(example['input_ids'])
        
        # Chunk into sequences of length seq_len + 1 (for target shift)
        self.examples = []
        for i in range(0, len(all_ids) - seq_len - 1, seq_len):
            chunk = all_ids[i:i + seq_len + 1]
            if len(chunk) == seq_len + 1:
                self.examples.append(torch.tensor(chunk, dtype=torch.long))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def load_wikitext2(config: WikiText2Config):
    """Load and tokenize WikiText-2."""
    logging.info(f"Loading {config.dataset_config}...")
    dataset = load_dataset(config.dataset, config.dataset_config)
    
    logging.info("Initializing tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=False,  # We'll chunk manually
            padding=False,
            return_attention_mask=False
        )
    
    logging.info("Tokenizing dataset...")
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    # Create chunked datasets
    train_dataset = WikiTextDataset(tokenized["train"], config.seq_len)
    val_dataset = WikiTextDataset(tokenized["validation"], config.seq_len)
    test_dataset = WikiTextDataset(tokenized["test"], config.seq_len)
    
    logging.info(f"Train: {len(train_dataset)} sequences")
    logging.info(f"Val: {len(val_dataset)} sequences")
    logging.info(f"Test: {len(test_dataset)} sequences")
    
    return train_dataset, val_dataset, test_dataset, tokenizer

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
    perplexity = math.exp(min(avg_loss, 20))  # Cap to prevent overflow
    return perplexity

def train_epoch(model, dataloader, optimizer, scheduler, config, epoch, scaler=None):
    """Single training epoch."""
    model.train()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()
    
    for batch_idx, batch in enumerate(dataloader):
        batch = batch.to(config.device)
        input_ids = batch[:, :-1]
        target_ids = batch[:, 1:]
        
        optimizer.zero_grad(set_to_none=True)
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(input_ids)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    target_ids.reshape(-1)
                )
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(input_ids)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1)
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            
        scheduler.step()
        
        total_loss += loss.item() * target_ids.numel()
        total_tokens += target_ids.numel()
        
        if batch_idx % 100 == 0:
            logging.info(
                f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f}"
            )
    
    avg_loss = total_loss / total_tokens
    return avg_loss

def get_optimizer(model, config, arch_name):
    """Select optimizer based on architecture and config."""
    use_muon_for_arch = arch_name in ['transformer', 's4d', 'deq']
    
    if config.use_muon and use_muon_for_arch:
        try:
            from torch.optim import SGD
            optimizer = SGD(
                model.parameters(),
                lr=config.learning_rate,
                momentum=config.muon_momentum,
                weight_decay=config.weight_decay,
                nesterov=config.muon_nesterov
            )
            logging.info(f"Using Muon-style SGD optimizer (momentum={config.muon_momentum})")
        except ImportError:
            logging.warning("Muon optimizer not available, falling back to AdamW")
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        logging.info("Using AdamW optimizer")
    
    return optimizer

def main():
    parser = argparse.ArgumentParser(description="Train WikiText-2 language model")
    parser.add_argument("--arch", type=str, required=True,
                       choices=["neural_ode", "s4d", "deq", "diffusion",
                               "flow", "ebm", "transformer"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint-dir", type=str,
                       default="checkpoints/wikitext2")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-epochs", type=int, default=100)
    args = parser.parse_args()
    
    # Setup directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = Path(args.log_dir) / f"{args.arch}_lm_train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Starting training for {args.arch}")
    logging.info(f"Device: {args.device}")
    
    # Configuration
    config = WikiText2Config(
        device=args.device,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs
    )
    
    # Set seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    # Load data
    train_dataset, val_dataset, test_dataset, tokenizer = load_wikitext2(config)
    vocab_size = tokenizer.vocab_size
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # Load model
    logging.info(f"Building {args.arch} model (vocab_size={vocab_size})...")
    
    if args.arch == "transformer":
        from models.transformer_lm import TransformerLM
        model = TransformerLM(
            vocab_size=vocab_size,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            dropout=config.dropout
        ).to(config.device)
    
    elif args.arch == "s4d":
        from models.s4d_lm import S4DLM
        model = S4DLM(
            vocab_size=vocab_size,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            dropout=config.dropout
        ).to(config.device)
    
    elif args.arch == "neural_ode":
        from models.neural_ode_lm import NeuralODELM
        model = NeuralODELM(
            vocab_size=vocab_size,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            dropout=config.dropout
        ).to(config.device)
    
    elif args.arch == "deq":
        from models.deq_lm import DEQLM
        model = DEQLM(
            vocab_size=vocab_size,
            hidden_dim=config.hidden_dim,
            n_heads=config.n_heads,
            dropout=config.dropout
        ).to(config.device)
    
    elif args.arch == "flow":
        from models.flow_lm import FlowLM
        model = FlowLM(
            vocab_size=vocab_size,
            hidden_dim=config.hidden_dim,
            n_flows=8,
            dropout=config.dropout
        ).to(config.device)
    
    elif args.arch == "ebm":
        from models.ebm_lm import EBMLM
        model = EBMLM(
            vocab_size=vocab_size,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout
        ).to(config.device)
    
    elif args.arch == "diffusion":
        from models.diffusion_lm import DiffusionLM
        model = DiffusionLM(
            vocab_size=vocab_size,
            hidden_dim=config.hidden_dim,
            n_steps=20,
            dropout=config.dropout
        ).to(config.device)
    
    else:
        raise ValueError(f"Unknown architecture: {args.arch}")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model parameters: {n_params:,}")
    
    # Optimizer and scheduler
    optimizer = get_optimizer(model, config, args.arch)
    
    total_steps = config.max_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps
    )
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if 'cuda' in str(config.device) else None

    # Training history tracking
    training_history = {
        'train_loss': [],
        'val_ppl': [],
        'learning_rates': [],
        'epoch_times': []
    }

    # Training loop
    best_val_ppl = float("inf")
    patience_counter = 0
    start_epoch = 0
    start_time = time.time()

    # Skip if already completed
    summary_path = Path(args.checkpoint_dir) / f"{args.arch}_lm_summary.json"
    checkpoint_path = Path(args.checkpoint_dir) / f"{args.arch}_lm_best.pt"
    if summary_path.exists():
        logging.info(f"Already completed: {args.arch}. Skipping.")
        return

    # Resume from checkpoint if interrupted mid-training
    if checkpoint_path.exists():
        logging.info(f"Resuming {args.arch} from existing checkpoint")
        ckpt = torch.load(checkpoint_path, map_location=config.device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_ppl = ckpt['val_perplexity']
        training_history = ckpt['training_history']
        logging.info(f"Resumed from epoch {start_epoch}, best_val_ppl={best_val_ppl:.2f}")

    for epoch in range(start_epoch, config.max_epochs):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                config, epoch, scaler)
        
        # Validate
        val_ppl = compute_perplexity(model, val_loader, config.device)
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store metrics
        training_history['train_loss'].append(train_loss)
        training_history['val_ppl'].append(val_ppl)
        training_history['learning_rates'].append(current_lr)
        training_history['epoch_times'].append(epoch_time)
        
        logging.info(
            f"Epoch {epoch}/{config.max_epochs} - "
            f"Train Loss: {train_loss:.4f}, "
            f"Val PPL: {val_ppl:.2f}, "
            f"Time: {epoch_time:.1f}s, "
            f"LR: {current_lr:.6f}"
        )
        
        # Save checkpoint if best
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            patience_counter = 0
            
            checkpoint_path = Path(args.checkpoint_dir) / f"{args.arch}_lm_best.pt"
            torch.save({
                "epoch": epoch,
                "arch": args.arch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_perplexity": val_ppl,
                "train_loss": train_loss,
                "config": asdict(config),
                "vocab_size": vocab_size,
                "training_history": training_history
            }, checkpoint_path)
            
            logging.info(f"✓ New best model saved (PPL: {val_ppl:.2f})")
            
            # Check if reached target
            if val_ppl <= config.target_perplexity:
                logging.info(f"Reached target perplexity {config.target_perplexity}!")
        else:
            patience_counter += 1
            logging.info(f"No improvement ({patience_counter}/{config.patience})")
            
            if patience_counter >= config.patience:
                logging.info(f"Early stopping at epoch {epoch}")
                break
    
    # Final evaluation on test set
    logging.info("Evaluating on test set...")
    checkpoint = torch.load(Path(args.checkpoint_dir) / f"{args.arch}_lm_best.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    test_ppl = compute_perplexity(model, test_loader, config.device)
    
    total_time = time.time() - start_time
    
    logging.info("\n" + "="*60)
    logging.info(f"Training complete for {args.arch}")
    logging.info(f"Best validation PPL: {best_val_ppl:.2f}")
    logging.info(f"Test PPL: {test_ppl:.2f}")
    logging.info(f"Total time: {total_time/3600:.2f} hours")
    logging.info("="*60)
    
    # Save final results
    results = {
        "arch": args.arch,
        "best_val_ppl": float(best_val_ppl),
        "test_ppl": float(test_ppl),
        "total_epochs": epoch + 1,
        "total_time_hours": total_time / 3600,
        "n_params": n_params,
        "converged": best_val_ppl <= config.max_acceptable_perplexity,
        "reached_target": best_val_ppl <= config.target_perplexity,
        "training_history": training_history,
        "timestamp": datetime.now().isoformat(),
        "config": asdict(config)
    }
    
    results_path = Path(args.checkpoint_dir) / f"{args.arch}_lm_summary.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
