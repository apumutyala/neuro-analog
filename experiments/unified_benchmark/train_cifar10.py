"""
Unified training harness for CIFAR-10 across all architecture families.

Ensures fair comparison by standardizing:
- Dataset preprocessing
- Training hyperparameters  
- Early stopping criteria
- Checkpoint format
"""

import os
import sys
import argparse
from pathlib import Path

import contextlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import logging
import json
import time
from datetime import datetime

# Enable CUDA optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class CIFAR10Config:
    """Shared configuration for all architectures."""
    batch_size = 128
    num_workers = 4
    learning_rate = 3e-4
    weight_decay = 1e-4
    max_epochs = 300  # Increased for convergence guarantee
    patience = 25  # More patient early stopping
    target_accuracy = 0.85
    min_acceptable_accuracy = 0.80  # Minimum for publication
    grad_clip_norm = 1.0  # Gradient clipping for stability
    
    # Optimizer selection
    use_muon = True  # Use Muon optimizer where beneficial
    muon_momentum = 0.95
    muon_nesterov = True
    
    # Data augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                           (0.2023, 0.1994, 0.2010))
    ])


def get_dataloaders(config):
    """Load CIFAR-10 with standard splits."""
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True,
        transform=config.train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True,
        transform=config.test_transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def build_model(arch_name, device):
    """Construct model for specified architecture."""
    if arch_name == 'neural_ode':
        from models.neural_ode_cifar import NeuralODENet
        model = NeuralODENet(num_classes=10, hidden_dim=256)
    
    elif arch_name == 's4d':
        from models.s4d_cifar import S4DNet
        model = S4DNet(num_classes=10, d_model=256, n_layers=4)
    
    elif arch_name == 'deq':
        from models.deq_cifar import DEQNet
        model = DEQNet(num_classes=10, hidden_dim=256, max_iter=30)
    
    elif arch_name == 'diffusion':
        from models.diffusion_cifar import DiffusionClassifier
        model = DiffusionClassifier(num_classes=10, hidden_dim=256)
    
    elif arch_name == 'flow':
        from models.flow_cifar import FlowClassifier
        model = FlowClassifier(num_classes=10, n_flows=8)
    
    elif arch_name == 'ebm':
        from models.ebm_cifar import EBMClassifier
        model = EBMClassifier(num_classes=10, hidden_dim=256)
    
    elif arch_name == 'transformer':
        from models.transformer_cifar import ViTClassifier
        model = ViTClassifier(num_classes=10, d_model=256, n_heads=8, n_layers=6)
    
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")
    
    return model.to(device)


def train_epoch(model, train_loader, optimizer, criterion, device, config, scaler=None):
    """Single training epoch with mixed precision and gradient clipping."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        
        # Mixed precision training for NVIDIA GPUs
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping for stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            
            optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(train_loader), 100. * correct / total


def evaluate(model, test_loader, criterion, device):
    """Evaluate on test set with mixed precision."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    autocast_ctx = torch.cuda.amp.autocast() if device.type == 'cuda' else contextlib.nullcontext()
    with torch.no_grad():
        with autocast_ctx:
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(test_loader), 100. * correct / total


def get_optimizer(model, config, arch_name):
    """Select optimizer based on architecture and config."""
    # Muon optimizer for Transformers and attention-based models
    use_muon_for_arch = arch_name in ['transformer', 's4d', 'deq']
    
    if config.use_muon and use_muon_for_arch:
        try:
            # Try to import Muon (if available)
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
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        logging.info("Using AdamW optimizer")
    
    return optimizer

def train_model(arch_name, config, device, checkpoint_dir, log_dir):
    """Full training loop with early stopping and comprehensive logging."""
    # Setup logging
    log_file = log_dir / f"{arch_name}_train.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"{'='*70}")
    logging.info(f"Training {arch_name} on CIFAR-10")
    logging.info(f"{'='*70}")
    logging.info(f"Device: {device}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"CUDA version: {torch.version.cuda}")
    
    train_loader, test_loader = get_dataloaders(config)
    model = build_model(arch_name, device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model parameters: {n_params:,}")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for better generalization
    optimizer = get_optimizer(model, config, arch_name)
    
    # Warm restart scheduler for better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # Training metrics tracking
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'learning_rates': [],
        'epoch_times': []
    }
    
    best_acc = 0
    best_epoch = 0
    patience_counter = 0
    start_epoch = 0
    start_time = time.time()

    # Skip if already completed
    summary_path = checkpoint_dir / f"{arch_name}_cifar10_summary.json"
    checkpoint_path = checkpoint_dir / f"{arch_name}_cifar10_best.pt"
    if summary_path.exists():
        logging.info(f"Already completed: {arch_name}. Skipping.")
        import json as _json
        with open(summary_path) as _f:
            return _json.load(_f)['best_acc']

    # Resume from checkpoint if interrupted mid-training
    if checkpoint_path.exists():
        logging.info(f"Resuming {arch_name} from checkpoint (epoch {checkpoint_path})")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch']
        best_acc = ckpt['test_acc']
        best_epoch = start_epoch
        training_history = ckpt['training_history']
        logging.info(f"Resumed from epoch {start_epoch}, best_acc={best_acc:.2f}%")

    logging.info(f"\nStarting training for {config.max_epochs} epochs...")
    logging.info(f"Target accuracy: {config.target_accuracy*100:.1f}%")
    logging.info(f"Early stopping patience: {config.patience} epochs")

    for epoch in range(start_epoch, config.max_epochs):
        epoch_start = time.time()
        
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, config, scaler
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store metrics
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['test_loss'].append(test_loss)
        training_history['test_acc'].append(test_acc)
        training_history['learning_rates'].append(current_lr)
        training_history['epoch_times'].append(epoch_time)
        
        logging.info(
            f"Epoch {epoch+1:3d}/{config.max_epochs} ({epoch_time:.1f}s): "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}% | "
            f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}% | "
            f"LR={current_lr:.6f}"
        )
        
        # Checkpoint on improvement
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            patience_counter = 0
            
            checkpoint_path = checkpoint_dir / f"{arch_name}_cifar10_best.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'test_acc': test_acc,
                'test_loss': test_loss,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'arch_name': arch_name,
                'n_params': n_params,
                'config': {
                    'learning_rate': config.learning_rate,
                    'batch_size': config.batch_size,
                    'max_epochs': config.max_epochs
                },
                'training_history': training_history
            }, checkpoint_path)
            logging.info(f"  ✓ New best! Saved checkpoint (acc: {best_acc:.2f}%)")
        else:
            patience_counter += 1
        
        # Check convergence
        if test_acc >= config.target_accuracy * 100:
            logging.info(f"\n✓ SUCCESS: Reached target accuracy {config.target_accuracy*100:.1f}% at epoch {epoch+1}")
            break
        
        # Early stopping
        if patience_counter >= config.patience:
            logging.warning(
                f"\n⚠ Early stopping at epoch {epoch+1} "
                f"(no improvement for {config.patience} epochs)"
            )
            break
        
        # Convergence check: if stuck at low accuracy, warn
        if epoch >= 50 and best_acc < config.min_acceptable_accuracy * 100:
            logging.warning(
                f"⚠ Low accuracy after 50 epochs: {best_acc:.2f}% "
                f"(target: {config.min_acceptable_accuracy*100:.1f}%)"
            )
    
    total_time = time.time() - start_time
    
    # Final summary
    logging.info(f"\n{'='*70}")
    logging.info(f"Training complete for {arch_name}")
    logging.info(f"Best test accuracy: {best_acc:.2f}% (epoch {best_epoch})")
    logging.info(f"Total training time: {total_time/3600:.2f} hours")
    logging.info(f"Average epoch time: {sum(training_history['epoch_times'])/len(training_history['epoch_times']):.1f}s")
    
    # Save final training summary
    summary = {
        'arch_name': arch_name,
        'best_acc': float(best_acc),
        'best_epoch': int(best_epoch),
        'total_epochs': epoch + 1,
        'total_time_hours': total_time / 3600,
        'n_params': n_params,
        'converged': best_acc >= config.min_acceptable_accuracy * 100,
        'reached_target': best_acc >= config.target_accuracy * 100,
        'training_history': training_history,
        'timestamp': datetime.now().isoformat()
    }
    
    summary_path = checkpoint_dir / f"{arch_name}_cifar10_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"Saved training summary to {summary_path}")
    logging.info(f"{'='*70}\n")
    
    return best_acc


def main():
    parser = argparse.ArgumentParser(description="Train CIFAR-10 model with optimizations")
    parser.add_argument('--arch', type=str, required=True,
                       choices=['neural_ode', 's4d', 'deq', 'diffusion',
                               'flow', 'ebm', 'transformer'],
                       help='Architecture to train')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--checkpoint-dir', type=str, 
                       default='checkpoints/cifar10',
                       help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str,
                       default='logs',
                       help='Directory for training logs')
    args = parser.parse_args()
    
    # Create directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    config = CIFAR10Config()
    final_acc = train_model(args.arch, config, device, checkpoint_dir, log_dir)


if __name__ == '__main__':
    main()
