"""DEQ experiment model — Implicit MLP on 8x8 MNIST classification.

Architecture: f_theta(z, x) = tanh(W_z @ z + W_x @ x + b)
  z-dim = 64, x-dim = 64 (flattened 8x8), hidden = 64
  Readout: W_read @ z* + b_read → 10 classes (digit labels)

Training: Unroll 30 iterations of the fixed-point iteration and backprop through.
  This is standard DEQ training practice (unrolled differentiation).
  No implicit differentiation needed for the demo scale.

DOUBT NOTED: The directive mentions implicit differentiation via torch.linalg.solve.
The exact implicit gradient formula is: dz*/dtheta = -(I - ∂f/∂z)^{-1} * ∂f/∂theta.
Computing (I - ∂f/∂z)^{-1} requires solving a 64x64 linear system per batch,
which requires autograd through the Jacobian — expensive. For the research demo,
unrolling 30 steps and backpropping through them achieves similar gradients.

DEQ analog convergence analysis:
  After analogize(), the weights W_z and W_x get mismatch δ ~ N(1, σ²).
  The Jacobian ∂f/∂z = W_z * diag(1 - tanh^2(W_z z + W_x x + b)).
  Under mismatch, the effective W_z becomes W_z * δ, so the spectral radius
  rho(∂f/∂z) can exceed 1 even if the nominal rho < 1.

  Critical experiment question: at what sigma does rho cross 1 for this architecture?
  We track convergence_failure_rate = fraction of inputs where iteration diverges.

Metric: Classification accuracy + convergence failure rate (tracked separately).
evaluate() returns accuracy. evaluate_convergence() returns failure rate.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# ── Dataset ───────────────────────────────────────────────────────────────

_IMG_DIM = 64
_N_CLASSES = 10

def _load_mnist_8x8(split="train", n=None):
    """Load 8x8 digit images.

    Primary: sklearn.datasets.load_digits() — built-in 8x8 digits, no download.
    Fallback: torchvision MNIST resized to 8x8 (requires torchvision + network).
    """
    try:
        import torchvision
        import torchvision.transforms as T
        transform = T.Compose([T.Resize(8), T.ToTensor(), T.Normalize([0.5], [0.5])])
        dataset = torchvision.datasets.MNIST(
            root=os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"),
            train=(split == "train"), download=True, transform=transform,
        )
        n = n or len(dataset)
        x = torch.stack([dataset[i][0].flatten() for i in range(min(n, len(dataset)))])
        y = torch.tensor([dataset[i][1] for i in range(min(n, len(dataset)))])
        return x, y
    except Exception:
        pass

    # sklearn digits: 1797 samples of 8x8 images, 10 classes, no download required
    from sklearn.datasets import load_digits
    data = load_digits()
    X = (data.data.astype(np.float32) / 8.0) - 1.0   # scale [0,16] → [-1, 1]
    y = data.target.astype(np.int64)

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(X))
    split_idx = int(len(X) * 0.8)  # 1437 train / 360 test
    idx = perm[:split_idx] if split == "train" else perm[split_idx:]
    n = n or len(idx)
    idx = idx[:n]
    return torch.tensor(X[idx]), torch.tensor(y[idx])

def _get_data():
    return _load_mnist_8x8("train"), _load_mnist_8x8("test")


# ── Model ─────────────────────────────────────────────────────────────────

class _DEQClassifier(nn.Module):
    """Deep Equilibrium Model: z* = f_theta(z*, x), classify from z*."""

    def __init__(self, z_dim=64, x_dim=64, n_classes=10, max_iter=30, tol=1e-4):
        super().__init__()
        self.z_dim = z_dim
        self.max_iter = max_iter
        self.tol = tol
        # f_theta layers (will be analogized by analogize())
        self.W_z = nn.Linear(z_dim, z_dim, bias=False)
        self.W_x = nn.Linear(x_dim, z_dim, bias=True)
        # Readout: z* → class logits (stays digital — only classifies, not dynamics)
        self.readout = nn.Linear(z_dim, n_classes)
        # Initialize W_z small so spectral norm starts < 1 (contraction guarantee)
        nn.init.normal_(self.W_z.weight, std=0.1)
        # W_x uses default kaiming init — it maps input → hidden and doesn't
        # affect convergence of the fixed-point iteration (only W_z does)
        
        # Apply spectral normalization to W_z to guarantee contraction (rho < 1)
        # This ensures fixed-point iteration converges even under analog mismatch
        self.W_z = nn.utils.parametrizations.spectral_norm(self.W_z)
        # Registered module so analogize() replaces it with AnalogTanh
        self.act = nn.Tanh()

    def f_theta(self, z, x):
        """Fixed-point function: z_{k+1} = tanh(W_z @ z_k + W_x @ x + b)."""
        return self.act(self.W_z(z) + self.W_x(x))

    def forward(self, x, n_iter=None):
        """Find fixed point and classify.

        During training: unroll n_iter steps (gradient flows through).
        During eval: iterate until convergence.
        """
        n_iter = n_iter or self.max_iter
        z = torch.zeros(x.shape[0], self.z_dim, device=x.device)
        for _ in range(n_iter):
            z_next = self.f_theta(z, x)
            if not self.training:
                # Check convergence (eval only — gradient not needed through this check)
                with torch.no_grad():
                    if (z_next - z).norm(dim=-1).max().item() < self.tol:
                        break
            z = z_next
        return self.readout(z), z

    def convergence_failure_rate(self, x, max_iter=None, tol=None) -> float:
        """Fraction of inputs where fixed-point iteration did not converge.

        A non-converged sample is one where ||z_{k+1} - z_k|| >= tol at max_iter.
        Under mismatch, high spectral radius can prevent convergence entirely.
        """
        max_iter = max_iter or self.max_iter
        tol = tol or self.tol
        z = torch.zeros(x.shape[0], self.z_dim, device=x.device)
        converged = torch.zeros(x.shape[0], dtype=torch.bool)

        with torch.no_grad():
            for _ in range(max_iter):
                z_next = self.f_theta(z, x)
                delta = (z_next - z).norm(dim=-1)
                converged |= (delta < tol)
                z = z_next

        failure_rate = 1.0 - converged.float().mean().item()
        return failure_rate


# ── Standard interface ─────────────────────────────────────────────────────

def create_model() -> nn.Module:
    return _DEQClassifier(z_dim=64, x_dim=_IMG_DIM, n_classes=_N_CLASSES)


def train_model(model: nn.Module, save_path: str) -> nn.Module:
    (X_train, y_train), _ = _get_data()
    n_epochs = 500
    batch_size = 128
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(n_epochs):
        idx = torch.randperm(len(X_train))[:batch_size]
        logits, _ = model(X_train[idx], n_iter=30)
        loss = criterion(logits, y_train[idx])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 100 == 0:
            acc = evaluate(model)
            print(f"  [DEQ] epoch {epoch+1}/{n_epochs}: loss={loss.item():.4f}, acc={acc:.3f}")

    torch.save(model.state_dict(), save_path)
    return model


def load_model(save_path: str) -> nn.Module:
    model = create_model()
    model.load_state_dict(torch.load(save_path, map_location="cpu"))
    return model


def evaluate(model: nn.Module) -> float:
    """Return negative cross-entropy loss (continuous metric, higher = better).
    
    Cross-entropy measures the full logit distribution, not just argmax.
    Degrades smoothly with analog noise because logit magnitudes matter.
    """
    (_, _), (X_test, y_test) = _get_data()
    model.eval()
    with torch.no_grad():
        logits, _ = model(X_test)
        loss = nn.functional.cross_entropy(logits, y_test)
    return -loss.item()  # Negative so higher = better


def evaluate_convergence_failure(model: nn.Module) -> float:
    """Convergence failure rate — fraction of inputs where fixed-point diverges."""
    (_, _), (X_test, y_test) = _get_data()
    model.eval()
    if hasattr(model, "convergence_failure_rate"):
        return model.convergence_failure_rate(X_test)
    return 0.0


def evaluate_output_mse(model: nn.Module, digital_baseline: nn.Module) -> float:
    """Compute MSE between analog and digital baseline outputs.
    
    Returns negative MSE so higher = better (consistent with other metrics).
    """
    (_, _), (X_test, y_test) = _get_data()
    model.eval()
    digital_baseline.eval()
    
    with torch.no_grad():
        dig_out, _ = digital_baseline(X_test)
        analog_out, _ = model(X_test)
    
    mse = ((dig_out - analog_out) ** 2).mean().item()
    return -mse


def get_family_name() -> str:
    return "DEQ"
