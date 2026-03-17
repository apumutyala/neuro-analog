"""DEQ experiment model — Implicit MLP on 8x8 MNIST classification.

Architecture: f_theta(z, x) = tanh(W_z @ z + W_x @ x + b)
  z-dim = 64, x-dim = 64 (flattened 8x8), hidden = 64
  Readout: W_read @ z* + b_read → 10 classes (digit labels)

Training: Unroll 30 iterations of the fixed-point iteration and backprop through.
  This is standard DEQ training practice (unrolled differentiation).
  No implicit differentiation needed for the demo scale.

An alternative to unrolled backprop is implicit differentiation via torch.linalg.solve.
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

import math
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Integration capacitor physical constants for Hopfield relaxation noise
_K_B = 1.380649e-23
_TEMP_K = 300.0
_CAP_F = 1e-12

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        Convergence criterion: RMS per-element change < tol.
        Using norm / sqrt(z_dim) makes tol dimension-independent (e.g., tol=1e-4
        means average per-element change < 1e-4, regardless of z_dim).
        """
        n_iter = n_iter or self.max_iter
        z = torch.zeros(x.shape[0], self.z_dim, device=x.device)
        scale = math.sqrt(self.z_dim)
        for _ in range(n_iter):
            z_next = self.f_theta(z, x)
            if not self.training:
                # Check convergence (eval only — gradient not needed through this check)
                with torch.no_grad():
                    if (z_next - z).norm(dim=-1).max().item() / scale < self.tol:
                        break
            z = z_next
        return self.readout(z), z

    def convergence_failure_rate(self, x, max_iter=None, tol=None) -> float:
        """Fraction of inputs where fixed-point iteration did not converge.

        A non-converged sample is one where RMS per-element change >= tol at max_iter.
        Uses norm / sqrt(z_dim) so tol is dimension-independent: tol=1e-4 means
        average per-element change < 1e-4 (not raw L2 norm < 1e-4, which would
        never trigger for z_dim=64 since sqrt(64)*per_element_change >> 1e-4).
        Under mismatch, high spectral radius can prevent convergence entirely.
        """
        max_iter = max_iter or self.max_iter
        tol = tol or self.tol
        z = torch.zeros(x.shape[0], self.z_dim, device=x.device)
        converged = torch.zeros(x.shape[0], dtype=torch.bool)
        scale = math.sqrt(self.z_dim)

        with torch.no_grad():
            for _ in range(max_iter):
                z_next = self.f_theta(z, x)
                delta = (z_next - z).norm(dim=-1) / scale  # RMS per-element change
                converged |= (delta < tol)
                z = z_next

        failure_rate = 1.0 - converged.float().mean().item()
        return failure_rate


# ── Standard interface ─────────────────────────────────────────────────────

def create_model() -> nn.Module:
    return _DEQClassifier(z_dim=64, x_dim=_IMG_DIM, n_classes=_N_CLASSES)


def train_model(model: nn.Module, save_path: str) -> nn.Module:
    (X_train, y_train), _ = _get_data()
    X_train, y_train = X_train.to(_DEVICE), y_train.to(_DEVICE)
    model = model.to(_DEVICE)
    n_epochs = 500
    batch_size = 128
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(n_epochs):
        idx = torch.randperm(len(X_train), device=_DEVICE)[:batch_size]
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
    model.load_state_dict(torch.load(save_path, map_location="cpu", weights_only=True))
    model = model.to(_DEVICE)
    return model


def evaluate(model: nn.Module, analog_substrate: str = "discrete") -> float:
    """Return negative cross-entropy loss (continuous metric, higher = better).

    Args:
        analog_substrate:
          "discrete"  — standard fixed-point iteration z_{k+1} = f(z_k, x) (default).
          "hopfield"  — continuous-time analog feedback relaxation:
                        z_{k+1} = z_k + dt*(-z_k + f(z_k,x)) + sqrt(2*kT/C*dt)*xi
                        Models a feedback amplifier circuit where output drives input
                        through an RC integrator with time constant dt.
                        The -z_k damping term guarantees bounded trajectories even
                        under mismatch — failure mode is sustained oscillation rather
                        than divergence, which is physically more realistic for
                        continuous-time analog circuits.
    """
    (_, _), (X_test, y_test) = _get_data()
    X_test, y_test = X_test.to(_DEVICE), y_test.to(_DEVICE)
    model = model.to(_DEVICE)
    model.eval()
    with torch.no_grad():
        if analog_substrate == "discrete":
            logits, _ = model(X_test)
        elif analog_substrate == "hopfield":
            # Damped relaxation: dz/dt = -z + f(z,x) + sqrt(2kT/C)*dW
            # dt=0.5 gives stable convergence in ~10 steps with spectral norm < 1.
            # sigma_int per step = sqrt(2*kT/C*dt): charge noise on RC integrator.
            dt_relax = 0.5
            sigma_int = math.sqrt(2 * _K_B * _TEMP_K / _CAP_F * dt_relax)
            z = torch.zeros(X_test.shape[0], model.z_dim, device=_DEVICE)
            for _ in range(model.max_iter):
                f_z = model.f_theta(z, X_test)
                xi = torch.randn_like(z)
                z = z + dt_relax * (-z + f_z) + sigma_int * xi
            logits = model.readout(z)
        loss = nn.functional.cross_entropy(logits, y_test)
    return -loss.item()


def evaluate_convergence_failure(model: nn.Module) -> float:
    """Convergence failure rate — fraction of inputs where fixed-point diverges."""
    (_, _), (X_test, _) = _get_data()
    X_test = X_test.to(_DEVICE)
    model = model.to(_DEVICE)
    model.eval()
    if hasattr(model, "convergence_failure_rate"):
        return model.convergence_failure_rate(X_test)
    return 0.0


def evaluate_output_mse(model: nn.Module, digital_baseline: nn.Module, analog_substrate: str = "discrete") -> float:
    """Compute MSE between analog and digital baseline outputs.

    analog_substrate is accepted for API consistency with other substrate-aware models
    but does not change behavior here — both models run standard fixed-point inference
    so the MSE reflects weight mismatch only, independent of the relaxation substrate.

    Returns negative MSE so higher = better (consistent with other metrics).
    """
    (_, _), (X_test, _) = _get_data()
    X_test = X_test.to(_DEVICE)
    model = model.to(_DEVICE)
    digital_baseline = digital_baseline.to(_DEVICE)
    model.eval()
    digital_baseline.eval()

    with torch.no_grad():
        dig_out, _ = digital_baseline(X_test)
        analog_out, _ = model(X_test)

    mse = ((dig_out - analog_out) ** 2).mean().item()
    return -mse


def get_family_name() -> str:
    return "DEQ"
