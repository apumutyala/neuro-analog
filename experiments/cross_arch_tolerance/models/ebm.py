"""EBM experiment model — Restricted Boltzmann Machine on binarized 8x8 MNIST.

Architecture: RBM with visible=64 (8x8), hidden=32.
Training: Contrastive Divergence CD-1.
Evaluation: 100 steps of Gibbs sampling from test data, measure reconstruction error.
  Metric: Negative reconstruction MSE [HIGHER = BETTER].

EBM analog mapping:
  h = sigmoid(W^T v + c) — crossbar MVM + sigmoid (AnalogSigmoid)
  v = sigmoid(W v + b)   — transposed crossbar + sigmoid

During analog evaluation:
  The W matrix (crossbar) and sigmoid (diff pair) are both analogized.
  Tests the core Boltzmann machine analog computing paradigm.

DOUBT NOTED: The directive says "initialize from test data, run k=100 Gibbs steps."
100 steps from the data is essentially running the Markov chain close to equilibrium
and checking if the chain stays close to real data. This is a proxy for model quality:
a well-trained RBM's Gibbs chain will oscillate near the data manifold.
Reconstruction error = ||v_k - v_0||^2 averaged over test set.

DOUBT NOTED: Contrastive Divergence (CD-1) uses 1 step of Gibbs sampling
to approximate the negative phase gradient. CD-k with k=1 is the standard
training procedure and works well for RBMs on binarized data.

DOUBT NOTED: Binarization threshold is 0.5 (pixel > 0.5 → 1, else 0).
The 8x8 MNIST grayscale values are in [0,1] after normalization.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# ── Dataset ───────────────────────────────────────────────────────────────

_VIS = 64
_HID = 32

def _load_mnist_8x8_binary(split="train", n=None):
    try:
        import torchvision
        import torchvision.transforms as T
        transform = T.Compose([T.Resize(8), T.ToTensor()])
        dataset = torchvision.datasets.MNIST(
            root=os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"),
            train=(split == "train"), download=True, transform=transform,
        )
        n = n or len(dataset)
        x = torch.stack([dataset[i][0].flatten() for i in range(min(n, len(dataset)))])
        return (x > 0.5).float()
    except Exception:
        pass

    # sklearn digits: 1797 samples of 8x8 images (0–16 scale), binarize at midpoint
    from sklearn.datasets import load_digits
    data = load_digits()
    X = (data.data.astype(np.float32) > 8.0).astype(np.float32)  # binarize [0,16] at 8
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(X))
    split_idx = int(len(X) * 0.8)
    idx = perm[:split_idx] if split == "train" else perm[split_idx:]
    n = n or len(idx)
    return torch.tensor(X[idx[:n]])

def _get_data():
    return _load_mnist_8x8_binary("train", 5000), _load_mnist_8x8_binary("test", 500)


# ── Model ─────────────────────────────────────────────────────────────────

class _RBM(nn.Module):
    """Restricted Boltzmann Machine.

    The W matrix is the single crossbar array — used in BOTH directions
    (v→h: W^T, h→v: W). In analog hardware, this is the same crossbar read
    in two modes (voltage on rows vs columns).

    We use nn.Linear to hold W (for analogize() to replace with AnalogLinear).
    Transpose in h→v pass is handled explicitly.

    DOUBT NOTED: analogize() replaces nn.Linear, which in PyTorch stores W as
    a (out, in) matrix. For W_fwd (vis→hid): W is (hid, vis). The backward pass
    W^T is (vis, hid). We create W_bwd as a second nn.Linear(hid, vis) and
    at the start of training, set W_bwd.weight = W_fwd.weight.T (they share
    the transposed weights conceptually, but in PyTorch they're separate parameters).
    This means the analog model has independent mismatch on W_fwd and W_bwd,
    which is realistic: RRAM read in forward vs transposed mode has different
    noise characteristics (different column vs row sense amplifiers).
    """
    def __init__(self, n_vis=64, n_hid=32):
        super().__init__()
        self.n_vis = n_vis
        self.n_hid = n_hid
        self.W_fwd = nn.Linear(n_vis, n_hid, bias=True)   # v → h
        self.W_bwd = nn.Linear(n_hid, n_vis, bias=True)   # h → v
        # Initialize: W_bwd.weight = W_fwd.weight.T
        nn.init.xavier_uniform_(self.W_fwd.weight)
        with torch.no_grad():
            self.W_bwd.weight.data = self.W_fwd.weight.data.T.clone()
        nn.init.zeros_(self.W_fwd.bias)
        nn.init.zeros_(self.W_bwd.bias)
        # Registered modules so analogize() replaces them with AnalogSigmoid
        self.act_h = nn.Sigmoid()   # v → h activation
        self.act_v = nn.Sigmoid()   # h → v activation

    def h_given_v(self, v):
        return self.act_h(self.W_fwd(v))

    def v_given_h(self, h):
        return self.act_v(self.W_bwd(h))

    def forward(self, v):
        """One Gibbs step: v → h → v'."""
        h_prob = self.h_given_v(v)
        h_sample = (h_prob > torch.rand_like(h_prob)).float()
        v_recon = self.v_given_h(h_sample)
        return v_recon, h_prob, h_sample


# ── Standard interface ─────────────────────────────────────────────────────

def create_model() -> nn.Module:
    return _RBM(n_vis=_VIS, n_hid=_HID)


def train_model(model: nn.Module, save_path: str) -> nn.Module:
    X_train, _ = _get_data()
    lr = 0.01
    batch_size = 64
    n_epochs = 200
    model.train()

    for epoch in range(n_epochs):
        idx = torch.randperm(len(X_train))[:batch_size]
        v_pos = X_train[idx]
        h_prob_pos = model.h_given_v(v_pos)
        h_sample = (h_prob_pos > torch.rand_like(h_prob_pos)).float()
        v_neg = model.v_given_h(h_sample)
        h_prob_neg = model.h_given_v(v_neg)

        # CD-1 weight gradient: <v h>_data - <v h>_model
        pos_grad = v_pos.T @ h_prob_pos / batch_size
        neg_grad = v_neg.T @ h_prob_neg / batch_size

        with torch.no_grad():
            model.W_fwd.weight.data += lr * (pos_grad.T - neg_grad.T)
            model.W_bwd.weight.data = model.W_fwd.weight.data.T.clone()
            model.W_fwd.bias.data += lr * (h_prob_pos.mean(0) - h_prob_neg.mean(0))
            model.W_bwd.bias.data += lr * (v_pos.mean(0) - v_neg.mean(0))

        if (epoch + 1) % 50 == 0:
            recon_err = evaluate(model)
            print(f"  [EBM] epoch {epoch+1}/{n_epochs}: neg_recon={recon_err:.4f}")

    torch.save(model.state_dict(), save_path)
    return model


def load_model(save_path: str) -> nn.Module:
    model = create_model()
    model.load_state_dict(torch.load(save_path, map_location="cpu"))
    return model


def evaluate(model: nn.Module) -> float:
    """k=100 Gibbs steps from test data, measure negative reconstruction MSE."""
    _, X_test = _get_data()
    v = X_test.clone()
    model.eval()

    with torch.no_grad():
        for _ in range(100):
            h_prob = model.h_given_v(v)
            h_sample = (h_prob > torch.rand_like(h_prob)).float()
            v_recon = model.v_given_h(h_sample)
            # Keep continuous values (probabilities), not hard samples, for stability
            v = v_recon

    mse = ((v - X_test) ** 2).mean().item()
    return -mse  # higher = better (lower reconstruction error)


def evaluate_output_mse(model: nn.Module, digital_baseline: nn.Module) -> float:
    """Compute MSE between analog and digital baseline reconstructions.
    
    Returns negative MSE so higher = better (consistent with other metrics).
    """
    _, X_test = _get_data()
    model.eval()
    digital_baseline.eval()
    
    with torch.no_grad():
        # RBM reconstruction: visible → hidden → visible (use model methods)
        h_prob_dig = digital_baseline.h_given_v(X_test)
        v_recon_dig = digital_baseline.v_given_h(h_prob_dig)
        
        h_prob_analog = model.h_given_v(X_test)
        v_recon_analog = model.v_given_h(h_prob_analog)
    
    mse = ((v_recon_dig - v_recon_analog) ** 2).mean().item()
    return -mse


def get_family_name() -> str:
    return "EBM"
