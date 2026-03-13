"""SSM experiment model — Diagonal S4D-style SSM on synthetic sequence classification.

Architecture: Diagonal SSM following S4D (Gu et al. 2022).
  State: complex diagonal A, N=8 complex eigenvalues (16 real values).
  Input dim D=32, 2 layers, linear readout.

Discretization: Bilinear (Tustin) transform.
  For diagonal A_c (continuous), the discrete A_bar, B_bar are:
    A_bar[i] = (1 + dt/2 * A_c[i]) / (1 - dt/2 * A_c[i])   [element-wise complex]
    B_bar[i] = dt * B_c[i] / (1 - dt/2 * A_c[i])

  We parametrize A_c = -exp(log_A) (negative real part for stability).
  This ensures Re(A_c) < 0 always, so |A_bar| < 1 (discrete stability).

Recurrence: h[t] = A_bar * h[t-1] + B_bar * u[t]   [element-wise complex multiply]
            y[t] = Re(C * h[t]) + D * u[t]           [real-valued output]

Analog mapping:
  A_bar element-wise multiply → analog decay (RC exponential)
  B_bar * u → analogized as AnalogLinear (B_bar matrix-vector multiply)
  C * h → AnalogLinear

DOUBT NOTED: True S4D has complex-valued A_bar. analogize() only replaces
nn.Linear modules, not complex multiplications. The A_bar element-wise multiply
(diagonal state decay) is the "RC integrator" primitive — it stays as is
(implemented as torch.complex multiplication, which has no analog replacement).
What we're testing is the mismatch effect on the B and C projections
(which ARE AnalogLinear), plus the input D-linear projection.

This is realistic: on actual analog hardware, the diagonal A (RC decay) is
implemented as analog RC circuits (no replacement needed), while B and C
are the crossbar MVMs that get mismatch.

Task: Same synthetic sequence task as Transformer (pattern detection).
Same train/test data for direct comparison.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# ── Dataset — same as transformer for direct comparison ──────────────────

_VOCAB = 16
_SEQ_LEN = 10
_PATTERN = (3, 5)
_N_TRAIN = 4000
_N_TEST = 1000

def _generate_data(n, seed):
    rng = np.random.default_rng(seed)
    X, y = [], []
    while len(X) < n:
        seq = rng.integers(0, _VOCAB, size=_SEQ_LEN)
        label = any(seq[i] == _PATTERN[0] and seq[i+1] == _PATTERN[1] for i in range(_SEQ_LEN-1))
        X.append(seq)
        y.append(int(label))
    return torch.tensor(np.array(X), dtype=torch.long), torch.tensor(y, dtype=torch.long)

def _get_data():
    return _generate_data(_N_TRAIN, 42), _generate_data(_N_TEST, 43)


# ── Model ─────────────────────────────────────────────────────────────────

class _S4DLayer(nn.Module):
    """Diagonal SSM layer (S4D-style).

    State: h in C^N (complex)
    Parametrized A_c = -exp(log_A) → stable
    B, C, D: learned projections
    """
    def __init__(self, d_model=32, d_state=8, dt=1.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt = dt

        # Continuous-time A parameters (log-parametrized for stability)
        self.log_A_real = nn.Parameter(torch.zeros(d_state))
        self.log_A_imag = nn.Parameter(torch.randn(d_state) * 0.1)

        # B, C: projection matrices (will be analogized)
        self.B = nn.Linear(d_model, d_state * 2, bias=False)   # *2 for complex (re, im)
        self.C = nn.Linear(d_state * 2, d_model, bias=False)
        self.D = nn.Linear(d_model, d_model, bias=False)

        nn.init.normal_(self.B.weight, std=0.1)
        nn.init.normal_(self.C.weight, std=0.1)
        nn.init.eye_(self.D.weight)

    def _get_discrete_params(self):
        """Compute A_bar, B_bar from continuous-time params via bilinear."""
        dt = self.dt
        A_real = -torch.exp(self.log_A_real)     # Re < 0 → stable
        A_imag = self.log_A_imag
        A_c = torch.complex(A_real, A_imag)     # (d_state,) complex

        # Bilinear: A_bar = (1 + dt/2 * A_c) / (1 - dt/2 * A_c)
        half_dt_A = 0.5 * dt * A_c
        A_bar = (1 + half_dt_A) / (1 - half_dt_A)

        return A_bar  # (d_state,) complex

    def forward(self, u):
        """
        u: (batch, seq_len, d_model)
        Returns: (batch, seq_len, d_model)
        """
        B, T, D = u.shape
        A_bar = self._get_discrete_params()       # (d_state,) complex

        # B projection: (batch, T, d_model) → (batch, T, 2*d_state) → complex
        Bu = self.B(u)  # (batch, T, 2*d_state) — real representation
        # Split into real and imaginary parts
        Bu_re = Bu[..., :self.d_state]
        Bu_im = Bu[..., self.d_state:]
        Bu_c = torch.complex(Bu_re, Bu_im)   # (batch, T, d_state)

        # Recurrence: h[t] = A_bar * h[t-1] + B_bar * u[t]
        # B_bar = 1.0 (absorbed into B projection matrix for simplicity)
        h = torch.zeros(B, self.d_state, dtype=torch.complex64, device=u.device)
        ys = []
        for t in range(T):
            h = A_bar.unsqueeze(0) * h + Bu_c[:, t, :]
            h_real = torch.cat([h.real, h.imag], dim=-1)  # (batch, 2*d_state)
            y_t = self.C(h_real) + self.D(u[:, t, :])     # (batch, d_model)
            ys.append(y_t)

        return torch.stack(ys, dim=1)   # (batch, T, d_model)


class _SSMClassifier(nn.Module):
    def __init__(self, vocab=16, d_model=32, d_state=8, n_layers=2, n_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList([
            _S4DLayer(d_model=d_model, d_state=d_state) for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        h = self.embed(x)  # (B, T, D)
        for layer in self.layers:
            h = h + layer(h)  # residual connection
        return self.head(h.mean(dim=1))


# ── Standard interface ─────────────────────────────────────────────────────

def create_model() -> nn.Module:
    return _SSMClassifier(vocab=_VOCAB, d_model=12, d_state=8, n_layers=2)  # Reduced from 32 to operate near capacity


def train_model(model: nn.Module, save_path: str) -> nn.Module:
    (X_train, y_train), _ = _get_data()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    batch_size = 128
    n_epochs = 300

    model.train()
    for epoch in range(n_epochs):
        idx = torch.randperm(len(X_train))[:batch_size]
        logits = model(X_train[idx])
        loss = criterion(logits, y_train[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            acc = evaluate(model)
            print(f"  [SSM] epoch {epoch+1}/{n_epochs}: loss={loss.item():.4f}, acc={acc:.3f}")

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
    _, (X_test, y_test) = _get_data()
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        loss = nn.functional.cross_entropy(logits, y_test)
    return -loss.item()  # Negative so higher = better


def evaluate_output_mse(model: nn.Module, digital_baseline: nn.Module) -> float:
    """Compute MSE between analog and digital baseline outputs.
    
    Returns negative MSE so higher = better (consistent with other metrics).
    """
    _, (X_test, y_test) = _get_data()
    model.eval()
    digital_baseline.eval()
    
    with torch.no_grad():
        dig_out = digital_baseline(X_test)
        analog_out = model(X_test)
    
    mse = ((dig_out - analog_out) ** 2).mean().item()
    return -mse


def get_family_name() -> str:
    return "SSM"
