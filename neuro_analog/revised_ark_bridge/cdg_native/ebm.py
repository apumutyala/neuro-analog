"""
ebm.py — Energy-Based Model (RBM/Hopfield) compiled via Ark CDG.

Physics:  dx/dt = -x + sigmoid(W_sym @ x + b)

RBM units are {0,1}; mean-field Hopfield relaxation uses sigmoid.

W_sym is block-symmetric for visible+hidden bipartition:
  z = [v; h],  W_sym = [[0, W], [W^T, 0]]
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from ..core.paradigms import additive_recurrent
from ..core.compiler import compile_cdg


def build_ebm(rbm_model, mismatch_sigma: float = 0.0, vectorize: bool = True):
    """
    Compile a PyTorch RBM to Ark CDG via sigmoid mean-field relaxation.

    Args:
        rbm_model: PyTorch model with .W [n_hidden, n_visible],
                   .b_v [n_visible], .b_h [n_hidden]
        mismatch_sigma: per-weight relative mismatch std
        vectorize: use vectorized OptCompiler (recommended for n_total > 8)

    Returns (CktClass, mgr)
    """
    import torch

    # Map PyTorch RBM attributes to the Hopfield relaxation variables.
    # W_fwd: visible -> hidden (shape [n_hid, n_vis])
    W = rbm_model.W_fwd.weight.detach().cpu().numpy()  # [n_hid, n_vis]
    b_v = rbm_model.W_bwd.bias.detach().cpu().numpy()   # visible bias
    b_h = rbm_model.W_fwd.bias.detach().cpu().numpy()   # hidden bias

    n_vis, n_hid = W.shape[1], W.shape[0]
    n_total = n_vis + n_hid

    # Block-symmetric weight matrix
    W_sym = np.zeros((n_total, n_total))
    W_sym[:n_vis, n_vis:] = W.T
    W_sym[n_vis:, :n_vis] = W

    b = np.concatenate([b_v, b_h])

    # RBM units are {0,1}; mean-field Hopfield relaxation uses sigmoid.
    weights = {
        "J": W_sym,
        "b": b,
        "activation": jax.nn.sigmoid,
    }

    spec = additive_recurrent(n_total, activation="sigmoid", mismatch_sigma=mismatch_sigma)
    CktClass, mgr = compile_cdg(
        spec=spec,
        weights=weights,
        mismatch_sigma=mismatch_sigma,
        prog_name=f"EBM_{n_total}d",
        vectorize=vectorize,
        normalize_weight=False,
        do_clipping=False,
    )

    return CktClass, mgr
