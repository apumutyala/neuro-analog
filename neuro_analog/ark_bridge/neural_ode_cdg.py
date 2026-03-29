"""
CDG bridge: PyTorch Neural ODE weights → Ark Constrained Dynamical Graph.

Architecture
------------
    dx/dt = -x + J * tanh(x) + b + K * u

    x ∈ R^n   — ODE state variables
    J ∈ R^n×n — recurrent weight matrix  (analog, mismatch-tagged)
    b ∈ R^n   — per-neuron bias           (analog, mismatch-tagged)
    K ∈ R^n×m — input weight matrix       (analog, mismatch-tagged)
    u ∈ R^m   — fixed external input

CDGSpec structure
-----------------
    Node types
        StateVar  (order=1)  — ODE state x_i, carries bias b_i
        OutUnit   (order=0)  — algebraic activation out_i = tanh(x_i)
        InpNode   (order=1)  — fixed input u_j (Inp-style)

    Edge types
        MapEdge   (no attrs) — ReadOut and SelfDecay wiring
        FlowEdge  (attr: g)  — J and K weight edges; g tagged with
                               AttrDefMismatch(rstd=sigma) when sigma > 0

    Production rules
        ReadOut    MapEdge  StateVar → OutUnit   DST  DST.act(VAR(SRC))
        SelfDecay  MapEdge  StateVar → StateVar  SELF -VAR(SRC) + SRC.b
        JWeight    FlowEdge OutUnit  → StateVar  DST  EDGE.g * VAR(SRC)
        KWeight    FlowEdge InpNode  → StateVar  DST  EDGE.g * VAR(SRC)

    Full dynamics (per node i):
        out_i    = tanh(x_i)                          [algebraic]
        dx_i/dt  = -x_i + b_i                         [SelfDecay]
                 + Σ_j  J[i,j] · tanh(x_j)           [JWeight]
                 + Σ_k  K[i,k] · u_k                 [KWeight]

This spec is structurally identical to Ark's CNN paradigm
(examples/cnn/spec.py), which uses the same IdealV/Out/Inp split and the
same MapEdge/FlowEdge/SelfFeedback/ReadOut/Amat/Bmat rules.  We give the
types Neural-ODE-appropriate names and add a bias attribute instead of the
CNN's `z` field to make the mapping explicit.

Why this form?
--------------
dx/dt = -x + J*tanh(x) + b is a standard Hopfield / Cohen-Grossberg
network.  It maps 1-to-1 to the CANN/CNN dynamical paradigm in Ark.  Our
cross-arch experiments train Neural ODEs of exactly this form (state_dim=2,
no external input).  For models trained with time augmentation or deeper
MLPs, see notes in docs/ark_export.md — the current bridge handles the
single-hidden-layer recurrent case; deeper feedforward MLPs require
additional order-0 hidden-unit nodes (follow the same pattern, add more
ReadOut/Flow layers).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# ── Spec factory ──────────────────────────────────────────────────────────────

def make_neural_ode_spec(mismatch_sigma: float = 0.0):
    """
    Build and return the CDGSpec for a Hopfield-type Neural ODE.

    Args:
        mismatch_sigma: per-weight relative std (rstd) for AttrDefMismatch.
                        Pass 0.0 for a nominal (no-mismatch) spec.

    Returns:
        (spec, StateVar, OutUnit, InpNode, MapEdge, FlowEdge)
    """
    from ark.reduction import SUM
    from ark.specification.attribute_def import AttrDef, AttrDefMismatch
    from ark.specification.attribute_type import AnalogAttr, FunctionAttr
    from ark.specification.cdg_types import EdgeType, NodeType
    from ark.specification.production_rule import ProdRule
    from ark.specification.rule_keyword import DST, EDGE, SELF, SRC, VAR
    from ark.specification.specification import CDGSpec
    spec = CDGSpec("neural_ode")

    StateVar = NodeType(
        name="StateVar",
        attrs={
            "order": 1,
            "reduction": SUM,
            "attr_def": {
                "b": AttrDef(attr_type=AnalogAttr((-10, 10))),
            },
        },
    )
    OutUnit = NodeType(
        name="OutUnit",
        attrs={
            "order": 0,
            "reduction": SUM,
            "attr_def": {
                "act": AttrDef(attr_type=FunctionAttr(nargs=1)),
            },
        },
    )
    InpNode = NodeType(name="InpNode", attrs={"order": 1, "reduction": SUM})
    MapEdge = EdgeType(name="MapEdge")

    if mismatch_sigma > 0.0:
        FlowEdge = EdgeType(
            name="FlowEdge",
            attrs={
                "attr_def": {
                    "g": AttrDefMismatch(
                        attr_type=AnalogAttr((-10, 10)), rstd=mismatch_sigma
                    )
                }
            },
        )
    else:
        FlowEdge = EdgeType(
            name="FlowEdge",
            attrs={"attr_def": {"g": AttrDef(attr_type=AnalogAttr((-10, 10)))}},
        )

    spec.add_cdg_types([StateVar, OutUnit, InpNode, MapEdge, FlowEdge])

    ReadOut   = ProdRule(MapEdge, StateVar, OutUnit,  DST,  DST.act(VAR(SRC)))
    SelfDecay = ProdRule(MapEdge, StateVar, StateVar, SELF, -VAR(SRC) + SRC.b)
    JWeight   = ProdRule(FlowEdge, OutUnit,  StateVar, DST,  EDGE.g * VAR(SRC))
    KWeight   = ProdRule(FlowEdge, InpNode,  StateVar, DST,  EDGE.g * VAR(SRC))
    # Hold InpNodes constant (ddt=0).  Mirrors CNN spec.py's Dummy rule —
    # without this, OptCompiler generates next_ddt_InpNode_i but never assigns it.
    InpHold   = ProdRule(FlowEdge, InpNode,  StateVar, SRC,  0)

    spec.add_production_rules([ReadOut, SelfDecay, JWeight, KWeight, InpHold])

    return spec, StateVar, OutUnit, InpNode, MapEdge, FlowEdge


# ── CDG builder ───────────────────────────────────────────────────────────────

def neural_ode_to_cdg(
    J: np.ndarray,
    b: np.ndarray,
    K: Optional[np.ndarray] = None,
    mismatch_sigma: float = 0.05,
):
    """
    Build an Ark CDG from Neural ODE weight matrices.

    Architecture:  dx/dt = -x + J * tanh(x) + b + K * u

    Args:
        J:               (n, n) recurrent weight matrix.
        b:               (n,) bias vector.
        K:               (n, m) input weight matrix; None for autonomous system.
        mismatch_sigma:  relative weight mismatch std (0 = ideal).

    Returns:
        (cdg, spec, trainable_mgr, state_nodes, inp_nodes)
    """
    import jax.numpy as jnp
    from ark.cdg.cdg import CDG
    from ark.specification.trainable import TrainableMgr
    spec, StateVar, OutUnit, InpNode, MapEdge, FlowEdge = make_neural_ode_spec(
        mismatch_sigma
    )

    n = J.shape[0]
    m = K.shape[1] if K is not None else 0

    mgr = TrainableMgr()
    cdg = CDG()

    state_nodes = [
        StateVar(b=mgr.new_analog(init_val=float(b[i]))) for i in range(n)
    ]
    out_nodes = [OutUnit(act=lambda x: jnp.tanh(x)) for _ in range(n)]

    for s_node, o_node in zip(state_nodes, out_nodes):
        cdg.add_node(s_node)
        cdg.add_node(o_node)

    for s_node, o_node in zip(state_nodes, out_nodes):
        cdg.connect(MapEdge(), s_node, s_node)
        cdg.connect(MapEdge(), s_node, o_node)

    for i in range(n):
        for j in range(n):
            w = float(J[i, j])
            if abs(w) < 1e-12:
                continue
            edge = FlowEdge(g=mgr.new_analog(init_val=w))
            cdg.connect(edge, out_nodes[j], state_nodes[i])

    inp_nodes = []
    if K is not None:
        inp_nodes = [InpNode() for _ in range(m)]
        for inp_node in inp_nodes:
            cdg.add_node(inp_node)
        for i in range(n):
            for j in range(m):
                w = float(K[i, j])
                if abs(w) < 1e-12:
                    continue
                edge = FlowEdge(g=mgr.new_analog(init_val=w))
                cdg.connect(edge, inp_nodes[j], state_nodes[i])

    return cdg, spec, mgr, state_nodes, inp_nodes


# ── Full pipeline convenience ──────────────────────────────────────────────────

def compile_neural_ode_cdg(
    J: np.ndarray,
    b: np.ndarray,
    K: Optional[np.ndarray] = None,
    mismatch_sigma: float = 0.05,
    prog_name: str = "NeuralODECkt",
):
    """
    Full pipeline: weight matrices → BaseAnalogCkt subclass via OptCompiler.

    Returns:
        (CktClass, trainable_mgr)
    """
    from ark.optimization.opt_compiler import OptCompiler
    cdg, spec, mgr, state_nodes, _ = neural_ode_to_cdg(J, b, K, mismatch_sigma)

    CktClass = OptCompiler().compile(
        prog_name=prog_name,
        cdg=cdg,
        cdg_spec=spec,
        trainable_mgr=mgr,
        readout_nodes=state_nodes,
        normalize_weight=False,
        do_clipping=False,
        aggregate_args_lines=mismatch_sigma > 0.0,
    )

    return CktClass, mgr
