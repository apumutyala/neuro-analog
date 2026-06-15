"""
cdg_build.py — Generic CDG → OptCompiler → BaseAnalogCkt builder.

Provides a single reusable function: compile_cdg(spec, weights, mismatch_sigma, ...).
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import diffrax

from ark.cdg.cdg import CDG
from ark.optimization.base_module import BaseAnalogCkt, TimeInfo
from ark.optimization.opt_compiler import OptCompiler
from ark.specification.cdg_types import EdgeType, NodeType
from ark.specification.trainable import TrainableMgr


def build_additive_cdg(spec, weights, mgr):
    """
    Build a CDG instance for additive_recurrent form (Hopfield/EBM/DEQ).

    Args:
        spec: CDGSpec with additive_recurrent structure
        weights: dict with keys 'J' [n,n], 'b' [n], optional 'K' [n,m], 'activation'
        mgr: TrainableMgr instance

    Returns (cdg, state_nodes, inp_nodes)
    """
    StateVar = spec.node_type("StateVar")
    OutUnit = spec.node_type("OutUnit")
    MapEdge = spec.edge_type("MapEdge")
    FlowEdge = spec.edge_type("FlowEdge")
    J = weights["J"]
    b = weights["b"]
    K = weights.get("K", None)
    n = J.shape[0]

    _raw_act = weights.get("activation", jnp.tanh)
    # JAX pjit primitives (jnp.tanh, jax.nn.sigmoid) lack __code__, which Ark's
    # FunctionAttr validator requires to verify co_argcount == 1.  Wrapping in a
    # real Python function satisfies the spec check without affecting JAX tracing.
    def _act_wrapper(x):
        return _raw_act(x)
    activation = _act_wrapper

    cdg = CDG()
    state_nodes = [
        StateVar(b=mgr.new_analog(init_val=float(b[i]))) for i in range(n)
    ]
    out_nodes = [OutUnit(act=activation) for _ in range(n)]

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
        m = K.shape[1]
        from ark.specification.cdg_types import NodeType
        InpNode = NodeType(name="InpNode", attrs={"order": 0, "attr_def": {}})
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

    return cdg, state_nodes, inp_nodes


def compile_cdg(
    spec,
    weights: dict,
    mismatch_sigma: float = 0.0,
    prog_name: str = "Ckt",
    vectorize: bool = False,
    readout_nodes=None,
    normalize_weight: bool = False,
    do_clipping: bool = False,
    aggregate_args_lines: bool = False,
):
    """
    Generic CDG compiler: spec + weights → OptCompiler → BaseAnalogCkt subclass.

    Returns (CktClass, mgr)
    """
    from ark.specification.trainable import TrainableMgr
    from ark.optimization.opt_compiler import OptCompiler

    mgr = TrainableMgr()

    # Build CDG based on spec family
    if "additive_recurrent" in spec.name or "deq" in spec.name:
        StateVar = spec.node_type("StateVar")
        OutUnit = spec.node_type("OutUnit")
        MapEdge = spec.edge_type("MapEdge")
        FlowEdge = spec.edge_type("FlowEdge")
        cdg, state_nodes, inp_nodes = build_additive_cdg(
            spec, weights, mgr
        )
        if readout_nodes is None:
            readout_nodes = state_nodes
    elif "linear_ssm" in spec.name:
        # Linear SSM builder (spike-gated)
        StateVar = spec.node_type("StateVar")
        InpNode = spec.node_type("InpNode")
        AEdge = spec.edge_type("AEdge")
        BEdge = spec.edge_type("BEdge")

        A = weights["A"]
        B = weights.get("B", None)
        n = A.shape[0]

        cdg = CDG()
        state_nodes = [StateVar() for _ in range(n)]
        for s_node in state_nodes:
            cdg.add_node(s_node)

        for i in range(n):
            for j in range(n):
                a = float(A[i, j])
                if abs(a) < 1e-12:
                    continue
                edge = AEdge(a=mgr.new_analog(init_val=a))
                cdg.connect(edge, state_nodes[j], state_nodes[i])

        inp_nodes = []
        if B is not None:
            m = B.shape[1]
            inp_nodes = [InpNode() for _ in range(m)]
            for inp_node in inp_nodes:
                cdg.add_node(inp_node)
            for i in range(n):
                for j in range(m):
                    b = float(B[i, j])
                    if abs(b) < 1e-12:
                        continue
                    edge = BEdge(b=mgr.new_analog(init_val=b))
                    cdg.connect(edge, inp_nodes[j], state_nodes[i])

        if readout_nodes is None:
            readout_nodes = state_nodes
    else:
        raise ValueError(f"Unknown spec family: {spec.name}")

    CktClass = OptCompiler().compile(
        prog_name=prog_name,
        cdg=cdg,
        cdg_spec=spec,
        trainable_mgr=mgr,
        readout_nodes=readout_nodes,
        normalize_weight=normalize_weight,
        do_clipping=do_clipping,
        aggregate_args_lines=aggregate_args_lines if mismatch_sigma > 0 else False,
        vectorize=vectorize,
    )

    return CktClass, mgr
