"""
paradigms.py — Reusable CDGSpec factories for neuro-analog dynamics.

Mismatch toggles between AttrDef and AttrDefMismatch on the same EdgeType
(not inheritance), which is the pattern OptCompiler expects.
"""

from __future__ import annotations

import numpy as np


def additive_recurrent(n_neurons: int, activation: str = "tanh", mismatch_sigma: float = 0.0):
    """
    CDGSpec for additive recurrent networks (Hopfield / EBM form).

    Dynamics:  dx_i/dt = -x_i + b_i + sum_j J_ij * phi(x_j)
    where phi = activation (tanh or sigmoid).

    Node types
      StateVar  (order=1, SUM)  — ODE state x_i, carries bias b_i
      OutUnit   (order=0, SUM)  — algebraic out_i = phi(x_i)

    Edge types
      MapEdge   (no attrs)      — wiring for ReadOut / SelfDecay
      FlowEdge  (attr: g)         — weight edges; g mismatched if sigma > 0

    Production rules
      ReadOut:    MapEdge  StateVar -> OutUnit   DST   DST.act(VAR(SRC))
      SelfDecay:  MapEdge  StateVar -> StateVar  SELF  -VAR(SELF) + SRC.b
      JWeight:    FlowEdge OutUnit  -> StateVar  DST   EDGE.g * VAR(SRC)

    Returns (spec, StateVar, OutUnit, MapEdge, FlowEdge)
    """
    from ark.reduction import SUM
    from ark.specification.attribute_def import AttrDef, AttrDefMismatch
    from ark.specification.attribute_type import AnalogAttr, FunctionAttr
    from ark.specification.cdg_types import EdgeType, NodeType
    from ark.specification.production_rule import ProdRule
    from ark.specification.rule_keyword import DST, EDGE, SELF, SRC, VAR
    from ark.specification.specification import CDGSpec

    spec = CDGSpec(name=f"additive_recurrent_{n_neurons}_{activation}")

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

    MapEdge = EdgeType(name="MapEdge")

    if mismatch_sigma > 0.0:
        FlowEdge = EdgeType(
            name="FlowEdge",
            attrs={
                "attr_def": {
                    "g": AttrDefMismatch(
                        attr_type=AnalogAttr((-50, 50)),
                        rstd=mismatch_sigma,
                    ),
                }
            },
        )
    else:
        FlowEdge = EdgeType(
            name="FlowEdge",
            attrs={
                "attr_def": {
                    "g": AttrDef(attr_type=AnalogAttr((-50, 50))),
                }
            },
        )

    spec.add_cdg_types([StateVar, OutUnit, MapEdge, FlowEdge])

    ReadOut = ProdRule(MapEdge, StateVar, OutUnit, DST, DST.act(VAR(SRC)))
    SelfDecay = ProdRule(MapEdge, StateVar, StateVar, SELF, -VAR(SELF) + SRC.b)
    JWeight = ProdRule(FlowEdge, OutUnit, StateVar, DST, EDGE.g * VAR(SRC))

    spec.add_production_rules([ReadOut, SelfDecay, JWeight])

    return spec


def deq_zform(n_state: int, mismatch_sigma: float = 0.0):
    """
    CDGSpec for DEQ relaxation: dz/dt = -z + tanh(W_z @ z + b_eff).

    This is the additive_recurrent form with effective bias b_eff = W_x @ x + b
    for a fixed input x. The compiled circuit converges to the DEQ fixed point.

    For a fixed input x, b_eff is baked into StateVar.b.
    """
    return additive_recurrent(n_state, activation="tanh", mismatch_sigma=mismatch_sigma)


def linear_ssm(d_state: int, mismatch_sigma: float = 0.0):
    """
    CDGSpec for purely linear state-space dynamics.

    OptCompiler is primarily validated on nonlinear specs; the linear path
    may fall back to a plain BaseAnalogCkt if compilation fails.

    Dynamics:  dh/dt = A @ h + B @ u   (real-only, diagonal A for demo)

    Node types
      StateVar  (order=1, SUM)  — state h_i
      InpNode   (order=0)       — external input u_j

    Edge types
      AEdge     (attr: a)         — state-to-state coupling
      BEdge     (attr: b)         — input-to-state coupling

    Production rules
      ARule:  AEdge  StateVar -> StateVar  DST  EDGE.a * VAR(SRC)
      BRule:  BEdge  InpNode  -> StateVar  DST  EDGE.b * VAR(SRC)
    """
    from ark.reduction import SUM
    from ark.specification.attribute_def import AttrDef, AttrDefMismatch
    from ark.specification.attribute_type import AnalogAttr
    from ark.specification.cdg_types import EdgeType, NodeType
    from ark.specification.production_rule import ProdRule
    from ark.specification.rule_keyword import DST, EDGE, VAR
    from ark.specification.specification import CDGSpec

    spec = CDGSpec(name=f"linear_ssm_{d_state}")

    StateVar = NodeType(
        name="StateVar",
        attrs={"order": 1, "reduction": SUM, "attr_def": {}},
    )
    InpNode = NodeType(name="InpNode", attrs={"order": 0, "attr_def": {}})

    if mismatch_sigma > 0.0:
        AEdge = EdgeType(
            name="AEdge",
            attrs={
                "attr_def": {
                    "a": AttrDefMismatch(
                        attr_type=AnalogAttr((-100, 100)), rstd=mismatch_sigma
                    ),
                }
            },
        )
        BEdge = EdgeType(
            name="BEdge",
            attrs={
                "attr_def": {
                    "b": AttrDefMismatch(
                        attr_type=AnalogAttr((-100, 100)), rstd=mismatch_sigma
                    ),
                }
            },
        )
    else:
        AEdge = EdgeType(
            name="AEdge",
            attrs={"attr_def": {"a": AttrDef(attr_type=AnalogAttr((-100, 100)))}},
        )
        BEdge = EdgeType(
            name="BEdge",
            attrs={"attr_def": {"b": AttrDef(attr_type=AnalogAttr((-100, 100)))}},
        )

    spec.add_cdg_types([StateVar, InpNode, AEdge, BEdge])

    ARule = ProdRule(AEdge, StateVar, StateVar, DST, EDGE.a * VAR(SRC))
    BRule = ProdRule(BEdge, InpNode, StateVar, DST, EDGE.b * VAR(SRC))

    spec.add_production_rules([ARule, BRule])
    return spec
