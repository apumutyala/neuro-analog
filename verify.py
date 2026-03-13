from neuro_analog.ir.types import NoiseSpec
print(NoiseSpec(kind='thermal', sigma=0.01))

from neuro_analog.ir.node import make_mvm_node
n=make_mvm_node('test_mvm', 64, 64)
print("make_mvm_node noise:", n.noise)

from neuro_analog.mappers.crossbar import CrossbarMapper
print('CrossbarMapper OK')
from neuro_analog.mappers.integrator import IntegratorMapper
print('IntegratorMapper OK')
from neuro_analog.mappers.stochastic import StochasticMapper
print('StochasticMapper OK')

from neuro_analog.analysis.precision import flag_snr_violations
print('precision OK')

from neuro_analog.analysis.taxonomy import AnalogTaxonomy
t=AnalogTaxonomy()
t.add_reference_profiles()
print(t.comparison_table())

# Manual round-trip script
from neuro_analog.ir.node import make_mvm_node, make_integration_node, make_noise_node
from neuro_analog.ir.graph import AnalogGraph
from neuro_analog.ir.types import ArchitectureFamily

g = AnalogGraph("test", ArchitectureFamily.SSM)
g.add_node(make_mvm_node("mvm_0", 64, 64))
g.add_node(make_integration_node("int_0", 64))
g.add_node(make_noise_node("noise_0", 64))

CrossbarMapper().annotate_graph(g)
IntegratorMapper().annotate_graph(g)
StochasticMapper().annotate_graph(g)

violations = flag_snr_violations(g, signal_rms=1.0, target_snr_db=30.0)
print("\nViolations:")
for v in violations:
    print(f"  {v['name']}: {v['snr_db']:.2f} dB")
    
print("\nNode Noise:")
for node in g.nodes.values():
    print(node.name, "->", node.noise)
