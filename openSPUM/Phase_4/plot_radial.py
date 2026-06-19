"""Plot VSPT radial probability distribution for hydrogen 1s state."""
import sys, os
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from Phase_4.vspt_electron import ElectronVSPTEngine, VSPTConfig, ElectronConfig
from Phase_4.nucleus_assembly import build_nucleus, NucleusType
from Phase_4.eternal_particle import EternalParticle, EternalParticleOrientation

nuc = build_nucleus(NucleusType.PROTON)

ecfg = ElectronConfig(electron_count=1, feedback_gamma=0.08, n_ref=600)
vcfg = VSPTConfig(max_layers=10, max_nodes_per_tree=500, seed=42)
eng = ElectronVSPTEngine(config=vcfg, electron_cfg=ecfg)
g = eng.run_growth(nuc.outward_solid, "H", nuc.particles, nuc.positions)

R_nuc = 1.0
layers = sorted(g['layer_distribution'].keys())
radii = [R_nuc + l * vcfg.layer_spacing for l in layers]
counts = [g['layer_distribution'][l] for l in layers]
probs = [g['electron_density'][l] for l in layers]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: node counts
ax1.bar(radii, counts, width=0.6, alpha=0.7, color='steelblue', edgecolor='navy')
ax1.axvline(x=radii[probs.index(max(probs))], color='red', ls='--', lw=2, label=f"Peak r={radii[probs.index(max(probs))]:.1f}")
ax1.set_xlabel("Radius (nuclear radius units)", fontsize=12)
ax1.set_ylabel("VSPT node count N(r)", fontsize=12)
ax1.set_title("VSPT Radial Distribution (H 1s)", fontsize=13)
ax1.legend()
ax1.grid(alpha=0.3)

# Right: probability
ax2.bar(radii, probs, width=0.6, alpha=0.7, color='coral', edgecolor='darkred')
ax2.axvline(x=radii[probs.index(max(probs))], color='red', ls='--', lw=2, label=f"Peak r={radii[probs.index(max(probs))]:.1f}")
ax2.set_xlabel("Radius (nuclear radius units)", fontsize=12)
ax2.set_ylabel("Electron probability p(r)", fontsize=12)
ax2.set_title("Electron Radial Probability", fontsize=13)
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vspt_radial.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
print(f"E_bind = {g['binding_energy_eV']:.2f} eV")
print(f"Peak at layer {g['electron_peak_layer']}, r = {radii[probs.index(max(probs))]:.2f}")
print(f"N distribution: {dict(zip(layers, counts))}")
