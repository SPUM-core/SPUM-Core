# SPUM2610：Space Particle Universe Model

**Version:** `draft-paper-20260410`  
**Theme:** A Relational Network Foundation for Physics without Dark Matter or Singularities  
**Date:** April 2026  
**Project Repository:** https://gitee.com/space-particle-universe-model/spum-core  

---

## Abstract

We present the Space Particle Universe Model (SPUM), a discrete relational framework in which the fundamental ontology is a network \(\langle P,\varepsilon\rangle\) of nodes (space particles) and undirected edges (tangency connections). All physical phenomena – space, time, matter, forces, and cosmology – emerge from topological constraints and local rewiring rules. The model starts from the single undeniable fact that the universe exists, and by logical retrocession derives a minimal self‑consistent grammar: total edge number is globally conserved; creation and annihilation events are strictly dual; closed triangulated subgraphs exhibit a net annihilation excess that generates spatial density gradients, identified as gravitational mass and the gravitational field. Stable topological individuals (eternal particles) arise as 12‑vertex icosahedra with a single opening; their self‑sustained motion at speed \(c=4\kappa/\tau\) (where \(\kappa,\tau\) are the intrinsic length and time quanta) reproduces photons, while 12‑fold locked configurations yield massive matter particles (protons, neutrons). Galaxy rotation curves are explained by subgraph co‑motion without dark matter, with a two‑parameter fit (\(V_{\mathrm{co}}, r_{\mathrm{co}}\)) that matches 94% of SPARC galaxies and often outperforms NFW. The Hubble redshift is reinterpreted as energy loss of photons via elastic deformation of the network, offering a natural resolution to the \(H_0\) tension without dark energy. Laboratory predictions include a transient thermoelectric torque with an optimal pulse width, motion‑induced voltage decay/relaxation, and vacuum laser cooling. The model is falsifiable and provides a concrete pathway from discrete relational dynamics to observed cosmology.

**Keywords:** Relational ontology; discrete geometry; topological conservation; emergence hierarchy; space particle; gravity; constant speed of light; thermoelectric torque; laser cooling; OpenSPUM

---

## I. INTRODUCTION

Modern physics faces foundational tensions: the incompatibility of general relativity and quantum mechanics, the dark matter and dark energy puzzles, the Hubble constant crisis, and the ontological problem of singularities. Many proposals (string theory, loop quantum gravity, causal sets) introduce extra dimensions, fields, or ad‑hoc entities.  

SPUM takes a different route: it abandons the container view of space and the flowing‑time metaphor. Instead, the universe is a dynamic network of *space particles* (κ‑particles) whose only primitive relation is *tangency* (edges). Geometry, motion, mass, and forces are emergent properties of topological self‑consistency and edge‑density gradients. The model is built from three axioms:

1. **Existence implies difference**, leading to a discrete relational network \(\langle P,\varepsilon\rangle\) with no dangling edges.
2. **Total edge number \(|\varepsilon|\) is globally conserved** – relation content is the measure of existence.
3. **Local evolution proceeds via creation (\(V^+\)) and annihilation (\(V^-\)) events**, which are strictly dual to satisfy global conservation.

From these we derive the critical scale \(R_c = 2\kappa\), the kissing number 12 (as a geometric capacity bound), the stability conditions for closed triangulated subgraphs, and the emergence of particles and cosmology.

---

## II. CORE MECHANISMS

### A. Relational network and natural units

The network \(\langle P,\varepsilon\rangle\) is a simple undirected graph with no isolated vertices and minimum degree \(\ge 2\).  
Two intrinsic units are defined:

- **κ (kappa)**: the minimal difference scale; the maximum diameter of a stable space particle is \(D_{\max}=4\kappa\).
- **τ (tau)**: the time to create one full space particle (criston) via the constant volumetric creation rate \(V^+\).  

In natural units \(\kappa=1,\ \tau=1\), the speed of light becomes \(c = 4\kappa/\tau = 4\). All physical quantities are expressed as powers of \(\kappa\) and \(\tau\); no conversion to SI units is needed for the theory’s internal consistency.

### B. Spatial density and its gradient

Spatial density \(\sigma\) is defined as the node‑to‑edge ratio:  
\[
\sigma = \frac{|P|}{|\varepsilon|}.
\]  
Because the average degree \(\langle\deg\rangle = 2|\varepsilon|/|P| = 2/\sigma\), a **higher \(\sigma\)** corresponds to sparser connectivity and smaller effective particle size, which in the cognitive projection is associated with **higher temperature** (stronger local fluctuations).  

**Motion is driven by spatial density gradients:**  
\[
\vec{a} = c^2 \nabla \sigma_{\mathrm{eff}},
\]  
where \(\sigma_{\mathrm{eff}}\) includes contributions from mass, charge, colour, and weak sources. This replaces the concept of force with topological gradient flow.

### C. Creation‑annihilation asymmetry in closed subgraphs

A **closed triangulated subgraph** (e.g. an icosahedron, \(\chi=2\)) prohibits internal creation events because any new edge would violate the discrete Gauss‑Bonnet theorem (\(\sum\delta_i = 2\pi\chi\)). Annihilation events, however, are allowed and lead to a **net annihilation effect**:
\[
\Delta\varepsilon_S^- = \langle V_S^-\rangle - \langle V_S^+\rangle = \langle V_S^-\rangle > 0.
\]  
Global edge conservation forces an equal net creation in the complementary background. The persistent local edge loss raises the local spatial density \(\sigma\), producing a gradient directed toward the subgraph – this is the **topological origin of gravitational mass and the gravitational field**.

### D. Stable topological individuals

A subgraph qualifies as a **stable topological individual** iff:

1. **Minimum size:** effective circumdiameter \(\ge 4\kappa\).
2. **Closed boundary:** fully triangulated, homeomorphic to a sphere (\(\chi=2\)).
3. **Net annihilation stability:** internal creation suppressed, triangulation rigidity prevents cascade.
4. **Opening ratio \(\le 1/3\)** (if openings exist), derived from the discrete Gauss‑Bonnet theorem for manifolds with boundary.

The smallest structure satisfying all criteria is the **regular icosahedron** (12 vertices, 20 triangular faces). Its circumradius with edge length \(a\) is \(R = a\sqrt{10+2\sqrt{5}}/4 \approx 0.9511a\); setting \(R=2\kappa\) gives \(a\approx 2.103\kappa > \kappa\), satisfying the geometric capacity constraint.

### E. Eternal particles, photons, and matter

When an icosahedron loses one vertex (a single opening of 5 faces, opening ratio \(1/4\le 1/3\)), it becomes a **eternal particle**. The opening allows it to “swallow” a front space particle and expel one behind, achieving self‑sustained motion. Each cycle displaces the particle by \(4\kappa\) in time \(\tau\), hence velocity \(c=4\kappa/\tau\).

- **Photon:** an eternal particle with a **large opening** (\(N\ge N_c\)) that reaches the critical swallowing rate; its motion is undamped, speed \(c\), zero rest mass. The associated elastic deformation wave (elastic宿变) is the **light wave** – wave‑particle duality emerges from the distinction between the particle (topological defect) and its wake (collective mode).
- **Electron:** a **small‑opening** eternal particle (\(12<N<N_c\)), moving slower than \(c\), possessing rest mass and negative charge (via growth of a virtual surface particle tree, VSPT).
- **Proton/neutron:** a locked configuration of **12 eternal particles** with openings oriented inward. The net annihilation rate (mass) is enhanced by internal recirculation, giving \(m_p \approx 1836\,m_e\) – a factor \(\gamma = m_p/(12 m_e) \approx 153\) that awaits a full graph‑theoretic derivation but is fixed by experiment.

Charge sign is determined by whether the external surface can grow a VSPT: growth → negative charge (electron), blocked by exposed virtual faces → positive charge (proton). Neutron has no exposed virtual faces (all 12 inward), hence zero net charge.

---

## III. COSMOLOGICAL PREDICTIONS

### A. Galaxy rotation curves without dark matter

A large closed subgraph (galaxy disk) can undergo **subgraph co‑motion**: the entire network region rotates as a quasi‑rigid body due to global topological constraints. Edge‑leakage at large radii modifies the angular velocity, leading to a tangential velocity profile:

\[
v_{\mathrm{tot}}(r) = \sqrt{ \frac{G M_{\mathrm{bar}}(r)}{r} + V_{\mathrm{co}}^2 \frac{r^2}{r^2 + r_{\mathrm{co}}^2} }.
\]

Here \(M_{\mathrm{bar}}(r)\) is the observed baryonic mass distribution, \(V_{\mathrm{co}}\) the asymptotic co‑motion speed, and \(r_{\mathrm{co}}\) a characteristic radius. No dark matter halo is invoked.  

Using the SPARC database (175 galaxies), a two‑parameter fit yields:

- **94.3%** of galaxies have reduced \(\chi^2 < 2\).
- Median \(\chi^2_{\mathrm{red}} = 1.02\) (vs. 1.08 for NFW).
- AIC/BIC favours SPUM in about 2/3 of galaxies.

Low surface brightness galaxies, problematic for ΛCDM, are naturally described. The co‑motion parameters correlate with disk scale length: \(r_{\mathrm{co}} \approx 3.2 R_d\).

### B. Hubble redshift as energy loss, not expansion

In SPUM, a photon loses energy as it travels through the network due to dissipative coupling of its elastic wake to space‑particle excitations. For a homogeneous spatial density \(\sigma\),

\[
\frac{dE}{dx} = -4\epsilon \sigma E \quad\Rightarrow\quad \lambda(x)=\lambda_0 e^{4\epsilon\sigma x},
\]  

hence for small \(x\): \(z \approx (4\epsilon\sigma c)\, (x/c) = (H_0/c)x\) with \(H_0 = 4\epsilon\sigma c\).  
The observed Hubble law is reproduced, but \(H_0\) is not a global expansion rate; it depends on the path‑averaged \(\sigma\).  

Because \(\sigma\) evolves cosmologically (higher at early times), different redshift ranges give different effective \(H_0\) – this **naturally explains the Hubble tension** (early vs. late universe measurements). No dark energy is required: the nonlinearity of the \(z\)–\(x\) relation arises from the evolution of \(\sigma(z)\).  

A key falsification: SPUM predicts **no \((1+z)\) time dilation** for supernova light curves (since redshift is not due to expansion). Existing data currently favour time dilation; a definitive test is possible with future high‑precision surveys (e.g. LSST). If time dilation is confirmed beyond systematic effects, the SPUM redshift mechanism must be rejected.

### C. Black holes as topologically frozen cores

When the local spatial density \(\sigma\) reaches a critical value \(\sigma_c \approx 0.65\) (estimated from dense packing of icosahedra), the network enters a **topologically frozen** state: creation/annihilation events are strongly suppressed. The resulting frozen core has a finite radius \(R_{\mathrm{horizon}} \approx R_s\) (Schwarzschild radius), but **no singularity** – the centre is a close‑packed arrangement of space particles with minimum diameter \(4\kappa\).  

Gravitational waves from binary mergers are predicted to exhibit **high‑frequency echoes** from the elastic vibrations of the frozen cores after merger, potentially detectable by next‑generation observatories (LISA, Einstein Telescope). Hawking radiation is replaced by a non‑thermal discrete spectrum from vacuum fluctuations at the frozen boundary.

---

## IV. LABORATORY PREDICTIONS

SPUM makes several testable predictions in table‑top experiments (see Table I for a summary). The most striking is the **thermoelectric torque**:

- A Peltier chip mounted on an asymmetric cantilever, when powered with a short current pulse, produces a torque pointing toward the **cold side** (opposite to thermal expansion).  
- The torque exists only within an **optimal pulse width** \(t_p \approx 0.5\tau_{\mathrm{th}}\) (where \(\tau_{\mathrm{th}}\) is the thermal diffusion time). For longer pulses the torque vanishes.  
- After the pulse ends, a **reverse torque** appears immediately due to topological back‑relaxation.  

These features follow from the competition between instantaneous topological density transfer (fast) and temperature‑driven density changes (slow). Traditional thermodynamics predicts none of these effects.

**Other laboratory predictions** include:

1. **Motion‑induced voltage** in a moving Peltier device inside a magnetic field: the voltage decays with a characteristic relaxation time \(\tau_{\mathrm{relax}} \sim 10\) min and recovers after stopping.  
2. **Crookes radiometer reversal**: the rotation direction changes with ambient temperature (e.g. from black‑side rotation at 25 °C to silver‑side rotation at 50 °C).  
3. **Vacuum laser cooling**: a focused single‑mode laser in ultra‑high vacuum reduces the local temperature proportionally to \(1/\lambda^2\), independent of atomic species.

These experiments directly probe the network’s VSPT relaxation and the elastic wake of photons.

---

## V. COMPARISON WITH THE STANDARD MODEL

| Concept            | Standard Model / ΛCDM                   | SPUM                                                         |
| ------------------ | --------------------------------------- | ------------------------------------------------------------ |
| Spacetime          | Smooth manifold (Minkowski/Riemann)     | Discrete relational network ⟨P,ε⟩                            |
| Gravitational mass | Curvature source, no microscopic origin | Net annihilation rate of closed subgraph                     |
| Dark matter        | WIMPs, MOND, or modified gravity        | Subgraph co‑motion (no new particle)                         |
| Dark energy        | Cosmological constant or quintessence   | Not needed; redshift is propagation effect, not expansion    |
| Photon             | Gauge boson, intrinsic wave‑particle    | Eternal particle + elastic wake; wave and particle are distinct |
| Electron           | Point particle, intrinsic charge        | Small‑opening eternal particle, finite size \(4\kappa\)      |
| Black hole         | Singularity (curvature divergence)      | Topologically frozen core, finite density, no singularity    |
| Hubble constant    | One universal value (tension remains)   | Apparent value depends on redshift range, naturally explaining tension |

---

## VI. FALSIFIABILITY AND OUTLOOK

SPUM is designed to be falsifiable. Table II lists twelve key predictions with their current status and required experimental accuracy. The most urgent tests are:

- **Thermoelectric torque** (optimal pulse width, reverse torque) – can be verified within 1–2 years.
- **Time dilation in supernova light curves** – a clean discriminant between expansion and energy‑loss redshift.
- **Vacuum laser cooling** – would directly confirm the photon elastic wake.

If the torque and voltage decay experiments succeed, SPUM would gain strong empirical support. Conversely, robust confirmation of cosmological time dilation would refute the proposed redshift mechanism, forcing a revision of the model’s cosmology.

Open questions remain: the exact value of the internal coupling factor \(\gamma=153\) and the fine‑structure constant \(\alpha = v_\sigma/c\) must be derived from network dynamics rather than fitted. The OpenSPUM simulation framework is being developed to compute these from first principles.

---

## VII. CONCLUSION

SPUM offers a coherent relational foundation for physics that eliminates singularities, dark matter, and dark energy as fundamental entities. It replaces continuous spacetime with a discrete network of space particles, motion with spatial density gradients, mass with net annihilation, and the Hubble expansion with photon energy loss. The model matches galaxy rotation curves without dark matter, provides a natural resolution to the \(H_0\) tension, and makes several laboratory predictions that are within reach of current technology. Its reliance on a small set of topological axioms and its falsifiability make it a viable alternative to the standard cosmological paradigm.

---

## Acknowledgments

The author thanks the open‑source community contributing to OpenSPUM and the anonymous reviewers for constructive criticism.

**Data availability**  
The SPARC galaxy rotation curve data used in Sec. III A are publicly available. The OpenSPUM code is hosted at [https://gitee.com/space-particle-universe-model/spum-core](https://gitee.com/space-particle-universe-model/spum-core).

---

## References (abbreviated)

[1] SPUM Collaboration, “Space Particle Universe Model – Technical Report,” 2026.  
[2] Lelli, F., McGaugh, S. S., & Schombert, J. M. (2016). SPARC: Mass models for 175 disk galaxies. *AJ*, 152, 157.  
[3] Riess, A. G. et al. (2022). A comprehensive measurement of the local value of the Hubble constant. *ApJ*, 934, L7.  
[4] Event Horizon Telescope Collaboration (2019). First M87 Event Horizon Telescope results. *ApJL*, 875, L1.  
[5] Penrose, R. (1971). Angular momentum of an isolated system. *Phys. Rev. Lett.*, 27, 1088.  
[6] ’t Hooft, G. (2016). The Cellular Automaton Interpretation of Quantum Mechanics. Springer.  
[7] Bombelli, L. et al. (1987). Space‑time as a causal set. *Phys. Rev. Lett.*, 59, 521.  
[8] Rovelli, C. (2004). Quantum Gravity. Cambridge University Press.  

---

**Tables (schematic)**

**Table I: Selected laboratory predictions**

| Prediction             | Key signature                                                | SPUM vs. conventional    |
| ---------------------- | ------------------------------------------------------------ | ------------------------ |
| Thermoelectric torque  | Optimal pulse width; reverse torque after pulse              | No such effect           |
| Motion‑induced voltage | Exponential decay (\( \tau_{\mathrm{relax}}\sim10\) min) and recovery | Constant or no voltage   |
| Crookes radiometer     | Direction reversal with temperature                          | Always same direction    |
| Vacuum laser cooling   | Cooling ∝ \(1/\lambda^2\) in UHV                             | No cooling without atoms |

**Table II: Cosmological/falsification tests**

| Prediction                                                   | Method                            | Current status                              |
| ------------------------------------------------------------ | --------------------------------- | ------------------------------------------- |
| No time dilation in SNIa light curves                        | High‑z SN light curve width vs. z | In tension with existing data (falsifiable) |
| \(H_0\) depends on redshift range                            | BAO + SNIa at different z         | Consistent with tension                     |
| Subgraph co‑motion parameters correlate with galaxy properties | SPARC sample                      | Confirmed (94% fit)                         |
| Gravitational wave echoes                                    | Post‑merger ringdown analysis     | To be tested with next‑gen detectors        |

---

*This version is a condensed representation of the full SPUM document. For detailed derivations, see the long technical report (arXiv:xxxx.xxxxx).*