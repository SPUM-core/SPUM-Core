# OpenSPUM Full-layer Regression Baseline

> **Snapshot Date**: 2026-09-14 ｜ **HEAD**: `514ffeb`
> **Purpose**: After any subsequent incremental implementation, reproduce this document's "reproduction commands" one by one. The readings must be **bit-for-bit identical** to the table below (tolerance see per-layer notes). **Inconsistency means regression**, which requires locating the issue before proceeding.
> **Scope**: All readings are **measured** (not expected values); only read projection layers (L1–L4), do not write back to L0.

---

## 1. Environment

| Item | Value |
|---|---|
| Python | `C:\Users\macotai\python-sdk\python3.13.2\python.exe` (3.13.2, portable, not in PATH) |
| numpy / scipy | 2.5.3 / 1.18.1 |
| cupy | 14.2.0 (**no CUDA path** ⇒ `l0_gpu` falls back to CPU; this baseline is **pure CPU**) |
| Encoding | Terminal must have `$env:PYTHONIOENCODING="utf-8"` (to avoid Chinese output corruption) |

---

## 2. Verdict Summary

| # | Entry | Command (relative to `openSPUM/src/`) | Verdict |
|---|---|---|---|
| 1 | L0 prototype | `l0/combinatorial_proto.py selftest=1` | **All passed** |
| 2 | L0 frame evolution | `l0/l0_core.py selftest=1` | **All passed** |
| 3 | L0 GPU decision | `l0/l0_gpu.py selftest=1` | **All passed** |
| 4 | L0 locality assertion | `l0/l0_locality.py selftest=1` | **All passed** |
| 5 | L0 count projection Π_c | `l0/l0_count.py selftest=1` | **All passed** (forward reading + dual backward control) |
| 6 | L1 local projection | `l1/l1_projection.py selftest=1` | **All passed** |
| 7 | L2 global projection | `l2/l2_projection.py selftest=1` | **All passed** (fixture A–H) |
| 8 | L3 frame sequence | `l3/l3_projection.py selftest=1` | **All passed** |
| 9 | L4 evolution observation | `l4/l4_evolution.py selftest=1` | **All passed** |
| 10 | Root test suite | `python -m unittest discover -s tests` (warehouse root) | **Ran 207 tests / OK** |
| 11 | L2 radial spectrum CLI | `l2/l2_projection.py spectrum=1 seed=icosa nframes=8 nbins=8` | adoption trajectory t=2..8 **all `neither`** |
| 12 | L2 finite window CLI | `l2/l2_projection.py window=1 seed=icosa nframes=4 sizes=8,16,24,32` | t=3 `stable`, others `unstable` |
| 13 | L2 existing regression CLI | `l2/l2_projection.py seed=icosa nframes=4` | consistent with frozen values in §7 "Phase quartic term landing" |

> ⚠️ `l0_gpu.py` will print `UserWarning: CUDA path could not be detected...` — **this is a CuPy loading hint, not a failure**. To judge as FAIL, it must match "there are failed items", do not use bare `FAIL` string (it may be mistakenly matched by `CuPy fails to load`).

---

## 3. Frozen Readings at Each Layer

### L0-① Prototype `combinatorial_proto.py`

```
tetra  (V,E,F,χ)=(4, 6, 4, 2)      expected (4, 6, 4, 2)   S=12 d=0 link(cyc/chord/disc)=4/0/0
icosa  (V,E,F,χ)=(12, 30, 20, 2)   expected (12, 30, 20, 2) S=12 d=0 all deg5 Σ(6−deg)=12 crystallite 12=1
patch  V=19 E=42 F=25 χ=2 d=9 holes=1(max12) open=0.480
flip_edge: 30/30 edges can be flipped; first (0,2)->(6,3) ΔΨ=4; flip twice to restore (rotation-system equivalence)=True
```

### L0-② Frame Evolution `l0_core.py`

```
icosa 4 frames   V=680 E=2034 F=1356 χ=2 triangles=1356 holes=0 separated=668 dangling=0 maxdeg=64
tetra 4 frames   V=164 E=486  F=324  χ=2 triangles=324  holes=0 separated=160 dangling=0 maxdeg=48
Deterministic=True (two runs frame-by-frame identical)｜Order-Independent=True (candidate reverse order consistent)
Adopted dynamics dense ten frames V/Vprev=[1.125, 1.195, 1.242, 1.292, 1.468, 1.571, 1.72, 1.833, 1.912, 2.667] (non-constant ⇒ off-track)｜ten frames later V=960
L0-A9  : baseline dangling four frames V⁻=0 (sleeping); adopted dense ten frames V⁻=5 (activated ⇒ dangling-end regeneration)
L0-A11 : icosa⊔tetra cap=5 21 frames n_comp 2→2 (min/max=2/2, frame-by-frame not increasing=True)※ limited to default dmin=3
Switch pairing: conserved first frame creation 3 / ΔE=9, cut edges 9 (must ==ΔE); none creation 20 / cut edges 0
Platform shutdown D: V=16 Ψ=12 (lower bound 12) reason=cycle cycle_len=2 best_t=4
Platform shutdown E: V=42 Ψ=12 (lower bound 12) reason=cycle period=(54, 56) best_t=9
```

### L0-③ GPU Decision `l0_gpu.py`

```
Four increments (decision segment / commit segment / write operation / adopted dynamics) + end-to-end frame_gpu all GPU≡CPU bit-for-bit identical
End-to-end anchor: icosa cap=12 dmin=3 slack/none punc=0 4 frames V=40 E=114 hash=87de63c3568be8c7
End-to-end anchor: tetra cap=64 dmin=3 slack/none punc=0 4 frames V=164 E=486 hash=9b8486986f9ea557
X1 guardian  : cap=12 10 frames maxdeg≤cap (CPU∧GPU)=True｜GPU≡CPU=True｜V=960
           cap=64  5 frames maxdeg≤cap (CPU∧GPU)=True｜GPU≡CPU=True｜V=1550
```

> `state_hash` **does not contain `nid`** — bit-for-bit identical requires separate verification of rotation-system dict and event count (see work mode document §4).

### L0-④ Locality Assertion `l0_locality.py` (Stage 6 Item ④, Block B)

```
① Single-component icosa: 6 items, all non-exempt passed, 1 exempt (face_rep, report the measured value as is)
② Reverse control: cross-component measurement inf ⇒ rejected (criterion not always true)
③ t=1..3 (V=66): dart_next measured [1, 2, 1] (bound 2, actually reached); reverse_dart/face_vertices always {1}
480 audits (seed{icosa,tetra,patch} × cap{12,16} × dmin{2..5} × vminus × vplus × t0..4) zero failures;
each non-exempt item measured maximum exactly equals registered bound: rot_idx=1 / owner_of_dart=0 / reverse_dart=1 / dart_next=2
```

### L0-⑤ Count Projection `l0_count.py` (Number Theory Branch, 2026-09-14)

```
① Preset gating: closure + no holes(tri_only) + uniform saturation C=5 ⇒ reading V=12 / E=30 / F=20; n_shells=1 (5-regular connected component)
② Reverse control a: seed=patch's χ is also 2 (outer boundary is a 12-gon **face**)
   ⇒ closed_chi2 holds, reading 30; overlay tri_only ⇒ None (blocked by tri_only) — [χ=2 ≠ no boundary]
③ Reverse control b: icosa + uniform_C=4 ⇒ None (blocked by uniform_C=4)
④ No preset: read as is, preset=None (no claim of validity under preset)
⑤ forced_N(C) (divisibility strong theorem N(6−C)=12 ⟺ (6−C)|12, C≥3): C=3,4,5 → 4,6,12; C∈{0,1,2,6,7,8} → None
```

> **Reading convention**: `value=None` indicates **preset not valid** (`kind="undefined"`), **not** 0 — "12" only holds under the declared preset.
> **Frozen new fact**: `χ=2` alone **does not** imply "no boundary"; C1's object condition must be `χ=2 ∧ no holes` (`l0_count` reverse control ② provides).

### L1 Local Projection `l1_projection.py`

```
tetra : A_v=π, defect=π, Σ=4π
icosa : A_v=5π/3, defect=π/3, Σ=4π, ∇σ≡0
icosa local frame: r(5)=3.784699, 5 directions in rotation-system ✓; skeleton 30 edges ✓
L0 adoption trajectory: t=0 saddle point 0 → t=1 saddle point 12 (max A_v/2π=1.3820, curvature reading)
```

### L2 Global Projection `l2_projection.py`

```
icosa t=0 alignment propagation: 11 tree edges all exact = 2r(5)=7.569398
holonomy: 19 edges mean=11.9701 (= curvature reading, not error)
Refined 60 rounds: edge residuals mean=0.1339 max=0.3864 (itemconvention: not pursuing precision)
Equi-degree graph: σ≡0.4, no anchor points, redshift≡0; t=1: anchor points 20/12 sink/source, tree edges still exact
```

### L3 Frame Sequence Projection `l3_projection.py`

```
id alignment: t=0→t=1 persistent 12 / new 20 / vanished 0 (V 12→32→36)
δ reading: δ_2 ≡ 0 (dangling-end pruning deleted deg<dmin at frame end), δ_5 = 0, δ_6 = 1.0 (all positive angle defect ends)
Closure criterion: δ all 0 while Δμ=9 ⇒ not closed (N013 "δ insensitive to anchor drift"structural reproduction)
Export: CSV 21 columns + JSON (rows/closure/trace) round-trip consistent; two runs frame-by-frame identical
```

### L4 Evolution Observation `l4_evolution.py`

```
Net annihilation portrait (icosa cap=12 dense/any 10 frames): rho_core=+1.000 (net annihilation) rho_bg=-1.000 rho_sh=-1.000
Gate reading: viol_sat ≡ 0 (10 frames); core net ΔE=-532 background net ΔE=+130
Convention two: V⁻sh ≡ V⁻bg ≡ 0 (rule enforced); cap=64 V⁺core=96>0 ⇒ core net annihilation not identity
Identity: ΔE = Σ_new_cells deg − (Σ_cut_edges + deleted points) frame-by-frame holds
Structural closure: n_comp≡1 / boundary_edges≡0 / χ≡2 (throughout) ⇒ component-level degeneracy (merged into core/bg partition)
Closure identity: Σ(6−deg)=12+2·holes frame-by-frame holds (t=10: 446 = 12+2×217)
Imperfection Theorem: edge arm δ_dmin only t=9 non-zero (=1/558, masked by same-frame creation), δ_κ∈[0.5102,0.8889] always non-zero
```

---

## 4. L2 CLI Frozen Readings (Anchor Point Comparison)

### 4.1 Radial Power Spectrum `spectrum=1 seed=icosa nframes=8 nbins=8`

> Verdict criterion: `slope∇ΔN = a` power, −2 ⇒ `newton` (authority), −3 ⇒ `trap` (deprecated path), tolerance band ±0.35.

| t | V | E | Bin Count | slopeσ | slopeΔN | slope∇σ | slope∇ΔN | Verdict |
|---|---|---|---|---|---|---|---|---|
| 0 | 12 | 30 | 3 | -- | -- | -- | -- | undefined |
| 1 | 32 | 90 | 2 | -- | -- | -- | -- | undefined |
| 2 | 36 | 72 | 8 | 0.1281 | −1.1360 | −1.5831 | −0.6455 | neither |
| 3 | 66 | 192 | 8 | −0.1906 | −1.4204 | 0.6787 | −0.7781 | neither |
| 4 | 82 | 240 | 8 | 0.0202 | −0.8939 | 2.7294 | 0.0726 | neither |
| 5 | 98 | 264 | 8 | 0.2063 | −0.2095 | 0.9429 | 1.5090 | neither |
| 6 | 154 | 452 | 8 | 0.1535 | −0.4123 | 2.0172 | 0.9511 | neither |
| 7 | 226 | 568 | 8 | 0.0830 | −0.5007 | 1.3679 | 0.6581 | neither |
| 8 | 432 | 1243 | 8 | 0.0876 | −1.0507 | 0.7491 | 0.0796 | neither |

Calibration (same run): `σ∝r⁻²` ⇒ four `−2.000/−1.121/−3.000/−2.000`; `σ∝r⁻³` ⇒ `−3.000/−2.029/−4.000/−3.000`.
Null-hypothesis control (equal-degree graph icosa t=0): σ range `0.00e+00`, anchor 0, `z_max=0.00e+00` ⇒ no signal across all paths.

> **Verdict (unchanged)**: This round **did not read gravitational power signals on the coordinate artifact**—neither a falsification (L2 is a projection artifact) nor a reading defect (calibration + null hypothesis prove reading validity). **Open item, not closed by data**.

### 4.2 Finite Window `window=1 seed=icosa nframes=4 sizes=8,16,24,32`

> Verdict only considers `σ_window` final relative residual `rel_last ≤ tol_rel(0.05)`; `slope_σ`/`anchor_density` are only for reference.

| t | V | E | rel_last | Trend | Verdict |
|---|---|---|---|---|---|
| 0 | 12 | 30 | 0.3333 | converging | unstable |
| 1 | 32 | 90 | 0.0547 | converging | unstable |
| 2 | 36 | 72 | 0.1346 | non-monotone | unstable |
| 3 | 66 | 192 | **0.0042** | converging | **stable** |
| 4 | 82 | 240 | 0.0921 | converging | unstable |

> **Verdict (unchanged)**: Window stability is **conditional**—holds when `V_window ≪ V` (t=3: 32 vs 66); boundary effects dominate when V and window are comparable (`frac_boundary` as high as 0.75–1.0). **Cannot unconditionally claim "window reading stable"**.

### 4.3 Existing Regression `seed=icosa nframes=4`

| t | V | E | σ=V/E | σmax | Anchor± | z_max | Tree Residual | Edge Residual | Holonomy | Overlap |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 12 | 30 | 0.4000 | 0.4000 | 0 | 0.0000 | 8.88e-16 | 0.1339/0.3864 | 11.9701/26.1846 | 6 |
| 1 | 32 | 90 | 0.3556 | 0.6667 | 32 | 2.3333 | 5.33e-15 | 0.0000/0.0000 | 8.8194/34.8658 | 0 |
| 2 | 36 | 72 | 0.5000 | 0.6667 | 36 | 2.8571 | 1.78e-15 | 0.6398/2.7186 | 10.4416/33.9909 | 49 |
| 3 | 66 | 192 | 0.3438 | 0.5000 | 66 | 3.8333 | 3.55e-15 | 0.5503/2.8123 | 15.6942/72.5876 | 39 |
| 4 | 82 | 240 | 0.3417 | 0.6667 | 80 | 4.8333 | 5.33e-15 | 0.5307/2.4970 | 16.5335/79.5305 | 55 |

---

## 5. Reproduction commands

```powershell
$py = "C:\Users\macotai\python-sdk\python3.13.2\python.exe"
$env:PYTHONIOENCODING = "utf-8"
$root = "openSPUM\src"

# 9 layers selftest (layer by layer)
foreach ($p in @("l0\combinatorial_proto","l0\l0_core","l0\l0_gpu","l0\l0_locality","l0\l0_count",
                 "l1\l1_projection","l2\l2_projection","l3\l3_projection","l4\l4_evolution")) {
  & $py "$root\$p.py" selftest=1
}

# Root test suite (executed at the warehouse root)
& $py -m unittest discover -s tests

# Count projection operator (number theory branch)
& $py "$root\l0\l0_count.py" seed=icosa obj=V closed=1 tri=1 uniform=5   # → 12, readout
& $py "$root\l0\l0_count.py" seed=patch obj=sum6deg closed=1 tri=1       # → None, undefined
& $py "$root\l0\l0_count.py" table=1                                     # → C=3,4,5 → 4,6,12

# L2 three CLI regressions
& $py "$root\l2\l2_projection.py" spectrum=1 seed=icosa nframes=8 nbins=8
& $py "$root\l2\l2_projection.py" window=1   seed=icosa nframes=4 sizes=8,16,24,32
& $py "$root\l2\l2_projection.py" seed=icosa nframes=4
```

---

## 6. Revalidation Discipline

1. **Inconsistency triggers regression**: If any layer selftest changes from "all passed" to failure, or numerical drift occurs in this file, first locate the issue before proceeding—**do not** adjust the baseline to accommodate the code.
2. **Randomness consistency**: All readings must be **deterministically reproducible** under a fixed `seed`; if drift occurs, first check whether an un-fixed random source or lexicographical dependency was introduced.
3. **`state_hash` excludes `nid`**: For GPU≡CPU bit-for-bit consistency, all three must be present (rotation-system dict + `nid` + event count).
4. **`--` is not equal to 0**: `--` denotes "no structure" or "insufficient samples", and both must be read separately (see L2 verdict 3).
5. **Update this file**: Only update the corresponding table row when **frozen** new anchors (pressure-tested theorems/readings) are added, and synchronize with HEAD.
