# SPUM 理论主张的引用支撑与可复现实验清单

> 目的：把 SPUM 的理论主张区分为三类——(A) 数学恒等式（可证明）、
> (B) 有学界先例的模型主张（有 peer-reviewed 文献锚点）、
> (C) SPUM 原创定量预言（尚无外部验证，需实验/观测裁决）。
> 并为每一类提供仓库内可一键复现的脚本。
>
> 对应规则基准：`spum-core.md`、`SPUM2610.md`、`knowledge.md`。

---

## 一、主张分类总表

| SPUM 主张 | 类别 | 复现脚本（本仓库） | 学界锚点 |
|-----------|------|--------------------|----------|
| Σ(6−deg(v)) = 12 / 拓扑常数 12 | A 数学恒等式 | `tools/repro_core_invariants.py` [2] | 离散高斯-博内 |
| 握手引理 / χ = V−E+F = 2 | A 数学恒等式 | `tools/repro_core_invariants.py` [1] | 经典图论 |
| 填补空隙必然产生新空隙（悬挂端不可消除） | A（几何论证）+ C（量化） | `tools/repro_core_invariants.py` [3] | 密堆/填充问题文献 |
| 锚点永不锁定（双层不完美） | B + C | `tools/repro_core_invariants.py` [4]；`experiments/n015|n016` | 非平稳学习/概念漂移 |
| 宇宙是离散关系网络（⟨P, ε⟩） | B（哲学-物理纲领） | `spum/` 单元测试 | 因果集、量子图性、Wolfram 超图 |
| 空间 = 连接关系（背景无关） | B（纲领） | — | Rovelli 圈量子引力、因果集 |
| 引力 = σ 密度梯度 / 净湮灭，预言 1/r³ | **C 原创预言** | `docs/Subgraph_CoMotion/fit_galaxy.py` | 需观测裁决（对比 MOND/ΛCDM） |
| 子图协动替代暗物质解释旋转曲线 | **C 原创模型** | `docs/Subgraph_CoMotion/fit_galaxy.py --nfw` | 对比 NFW：SPARC 175 数据 |
| 连续是离散的认知投影 / π 降级 | B（立场声明） | — | 离散外微分、Regge 微积分 |

---

## 二、类别 A：数学恒等式（可证明，非实证主张）

这些是图论的**定理**，SPUM 借用其形式，不构成经验主张：

1. **握手引理**：Σ deg(v) = 2E。任何有限图成立。
2. **欧拉公式 / 示性数**：闭合三角剖分曲面 χ = V − E + F = 2。
3. **离散高斯-博内 / Descartes 角亏损定理**：对闭合三角剖分，
   Σ(6 − deg(v)) = 12。这正是 SPUM "拓扑常数 12" 的来源——
   它是组合恒等式，不是测量结果。

> 学界对应（均为正式出版物，非 SPUM 原创）：
> - Descartes, R. (1630s) 角亏损定理——"Σ角亏损 = 4π" 的古典来源。
> - Knill, O. (2011). *A graph theoretical Gauss-Bonnet-Chern theorem*.
>   arXiv:1111.5395 —— 图上的离散高斯-博内现代形式。
> - Higuchi, Y. (2001). *Combinatorial curvature for planar graphs*.
>   J. Graph Theory 38, 220–229.

**复现**：

```bash
python tools/repro_core_invariants.py --frames 200 --steps 5000
# → [1][2] 项验证 Σdeg=2E、χ=2、Σ(6−deg)=12
```

对应单元测试：`tests/test_spum_graph.py`（握手/欧拉/Σ(6−deg)=12）。

---

## 三、类别 B：有学界先例的模型主张

SPUM 的离散-关系世界观并非孤例，以下 peer-reviewed / 正式出版物
与 SPUM 的对应主张共享核心思想：

### B1. 宇宙 = 离散关系网络（⟨P, ε⟩）

- **因果集理论**：时空是离散偏序集。
  Bombelli, L., Lee, J., Meyer, D., Sorkin, R. D. (1987).
  *Space-time as a causal set*. Phys. Rev. Lett. **59**, 521–524.
- **量子图性（Quantum Graphity）**：宇宙早期是图，几何从中涌现。
  Konopka, T., Markopoulou, F., Smolin, L. (2008).
  *Quantum graphity*. arXiv:hep-th/0611197.
- **Wolfram 超图模型**：以超图重写为底层，空间是图。
  Wolfram, S. (2020). *A class of models with the potential to
  represent fundamental physics*. Complex Systems **29**(2), 107–134.
- **因果动力学三角剖分（CDT）**：
  Ambjørn, J., Jurkiewicz, J., Loll, R. (2006).
  *The universe from scratch*. Contemp. Phys. **47**, 103–117.
- **Regge 微积分**：无坐标广义相对论，几何由离散单纯形连接决定。
  Regge, T. (1961). *General relativity without coordinates*.
  Nuovo Cimento **19**, 558–571.

### B2. 空间 = 关系，无背景容器

- Rovelli, C. (2004). *Quantum Gravity*. Cambridge University Press.
  圈量子引力：空间本身由自旋网络（图）构成。

### B3. 复杂网络的度量体系（σ、度分布、介数、小世界）

SPUM 的 σ、平均度数、边介数、聚类系数等指标来自网络科学：

- Watts, D. J., Strogatz, S. H. (1998). *Collective dynamics of
  'small-world' networks*. Nature **393**, 440–442.
- Barabási, A.-L., Albert, R. (1999). *Emergence of scaling in
  random networks*. Science **286**, 509–512.
- Albert, R., Barabási, A.-L. (2002). *Statistical mechanics of
  complex networks*. Rev. Mod. Phys. **74**, 47–97.
- Girvan, M., Newman, M. E. J. (2002). *Community structure in
  social and biological networks*. PNAS **99**, 7821–7826.
- Albert, R., Jeong, H., Barabási, A.-L. (2000). *Error and attack
  tolerance of complex networks*. Nature **406**, 378–382.
  （与"悬挂端删除级联"机制同源。）

### B4. 悬挂端不可消除 / k-core 组织

SPUM "删除悬挂端产生新悬挂端" 与图论 k-core / 2-core 骨架理论对应：

- Dorogovtsev, S. N., Goltsev, A. V., Mendes, J. F. F. (2006).
  *k-core organization of complex networks*.
  Phys. Rev. Lett. **96**, 040601.

### B5. 连续是离散的认知投影（π 降级 / 微积分离散化）

"连续可用离散系统充分模拟" 的工程与数学先例：

- Desbrun, M., Hirani, A. N., Leok, M., Marsden, J. E. (2005).
  *Discrete exterior calculus*. arXiv:math/0508341.
- Regge (1961) 同 B1：微分几何的离散替代。

### B6. 群体拓扑（邓巴数）

SPUM 社会学中"群体规模上界"与人类学结论一致：

- Dunbar, R. I. M. (1992). *Neocortex size as a constraint on group
  size in primates*. J. Human Evolution **22**, 469–493.

---

## 四、类别 C：SPUM 原创定量预言（需外部验证）

以下主张超出既有文献，SPUM 内部已有实验支撑，但**尚未经独立同行评审**。
这正是本仓库可复现脚本的价值所在——任何人有 numpy 即可复核：

| 预言 | 仓库内实验 | 运行命令 | 依赖 |
|------|-----------|----------|------|
| 悬挂端梯度范数显著更高（P0） | `experiments/p0_gradient_norm/run.py` | `python experiments/p0_gradient_norm/run.py` | torch |
| δ 驱动 Early Stopping（P1） | `experiments/p1_early_stopping/run.py` | `python experiments/p1_early_stopping/run.py` | torch |
| 锚点漂移永不锁定（N015） | `experiments/n015_anchor_drift/run.py` | `python experiments/n015_anchor_drift/run.py` | torch |
| 完整拓扑闭合在标准 SGD 不可达（N016） | `experiments/n016_topological_closure/run.py` | `python experiments/n016_topological_closure/run.py` | torch |
| 子图协动替代暗物质（旋转曲线） | `docs/Subgraph_CoMotion/fit_galaxy.py` | `python docs/Subgraph_CoMotion/fit_galaxy.py --nfw` | numpy/scipy |
| 温差力矩（不对称空间抽运） | `温差力矩实验/invert_competition.py` | `python 温差力矩实验/invert_competition.py` | numpy |
| 引力 r⁻³ 预言（区别于牛顿 r⁻²） | 理论推导见 `SPUM2610.md` | — | 需天文观测裁决 |

> 学界对比基线（已发表）：
> - 修正牛顿动力学：Milgrom, M. (1983). *A modification of the
>   Newtonian dynamics as a possible alternative to the hidden mass
>   hypothesis*. ApJ **270**, 365–370.（SPUM 与 MOND 同为
>   非暗物质路径，但机制与幂次不同——SPUM 预言 r⁻³。）
> - 暗物质范式（对比对象）：Navarro, J. F., Frenk, C. S., White,
>   S. D. M. (1996). *The structure of cold dark matter halos*.
>   ApJ **462**, 563.（NFW 轮廓，`fit_galaxy.py --nfw` 内置比较。）

---

## 五、零依赖核心不变量复现（CI 内置）

`tools/repro_core_invariants.py` 只依赖标准库，已接入 CI
（`.github/workflows/validate.yml` 与 `.gitee-ci.yml`）：

```bash
# 四项核心主张，确定性输出，退出码 0=全通过
python tools/repro_core_invariants.py
# [1] 握手引理+欧拉 [2] 拓扑常数12 [3] 悬挂端不可消除 [4] 锚点永不锁定
```

---

## 六、诚实边界

1. 类别 A（恒等式）**不构成对宇宙的描述**——它只保证 SPUM 框架
   内部自洽。宇宙本体是否为 ⟨P, ε⟩ 网络，是类别 C 的实证问题。
2. 类别 B 的文献锚点证明 SPUM "不是第一个这么想的"，但**共享思想
   不等于 SPUM 已被验证**。
3. 类别 C 是 SPUM 的试金石：r⁻³ 引力、悬挂端梯度、锚点漂移、
   子图协动——均以可运行脚本公开，等待独立复现与裁决。
4. 引用版本信息以出版方为准；如引用条目与出版方记录有出入，
   以出版社/期刊官方页面为准。

---

## 附：一键运行全部验证

```bash
# 1. 单元测试（207 项，纯标准库）
python -m unittest discover -s tests -v

# 2. 核心拓扑不变量（零依赖）
python tools/repro_core_invariants.py

# 3. 四维评分（信息性）
python tools/score_calculator.py knowledge.md theory
```
