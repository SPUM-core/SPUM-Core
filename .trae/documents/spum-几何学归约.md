# SPUM 几何学归约模块构建计划

## Context

几何相关内容散落在 6+ 处（SGG 1000 行、CGB、GT-027~032、spum-structure.md、L0_闭合与隔绝、L0L1L2 架构），但**没有统一模块**组织它们。同时，上一轮对话中确认了 Finsler/Randers 度量是 L2 投影层的正确语言——它解决了 AGENT.md §矛盾二「嵌入度量未定义」的悬置项。现有 `宇宙学/L0.5_induced_metric.md` 用 Riemannian 度量（只依赖位置），无法捕捉 σ 场方向依赖性与 V⁺/V⁻ 不可逆性；Finsler 是其自然升级。

**目标**：在 `几何学/` 下构建统一模块，精炼索引现有几何内容 + 创作 Finsler 新内容，共 38 节点（GEO-001~038）。

## 文件结构（8 文件 + 2 子目录）

```
几何学/
├── README.md                       # 总索引 + 节点速查
├── skill.md                        # 层级 Skill + 加载策略
├── spum-几何公理.md                  # GEO-001~008：8 公理 + L0-L4 五层几何角色
├── 欧氏几何/
│   ├── spum-发生学.md               # GEO-009~017：12 发生（SGG 精炼索引）
│   └── spum-组合恒等式.md            # GEO-018~020：Σ(6−deg)=12 / 高斯-博内（CGB 精炼索引）
├── Finsler几何/
│   ├── spum-Finsler公设.md           # GEO-021~026：F1-F6 公设（新创核心）
│   ├── spum-Randers度量.md           # GEO-027~030：σ 场几何化（新创核心）
│   └── spum-矛盾二解答.md            # GEO-031~033：AGENT.md §矛盾二结案（新创核心）
└── spum-几何应用桥.md                # GEO-034~038：跨模块桥
```

## 实施顺序

### 第 1 步：骨架文件（README + skill + 几何公理）

- `README.md`：模块定位、目录图、38 节点速查表、R1-R9 映射、依赖关系图
- `skill.md`：触发词、范式总纲、节点注册表、加载策略 7 种、伪加载检测
- `spum-几何公理.md`：8 公理 + **L0-L4 五层几何角色形式化**（核心新内容——每层几何角色 + 代码引用）

### 第 2 步：Finsler 新创内容（3 文件）

- `Finsler几何/spum-Finsler公设.md`：F1 方向依赖 / F2 不可逆度量 / F3 σ场各向异性 / F4 V⁺V⁻对偶 / F5 诱导坐标 / F6 Randers 形式。与 G1-G5（GT 球体公设）分层
- `Finsler几何/spum-Randers度量.md`：g_ij 由 σ 诱导（引用 `宇宙学/L0.5_induced_metric.md` §2.2.1 的 f(σ)=(σ₀/σ)^{1/3}）、b_i 由 ∇σ 诱导、引力幂次 r⁻² 退化、红移累积读数
- `Finsler几何/spum-矛盾二解答.md`：AGENT.md §矛盾二原文 → Finsler 解答 → Riemannian 局限 → 结案声明

### 第 3 步：欧氏几何索引（2 文件）

- `欧氏几何/spum-发生学.md`：12 个发生精炼（每个 ~150 字 + SGG 章节锚点），不复制 SGG 全文
- `欧氏几何/spum-组合恒等式.md`：V−E+F=2 / Σ(6−deg)=12 / 4π=12·(π/3)，指向 CGB 完整推导

### 第 4 步：应用桥 + 收尾

- `spum-几何应用桥.md`：5 桥（物理→引力 / 宇宙学→红移 / 化学→球体 / 图论→指标 / L0-L4 统一）

## 关键约束

1. **不复制现有文档**——SGG/CGB 精炼为索引 + 锚点
2. **Finsler 是新创核心**——三份 Finsler 文件是本模块的实质性贡献
3. **Riemannian 是 Finsler 在 b=0 的特例**——升级而非废弃 `宇宙学/L0.5_induced_metric.md`
4. **不直接修改 AGENT.md**——在 GEO-033 给出结案建议，由用户决定回写
5. **范式合规**：几何非本体（GEO-001）、π 降级（GEO-004）、离散优先（GEO-003）

## 关键引用源

| 节点范围 | 引用目标 | 引用类型 |
|---------|---------|---------|
| GEO-009~017 | `docs/Spatial_Geometry_Genesis/SPUM_Spatial_Geometry_Genesis.md` | 精炼索引 |
| GEO-018~020 | `docs/Combinatorial_Gauss_Bonnet/SPUM_Combinatorial_Gauss_Bonnet.md` | 精炼索引 |
| GEO-006 | `图论/spum-几何图论.md` GT-028~032 | 分层互补 |
| GEO-008 | `openSPUM/docs/L0L1L2_Architecture.md` | 形式化补充 |
| GEO-025~030 | `宇宙学/L0.5_induced_metric.md` + `induced_metric/embedding.py` | 升级取代 |
| GEO-019 | `openSPUM/docs/L0_Closure_and_Isolation_chi2.md` B1-B6 | 闭合判据 |
| GEO-033 | `AGENT.md` §矛盾二 | 状态升级建议 |

## 验证

1. 每个文件可独立阅读（不依赖同模块其他文件也能理解）
2. 范式自检：对照 spum-anti-pattern.md 8 种模式逐一检查
3. 交叉引用完整性：所有引用的文件路径真实存在
4. 节点编号连续：GEO-001~038 无缺号无重号
5. 加载策略可执行：skill.md 中 7 种加载组合的文件都存在
