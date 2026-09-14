# SPUM-OpenSPUM — 开源宇宙实验室 Skill

> OpenSPUM 是 SPUM 公理体系的 Python 确定性实现：把 ⟨P, ε⟩ 关系网络的拓扑演化实例化为可运行代码——不是"模拟宇宙"，而是**运行宇宙的一段局部规则集**。
> 现行实现是 **L0–L4 分层投影架构**（`src/l0` ~ `src/l4`）；早期 v1 `Phase_0~4` 模拟器整体归档至 `_archive_v1/`（冻结，不再维护）。

---

## 一、现行目录结构

```
openSPUM/
├── src/
│   ├── l0/   combinatorial_proto.py  # L0 纯组合原型：RotNet 旋转系统（环序邻居表）
│   │         l0_core.py              # L0 帧演化：五步帧 + 四操作 + 悬挂边删除
│   │         l0_gpu.py               # L0 GPU 内核（CuPy；无 GPU 时 numpy 兜底）
│   │         l0_locality.py          # L0-A10 局部性自动断言（逐工件实测跳度）
│   ├── l1/   l1_projection.py        # L1 局部投影：角亏/局部标架/局部 σ
│   ├── l2/   l2_projection.py        # L2 全局坐标工件 + 全局读数（永不回写 L0）
│   ├── l3/   l3_projection.py        # L3 帧序列投影：δ / Δμ / d_topo / 闭合判据
│   └── l4/   l4_evolution.py         # L4 演化观察：净湮灭 / 集中 / 不完美定理
├── docs/
│   ├── L0L1L2_架构设计.md            # 唯一架构基准（含分岔判决 N1~N8、阶段锚点、断言表）
│   └── 本体模型协同推演工作模式.md    # 本地模型协同推演工作流（5 步 + 7 项审核清单）
└── _archive_v1/                      # v1 冻结归档：Phase_0~4 / tests / universe / webviz / l0_legacy
```

**分层原则**：L0 是**纯组合的二维闭合单纯复形**（核心规则「每条边恰好属于两个面」），不出现浮点、坐标、距离、角度；L1–L4 全是**只读投影**，永不回写 L0。

---

## 二、统一自检入口（各层唯一调用方式）

全部以 `python <模块>.py selftest=1` 调用，无外部测试框架依赖：

| 层 | 命令（在 `openSPUM/src/<层>/` 下执行） |
|----|----------------------------------------|
| L0 原型 | `python combinatorial_proto.py selftest=1` |
| L0 帧演化 | `python l0_core.py selftest=1` |
| L0 GPU | `python l0_gpu.py selftest=1` |
| L0 局部性断言 | `python l0_locality.py selftest=1` |
| L1 局部投影 | `python l1_projection.py selftest=1` |
| L2 全局投影 | `python l2_projection.py selftest=1` |
| L3 帧序列投影 | `python l3_projection.py selftest=1` |
| L4 演化观察 | `python l4_evolution.py selftest=1` |

> 便携 Python 全路径：`C:\Users\macotai\python-sdk\python3.13.2\python.exe`（未入 PATH）。
> 需要 GPU 的只有 `l0_gpu.py` 的 CuPy 路径；无 GPU 时自动 numpy 兜底，自检照样通过。

---

## 三、各层一句话与关键读数

| 层 | 一句话 | 代表读数（自检输出） |
|----|--------|----------------------|
| L0 原型 | 旋转系统承载全部组合信息，四个保 χ 操作 | icosa V/E/F/χ = 12/30/20/2；S=Σ(6−deg)=12 |
| L0 帧演化 | 一帧 = 创生→连接→变化体积→判断→删除（只删一层） | L0-A11 分量数不增（限默认 dmin=3） |
| L0 局部性 | L0-A10「所有 kernel 访问跳度 ≤ 二阶邻居」的**自动断言** | 逐工件界 1/0/1/2，480 次审计 0 失败；反向控制跨分量 ⇒ inf 被拒 |
| L1 | 局部投影：角亏、局部标架、局部 σ | tetra Σ=4π、icosa Σ=4π；邻角 A_v、defect |
| L2 | 全局坐标**工件**（不主张为真实几何）+ 全局读数 | 树边精确 = 2r(5)、holonomy = 曲率读数、边残差均值 |
| L3 | 帧与帧之间的投影（id 对齐 + 环序） | δ 臂、Δμ 锚点漂移；N013：δ 对 Δμ 完全无感 |
| L4 | 事件落在哪个区域（度数分区 core/sh/bg） | 净湮灭率 ρ；不完美双层判据持续不满足 |

**联合不变量**（逐帧可检验）：`χ ≡ 2`、`Σ(6−deg) = 12 + 2·holes`、闭合子图创生抑制、不完美定理（边缘 + 中心双臂）。

---

## 四、协同推演工作模式（本地模型）

修改/扩展 OpenSPUM 代码时按 `docs/本体模型协同推演工作模式.md` 推进：

1. 主 agent 写**接口规范**（放 `%TEMP%`，不入仓）
2. Python `urllib.request` 直发 **`qwen2.5-coder:14b`** @ `http://localhost:11434`（**不要用 PowerShell `Invoke-RestMethod`**）
3. 主 agent 按 7 项审核清单判定本地草稿
4. 重写落地（本地草稿**只作骨架**，正确性基准永远是 CPU 参考实现）
5. 三级验证 + 冻结进度锚点进 `docs/L0L1L2_架构设计.md` §7

硬约束：逐位一致 = 环序 dict + `nid` + 事件计数三者齐备（`state_hash` 不含 `nid`，须单独校验）。

---

## 五、归档 v1（`_archive_v1/`，冻结，仅供追溯）

v1 是 **L1 模拟器**（存在全局度数硬顶、O(N³) 全局扫描、力导向"摆放"12 晶子等问题，见架构文档 §0），已整体归档：
`_archive_v1/Phase_0~Phase_4/`、`_archive_v1/universe/`、`_archive_v1/tests/`（`measure_sigma_natural.py`、`test_phase*_all.py`、`verify_*.py` 等）、`_archive_v1/l0_legacy/`、`_archive_v1/webviz/`。
引用这些历史脚本时一律带 `_archive_v1/` 前缀；**现行开发入口只有 `src/l0`~`src/l4`**。

---

## 六、加载策略

| 场景 | 加载 |
|------|------|
| 运行 / 自检 OpenSPUM | 本节 §二 —— `python <layer>.py selftest=1` |
| 修改任一层的代码 | `docs/L0L1L2_架构设计.md`（§7 分岔判决 + §9 断言表）+ 对应 `src/lN/*.py` |
| 扩展代码（协同模式） | `docs/本体模型协同推演工作模式.md` |
| 理解公理↔代码映射 | `.trae/rules/spum-core.md` + `SPUM_系统总纲.md` |
| 追溯 v1 历史实现 | `_archive_v1/`（冻结，不维护） |
