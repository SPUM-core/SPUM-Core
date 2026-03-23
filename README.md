cat > README.md << 'EOF'
# SPUM Core

**SPUM（空间粒子宇宙模型）** 是一个基于离散关系网络的宇宙理论模型。它宣告：宇宙的根本实在不是物体、时空或力，而是一个动态演化的关系网络 ⟨P, ε⟩。

- **空间** = 网络的编织方式
- **时间** = 创生事件的节律回响
- **物质** = 网络中的稳定拓扑模式
- **光** = 网络中自持传播的激发态（永恒粒子）
- **引力** = 网络为补偿湮灭而产生的集体重配

完整的理论阐述见 [`theory/SPUM2605.md`](theory/SPUM2605.md)。

---

## 目录结构
spum-core/
├── theory/ # 理论核心文档（与领域无关）
├── data/ # 原始数据与清洗后数据
│ ├── raw/ # 原始文件（PDF、截图等）
│ ├── processed/ # 结构化数据（按领域分目录）
│ └── external/ # 外部数据源链接
├── predictions/ # 各领域可检验预言（按领域分目录）
├── experiments/ # 实验方案与结果
├── code/ # 模拟、数据处理、分析代码
├── docs/ # 项目文档（贡献指南、字段规范等）
├── versioning/ # 版本管理元数据
└── open_spum/ # 开源协作计划

---

## 当前重点

**辐射计温差驱动旋转数据考古**  
我们正在系统收集克鲁克斯辐射计在纯温差场中的旋转方向数据，验证 SPUM 预测的“方向随温度反转”效应。

- 数据：`data/processed/radiometer/`
- 预言：`predictions/radiometer/01_direction_reversal.md`
- 阶段性报告：`docs/findings/radiometer_qualitative_findings.md`

---

## 如何贡献

1. 阅读 `docs/contribution_guide.md`
2. 查阅 `docs/data_field_spec.md` 了解数据格式规范
3. 在 Issues 中认领任务或提出问题
4. Fork 仓库，提交 Pull Request

欢迎为任何领域（天体物理、化学、生物、社科等）贡献数据、预言或代码。

---

## 许可证

- **代码**：MIT 协议（见 `LICENSE`）
- **数据**：CC0 1.0 公共领域（见 `data/LICENSE`）
- **文档**：MIT 协议（与代码相同）

---

## 联系

技术讨论：332556447@qq.com（临时）

---

**版本状态：** `a7f3e2d8.e27.r2.c4-8a3f1b.2c7e9d.4b1a6c.stable`