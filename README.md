# SPUM Core - 空间粒子宇宙模型

## 这是什么？

SPUM 是一个基于离散关系网络的宇宙模型。它宣告：宇宙的根本实在不是物体、时空或力，而是一个动态演化的关系网络 ⟨P, ε⟩。

- **空间** = 网络的编织方式
- **时间** = 创生事件的节律回响
- **物质** = 网络中的稳定拓扑模式
- **光** = 网络中自持传播的激发态
- **引力** = 网络为补偿湮灭而产生的集体重配

完整的理论阐述见 [`SPUM2605.md`](SPUM2605.md)（当前版本 `a7f3e2d8.e27.r2.c4-8a3f1b.2c7e9d.4b1a6c.stable`）。

---

## 当前任务：数据考古 — 克鲁克斯辐射计温差驱动旋转

我们正在通过系统整理已有公开数据，验证 SPUM 对辐射计温差驱动方向反转的预测。目前已完成：

- 收集并上传谭庆仁等 (2013) 论文（含冰水实验）
- 录入定性数据至 `data/processed/radiometer_rotation_qualitative.csv`
- 更新定性发现报告 `docs/radiometer_qualitative_findings.md`

**下一步**：继续收集更多文献（如吕章德 2004、英文网页资料），或深挖现有论文的压强分析。

---

## 如何参与

1. 阅读理论文档（`SPUM2605.md`），理解核心概念
2. 在 [Issues](https://gitee.com/您的用户名/spum-core/issues) 中查看可认领的任务
3. Fork 本仓库，修改后提交 Pull Request

您也可以直接在 Issues 中提出建议或疑问。

---

## 目录结构

```
spum-core/
├── SPUM2605.md                  # 理论完整文档
├── README.md                    # 本文件
├── LICENSE                      # MIT 协议
├── data/
│   ├── raw/                     # 原始文献 PDF、截图等
│   │   └── 谭庆仁_2013_辐射计是如何工作的.pdf
│   ├── processed/               # 清洗后的结构化数据
│   │   └── radiometer_rotation_qualitative.csv
│   └── external/                # 外部数据链接
├── docs/
│   ├── radiometer_lit_review.md # 文献检索指南
│   ├── data_field_spec.md       # 数据字段规范
│   ├── contribution_guide.md    # 贡献指南
│   └── radiometer_qualitative_findings.md # 定性发现报告
├── scripts/
│   ├── clean_data.py
│   └── extract_webplot.py
└── notebooks/
    └── README.md
```

---

## 联系方式

技术讨论：332556447@qq.com（临时）

---

**版本状态：** `a7f3e2d8.e27.r2.c4-8a3f1b.2c7e9d.4b1a6c.rc`（数据考古阶段）