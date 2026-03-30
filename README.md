# 空间粒子宇宙模型（SPUM）核心仓库
Space Particle Universe Model (SPUM) - Core Repository

## 📌 项目简介
SPUM（空间粒子宇宙模型）是一套全新的宇宙学理论框架，以「空间粒子」为基本单元，重构宇宙的起源、演化与相互作用机制，彻底摆脱传统宇宙学对暗物质、暗能量、大爆炸奇点等不可观测实体的依赖。

本仓库为SPUM理论的**核心代码、理论文档、实验数据与自动化工具**集合，包含：
- 理论架构与核心论文
- AI驱动的理论评分系统
- 实验数据处理脚本
- CI/CD自动化流程
- 温差力矩效应等核心实验验证工具

---

## 🧩 目录结构
```
spum-core/
├── .github/workflows/    # CI/CD自动化配置（AI评分、版本生成）
├── core/v1.0/            # 核心理论文档（首个正式版本）
│   ├── theory.md         # SPUM2605核心理论全文
│   └── assets/           # 理论配图、公式、补充材料
├── draft/                # 草稿、迭代版本、实验设计
├── tools/                # 工具脚本
│   ├── score_calculator.py  # AI架构评分系统（逻辑图版）
│   ├── hash_validator.py    # 哈希校验、版本号生成
│   └── requirements.txt     # Python依赖
├── citations/            # 参考文献、引用文献
├── references/           # 对照文献、传统宇宙学资料
└── README.md             # 项目说明（本文件）
```

---

## ✅ 核心功能
### 1. AI架构评分系统
- **原理**：AI抽取理论架构 → 构建「公理→推导→现象」逻辑图 → 基于图结构量化评分
- **四维评分维度**：
  - **A 架构纯净度**：公理节点中SPUM原生实体占比，排除批判旧范式的误判
  - **B 推导严密性**：逻辑图无环性、连通性、无悬空节点
  - **C 归约完备度**：所有现象节点到公理的路径覆盖率
  - **D 方法论透明度**：数学工具与物理实体的区分度
  - **LE 梯子依赖指数**：移除传统术语后逻辑图的连通性
- **使用方式**：
  ```bash
  python tools/score_calculator.py core/v1.0/theory.md theory
  ```

### 2. CI/CD自动化流程
- 每次提交/PR自动触发AI评分
- 自动生成版本号、校验核心库准入阈值
- 自动输出评分报告、实验数据校验
- 配置文件：`.github/workflows/ci.yml`

### 3. 核心实验验证
- 温差力矩效应实验设计与数据处理
- 激光制冷实验模拟脚本
- 空间粒子相互作用数值模拟

---

## 🚀 快速开始
### 1. 环境依赖
```bash
pip install -r tools/requirements.txt
```

### 2. 克隆仓库
```bash
git clone https://gitee.com/space-particle-universe-model/spum-core.git
cd spum-core
```

### 3. 运行AI评分
```bash
python tools/score_calculator.py core/v1.0/theory.md theory
```

### 4. 提交修改
```bash
git add .
git commit -m "feat: 理论迭代/实验更新"
git push origin master
```

---

## 📊 核心理论亮点
1.  **无暗物质、暗能量假设**：以空间粒子的连接密度梯度解释引力效应，无需引入不可观测实体
2.  **自洽的宇宙演化模型**：从空间粒子的基本相互作用推导宇宙膨胀、星系旋转等现象
3.  **可验证的实验预言**：提出温差力矩效应等可直接验证的核心实验
4.  **统一的相互作用框架**：将引力、电磁力等基本相互作用统一于空间粒子的拓扑结构

---

## 🤝 贡献指南
1.  Fork 本仓库
2.  创建特性分支 (`git checkout -b feature/AmazingFeature`)
3.  提交修改 (`git commit -m 'Add some AmazingFeature'`)
4.  推送到分支 (`git push origin feature/AmazingFeature`)
5.  开启 Pull Request

---

## 📄 许可证
本项目遵循 [MIT 许可证](LICENSE) 开源。

---

## 📞 联系方式
- 项目维护：空间粒子宇宙模型研究团队
- 仓库地址：https://gitee.com/space-particle-universe-model/spum-core
```
