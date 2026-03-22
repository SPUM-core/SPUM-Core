# 克鲁克斯辐射计文献综述与数据检索指南

## 背景
克鲁克斯辐射计（Crookes radiometer，又称光动辐射计）自1873年发明以来，其旋转行为一直存在争议。传统解释（光压、热蠕流）无法完整解释无光照下的温差驱动旋转，特别是旋转方向随环境温度反转的现象。

## SPUM 预测
- 在无光照、仅存在环境温差的条件下，叶片旋转方向由温度唯一决定：低温时涂黑面朝前，高温时镜面朝前。
- 存在临界温度 \(T_c\)（与叶片表面 VSPT 结构相关），在此温度下转速为零。
- 转速与温差线性相关，比例系数取决于材料耦合系数。

## 文献检索策略

### 关键词
- 英文：`Crookes radiometer`, `light mill`, `radiometer`, `temperature`, `rotation direction`, `vacuum`, `blackened`
- 中文：`克鲁克斯辐射计`, `光动辐射计`, `辐射计`, `温度`, `旋转方向`, `真空`

### 数据库
1. **NASA ADS** (https://ui.adsabs.harvard.edu/) – 物理学、天文学核心文献。
2. **Internet Archive** (https://archive.org) – 可下载1870–1950年代的扫描期刊、实验教材。
3. **Google Scholar** – 检索现代论文，注意筛选“无光照”实验。
4. **专业期刊**：*Nature*, *Philosophical Transactions of the Royal Society*, *Physical Review*, *American Journal of Physics*。
5. **专利数据库**：Espacenet, Google Patents（关键词 “radiometer” + “temperature”）。

### 重点文献示例（起步用）
1. **Crookes, W. (1874)**. On the repulsion resulting from radiation. *Philosophical Transactions of the Royal Society*, 164, 501–527. （最早的系统研究）
2. **Lewis, J. W. (1933)**. The effect of temperature on the rotation of the radiometer. *Physical Review*, 44, 529. （明确提出方向反转与温度相关）
3. **Oseen, C. W. (1925)**. The theory of the radiometer. *Arkiv för Matematik, Astronomi och Fysik*, 19, 1–28. （热蠕流理论经典，但未解释无光照反转）

## 数据提取方法
- **定性数据**：文献中明确描述“低温时黑面超前，高温时镜面超前”的观察，记录临界温度范围。
- **定量数据**：若文献给出转速-温度曲线，使用 WebPlotDigitizer (https://automeris.io/WebPlotDigitizer/) 提取数值点。
- **材料信息**：记录涂黑层（如“碳黑”“铂黑”）和镜面（如“银”“铝”）的具体描述。

## 数据记录模板
请使用仓库中 `data/processed/radiometer_rotation.csv` 的格式填写。

## 进度跟踪
当前任务：收集至少20篇包含**定性方向**信息的文献，并录入数据库。