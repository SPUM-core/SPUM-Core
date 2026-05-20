# SPUM — 空间粒子宇宙模型

本仓库是 SPUM 理论的**关系网络实现**。

## 核心思想

> 关系定义存在。一切存在皆为宇宙。

SPUM 以**离散关系网络 ⟨P, ε⟩** 为宇宙唯一本体基底，通过拓扑约束演化，涌现全部物理现象。

## 仓库结构
network/ # 理论的关系网络核心（AI 可直接读取推理）
nodes.txt # 概念节点清单（ID、名称、类型、状态）
edges.txt # 节点间关系边（derives_from、requires、constrains 等）
prompt.txt # 给 AI 的演化推理指令
texts/ # 传统长文（总论、推导、实验细节）
experiments/ # 实验数据与报告
sim/ # OpenSPUM 模拟

text

## 如何使用本仓库

1. 读取 `network/nodes.txt` 和 `network/edges.txt` 获取理论结构
2. 按 `network/prompt.txt` 的指令在网络中推理
3. 可追问 AI：脉络回溯、稳固性检查、倒塌分析、冲突检测

## 网络演化规则

- 修改理论 = 修改 `nodes.txt` 和 `edges.txt`
- 每次提交记录演化的一个步骤
- 总边数守恒是本体的约束（公理 N008）

## 许可证

本理论及相关文件以开放原则发布。检验它，修改它，直到与宇宙的原始数据对上。
