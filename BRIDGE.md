# 桥接契约 · spum-core ⇄ 青檬引擎

> 本文件是**人读**的桥接契约。机器读版本在同目录 [`bridge/manifest.json`](bridge/manifest.json)。
> 对侧对应文件：`C:/Users/macotai/Desktop/工作/qingmeng/BRIDGE.md`

## 1. 定位

`qingmeng/` 与 `qingmeng-weapp/` 已于 2026-09-14 **分离出本仓**，成为独立项目「青檬引擎」。
本仓**不再包含青檬代码**，两者代码零耦合，只通过本契约交换**成果产物**。

| | spum-core（本仓） | 青檬引擎（对侧） |
|:---|:---|:---|
| 路径 | `C:/Users/macotai/Desktop/工作/spum-core` | `C:/Users/macotai/Desktop/工作/qingmeng` |
| 角色 | 源头：定义公理、本体论与推导图谱 | 执行体：把公理**跑成**可审计的推理帧 |
| 产出 | 公理规则 / 节点-边图谱 / 接口契约 | 帧快照 / S向量 / 共识报告 / 推导链 |
| 入口 | `spum.SPUM`（`from spum import SPUM`） | `qingmeng.QingmengEngine` |
| 前端 | — | 对侧 `weapp/`（Taro 小程序） |

## 2. 交换目录

```
bridge/
├── manifest.json          # 机器读契约（路径、清单、校验方式）
├── exchange/
│   ├── inbox/             # 对侧投递进来的产物（本侧只读）
│   └── outbox/            # 本侧产出待投递的产物
```

产物不入 git（`.gitignore` 只保留目录结构占位）。搬运为**人工或脚本按 manifest 执行**，不做自动同步。

### 投递流程

1. 本侧导出：产物写入 `bridge/exchange/outbox/<artifact-id>.<ext>`；
2. 生成边车：同名 `.manifest.json`，字段见下；
3. 放置到对侧 `bridge/exchange/inbox/`（同一路径名）；
4. 对侧按边车 `sha256` 校验后使用。

### 边车（sidecar）字段

```json
{
  "artifact": "s-vector",
  "producer": "qingmeng@0.1.0",
  "generated_at": "2026-09-14T10:00:00+08:00",
  "sha256": "<payload 的 sha256 十六进制>",
  "source_frames": 12,
  "notes": "可选"
}
```

## 3. 本侧导出（写给对侧）

| id | 本仓路径 | 用途 |
|:---|:---|:---|
| `spum-core-rules` | `.trae/rules/spum-core.md` | 公理与本体论基准 |
| `spum-evolution-rules` | `.trae/rules/spum-evolution.md` | 五步帧 / 创生湮灭 / 不完美定理 |
| `spum-wuxing-rules` | `.trae/rules/spum-wuxing.md` | 五形 → S 向量定义 |
| `spum-knowledge-core` | `knowledge-core.md` | 精简权威基准 |
| `graph-nodes` | `network/nodes.txt` | 22 个核心节点定义 |
| `graph-edges` | `network/edges.txt` | 推导边（derives_from / requires / refines…） |
| `qingmeng-guard` | `.trae/rules/spum-qingmeng-guard.md` | **接口契约 + 11 条不变量** |

## 4. 本侧导入（读对侧）

| id | 产出方法 | 内容 |
|:---|:---|:---|
| `frame-snapshot` | `QingmengEngine.summarize()` | 帧结构摘要（拓扑指标） |
| `s-vector` | `QingmengEngine.emergence()` | 五形相位向量 `S = (水,木,土,金,火)` |
| `consensus-report` | `QingmengEngine.check()` | 11 条不变量校验结果（含违规明细） |
| `reasoning-trace` | `QingmengEngine.reason(prompt)` | L2 帧推理循环的推导链 |
| `domain-interpretation` | `QingmengEngine.interpret(domain)` | 拓扑信号 → 领域语义（physics / sociology） |

## 5. 硬约束

1. **禁止跨仓 import**：本仓不得 `import qingmeng`，对侧不得 `import spum`。
   需要对方数据时走交换目录的产物文件。
2. **公理层只在本仓修订**。对侧回传的 `consensus-report` 若暴露公理问题，
   由本仓改规则文件（`.trae/rules/spum-*.md` + `knowledge*.md`），对侧跟随升级。
3. **护栏保留**：`.trae/rules/spum-qingmeng-guard.md` 定义本仓对青檬引擎的最低接口契约
   （11 条不变量），修改对侧代码前须按该文件校验。
4. **产物必须带边车**：无 `sha256` 的产物视为未投递。

## 6. 对侧文件

- `C:/Users/macotai/Desktop/工作/qingmeng/BRIDGE.md`
- `C:/Users/macotai/Desktop/工作/qingmeng/bridge/manifest.json`
