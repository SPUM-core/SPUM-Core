#!/usr/bin/env python3
"""
派生文件维护漂移检测脚本

检查 .trae/rules/ 下的精简版规则文件是否在 knowledge.md 修改后
得到更新。若 knowledge.md 的修改时间 > 派生文件的修改时间 + 30天，
则输出警告。

用法：
    python tools/check_derived_drift.py          # 检查所有派生文件
    python tools/check_derived_drift.py --ci     # CI模式（检测到漂移则退出码1）

依赖：无外部库，仅使用 os.path.getmtime 和 datetime。

关联：审查问题 P1-3
"""

import os
import sys
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 权威基准
BENCHMARK = os.path.join(ROOT, "knowledge.md")

# 派生文件映射：(派生文件路径, 描述)
DERIVED_FILES = {
    os.path.join(ROOT, ".trae", "rules", "spum-core.md"): "核心公理精简版",
    os.path.join(ROOT, ".trae", "rules", "spum-structure.md"): "空间结构精简版",
    os.path.join(ROOT, ".trae", "rules", "spum-evolution.md"): "演化规则精简版",
    os.path.join(ROOT, ".trae", "rules", "spum-vocabulary.md"): "词汇规范精简版",
    os.path.join(ROOT, ".trae", "rules", "spum-reasoning.md"): "推理原则精简版",
    os.path.join(ROOT, ".trae", "rules", "spum-review.md"): "评审规则精简版",
    os.path.join(ROOT, ".trae", "rules", "spum-anti-pattern.md"): "伪加载检测精简版",
}

DRIFT_DAYS = 30  # 超过此天数未同步视为漂移


def check_drift(ci_mode: bool = False) -> int:
    """检查派生文件漂移。返回漂移文件数。"""
    if not os.path.exists(BENCHMARK):
        print(f"[ERROR] 权威基准文件不存在: {BENCHMARK}")
        return 1

    benchmark_mtime = datetime.fromtimestamp(os.path.getmtime(BENCHMARK))
    print(f"[INFO] knowledge.md 最后修改时间: {benchmark_mtime.isoformat()}")
    print(f"[INFO] 漂移阈值: {DRIFT_DAYS} 天\n")

    drift_count = 0
    missing_count = 0

    for derived_path, description in DERIVED_FILES.items():
        if not os.path.exists(derived_path):
            print(f"[MISS] {description} ({derived_path}) — 文件不存在")
            missing_count += 1
            continue

        derived_mtime = datetime.fromtimestamp(os.path.getmtime(derived_path))
        days_since_benchmark = (derived_mtime - benchmark_mtime).days

        if days_since_benchmark < 0:
            status = "DRIFT"
            drift_count += 1
            lag = abs(days_since_benchmark)
            print(f"[{status}] {description} — knowledge.md 更新后已 {lag} 天未同步")
            print(f"       派生文件: {derived_mtime.isoformat()}")
        else:
            status = "OK"
            print(f"[{status}] {description} — 派生文件较基准新 {days_since_benchmark} 天")

    print(f"\n{'='*60}")
    print(f"结果: {drift_count} 个文件漂移, {missing_count} 个文件缺失")

    if drift_count > 0 or missing_count > 0:
        print("建议: 检查是否需要在 knowledge.md 修改后同步更新派生文件")
        if ci_mode:
            return 1
    else:
        print("所有派生文件与 knowledge.md 同步正常")

    return 0 if ci_mode else 0


if __name__ == "__main__":
    ci = "--ci" in sys.argv
    ret = check_drift(ci_mode=ci)
    sys.exit(ret)
