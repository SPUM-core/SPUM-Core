"""
共识规则 Schema 化测试 — ConsensusValidator 由 consensus_schema.json 驱动。
"""

import json

import pytest

from qingmeng.core.axioms import Core
from qingmeng.core.consensus import ConsensusValidator


def _base_graph():
    return Core().build_icosahedron()


def test_schema_loaded_from_package():
    """内置 Schema：8 条规则，含 I/II/V/VI。"""
    v = ConsensusValidator()
    ids = [r["id"] for r in v.rules]
    assert len(v.rules) == 8
    assert "I" in ids and "II" in ids and "V" in ids and "VI" in ids


def test_schema_has_required_fields():
    """每条规则携带 id/name/scope/check/enabled/params 六要素。"""
    v = ConsensusValidator()
    for r in v.rules:
        assert {"id", "name", "scope", "check", "enabled", "params"} <= set(r)


def test_validate_runs_all_enabled_rules():
    """干净骨架：8 项检查全过。"""
    v = ConsensusValidator()
    report = v.validate(_base_graph(), ledger={"total_created": 0, "total_annihilated": 0})
    assert report.passed is True
    assert len(report.checks) == 8


def test_from_schema_external(tmp_path):
    """外部 Schema 可加载且规则生效（自定义 max_drift 阈值）。"""
    schema = {
        "version": 1,
        "rules": [
            {"id": "II", "name": "对偶", "scope": "ledger", "check": "check_II",
             "enabled": True, "params": {"max_drift": 2}},
        ],
    }
    p = tmp_path / "custom_rules.json"
    p.write_text(json.dumps(schema, ensure_ascii=False), encoding="utf-8")

    v = ConsensusValidator.from_schema(str(p))
    # 漂移 5 > 阈值 2 → 违规
    report = v.validate(_base_graph(), ledger={"total_created": 5, "total_annihilated": 0})
    assert report.passed is False
    assert any(f.invariant == "II" for f in report.failures)
    # 漂移 1 ≤ 阈值 2 → 通过
    ok = v.validate(_base_graph(), ledger={"total_created": 1, "total_annihilated": 0})
    assert ok.passed is True


def test_disabled_rule_skipped(tmp_path):
    """enabled=false 的规则被跳过（保留工程弹性）。"""
    schema = {
        "version": 1,
        "rules": [
            {"id": "III", "name": "无悬挂", "scope": "structural", "check": "check_III",
             "enabled": False, "params": {}},
        ],
    }
    p = tmp_path / "partial_rules.json"
    p.write_text(json.dumps(schema, ensure_ascii=False), encoding="utf-8")

    v = ConsensusValidator.from_schema(str(p))
    G = _base_graph()
    G.add_node(99)  # 孤立节点——若 III 启用则违规；禁用后跳过
    report = v.validate(G)
    assert report.passed is True
    assert all(c["invariant"] != "III" for c in report.checks)


def test_unknown_check_fails_closed(tmp_path):
    """规则表引用未注册函数 → 失败关闭（防规则漂移静默放行）。"""
    schema = {
        "version": 1,
        "rules": [
            {"id": "X99", "name": "幽灵规则", "scope": "structural",
             "check": "check_does_not_exist", "enabled": True, "params": {}},
        ],
    }
    p = tmp_path / "ghost_rules.json"
    p.write_text(json.dumps(schema, ensure_ascii=False), encoding="utf-8")

    v = ConsensusValidator.from_schema(str(p))
    report = v.validate(_base_graph())
    assert report.passed is False
    assert any(f.invariant == "X99" for f in report.failures)


def test_default_schema_immutable():
    """两次构造的规则一致（公理底线不漂移）。"""
    a = ConsensusValidator()
    b = ConsensusValidator()
    assert [r["id"] for r in a.rules] == [r["id"] for r in b.rules]
