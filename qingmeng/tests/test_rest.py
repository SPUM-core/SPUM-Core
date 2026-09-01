"""REST API 测试 — FastAPI TestClient（密闭 · 无真实网络）"""

from fastapi.testclient import TestClient

from qingmeng import QingmengEngine
from qingmeng.api.rest import create_app


def _client():
    eng = QingmengEngine()
    return TestClient(create_app(engine=eng)), eng


# ════════════════════════════════════════════════════════════════════
# 基础端点
# ════════════════════════════════════════════════════════════════════

def test_health():
    client, eng = _client()
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"] == "0.1.0"
    assert "青檬引擎" in body["engine"]
    assert body["consensus"]["passed"] is True


def test_state_snapshot():
    client, eng = _client()
    r = client.get("/state")
    assert r.status_code == 200
    body = r.json()
    assert body["frame"] == 0
    assert body["graph"]["nodes"] == 12
    assert body["graph"]["edges"] == 30
    assert body["graph"]["sigma"] == 0.4
    assert set(body["wuxing"].keys()) == {"水", "木", "土", "金", "火"}
    assert body["ledger"]["net_drift"] == 0
    assert body["consensus"]["passed"] is True


# ════════════════════════════════════════════════════════════════════
# 演化
# ════════════════════════════════════════════════════════════════════

def test_evolve_endpoint():
    client, eng = _client()
    r = client.post("/evolve", json={"frames": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["frame_id"] == 4          # evolve_many 返回最后一帧快照（帧从 0 起）
    assert body["consensus"]["passed"] is True


def test_evolve_validation():
    client, eng = _client()
    assert client.post("/evolve", json={"frames": 0}).status_code == 422
    assert client.post("/evolve", json={"frames": -1}).status_code == 422


# ════════════════════════════════════════════════════════════════════
# 触觉桥
# ════════════════════════════════════════════════════════════════════

def test_touch_endpoint():
    client, eng = _client()
    r = client.post("/touch", json={"text": "告诉你一个秘密，我有点紧张。"})
    assert r.status_code == 200
    body = r.json()
    assert "feeling_text" in body
    assert "user_intent" in body
    assert 0.2 <= body["intensity"] <= 1.0
    assert body["consensus"]["passed"] is True


def test_touch_requires_text():
    client, eng = _client()
    assert client.post("/touch", json={}).status_code == 422
    assert client.post("/touch", json={"text": ""}).status_code == 422


# ════════════════════════════════════════════════════════════════════
# L2 帧推理循环（确定性后端——无网络）
# ════════════════════════════════════════════════════════════════════

def test_reason_endpoint_deterministic():
    client, eng = _client()
    r = client.post("/reason", json={"prompt": "引力驱动行星运动"})
    assert r.status_code == 200
    body = r.json()
    assert body["total_frames"] >= 1
    assert len(body["frames"]) == body["total_frames"]
    # 每帧五步序列完整
    for f in body["frames"]:
        steps = f["steps"]
        assert all(steps.values()), f"帧{f['frame_id']} 五步未走完: {steps}"
    assert body["consensus"]["passed"] is True


def test_reason_endpoint_http_backend_no_key_falls_back(monkeypatch):
    """无 API Key 时 --backend http 自动回退确定性后端（引擎不被 LLM 阻塞）。

    强制清空 Key 环境变量（覆盖 .env），确保测试密闭、不发真实网络请求。
    """
    monkeypatch.setenv("QINGMENG_LLM_API_KEY", "")
    client, eng = _client()
    r = client.post("/reason", json={
        "prompt": "需求依赖供给", "backend": "http", "preset": "deepseek",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["total_frames"] >= 1       # 回退成功，循环照常产出
    assert body["consensus"]["passed"] is True


def test_reason_validation():
    client, eng = _client()
    assert client.post("/reason", json={"prompt": ""}).status_code == 422
    assert client.post("/reason", json={"prompt": "x", "max_frames": 0}).status_code == 422
    assert client.post("/reason", json={"prompt": "x", "backend": "幻觉"}).status_code == 422


# ════════════════════════════════════════════════════════════════════
# 端到端编排
# ════════════════════════════════════════════════════════════════════

def test_full_pipeline_via_api():
    """触觉 → 推理 → 演化 → 状态，全链路经 REST，共识持续通过。"""
    client, eng = _client()

    r = client.post("/touch", json={"text": "我有些担心，想靠近一点"})
    assert r.status_code == 200 and r.json()["consensus"]["passed"]

    r = client.post("/reason", json={"prompt": "经济周期驱动市场波动"})
    assert r.status_code == 200 and r.json()["consensus"]["passed"]

    r = client.post("/evolve", json={"frames": 20})
    assert r.status_code == 200 and r.json()["consensus"]["passed"]

    r = client.get("/state")
    body = r.json()
    assert body["consensus"]["passed"]
    # 触觉/推理注入的内容节点与核心骨架共存
    assert body["graph"]["nodes"] >= 12
    assert body["graph"]["edges"] >= 30
