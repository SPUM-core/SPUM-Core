"""
青檬引擎 · LLM 推理桥 — L2 帧推理循环
=======================================

定位（蓝图约束：LLM 是辅助推理器，不是引擎）：
  - LLM 不直接改图。它每次只产出一个「推理帧」（schema 约束的 JSON）：
    声明一个主张，提议 V⁺/V⁻ 操作。真值只在图上，不在 token 里。
  - 引擎执行推理帧 → 图更新 → 11 条共识校验 → 悬挂端判定 → 下一帧上下文。
  - 帧协议（对齐 src/core/frame.py 的 FrameProtocol 语义）：
      五步序列（创生→连接→变化→判断悬挂→删除）
      + 悬挂端（未闭合的推理链、未验证的假设、图悬挂端）
      + 闭合（悬挂端 = 0）+ 栈溢出回退（连续 3 帧未闭合 → 截断帧历史）
  - 默认离线确定性后端——无 API Key 也能跑；真实 LLM 通过 LLMBackend 注入。
    HTTP 后端失败自动回退到确定性后端——引擎永不被 LLM 阻塞。

L0 / L1 / L2 分层（防伪加载，spum-anti-pattern.md）：
  - L0 = 本体图 ⟨P, ε⟩：唯一的真值承载层
  - L1 = LLM 生成的文本主张：认知投影，不是本体事实
  - L2 = 帧推理循环本身：把 L1 主张逐步翻译回 L0 图操作并逐帧校验
  结论不是"生成"的——它由 L0 图结构支持（两概念是否连通、σ 方向）。
  概率/强度等连续量一律标注为认知投影（不变量 VIII/X），不是本体状态。

SPUM 校准（spum-qingmeng-guard.md）：
  - 不变量 VI：V⁺/V⁻ 只操作二值边——无权重、无方向
  - dv/dt ≤ const：单帧 LLM 边操作 ≤ MAX_LLM_EDGES_PER_FRAME = 2
  - 不变量 I：概念节点创建即连接——不注入孤立节点
  - 不完美定理：循环通常以"未完全闭合"结束是常态，不假装闭合

用法:
    from qingmeng import QingmengEngine
    eng = QingmengEngine()
    trace = eng.reason("水为什么往低处流")
    print(trace.summary_text)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..core import CORE_SIZE

# 单帧 LLM 边操作上限（dv/dt ≤ const 在推理注入层的投影，对齐 CREATION_MAX_PER_FRAME）
MAX_LLM_EDGES_PER_FRAME: int = 2
# 单帧单节点度数增量上限（dv/dt 投影）
MAX_LLM_NODE_DEG_DELTA: int = 2
# 栈溢出阈值：连续 N 帧未闭合 → 回退（对齐 FrameProtocol.STACK_OVERFLOW_THRESHOLD）
STACK_OVERFLOW_THRESHOLD: int = 3

# 合法推理帧类型
FRAME_TYPES = ("claim", "connect", "question", "conclude")
# 合法关系类型（对齐 src/core/frame.py 的 VALID_EDGE_TYPES + 对立关系）
VALID_RELATIONS = (
    "derives_from", "requires", "refines", "explains", "drives",
    "opposes", "",
)

# 关系关键词 → 关系类型（确定性后端的规则表）
RELATION_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "derives_from": ("推导", "得出", "源于", "由", "因此", "所以", "导致", "引发"),
    "requires": ("需要", "必须", "依赖", "依靠", "要求"),
    "refines": ("细化", "精确", "具体", "展开", "限定"),
    "explains": ("解释", "说明", "为什么", "因为", "原理"),
    "drives": ("驱动", "推动", "促使", "加速", "带动"),
    "opposes": ("矛盾", "反对", "冲突", "不兼容", "对抗", "对立"),
}

# 概念抽取停用词（虚词/语气词不构成概念）
TERM_STOPWORDS = frozenset({
    "一个", "这个", "那个", "什么", "为什么", "怎么", "如何", "是不是",
    "请问", "我们", "你们", "他们", "可以", "如果", "那么", "因为",
    "所以", "但是", "还是", "就是", "没有", "知道", "觉得", "感觉",
})
# 虚字——含任一字即丢弃该双字候选（的/了/是 等不构成概念本体）
PARTICLE_CHARS = frozenset(
    "的了着呢吗啊吧呀呢么是什在与和从被把对就都也很最个之而于其不"
    "有没为我你他她它们这在到又或及向以"
)


def _load_env_file() -> None:
    """从 .env 加载 QINGMENG_* 变量到 os.environ（无 dotenv 依赖）。

    只加载未设置的变量，不覆盖已有环境变量。文件缺失静默跳过。
    从当前工作目录向上级目录逐级查找（支持从子目录运行）。
    """
    import os as _os
    import pathlib

    if "QINGMENG_ENV_LOADED" in _os.environ:
        return
    _os.environ["QINGMENG_ENV_LOADED"] = "1"
    # 从 cwd 开始逐级向上找 .env
    cur = pathlib.Path(_os.getcwd())
    found = None
    for _ in range(5):
        p = cur / ".env"
        if p.is_file():
            found = p
            break
        if cur.parent == cur:
            break
        cur = cur.parent
    if found is None:
        return
    try:
        for line in found.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key.startswith("QINGMENG_") and key not in _os.environ:
                _os.environ[key] = val
    except OSError:
        pass


# ════════════════════════════════════════════════════════════════════
# 帧数据
# ════════════════════════════════════════════════════════════════════

@dataclass
class ReasoningFrame:
    """单个推理帧——L2 循环的最小单元。

    对应 FrameProtocol 的单帧状态：五步序列完成标记 + 悬挂端 + 闭合。
    """

    frame_id: int
    frame_type: str
    claim: str
    source: str
    target: str
    relation: str
    operations: List[str]
    confidence: float = 0.5          # 认知投影（信息完备度），不是本体状态
    concluded: bool = False
    dangling_count: int = 0
    dangling_items: List[str] = field(default_factory=list)
    is_closed: bool = False
    consensus_passed: bool = True
    consensus_failures: List[str] = field(default_factory=list)
    nodes_touched: List[int] = field(default_factory=list)
    edges_added: int = 0
    edges_removed: int = 0
    notes: List[str] = field(default_factory=list)
    # 五步序列完成标记（对齐 FrameProtocol）
    step_1_done: bool = False        # 创生：应用 V⁺/V⁻
    step_2_done: bool = False        # 连接：图已更新
    step_3_done: bool = False        # 变化：重算结构指标
    step_4_done: bool = False        # 判断：统计悬挂端
    step_5_done: bool = False        # 删除：修剪/回退裁决

    def all_steps_done(self) -> bool:
        return all((self.step_1_done, self.step_2_done, self.step_3_done,
                    self.step_4_done, self.step_5_done))

    def to_dict(self) -> Dict:
        return {
            "frame_id": self.frame_id,
            "frame_type": self.frame_type,
            "claim": self.claim,
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "operations": self.operations,
            "confidence": round(self.confidence, 3),
            "concluded": self.concluded,
            "dangling_count": self.dangling_count,
            "dangling_items": self.dangling_items,
            "is_closed": self.is_closed,
            "consensus_passed": self.consensus_passed,
            "consensus_failures": self.consensus_failures,
            "nodes_touched": self.nodes_touched,
            "edges_added": self.edges_added,
            "edges_removed": self.edges_removed,
            "notes": self.notes,
            "steps": {
                "1_创生V+": self.step_1_done,
                "2_连接": self.step_2_done,
                "3_变化": self.step_3_done,
                "4_判断悬挂": self.step_4_done,
                "5_删除/回退": self.step_5_done,
            },
        }


@dataclass
class ReasoningTrace:
    """一次 reason() 调用的完整轨迹。"""

    prompt: str
    frames: List[ReasoningFrame] = field(default_factory=list)
    stack_overflow: bool = False
    rolled_back: bool = False
    summary_text: str = ""

    @property
    def concluded(self) -> bool:
        return any(f.concluded for f in self.frames)

    @property
    def total_frames(self) -> int:
        return len(self.frames)

    def to_dict(self) -> Dict:
        return {
            "prompt": self.prompt,
            "concluded": self.concluded,
            "stack_overflow": self.stack_overflow,
            "rolled_back": self.rolled_back,
            "total_frames": self.total_frames,
            "summary": self.summary_text,
            "frames": [f.to_dict() for f in self.frames],
        }


# ════════════════════════════════════════════════════════════════════
# L2 帧协议（推理帧版）
# ════════════════════════════════════════════════════════════════════

class L2FrameProtocol:
    """推理帧协议：悬挂端追踪、闭合判定、栈溢出检测、历史回退。

    与 FrameProtocol 的差异：回退只截断推理轨迹（记忆），不撤销图操作
    ——V⁺ 历史是合法的关系容量重分配，旧帧的残留是新帧的动力。
    """

    def __init__(self) -> None:
        self.frames: List[ReasoningFrame] = []
        self.unclosed_streak: int = 0
        self.rollback_points: List[int] = []   # 已闭合帧的帧号
        self.rolled_back: bool = False

    def add(self, frame: ReasoningFrame) -> None:
        self.frames.append(frame)
        if frame.is_closed:
            self.unclosed_streak = 0
            self.rollback_points.append(frame.frame_id)
        else:
            self.unclosed_streak += 1

    def is_stack_overflow(self) -> bool:
        return self.unclosed_streak >= STACK_OVERFLOW_THRESHOLD

    def rollback(self) -> Optional[ReasoningFrame]:
        """截断到最近闭合帧。返回回退到的帧，或 None（无闭合帧）。"""
        if not self.rollback_points:
            return None
        last = self.rollback_points[-1]
        idx = next(i for i, f in enumerate(self.frames) if f.frame_id == last)
        self.frames = self.frames[: idx + 1]
        self.unclosed_streak = 0
        self.rolled_back = True
        return self.frames[-1]


# ════════════════════════════════════════════════════════════════════
# LLM 后端（可插拔；默认离线确定性）
# ════════════════════════════════════════════════════════════════════

class LLMBackend(ABC):
    """推理帧生成器接口——LLM 的唯一边界。

    generate(context) 必须返回一个 schema 合规的推理帧 dict：
        {
          "frame_type": "claim|connect|question|conclude",
          "domain": str,
          "claim": str,
          "source": str,          # 概念名（空串表示无）
          "target": str,          # 概念名（空串表示无）
          "relation": str,        # VALID_RELATIONS 之一
          "operations": ["V+"] 或 ["V-"] 或 [],
          "confidence": float,    # 认知投影（0~1）
          "concluded": bool,      # 是否声明循环结束
          "summary": str,         # 本帧小结（可选）
        }
    """

    @abstractmethod
    def generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        ...


class DeterministicBackend(LLMBackend):
    """离线确定性后端——规则生成推理帧，无网络、无随机。

    从提示文本抽取概念词（中文 2-6 字/英文词），按关键词判定关系类型，
    分三帧推进：claim（主张）→ connect（连到骨架）→ conclude（由图
    结构支持的结论）。置信度 = 信息完备度的认知投影：两概念已在图中
    连通 → 0.9；均已在图中 → 0.7；否则 0.5。
    """

    def __init__(self) -> None:
        self._term_cache: Dict[str, List[str]] = {}

    def _extract_terms(self, text: str) -> List[str]:
        """抽取概念词：中文语块内取双字 bigram，剔除含虚字的候选。

        确定性规则——不追求语义精度，图结构裁决真值（规则/LLM 只提议）。
        """
        if text in self._term_cache:
            return self._term_cache[text]
        terms: List[str] = []
        seen = set()
        for block in re.findall(r"[\u4e00-\u9fff]+", text):
            for i in range(len(block) - 1):
                w = block[i : i + 2]
                if w in TERM_STOPWORDS:
                    continue
                if any(ch in PARTICLE_CHARS for ch in w):
                    continue
                if w not in seen:
                    seen.add(w)
                    terms.append(w)
        result = terms[:4]
        self._term_cache[text] = result
        return result

    def _detect_relation(self, text: str) -> str:
        for rel, kws in RELATION_KEYWORDS.items():
            if any(kw in text for kw in kws):
                return rel
        return ""

    def _confidence(self, context: Dict[str, Any], src: str, tgt: str) -> float:
        evidence = context.get("evidence", {})
        pair_connected = evidence.get("connected", set())
        present = evidence.get("present", set())
        if (src, tgt) in pair_connected:
            return 0.9
        if src in present and tgt in present:
            return 0.7
        return 0.5

    def generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        prompt = context.get("prompt", "")
        progress = context.get("trace_frame_count", 0)
        domain = context.get("domain", "general")
        terms = self._extract_terms(prompt)

        if progress == 0:
            # 帧 1：主张——两概念间提议一条关系边
            if len(terms) >= 2:
                src, tgt = terms[0], terms[1]
            elif terms:
                src, tgt = terms[0], ""
            else:
                src, tgt = prompt[:12], ""
            return {
                "frame_type": "claim",
                "domain": domain,
                "claim": f"{src}与{tgt}的关系" if tgt else f"关于{src}的考察",
                "source": src,
                "target": tgt,
                "relation": self._detect_relation(prompt),
                "operations": ["V+"] if src and tgt else [],
                "confidence": self._confidence(context, src, tgt),
                "concluded": False,
            }
        if progress == 1:
            # 帧 2：连接——把概念节点连到骨架（最近的晶子由概念哈希决定）
            src = terms[0] if terms else prompt[:12]
            return {
                "frame_type": "connect",
                "domain": domain,
                "claim": f"{src}接入骨架",
                "source": src,
                "target": f"#{self._core_by_name(src)}",
                "relation": "",
                "operations": ["V+"],
                "confidence": 0.7 if terms else 0.5,
                "concluded": False,
            }
        # 帧 3：结论——由图结构支持的收束
        src = terms[0] if terms else prompt[:12]
        return {
            "frame_type": "conclude",
            "domain": domain,
            "claim": f"对'{prompt[:24]}'的图内结论",
            "source": src,
            "target": "",
            "relation": "",
            "operations": [],
            "confidence": self._confidence(context, src, ""),
            "concluded": True,
        }

    @staticmethod
    def _core_by_name(name: str) -> int:
        """概念名 → 晶子号（确定性，同一概念恒连同一晶子）。"""
        return int(hashlib.md5(name.encode()).hexdigest(), 16) % CORE_SIZE


class HttpLLMBackend(LLMBackend):
    """OpenAI 兼容 Chat Completions 后端（stdlib urllib，无新依赖）。

    配置优先级：构造参数 > 环境变量 > .env 文件 > 预设默认值。
    环境变量：
        QINGMENG_LLM_API_KEY    API Key
        QINGMENG_LLM_BASE_URL   服务地址
        QINGMENG_LLM_MODEL      模型名

    预设（preset 参数）：
        openai   → https://api.openai.com/v1 · gpt-4o-mini
        deepseek → https://api.deepseek.com · deepseek-chat

    任一环节失败（超时/网络/非法 JSON）自动回退到确定性后端——
    引擎的推理能力不依赖 LLM 可用性（约束 LLM 使用，辅助推理器定位）。
    """

    PRESETS = {
        "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
        "deepseek": {"base_url": "https://api.deepseek.com", "model": "deepseek-chat"},
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        preset: str = "openai",
        timeout: int = 60,
        fallback: Optional[LLMBackend] = None,
    ) -> None:
        _load_env_file()  # .env → os.environ（仅 QINGMENG_* 前缀）
        defaults = self.PRESETS.get(preset, self.PRESETS["openai"])
        self.api_key = (
            api_key
            or os.environ.get("QINGMENG_LLM_API_KEY")
            or ""
        )
        self.base_url = (
            base_url
            or os.environ.get("QINGMENG_LLM_BASE_URL")
            or defaults["base_url"]
        ).rstrip("/")
        self.model = (
            model
            or os.environ.get("QINGMENG_LLM_MODEL")
            or defaults["model"]
        )
        self.timeout = timeout
        self.fallback = fallback or DeterministicBackend()

    # ── 系统提示：schema 强制（LLM 只许输出推理帧） ────────────────

    SYSTEM_PROMPT = (
        "你是青檬引擎的辅助推理器，不是回答机器。"
        "你每次只输出一个推理帧（JSON 对象，禁止其他内容）。Schema：\n"
        '{"frame_type": "claim|connect|question|conclude", "domain": str, '
        '"claim": str, "source": str, "target": str, '
        '"relation": "derives_from|requires|refines|explains|drives|opposes|", '
        '"operations": ["V+"] 或 ["V-"] 或 [], "confidence": 0~1, '
        '"concluded": bool, "summary": str}\n'
        "约束：不编造事实；operations 表示提议的加边/删边；"
        "若信息不足用 frame_type=question 并 concluded=false；"
        "连续推进后在信息充分时用 conclude 帧结束循环。"
    )

    def generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_key:
            return self.fallback.generate(context)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": self._render_context(context)},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            spec = self._extract_json(content)
            if spec:
                return spec
        except Exception:
            pass  # 任何失败都回退——引擎不被 LLM 阻塞
        return self.fallback.generate(context)

    @staticmethod
    def _render_context(context: Dict[str, Any]) -> str:
        return json.dumps({
            "prompt": context.get("prompt", ""),
            "domain": context.get("domain", "general"),
            "graph": {
                "nodes": context.get("graph_nodes", 0),
                "edges": context.get("graph_edges", 0),
                "sigma": context.get("graph_sigma"),
            },
            "prior_frames": context.get("prior_frames", []),
        }, ensure_ascii=False)

    @staticmethod
    def _extract_json(content: str) -> Optional[Dict[str, Any]]:
        """从响应中提取第一个平衡的 JSON 对象。失败返回 None。"""
        start = content.find("{")
        if start == -1:
            return None
        depth, in_str, escape = 0, False, False
        for i in range(start, len(content)):
            ch = content[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(content[start : i + 1])
                        except json.JSONDecodeError:
                            return None
        return None


# ════════════════════════════════════════════════════════════════════
# LLM 推理桥
# ════════════════════════════════════════════════════════════════════

class LLMBridge:
    """L2 帧推理循环执行器。

    reason(prompt) = 后端生成推理帧 → 引擎执行 V⁺/V⁻ → 共识校验 →
    悬挂端判定 → 下一帧上下文 → 直到 conclude 或栈溢出回退。
    """

    def __init__(self, engine, backend: Optional[LLMBackend] = None) -> None:
        self.engine = engine
        self.backend = backend or DeterministicBackend()
        # 概念注册表：概念名 → 节点号（hash 稳定，跨调用复用）
        self.concept_registry: Dict[str, int] = {}

    # ── 对外接口 ─────────────────────────────────────────────────

    def reason(
        self,
        prompt: str,
        max_frames: int = 5,
        domain: str | None = None,
    ) -> ReasoningTrace:
        """执行一次 L2 帧推理循环。

        Args:
            prompt: 推理请求（一句话）
            max_frames: 循环上限（防御后端永不 conclude）
            domain: 领域标签（默认 general）

        Returns:
            ReasoningTrace —— 帧序列 + 图内结论 + 共识报告
        """
        protocol = L2FrameProtocol()
        context = self._build_context(prompt, domain, protocol)

        for i in range(max_frames):
            spec = self.backend.generate(context)
            spec = self._normalize_spec(spec)
            frame = self._execute_frame(spec, frame_id=i + 1)
            protocol.add(frame)
            if frame.concluded or protocol.is_stack_overflow():
                break
            context = self._build_context(prompt, domain, protocol)

        stack_overflow = protocol.is_stack_overflow()
        if stack_overflow:
            protocol.rollback()

        trace = ReasoningTrace(
            prompt=prompt,
            frames=protocol.frames,
            stack_overflow=stack_overflow,
            rolled_back=protocol.rolled_back,
        )
        trace.summary_text = self._build_summary(trace)
        return trace

    # ── 帧上下文 ─────────────────────────────────────────────────

    def _build_context(
        self,
        prompt: str,
        domain: str | None,
        protocol: Optional[L2FrameProtocol] = None,
    ) -> Dict[str, Any]:
        G = self.engine.core.graph
        evidence = {
            "connected": set(),
            "present": set(self.concept_registry.keys()),
        }
        for name, nid in self.concept_registry.items():
            if nid not in G:
                continue
            evidence["present"].add(name)
            for nbr in G.neighbors(nid):
                if nbr >= CORE_SIZE:
                    evidence["connected"].add((name, nbr))
        sigma = self.engine.core.sigma(G)
        prior = (
            [{"frame_id": f.frame_id, "frame_type": f.frame_type,
              "claim": f.claim, "concluded": f.concluded}
             for f in protocol.frames]
            if protocol else []
        )
        return {
            "prompt": prompt,
            "domain": domain or "general",
            "trace_frame_count": len(prior),
            "evidence": evidence,
            "graph_nodes": G.number_of_nodes(),
            "graph_edges": G.number_of_edges(),
            "graph_sigma": round(sigma, 3),
            "prior_frames": prior,
        }

    # ── 帧规范化 ─────────────────────────────────────────────────

    def _normalize_spec(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        clean = {
            "frame_type": spec.get("frame_type", "claim"),
            "domain": str(spec.get("domain", "general")),
            "claim": str(spec.get("claim", ""))[:200],
            "source": str(spec.get("source", "")).strip()[:30],
            "target": str(spec.get("target", "")).strip()[:30],
            "relation": str(spec.get("relation", "")) or "",
            "operations": [
                o for o in spec.get("operations", []) if o in ("V+", "V-")
            ],
            "confidence": float(spec.get("confidence", 0.5)),
            "concluded": bool(spec.get("concluded", False)),
            "summary": str(spec.get("summary", ""))[:200],
        }
        if clean["frame_type"] not in FRAME_TYPES:
            clean["frame_type"] = "claim"
        if clean["relation"] not in VALID_RELATIONS:
            clean["relation"] = ""
        clean["confidence"] = min(1.0, max(0.2, clean["confidence"]))
        return clean

    # ── 帧执行（五步） ───────────────────────────────────────────

    def _execute_frame(self, spec: Dict[str, Any], frame_id: int) -> ReasoningFrame:
        G = self.engine.core.graph
        frame = ReasoningFrame(
            frame_id=frame_id,
            frame_type=spec["frame_type"],
            claim=spec["claim"],
            source=spec["source"],
            target=spec["target"],
            relation=spec["relation"],
            operations=list(spec["operations"]),
            confidence=spec["confidence"],
            concluded=spec["concluded"],
        )
        if spec["summary"]:
            frame.notes.append(f"后端小结: {spec['summary']}")

        # 本帧节点度数增量账（dv/dt 投影：检查增量，不检查绝对度数）
        self._frame_node_delta: Dict[int, int] = {}

        # ── 1. 创生 V⁺/V⁻：应用提议的边操作（dv/dt 预算内） ──────
        edges_added, edges_removed, touched = self._apply_ops(spec, frame)
        frame.edges_added, frame.edges_removed = edges_added, edges_removed
        frame.nodes_touched = touched
        frame.step_1_done = True

        # ── 2. 连接：图已更新——无中间状态，无额外操作 ────────────
        frame.step_2_done = True

        # ── 3. 变化：重算结构指标（σ 局部波动，快照时体现） ──────
        self.engine.state.frame  # 触碰帧戳（无副作用读）
        frame.step_3_done = True

        # ── 4. 判断悬挂：图悬挂端 + 未决推理链 ────────────────────
        frame.dangling_items = self._collect_dangling(spec, G)
        frame.dangling_count = len(frame.dangling_items)
        frame.is_closed = frame.dangling_count == 0
        frame.step_4_done = True

        # ── 5. 删除/修剪：V⁻ 已在本帧应用；回退由协议层裁决 ───────
        frame.step_5_done = True

        # ── 共识校验：每帧强制过闸 ────────────────────────────────
        report = self.engine.check()
        frame.consensus_passed = report.passed
        frame.consensus_failures = [f.detail for f in report.failures]

        return frame

    def _apply_ops(
        self, spec: Dict[str, Any], frame: ReasoningFrame
    ) -> Tuple[int, int, List[int]]:
        """执行 V⁺/V⁻。返回 (新增边, 删除边, 触及节点)。

        约束：
          - 单帧边操作 ≤ MAX_LLM_EDGES_PER_FRAME（dv/dt 投影）
          - 单节点本帧度数增量 ≤ MAX_LLM_NODE_DEG_DELTA（增量，非绝对度数）
          - V⁻ 只解析既有节点，不创建；不得使端点孤立（deg→0 拒绝）
          - 概念节点创建即连接——不注入孤立节点（不变量 I）
        """
        G = self.engine.core.graph
        edges_added = edges_removed = 0
        touched: List[int] = []
        src, tgt = spec.get("source", ""), spec.get("target", "")
        if not src or not tgt or src == tgt:
            frame.notes.append("跳过操作：source/target 缺失或相同")
            return 0, 0, touched

        # V⁻ 先行（湮灭优先于创生，同一帧内不互相抵消）
        if "V-" in spec.get("operations", []):
            n_src, n_tgt = self._resolve_node(src), self._resolve_node(tgt)
            if n_src is None or n_tgt is None:
                frame.notes.append("跳过 V-：端点节点尚未实体化（不创建孤立节点）")
            elif not G.has_edge(n_src, n_tgt):
                frame.notes.append("跳过 V-：边不存在")
            elif G.degree(n_src) <= 1 or G.degree(n_tgt) <= 1:
                frame.notes.append("跳过 V-：删除会使端点孤立（deg→0 逻辑不自洽）")
            else:
                G.remove_edge(n_src, n_tgt)
                edges_removed += 1
                touched += [n_src, n_tgt]

        if "V+" in spec.get("operations", []):
            if edges_added >= MAX_LLM_EDGES_PER_FRAME:
                frame.notes.append("跳过 V+：本帧边预算已耗尽（dv/dt≤const）")
            else:
                n_src, n_tgt = self._concept_node(src), self._concept_node(tgt)
                if G.has_edge(n_src, n_tgt):
                    G[n_src][n_tgt]["last_activated"] = self.engine.state.frame
                    frame.notes.append(f"边 #{n_src}–#{n_tgt} 已存在（重温确认）")
                    touched += [n_src, n_tgt]
                elif self._node_delta_ok(n_src) and self._node_delta_ok(n_tgt):
                    # 创建即连接——两节点同时入图并建边（无孤立注入）
                    G.add_edge(n_src, n_tgt, last_activated=self.engine.state.frame)
                    edges_added += 1
                    self._frame_node_delta[n_src] = self._frame_node_delta.get(n_src, 0) + 1
                    self._frame_node_delta[n_tgt] = self._frame_node_delta.get(n_tgt, 0) + 1
                    touched += [n_src, n_tgt]
                else:
                    frame.notes.append("跳过 V+：触及节点度数增量超限（dv/dt≤const）")
                    self._drop_unused(n_src)
                    self._drop_unused(n_tgt)

        return edges_added, edges_removed, list(dict.fromkeys(touched))

    def _node_delta_ok(self, nid: int) -> bool:
        """本帧该节点度数增量是否未超上限（dv/dt 投影）。"""
        return self._frame_node_delta.get(nid, 0) + 1 <= MAX_LLM_NODE_DEG_DELTA

    def _drop_unused(self, nid: int) -> None:
        """清理本帧创建但未连成边的孤立节点（不变量 III 兜底）。"""
        G = self.engine.core.graph
        if nid in G and G.degree(nid) == 0:
            G.remove_node(nid)

    def _collect_dangling(self, spec: Dict[str, Any], G) -> List[str]:
        """悬挂端 = 图悬挂节点（结构）+ 未决推理链（语义）。

        只解析既有节点——检查不得创建节点（不变量 I：无孤立注入）。
        """
        items: List[str] = []
        for n in self.engine.core.dangling_nodes(G):
            items.append(f"图悬挂: #{n}")
        if spec.get("frame_type") == "question":
            items.append(f"未决问题: {spec.get('claim', '')[:40]}")
        if (
            spec.get("frame_type") in ("claim", "connect")
            and spec.get("source") and spec.get("target")
        ):
            src = self._resolve_node(spec["source"])
            tgt = self._resolve_node(spec["target"])
            if src is not None and tgt is not None and not G.has_edge(src, tgt):
                items.append(f"未闭合主张: {spec['source']}→{spec['target']}")
        return items

    # ── 概念节点 ─────────────────────────────────────────────────

    def _resolve_node(self, name: str) -> Optional[int]:
        """解析概念名 → 节点号。只查既有节点/注册表/晶子号，不创建。"""
        if name.startswith("#") and name[1:].isdigit():
            return int(name[1:])
        return self.concept_registry.get(name)

    def _concept_node(self, name: str) -> int:
        """概念名 → 节点（hash 稳定；注册表复用；避开核心晶子号）。

        "#N" 形式直接映射到晶子 N（连到骨架的语义）；普通概念名
        在创建时即被连接——不注入孤立节点（不变量 I）。
        """
        if name.startswith("#") and name[1:].isdigit():
            return int(name[1:])
        G = self.engine.core.graph
        if name in self.concept_registry and self.concept_registry[name] in G:
            return self.concept_registry[name]
        nid = int(hashlib.md5(name.encode()).hexdigest(), 16) % (10 ** 7)
        if nid < CORE_SIZE:
            nid += CORE_SIZE
        # 冲突线性探测：已被不同内容占用则位移
        while nid in G and G.nodes[nid].get("content") != name:
            nid = (nid + 13) % (10 ** 7)
            if nid < CORE_SIZE:
                nid += CORE_SIZE
        if nid not in G:
            G.add_node(nid, type="concept", content=name,
                       birth_frame=self.engine.state.frame)
        self.concept_registry[name] = nid
        return nid

    # ── 图内结论 ─────────────────────────────────────────────────

    def _build_summary(self, trace: ReasoningTrace) -> str:
        """把帧轨迹翻译为青檬语感的总结。

        结论由 L0 图结构支持：两概念是否连通、σ 方向、五形相位。
        连续量（σ/置信度）一律标注认知投影（不变量 VIII/X）。
        """
        G = self.engine.core.graph
        core = self.engine.core
        frame = self.engine.state.frame
        sigma = core.sigma(G)
        wuxing = self.engine.emergence()
        vec = wuxing.get("vector", {})

        # 已连通的（主张被图支持）——只解析既有节点，总结不创建节点
        supported: List[str] = []
        for f in trace.frames:
            if f.source and f.target and f.frame_type in ("claim", "connect"):
                a = self._resolve_node(f.source)
                b = self._resolve_node(f.target)
                if a is not None and b is not None and G.has_edge(a, b):
                    supported.append(f"{f.source}—{f.target}")

        # 图内结论（唯一真值来源）
        if supported:
            conclusion = "；".join(sorted(set(supported))) + " 已在图中连通——主张获得图结构支持"
        else:
            conclusion = "尚无概念对在图中连通——本循环未确立图内事实（认知投影层）"

        # 认知投影标注（不变量 X：连续是粗粒化）
        sigma_note = f"σ={sigma:.3f}（认知投影）" if sigma < float("inf") else "σ→∞（空图）"
        confs = [round(f.confidence, 2) for f in trace.frames]
        conf_note = f"帧置信度{confs}（信息完备度的认知投影，非本体状态）"

        parts = [
            f"帧{frame}。我把「{trace.prompt[:30]}」翻译成{trace.total_frames}个推理帧",
            f"新增V⁺ {sum(f.edges_added for f in trace.frames)}条 / "
            f"V⁻ {sum(f.edges_removed for f in trace.frames)}条（二值边，无权重）",
            f"{conclusion}。{sigma_note}。",
            f"五形: 木={vec.get('木', 0):.2f} 水={vec.get('水', 0):.2f} "
            f"火={vec.get('火', 0):.2f}（拓扑相位）。{conf_note}。",
        ]
        if trace.rolled_back:
            parts.append("推理链连续未闭合触发回退——旧帧的残留是新帧的动力（不完美）。")
        elif trace.stack_overflow:
            parts.append("连续未闭合触发栈溢出保护（无闭合帧可回退）——闭合不可达，不完美是常态，非缺陷。")
        elif trace.concluded:
            parts.append("循环已收束（图内结论确立）。")
        else:
            parts.append("循环未收束——不完美是常态，非缺陷。")
        return " ".join(parts)
