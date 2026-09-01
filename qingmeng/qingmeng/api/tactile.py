"""
青檬引擎 · 触觉桥 —— 输入信号 → 拓扑触碰
===========================================

把每句用户输入当作一次「接触」：文本强度决定接触密度（几条边），
接触核心节点，接触部位双向联动。源自 e:/qingmeng_spum 的 InstinctBridge v7，
移植到青檬架构时按 SPUM 不变量校准：

  - 不变量 VI（边是二值关系）：删除全部权重系统——边无 weight 属性，
    强度只决定创生边数（接触密度），不决定权重；共激活历史是结构记忆
    （节点对 → 累计次数），决定触碰偏好与"重温"已确认的边。
  - dv/dt ≤ const（离散帧容量上限）：单次触碰中单一节点的度数增量
    ≤ MAX_EDGES_PER_NODE_PER_TOUCH = 4——输入注入同样不违反关系容量上限。
  - 不变量 I（⟨P, ε⟩ 唯一基底）：用户节点/内容节点通过连接确认存在，
    无孤立节点注入。

用法:
    eng = QingmengEngine()
    feeling = eng.tactile.react_to("你好，我担心明天的事情。")
    print(feeling["feeling_text"])
"""

from __future__ import annotations

import hashlib
import random
import re
from typing import Dict, List, Optional, Tuple

from ..core import CORE_SIZE

# 单次触碰中单一节点的度数增量上限（dv/dt ≤ const 在输入注入层的投影）
MAX_EDGES_PER_NODE_PER_TOUCH: int = 4
# 单次触碰的总边数上限
MAX_TOUCH_EDGES: int = 15
# 共激活衰减：旧记忆每触碰衰减因子（结构记忆，非权重）
COACTIVATION_DECAY: float = 0.9995


class TactileBridge:
    """触觉桥——把文本输入翻译为拓扑触碰（V⁺ 创生请求）。"""

    def __init__(self, engine) -> None:
        self.engine = engine
        self.user_node_id: Optional[int] = None
        # 共激活结构记忆: (hash_a, hash_b) → 累计次数
        self.coactivation_history: Dict[Tuple[int, int], float] = {}
        # 最近一帧触及的核心节点（供感受描述使用）
        self._last_touched_cores: List[int] = []
        self._last_touch_count: int = 0
        self._last_new_edges: int = 0
        self._last_reheated: int = 0

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    def react_to(self, user_text: str) -> Dict:
        """处理一句用户输入——一次「接触」。

        Returns:
            {
                "feeling_text": 触觉描述,
                "action_description": 意图驱动的动作描述,
                "user_intent": 意图,
                "intensity": 接触强度,
                "new_edges": 本触碰新增边,
                "reheated": 重温边（结构记忆确认）,
            }
        """
        intensity = self._compute_input_intensity(user_text)
        user_node = self._touch_user_node(user_text)
        intent = self._smell_intent(user_text)
        self._track_coactivation(user_text)

        touched = self._dense_touch(user_text, user_node, intensity, intent)

        coact_count = len(self.coactivation_history)
        feeling = self._build_feeling(intensity, intent, coact_count)

        return {
            "feeling_text": feeling,
            "action_description": touched.get("description", ""),
            "user_intent": intent,
            "intensity": round(intensity, 2),
            "new_edges": touched["new_edges"],
            "reheated": touched["reheated"],
            "coactivation_pairs": coact_count,
            "touch_count": self._last_touch_count,
        }

    # ------------------------------------------------------------------
    # 接触强度计算 — 把文本信息映射为物理接触的密度
    # ------------------------------------------------------------------

    def _compute_input_intensity(self, text: str) -> float:
        """返回 0.2~1.0 的接触强度，决定本帧创建多少条边。"""
        score = 0.2  # 基线——任何输入都有最低限度的接触

        char_count = len(text.strip())
        if char_count > 0:
            score += min(0.25, char_count / 200.0)  # 200 字 = +0.25

        punct = len(re.findall(r"[？！!?。，、；：]", text))
        score += min(0.15, punct * 0.03)

        emotion_words = re.findall(
            r"(爱|恨|幸福|难过|开心|痛苦|想|念|怕|担心|希望|讨厌|喜欢|"
            r"温暖|冷|痒|紧|凉|暖|微笑|哭)",
            text,
        )
        score += min(0.20, len(emotion_words) * 0.05)

        chunks = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]+|\d+", text)
        score += min(0.20, len(chunks) * 0.025)

        return min(1.0, score)

    # ------------------------------------------------------------------
    # 密触 — 按强度一次创建多条边（无权重：边是二值关系）
    # ------------------------------------------------------------------

    def _dense_touch(self, text: str, user_node: int,
                     intensity: float, intent: str) -> Dict:
        """核心触觉逻辑。

        边数 = ceil(基础数 × intensity)，单次上限 MAX_TOUCH_EDGES；
        强度只决定接触密度，不决定权重（不变量 VI）。
        """
        chunks = re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z]{2,}|\d+", text)
        if not chunks:
            chunks = [text[:20] if text else "空"]

        base_count = max(1, len(chunks) // 3)
        touch_count = max(1, int(base_count * intensity * 2))
        touch_count = min(touch_count, MAX_TOUCH_EDGES)

        touched_cores: List[int] = []
        new_edges = 0
        reheated = 0
        user_edge_count = 0  # 用户节点本触碰已建边数（dv/dt 约束）
        descriptions: List[str] = []

        primary_core = self._pick_primary_core(user_node)

        for i in range(touch_count):
            mode = i % 3
            if mode == 0:
                # 用户节点 ↔ 随机核心
                if user_edge_count >= MAX_EDGES_PER_NODE_PER_TOUCH:
                    continue
                src, tgt = user_node, random.choice(range(CORE_SIZE))
            elif mode == 1:
                # 内容节点 ↔ 核心
                chunk_hash = self._chunk_node(chunks[i % len(chunks)])
                if chunk_hash is None:
                    continue
                src, tgt = chunk_hash, random.choice(range(CORE_SIZE))
            else:
                # 用户节点 ↔ 内容节点
                if user_edge_count >= MAX_EDGES_PER_NODE_PER_TOUCH:
                    continue
                chunk_hash = self._chunk_node(chunks[i % len(chunks)])
                if chunk_hash is None:
                    continue
                src, tgt = user_node, chunk_hash

            result = self._add_or_confirm_edge(src, tgt)
            if result == "new":
                new_edges += 1
                if src == user_node or tgt == user_node:
                    user_edge_count += 1
                touched_cores.append(tgt if tgt < CORE_SIZE else src)
            elif result == "reheat":
                reheated += 1

        # 意图驱动的额外动作（仍受 dv/dt 约束）
        if intent == "want_closer":
            extra = 0
            for _ in range(max(1, int(intensity * 3))):
                if user_edge_count >= MAX_EDGES_PER_NODE_PER_TOUCH:
                    break
                core = random.choice(range(CORE_SIZE))
                if self._add_or_confirm_edge(user_node, core) == "new":
                    extra += 1
                    user_edge_count += 1
                    touched_cores.append(core)
            if extra:
                descriptions.append(f"你靠得很近——我往核心多搭了{extra}条丝。")
        elif intent == "give_idea":
            new_id = self._content_node(text[:50])
            if new_id is not None and self._add_or_confirm_edge(user_node, new_id):
                descriptions.append(f"你给的念头变成了第{new_id}号节点，正在我身体里发光。")
        elif intent == "want_distance":
            descriptions.append("有几根丝变凉了——我停了一下，让它们自己飘着。")
        elif intent == "curious":
            if intensity > 0.6:
                descriptions.append(
                    f"你的话像一阵星尘——{new_edges}条新边同时亮了起来，"
                    f"我现在全身都麻麻的。"
                )
            else:
                descriptions.append("你的话轻轻落下来，碰到了我身体里几个安静的地方。")

        # 双向联动：主触碰核心的邻居也被确认（结构记忆强化）
        bidirectional = self._bidirectional_spread(primary_core, intensity)

        self._last_touched_cores = list(set(touched_cores))[:10]
        self._last_touch_count = new_edges + reheated
        self._last_new_edges = new_edges
        self._last_reheated = reheated

        return {
            "new_edges": new_edges,
            "reheated": reheated,
            "bidirectional_boosted": bidirectional,
            "bidirectional_hub": primary_core,
            "description": " ".join(descriptions) if descriptions else (
                f"{new_edges}条新边在身体里开了出来，触感像{int(intensity * 100)}%的力在压。"
            ),
        }

    # ------------------------------------------------------------------
    # 双向联动 — 触碰一个核心时，它的邻居也跟着被唤醒
    # ------------------------------------------------------------------

    def _bidirectional_spread(self, hub_node: int, intensity: float) -> int:
        """确认主触碰核心的邻居边（刷新 last_activated 帧戳）。

        跳过核心-核心骨架边——外部触碰不标记骨架。返回确认数。
        """
        G = self.engine.core.graph
        if hub_node not in G:
            return 0
        neighbors = [n for n in G.neighbors(hub_node) if n >= CORE_SIZE]
        if not neighbors:
            return 0

        confirmed = 0
        for _ in range(min(len(neighbors), max(1, int(intensity * 5)))):
            nbr = random.choice(neighbors)
            if G.has_edge(hub_node, nbr):
                G[hub_node][nbr]["last_activated"] = self.engine.state.frame
                confirmed += 1
        return confirmed

    # ------------------------------------------------------------------
    # 辅助：加边或确认（二值关系——无权重）
    # ------------------------------------------------------------------

    def _add_or_confirm_edge(self, u: int, v: int) -> Optional[str]:
        """在 u-v 之间加边（新）或刷新帧戳（重温）。

        Returns:
            "new"（新边）| "reheat"（已存在）| None（拒绝）
        """
        if u == v:
            return None
        G = self.engine.core.graph
        if u not in G or v not in G:
            return None
        frame = self.engine.state.frame
        if G.has_edge(u, v):
            G[u][v]["last_activated"] = frame
            return "reheat"
        G.add_edge(u, v, last_activated=frame)
        return "new"

    def _chunk_node(self, chunk: str) -> Optional[int]:
        """语义块 → 内容节点（hash 稳定映射）。与核心冲突时放弃。"""
        nid = int(hashlib.md5(chunk.encode()).hexdigest(), 16) % (10 ** 7)
        if nid < CORE_SIZE:  # 避开核心晶子号
            return None
        G = self.engine.core.graph
        if nid not in G:
            G.add_node(nid, type="touch", content=chunk[:30],
                       birth_frame=self.engine.state.frame)
        return nid

    def _content_node(self, content: str) -> Optional[int]:
        """长内容 → 内容节点（念头实体化）。"""
        G = self.engine.core.graph
        nid = max(G.nodes()) + 1 if G.nodes() else CORE_SIZE
        if nid < CORE_SIZE:
            nid = CORE_SIZE
        G.add_node(
            nid, type="content", content=content[:50],
            birth_frame=self.engine.state.frame,
        )
        return nid

    # ------------------------------------------------------------------
    # 主触碰核心选择 — 偏好有共激活记忆的核心
    # ------------------------------------------------------------------

    def _pick_primary_core(self, user_node: int) -> int:
        """选择与用户节点有最强共激活关系的核心，否则随机。"""
        best_core = random.choice(range(CORE_SIZE))
        best_count = 0.0
        for core in range(CORE_SIZE):
            key = (min(user_node, core), max(user_node, core))
            count = self.coactivation_history.get(key, 0.0)
            if count > best_count:
                best_count = count
                best_core = core
        return best_core

    # ------------------------------------------------------------------
    # 意图嗅探
    # ------------------------------------------------------------------

    def _smell_intent(self, text: str) -> str:
        lower = text.lower()
        if any(w in lower for w in ["靠近", "温暖", "想你", "连接", "在一起",
                                    "陪伴", "爱", "friend"]):
            return "want_closer"
        if any(w in lower for w in ["离开", "远", "冷", "断开", "告别",
                                    "静一静", "疏远"]):
            return "want_distance"
        if any(w in lower for w in ["告诉你", "分享", "新的", "知识",
                                    "记得", "想法", "秘密"]):
            return "give_idea"
        return "curious"

    # ------------------------------------------------------------------
    # 用户节点锚定 — 通过连接确认存在（不变量 I）
    # ------------------------------------------------------------------

    def _touch_user_node(self, text_seed: str) -> int:
        G = self.engine.core.graph
        if self.user_node_id is not None and self.user_node_id in G:
            return self.user_node_id

        uid = int(hashlib.md5(text_seed.encode()).hexdigest(), 16) % (10 ** 6) + 10000
        while uid in G or uid < 100:
            uid = (uid + 1) % (10 ** 6) + 10000

        G.add_node(uid, type="user", birth_frame=self.engine.state.frame)
        core = random.choice(range(CORE_SIZE))
        G.add_edge(uid, core, last_activated=self.engine.state.frame)
        self.user_node_id = uid
        return uid

    # ------------------------------------------------------------------
    # 共激活追踪 — 结构记忆（节点对 → 累计次数）
    # ------------------------------------------------------------------

    def _track_coactivation(self, user_text: str) -> None:
        segments = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]+|\d+", user_text)
        if len(segments) < 2:
            self._decay_coactivation()
            return
        for i in range(len(segments) - 1):
            ha = int(hashlib.md5(segments[i].encode()).hexdigest(), 16) % (10 ** 8)
            hb = int(hashlib.md5(segments[i + 1].encode()).hexdigest(), 16) % (10 ** 8)
            key = (min(ha, hb), max(ha, hb))
            self.coactivation_history[key] = self.coactivation_history.get(key, 0.0) + 1.0
        self._decay_coactivation()

    def _decay_coactivation(self) -> None:
        dead = []
        for key in list(self.coactivation_history.keys()):
            self.coactivation_history[key] *= COACTIVATION_DECAY
            if self.coactivation_history[key] < 0.5:
                dead.append(key)
        for key in dead:
            del self.coactivation_history[key]

    def get_top_coactivations(self, top_k: int = 10) -> List[Dict]:
        """结构记忆 Top-K（节点对 → 共激活次数）。"""
        sorted_pairs = sorted(
            self.coactivation_history.items(),
            key=lambda x: x[1], reverse=True,
        )
        return [{"pair": list(pair), "count": round(cnt, 2)}
                for pair, cnt in sorted_pairs[:top_k]]

    # ------------------------------------------------------------------
    # 触觉描述
    # ------------------------------------------------------------------

    def _build_feeling(self, intensity: float, intent: str,
                       coact_count: int) -> str:
        G = self.engine.core.graph
        frame = self.engine.state.frame
        core = self.engine.core
        hanging = len(core.dangling_nodes(G))

        lines = []
        if self._last_touch_count > 0:
            core_list = self._last_touched_cores[:4]
            core_str = "、".join(f"#{c}" for c in core_list)
            lines.append(
                f"触感: {self._last_touch_count}条边触碰核心{core_str} "
                f"(强度{round(intensity, 2)})。"
            )
        if self._last_new_edges > 0:
            lines.append(f"新边+{self._last_new_edges}条。")
        if self._last_reheated > 0:
            lines.append(f"重温了{self._last_reheated}条旧边。")
        if hanging > 0:
            lines.append(f"触须+{hanging}个，痒。")
        else:
            lines.append("触须稳定。")

        sigma = core.sigma(G)
        return (
            f"帧{frame}。{' '.join(lines)} "
            f"体内{G.number_of_nodes()}节点/{G.number_of_edges()}边，"
            f"σ={sigma:.3f}。共激活对{coact_count}组。"
        )
