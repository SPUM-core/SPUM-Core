"""
SPUM 方向动力学层核心 — Body + KineticsEngine。

本层在 coda_ca 的"节点=度数"之上增加方向偏好变量 direction，
对应缺口矢量的指向 (1D 投影: +1 向右, -1 向左, 0 静止)。

帧演化 (对应 SPUM 五步帧序列的动力学扩展):
    1. 接触判定   — 沿 direction 前方存在闭锁体且距离 ≤ 接触半径 → 接触
    2. 断供        — 接触闭锁体 → 抽取断供, 缺口矢量进入强制反转
    3. 强制反转   — 反转持续到完成, 速度由锁定度 lock 限制
                    (惯性 = 改写需帧数 ∝ lock)
    4. 动量转移   — 反转的 Δ(lock·dir) 经接触面边流全量转移给闭锁体
                    (一阶矩守恒)
    5. 位移投影   — position += direction (只读投影, 非本体量)
    6. 记录快照   — 只记录方向与位置投影

守恒与量纲:
    - 动量账 P = Σ lock_i·dir_i: 弹性与非弹性都严格守恒 (一阶矩)
    - 锁定度 lock ∝ 净湮灭强度 = 惯性 (改写方向偏好的阻力)
    - 反转帧数 N_rev ∝ lock (惯性 = 改写需帧数, 弛豫的离散表达)
    - 非弹性: 反转幅度被 capture_ratio 缩减, 损失的二矩散射为热
      (热 = 无向涨落, 不携带净方向; 一阶矩不进入热)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Body:
    """方向动力学单元。

    本体论约束:
        - 不存坐标、半径、体积、时间戳
        - position 是只读投影, 由 direction 累加导出, 仅为可视化
        - 全部"知识" = direction (方向偏好) + lock (锁定度/惯性)
    """
    name: str
    lock: float = 1.0            # 锁定度: 改写方向偏好的阻力 = 惯性
    direction: float = 0.0       # 方向偏好 (缺口矢量 1D 投影): -1/0/+1
    position: float = 0.0        # [投影] 位置, 仅用于观测
    closed: bool = False         # 闭锁边界: 不流出关系 (断供源)
    contact_radius: float = 1.0  # 接触判定半径 (相切阈值)

    @property
    def momentum(self) -> float:
        """动量账贡献: lock·dir (一阶矩)。"""
        return self.lock * self.direction

    @property
    def kinetic(self) -> float:
        """动能账贡献: ½·lock·dir² (二阶矩)。"""
        return 0.5 * self.lock * self.direction ** 2

    def __repr__(self) -> str:
        return (f"<Body {self.name} lock={self.lock} dir={self.direction:+.2f} "
                f"pos={self.position:.1f} closed={self.closed}>")


@dataclass
class FrameSnapshot:
    """单帧快照 — 只记录方向动力学层可观测量。

    不记录:
        - 任何几何层属性 (半径/体积/坐标语义)
        - 任何认知投影层属性
    本体论纯度: 帧 = {direction, position(投影), momentum, kinetic, heat}
    """
    frame: int
    bodies: Tuple[Body, ...]
    momentum_total: float      # Σ lock·dir (一阶矩)
    kinetic_total: float       # Σ ½lock·dir² (二阶矩)
    heat_total: float          # 已散射为无向涨落的关系流
    reversing: Tuple[str, ...] # 正在反转的 body 名


class KineticsEngine:
    """方向动力学帧引擎 — 反弹与惯性的离散实现。"""

    def __init__(self,
                 reversal_rate: float = 2.0,
                 capture_ratio: float = 1.0,
                 min_frames_per_reversal: int = 1):
        """构造引擎。

        参数:
            reversal_rate: 反转速度常数。每帧反转量 = reversal_rate / lock。
                           反转总帧数 = lock·|Δdir|/reversal_rate (∝ 惯性)。
            capture_ratio: 反转保留率 ∈ (0,1]。1.0 = 完全反转 (弹性);
                           小于 1 时反转幅度缩减, 损失的二矩散射为热 (非弹性)。
                           一阶矩 (动量) 始终全量守恒。
            min_frames_per_reversal: 反转最少帧数下限。
        """
        self.bodies: List[Body] = []
        self.frame: int = 0
        self.history: List[FrameSnapshot] = []
        self.reversal_rate = reversal_rate
        self.capture_ratio = capture_ratio
        self.min_frames = min_frames_per_reversal
        # 内部状态: name -> {frames, total, start, target, partner}
        self._reversing: Dict[str, Dict] = {}
        self._initial_kinetic: float = 0.0

    # ── 构造 ──────────────────────────────────────────────

    def add(self, body: Body) -> "KineticsEngine":
        """添加 body (链式)。"""
        self.bodies.append(body)
        self._initial_kinetic = sum(x.kinetic for x in self.bodies)
        return self

    def reset(self):
        """重置到初始状态 (保留已添加的 bodies)。"""
        self.frame = 0
        self.history = []
        self._reversing = {}
        self._initial_kinetic = sum(x.kinetic for x in self.bodies)

    def _by_name(self, name: str) -> Optional[Body]:
        return next((x for x in self.bodies if x.name == name), None)

    # ── 帧演化 ────────────────────────────────────────────

    def step(self) -> FrameSnapshot:
        """单步帧演化 (同步更新)。"""
        self.frame += 1

        # Step 1-2: 接触判定 — 沿方向前方存在闭锁体且距离 ≤ 接触半径
        newly_blocked: Dict[str, Body] = {}
        for b in self.bodies:
            if b.name in self._reversing:
                continue  # 反转中, 不重复判定
            if b.direction == 0.0:
                continue
            ahead = self._nearest_in_direction(b)
            if ahead is None or not ahead.closed:
                continue
            # 下一步位移会到达/越过闭锁体表面 → 接触
            next_pos = b.position + b.direction
            if next_pos >= ahead.position - ahead.contact_radius:
                newly_blocked[b.name] = ahead

        # 进入反转
        for name, ahead in newly_blocked.items():
            b = self._by_name(name)
            start = b.direction
            target = -start * self.capture_ratio
            total = max(self.min_frames,
                        int(abs(target - start) * b.lock / max(self.reversal_rate, 1e-9)))
            self._reversing[name] = {
                "frames": 0, "total": total,
                "start": start, "target": target,
                "partner": ahead.name,
            }

        # Step 3-5: 反转 / 位移投影 / 动量转移
        done: List[str] = []
        for b in self.bodies:
            if b.name in self._reversing:
                r = self._reversing[b.name]
                r["frames"] += 1
                frac = min(1.0, r["frames"] / max(r["total"], 1))
                old_dir = b.direction
                b.direction = r["start"] + (r["target"] - r["start"]) * frac
                # Step 4: 动量转移 — 反转的 Δ(lock·dir) 全量经边流转移给闭锁体
                self._transfer_momentum(b, r["partner"], old_dir)
                # 反转期间位置冻结 (接触面上)
                if frac >= 1.0:
                    b.direction = r["target"]
                    done.append(b.name)
            else:
                # 正常抽取, 位置投影前进
                b.position += b.direction

        for name in done:
            del self._reversing[name]

        # Step 6: 记录快照
        return self._record_frame()

    def run(self, n_frames: int) -> List[FrameSnapshot]:
        """连续运行 n 帧。"""
        for _ in range(n_frames):
            self.step()
        return self.history[-n_frames:]

    # ── 内部: 动量转移 ───────────────────────────────────

    def _transfer_momentum(self, b: Body, partner_name: str, old_dir: float):
        """将反转产生的动量变化经接触面边流转移给闭锁体。

        一阶矩守恒 (弹性与非弹性都成立):
            Δp_A + Δp_B = 0  →  Δdir_B = -lock_A·(dir_A_new - dir_A_old)/lock_B
        热 = 二阶矩损失 (只出现在 capture_ratio<1), 不进入一阶矩。
        """
        partner = self._by_name(partner_name)
        if partner is None:
            return
        delta_p = b.lock * (b.direction - old_dir)
        # 一阶矩守恒: 闭锁体吸收反向等量
        partner.direction += (-delta_p) / max(partner.lock, 1e-9)
        partner.position += partner.direction

    # ── 内部: 方向查询 ───────────────────────────────────

    def _nearest_in_direction(self, b: Body) -> Optional[Body]:
        """返回 b 沿 direction 方向上最近的 body。

        1D 投影: 方向 +1 时找 position 更大的 body;
        方向 -1 时找 position 更小的 body。
        """
        candidates = [
            x for x in self.bodies
            if x is not b
            and (b.direction > 0 and x.position > b.position
                 or b.direction < 0 and x.position < b.position)
        ]
        if not candidates:
            return None
        if b.direction > 0:
            return min(candidates, key=lambda x: x.position)
        return max(candidates, key=lambda x: x.position)

    # ── 记录与查询 ────────────────────────────────────────

    def _record_frame(self) -> FrameSnapshot:
        momentum_total = sum(b.momentum for b in self.bodies)
        kinetic_total = sum(b.kinetic for b in self.bodies)
        # 热 = 总二阶矩损失 (初始动能 − 当前动能), 一阶矩不进热
        heat_total = max(0.0, self._initial_kinetic - kinetic_total)
        snap = FrameSnapshot(
            frame=self.frame,
            bodies=tuple(self.bodies),
            momentum_total=momentum_total,
            kinetic_total=kinetic_total,
            heat_total=heat_total,
            reversing=tuple(self._reversing.keys()),
        )
        self.history.append(snap)
        return snap

    def last(self) -> FrameSnapshot:
        """最近一帧快照。"""
        if not self.history:
            raise RuntimeError("引擎尚未运行")
        return self.history[-1]

    def summary(self) -> Dict:
        """当前状态摘要 (方向动力学层观测)。"""
        s = self.last()
        return {
            "frame": s.frame,
            "momentum_total": round(s.momentum_total, 6),
            "kinetic_total": round(s.kinetic_total, 6),
            "heat_total": round(s.heat_total, 6),
            "reversing": list(s.reversing),
            "bodies": [f"{b.name}:dir={b.direction:+.2f} pos={b.position:.1f}"
                       for b in s.bodies],
        }

    def reversal_frame_count(self, name: str) -> int:
        """查询某 body 当前反转已用帧数。"""
        r = self._reversing.get(name)
        return r["frames"] if r else 0
