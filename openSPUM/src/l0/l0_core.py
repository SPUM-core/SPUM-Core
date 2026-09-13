# -*- coding: utf-8 -*-
"""l0_core.py — L0 内核：确定性协商式帧演化。

架构基准: docs/L0L1L2_架构设计.md 第一部分（§2 本体结构 / §3 协商引擎 / §4 帧执行）
复用: combinatorial_proto.RotNet —— 旋转系统与四个保 χ 组合操作（规则已验证）

与原型 combinatorial_proto.py 的区别
------------------------------------
  原型 run_frame(mode="all")：帧首快照全部真面，**逐个全锥化**（无容量、无优先级）。
  本内核：真面/洞 → 候选 → 确定性优先级 → **每节点局部消解（cap-deg 预算）**
          → 端点接受集**交集提交** → **双缓冲**写入下一状态。

  即：把原型的"全锥化"泛化为"带容量约束的确定性协商"——结果与候选生成顺序无关。

帧语义（§4.2 双缓冲 / §4.1 五步帧）
-----------------------------------
  读 S_t：propose（生成候选）/ resolve（局部消解）只读当前状态，不改。
  写 S_{t+1}：commit / 悬挂修剪在**副本**上执行。
  交换：帧末 self.net = S_{t+1}。
  ⇒ 帧内不读写同一份状态；一帧只删一次（不级联）。

候选三类（§3.2）
----------------
  创生 V+：真三角面锥化（A 类）/ k≥4 洞锥化（B 类）——受 cap-deg 预算约束
  湮灭 V-：悬挂节点（deg < dmin）删除——不参与容量竞争（悬挂节点 deg<dmin，
           不可能是真面端点，故与创生候选天然不冲突）
  重连：删一条加一条（保持计数）——留作下一步扩展，接口见 TODO

驱动开关（两开关，不预设结论；供三探针判定「晶子是否涌现」）
------------------------------------------------------------
  drive   = "slack" | "saturate"
      "slack"    : 端点剩余容量之和**大**者优先 → 把创生摊到还没满的区域（现役默认）
      "saturate" : 端点剩余容量之和**小**者优先 → 把创生集中到接近饱和的区域
                   （§3.3「局部饱和驱动」：只要求局部度数最大化，不引入闭合判据）
  pairing = "none" | "conserved"
      "none"      : 不配平（现役默认）
      "conserved" : 帧内 ΣΔE = 0 —— **湮灭预算约束创生**（§3.2「每个创生必是某湮灭
                    的对偶补偿」）。顺序固定为「先定湮灭 → 再截断创生」：
                      步骤 1（只读 S_t）：规划本帧 V⁻ 边集 —— 从帧首旧边里按
                          「两端度数之和最大」优先（平局 (min id, max id) 字典序）贪心取，
                          维护虚拟度数，两端均 >= 2 才取；所得条数即本帧**湮灭预算 N**。
                      步骤 2：按优先级顺序提交创生，**逐条对预算**——cost = 3（A 类真面）
                          / k（B 类 k 边形洞）； spent + cost > N 则**跳过**该候选
                          （continue 而非 break，后续小 cost 候选仍可入选）。
                      步骤 3：断掉规划集的前 spent 条 → 创生/湮灭两类事件净 ΔE = 0。
                    注 1：prune（§4.3 判断→删除）是帧的独立步骤，**不计入**配平，
                          故帧净 ΔE 仍可能 != 0（= 被删节点度数之和）；探针实测并打印。
                    注 2：断边把两个三角面并成 k>=4 边形 → 破坏纯三角剖分，
                          于是 Σ(6−deg) 由恒 12 抬为 12 + 2d（d = 3V−6−E）。
                    注 3：N 是**结构上限**——「断后两端仍 >= dmin」约束下 S_t 上可断的边数，
                          非外生参数。该守卫使配平**不会**触发 §4.3 的 prune；若放宽到公理
                          下限 2，N 会被推到「所有可断边」= O(|E|)，湮灭端碾压创生端。
  vminus = "dangling" | "dense"   （非对称 V⁻：**湮灭判据不再等于创生的逆操作**）
      "dangling"（L0Core 类默认；§7.1/§7.2 锚点口径）：
                V⁻ 只有 §4.3 的悬挂修剪 + conserved 下的 `_plan_breaks`。
      "dense"  ：V⁻ = 断「过密边」——**两端皆为负曲率**（deg(u) > κ 且 deg(v) > κ）的边，
                 按度数之和最大优先，守卫取**公理下限 2**（而非 dmin）。两条后果：
                   · 判据（过密）与创生判据（正角亏）作用于**不同区域** ⇒ 破互逆；
                   · 断边可把端点压到 deg 2 < dmin ⇒ 触发 prune ⇒ 悬挂端再生（L0-A9）。
                 本模式下 plan 独立于 pairing：pairing="none" 断**全部** plan（净 ΔE ≠ 0），
                 pairing="conserved" 只断前 spent 条（ΣΔE = 0）。
  vplus = "any" | "gap"           （创生抑制）
      "any"（默认）：所有真面 / 洞都是创生候选。
      "gap"            ：只保留**正角亏**候选（δ = Σ_{v∈corners}(κ − deg(v)) > 0）。
                         δ <= 0 ⇒ 该处已饱和/过密 ⇒ 创生被抑制 =「闭合壳内创生抑制」。
  kappa = 6                       # 平坦阈值：角亏 δ(v) = κ − deg(v)；三角剖分 κ=6

采纳的内核动力学（2026-09-13，§7.4 四组对照裁定）
--------------------------------------------------
  **vminus="dense" + vplus="any"** —— 非对称湮灭开启，创生不抑制。
  对照（seed=icosa cap=12 dmin=3，10 帧）：
    基线 dangling/any → **饱和冻结于 V=44**（cap 绑住度数，t≥6 不再变）；
                        注：无 cap（cap=64）时才是「V/E 每帧 ×3」的自相似通胀。
    dense/any        → V/Vprev 非常数（1.125~2.667）、持续生长、L0-A9 激活 ⇒ **采纳**
    dangling/gap     → t=1 冻结 V=32（创生在饱和区关闭 + 无过密边 ⇒ 两端同时熄火）
    dense/gap        → t=3 冻结 V=62（同上）
  引擎：**湮灭降度 → 再生正角亏 → 创生回填 → 新枢纽 → 再过密**；抑制创生即切断此回路。
  代价：纯组合无尺度层级 ⇒ V 无界增长（持续演化，但非自相似）；maxdeg 被 cap 钉住。
  口径：CLI `dyn=1` 默认走采纳配置（dense/any）；`dyn=1 vminus=dangling` 回到基线。
  注（2026-09-13 修正 X1）：propose 新增「新点度数 ≤ cap」校验——cone_hole 的新点度数 = 洞
      边数 k，此前未校验，锥化大洞会造出 deg=k > cap 的超容点（L0-A5 违反）。

已知边界（实测；见 selftest 第 6/7 项）
--------------------------------------
  锥化创生与悬挂修剪**精确互逆**：每次 A 类锥化恰新生 1 个 deg-3 元胞，
  按 dmin=3 修剪恰好删回它，净结构不变。故（基线参数下）：
    dmin <= 3：创生只增度，没有任何节点能降到 dmin 以下 => 删除分支恒不触发；
    dmin >= 4：创生与删除逐帧精确抵消，系统回到种子（零净演化）。
  §9 的 L0-A9（悬挂修剪后必再生悬挂）需要**非对称的 V- 湮灭规则**才能激活。
  已由 vminus="dense" 提供：过密边湮灭的判据不是锥化的逆操作，且守卫取公理下限 2，
  故断边可真的把节点压到 dmin 之下 —— 见 §7.4 探针（`dyn=1`）。

用法
----
  python l0_core.py selftest=1
  python l0_core.py seed=tetra nframes=6 cap=64 dmin=3
  python l0_core.py seed=icosa nframes=8 cap=5       （小 cap → 演示协商预算生效）
  python l0_core.py seed=icosa puncture=1 nframes=4  （V- 单点穿刺 → B 类洞锥化路径）
  python l0_core.py seed=icosa nframes=8 cap=12 probe=1 drive=saturate pairing=conserved
      （三探针：度数分布 / 闭合壳 / 周期检测；drive 与 pairing 见下节）
  python l0_core.py seed=icosa nframes=9 bare=1
      （裸动力学：cap / 优先级 / 驱动全删，只留 锥化所有真面 + 帧末修剪）
  python l0_core.py seed=icosa nframes=10 dyn=1
      （非对称动力学读数（默认走采纳配置 dense/any）：σ / V+创生 / V-断边 / V-删点 /
        netΔE / 度直方图；判据 V/Vprev 非常数 ⇒ 离开自相似轨道。
        加 vminus=dangling 回基线；加 vplus=gap 复现「创生抑制 ⇒ 冻结」对照）
  python l0_core.py recon=1 mode=phi1 V=13 nframes=40 perturb=0
      （重连动力学：固定 V，动作集 = 边翻转（对合）；mode = none | phi1 | phi2；
        perturb = 起步前先走的 Φ₂ 步数；V=0 时用 seed）
  python l0_core.py seed=icosa extra=tetra cap=5 nframes=21 product=1
      （稳定结构读数：连通分量 / 分量级 12 壳（V=12 且全 deg==5）/ 缺口清单。
        extra 与 seed **不连通**地并成多分量种子 ⇒ 0..11 恒为独立 12 壳；
        同时逐帧取证 L0-A11「分量数不增」）
"""

import math
import sys
from collections import Counter

from combinatorial_proto import RotNet, SEEDS

CREATION = "V+"
ANNIHILATION = "V-"


def _clone(net):
    """深拷贝旋转系统（环序表逐行复制），保留 id 计数器（避免复用已删 id）。"""
    c = RotNet({v: list(r) for v, r in net.rot.items()})
    c.nid = net.nid
    return c


def _snapshot(net):
    """用于确定性比较的规范化视图。"""
    return {v: tuple(r) for v, r in sorted(net.rot.items())}


def _edge_set(net):
    """无向边集合（规范化 (min,max)，确定性排序）。"""
    return sorted({(v, w) if v < w else (w, v)
                   for v in net.ids() for w in net.rot[v]})


def _net_key(net):
    """状态键：环序表的可哈希规范形式（用于重连动力学的周期检测）。"""
    return tuple((v, tuple(net.rot[v])) for v in sorted(net.rot))


def _psi(net):
    """重连势 Ψ = Σ(6−deg)²。Σ(6−deg) ≡ 12 是恒等式（帧不变），非约束；
    故 Ψ 只能在下界 ⌈144/V⌉ 及以上取值，取到下界即达该 V 的平台。"""
    return sum((6 - net.deg(v)) ** 2 for v in sorted(net.ids()))


class L0Core:
    """L0 内核。net 为旋转系统状态，cap 为容量上界（安全预算，非涌现极限）。"""

    def __init__(self, net, cap=64, dmin=3, drive="slack", pairing="none",
                 vminus="dangling", vplus="any", kappa=6):
        self.net = net
        self.cap = int(cap)      # §2.2 容量约束 deg ≤ cap（内存/安全上界）
        self.dmin = int(dmin)    # §4.3 删除阈值（公理下限 2；三角剖分下限 3）
        self.drive = str(drive)      # "slack" | "saturate"（见模块头「驱动开关」）
        self.pairing = str(pairing)  # "none" | "conserved"
        self.vminus = str(vminus)    # "dangling"（现状）| "dense"（过密边湮灭）
        self.vplus = str(vplus)      # "any"（现状）| "gap"（只在正角亏处创生）
        self.kappa = int(kappa)      # 平坦阈值 κ：角亏 δ(v) = κ − deg(v)
        if self.vminus not in ("dangling", "dense"):
            raise ValueError(f"未知 vminus={vminus}，可选 dangling | dense")
        if self.vplus not in ("any", "gap"):
            raise ValueError(f"未知 vplus={vplus}，可选 any | gap")
        self.t = 0

    def gap(self, vs):
        """角亏 δ = Σ_{v∈vs}(κ − deg(v))。

        κ=6 是三角剖分的平坦阈值（deg=6 ⇒ 角亏 0）。δ > 0 ⇒ 正曲率/缺口；
        δ ≤ 0 ⇒ 该处已饱和或过密。这是 vplus="gap" 与 vminus="dense" 共用的标量。
        """
        return sum(self.kappa - self.net.deg(v) for v in vs)

    # ========================================================
    # §2 本体结构：状态只读查询（委托旋转系统）
    # ========================================================
    def deg(self, v):
        return self.net.deg(v)

    def slack(self, vs):
        """端点剩余容量之和（§3.3 优先级来源之一）。"""
        return sum(self.cap - self.net.deg(v) for v in set(vs))

    # ========================================================
    # §3.1 候选生成（只读当前状态，确定性顺序）
    # ========================================================
    def propose(self):
        """返回候选列表。候选为 dict：
            type   : "V+" / "V-"
            kind   : "face" | "hole" | "node"
            corners: 参与端点（face/hole 为环序；node 为单点）
            key    : 确定性优先级键（越小越优先）
        """
        cands = []
        # 面迹 = 旋转系统的 dart 环。长度 3 = 真三角面；长度 ≥4 = 洞。
        # 从 faces() 取候选 ⟺ 只锥化真面 ⟺ 天然排除"分离三角"（§2.4）。
        for cyc in self.net.faces():
            vs = self.net.face_vertices(cyc)
            if len(vs) == 3 and len(set(vs)) == 3:
                kind = "face"
            elif len(vs) >= 4 and len(set(vs)) == len(vs):
                kind = "hole"
            else:
                continue
            # §2.2 容量约束（L0-A5）：创生新建的点度数 = 3（face）/ k（hole）。
            # 新点同样是 "deg ≤ cap" 的承载者，而 §3.4 的 resolve 只约束旧角点预算，
            # 故新点度数必须在此独立校验——否则锥化大洞会造出 deg=k 的超容点。
            if len(vs) > self.cap:
                continue
            # vplus="gap"：只在**正角亏**（缺口/正曲率）处创生 —— 闭合壳内创生抑制
            if self.vplus == "gap" and self.gap(vs) <= 0:
                continue
            cands.append(self._mk(CREATION, kind, tuple(vs)))
        # 湮灭候选：悬挂/孤立节点
        for v in self.net.ids():
            if self.net.deg(v) < self.dmin:
                cands.append(self._mk(ANNIHILATION, "node", (v,)))
        return cands

    def _mk(self, ctype, kind, corners):
        return {"type": ctype, "kind": kind, "corners": corners,
                "key": self.priority(ctype, kind, corners)}

    def priority(self, ctype, kind, corners):
        """确定性优先级：对称（只看端点集合）/ 局部 / 可比较 / 确定。
        规则（drive 决定方向）：
            "slack"    : 端点剩余容量之和越大越优先（把创生摊到没满的区域）；
            "saturate" : 端点剩余容量之和越小越优先（把创生推向饱和）。
        平局按端点字典序。终审仍在 commit（重新校验判据），故优先级只需给出一致全序。
        """
        s = self.slack(corners)
        sign = 1 if self.drive == "saturate" else -1
        return (0 if ctype == CREATION else 1,
                sign * s,
                tuple(sorted(corners)))

    # ========================================================
    # §3.4 局部消解（每个节点在本地按预算取前若干候选）
    # ========================================================
    def resolve(self, cands):
        """返回 {节点: 该节点接受的候选 id 集合}。只处理创生候选（消耗容量）。
        湮灭候选不占容量、也不与创生竞争，故不参与局部消解。"""
        by_node = {}
        for c in cands:
            if c["type"] != CREATION:
                continue
            for v in set(c["corners"]):
                by_node.setdefault(v, []).append(c)
        accept = {}
        for v, lst in by_node.items():
            budget = self.cap - self.net.deg(v)
            if budget <= 0:
                accept[v] = set()
                continue
            order = sorted(lst, key=lambda c: c["key"])
            accept[v] = {id(c) for c in order[:budget]}
        return accept

    # ========================================================
    # §3.5 交集提交（双边/多边：所有端点都接受才成立）
    # ========================================================
    def commit(self, nx, cands, accept, budget=None):
        """在副本 nx 上按确定性顺序提交创生请求，返回 (新节点 id 列表, 实际 ΔE)。

        每次写入前用 cone_* 内部判据重新终审（帧首快照可能已失效）。
        budget 非 None 时逐条对湮灭预算（pairing=conserved）：
        cost = 3（A 类真面）/ k（B 类 k 边形洞）；spent + cost > budget 则**跳过**
        该候选（用 continue 而非 break，允许后续小 cost 候选入选）。
        """
        born, spent = [], 0
        for c in sorted((c for c in cands if c["type"] == CREATION),
                        key=lambda c: c["key"]):
            if not all(id(c) in accept.get(v, ()) for v in set(c["corners"])):
                continue
            cost = 3 if c["kind"] == "face" else len(c["corners"])
            if budget is not None and spent + cost > budget:
                continue
            if c["kind"] == "face":
                w = nx.cone_tri_face(*c["corners"])
            else:
                w = nx.cone_hole(list(c["corners"]))
            if w is not None:
                born.append(w)
                spent += cost
        return born, spent

    # ========================================================
    # §3.2 创生-湮灭对偶：湮灭预算（pairing=conserved 的 V- 侧，先于创生）
    # ========================================================
    def _plan_breaks(self, net):
        """只读 S_t，规划本帧的 V⁻ 边集；返回边表，len = 本帧湮灭预算 N。

        贪心：按「两端度数之和最大」优先（平局 (min id, max id) 字典序），
        维护**虚拟度数**，仅当**断后两端仍 >= dmin** 时才取该边（守卫 vdeg > dmin）。
        创生只增边，故该规划在创生后仍可行（旧边不会消失、度数只升）。
        注 1：守卫取 dmin 而非公理下限 2 —— 断后仍 >= dmin 的节点不会被 §4.3 的 prune
              删除，故配平本身不会触发悬挂分支。若只守 >= 2，N 会被推到「所有可断边」
              = O(|E|)：实测 icosa 上 N=23/30，湮灭端碾压创生端，5 帧内清空网络。
        注 2：N 因此是**结构上限**——由 dmin 与当前度数分布共同封顶，无外生比例参数。
        """
        pool = sorted(_edge_set(net),
                      key=lambda e: (-(net.deg(e[0]) + net.deg(e[1])), e))
        vdeg = {v: net.deg(v) for v in net.ids()}
        out = []
        for (u, v) in pool:
            if vdeg[u] <= self.dmin or vdeg[v] <= self.dmin:
                continue
            vdeg[u] -= 1
            vdeg[v] -= 1
            out.append((u, v))
        return out

    def _plan_dense_breaks(self, net):
        """过密边湮灭规划（vminus="dense"）。返回边表，len = 本帧湮灭条数。

        与 `_plan_breaks` 同构，只有两处不同：
          ① 入池判据：只取**两端皆为负曲率**的边（deg(u) > κ 且 deg(v) > κ），
             即整条边都落在负曲率区。这是「过密」的局部形式。
             （初版曾用「度数之和 >= 2κ」，实测过松：枢纽一旦形成，它与任何邻居的
             度数之和都超阈，90/90 条边全部入池 → 两帧清空网络。见 §7.4。）
          ② 守卫下线取**公理下限 2**（不是 dmin）：断边因而可以把端点压到 deg 2；
             若该节点随后 deg < dmin，帧末 §4.3 的修剪就真的删掉它 —— 这是
             **非互逆的删除**，正是要激活的「不完美 / 悬挂端再生（L0-A9）」。
             若守 dmin，湮灭会被守卫完全挡回互逆区，等于白改。

        注：判据（负曲率区）与创生判据（正角亏 ⇒ 缺口处）**作用在不同区域**，
        故不再是「锥化 / 修剪」那一对互逆操作 —— 这是离开自相似轨道的机制来源。
        """
        pool = sorted((e for e in _edge_set(net)
                       if net.deg(e[0]) > self.kappa and net.deg(e[1]) > self.kappa),
                      key=lambda e: (-(net.deg(e[0]) + net.deg(e[1])), e))
        vdeg = {v: net.deg(v) for v in net.ids()}
        out = []
        for (u, v) in pool:
            if vdeg[u] <= 2 or vdeg[v] <= 2:
                continue
            vdeg[u] -= 1
            vdeg[v] -= 1
            out.append((u, v))
        return out

    # ========================================================
    # §4 帧执行（双缓冲：读 S_t → 写 S_{t+1}）
    # ========================================================
    def _step(self, cands):
        """纯函数：给定候选序列产出下一状态（不改 self）。
        返回 (S_{t+1}, born, dead, broken, spent)。"""
        accept = self.resolve(cands)                 # 读 S_t
        nx = _clone(self.net)                        # 写缓冲 S_{t+1}
        e0 = self.net.E()
        if self.vminus == "dense":                   # 过密边：湮灭判据独立于创生
            plan = self._plan_dense_breaks(self.net)
            budget = len(plan) if self.pairing == "conserved" else None
        elif self.pairing == "conserved":
            plan = self._plan_breaks(self.net)       # 步骤 1：先定湮灭
            budget = len(plan)
        else:
            plan, budget = [], None
        born, spent = self.commit(nx, cands, accept, budget)   # 步骤 2：截断创生
        broken = []
        # budget is None（不配平）⇒ 湮灭独立发生 ⇒ 净 ΔE ≠ 0；否则只断前 spent 条 ⇒ ΣΔE = 0
        for (u, v) in (plan if budget is None else plan[:spent]):
            if nx.break_edge(u, v) is not None:
                broken.append((u, v))
        # §4.3 判断 + 删除：只删一层，不级联
        dead = [v for v in nx.ids() if nx.deg(v) < self.dmin]
        for v in dead:
            nx.remove_vertex(v)
        return nx, sorted(born), sorted(dead), sorted(broken), spent

    def frame(self):
        """一帧：创生 → 连接 → 变化体积 → 判断 → 删除。"""
        e0 = self.net.E()
        nx, born, dead, broken, spent = self._step(self.propose())
        self.net = nx
        self.t += 1
        return {"t": self.t, "born": len(born), "dead": len(dead),
                "broken": len(broken), "spent": spent, "dE": self.net.E() - e0}

    def puncture(self):
        """V- 单点穿刺（帧外、确定性）：删除最大度元胞 → 其 d 个面并成一个 d 边形洞。

        闭合种子下这是产生「开口」的唯一入口 —— 全部创生操作 Δχ=0 且不造洞。
        它同时打开两条此前不可达的路径：B 类（cone_hole）与删除分支。
        返回 (被删元胞, 其度数)。
        """
        ids = self.net.ids()
        if not ids:
            return None
        v = max(ids, key=lambda x: (self.net.deg(x), -x))
        d = self.net.deg(v)
        self.net.remove_vertex(v)
        return v, d

    # ========================================================
    # §4.4 / §6 审计（只读）
    # ========================================================
    def edge_face_incidence(self, cycles=None):
        """每条无向边被多少条 dart 环引用。闭合流形应恒为 2（L0-A1）。"""
        if cycles is None:
            cycles = self.net.faces()
        inc = Counter()
        for cyc in cycles:
            for (a, b) in cyc:
                inc[(a, b) if a < b else (b, a)] += 1
        return inc

    def link_type(self, v):
        """元胞 v 的 link（邻居诱导子图）类型：cycle / path / chord / disc（§4.4 判据 C2）。

        cycle = 干净内部元胞；path = 边界元胞；chord = 该处存在分离三角；
        disc = link 不连通 = 真正的非流形（领结/交叉）。
        """
        nb = sorted(self.net.rot[v])
        d = len(nb)
        if d == 0:
            return "disc"
        ld = {x: 0 for x in nb}
        la = {x: [] for x in nb}
        le = 0
        for ai in range(d):
            for bi in range(ai + 1, d):
                x, y = nb[ai], nb[bi]
                if self.net.adj(x, y):
                    le += 1
                    ld[x] += 1
                    ld[y] += 1
                    la[x].append(y)
                    la[y].append(x)
        seen = {nb[0]}
        st = [nb[0]]
        while st:
            u = st.pop()
            for w in la[u]:
                if w not in seen:
                    seen.add(w)
                    st.append(w)
        if len(seen) != d:
            return "disc"
        degs = sorted(ld.values())
        if le == d and all(x == 2 for x in degs):
            return "cycle"
        if le == d - 1 and degs.count(1) == 2 and all(x in (1, 2) for x in degs):
            return "path"
        return "chord"

    def separating_triangles(self):
        """3-团中「两个定向都不是真面」的个数（§2.4 分离三角）。

        这些正是 v1.x 判据「j,k 互邻即缝隙」会误锥化的位置。本内核从
        faces() 取候选，天然不触及它们 —— 该计数 > 0 属正常（每次锥化都会
        把原面变成分离三角），只有「被锥化」才是缺陷。
        """
        n = 0
        for i in self.net.ids():
            ni = self.net.rot[i]
            for j in ni:
                if j <= i:
                    continue
                for k in ni:
                    if k > j and self.net.adj(j, k):
                        if not (self.net.tri_is_face(i, j, k)
                                or self.net.tri_is_face(i, k, j)):
                            n += 1
        return n

    def stats(self):
        """单次全局面追踪，汇总 §4.4 四判据 + §6.1 审计所需全部计数。"""
        net = self.net
        ids = net.ids()
        V, E = net.V(), net.E()
        cyc = net.faces()
        n_iso = sum(1 for v in ids if net.deg(v) == 0)
        F = len(cyc) + n_iso          # 与 prototype audit 同口径（孤立点计一个退化面）

        cyc_id = {}
        for ci, c in enumerate(cyc):
            for d in c:
                cyc_id[d] = ci
        tris, holes = 0, []
        for c in cyc:
            vs = net.face_vertices(c)
            if len(vs) == 3 and len(set(vs)) == 3:
                tris += 1
            elif len(vs) >= 4 and len(set(vs)) == len(vs):
                holes.append(len(vs))
        inc = self.edge_face_incidence(cyc)
        # C3 无领结：同一无向边的两个 dart 必须落在两个不同的面里
        bowtie = sum(1 for (a, b) in inc
                     if (a, b) in cyc_id and (b, a) in cyc_id
                     and cyc_id[(a, b)] == cyc_id[(b, a)])
        return {
            "V": V, "E": E, "F": F, "chi": V - E + F,
            "tri": tris, "n_holes": len(holes),
            "max_hole": max(holes) if holes else 0,
            "edge_face_bad": sum(1 for e, n in inc.items() if n != 2),
            "bowtie": bowtie,
            "sep": self.separating_triangles(),
            "iso": n_iso,
            "dangling": sum(1 for v in ids if net.deg(v) < self.dmin),
            "max_deg": max((net.deg(v) for v in ids), default=0),
            "handshake": sum(net.deg(v) for v in ids) == 2 * E,
            "cap_ok": all(net.deg(v) <= self.cap for v in ids),
            "planar_ok": E <= max(3 * V - 6, 0),
            "link": dict(Counter(self.link_type(v) for v in ids)),
        }

    # ========================================================
    # §7 三探针（判定「晶子是否涌现」的读数；只读、无副作用）
    # ========================================================
    def degrees(self):
        return [self.net.deg(v) for v in self.net.ids()]

    def gini(self, xs=None):
        """度数基尼系数（0 = 完全均匀；→1 = 向少数节点集中）。

        gini = (2·Σ i·x_(i)) / (n·Σx) − (n+1)/n，x_(i) 为升序第 i 个（i 从 1 起）。
        """
        xs = sorted(self.degrees() if xs is None else xs)
        n = len(xs)
        tot = sum(xs)
        if n == 0 or tot == 0:
            return 0.0
        acc = sum((i + 1) * x for i, x in enumerate(xs))
        return (2.0 * acc) / (n * tot) - (n + 1.0) / n

    def shell_probe(self):
        """闭合壳读数（两个口径，分开报，避免把恒等式当观测量）。

        口径一（字面）：Σ(6−deg) 的读数。对**纯三角剖分闭曲面**它恒为 12
            （Σ(6−deg) = 6V − 2E = 12 + 2d，d = 3V−6−E ⇒ d=0 ⇔ 恰为 12）。
            故它只能给出「整体是否仍是闭合三角剖分」这一个 bit，不是可数的壳个数。
        口径二（可数）：deg==5 元胞的诱导子图连通分量，其中 12 点 5-正则
            = 正二十面体 = 晶子（12 顶点唯一的 5-正则平面三角剖分）。
        """
        net = self.net
        ids = net.ids()
        S = sum(6 - net.deg(v) for v in ids)
        deg5 = {v for v in ids if net.deg(v) == 5}
        seen, comps = set(), []
        for v in sorted(deg5):
            if v in seen:
                continue
            seen.add(v)
            stack, comp = [v], []
            while stack:
                u = stack.pop()
                comp.append(u)
                for w in net.rot[u]:
                    if w in deg5 and w not in seen:
                        seen.add(w)
                        stack.append(w)
            cs = set(comp)
            m = sum(1 for u in comp for w in net.rot[u] if w in cs) // 2
            reg = all(sum(1 for w in net.rot[u] if w in cs) == 5 for u in comp)
            comps.append({"n": len(comp), "m": m, "reg5": reg,
                          "icosa12": (len(comp) == 12 and m == 30 and reg)})
        sizes = sorted((c["n"] for c in comps), reverse=True)
        return {"S": S,                                   # Σ(6−deg)：纯三角剖分闭曲面恒 12
                "closed_tri": S == 12,                    # 字面口径的读数（1 bit）
                "n_comp": len(comps),                     # 度5诱导分量的个数
                "n_icosa12": sum(1 for c in comps if c["icosa12"]),  # 真·晶子壳计数
                "max_comp": sizes[0] if sizes else 0,
                "sizes": sizes[:6]}

    def state_key(self):
        """规范化状态键（exact，用于周期检测与确定性比较）。"""
        return tuple((v, tuple(r)) for v, r in sorted(self.net.rot.items()))

    def state_hash(self):
        """状态键的定长摘要（FNV-1a 64 位；仅用于打印与查表，检测仍用 state_key）。"""
        MOD = (1 << 64) - 1
        h = 1469598103934665603
        for v, r in sorted(self.net.rot.items()):
            for tok in (v, len(r)):
                h = ((h ^ (tok & MOD)) * 1099511628211) & MOD
            for w in r:
                h = ((h ^ ((w + 1) & MOD)) * 1099511628211) & MOD
        return format(h, "016x")


# ============================================================
# 自检 / 演示
# ============================================================
def _fmt(s, extra=""):
    lt = s["link"]
    return (f"V={s['V']:4d} E={s['E']:5d} F={s['F']:5d} chi={s['chi']:3d} "
            f"三角={s['tri']:5d} 洞={s['n_holes']}(max{s['max_hole']}) "
            f"边面!=2:{s['edge_face_bad']:2d} 领结:{s['bowtie']:2d} 分离:{s['sep']:5d} "
            f"悬挂={s['dangling']:2d} maxdeg={s['max_deg']:3d} "
            f"link[c{lt.get('cycle', 0)}/p{lt.get('path', 0)}"
            f"/h{lt.get('chord', 0)}/d{lt.get('disc', 0)}]{extra}")


def _invariants_ok(s):
    """§4.4 + §6.1 四判据（闭合三角剖分）：χ=2 / 边面入射=2 / 无领结 / 无 disc /
    无孤立点 / 握手 / 容量 / 平面界。"""
    return (s["chi"] == 2 and s["edge_face_bad"] == 0 and s["bowtie"] == 0
            and s["link"].get("disc", 0) == 0 and s["iso"] == 0
            and s["handshake"] and s["cap_ok"] and s["planar_ok"])


# ============================================================
# 三探针运行器（drive / pairing 的判定读数）
# ============================================================
def _cell(x, w=4):
    return "-".rjust(w) if x is None else f"{x}".rjust(w)


def _probe_row(core, rec=None):
    sh = core.shell_probe()
    degs = core.degrees()
    V, E = core.net.V(), core.net.E()
    F = len(core.net.faces()) + sum(1 for d in degs if d == 0)
    return {"t": core.t, "V": V, "E": E, "F": F, "chi": V - E + F,
            "maxdeg": max(degs, default=0), "gini": core.gini(degs),
            "S": sh["S"], "closed": sh["closed_tri"],
            "shell12": sh["n_icosa12"], "n_comp": sh["n_comp"],
            "max_comp": sh["max_comp"], "sizes": sh["sizes"],
            "hash": core.state_hash(),
            "born": None if rec is None else rec["born"],
            "dead": None if rec is None else rec["dead"],
            "broken": None if rec is None else rec["broken"],
            "dE": None if rec is None else rec["dE"]}


def run_probe(core, nframes=8, guard=4000):
    """跑三探针，返回 (逐帧记录, 周期, 预周期)。

    周期检测用 exact state_key：动力学是状态的确定性函数 ⇒ 键重复即真循环。
    """
    seen, rows = {}, []
    seen[core.state_key()] = core.t
    rows.append(_probe_row(core))
    period = pre = None
    for _ in range(nframes):
        rec = core.frame()
        key = core.state_key()
        row = _probe_row(core, rec)
        if key in seen:
            if period is None:
                pre, period = seen[key], core.t - seen[key]
            row["loop"] = f"{pre}->{core.t}"
        else:
            seen[key] = core.t
        rows.append(row)
        if core.net.V() == 0 or core.net.V() > guard:
            break
    return rows, period, pre


def _print_probe(rows):
    print("   t      V       E   chi  maxdeg    gini      S  d=3V-6-E  壳12 度5n 度5max"
          "      V+     V-    断边  netdE  状态键")
    for r in rows:
        d = 3 * r["V"] - 6 - r["E"]
        loop = ("  [循环 " + r["loop"] + "]") if "loop" in r else ""
        print(f"  {r['t']:<3d} {r['V']:6d} {r['E']:7d} {r['chi']:5d} {r['maxdeg']:6d} "
              f"{r['gini']:7.4f} {r['S']:6d} {d:8d} {r['shell12']:6d} "
              f"{r['n_comp']:5d} {r['max_comp']:7d} "
              f"{_cell(r['born'], 6)} {_cell(r['dead'], 6)} {_cell(r['broken'], 7)} "
              f"{_cell(r['dE'], 7)}  {r['hash']}{loop}")


# ============================================================
# §7.2 裸动力学探针：cap / 优先级 / 驱动**全部删除**
#   只留两条规则：① 锥化帧首全部真面（唯一 V⁺ 通道）
#                 ② 帧末修剪 deg < dmin（唯一 V⁻ 通道，一帧只删一次）
#   判定三件事：① maxdeg 增长曲线（线性/次线性/自行饱和）
#              ② 度直方图随 V 的演化——曲率被搬到哪里
#              ③ 是否反复出现同一种局部构型（度数签名的晶子候选）
#   注：不经 L0Core（无 cap、无优先级、无配平），故与 §3 协商引擎无关。
# ============================================================
_DEG_BUCKETS = ((3, 4), (4, 6), (6, 10), (10, 20), (20, 40), (40, None))


def _bucket(d):
    for (lo, hi) in _DEG_BUCKETS:
        if d >= lo and (hi is None or d < hi):
            return f"{lo}" if hi is None else f"{lo}-{hi - 1}"
    return f"<{_DEG_BUCKETS[0][0]}"


def _bucket_keys():
    """度数直方图的桶标签序列（与 _bucket 返回值一致，含 <3 兜底桶）。"""
    return ([f"<{_DEG_BUCKETS[0][0]}"]
            + [f"{lo}" if hi is None else f"{lo}-{hi - 1}"
               for (lo, hi) in _DEG_BUCKETS])


def _msig(key):
    return "+".join(f"{b}x{c}" for b, c in key)


def bare_step(net, dmin=3):
    """裸一帧：锥化全部真面 → 帧末修剪。返回 (S_{t+1}, 锥化数, 被删数)。

    读 S_t 的面迹快照，写副本 S_{t+1}；一帧只删一次，不级联。
    """
    nx = _clone(net)
    coned = 0
    for cyc in net.faces():
        vs = net.face_vertices(cyc)
        if len(vs) == 3 and len(set(vs)) == 3 and nx.cone_tri_face(*vs) is not None:
            coned += 1
    dead = [v for v in nx.ids() if nx.deg(v) < dmin]
    for v in dead:
        nx.remove_vertex(v)
    return nx, coned, len(dead)


def bare_probe(net, nframes=9, dmin=3, guard=600000):
    """裸动力学长程读数（不改传入的 net）。返回逐帧记录表。"""
    net = _clone(net)
    core_ids = tuple(net.ids())          # 初始元胞集 = 冻结核候选
    rows = []
    for t in range(nframes + 1):
        ids = net.ids()
        degs = [net.deg(v) for v in ids]
        V, E = net.V(), net.E()
        cs = set(core_ids) & set(ids)
        cdeg = {u: sum(1 for w in net.rot[u] if w in cs) for u in cs}
        motifs = Counter()
        for v in ids:
            nb = Counter(_bucket(net.deg(w)) for w in net.rot[v])
            motifs[tuple(sorted(nb.items()))] += 1
        rows.append({
            "t": t, "V": V, "E": E, "F": len(net.faces()), "maxdeg": max(degs),
            "hist": Counter(_bucket(d) for d in degs),
            "dust": sum(6 - d for d in degs if d < 6),     # 正曲率：尘埃
            "debt": sum(d - 6 for d in degs if d > 6),     # 负曲率：枢纽
            "S": sum(6 - d for d in degs),
            "core_n": len(cs), "core_m": sum(cdeg.values()) // 2,
            "core_degs": sorted(set(cdeg.values())),
            "motif_top": motifs.most_common(4), "motif_n": len(motifs)})
        if t >= nframes or V > guard:
            break
        net, _, _ = bare_step(net, dmin)
    return rows


def _print_bare(rows):
    print("  t        V        E        F  maxdeg   md×     α   曲率尘埃+  枢纽-    S"
          "  冻结核n/m/内度   构型(种类/最大类)")
    prev = None
    for r in rows:
        a = ("-" if r["t"] == 0 else
             f"{math.log(r['maxdeg'] / 5.0) / math.log(r['V'] / 12.0):.4f}")
        ratio = "-" if prev is None else f"{r['maxdeg'] / prev:.3f}"
        top = r["motif_top"][0] if r["motif_top"] else ((), 0)
        print(f"  {r['t']:<3d} {r['V']:8d} {r['E']:8d} {r['F']:8d} {r['maxdeg']:7d} "
              f"{ratio:>6} {a:>7} "
              f"{r['dust']:10d} {r['debt']:7d} {r['S']:5d} "
              f"{r['core_n']:5d}/{r['core_m']:<3d}/{str(r['core_degs']):<9} "
              f"{r['motif_n']:5d}/{top[1]:<8d} {_msig(top[0])}")
        prev = r["maxdeg"]
    keys = _bucket_keys()
    print("  度数直方图（桶下限）：")
    print("    t     " + "".join(f"{k:>9}" for k in keys))
    for r in rows:
        print(f"    {r['t']:<4d}  " + "".join(f"{r['hist'].get(k, 0):9d}" for k in keys))


# ============================================================
# §7.4 非对称动力学：vminus / vplus 打破锥化-修剪互逆
#   基线（vminus="dangling", vplus="any"）：V/E 每帧精确 ×3（纯通胀）或恒等映射。
#   本探针把两个新开关的读数摊开：σ=V/E、V+ / V-断边 / V-删点、净 ΔE、度直方图。
#   判据：V/Vprev 非常数 ⇒ 已离开自相似轨道；同帧既有断边又有删点 ⇒ 非互逆删除生效。
# ============================================================
def dyn_probe(core, nframes=8, guard=400000):
    """非对称动力学逐帧读数（core 就地演化）。返回记录表。"""
    rows = []
    for t in range(nframes + 1):
        s = core.stats()
        degs = core.degrees()
        rows.append({
            "t": core.t, "V": s["V"], "E": s["E"], "F": s["F"], "chi": s["chi"],
            "maxdeg": s["max_deg"], "sigma": (s["V"] / s["E"]) if s["E"] else 0.0,
            "hist": Counter(_bucket(d) for d in degs),
            "dust": sum(6 - d for d in degs if d < 6),
            "debt": sum(d - 6 for d in degs if d > 6),
            "inv_ok": _invariants_ok(s),
            "born": None, "broken": None, "dead": None, "dE": None})
        if t >= nframes or s["V"] == 0 or s["V"] > guard:
            break
        r = core.frame()
        rows[-1].update({"born": r["born"], "broken": r["broken"],
                         "dead": r["dead"], "dE": r["dE"]})
    return rows


def _print_dyn(rows):
    print("  t        V        E      F   χ   maxdeg    σ=V/E  V/Vprev   V+创生 V-断边 V-删点"
          " netΔE   尘埃   枢纽")
    prev = None
    for r in rows:
        ratio = "-" if not prev else f"{r['V'] / prev:.3f}"
        ev = ("-" * 30 if r["born"] is None else
              f"{r['born']:7d} {r['broken']:7d} {r['dead']:7d} {r['dE']:6d}")
        flag = "" if r["inv_ok"] else "  [WARN:不变量!]"
        print(f"  {r['t']:<3d} {r['V']:8d} {r['E']:8d} {r['F']:6d} {r['chi']:4d} "
              f"{r['maxdeg']:8d} {r['sigma']:8.4f} {ratio:>8}   {ev} "
              f"{r['dust']:6d} {r['debt']:6d}{flag}")
        prev = r["V"]
    keys = _bucket_keys()
    print("  度数直方图（桶下限）：")
    print("    t     " + "".join(f"{k:>9}" for k in keys))
    for r in rows:
        print(f"    {r['t']:<4d}  " + "".join(f"{r['hist'].get(k, 0):9d}" for k in keys))
    ratios = sorted({round(rows[i]["V"] / rows[i - 1]["V"], 3)
                     for i in range(1, len(rows)) if rows[i - 1]["V"]})
    noninverse = sum(1 for r in rows if r["broken"] and r["dead"])
    print(f"  V/Vprev 取值集 = {ratios}  （恒 {{3.0}} 或 {{2.0}} = 自相似通胀轨道；"
          f"非常数 ⇒ 已离开该轨道）")
    print(f"  同帧「既有断边又有删点」的帧数 = {noninverse} / "
          f"{sum(1 for r in rows if r['born'] is not None)}"
          f"  （> 0 ⇒ 非互逆的湮灭已生效 = L0-A9 悬挂端再生激活）")


# ============================================================
# §7.3 重连动力学：固定 V 的势函数 Φ 探针
#   动作集 = 边翻转（flip_edge，见 combinatorial_proto）。它是**对合**（自逆）
#   ——与「锥化 / 删点」互逆同源（庞加莱对偶：deg-3 ↔ 三角形）——故不选方向
#   必落周期。要箭头就得选方向，于是势函数不是可选输入，是唯一结构解。
#   势 Ψ = Σ(6−deg)²（注意 Σ(6−deg) ≡ 12 是恒等式，不是约束）：
#     mode=phi1 : Φ₁ = −Ψ，只走 ΔΨ < 0（摊开曲率）→ 预期收敛到最均匀剖分
#     mode=phi2 : Φ₂ = +Ψ，只走 ΔΨ > 0（集中曲率）→ 预期收敛到曲率集中态
#     mode=none : 不选，全翻 → 验证自逆 ⇒ 必周期
#   ΔΨ = 2(d_k+d_l−d_i−d_j)+4；ΔΨ<0 ⇒ ΔΨ ≤ −2，Ψ ≥ 12 ⇒ 必有限步终止。
#   固定 V：翻转保 V/E/F/χ ⇒ E=3V−6、F=2V−4 全程锁定。有界性由上一层给定。
# ============================================================
def _defect_clusters(net):
    """正缺陷顶点（deg < 6）诱导子图的连通分量，按 (点数, 内部边数) 降序。

    返回 [(顶点表, 内部边数, 缺陷和), ...]。
    """
    pos = {v for v in net.ids() if net.deg(v) < 6}
    seen, out = set(), []
    for v in sorted(pos):
        if v in seen:
            continue
        seen.add(v)
        st, comp = [v], []
        while st:
            u = st.pop()
            comp.append(u)
            for w in net.rot[u]:
                if w in pos and w not in seen:
                    seen.add(w)
                    st.append(w)
        cs = set(comp)
        m = sum(1 for u in comp for w in net.rot[u] if w in cs) // 2
        comp.sort()
        out.append((comp, m, sum(6 - net.deg(u) for u in comp)))
    out.sort(key=lambda c: (-len(c[0]), -c[1]))
    return out


def flip_step(net, mode="none"):
    """重连一帧：按字典序处理帧首全部边，逐条在副本上**重新复核**后才翻。

    读 S_t 的边表快照，写副本 S_{t+1}（同 §4.2 双缓冲）。返回 (S_{t+1}, 翻成数)。
    翻转不改 V/E/F/χ、不动悬挂（deg≥3 恒成立，见 flip_delta 注），故无帧末修剪。

    mode（谓词作用于 ΔΨ = flip_delta(...)[2]）：
      "none"  : 全部接受（任意 ΔΨ）
      "phi1"  : 严格下降 ΔΨ < 0            —— 单调收敛，会卡在伪局部极小
      "phi2"  : 严格上升 ΔΨ > 0            —— 用于把起点推离最优（perturb）
      "phi1n" : 非严格下降 ΔΨ <= 0（含中立 ΔΨ == 0）—— 可达 Ψ 下界，但在
                下界平台上停不下来（中立翻转 ⇒ 极限环），故须配 recon_settle
    """
    nx = _clone(net)
    done = 0
    for (i, j) in _edge_set(net):
        r = nx.flip_delta(i, j)
        if r is None:
            continue
        if mode == "phi1" and not r[2] < 0:
            continue
        if mode == "phi2" and not r[2] > 0:
            continue
        if mode == "phi1n" and not r[2] <= 0:
            continue
        if nx.flip_edge(i, j) is not None:
            done += 1
    return nx, done


def _flip_row(net, t):
    degs = [net.deg(v) for v in net.ids()]
    cl = _defect_clusters(net)
    if cl:
        comp, cm, cd = cl[0]
        cs = set(comp)
        reg5 = all(sum(1 for w in net.rot[u] if w in cs) == 5 for u in comp)
        cn, ic12 = len(comp), (len(comp) == 12 and cm == 30 and cd == 12 and reg5)
    else:
        cn, cm, cd, ic12 = 0, 0, 0, False
    return {"t": t, "V": net.V(), "E": net.E(), "F": len(net.faces()),
            "maxdeg": max(degs, default=0),
            "psi": sum((6 - d) ** 2 for d in degs),
            "dust": sum(6 - d for d in degs if d < 6),
            "debt": sum(d - 6 for d in degs if d > 6),
            "hist": Counter(degs),
            "nc": len(cl), "cn": cn, "cm": cm, "cd": cd, "icosa12": ic12}


def recon_probe(net, mode="phi1", nframes=40, perturb=0):
    """重连长程读数（不改传入的 net）。返回 (rows, 收敛帧, 周期)。

    perturb：起步前先做 perturb 次 Φ₂ 步，把起点推离 Φ₁ 最优。V=12 时必需——
    正二十面体自身就是 Φ₁ 全局最优（所有缺陷=1 ⇒ Ψ=12 取到下界），无改进翻转。
    """
    net = _clone(net)
    for _ in range(perturb):
        net, _ = flip_step(net, "phi2")
    rows, seen, conv, period = [], {}, None, None
    for t in range(nframes + 1):
        rows.append(_flip_row(net, t))
        key = tuple((v, tuple(r)) for v, r in sorted(net.rot.items()))
        if key in seen and period is None:
            period = (seen[key], t)
        seen.setdefault(key, t)
        if t >= nframes:
            break
        nxt, done = flip_step(net, mode)
        net = nxt
        if done == 0:
            conv = t + 1
            rows.append(_flip_row(net, t + 1))
            break
    return rows, conv, period


def recon_settle(net, mode="phi1n", perturb=0, cap=4096):
    """把 net 跑到吸引子并**停机**，返回 (settled_net, info)。不改传入的 net。

    为什么需要它：严格 Φ₁（ΔΨ<0）单调但会卡在伪局部极小；非严格 Φ₁（ΔΨ<=0）
    能到达 Ψ 下界平台，却因平台上的中立翻转（ΔΨ==0）而**永不停机**——这不是
    无结构漂移，而是周期 2/3 的**极限环**。故停机判据必须是「固定点或周期命中」
    二者之一，且返回**环上/路径上 Ψ 最小的构型**（极限环没有"最后一帧"，返回
    最优者才确定；并列取最早出现，结果与扫描顺序无关）。

    perturb：起步前先走 perturb 次 Φ₂（ΔΨ>0），把起点推离 Φ₁ 最优。V=12 时必需
    ——正二十面体自身即 Φ₁ 全局最优（Ψ=12 取下界），无改进翻转。
    info 字段：reason ∈ {halting, cycle, cap}；halt_t（固定点帧号）；
              period=(t0,t1) 与 cycle_len（周期命中）；best_t / best_psi / frames。
    """
    # --- S1 推离最优：先走 perturb 次 Φ₂（ΔΨ>0） ---
    nx = _clone(net)
    for _ in range(perturb):
        nx, _ = flip_step(nx, "phi2")

    # --- S2 初始化 ---
    seen = {_net_key(nx): 0}                 # 状态键 -> 首次出现的帧号
    best = _clone(nx)
    best_psi = _psi(nx)
    best_t = 0
    frames = 0
    result = {"reason": None, "halt_t": None, "period": None,
              "cycle_len": None, "best_t": 0, "best_psi": best_psi, "frames": 0}

    # --- S3 主循环：帧号 t 从 1 到 cap ---
    for t in range(1, cap + 1):
        nxt, done = flip_step(nx, mode)
        if done == 0:
            # 3a 局部极小：无任何可接受翻转 => 停机（固定点）
            result["reason"] = "halting"
            result["halt_t"] = t
            break
        nx = nxt
        frames = t
        k = _net_key(nx)
        if k in seen:
            # 3b 命中周期：状态键重复 => 已进入吸引子（极限环）
            result["reason"] = "cycle"
            result["period"] = (seen[k], t)
            result["cycle_len"] = t - seen[k]
            break
        seen[k] = t
        s = _psi(nx)
        # 3c 记录最优：Ψ 最小者；并列取**最早出现**的帧
        if s < best_psi:
            best_psi = s
            best_t = t
            best = _clone(nx)
    else:
        # 3d cap 用尽仍未停机
        result["reason"] = "cap"

    # --- S4 组装返回 ---
    result["best_t"] = best_t
    result["best_psi"] = best_psi
    result["frames"] = frames
    return best, result


def seed_at_V(V, base="icosa"):
    """构造 V ≥ 12 的三角剖分起点：由 icosa 逐次锥化首个真面，每次 +1 元胞。"""
    net = RotNet(SEEDS[base]())
    while net.V() < V:
        for cyc in net.faces():
            vs = net.face_vertices(cyc)
            if (len(vs) == 3 and len(set(vs)) == 3
                    and net.cone_tri_face(*vs) is not None):
                break
        else:
            break
    return net


def _print_flip(rows, conv, period, tag):
    print(f"  {tag}")
    print("   t      V      E      F  maxdeg   Ψ=Σ(6-d)²  dust  debt"
          "   正缺陷团簇 个数/最大点数/内部边/缺陷和  正二十面体  度直方图")
    for r in rows:
        print(f"  {r['t']:<3d} {r['V']:6d} {r['E']:6d} {r['F']:6d} {r['maxdeg']:7d} "
              f"{r['psi']:10d} {r['dust']:5d} {r['debt']:5d}   "
              f"{r['nc']:3d}/{r['cn']:<4d}/{r['cm']:<4d}/{r['cd']:<4d}    "
              f"{'是' if r['icosa12'] else '否':<6s}  "
              + " ".join(f"{k}:{v}" for k, v in sorted(r["hist"].items())))
    print(f"  收敛：{'第 ' + str(conv) + ' 帧后无可接受翻转（终态 Ψ=' + str(rows[-1]['psi']) + '）' if conv is not None else '未在 ' + str(rows[-1]['t']) + ' 帧内收敛'}")
    print(f"  周期：{'状态键 t=' + str(period[0]) + ' 与 t=' + str(period[1]) + ' 重复 → 周期=' + str(period[1] - period[0]) + '，预周期=' + str(period[0]) if period is not None else '无重复状态键'}")


# ============================================================
# §7.5 稳定结构判据（产物读数；只读）
#   §7.3 判决把「闭合 12 球作为**独立连通分量**」立为可判定目标。判据的唯一严格
#   形式是**分量级**：某连通分量恰 V=12，且其内每个元胞的**全局** deg == 5。
#     依据：分量与外界无边 ⇒ 该 12 点诱导子图必 5-正则；12 顶点唯一的 5-正则
#           平面三角剖分 = 正二十面体 ⇒ 该分量必是独立连通分量。
#   （旧口径 find_crystallites 只按「deg==5 诱导块点数 >= 12」列，不要求正则，
#     V=14/16 会列出假阳性块。）
#   缺口（洞）= k >= 4 的面环，k = 边数；缺口**方向** n̂ 属投影层（L1/L2 坐标）
#   量，L0 无坐标，本节不做。
#   L0-A11（分量数不增）由本节 product_probe 逐帧取证。**适用域（实测收紧）**：
#     仅默认 dmin=3（三角剖分下限，悬挂修剪不作用于低度割点）下恒成立；
#     已登记 3 类反例（详见 docs §9 断言表 L0-A11）——① dmin=2 ∧ pairing=conserved
#     （低阈值开放修剪 + 强制断边可割断桥边/割点）；② 开放种子 + 大 dmin 剪除
#     （可把曲面剪碎，patch cap=6 dmin=5 → n_comp 1→1→2）；③ 开放种子 + ① 组合。
# ============================================================
def components(net):
    """全图连通分量（无向，按 rot 邻接）。只读、确定性。

    返回 list[list[int]]：每个分量是**升序 id 列表**；分量之间按**首元素升序**
    排列（net.ids() 已升序 ⇒ 每分量第一个被访问的点必是它的最小 id）。
    """
    # --- S1 初始化 ---
    seen = set()
    out = []
    # --- S2 逐点泛洪（ids() 已升序，故 out 天然按分量最小 id 排序）---
    for v in net.ids():
        if v in seen:
            continue
        seen.add(v)
        stack, comp = [v], []
        while stack:
            u = stack.pop()
            comp.append(u)
            for w in sorted(net.rot[u]):        # 必须 sorted，保证确定性
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        out.append(sorted(comp))
    # --- S3 返回 ---
    return out


def shells(net):
    """分量级「闭合 12 壳」判定（判据收紧口径）。只读、确定性。

    判据：某个连通分量恰为 12 个元胞，且该分量内**每个**元胞全局 deg == 5。
    依据：分量与外界无边 ⇒ 5-正则 ⇒ 12 顶点唯一的 5-正则平面三角剖分 = 正二十面体
         ⇒ 该分量必是**独立连通分量**（不可能有边伸向 deg≠5 的邻居）。

    返回 {"n_shells": 命中壳个数,
          "shells": [{"ids": 分量升序 id, "V": 12, "E": 30}, ...],
          "n_comp": 全图连通分量数,
          "sizes": 各分量点数降序（最多前 6）}
    """
    # --- S1 取全图连通分量 ---
    comps = components(net)
    # --- S2 逐分量判据（V=12 且全 deg==5 ⇒ 5-正则独立壳）---
    found = []
    for comp in comps:
        if len(comp) == 12 and all(net.deg(v) == 5 for v in comp):
            m = sum(len(net.rot[v]) for v in comp) // 2
            found.append({"ids": comp, "V": len(comp), "E": m})
    # --- S3 组装 ---
    sizes = sorted((len(c) for c in comps), reverse=True)
    return {"n_shells": len(found), "shells": found,
            "n_comp": len(comps), "sizes": sizes[:6]}


def gap_report(net):
    """缺口（洞）清单：k >= 4 的面环即开口，k 为其边数。只读、确定性。

    对每个洞报出 k 与**有序边界环**（面环顶点序列）；并给 k 的分布直方图。
    口径与 L0Core.stats() 的洞口径一致（len(vs)>=4 且顶点互异）。
    注：缺口**方向** n̂（GT-032/G4）是几何量 —— 需要 L1/L2 坐标，L0 无坐标，不做。

    返回 {"n_holes", "max_hole"（无洞为 0）, "dist": {k: 个数}, "holes": [{"k","vs"}]}
    """
    # --- S1 扫描全部面环 ---
    holes, dist = [], Counter()
    for cyc in net.faces():
        vs = net.face_vertices(cyc)
        if len(vs) >= 4 and len(set(vs)) == len(vs):
            holes.append({"k": len(vs), "vs": list(vs)})
            dist[len(vs)] += 1
    # --- S2 确定性排序（按 (k, 边界环) 升序）---
    holes.sort(key=lambda h: (h["k"], h["vs"]))
    # --- S3 组装（空输入守卫：max 带 default）---
    return {"n_holes": len(holes),
            "max_hole": max((h["k"] for h in holes), default=0),
            "dist": dict(sorted(dist.items())),
            "holes": holes}


def _union_nets(nets):
    """把若干旋转系统并成一个**不连通**的旋转系统（各分量 id 依次平移，互不干涉）。

    用于构造多分量种子（如 icosa ⊔ tetra = 12 壳 + 四面体两个独立分量）。
    这是「独立 12 壳作为分量」的唯一可达构造入口 —— 单连通种子下分量数不可增
    （L0-A11），故独立壳只可能是初始条件，不能自发涌现。
    """
    # --- S1 初始化 ---
    out = RotNet({})
    off = 0
    # --- S2 逐网平移并入 ---
    for n in nets:
        for v in sorted(n.rot):
            out.rot[v + off] = [w + off for w in n.rot[v]]
        if n.rot:
            off = max(v + off for v in n.rot) + 1        # 下一网的首 id
    # --- S3 定 id 计数器（防止新 id 与既有 id 撞车）---
    out.nid = off
    return out


def product_probe(core, nframes=0):
    """稳定结构读数（core 就地演化；不改调用语义）。返回 (逐帧记录, 守恒判定)。

    rows[0] = 起点读数，随后每帧追加一条；conserved 给「分量数不增」的取证。
    """
    def _row():
        cp = components(core.net)
        sh = shells(core.net)
        gp = gap_report(core.net)
        return {"t": core.t, "V": core.net.V(), "E": core.net.E(),
                "n_comp": len(cp), "sizes": [len(c) for c in cp],
                "n_shells": sh["n_shells"],
                "shell_ids": [s["ids"] for s in sh["shells"]],
                "n_holes": gp["n_holes"], "max_hole": gp["max_hole"],
                "dist": gp["dist"]}

    rows = [_row()]
    for _ in range(nframes):
        if core.net.V() == 0:
            break
        core.frame()
        rows.append(_row())
    ns = [r["n_comp"] for r in rows]
    conserved = {"n0": ns[0], "n_end": ns[-1], "n_min": min(ns), "n_max": max(ns),
                 "never_up": all(ns[i] <= ns[i - 1] for i in range(1, len(ns))),
                 "frames": len(rows) - 1}
    return rows, conserved


def _print_product(rows, conserved):
    print("   t      V      E  分量数  分量点数(降序)   壳12  壳内id"
          "                     洞数  最大洞  k分布")
    for r in rows:
        sizes = ",".join(str(x) for x in sorted(r["sizes"], reverse=True)[:6])
        dist = ",".join(f"{k}x{v}" for k, v in sorted(r["dist"].items())) or "-"
        sh = r["shell_ids"]
        shids = (";".join("[" + ",".join(str(x) for x in s) + "]" for s in sh)
                 if sh else "-")
        print(f"  {r['t']:<3d} {r['V']:6d} {r['E']:6d} {r['n_comp']:6d}  "
              f"{sizes:<14s} {r['n_shells']:5d}  {shids:<28s} "
              f"{r['n_holes']:5d} {r['max_hole']:7d}  {dist}")
    print(f"  分量数守恒（L0-A11）：起点 {conserved['n0']} → 终点 {conserved['n_end']}，"
          f"全程 min/max = {conserved['n_min']}/{conserved['n_max']}，"
          f"逐帧不增 = {conserved['never_up']}（{conserved['frames']} 帧）")


def selftest():
    print("=" * 78)
    print("[自检] L0 内核：种子不变量 / 帧不变量 / 确定性 / 顺序无关")
    print("=" * 78)
    ok = True

    # 1. 种子计数
    for name, want in (("tetra", (4, 6, 4)), ("icosa", (12, 30, 20))):
        core = L0Core(RotNet(SEEDS[name]()))
        s = core.stats()
        got = (s["V"], s["E"], s["F"])
        flag = got == want
        ok &= flag
        print(f"  [{ 'OK' if flag else 'FAIL'}] 种子 {name:6s} (V,E,F)={got} 期望 {want}")

    # 2. 帧不变量：χ=2 / 握手 / 边面入射=2 / 无领结 / 无 disc / 容量 / 平面界
    #    注：锥化全部真面 → 面数每帧 ×3（"填补空隙→更多空隙"的驱动力），故帧数取小。
    for name in ("tetra", "icosa"):
        core = L0Core(RotNet(SEEDS[name]()), cap=64)
        inv_ok = True
        for _ in range(4):
            core.frame()
            s = core.stats()
            if not _invariants_ok(s):
                inv_ok = False
                print(f"  [FAIL] {name} 帧 {core.t} 不变量破坏: {s}")
        ok &= inv_ok
        print(f"  [{'OK' if inv_ok else 'FAIL'}] {name} 4 帧：{_fmt(core.stats())}")

    # 3. 确定性：两个独立内核逐帧一致
    a = L0Core(RotNet(SEEDS["tetra"]()))
    b = L0Core(RotNet(SEEDS["tetra"]()))
    det = True
    for _ in range(4):
        a.frame(); b.frame()
        if _snapshot(a.net) != _snapshot(b.net):
            det = False
            break
    ok &= det
    print(f"  [{'OK' if det else 'FAIL'}] 确定性：两次运行逐帧状态一致")

    # 4. 顺序无关：候选反序后结果一致
    c = L0Core(RotNet(SEEDS["tetra"]()))
    d = L0Core(RotNet(SEEDS["tetra"]()))
    order_ok = True
    for _ in range(4):
        nx1 = c._step(c.propose())[0]
        nx2 = d._step(list(reversed(d.propose())))[0]  # 人为反序
        if _snapshot(nx1) != _snapshot(nx2):
            order_ok = False
            break
        c.net, d.net = nx1, nx2
        c.t += 1; d.t += 1
    ok &= order_ok
    print(f"  [{'OK' if order_ok else 'FAIL'}] 顺序无关：候选反序结果一致")

    # 5. 容量预算生效：小 cap 下协商预算真正约束生长，且 deg 不超过 cap
    e = L0Core(RotNet(SEEDS["tetra"]()), cap=5)
    for _ in range(6):
        e.frame()
    s = e.stats()
    cap_ok = s["cap_ok"]
    ok &= cap_ok
    print(f"  [{'OK' if cap_ok else 'FAIL'}] 容量预算：tetra cap=5 六帧 {_fmt(s)}")

    # 6. 洞路径：V- 穿刺 → B 类（cone_hole）→ 开口闭合。这是 cone_hole 的唯一入口，
    #    因为四个保 χ 操作中只有 remove_vertex 会造洞（Δd = deg-3 > 0）。
    f = L0Core(RotNet(SEEDS["icosa"]()))
    v0, d0 = f.puncture()
    s0 = f.stats()
    hole_ok = (s0["n_holes"] == 1 and s0["max_hole"] == d0
               and s0["tri"] == 20 - d0 and _invariants_ok(s0))
    for _ in range(2):
        f.frame()
    s = f.stats()
    closed_ok = s["n_holes"] == 0 and _invariants_ok(s)
    ok &= hole_ok and closed_ok
    print(f"  [{'OK' if hole_ok else 'FAIL'}] 穿刺 V-：删元胞 {v0}(deg={d0}) → "
          f"洞 {s0['max_hole']} 边形/三角 {s0['tri']}/chi={s0['chi']}")
    print(f"  [{'OK' if closed_ok else 'FAIL'}] 洞锥化 B 类：两帧后开口闭合 {_fmt(s)}")

    # 7. L0-A9（悬挂修剪后必再生悬挂）：基线 (vminus=dangling) 休眠——纯锥化只增度，
    #    无边可降到 dmin 之下；采纳的非对称湮灭 (vminus=dense) 激活——断过密边可把
    #    端点压到 dmin 之下 → prune 再生悬挂端。判据：基线累计 V-==0 且 dense 累计 V->0。
    g = L0Core(RotNet(SEEDS["tetra"]()))
    fired = 0
    for _ in range(4):
        fired += g.frame()["dead"]
    gd = L0Core(RotNet(SEEDS["icosa"]()), cap=12, dmin=3,
                vminus="dense", vplus="any")
    fired_d = 0
    for _ in range(10):
        fired_d += gd.frame()["dead"]
    a9_ok = (fired == 0 and fired_d > 0)
    ok &= a9_ok
    print(f"  [{'OK' if a9_ok else 'FAIL'}] L0-A9：基线 dangling 四帧 V-={fired}（休眠）；"
          f"采纳动力学 dense 十帧 V-={fired_d}（激活 ⇒ 悬挂端再生）")

    # 8. 开关（drive=saturate / pairing=conserved）：预算真正约束创生 + 确定性 + χ 保持
    #    断言：① conserved 下必须真的断边（非空转）；② 断边数 == 创生 ΔE（净 ΔE = 0）；
    #          ③ conserved 的创生数必须少于 none（预算真的在截断）；④ 多帧后 V 仍 > 0
    #          （配平守卫取 dmin，故不能触发 prune 自清空）；⑤ 确定性；⑥ χ 一并打印。
    #    注：断边若落在「两个 dart 同面」的边上（图论割边），会使 χ → χ+2，
    #        故此处把 χ 一并打印出来（用 PASS/BREAK 标记），不掩盖。
    h1 = L0Core(RotNet(SEEDS["icosa"]()), cap=12, dmin=3,
                drive="saturate", pairing="conserved")
    h0 = L0Core(RotNet(SEEDS["icosa"]()), cap=12, dmin=3,
                drive="slack", pairing="none")
    h2 = L0Core(RotNet(SEEDS["icosa"]()), cap=12, dmin=3,
                drive="saturate", pairing="conserved")
    r1 = h1._step(h1.propose())
    r0 = h0._step(h0.propose())
    br1, sp1, bo1 = len(r1[3]), r1[4], len(r1[1])
    br0, bo0 = len(r0[3]), len(r0[1])
    pair_ok = br1 > 0 and br0 == 0 and br1 == sp1 and bo1 < bo0
    dead3 = 0
    for _ in range(3):
        dead3 += h1.frame()["dead"]
        h2.frame()
    sw_det = _snapshot(h1.net) == _snapshot(h2.net)
    surv_ok = h1.net.V() > 0          # 守卫取 dmin ⇒ 配平不得触发 prune 清空
    ok &= pair_ok and sw_det and surv_ok
    print(f"  [{'OK' if pair_ok else 'FAIL'}] 开关 pairing：conserved 首帧 "
          f"创生 {bo1} 个/ΔE={sp1}，断边 {br1}（须 ==ΔE）；none 创生 {bo0} 断边 {br0}")
    print(f"  [{'OK' if surv_ok else 'FAIL'}] 配平不清空：三帧后 V={h1.net.V()}（须 >0）、"
          f"累计 V-={dead3}")
    print(f"  [{'OK' if sw_det else 'FAIL'}] 开关下确定性（saturate+conserved 三帧逐位一致）"
          f" | χ={h1.stats()['chi']} "
          f"{'PASS(闭合流形)' if _invariants_ok(h1.stats()) else 'BREAK(见 χ/边面/领结/disc)'}")

    # 9. 采纳的非对称动力学（vminus=dense vplus=any）：① 离开自相似通胀轨道
    #    —— 基线 V/Vprev 恒 {3}，采纳配置 V/Vprev 非常数；② 确定性逐帧一致；
    #    ③ cap_ok 恒真（守护 X1：cone_hole 新点 deg=k 必须 ≤ cap）。
    x1 = L0Core(RotNet(SEEDS["icosa"]()), cap=12, dmin=3,
                vminus="dense", vplus="any")
    x2 = L0Core(RotNet(SEEDS["icosa"]()), cap=12, dmin=3,
                vminus="dense", vplus="any")
    vs, det2, cap_ok2 = [x1.net.V()], True, True
    for _ in range(10):
        x1.frame(); x2.frame()
        vs.append(x1.net.V())
        cap_ok2 &= x1.stats()["cap_ok"]
        if _snapshot(x1.net) != _snapshot(x2.net):
            det2 = False
    ratios = sorted({round(vs[i] / vs[i - 1], 3)
                     for i in range(1, len(vs)) if vs[i - 1]})
    orbit_ok = (len(ratios) > 1) and det2 and cap_ok2
    ok &= orbit_ok
    print(f"  [{'OK' if orbit_ok else 'FAIL'}] 采纳动力学 dense：V/Vprev={ratios}"
          f"（非常数 ⇒ 离轨）| 确定性={det2} | cap_ok={cap_ok2} | 十帧后 V={vs[-1]}")

    # 10. 重连动力学（flip_step 的 phi1n + recon_settle 平台停机）：动作集唯一地是边
    #     翻转（对合，自逆），保 V/E/F/χ。核心难点是「非严格下降 ΔΨ<=0 能达 Ψ 下界，
    #     但下界平台上存在中立翻转 ⇒ 永不停机」——实为周期 2/3/6 的极限环，故停机
    #     判据 = 固定点或周期命中，且返回环上/路径上 Ψ 最小的构型。
    #     A/B/C 校验 phi1n/phi2 谓词；D/E 校验「达下界 + 周期停机 + 返回最优」。
    fa, da = flip_step(RotNet(SEEDS["tetra"]()), "phi1n")
    fb, db = flip_step(RotNet(SEEDS["icosa"]()), "phi1n")
    fc, dc = flip_step(RotNet(SEEDS["icosa"]()), "phi2")
    a_ok = (da == 0 and (fa.V(), fa.E(), len(fa.faces())) == (4, 6, 4)
            and _psi(fa) == 36)
    b_ok = (db == 0 and (fb.V(), fb.E(), len(fb.faces())) == (12, 30, 20)
            and _psi(fb) == 12)
    c_ok = (dc > 0 and (fc.V(), fc.E(), len(fc.faces())) == (12, 30, 20)
            and _psi(fc) > 12)
    ok &= a_ok and b_ok and c_ok
    print(f"  [{'OK' if a_ok else 'FAIL'}] 重连谓词 A：tetra phi1n done={da} "
          f"VEF=(4,6,4) Ψ=36（K4 无可翻边）")
    print(f"  [{'OK' if b_ok else 'FAIL'}] 重连谓词 B：icosa phi1n done={db} "
          f"Ψ={_psi(fb)}（已是下界，无可接受翻转）")
    print(f"  [{'OK' if c_ok else 'FAIL'}] 重连谓词 C：icosa phi2 done={dc} "
          f"Ψ={_psi(fc)}（推离下界）")

    fd, infod = recon_settle(seed_at_V(16), "phi1n", perturb=3)
    fe, infoe = recon_settle(seed_at_V(42), "phi1n", perturb=3)
    d_ok = (_psi(fd) == 12 and (fd.V(), fd.E(), len(fd.faces())) == (16, 42, 28)
            and infod["reason"] == "cycle" and infod["cycle_len"] == 2)
    e_ok = _psi(fe) == 12 and infoe["reason"] == "cycle"
    ok &= d_ok and e_ok
    print(f"  [{'OK' if d_ok else 'FAIL'}] 平台停机 D：V=16 Ψ={_psi(fd)}（下界 12）"
          f" reason={infod['reason']} cycle_len={infod['cycle_len']} best_t={infod['best_t']}")
    print(f"  [{'OK' if e_ok else 'FAIL'}] 平台停机 E：V=42 Ψ={_psi(fe)}（下界 12）"
          f" reason={infoe['reason']} period={infoe['period']} best_t={infoe['best_t']}")

    # 11. 稳定结构判据（阶段六②）：① 分量级 12 壳判据（V=12 且全 deg==5 ⇒ 5-正则
    #     独立分量）② 缺口清单（k + 有序边界环 + k 分布）③ 多分量构造 _union_nets
    #     ④ L0-A11 分量数不增（多分量种子 + cap=κ−1=5 饱和 ⇒ 逐帧不增）。
    u = _union_nets([RotNet(SEEDS["icosa"]()), RotNet(SEEDS["tetra"]())])
    cu = components(u)
    sh_ic = shells(RotNet(SEEDS["icosa"]()))
    sh_t = shells(RotNet(SEEDS["tetra"]()))
    sh_u = shells(u)
    ic_ok = (sh_ic["n_shells"] == 1 and sh_ic["n_comp"] == 1
             and sh_ic["shells"][0]["V"] == 12 and sh_ic["shells"][0]["E"] == 30
             and sh_ic["shells"][0]["ids"] == list(range(12)))
    t_ok = sh_t["n_shells"] == 0
    u_ok = (u.V() == 16 and len(cu) == 2 and cu[0] == list(range(12))
            and cu[1] == [12, 13, 14, 15]
            and sh_u["n_comp"] == 2 and sh_u["n_shells"] == 1
            and sh_u["shells"][0]["ids"] == list(range(12))
            and sh_u["sizes"] == [12, 4])
    g0 = gap_report(RotNet(SEEDS["icosa"]()))
    gp = L0Core(RotNet(SEEDS["icosa"]()))
    gp.puncture()
    g1 = gap_report(gp.net)
    gap_ok = (g0["n_holes"] == 0 and g0["max_hole"] == 0 and g0["dist"] == {}
              and g1["n_holes"] == 1 and g1["max_hole"] == 5 and g1["dist"] == {5: 1}
              and len(g1["holes"][0]["vs"]) == 5)
    pc = L0Core(_union_nets([RotNet(SEEDS["icosa"]()), RotNet(SEEDS["tetra"]())]),
                cap=5, dmin=3, vminus="dense", vplus="any")
    rows, cons = product_probe(pc, 21)
    cons_ok = (cons["n0"] == 2 and cons["never_up"] and cons["n_max"] == 2)
    ok &= ic_ok and t_ok and u_ok and gap_ok and cons_ok
    print(f"  [{'OK' if ic_ok else 'FAIL'}] 12 壳判据：icosa n_shells="
          f"{sh_ic['n_shells']} V/E={sh_ic['shells'][0]['V']}/{sh_ic['shells'][0]['E']} "
          f"n_comp={sh_ic['n_comp']} ids=[0..11]")
    print(f"  [{'OK' if t_ok else 'FAIL'}] 12 壳判据：tetra n_shells={sh_t['n_shells']}"
          f"（不命中，非 12 点）")
    print(f"  [{'OK' if u_ok else 'FAIL'}] 多分量：icosa⊔tetra V={u.V()} n_comp="
          f"{sh_u['n_comp']} n_shells={sh_u['n_shells']} sizes={sh_u['sizes']} "
          f"（壳恒为原始分量 0..11）")
    print(f"  [{'OK' if gap_ok else 'FAIL'}] 缺口清单：icosa n_holes={g0['n_holes']}；"
          f"穿刺后 n_holes={g1['n_holes']} k={g1['max_hole']} dist={g1['dist']}")
    print(f"  [{'OK' if cons_ok else 'FAIL'}] L0-A11 分量数不增：icosa⊔tetra cap=5 "
          f"{cons['frames']} 帧 n_comp {cons['n0']}→{cons['n_end']}"
          f"（min/max={cons['n_min']}/{cons['n_max']}，逐帧不增={cons['never_up']}）")

    print("-" * 78)
    print(f"自检总体: {'全部通过' if ok else '存在失败项'}")
    return ok


def main():
    kv = {}
    for a in sys.argv[1:]:
        if "=" in a:
            k, v = a.split("=", 1)
            kv[k] = v
    if kv.get("selftest", "0") == "1":
        selftest()
        return

    seed = kv.get("seed", "tetra")
    nframes = int(kv.get("nframes", 6))
    cap = int(kv.get("cap", 64))
    dmin = int(kv.get("dmin", 3))
    drive = kv.get("drive", "slack")
    pairing = kv.get("pairing", "none")
    puncture = kv.get("puncture", "0") == "1"
    probe = kv.get("probe", "0") == "1"
    bare = kv.get("bare", "0") == "1"
    dyn = kv.get("dyn", "0") == "1"
    product = kv.get("product", "0") == "1"
    extra = kv.get("extra", "")          # 逗号分隔：与 seed **不连通**地并成多分量种子
    vminus = kv.get("vminus", "dense" if dyn else "dangling")
    vplus = kv.get("vplus", "any")
    kappa = int(kv.get("kappa", 6))
    names = [seed] + ([s for s in extra.split(",") if s] if extra else [])
    for nm in names:
        if nm not in SEEDS:
            raise SystemExit(f"未知种子 {nm}，可选 {sorted(SEEDS)}")

    def _seed_net():
        """seed（+ extra）→ 旋转系统；extra 非空时用 _union_nets 并成多分量。"""
        nets = [RotNet(SEEDS[nm]()) for nm in names]
        return nets[0] if len(nets) == 1 else _union_nets(nets)

    if drive not in ("slack", "saturate"):
        raise SystemExit(f"未知 drive={drive}，可选 slack | saturate")
    if pairing not in ("none", "conserved"):
        raise SystemExit(f"未知 pairing={pairing}，可选 none | conserved")

    if kv.get("recon", "0") == "1":
        mode = kv.get("mode", "phi1")
        if mode not in ("none", "phi1", "phi2", "phi1n"):
            raise SystemExit(f"未知 mode={mode}，可选 none | phi1 | phi2 | phi1n")
        perturb = int(kv.get("perturb", 0))
        vt = int(kv.get("V", 0))
        base = seed_at_V(vt) if vt else RotNet(SEEDS[seed]())
        if kv.get("settle", "0") == "1":
            settled, info = recon_settle(base, mode, perturb)
            print("=" * 78)
            print(f"[L0 重连停机] V={base.V()} mode={mode} perturb={perturb} "
                  f"—— 固定 V；动作集 = 边翻转（对合，自逆）")
            print("=" * 78)
            r0 = _flip_row(base, 0)
            print(f"  起点 V={r0['V']} E={r0['E']} F={r0['F']} "
                  f"χ={r0['V'] - r0['E'] + r0['F']} Ψ={r0['psi']} maxdeg={r0['maxdeg']} "
                  + " ".join(f"{k}:{v}" for k, v in sorted(r0["hist"].items())))
            print(f"  停机：reason={info['reason']} halt_t={info['halt_t']} "
                  f"period={info['period']} cycle_len={info['cycle_len']} "
                  f"best_t={info['best_t']} best_Ψ={info['best_psi']} "
                  f"frames={info['frames']}")
            print(f"  终态 Ψ={_psi(settled)}（下界 ⌈144/V⌉="
                  f"{-(-144 // settled.V())}）V={settled.V()} E={settled.E()} "
                  f"F={len(settled.faces())}")
            return
        print("=" * 78)
        print(f"[L0 重连动力学] V={base.V()} mode={mode} nframes={nframes} "
              f"perturb={perturb} —— 固定 V；动作集 = 边翻转（对合，自逆）")
        print("=" * 78)
        r0 = _flip_row(base, 0)
        print(f"  起点 V={r0['V']} E={r0['E']} F={r0['F']} "
              f"χ={r0['V'] - r0['E'] + r0['F']} Ψ={r0['psi']} maxdeg={r0['maxdeg']} "
              + " ".join(f"{k}:{v}" for k, v in sorted(r0["hist"].items())))
        rows, conv, period = recon_probe(base, mode, nframes, perturb)
        _print_flip(rows, conv, period, f"mode={mode} perturb={perturb}")
        return

    core = L0Core(_seed_net(), cap=cap, dmin=dmin,
                  drive=drive, pairing=pairing,
                  vminus=vminus, vplus=vplus, kappa=kappa)
    if dyn:
        adopted = (vminus == "dense" and vplus == "any")
        tag = "采纳配置" if adopted else "对照"
        vminus_note = ("断两端皆负曲率的过密边" if vminus == "dense"
                       else "仅悬挂修剪（基线）")
        vplus_note = "创生不抑制" if vplus == "any" else "创生抑制于正角亏处"
        print("=" * 78)
        print(f"[L0 非对称动力学·{tag}] seed={seed} cap={cap} dmin={dmin} "
              f"drive={drive} pairing={pairing} nframes={nframes}")
        print(f"                  vminus={vminus}（{vminus_note}）"
              f" vplus={vplus}（{vplus_note}）κ={kappa}")
        print("=" * 78)
        _print_dyn(dyn_probe(core, nframes))
        return
    if bare:
        print("=" * 78)
        print(f"[L0 裸动力学] seed={seed} dmin={dmin} nframes={nframes} "
              f"—— cap / 优先级 / 驱动已删除：只留 锥化所有真面 + 帧末修剪")
        print("=" * 78)
        _print_bare(bare_probe(core.net, nframes, dmin))
        return
    if product:
        print("=" * 78)
        print(f"[L0 稳定结构] seed={seed} extra={extra or '-'} cap={cap} dmin={dmin} "
              f"drive={drive} pairing={pairing} nframes={nframes}")
        print(f"              vminus={vminus} vplus={vplus} κ={kappa}")
        print("  判据：分量级 12 壳 = 某连通分量恰 V=12 且其内全 deg==5"
              "（5-正则 ⇒ 独立连通分量 = 正二十面体）")
        print("=" * 78)
        rows, conserved = product_probe(core, nframes)
        _print_product(rows, conserved)
        return
    print("=" * 78)
    print(f"[L0 内核] seed={seed} cap={cap} dmin={dmin} nframes={nframes} "
          f"drive={drive} pairing={pairing} probe={int(probe)} puncture={int(puncture)}")
    if probe:
        print("  三探针 = 度数分布(maxdeg/gini) | 闭合壳(Σ(6−deg)/晶子12) | 周期(状态键短循环)")
        print("=" * 78)
        if puncture:
            v, d0 = core.puncture()
            print(f"[穿刺] V- 删元胞 {v}(deg={d0}) → 洞 {d0} 边形")
        rows, period, pre = run_probe(core, nframes)
        _print_probe(rows)
        if period is None:
            print(f"  周期检测：{len(rows) - 1} 帧内无重复状态键（未进入短循环）")
        else:
            print(f"  周期检测：状态键在 t={pre} 与 t={pre + period} 重复 "
                  f"→ 周期={period}，预周期={pre}")
        return
    print("  一帧 = 候选生成 → 优先级 → 局部消解 → 交集提交 → 悬挂修剪（双缓冲）")
    print("=" * 78)
    print("f0 " + _fmt(core.stats()))
    if puncture:
        v, d0 = core.puncture()
        print(f"[穿刺] V- 删元胞 {v}(deg={d0}) → 洞 {d0} 边形 "
              f"（Δd 预期 = deg-3 = {d0 - 3}）")
        print("p0 " + _fmt(core.stats()))
    for _ in range(nframes):
        r = core.frame()
        print(f"f{r['t']:<2d} " + _fmt(core.stats(), f" | V+={r['born']} V-={r['dead']} "
              f"创生ΔE={r['spent']} 断边={r['broken']} netΔE={r['dE']}"))
        if core.net.V() == 0:
            print("  (网络已清空)")
            break
    print("-" * 78)
    print("说明: chi=2 且 边面!=2=0 且 领结=0 且 link 无 disc → 闭合流形（无交叉）；"
          "洞=0 → 纯三角剖分。")


if __name__ == "__main__":
    main()
