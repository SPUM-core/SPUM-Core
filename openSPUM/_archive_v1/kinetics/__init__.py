"""
SPUM 方向动力学层 (kinetics) — 反弹与惯性的离散帧引擎。

定位:
    在 coda_ca 纯拓扑层 (只有度数) 之上，为节点增加"方向偏好"变量，
    对应缺口矢量的指向。本层演示两件事:

        1. 反弹 = 缺口矢量在闭锁边界断供后被强制反转
        2. 惯性 = 改写方向偏好需要帧数, 帧数正比于锁定度 lock

本体论约束 (继承 coda_ca):
    - 节点不存坐标/半径/力
    - 位置 position 是只读投影 (由 direction 每帧累加导出)
    - 动量账 = Σ lock·dir, 弹性碰撞中守恒 (可验证涌现, 非公设)
    - 热 = 被散射为无向涨落的关系流 (非弹性时出现)
    - 不用"压缩/储存/弹性势能"等旧范式词汇

使用方式:
    from kinetics import Body, KineticsEngine
    eng = KineticsEngine()
    a = Body(name="A", lock=1.0, direction=1.0, position=0.0)
    b = Body(name="B", lock=100000.0, direction=0.0, position=10.0, closed=True)
    eng.add(a).add(b)
    eng.run(60)
    print(eng.last())
"""

from .core import Body, KineticsEngine, FrameSnapshot

__all__ = ["Body", "KineticsEngine", "FrameSnapshot"]
