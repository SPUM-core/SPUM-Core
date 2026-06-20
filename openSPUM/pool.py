"""
坐标数字池（Coordinate Pool）.

精度 4 位小数。池管理两类坐标的占用状态：
    1. 球心坐标 —— 每个球唯一占用
    2. 球面坐标 —— 最多被 2 球共享（即切点）

球体内部坐标不计入池（量太大），体积用解析公式计算。
球面的"在面上"判断用容差逼近网格对齐误差。
"""

import math

DECIMALS = 4
STEP = 10 ** (-DECIMALS)          # 0.0001
TOLERANCE = STEP * 1.5            # 0.00015 — 网格对齐径向容差


def align(v: float) -> float:
    return round(v, DECIMALS)


def radius_of(volume: int) -> float:
    return align(((3.0 * volume) / (4.0 * math.pi)) ** (1.0 / 3.0))


def _dist2(a: tuple, b: tuple) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return dx*dx + dy*dy + dz*dz


def is_on_surface(pt: tuple, center: tuple, radius: float) -> bool:
    """网格对齐点 pt 是否在球面附近（容差内）？"""
    d = math.sqrt(_dist2(pt, center))
    return abs(d - radius) <= TOLERANCE


class Pool:
    """
    坐标数字池。

    每个坐标的状态（由 _owners 字典编码）：
        · 空闲 —— 键不存在
        · 被球 i 单独占用 —— [i]
        · 被球 i, j 共享（切点） —— [i, j]
    """

    def __init__(self):
        self._owners: dict[tuple, list[int]] = {}

    # ── 声称坐标 ──

    def claim_center(self, sphere_idx: int, pt: tuple) -> bool:
        """球心坐标（只能被 1 个球占用）。"""
        if pt in self._owners:
            return False
        self._owners[pt] = [sphere_idx]
        return True

    def claim_surface(self, sphere_idx: int, pt: tuple) -> bool:
        """球面坐标（最多被 2 个球共享即为切点）。"""
        if pt not in self._owners:
            self._owners[pt] = [sphere_idx]
            return True
        owners = self._owners[pt]
        if len(owners) >= 2:
            return False
        if sphere_idx in owners:
            return True
        owners.append(sphere_idx)
        return True

    def unclaim(self, sphere_idx: int, pt: tuple):
        """释放一个坐标占用。"""
        if pt not in self._owners:
            return
        remaining = [s for s in self._owners[pt] if s != sphere_idx]
        if remaining:
            self._owners[pt] = remaining
        else:
            del self._owners[pt]

    # ── 查询 ──

    def is_free(self, pt: tuple) -> bool:
        return pt not in self._owners

    def owners_of(self, pt: tuple) -> list[int]:
        return self._owners.get(pt, []).copy()

    def is_tangent(self, pt: tuple) -> bool:
        return len(self._owners.get(pt, [])) == 2

    @property
    def count(self) -> int:
        return len(self._owners)

    # ── 球面空闲点枚举 ──

    def free_surface_points(self, center: tuple, radius: float) -> list[tuple]:
        """
        返回球面上所有空闲的网格对齐坐标。

        策略：
          1. 用自适应角度步长在球面上采样
          2. 每个采样点对齐到 4 位小数网格
          3. 只保留真正在球面附近（容差内）的网格点
          4. 去重 + 过滤已被占用的坐标
        """
        cx, cy, cz = center
        r = radius
        if r < TOLERANCE:
            pt = (align(cx + r), align(cy), align(cz))
            return [pt] if self.is_free(pt) and is_on_surface(pt, center, r) else []

        # 角度步长：保证每个网格 cell 约被采样 1 次
        step_angle = max(0.02, STEP / r) if r > 0 else 0.1

        candidates: set[tuple] = set()
        theta = 0.0
        while theta <= math.pi + step_angle:
            phi = 0.0
            while phi <= 2 * math.pi + step_angle:
                x = cx + r * math.sin(theta) * math.cos(phi)
                y = cy + r * math.sin(theta) * math.sin(phi)
                z = cz + r * math.cos(theta)
                pt = (align(x), align(y), align(z))
                if is_on_surface(pt, center, r):
                    candidates.add(pt)  # set 自动去重
                phi += step_angle
            theta += step_angle

        return [pt for pt in candidates if self.is_free(pt)]

    # ── 统计 ──

    def summary(self):
        single = sum(1 for v in self._owners.values() if len(v) == 1)
        shared = sum(1 for v in self._owners.values() if len(v) == 2)
        print(f"\n  Pool: {self.count} 个坐标")
        print(f"    单球: {single}  双球共享(切点): {shared}")


# ══════════════════════════════════════════════════════════
#  测试
# ══════════════════════════════════════════════════════════

def _verify_on_surface(label: str, pt: tuple, center: tuple, r: float):
    d = math.sqrt(_dist2(pt, center))
    ok = "✓" if abs(d - r) <= TOLERANCE else "✗"
    print(f"    {label}: |P-C|={d:.6f}  r={r:.4f}  {ok}")


if __name__ == "__main__":
    pool = Pool()

    # ── 球 0：原点，体积 1 ──
    r0 = radius_of(1)
    c0 = (0.0, 0.0, 0.0)
    pool.claim_center(0, c0)
    surface0 = pool.free_surface_points(c0, r0)
    print(f"球 0: center={c0}  r={r0:.4f}  vol=1")
    print(f"  空闲球面网格点: {len(surface0)}")

    # 选第 1 个空闲点作为切点
    tp1 = surface0[0]
    pool.claim_surface(0, tp1)
    _verify_on_surface("切点 → 球 0", tp1, c0, r0)
    print(f"  切点坐标: {tp1}")

    # ── 球 1：沿径向延伸，体积 2 ──
    r1 = radius_of(2)
    dx = tp1[0] - c0[0]
    dy = tp1[1] - c0[1]
    dz = tp1[2] - c0[2]
    d = math.sqrt(dx*dx + dy*dy + dz*dz)
    u = (dx/d, dy/d, dz/d)
    c1 = (align(c0[0] + u[0] * (r0 + r1)),
          align(c0[1] + u[1] * (r0 + r1)),
          align(c0[2] + u[2] * (r0 + r1)))
    pool.claim_center(1, c1)
    pool.claim_surface(1, tp1)
    print(f"\n球 1: center={c1}  r={r1:.4f}  vol=2")
    print(f"  球心距 = {math.sqrt(_dist2(c1, c0)):.4f}  (r0+r1={r0+r1:.4f})")
    _verify_on_surface("切点 → 球 1", tp1, c1, r1)
    _verify_on_surface("切点 → 球 0", tp1, c0, r0)
    print(f"  切点 {tp1} 是切点? {pool.is_tangent(tp1)}")

    # ── 球 2：在球 0 表面找另一个空闲切点，体积 1 ──
    # 要求与球 1 不穿透：两切点方向夹角 >= 60°
    r2 = radius_of(1)
    surface0_2 = pool.free_surface_points(c0, r0)
    print(f"\n球 0 剩余空闲球面点: {len(surface0_2)}")

    # 计算球 1 的方向单位向量
    u1 = (tp1[0]/r0, tp1[1]/r0, tp1[2]/r0)
    # 找与 u1 夹角 >= 90° (dot <= 0) 的空闲切点
    tp2 = None
    for pt in surface0_2:
        dot = (pt[0]*u1[0] + pt[1]*u1[1] + pt[2]*u1[2]) / r0
        if dot <= 0.0:  # >= 90°
            tp2 = pt
            break

    if tp2 is None:
        print("  未找到不与球 1 穿透的切点")
    else:
        pool.claim_surface(0, tp2)
        dx2 = tp2[0] - c0[0]
        dy2 = tp2[1] - c0[1]
        dz2 = tp2[2] - c0[2]
        d2 = math.sqrt(dx2*dx2 + dy2*dy2 + dz2*dz2)
        u2 = (dx2/d2, dy2/d2, dz2/d2)
        c2 = (align(c0[0] + u2[0] * (r0 + r2)),
              align(c0[1] + u2[1] * (r0 + r2)),
              align(c0[2] + u2[2] * (r0 + r2)))
        pool.claim_center(2, c2)
        pool.claim_surface(2, tp2)
        angle = math.degrees(math.acos(
            (tp1[0]*tp2[0] + tp1[1]*tp2[1] + tp1[2]*tp2[2]) / (r0*r0)))
        print(f"球 2: center={c2}  r={r2:.4f}  vol=1")
        print(f"  切点={tp2}  与 tp1 夹角={angle:.1f}°")
        _verify_on_surface("切点 → 球 2", tp2, c2, r2)
        _verify_on_surface("切点 → 球 0", tp2, c0, r0)
        print(f"  切点 {tp2} 是切点? {pool.is_tangent(tp2)}")

        # 验证球 1 和球 2 不相交
        d12 = math.sqrt(_dist2(c1, c2))
        print(f"  球 1-2 球心距: {d12:.4f}  r1+r2={r1+r2:.4f}")
        if d12 < r1 + r2 - TOLERANCE:
            print("  ⚠ 球 1 与球 2 穿透！")
        else:
            print("  球 1 与球 2 不穿透 ✓")

    pool.summary()
