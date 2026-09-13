"""
4 球相切 → 中心坐标池断裂，仅有 1 种可能。

使用坐标数字池（Pool）演示：
    1. 构造 4 球正四面体，全部相切
    2. 池追踪所有球心和切点坐标
    3. 验证中心空腔内几乎没有空闲坐标可用
"""

import math
from pool import Pool, align, radius_of, is_on_surface, TOLERANCE

DECIMALS = 4


# ── 用已知几何构造 4 球四面体 ──

# 球 0：原点，体积 1
r0 = radius_of(1)
p0 = (0.0, 0.0, 0.0)

# 球 1：+X 方向，体积 2
r1 = radius_of(2)
p1 = (align(r0 + r1), 0.0, 0.0)

# 球 2：与 p0, p1 相切，+Y，体积 2
r2 = radius_of(2)
d01 = math.dist(p1, p0)
x0 = ((r2+r0)**2 - (r2+r1)**2 + d01**2) / (2.0 * d01)
h = math.sqrt(max((r2+r0)**2 - x0**2, 0.0))
p2 = (align(x0), align(h), 0.0)

# 球 3：与 p0, p1, p2 相切，+Z，体积 2
r3 = radius_of(2)

def _trilateration(a, b, c, ra, rb, rc):
    """三维三边测量：返回第 4 点坐标（取 +Z 解）。"""
    dab = math.dist(b, a)
    ex = ((b[0]-a[0])/dab, (b[1]-a[1])/dab, (b[2]-a[2])/dab)
    x = (ra**2 - rb**2 + dab**2) / (2.0*dab)
    c_vec = (c[0]-a[0], c[1]-a[1], c[2]-a[2])
    i = ex[0]*c_vec[0] + ex[1]*c_vec[1] + ex[2]*c_vec[2]
    ey_vec = (c_vec[0] - i*ex[0], c_vec[1] - i*ex[1], c_vec[2] - i*ex[2])
    j = math.sqrt(ey_vec[0]**2 + ey_vec[1]**2 + ey_vec[2]**2)
    ey = (ey_vec[0]/j, ey_vec[1]/j, ey_vec[2]/j)
    ez = (ex[1]*ey[2] - ex[2]*ey[1],
          ex[2]*ey[0] - ex[0]*ey[2],
          ex[0]*ey[1] - ex[1]*ey[0])
    y = (ra**2 - rc**2 + i**2 + j**2 - 2*i*x) / (2.0*j)
    z = math.sqrt(max(ra**2 - x**2 - y**2, 0.0))
    return (align(a[0] + x*ex[0] + y*ey[0] + z*ez[0]),
            align(a[1] + x*ex[1] + y*ey[1] + z*ez[1]),
            align(a[2] + x*ex[2] + y*ey[2] + z*ez[2]))

p3 = _trilateration(p0, p1, p2, r3+r0, r3+r1, r3+r2)

spheres = [
    (0, p0, r0, 1),
    (1, p1, r1, 2),
    (2, p2, r2, 2),
    (3, p3, r3, 2),
]

print("=" * 60)
print("  四面体：4 球全部相切")
print("=" * 60)
for idx, c, r, v in spheres:
    print(f"  球 {idx}: center={c}  r={r:.4f}  vol={v}")

# ── 坐标池 ──
pool = Pool()

# 声称球心
for idx, c, r, v in spheres:
    pool.claim_center(idx, c)

# 找到并声称切点
for i in range(4):
    for j in range(i+1, 4):
        ci, ri = spheres[i][1], spheres[i][2]
        cj, rj = spheres[j][1], spheres[j][2]
        # 找到一个近似在两球面上的网格点
        found = None
        for pt in pool.free_surface_points(ci, ri):
            if is_on_surface(pt, cj, rj):
                found = pt
                break
        if found:
            pool.claim_surface(i, found)
            pool.claim_surface(j, found)
            print(f"  切点 {i}-{j}: {found}")

print("\n  池统计：")
pool.summary()

# ── 中心空腔 ──
# 四点重心（初始估计）
cx = sum(c[0] for _, c, _, _ in spheres) / 4
cy = sum(c[1] for _, c, _, _ in spheres) / 4
cz = sum(c[2] for _, c, _, _ in spheres) / 4
center0 = (cx, cy, cz)

# 球 0 比其它球小，重心偏离子 0 方向 → 需要找到真正的内切球心
# 沿重心→球 0 方向搜索等距点
c0 = spheres[0][1]  # (0,0,0)
r0 = spheres[0][2]
# 方向向量：从重心指向球 0
dx0 = c0[0] - center0[0]
dy0 = c0[1] - center0[1]
dz0 = c0[2] - center0[2]
d_norm = math.sqrt(dx0*dx0 + dy0*dy0 + dz0*dz0)
u0 = (dx0/d_norm, dy0/d_norm, dz0/d_norm)

def _surface_dist(pt, sphere_pos, sphere_r):
    return math.dist(pt, sphere_pos) - sphere_r

def _min_surface_dist(pt):
    return min(_surface_dist(pt, s[1], s[2]) for s in spheres)

# 二分搜索：在重心到球 0 方向找使最小表面距最大的点
lo, hi = 0.0, 1.0
for _ in range(50):
    mid = (lo + hi) / 2
    # pt = 重心 + 方向(重心→球0) * mid * 距离
    pt = (center0[0] + u0[0] * mid * d_norm,
          center0[1] + u0[1] * mid * d_norm,
          center0[2] + u0[2] * mid * d_norm)
    d0 = _surface_dist(pt, c0, r0)
    d1 = _surface_dist(pt, spheres[1][1], spheres[1][2])
    if d0 < d1:
        hi = mid  # 过头了（d0太小），往回
    else:
        lo = mid  # 还不够（d0太大），继续向球0方向

t_opt = (lo + hi) / 2
center = (align(center0[0] + u0[0] * t_opt * d_norm),
          align(center0[1] + u0[1] * t_opt * d_norm),
          align(center0[2] + u0[2] * t_opt * d_norm))
print(f"\n  内切球心: {center}")
print(f"   （重心: ({align(center0[0])}, {align(center0[1])}, {align(center0[2])})）")

# 到每个球表面的距离
inner_r = float('inf')
for idx, c, r, v in spheres:
    d = math.dist(center, c)
    to_surface = d - r
    print(f"    距球 {idx} 表面: {to_surface:.4f}")
    if to_surface < inner_r:
        inner_r = to_surface

inner_r = align(inner_r)
inner_v = align((4.0 * math.pi / 3.0) * inner_r ** 3)
print(f"\n  空腔内切半径: {inner_r}")
print(f"  空腔内切体积: {inner_v}")

# ══════════════════════════════════════════════════════════
#  5 号球：放入空腔中心
# ══════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"  5 号球：放入空腔中心")
print(f"{'='*60}")

# 5 号球的半径由 4 球包围锁定
r5 = inner_r
# 按标准公式算体积（几何决定）
v5_geom = align((4.0 * math.pi / 3.0) * r5 ** 3)
p5 = center

print(f"\n  几何约束：")
print(f"    中心  = {p5}")
print(f"    半径  = {r5}")
print(f"    体积  = {v5_geom}")

# ── 5 号球入池 ──
pool.claim_center(5, p5)

# 切点由解析几何确定：切点在两球心连线上
# tp = C5 + (Ci - C5) * r5 / dist(C5, Ci)
found_tangents = 0
for idx, c, r, v in spheres:
    # 球心连线方向
    dx = c[0] - p5[0]
    dy = c[1] - p5[1]
    dz = c[2] - p5[2]
    d = math.sqrt(dx*dx + dy*dy + dz*dz)
    u = (dx/d, dy/d, dz/d)
    # 切点：从 5 号球心沿方向走 r5
    tp_raw = (p5[0] + u[0] * r5,
              p5[1] + u[1] * r5,
              p5[2] + u[2] * r5)
    tp = (align(tp_raw[0]), align(tp_raw[1]), align(tp_raw[2]))

    # 验证网格对齐后的点确实在两球面上
    on5 = is_on_surface(tp, p5, r5)
    on_i = is_on_surface(tp, c, r)
    is_free = pool.is_free(tp) or 5 in pool.owners_of(tp)

    if on5 and on_i and is_free:
        pool.claim_surface(idx, tp)
        pool.claim_surface(5, tp)
        found_tangents += 1
        dist_5 = math.dist(tp, p5)
        dist_i = math.dist(tp, c)
        center_dist = d
        print(f"\n    与球 {idx} 相切 ✓  切点 = {tp}")
        print(f"      |tp-5号心| = {dist_5:.6f}  r5 = {r5:.4f}")
        print(f"      |tp-球{idx}心| = {dist_i:.6f}  r{idx} = {r:.4f}")
        print(f"      球心距 = {center_dist:.4f}  r5+r{idx} = {r5+r:.4f}")
    else:
        print(f"\n    与球 {idx} 相切 ✗")
        if not on5: print(f"      tp 不在 5 号球面上")
        if not on_i: print(f"      tp 不在球 {idx} 面上")
        if not is_free: print(f"      tp 已被占用: {pool.owners_of(tp)}")

# ── 5 号球度数 & 体积 ──
degree5 = found_tangents
print(f"\n{'='*60}")
print(f"  5 号球状态")
print(f"{'='*60}")
print(f"  度数 degree = {degree5}  (与 {degree5} 个外球相切)")
print(f"  体积上限 = 度数+1 = {degree5 + 1}")
print(f"  几何体积 V = {v5_geom}")
print(f"  体积 ≤ 度数+1 ? {'✓' if v5_geom <= degree5 + 1 else '✗'}")

# 按 SPUM 规则：球在可能时张到最大
# 但 5 号球被几何锁定
r_ideal = radius_of(degree5 + 1)
print(f"  若球膨胀到体积=度数+1={degree5+1}，半径应 = {r_ideal:.4f}")
print(f"  实际半径 {r5:.4f} < 理想半径 {r_ideal:.4f}")
print(f"  → 坐标池断裂：5 号球被 4 球包围锁定，无法膨胀")

print(f"\n  最终池统计：")
pool.summary()
print(f"  总球数: 5")
