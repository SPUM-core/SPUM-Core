"""
OpenSPUM Phase 1 — 基础常量定义

包含 SPUM 宇宙模型的基础物理/拓扑常量:
    - κ (kappa):       最小差异尺度，关系网络自校准形成的基本长度单位
    - τ (tau):         离散帧时间单位
    - CRYSTALLITE_DEGREE_THRESHOLD:  晶子度数阈值（度数 ≥ 此值即视为晶子）
    - MAX_CONTACTS:    晶子最大接触数（三维密堆接吻数 = 12，**此为涌现结果**
                       而非预设参数，仅保留供对比验证）
    - CRYSTALLITE_DIAMETER_MAX: 晶子等效直径临界值 D_max = 4κ
    - CRYSTALLITE_CAPACITY:    晶子几何容量上限 N_max ≈ 16π ≈ 50.3

重要设计原则 (SPUM2611 v5.0 新理论):
    - 12 是计数事实，不是输入。Σ(6−deg(v)) = 12 是闭合子图的欧拉恒等式强制解
      (T2)，正二十面体是推导的输出而非种子构型（主定理：计数 → 空间）。
    - 稳定解为 deg(O) = 42 (T5)：中心粒子与全部轨道位置（12 顶点 + 30 边中点）
      相切，42 < 50.3 容量可容纳。没有"12 闭锁"——12 只作为计数链
      (12, 30, 42) 的一环存在。
    - 容量极致为 N_max = 16π ≈ 50.3 (T8)，仅作几何容量上限参考，
      不作为度数硬阈值。
    - degree 硬阈值 = CRYSTALLITE_DEGREE_THRESHOLD (42，T5 稳定解)，
      晶子聚簇连接通过 crystallite_connections 单独追踪，不增加 degree。
"""

# 最小差异尺度 — 关系网络自校准形成的基本长度单位
KAPPA: float = 1.0

# 离散帧时间单位
TAU: int = 1

# 晶子度数阈值 — 节点度数达到此值即视为晶子
# = T5 稳定解 deg(O) = 42（中心粒子与 12 顶点 + 30 边中点全部轨道位置相切）
# 这是 degree 的硬上限：任何节点的 degree 不得超过此值
# 容量极致 50.3 (T8) 仅作几何参考，不是度数硬阈值
CRYSTALLITE_DEGREE_THRESHOLD: int = 42

# 晶子最大接触数（三维密堆接吻数）
# 12 是计数事实 (T2 饱和解 n=12, d=5)，非预设输入
# 注意: 12 是涌现结果，不是预设的度数上限。
# 节点的度数可以从 0 增长到晶子阈值 (42)，12 会在闭合子图分析中自然显现。
MAX_CONTACTS: int = 12

# 晶子等效直径临界值 D_max = 4κ
CRYSTALLITE_DIAMETER_MAX: float = 4.0 * KAPPA  # = 4.0

# 晶子几何容量上限 N_max = 16π ≈ 50.265
# virtual_faces 的上限基于此值
CRYSTALLITE_CAPACITY: float = 16.0 * 3.141592653589793

# 空间几何约束容差：球体相切判定允许的相对误差
GEOMETRIC_TOLERANCE: float = 0.1

# 晶子聚簇最大规模 — 每个晶子在晶子子图中的最多连接数
# 三维密堆中每个球体最多与 12 个同类球体相切
# 此连接不增加 degree（保持 degree ≤ 42），单独通过 crystallite_connections 追踪
MAX_CRYSTALLITE_CLUSTER_SIZE: int = 12

# 晶子子图平面三角剖分度数上限 — 平面图每个顶点度数 ≤ 6
# （平面三角形的极限，超过则子图不再是平面三角剖分）
# 使用 min(MAX_CRYSTALLITE_CLUSTER_SIZE, MAX_CRYSTALLITE_PLANAR_DEGREE) 作为实际限制
MAX_CRYSTALLITE_PLANAR_DEGREE: int = 6

# 聚簇阶段最大迭代帧数 — 防止无限循环
MAX_CLUSTER_ITERATIONS: int = 200

# --- 分层扩散增长策略配额（已弃用，保留仅为引用对比） ---
# 从 v3 起，seed_epoch_engine 使用 _compute_budgets_from_degree_distribution()
# 从度数分布动态推导配额，不再使用这些固定值。
# 当前保留仅为测试和历史引用。
DANGLING_QUOTA: float = 0.20
MIDRANGE_QUOTA: float = 0.70
NEWNODE_QUOTA: float = 0.10
