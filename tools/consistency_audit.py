#!/usr/bin/env python3
"""
跨领域一致性审计工具 v1.0
===============================

功能:
  1. 解析推导依赖图 (network/nodes.txt + edges.txt + module_nodes.txt)
  2. 构建正向/反向依赖映射（谁依赖我 / 我依赖谁）
  3. 扫描全仓库 .md 文件，提取核心概念引用
  4. 审计输出:
     - 变更影响图：修改 N-xxx → 列出所有需同步的文件
     - 悬挂引用：文件引用了未注册/不存在的节点
     - 未注册依赖：文件隐含依赖但未在 module_nodes.txt 注册
     - 推导链完整性：检查推导边是否有断裂

用法:
  cd spum-core && python tools/consistency_audit.py             # 完整审计
  cd spum-core && python tools/consistency_audit.py --node N013  # 仅审计 N013 变更影响
  cd spum-core && python tools/consistency_audit.py --impact     # 仅变更影响图
  cd spum-core && python tools/consistency_audit.py --scan-only  # 仅文件扫描

输出:
  - 终端彩色报告
  - tools/audit_report.txt（纯文本版）
"""
import os
import re
import sys
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─── 路径 ───────────────────────────────────────────────────
NODES_PATH = os.path.join(ROOT, 'network', 'nodes.txt')
EDGES_PATH = os.path.join(ROOT, 'network', 'edges.txt')
MODULE_NODES_PATH = os.path.join(ROOT, 'network', 'module_nodes.txt')
OUT_PATH = os.path.join(ROOT, 'tools', 'audit_report.txt')

# ─── 排除目录 ───────────────────────────────────────────────
EXCLUDE_DIRS = {
    '.git', '__pycache__', 'node_modules', '.trae',
    '_vol_cache', 'venv', '.venv', 'env', '.env',
    'dashboard/node_modules', 'dashboard/dist',
    'openSPUM/tests',  # 测试文件通常不引用核心概念
}

EXCLUDE_FILES = {
    'repo.json', 'config.json', 'package-lock.json',
}

# ============================================================
#  1. 解析网络文件
# ============================================================

@dataclass
class Node:
    node_id: str       # N001, GT-001, ECON-001, R1, etc.
    label: str
    definition: str
    parent: str = ''   # 父 N 节点（用于模块节点）

@dataclass
class Edge:
    source: str
    target: str
    edge_type: str      # derives_from, requires, refines, explains, drives
    note: str = ''

@dataclass
class DependencyGraph:
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    forward: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    reverse: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))


def parse_nodes(path: str) -> dict[str, Node]:
    """解析 nodes.txt 或 module_nodes.txt"""
    nodes = {}
    if not os.path.exists(path):
        print(f'[WARN] 节点文件不存在: {path}')
        return nodes
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('---'):
                continue
            # 格式: N001 | 标签 | 定义 | 父节点(可选)
            parts = [p.strip() for p in line.split('|')]
            if len(parts) < 3:
                continue
            node_id = parts[0]
            label = parts[1]
            definition = parts[2] if len(parts) > 2 else ''
            parent = parts[3] if len(parts) > 3 else ''
            nodes[node_id] = Node(
                node_id=node_id, label=label,
                definition=definition, parent=parent
            )
    return nodes


def parse_edges(path: str) -> list[Edge]:
    """解析 edges.txt"""
    edges = []
    if not os.path.exists(path):
        print(f'[WARN] 边文件不存在: {path}')
        return edges
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('---'):
                continue
            parts = [p.strip() for p in line.split('|')]
            if len(parts) < 4:
                continue
            source, target = parts[0], parts[1]
            edge_type = parts[2]
            note = parts[3] if len(parts) > 3 else ''
            edges.append(Edge(source, target, edge_type, note))
    return edges


def build_graph() -> DependencyGraph:
    """构建完整推导依赖图"""
    g = DependencyGraph()

    # 核心节点
    g.nodes.update(parse_nodes(NODES_PATH))

    # 模块节点 + 根节点
    g.nodes.update(parse_nodes(MODULE_NODES_PATH))

    # 边
    g.edges = parse_edges(EDGES_PATH)

    # 构建正向/反向索引
    for e in g.edges:
        g.forward[e.source].append(e.target)
        g.reverse[e.target].append(e.source)

    return g


# ============================================================
#  2. 文件扫描
# ============================================================

# 核心概念引用模式
PATTERNS = {
    'node_ref': re.compile(r'\b(N\d{3})\b'),                     # N001, N013
    'module_ref': re.compile(r'\b((?:GT|SOC|ECON|LING|FI|VSPT|AGT|COSM)-\d{3})\b'),
    'root_ref': re.compile(r'\b(R[1-9])\b'),                     # R1-R9
    'concept_ref': re.compile(
        r'\b(晶子|永恒粒子|空间粒子|κ粒子|悬挂边|离散帧|'
        r'不完美定理|拓扑常数12|欧拉恒等式|'
        r'净湮灭|创生事件|湮灭事件|'
        r'空间密度|σ梯度|∇σ|'
        r'五形|水形|木形|土形|金形|火形|S向量|'
        r'正二十面体|接吻数|分形生长|'
        r'反转图论|关系第一性|⟨P,ε⟩|'
        r'子图协动|σ红移|膨胀表象)\b'
    ),
}

# 推理链关键词
CHAIN_KEYWORDS = {
    'N001': ['存在', '宇宙存在', '根节点'],
    'N002': ['差异', '第一刻痕', '不可区分'],
    'N003': ['边界', '界定'],
    'N004': ['节点', '空间粒子', 'κ粒子'],
    'N005': ['边', '连接', '关联'],
    'N006': ['闭合网络', '自洽', '闭环', '内外'],
    'N007': ['最小度数', '度数≥2'],
    'N008': ['总边数守恒', '关系总量', '守恒'],
    'N009': ['创生事件', 'V⁺', '创生'],
    'N010': ['体积', '存在权重'],
    'N011': ['悬挂边删除', '悬挂边'],
    'N012': ['离散帧', '帧'],
    'N013': ['不完美', '不完美定理', '悬挂端', '锚点漂移'],
    'N014': ['晶子', '饱和', 'D_max'],
    'N015': ['分形生长', '缝隙', '层级'],
    'N016': ['拓扑常数12', '12', 'Σ(6−deg)', '角度亏损'],
    'N017': ['π', '降级', '认知投影', '认知压缩'],
    'N018': ['永恒粒子', '自持位移'],
    'N019': ['几何发生学', '欧氏几何', '认知投影', '统计投影'],
    'N020': ['运动', '引力', '密度梯度', '∇σ'],
    'N021': ['拓扑认知', '锚点漂移', '悬挂端梯度', 'DeepCNN', 'MNIST'],
    'N022': ['SPUM-图论', '图论', '连接优先'],
}


@dataclass
class FileScanResult:
    path: str
    relative_path: str
    size_bytes: int
    line_count: int
    refs_nodes: set[str] = field(default_factory=set)
    refs_modules: set[str] = field(default_factory=set)
    refs_roots: set[str] = field(default_factory=set)
    refs_concepts: set[str] = field(default_factory=set)
    detected_keywords: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    error: str = ''


def scan_file(filepath: str, root: str) -> FileScanResult:
    """扫描单个 .md 文件，提取所有 SPUM 引用"""
    rel = os.path.relpath(filepath, root)
    try:
        with open(filepath, encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        return FileScanResult(
            path=filepath, relative_path=rel,
            size_bytes=0, line_count=0, error=str(e)
        )

    lines = content.split('\n')
    result = FileScanResult(
        path=filepath, relative_path=rel,
        size_bytes=os.path.getsize(filepath),
        line_count=len(lines),
    )

    # 扫描引用
    for match in PATTERNS['node_ref'].finditer(content):
        result.refs_nodes.add(match.group(1))
    for match in PATTERNS['module_ref'].finditer(content):
        result.refs_modules.add(match.group(1))
    for match in PATTERNS['root_ref'].finditer(content):
        result.refs_roots.add(match.group(1))
    for match in PATTERNS['concept_ref'].finditer(content):
        result.refs_concepts.add(match.group(1))

    # 扫描推理链关键词
    for node_id, keywords in CHAIN_KEYWORDS.items():
        found = [kw for kw in keywords if kw in content]
        if found:
            result.detected_keywords[node_id] = found

    return result


def scan_repository(root: str) -> list[FileScanResult]:
    """遍历仓库扫描所有 .md 文件"""
    results = []
    total = 0
    skipped = 0

    for dirpath, dirnames, filenames in os.walk(root):
        # 排除目录
        dirnames[:] = [d for d in dirnames
                       if d not in EXCLUDE_DIRS
                       and not any(ex in dirpath for ex in EXCLUDE_DIRS)]

        for fname in filenames:
            if not fname.endswith('.md') or fname in EXCLUDE_FILES:
                continue
            fpath = os.path.join(dirpath, fname)
            result = scan_file(fpath, root)
            if result.error:
                skipped += 1
            else:
                results.append(result)
            total += 1

    print(f'[INFO] 扫描文件: {total} 总文件, {len(results)} 有效, {skipped} 跳过')
    return results


# ============================================================
#  3. 审计逻辑
# ============================================================

@dataclass
class AuditReport:
    graph: DependencyGraph
    file_scans: list[FileScanResult]

    # 审计结果
    orphan_refs: list[tuple[str, str, str]] = field(default_factory=list)       # (文件, 引用, 类型)
    unregistered_modules: list[tuple[str, str]] = field(default_factory=list)    # (文件, 模块节点)
    impact_map: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    chain_gaps: list[str] = field(default_factory=list)
    inconsistent_refs: list[tuple[str, str, str]] = field(default_factory=list)  # (文件, 应引用, 未引用)


def build_impact_map(g: DependencyGraph) -> dict[str, list[str]]:
    """
    构建变更影响图。
    如果修改节点 X（如 N013 不完美定理），
    所有直接或间接依赖 X 的节点都需审计。

    使用 BFS 遍历反向依赖图。
    """
    impact = {}

    for node_id in g.nodes:
        # BFS 找所有依赖此节点的下游
        visited = set()
        queue = list(g.forward.get(node_id, []))
        while queue:
            dep = queue.pop(0)
            if dep in visited:
                continue
            visited.add(dep)
            # 模块节点继续向下遍历
            for sub in g.forward.get(dep, []):
                queue.append(sub)

        # 只输出有下游依赖的节点
        if visited:
            impact[node_id] = sorted(visited)

    return impact


def audit_orphan_references(scans: list[FileScanResult],
                            g: DependencyGraph) -> list[tuple[str, str, str]]:
    """检测悬挂引用：文件引用了不存在的节点"""
    orphan = []
    for s in scans:
        for ref in s.refs_nodes:
            if ref not in g.nodes:
                orphan.append((s.relative_path, ref, 'N-节点'))
        for ref in s.refs_modules:
            if ref not in g.nodes:
                orphan.append((s.relative_path, ref, '模块节点'))
        for ref in s.refs_roots:
            if ref not in g.nodes:
                orphan.append((s.relative_path, ref, '根节点'))
    return orphan


def audit_unregistered_modules(scans: list[FileScanResult],
                               g: DependencyGraph) -> list[tuple[str, str]]:
    """检测未注册模块：文件使用了模块前缀但未在 module_nodes.txt 注册"""
    # 已知的模块前缀
    known_prefixes = {'GT', 'SOC', 'ECON', 'LING', 'FI', 'VSPT', 'AGT', 'COSM'}
    unreg = []
    for s in scans:
        for ref in s.refs_modules:
            prefix = ref.split('-')[0]
            if prefix in known_prefixes and ref not in g.nodes:
                unreg.append((s.relative_path, ref))
    return unreg


def audit_inconsistent_references(scans: list[FileScanResult],
                                  g: DependencyGraph) -> list[tuple[str, str, str]]:
    """
    检测不一致引用：文件提到了某个核心概念的关键词，
    但未引用对应的 N 节点编号。
    """
    inconsistent = []
    for s in scans:
        for node_id, keywords in s.detected_keywords.items():
            # 如果检测到关键词但没引用对应节点
            if node_id not in s.refs_nodes and node_id not in s.refs_modules:
                inconsistent.append((
                    s.relative_path, node_id,
                    f'提到了关键词 {keywords[0]} 但未引用 {node_id}'
                ))
    return inconsistent


def find_all_paths_bidi(g: DependencyGraph, src: str, tgt: str, max_depth: int = 8) -> list[list[str]]:
    """
    双向 BFS 查找 src 到 tgt 的所有路径。
    正向边和反向边都可以走（推导链不要求方向一致）。
    """
    paths = []
    queue = [(src, [src])]
    while queue:
        node, path = queue.pop(0)
        if len(path) > max_depth:
            continue
        # 正向边
        for neighbor in g.forward.get(node, []):
            if neighbor == tgt:
                paths.append(path + [neighbor])
            elif neighbor not in path:
                queue.append((neighbor, path + [neighbor]))
        # 反向边（也走 reverse 映射中的节点）
        for neighbor in g.reverse.get(node, []):
            if neighbor == tgt:
                paths.append(path + [neighbor])
            elif neighbor not in path:
                queue.append((neighbor, path + [neighbor]))
    return paths


def audit_chain_integrity(g: DependencyGraph) -> list[str]:
    """
    检查推导链完整性。
    
    核心推导链 N001→N002→...→N022 中，相邻节点不必直接相连，
    只要在双向图中有任何连通路径即可。
    """
    gaps = []
    chain = [f'N{i:03d}' for i in range(1, 23)]
    for i in range(len(chain) - 1):
        src, tgt = chain[i], chain[i + 1]
        paths = find_all_paths_bidi(g, src, tgt)
        if not paths:
            gaps.append(f'{src} → {tgt} 完全无连通路径')
        else:
            # 检查最短路径（仅供信息）
            shortest = min(paths, key=len)
            if len(shortest) > 2:
                gaps.append(f'{src} → {tgt} 间接路径 (长{len(shortest)-1}步): {"→".join(shortest)}')
    return gaps


# ============================================================
#  4. 报告输出
# ============================================================

def generate_report(report: AuditReport, show_all_files: bool = False) -> str:
    """生成可读的审计报告"""
    lines = []
    g = report.graph

    def w(*args):
        lines.append(' '.join(str(a) for a in args))

    w('=' * 90)
    w('SPUM 跨领域一致性审计报告')
    w(f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    w(f'仓库根目录: {ROOT}')
    w('=' * 90)

    # ── 0. 统计概览 ──
    w()
    w('【0】统计概览')
    w('─' * 60)
    w(f'  核心节点 (N001-N022):   {sum(1 for n in g.nodes if n.startswith("N"))}/{22}')
    w(f'  模块节点:                {sum(1 for n in g.nodes if not n.startswith("N") and not n.startswith("R"))}')
    w(f'  根节点 (R1-R9):          {sum(1 for n in g.nodes if n.startswith("R"))}')
    w(f'  推导边:                  {len(g.edges)}')
    w(f'  扫描文件:                {len(report.file_scans)}')

    # ── 1. 变更影响图 ──
    w()
    w('【1】变更影响图')
    w('─' * 60)
    w('  说明: 修改左侧节点 → 需同步右侧所有依赖节点')
    w()

    # 只显示有实质影响的
    significant_impacts = {k: v for k, v in report.impact_map.items()
                           if len(v) > 0 and k.startswith('N')}
    for node_id in sorted(significant_impacts, key=lambda x: -len(significant_impacts[x])):
        deps = significant_impacts[node_id]
        node_label = g.nodes[node_id].label if node_id in g.nodes else '?'
        w(f'  {node_id} {node_label:<12} → {len(deps)} 个依赖:')
        for dep in deps:
            dep_label = g.nodes[dep].label if dep in g.nodes else '?'
            w(f'      {dep:<12} {dep_label}')

    # ── 2. 变更影响 → 文件级 ──
    w()
    w('【2】变更影响 → 文件级映射')
    w('─' * 60)
    w('  说明: 核心节点变更时，需同步的实际文件清单')
    w()

    # 构建 节点→文件 映射
    node_to_files = defaultdict(list)
    for s in report.file_scans:
        for ref in s.refs_nodes | s.refs_modules | s.refs_roots:
            node_to_files[ref].append(s.relative_path)

    for node_id in sorted(significant_impacts, key=lambda x: -len(significant_impacts[x])):
        if node_id not in g.nodes:
            continue
        # 该节点本身引用的文件
        direct_files = node_to_files.get(node_id, [])
        w(f'  [{node_id}] {g.nodes[node_id].label} — {len(direct_files)} 个文件直接引用')
        if direct_files and show_all_files:
            for f in sorted(direct_files):
                w(f'    {f}')

        # 依赖节点引用的文件（间接影响）
        for dep in significant_impacts.get(node_id, []):
            dep_files = node_to_files.get(dep, [])
            if dep_files:
                w(f'    → {dep} 影响 {len(dep_files)} 文件:')
                if show_all_files:
                    for f in sorted(dep_files)[:10]:
                        w(f'      {f}')
                    if len(dep_files) > 10:
                        w(f'      ... 还有 {len(dep_files)-10} 个')

    # ── 3. 悬挂引用检测 ──
    w()
    w('【3】悬挂引用检测')
    w('─' * 60)
    w('  说明: 文件引用了未在 network/ 中注册的节点')
    w()
    if report.orphan_refs:
        w(f'  发现 {len(report.orphan_refs)} 处悬挂引用:')
        for fpath, ref, rtype in sorted(report.orphan_refs):
            w(f'    {fpath:<50} 引用了未注册的 {rtype}: {ref}')
    else:
        w('  ✅ 无悬挂引用')

    # ── 4. 未注册模块检测 ──
    w()
    w('【4】未注册模块节点检测')
    w('─' * 60)
    w('  说明: 文件使用了模块节点 ID 但未在 module_nodes.txt 注册')
    w()
    if report.unregistered_modules:
        w(f'  发现 {len(report.unregistered_modules)} 处未注册引用:')
        for fpath, ref in sorted(report.unregistered_modules):
            w(f'    {fpath:<50} 未注册: {ref}')
    else:
        w('  ✅ 所有模块节点均已注册')

    # ── 5. 引用不一致检测 ──
    w()
    w('【5】内容-引用不一致检测')
    w('─' * 60)
    w('  说明: 文件提到了核心概念的关键词但未引用对应 N 节点')
    w()

    # 优先级文件：领域核心文档
    PRIORITY_PATTERNS = [
        r'spum-.*\.md$',                      # 领域规则文件
        r'^五行\\', r'^物理学\\',               # 五行/物理领域
        r'^儒释道哲学\\',                       # 哲学领域
        r'^社会学\\', r'^经济学\\', r'^语言学\\', # 社科领域
        r'^数学\\', r'^图论\\',                # 数学/图论
        r'^宇宙学\\',                          # 宇宙学
        r'^元素化学\\',                        # 元素化学
        r'knowledge\.md$', r'SPUM2610\.md$',
        r'SPUM_系统总纲\.md$',
    ]

    def is_priority_file(rel_path: str) -> bool:
        return any(re.search(p, rel_path) for p in PRIORITY_PATTERNS)

    if report.inconsistent_refs:
        # 按文件分组
        by_file = defaultdict(list)
        for fpath, node_id, note in report.inconsistent_refs:
            by_file[fpath].append((node_id, note))

        # 优先级文件
        priority_items = {f: v for f, v in by_file.items() if is_priority_file(f)}
        other_items = {f: v for f, v in by_file.items() if not is_priority_file(f)}

        w(f'  发现 {len(report.inconsistent_refs)} 处不一致 (去重后 {len(by_file)} 文件)')
        w(f'  其中优先级文件 {len(priority_items)} 个，其余 {len(other_items)} 个')
        w()

        if priority_items:
            w(f'  ⚠ 优先级文件需要关注:')
            for fpath in sorted(priority_items, key=lambda x: -len(priority_items[x])):
                items = priority_items[fpath]
                w(f'    [{len(items)}处] {fpath}')
                for node_id, note in sorted(items)[:4]:
                    w(f'      → 建议引用 {node_id}: {note}')
                if len(items) > 4:
                    w(f'      ... 还有 {len(items)-4} 处')
        w()
        if other_items and show_all_files:
            w(f'  其余文件 (仅列出前 10):')
            for fpath in sorted(other_items, key=lambda x: -len(other_items[x]))[:10]:
                items = other_items[fpath]
                w(f'    [{len(items)}处] {fpath}')
    else:
        w('  ✅ 未发现明显不一致')

    # ── 6. 推导链完整性 ──
    w()
    w('【6】推导链完整性检查')
    w('─' * 60)
    direct_gaps = [g for g in report.chain_gaps if '完全无连通' in g]
    indirect = [g for g in report.chain_gaps if '间接路径' in g]
    if direct_gaps:
        w(f'  🔴 完全断裂: {len(direct_gaps)} 处')
        for g in direct_gaps:
            w(f'    {g}')
    if indirect:
        w(f'  🟡 间接推导 (建议补充直接边): {len(indirect)} 处')
        for gap_str in indirect:
            w(f'    {gap_str}')
            # 解析路径并建议直接边
            parts = gap_str.split('):')
            if len(parts) > 1:
                path = parts[1].strip().split('→')
                if len(path) >= 2:
                    src, tgt = path[0], path[-1]
                    mid = '→'.join(path[1:-1])
                    w(f'      建议: 添加 {src} → {tgt} 直接推导边 (当前绕经 {mid})')
    if not report.chain_gaps:
        w('  ✅ 主推导链 N001→N022 完整')

    # ── 7. 文件引用热力图 ──
    w()
    w('【7】核心节点引用热力图')
    w('─' * 60)
    w('  说明: 各核心节点被引用的文件数')
    w()
    ref_counts: dict[str, int] = defaultdict(int)
    for s in report.file_scans:
        for ref in s.refs_nodes | s.refs_modules | s.refs_roots:
            ref_counts[ref] += 1

    for node_id in sorted(ref_counts, key=lambda x: -ref_counts[x]):
        count = ref_counts[node_id]
        label = g.nodes[node_id].label if node_id in g.nodes else '?'
        bar = '█' * min(count // 2, 40)
        w(f'  {node_id:<12} {label:<16} {count:>4} 文件 {bar}')

    return '\n'.join(lines)


def write_report(text: str):
    """写报告到文件"""
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f'\n[INFO] 报告已写入: {OUT_PATH}')


# ============================================================
#  5. 主入口
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='SPUM 跨领域一致性审计工具')
    parser.add_argument('--node', type=str, help='仅审计指定节点的变更影响')
    parser.add_argument('--impact', action='store_true', help='仅输出变更影响图')
    parser.add_argument('--scan-only', action='store_true', help='仅文件扫描，不审计')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细模式（展示文件列表）')
    args = parser.parse_args()

    print('=' * 60)
    print('SPUM 跨领域一致性审计工具 v1.0')
    print('=' * 60)

    # Step 1: 解析
    print('\n[1/4] 解析推导依赖图...')
    graph = build_graph()
    print(f'  节点: {len(graph.nodes)}  边: {len(graph.edges)}')

    if args.node:
        # 单节点审计模式
        if args.node not in graph.nodes:
            print(f'[ERROR] 节点 {args.node} 不存在')
            sys.exit(1)
        impact = build_impact_map(graph)
        deps = impact.get(args.node, [])
        print(f'\n[{args.node}] {graph.nodes[args.node].label}')
        print(f'  变更后需同步 {len(deps)} 个依赖节点:')
        for dep in deps:
            label = graph.nodes[dep].label if dep in graph.nodes else '?'
            print(f'    {dep:<12} {label}')
        print(f'\n  建议操作: 修改 {args.node} 定义后，逐个审计以上 {len(deps)} 个节点对应的文件')
        return

    # Step 2: 文件扫描
    print('\n[2/4] 扫描仓库文件...')
    scans = scan_repository(ROOT)

    if args.scan_only:
        print(f'\n扫描完成: {len(scans)} 个 .md 文件')
        # 输出前 20 个最密集引用的文件
        scans_sorted = sorted(scans, key=lambda s: -(len(s.refs_nodes) + len(s.refs_modules)))
        print(f'\n引用最密集的前 20 个文件:')
        for s in scans_sorted[:20]:
            total_refs = len(s.refs_nodes) + len(s.refs_modules) + len(s.refs_roots)
            print(f'  {total_refs:>3} 引用 | {s.relative_path}')
        return

    # Step 3: 审计
    print('\n[3/4] 执行审计...')

    impact_map = build_impact_map(graph)
    orphan_refs = audit_orphan_references(scans, graph)
    unreg = audit_unregistered_modules(scans, graph)
    inconsistent = audit_inconsistent_references(scans, graph)
    chain_gaps = audit_chain_integrity(graph)

    report = AuditReport(
        graph=graph, file_scans=scans,
        impact_map=impact_map,
        orphan_refs=orphan_refs,
        unregistered_modules=unreg,
        inconsistent_refs=inconsistent,
        chain_gaps=chain_gaps,
    )

    # Step 4: 输出
    print('\n[4/4] 生成报告...')
    report_text = generate_report(report, show_all_files=args.verbose)

    if args.impact:
        # 仅输出影响图部分
        impact_section = []
        in_impact = False
        for line in report_text.split('\n'):
            if '【1】变更影响图' in line:
                in_impact = True
            elif line.startswith('【') and in_impact:
                break
            if in_impact:
                impact_section.append(line)
        print('\n'.join(impact_section))
    else:
        print()
        print(report_text[:3000])  # 终端只打印前 3000 字符
        if len(report_text) > 3000:
            print(f'\n... (报告总长 {len(report_text)} 字符，更多内容见文件)')

    write_report(report_text)

    # 摘要
    print(f'\n{"=" * 60}')
    print(f'审计摘要:')
    print(f'  变更影响节点:     {len(impact_map)}')
    print(f'  悬挂引用:        {len(orphan_refs)}')
    print(f'  未注册模块:      {len(unreg)}')
    print(f'  内容不一致:      {len(inconsistent)}')
    print(f'  推导链间隙:      {len(chain_gaps)}')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
