# Source Generated with Decompyle++
# File: stock_sigma_multi.cpython-311.pyc (Python 3.11)

'''
SPUM × 多股 σ 耦合分析器 (G_econ 子图)
=========================================

理论基底: SPUM 经济学模块
  - ECON-001: G_econ ⊂ G_social ⊂ ⟨P, ε⟩
  - ECON-008: 市场子图 = 高连通替代路径子图
  - ECON-009: 竞争驱动 = ∇σ 跨股梯度

核心功能:
  1. 多股 σ 同步分析 —— 板块内个股的 σ 振荡是否耦合
  2. 板块 σ 共识 —— 用多股 σ 均值过滤单点假信号
  3. 跨股交叉验证 —— 单个信号在板块内获得 N 票支持才可信
  4. 背离预警 —— 个股 σ 脱离板块共识 = 异动预警

依赖: stock_sigma.py (单股 σ 分析器)
'''
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Callable
import warnings


@dataclass
class StockNode:
    """单个股票的 σ 状态节点"""
    code: str
    name: str
    sector: str
    df: pd.DataFrame
    sigma: pd.Series
    gradient: pd.Series
    phase: pd.Series
    phase_label: pd.Series
    prediction: dict


@dataclass
class GraphConfig:
    """多股图配置"""
    corr_window: int = 60
    edge_threshold: float = 0.5
    min_community_size: int = 2
    crossvote_min: int = 2
    crossvote_window: int = 5
    divergence_threshold: float = 0.15
    sector_weight: str = 'equal'


class MultiStockGraph:
    """
    多股 σ 耦合分析器。

    用法::

        graph = MultiStockGraph()

        # 添加个股 (支持 DataFrame 或 代码自动获取)
        graph.add_stock('000001', df=df_pingan, name='平安银行', sector='银行')
        graph.add_stock('600036', df=df_zhaoshang, name='招商银行', sector='银行')
        graph.add_stock('601166', df=df_xingye, name='兴业银行', sector='银行')

        # 或批量添加 (用 akshare 自动获取)
        graph.add_stocks_batch(
            codes=['000001', '600036', '601166'],
            sector='银行'
        )

        # 分析
        graph.build()
        report = graph.generate_report()
        graph.print_report(report)
    """

    def __init__(self, config: Optional[GraphConfig] = None):
        self.cfg = config if config else GraphConfig()
        self.nodes = {}
        self.graph_ready = False
        self.corr_matrix = None
        self.adjacency = None
        self.communities = []
        self.sector_sigma = {}
        self.sector_phase = {}
        self.crossvotes = {}

    def add_stock(self, code: str, df: pd.DataFrame, name: str = '', sector: str = ''):
        """
        添加个股数据 (直接传入 DataFrame)。

        参数
        ----------
        code : str
            股票代码
        df : pd.DataFrame
            日线数据 (必须含 close, volume, high, low)
        name : str
            股票名称
        sector : str
            所属板块
        """
        from stock_sigma import StockSigma, SigmaConfig
        analyzer = StockSigma()
        result = analyzer.analyze(df)
        node = StockNode(code=code, name=name or code, sector=sector, df=df,
                         sigma=result['sigma'], gradient=result['gradient'],
                         phase=result['phase'], phase_label=result['phase_label'],
                         prediction=result['prediction'])
        self.nodes[code] = node
        self.graph_ready = False

    def add_stock_auto(self, code: str, name: str = '', sector: str = '',
                       start: str = '20220101', end: Optional[str] = None, source: str = 'tx'):
        """
        自动获取数据并添加 (使用 akshare 备选源)。

        注意: 部分网络环境下可能失败, 可先用 fetch_a_stock 获取后
              用 add_stock() 传入。
        """
        from stock_sigma import StockSigma
        try:
            analyzer = StockSigma()
            df = analyzer.fetch_a_stock(code, start, end, source=source)
            self.add_stock(code, df, name, sector)
        except Exception as e:
            warnings.warn(f'无法获取 {code}: {e}')

    def add_stocks_batch(self, codes: List[str], sector: str = '',
                         start: str = '20220101', end: Optional[str] = None):
        """批量添加同板块个股"""
        for code in codes:
            self.add_stock_auto(code, '', sector, start=start, end=end)

    def load_local_batch(self, file_map: Dict[str, dict], sector: str = ''):
        """
        从本地 CSV 批量加载。

        file_map : { code: { 'filepath': ..., 'name': ... } }
        """
        from stock_sigma import StockSigma
        analyzer = StockSigma()
        for code, info in file_map.items():
            try:
                df = analyzer.load_local(info['filepath'])
                self.add_stock(code, df, info.get('name', code), info.get('sector', sector))
            except Exception as e:
                warnings.warn(f'加载 {code} 失败: {e}')

    def build(self):
        """
        构建 G_econ 子图:
          1. σ 时间序列对齐
          2. 相关系数矩阵 (滚动窗口)
          3. 邻接矩阵 (阈值过滤)
          4. 耦合群落检测
          5. 板块 σ 聚合
          6. 交叉验证票数计算
        """
        if len(self.nodes) < 2:
            raise ValueError(f'至少需要 2 只股票才能建图, 当前 {len(self.nodes)}')
        sigma_df = pd.DataFrame({code: node.sigma for code, node in self.nodes.items()}).dropna()
        self.sigma_aligned = sigma_df
        self.corr_matrix = sigma_df.rolling(self.cfg.corr_window).corr()
        self.latest_corr = sigma_df.corr()
        adj = self.latest_corr.abs() >= self.cfg.edge_threshold
        adj_values = adj.values.copy()
        np.fill_diagonal(adj_values, False)
        adj = pd.DataFrame(adj_values, index=adj.index, columns=adj.columns)
        self.adjacency = adj
        self.communities = self._detect_communities()
        self.sector_sigma = self._aggregate_sector_sigma()
        self.sector_phase = self._compute_sector_phases()
        self.crossvotes = self._compute_crossvotes()
        self.graph_ready = True

    def _detect_communities(self) -> List[List[str]]:
        """
        基于 σ 相关系数检测耦合群落。

        方法: 以相关系数 > edge_threshold 为边构建图,
              按连通分量分群。
        """
        adj = self.adjacency
        codes = list(self.nodes.keys())
        visited = set()
        communities = []
        for code in codes:
            if code in visited:
                continue
            community = []
            queue = [code]
            while queue:
                c = queue.pop(0)
                if c in visited:
                    continue
                visited.add(c)
                community.append(c)
                for neighbor in codes:
                    if neighbor not in visited and adj.loc[c, neighbor]:
                        queue.append(neighbor)
            if len(community) >= self.cfg.min_community_size:
                communities.append(community)
        isolated = [c for c in codes if c not in visited]
        for c in isolated:
            communities.append([c])
        return communities

    def _aggregate_sector_sigma(self) -> Dict[str, pd.Series]:
        """
        按板块聚合 σ (用板块均值代替个股噪声)。
        """
        sectors = {}
        for code, node in self.nodes.items():
            sec = node.sector or 'default'
            if sec not in sectors:
                sectors[sec] = []
            sectors[sec].append(code)
        sector_sigma = {}
        for sec, codes in sectors.items():
            if len(codes) == 1:
                sector_sigma[sec] = self.nodes[codes[0]].sigma
            else:
                series = [self.nodes[c].sigma for c in codes]
                sector_sigma[sec] = pd.concat(series, axis=1).mean(axis=1)
        return sector_sigma

    def _compute_sector_phases(self) -> Dict[str, str]:
        """计算板块整体相位"""
        from stock_sigma import StockSigma, SigmaConfig
        result = {}
        for sec, sigma_series in self.sector_sigma.items():
            grad = sigma_series.diff().fillna(0)
            latest_sigma = sigma_series.iloc[-1]
            latest_grad = grad.iloc[-1]
            cfg = SigmaConfig()
            if latest_sigma < cfg.threshold_low and abs(latest_grad) < 0.02:
                result[sec] = '近顶'
            elif latest_sigma > cfg.threshold_high and abs(latest_grad) < 0.02:
                result[sec] = '近底'
            elif latest_grad < -0.01:
                result[sec] = '扩张'
            elif latest_grad > 0.01:
                result[sec] = '收缩'
            else:
                result[sec] = '震荡'
        return result

    def _compute_crossvotes(self) -> Dict[str, float]:
        """
        交叉验证: 个股当前预测方向在同板块内获得多少支持票。

        返回值: { code: 支持率 }
          1.0 = 全板块一致
          0.0 = 孤军
          -1.0 = 与板块反向
        """
        votes = {}
        for code, node in self.nodes.items():
            sector = node.sector or 'default'
            my_dir = node.prediction['direction']
            peers = [c for c, n in self.nodes.items()
                     if (n.sector or 'default') == sector and c != code]
            if not peers:
                votes[code] = 0.0
            else:
                same = sum(1 for p in peers
                           if self.nodes[p].prediction['direction'] == my_dir)
                opposite = sum(1 for p in peers
                               if self.nodes[p].prediction['direction'] == -my_dir)
                total = len(peers)
                if total > 0:
                    votes[code] = (same - opposite) / total
                else:
                    votes[code] = 0.0
        return votes

    def enhance_prediction(self, code: str) -> dict:
        """
        基于板块共识增强个股预测。

        返回增强后的预测 (含原始预测 + 板块验证信息)。
        """
        if not self.graph_ready:
            self.build()
        node = self.nodes.get(code)
        if not node:
            raise ValueError(f'未找到 {code}')
        pred = node.prediction.copy()
        sector = node.sector or 'default'
        sector_sigma = self.sector_sigma.get(sector)
        if sector_sigma is not None:
            pred['sector_sigma'] = float(sector_sigma.iloc[-1])
        pred['sector_phase'] = self.sector_phase.get(sector, '未知')
        vote = self.crossvotes.get(code, 0.0)
        pred['crossvote'] = vote
        pred['vote_label'] = self._vote_label(vote)
        enhanced_conf = pred['confidence']
        if vote > 0.3:
            enhanced_conf = min(1.0, enhanced_conf + 0.15)
        elif vote < -0.3:
            enhanced_conf = max(0, enhanced_conf - 0.15)
        elif vote >= 0.6:
            enhanced_conf = min(1.0, enhanced_conf + 0.25)
        pred['confidence_enhanced'] = float(enhanced_conf)
        sigma = node.sigma
        if sector_sigma is not None:
            sigma_dev = abs(float(sigma.iloc[-1] - sector_sigma.iloc[-1]))
            pred['divergence'] = sigma_dev
            pred['divergence_warn'] = sigma_dev >= self.cfg.divergence_threshold
        else:
            pred['divergence'] = 0.0
            pred['divergence_warn'] = False
        return pred

    @staticmethod
    def _vote_label(vote: float) -> str:
        if vote >= 0.6:
            return '★ 板块强共识'
        if vote >= 0.3:
            return '✓ 板块偏共识'
        if vote >= -0.3:
            return '○ 无共识'
        if vote >= -0.6:
            return '✗ 板块偏反向'
        return '✗✗ 板块强反向'

    def detect_divergences(self) -> Dict[str, dict]:
        """
        检测所有个股与板块的 σ 背离。

        返回: { code: { 'deviation', 'warn', 'details', ... } }
        """
        if not self.graph_ready:
            self.build()
        divergences = {}
        for code, node in self.nodes.items():
            sector = node.sector or 'default'
            sector_sigma = self.sector_sigma.get(sector)
            if sector_sigma is None:
                continue
            sigma_dev = (node.sigma - sector_sigma).dropna()
            latest_dev = float(sigma_dev.iloc[-1])
            grad_dev = float(node.gradient.iloc[-1])
            sector_grad = float(sector_sigma.diff().fillna(0).iloc[-1])
            divergences[code] = {
                'deviation': latest_dev,
                'warn': abs(latest_dev) >= self.cfg.divergence_threshold,
                'grad_mismatch': grad_dev * sector_grad < 0,
                'sigma_latest': float(node.sigma.iloc[-1]),
                'sector_sigma_latest': float(sector_sigma.iloc[-1]),
            }
        return divergences

    def generate_report(self) -> dict:
        """生成本轮分析的完整报告"""
        if not self.graph_ready:
            self.build()
        report = {
            'overview': {
                'n_stocks': len(self.nodes),
                'n_communities': len(self.communities),
                'n_edges': int(self.adjacency.sum().sum() / 2),
                'stocks': [{'code': c, 'name': n.name, 'sector': n.sector}
                           for c, n in self.nodes.items()],
            },
            'communities': [],
            'sectors': {},
            'signals': {},
            'divergences': self.detect_divergences(),
        }
        for i, comm in enumerate(self.communities):
            comm_info = {
                'id': i + 1,
                'size': len(comm),
                'codes': comm,
                'names': [self.nodes[c].name for c in comm],
                'avg_correlation': 0.0,
            }
            if len(comm) > 1:
                corr_vals = []
                for c1 in comm:
                    for c2 in comm:
                        if c1 < c2:
                            corr_vals.append(self.latest_corr.loc[c1, c2])
                comm_info['avg_correlation'] = float(np.mean(corr_vals)) if corr_vals else 0.0
            report['communities'].append(comm_info)
        for sec, sigma_s in self.sector_sigma.items():
            codes = [c for c, n in self.nodes.items() if (n.sector or 'default') == sec]
            report['sectors'][sec] = {
                'phase': self.sector_phase.get(sec, '未知'),
                'sigma': float(sigma_s.iloc[-1]),
                'n_stocks': len(codes),
                'codes': codes,
            }
        for code in self.nodes:
            enhanced = self.enhance_prediction(code)
            report['signals'][code] = enhanced
        return report

    def print_report(self, report: dict):
        """打印可读报告"""
        ov = report['overview']
        print('============================================================')
        print('  SPUM 多股 σ 耦合分析报告')
        print('============================================================')
        print(f'  个股: {ov["n_stocks"]} 只, 板块: {len(report["sectors"])} 个')
        print(f'  耦合边: {ov["n_edges"]} 条, 群落: {ov["n_communities"]} 个')
        print(f'  {"─" * 56}')
        print('\n  📊 板块 σ 状态')
        for sec, info in report['sectors'].items():
            print(f'    {sec:12s} σ={info["sigma"]:.3f}  相位={info["phase"]}  ({info["n_stocks"]} 只)')
        print('\n  🔗 σ 耦合群落')
        for comm in report['communities']:
            if comm['size'] >= 2:
                names = '/'.join(comm['names'][:3])
                if len(comm['names']) > 3:
                    names += f'...(+{len(comm["names"]) - 3})'
                print(f'    群落 {comm["id"]}: {names}  (相关性={comm["avg_correlation"]:.2f})')
        print('\n  🎯 个股预测信号 (按板块验证排序)')
        signals = sorted(report['signals'].items(),
                         key=lambda x: x[1].get('crossvote', 0), reverse=True)
        print(f'  {"代码":>8s} {"名称":>8s} {"板块":>8s} {"方向":>6s} {"信心":>6s} {"增强":>6s} {"板块票":>8s}')
        for code, sig in signals:
            node = self.nodes.get(code)
            dir_arrow = {1: '↑涨', -1: '↓跌', 0: '—'}.get(sig['direction'], '?')
            vote_label = sig.get('vote_label', '')
            print(f'  {code:>8s} {node.name if node else "":>8s} '
                  f'{node.sector if node else "":>8s} {dir_arrow:>6s} '
                  f'{sig["confidence"]:.0%} {sig.get("confidence_enhanced", 0):.0%} '
                  f'{vote_label:>8s}')
        divs = {k: v for k, v in report['divergences'].items() if v['warn']}
        if divs:
            print('\n  ⚠  σ 背离预警')
            for code, info in divs.items():
                node = self.nodes.get(code)
                print(f'    {node.name if node else code:10s} '
                      f'个股σ={info["sigma_latest"]:.3f} 板块σ={info["sector_sigma_latest"]:.3f} '
                      f'偏差={info["deviation"]:+.3f}')
        else:
            print('\n  ✅ 无 σ 背离预警')
        print('\n' + '=' * 60)


class IndexSectorDemo:
    """
    用市场指数模拟多板块分析。

    演示 G_econ 子图概念 —— 不同指数代表不同"板块"。
    可用于验证多股 σ 耦合逻辑, 无需个股数据。
    """

    INDEX_MAP = {
        '000001': ('上证指数', '大盘'),
        '399001': ('深证成指', '大盘'),
        '399006': ('创业板指', '成长'),
        '000688': ('科创50', '成长'),
        '000300': ('沪深300', '蓝筹'),
        '000016': ('上证50', '蓝筹'),
        '000905': ('中证500', '中小盘'),
    }

    def __init__(self, start: str = '20220101', end: Optional[str] = None):
        self.start = start
        self.end = end

    def fetch_all(self) -> Dict[str, pd.DataFrame]:
        """
        获取所有指数数据。

        返回: { code: df }
        """
        import akshare as ak
        results = {}
        for code, (name, sector) in self.INDEX_MAP.items():
            try:
                if code.startswith('000'):
                    symbol = f'sh{code}'
                else:
                    symbol = f'sz{code}'
                df = ak.stock_zh_index_daily(symbol=symbol)
                df.index = pd.to_datetime(df['date'])
                df = df[df.index >= self.start].copy()
                if self.end:
                    df = df[df.index <= self.end].copy()
                if len(df) > 100:
                    results[code] = df
                    print(f'  ✅ {code} {name:10s}  ({len(df)} 帧)')
                else:
                    print(f'  ⚠️  {code} {name:10s}  (数据太少: {len(df)} 帧)')
            except Exception as e:
                print(f'  ❌ {code} {name:10s}  ({e})')
        return results

    def build_graph(self, data: Dict[str, pd.DataFrame]) -> MultiStockGraph:
        """构建多指数 σ 图"""
        graph = MultiStockGraph()
        for code, df in data.items():
            name, sector = self.INDEX_MAP[code]
            graph.add_stock(code, df, name=name, sector=sector)
        return graph


def quick_multi_analysis(data_map: Dict[str, Tuple[pd.DataFrame, str, str]], verbose: bool = True) -> dict:
    """
    一键多股分析。

    参数
    ----------
    data_map : { code: (df, name, sector) }
    verbose : bool
        是否打印报告

    返回
    -------
    dict : generate_report() 的结果
    """
    graph = MultiStockGraph()
    for code, (df, name, sector) in data_map.items():
        graph.add_stock(code, df, name=name, sector=sector)
    report = graph.generate_report()
    if verbose:
        graph.print_report(report)
    return report


def demo_from_indices(start: str = '20220101', end: Optional[str] = None):
    """
    用市场指数运行多股分析演示。

    验证 G_econ 子图的多指数 σ 耦合。
    """
    print('============================================================')
    print('  SPUM 多指数 σ 耦合演示')
    print('============================================================')
    print('\n▶ 获取指数数据...')
    demo = IndexSectorDemo(start=start, end=end)
    data = demo.fetch_all()
    print('\n▶ 构建 σ 耦合图...')
    graph = demo.build_graph(data)
    report = graph.generate_report()
    graph.print_report(report)
    return graph, report
