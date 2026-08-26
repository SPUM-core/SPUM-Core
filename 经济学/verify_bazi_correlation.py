'''
股票先天八字五行向量 —— 相关性验证
===================================

命题: 五行向量 (S_bazi) 接近的股票, 其行情走势相关性是否高于随机?
      (用户主张: 先天五行相同 → 同相位 → 走势趋同)

方法:
  1. 读 stock_bazi.py 输出的 bazi_data.csv (深交所主板, 含明字版 + 藏干版 S 向量与行业)
  2. 新浪源获取日线, 共同窗口 2022-01-01~ (保证覆盖率), 重采样周收益
  3. 两两 Pearson 相关矩阵 (N=84 → 3486 对)
  4. 分组:
     A  top-10% S 向量欧氏距离最小对 (~350 对)   ← 检验组
     B  随机对 (同规模)                          ← 基线
     C  同行业对                                 ← 行业混杂对照
  5. 置换检验: S 向量随机打乱 500 次, 重算 top-10% 对相关均值 → null 分布 → p 值
  6. 行业混杂检查: A 组内同行业对比例 vs B 组——若 A 组行业聚集,
     相关显著可能只是行业因子重复

诚实预期: 上市时刻八字与 30 年后股价相关性先验极弱, 很可能不显著。
         结果无论正负都如实报告。

用法:
  python -u verify_bazi_correlation.py [--start 2022-01-01] [--top-pct 0.10]
                                    [--vec ming|hide] [--log bazi_corr_run.log]
  --vec: 五行向量版本 —— ming=明字版(8字等权) / hide=藏干版(本气0.7/中气0.2/余气0.1), 默认 hide
'''
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
import pandas as pd

WX_ORDER = ['水', '木', '火', '土', '金']


_LOG_F = None


def log(msg, flush=True):
    '''同时输出到 stdout 与日志文件 (实时进度)'''
    print(msg, flush=flush)
    if _LOG_F is not None:
        _LOG_F.write(msg + '\n')
        _LOG_F.flush()


def load_prices(codes, start, end='2026-08-12'):
    '''新浪源获取日线 (限流防护: 间隔+重试), 对齐日期索引'''
    import akshare as ak
    import time
    frames = {}
    ok = []
    for idx, code in enumerate(codes):
        for attempt in range(3):
            try:
                df = ak.stock_zh_a_daily(symbol=f'sz{code}', start_date=start, end_date=end)
                if df is None or len(df) == 0:
                    raise ValueError('empty')
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')[['close']]
                frames[code] = df['close']
                ok.append(code)
                break
            except Exception as e:
                if attempt == 2:
                    log(f'  [跳过] {code}: {str(e)[:60]}')
                time.sleep(1.0 + attempt)
        if (idx + 1) % 10 == 0 or idx == len(codes) - 1:
            log(f'  下载进度 {idx + 1}/{len(codes)} (可用 {len(ok)})')
        time.sleep(0.6)  # 限流间隔
    px = pd.DataFrame(frames).sort_index()
    px = px.dropna(how='all')
    return ok, px


def weekly_returns(px):
    '''周收益: 每周五收盘 → 周收益序列'''
    w = px.resample('W-FRI').last().dropna(how='all')
    wr = w.pct_change().dropna(how='all')
    return wr


def corr_matrix(wr):
    '''两两 Pearson 相关矩阵 (N,N)'''
    return wr.corr()


def pair_corrs(cm, pairs):
    '''取指定配对集合的相关值数组'''
    vals = []
    for i, j in pairs:
        v = cm.iloc[i, j]
        if not np.isnan(v):
            vals.append(v)
    return np.array(vals)


def svec_matrix(bazi, vec='hide'):
    '''S 向量矩阵 (N,5), 按 codes 顺序; vec=ming(明字版)/hide(藏干版)'''
    cols = WX_ORDER if vec == 'ming' else [f'{w}_hide' for w in WX_ORDER]
    return bazi[cols].values.astype(float)


def top_pairs(dist, k):
    '''欧氏距离矩阵取 top-k 最小对 (i<j)'''
    n = dist.shape[0]
    idx = np.triu_indices(n, k=1)
    dvals = dist[idx]
    order = np.argsort(dvals)[:k]
    return [(idx[0][t], idx[1][t]) for t in order]


def random_pairs(n, k, rng):
    '''随机 k 对 (不重复)'''
    pool = [(i, j) for i in range(n) for j in range(i + 1, n)]
    sel = rng.choice(len(pool), size=k, replace=False)
    return [pool[t] for t in sel]


def permute_top_mean(smat, cm, k, n_perm=500, seed=2026):
    '''置换检验: 打乱 S 向量行 → 重算 top-k 对相关均值 → null 分布'''
    rng = np.random.default_rng(seed)
    nulls = []
    n = smat.shape[0]
    for _ in range(n_perm):
        perm = smat[rng.permutation(n)]
        d = np.linalg.norm(perm[:, None, :] - perm[None, :, :], axis=2)
        pairs = top_pairs(d, k)
        vals = pair_corrs(cm, pairs)
        nulls.append(vals.mean() if len(vals) else np.nan)
    return np.array(nulls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2022-01-01')
    ap.add_argument('--top-pct', type=float, default=0.10)
    ap.add_argument('--n-perm', type=int, default=500)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--log', default='bazi_corr_run.log', help='进度/结果日志文件')
    ap.add_argument('--vec', choices=['ming', 'hide'], default='hide',
                    help='五行向量版本: ming=明字版 8字等权 / hide=藏干版(本气0.7/中气0.2/余气0.1)')
    args = ap.parse_args()

    global _LOG_F
    _LOG_F = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), args.log),
                  'w', encoding='utf-8')

    here = os.path.dirname(os.path.abspath(__file__))
    bazi = pd.read_csv(os.path.join(here, 'bazi_data.csv'))
    # 关键: pandas 会把 "000001" 解析成整数 1, 必须补零还原为 6 位代码
    bazi['code'] = bazi['code'].astype(str).str.zfill(6)
    # 过滤: 上市早于窗口起点, 保证行情覆盖率
    bazi = bazi[pd.to_datetime(bazi['list_date']) <= pd.to_datetime(args.start)]
    codes = bazi['code'].tolist()
    log(f'股票池: {len(codes)} 只 (上市 ≤ {args.start})')

    log('获取行情...')
    ok_codes, px = load_prices(codes, args.start)
    # 关键: S 向量顺序必须与 px 列序(ok_codes)一致, 否则相关矩阵错位
    bazi = bazi[bazi['code'].isin(ok_codes)].set_index('code').loc[ok_codes].reset_index()
    log(f'行情可用: {len(ok_codes)} 只, {px.shape[0]} 交易日')

    wr = weekly_returns(px)
    log(f'周收益序列: {wr.shape[0]} 周, {wr.shape[1]} 只')
    cm = corr_matrix(wr)
    n = len(ok_codes)
    smat = svec_matrix(bazi, args.vec)
    vname = '藏干版' if args.vec == 'hide' else '明字版'
    log(f'相关矩阵: {n}x{n}, 总配对数 {n*(n-1)//2}, 五行向量: {vname}')

    # 欧氏距离
    dist = np.linalg.norm(smat[:, None, :] - smat[None, :, :], axis=2)
    k = int(n * (n - 1) / 2 * args.top_pct)
    tp = top_pairs(dist, k)
    rng = np.random.default_rng(args.seed)
    rp = random_pairs(n, k, rng)

    a_vals = pair_corrs(cm, tp)
    b_vals = pair_corrs(cm, rp)

    # 行业混杂: 同行业对比例
    ind = bazi['industry'].values
    ind_map = {c: i for c, i in zip(bazi['code'].tolist(), ind)}
    def same_ind(pairs):
        return sum(1 for i, j in pairs if ind[i] == ind[j]) / max(len(pairs), 1)
    a_same, b_same = same_ind(tp), same_ind(rp)

    # 同行业对照组的平均相关
    same_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)
                  if ind[i] == ind[j]]
    c_vals = pair_corrs(cm, same_pairs)

    log(f'\n{"=" * 74}')
    log(f'  结果 ({vname}, top-{args.top_pct:.0%} S 向量最近对 vs 随机对)')
    log(f'{"=" * 74}')
    log(f'  A  S向量最近对 ({len(a_vals)} 对):  平均相关 {a_vals.mean():.4f}  中位 {np.median(a_vals):.4f}')
    log(f'  B  随机对     ({len(b_vals)} 对):  平均相关 {b_vals.mean():.4f}  中位 {np.median(b_vals):.4f}')
    if len(c_vals):
        log(f'  C  同行业对   ({len(c_vals)} 对):  平均相关 {c_vals.mean():.4f}  中位 {np.median(c_vals):.4f}')
    log(f'\n  行业混杂检查:')
    log(f'    A 组同行业对比例: {a_same:.1%}    B 组: {b_same:.1%}')

    # 置换检验
    log(f'\n  置换检验 ({args.n_perm} 次, 打乱 S 向量):')
    nulls = permute_top_mean(smat, cm, k, args.n_perm, args.seed)
    real = a_vals.mean()
    p = (np.sum(nulls >= real) + 1) / (len(nulls) + 1)
    log(f'    真实 top-10% 组相关均值: {real:.4f}')
    log(f'    null 分布: 均值 {np.nanmean(nulls):.4f}  SD {np.nanstd(nulls):.4f}')
    log(f'    90/95/99 分位: {np.nanpercentile(nulls, [90, 95, 99])}')
    log(f'    置换 p 值: {p:.3f}  {"★显著" if p < 0.05 else "不显著"}')
    if len(c_vals):
        t_same = (real - c_vals.mean()) / (np.std(c_vals) / np.sqrt(len(c_vals)))
        log(f'    A 组 vs 同行业对照: 差 {real - c_vals.mean():+.4f} (t≈{t_same:.2f})')

    # 同行业对中, S 相似子集 vs 同行业全集的比较
    if len(c_vals):
        same_codes = [(i, j) for i in range(n) for j in range(i + 1, n)
                      if ind[i] == ind[j]]
        same_dist = np.array([dist[i, j] for i, j in same_codes])
        q = np.quantile(same_dist, args.top_pct)
        close_same = [p for p, d in zip(same_codes, same_dist) if d <= q]
        cv = pair_corrs(cm, close_same)
        log(f'\n  S 相似 × 同行业 双重筛选 ({len(close_same)} 对):')
        log(f'    平均相关 {cv.mean():.4f}  vs 同行业全集 {c_vals.mean():.4f}')
    log('\n[完成]')


if __name__ == '__main__':
    main()
