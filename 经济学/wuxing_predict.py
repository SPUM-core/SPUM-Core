"""
先天五行命格 × 波动率预测 — 主模块
=====================================
基于 SPUM 五行命格理论：S₀ = (木,火,土,金,水) 由上市日期三柱六字计算

功能:
  1. 计算股票先天五行命格 S₀
  2. 缓存/加载历史数据 + 超额波动率（去沪深300 β）
  3. 同命格波动曲线预测（Ridge + XGBoost + Walk-forward验证）

用法:
  python wuxing_predict.py            # 完整运行
  python wuxing_predict.py --s0       # 只计算 S₀ 并分组
  python wuxing_predict.py --predict  # 只运行预测
  python wuxing_predict.py --s0 --predict  # 先算 S₀ 再预测（完整流程）
"""
import json, os, numpy as np
import pandas as pd
from datetime import datetime, date
from collections import defaultdict
from zhdate import ZhDate
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# === 配置 ===
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(DATA_DIR, '_vol_cache')
JSON_PATH = os.path.join(DATA_DIR, 'mingge_results.json')
REPORT_PATH = os.path.join(DATA_DIR, 'predict_report.txt')
S0_LOG_PATH = os.path.join(DATA_DIR, 's0_groups.txt')

# XGBoost（可选）
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# === 天干地支五行 ===
TIANGAN = ['甲','乙','丙','丁','戊','己','庚','辛','壬','癸']
DIZHI   = ['子','丑','寅','卯','辰','巳','午','未','申','酉','戌','亥']
STEM_WM = {'甲':'木','乙':'木','丙':'火','丁':'火','戊':'土','己':'土','庚':'金','辛':'金','壬':'水','癸':'水'}
BRANCH_WM = {'寅':'木','卯':'木','巳':'火','午':'火','辰':'土','戌':'土','丑':'土','未':'土','申':'金','酉':'金','亥':'水','子':'水'}
WX = ['木','火','土','金','水']

# === 股票列表（58只核心A股） ===
STOCKS = [
    ('sz300750','宁德时代','2018-06-11'),('sz002415','海康威视','2010-05-28'),
    ('sz300124','汇川技术','2010-09-28'),('sz002475','立讯精密','2010-09-15'),
    ('sz000725','京东方A','2001-01-12'),('sh688981','中芯国际','2020-07-16'),
    ('sh603501','韦尔股份','2017-05-04'),('sh603986','兆易创新','2016-08-18'),
    ('sz002230','科大讯飞','2008-05-12'),('sz300033','同花顺','2009-12-25'),
    ('sh600570','恒生电子','2003-12-16'),('sz300059','东方财富','2010-03-19'),
    ('sh600519','贵州茅台','2001-08-27'),('sh600887','伊利股份','1996-03-12'),
    ('sz000333','美的集团','2013-09-18'),('sz000651','格力电器','1996-11-18'),
    ('sz000858','五粮液','1998-04-27'),('sh603288','海天味业','2014-02-11'),
    ('sh601888','中国中免','2009-10-15'),('sz002304','洋河股份','2009-11-06'),
    ('sh600809','山西汾酒','1994-01-06'),('sh600882','妙可蓝多','1995-12-06'),
    ('sh600276','恒瑞医药','2000-10-18'),('sz300760','迈瑞医疗','2018-10-16'),
    ('sh603259','药明康德','2018-05-08'),('sz000538','云南白药','1993-12-15'),
    ('sz300015','爱尔眼科','2009-10-30'),('sh600196','复星医药','1998-08-07'),
    ('sz002007','华兰生物','2004-06-25'),('sh600085','同仁堂','1997-06-25'),
    ('sh600036','招商银行','2002-04-09'),('sh601318','中国平安','2007-03-01'),
    ('sh601398','工商银行','2006-10-27'),('sh600030','中信证券','2003-01-06'),
    ('sh601166','兴业银行','2007-02-05'),('sh600016','民生银行','2000-12-19'),
    ('sh600031','三一重工','2003-07-03'),('sh600309','万华化学','2001-01-05'),
    ('sh601899','紫金矿业','2008-04-25'),('sh600585','海螺水泥','2002-02-07'),
    ('sz000338','潍柴动力','2007-04-30'),('sh600690','海尔智家','1993-11-19'),
    ('sh601012','隆基绿能','2012-04-11'),('sz300274','阳光电源','2011-11-02'),
    ('sz002594','比亚迪','2011-06-30'),('sh601633','长城汽车','2011-09-28'),
    ('sh600104','上汽集团','1997-11-25'),('sz002460','赣锋锂业','2010-08-10'),
    ('sh600438','通威股份','2004-03-02'),('sz002129','TCL中环','2007-04-20'),
    ('sz000002','万科A','1991-01-29'),('sh600048','保利发展','2006-07-31'),
    ('sh600900','长江电力','2003-11-18'),('sh601668','中国建筑','2009-07-29'),
    ('sz002352','顺丰控股','2010-02-05'),('sh601857','中国石油','2007-11-05'),
    ('sh600941','中国移动','2022-01-05'),('sz001979','招商蛇口','2015-12-30'),
]

# =============================================
#  核心函数
# =============================================

def calc_s0(listing_str):
    """上市日期 ISO 字符串 → S₀ 向量 (木,火,土,金,水)"""
    ld = datetime.strptime(listing_str, '%Y-%m-%d').date()
    lunar = ZhDate.from_datetime(datetime.combine(ld, datetime.min.time()))
    yg = TIANGAN[(ld.year-4)%10]; yz = DIZHI[(ld.year-4)%12]
    mg = TIANGAN[(((ld.year-4)%10)%5*2 + lunar.lunar_month)%10]; mz = DIZHI[(lunar.lunar_month+1)%12]
    dg = TIANGAN[(ld-date(1900,1,1)).days%10]; dz = DIZHI[((ld-date(1900,1,1)).days+10)%12]
    elems = [STEM_WM[yg], BRANCH_WM[yz], STEM_WM[mg], BRANCH_WM[mz], STEM_WM[dg], BRANCH_WM[dz]]
    return tuple(elems.count(w) for w in WX)

def vec_label(vec):
    """S₀ 向量 → 中文标签，如 (1,0,2,0,1) → '土旺缺火金'"""
    max_i = max(range(5), key=lambda i: vec[i])
    missing = [WX[i] for i, v in enumerate(vec) if v == 0]
    label = f'{WX[max_i]}旺'
    if missing: label += f"缺{''.join(missing)}"
    return label

def load_or_download_data():
    """从缓存加载数据，缺失则下载并缓存"""
    import akshare as ak
    if os.path.exists(CACHE_DIR) and len(os.listdir(CACHE_DIR)) >= 58:
        data = {}
        for fname in os.listdir(CACHE_DIR):
            if not fname.endswith('.npy'): continue
            sym = fname.replace('_', '.').replace('.npy', '')
            data[sym] = np.load(os.path.join(CACHE_DIR, fname), allow_pickle=True).item()
        print(f'  加载缓存: {len(data)} 只')
        return data

    os.makedirs(CACHE_DIR, exist_ok=True)
    data = {}
    for sym, name, _ in STOCKS:
        try:
            df = ak.stock_zh_a_daily(symbol=sym)
            if df is not None and len(df) > 100:
                df = df.sort_values('date')
                rets = df['close'].pct_change().values
                obj = {'dates': list(df['date'].astype(str)), 'returns': rets}
                np.save(os.path.join(CACHE_DIR, f'{sym}.npy'), obj)
                data[sym] = obj
        except Exception as e:
            print(f'  ✗ {sym} {name}: {e}')
        print(f'  下载中: {len(data)}/{len(STOCKS)}', end='\r')
    print(f'\n  下载完成: {len(data)} 只')
    return data

def calc_excess_vol(stock_data):
    """计算超额波动率（去沪深300 β）"""
    import akshare as ak
    print('  加载沪深300...')
    mkt = ak.stock_zh_index_daily(symbol='sh000300')
    mkt['date'] = pd.to_datetime(mkt['date'])
    mkt = mkt.sort_values('date')
    mkt['ret'] = mkt['close'].pct_change()
    mkt_d = dict(zip(mkt['date'].dt.strftime('%Y-%m-%d'), mkt['ret']))

    excess = {}
    for i, (sym, sd) in enumerate(stock_data.items()):
        try:
            dates, rets = sd['dates'], sd['returns']
            mv = np.array([mkt_d.get(d, np.nan) for d in dates])
            valid = ~(np.isnan(rets) | np.isnan(mv))
            rv, mv_v = rets[valid], mv[valid]
            if len(rv) < 120: continue
            beta = np.cov(rv, mv_v)[0,1] / np.var(mv_v)
            er = np.full(len(dates), np.nan); er[valid] = rv - beta * mv_v
            ev = pd.Series(er).rolling(20).std().values
            excess[sym] = {'dates': dates, 'vol': ev}
        except Exception as e:
            pass
        if (i+1) % 20 == 0:
            print(f'  超额波动: {len(excess)}/{i+1}', flush=True)
    return excess

def build_features(di_a, td, di_b, pd_d, common, i, h):
    """构造特征向量: AR(3) + 信号源多滞后(0,1,2,5,10,20,30)"""
    t, th = common[i], common[i+h]
    if t not in di_a or t not in di_b or th not in di_a:
        return None, None
    vt = td['vol'][di_a[th]]
    if np.isnan(vt) or np.isinf(vt):
        return None, None
    feats = []
    for lag in [1,2,3]:
        lt = common[i-lag] if i >= lag else None
        if lt and lt in di_a:
            av = td['vol'][di_a[lt]]
            feats.append(0 if (np.isnan(av) or np.isinf(av)) else av)
        else: feats.append(0)
    for sl in [0,1,2,5,10,20,30]:
        slt = common[i-sl] if i >= sl else None
        if slt and slt in di_b:
            sv = pd_d['vol'][di_b[slt]]
            feats.append(0 if (np.isnan(sv) or np.isinf(sv)) else sv)
        else: feats.append(0)
    return feats, vt

# =============================================
#  报告输出
# =============================================

def write_report(segments):
    """分段写入报告文件"""
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(segments))

def w(segments, *args):
    """添加行到 segments 列表"""
    line = args[0] if args else ''
    segments.append(str(line))

# =============================================
#  主流程
# =============================================

def s0_analysis(segments):
    """Phase 1: 计算 S₀ 并分组"""
    w(segments, f'\n{"="*80}')
    w(segments, 'Phase 1: 先天五行命格 S₀ 计算')
    w(segments, f'S₀ = 上市日期(阴历)三柱六字 × 五行统计 → (木,火,土,金,水)')
    w(segments, f'{"="*80}')

    s0_map, name_map, vec_groups = {}, {}, defaultdict(list)
    for sym, name, list_str in STOCKS:
        s0 = calc_s0(list_str)
        s0_map[sym] = s0
        name_map[sym] = name
        vec_groups[s0].append((sym, name))
    w(segments, f'{len(s0_map)} 只股票, {len(vec_groups)} 个不同命格组')

    # 按组大小排序
    for vec, members in sorted(vec_groups.items(), key=lambda x: -len(x[1])):
        label = vec_label(vec)
        names = ','.join(m[1] for m in members)
        w(segments, f'  S₀={str(vec):<15} {label:<22} ({len(members)}只) {names}')

    # 保存到 JSON（供后续预测使用）
    results = []
    for sym, name, list_str in STOCKS:
        s0 = calc_s0(list_str)
        results.append({'sym': sym, 'name': name, 'listing': list_str, 's0': list(s0)})
    with open(JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump({'results': results, 'count': len(results)}, f, ensure_ascii=False, indent=2)
    w(segments, f'\nS₀ 已保存至 {JSON_PATH}')

    # 写入精简版分组日志
    with open(S0_LOG_PATH, 'w', encoding='utf-8') as f:
        for vec, members in sorted(vec_groups.items(), key=lambda x: -len(x[1])):
            label = vec_label(vec)
            names = ','.join(m[1] for m in members)
            f.write(f'{str(vec):<15} {label:<22} ({len(members)}只) {names}\n')

    return s0_map, name_map, vec_groups

def predict_analysis(segments, s0_map, name_map, excess):
    """Phase 2: 同命格波动曲线预测"""
    # 分组
    vec_syms = defaultdict(list)
    for sym in excess:
        if sym in s0_map:
            vec_syms[s0_map[sym]].append(sym)

    multi_groups = {k: v for k, v in vec_syms.items() if len(v) >= 2}
    w(segments, f'\n{"="*80}')
    w(segments, f'Phase 2: 波动曲线预测 ({len(multi_groups)} 组 ≥2只)')
    w(segments, f'特征: AR(3) + 信号源多滞后(0,1,2,5,10,20,30)')
    w(segments, f'模型: Ridge + {"XGBoost" if HAS_XGB else "无"} XGBoost')
    w(segments, f'验证: Walk-forward 5折 (60%/10步进)')
    w(segments, f'{"="*80}')

    HORIZONS = [1, 2, 3, 5, 10]
    w(segments, f'{"命格":<22} {"目标":<8} {"信号源":<8} {"H+1 Ridge":<14} {"H+1 XGB":<14} {"H+5 Ridge":<14} {"H+5 XGB":<14}')
    w(segments, '-' * 80)

    results_a = []
    pair_count = 0

    for vec, members in sorted(vec_syms.items(), key=lambda x: -len(x[1])):
        if len(members) < 2: continue
        label = vec_label(vec)
        w(segments, f'\n  [{label} {len(members)}只]')

        for a in members:
            for b in members:
                if a == b or a not in excess or b not in excess: continue
                try:
                    common = sorted(set(excess[a]['dates']) & set(excess[b]['dates']))
                    if len(common) < 200: continue
                    common = common[40:]
                    di_a = {d: k for k, d in enumerate(excess[a]['dates'])}
                    di_b = {d: k for k, d in enumerate(excess[b]['dates'])}
                    td, pd_d = excess[a], excess[b]
                    all_h = {}

                    for h in HORIZONS:
                        if h >= len(common) - 120: continue
                        X, y = [], []
                        for i in range(len(common) - h):
                            feats, vt = build_features(di_a, td, di_b, pd_d, common, i, h)
                            if feats is not None and vt is not None:
                                X.append(feats); y.append(vt)
                        if len(y) < 120: continue
                        n, n_train = len(y), int(len(y) * 0.6)
                        preds_r, preds_x, y_test = [], [], []

                        for fold in range(5):
                            ts = n_train + fold * n_train // 10
                            te = min(ts + n_train // 10, n - h)
                            if te <= ts + 5: break
                            Xtr = np.array(X[:ts]); ytr = np.array(y[:ts])
                            Xte = np.array(X[ts:te]); yte = np.array(y[ts:te])
                            if len(ytr) < 60 or len(yte) < 10: continue
                            try:
                                m_r = Ridge(alpha=1.0); m_r.fit(Xtr, ytr)
                                preds_r.extend(m_r.predict(Xte).tolist())
                            except: pass
                            try:
                                if HAS_XGB:
                                    m_x = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, verbosity=0, random_state=42)
                                    m_x.fit(Xtr, ytr)
                                    preds_x.extend(m_x.predict(Xte).tolist())
                            except: pass
                            if len(y_test) < len(preds_r):
                                y_test.extend(yte.tolist())

                        if len(y_test) < 20: continue
                        # AR-only baseline
                        ar_X = np.array([x[:3] for x in X])
                        preds_ar = []
                        for fold in range(5):
                            ts = n_train + fold * n_train // 10
                            te = min(ts + n_train // 10, n - h)
                            if te <= ts + 5: break
                            Xtr = ar_X[:ts]; ytr = np.array(y[:ts])
                            Xte = ar_X[ts:te]; yte = np.array(y[ts:te])
                            if len(ytr) < 60 or len(yte) < 10: continue
                            try:
                                m_a = Ridge(alpha=1.0); m_a.fit(Xtr, ytr)
                                preds_ar.extend(m_a.predict(Xte).tolist())
                            except: pass
                        rmse_ar = np.sqrt(mean_squared_error(y_test[:len(preds_ar)], preds_ar)) if len(preds_ar) >= 20 else None
                        if rmse_ar is None or rmse_ar <= 0: continue
                        rmse_r = np.sqrt(mean_squared_error(y_test[:len(preds_r)], preds_r)) if preds_r else None
                        rmse_x = np.sqrt(mean_squared_error(y_test[:len(preds_x)], preds_x)) if preds_x else None
                        imp_r = (rmse_ar - rmse_r) / rmse_ar * 100 if rmse_r else None
                        imp_x = (rmse_ar - rmse_x) / rmse_ar * 100 if rmse_x else None
                        best = 'Ridge' if (imp_x is None or (imp_r is not None and imp_r >= imp_x)) else 'XGB'
                        all_h[h] = {'imp_r': imp_r, 'imp_x': imp_x, 'best': best}

                    if not all_h: continue
                    h1, h5 = all_h.get(1, {}), all_h.get(5, {})
                    ir = f'{h1.get("imp_r", 0):+.1f}%' if h1.get('imp_r') is not None else 'N/A'
                    ix = f'{h1.get("imp_x", 0):+.1f}%' if h1.get('imp_x') is not None else 'N/A'
                    i5r = f'{h5.get("imp_r", 0):+.1f}%' if h5.get('imp_r') is not None else 'N/A'
                    i5x = f'{h5.get("imp_x", 0):+.1f}%' if h5.get('imp_x') is not None else 'N/A'
                    tn = name_map.get(a, a[:6]); pn = name_map.get(b, b[:6])
                    w(segments, f'  {label:<22} {tn:<8} {pn:<8} {ir:<14} {ix:<14} {i5r:<14} {i5x:<14}')
                    pair_count += 1
                    results_a.append({'label': label, 'target': tn, 'pred': pn, 'h1_imp_r': h1.get('imp_r'), 'h1_imp_x': h1.get('imp_x')})
                except:
                    continue

    w(segments, f'\n  {pair_count} 配对完成')

    # 汇总
    if results_a:
        w(segments, f'\n{"="*80}')
        w(segments, '【汇总】')
        for h in HORIZONS:
            ir = [r.get(f'h{h}_imp_r') for r in results_a if r.get(f'h{h}_imp_r') is not None]
            ix = [r.get(f'h{h}_imp_x') for r in results_a if r.get(f'h{h}_imp_x') is not None]
            if ir:
                w(segments, f'  H+{h} Ridge: avg={np.mean(ir):+.2f}% med={np.median(ir):+.2f}% pos={sum(1 for x in ir if x>0)}/{len(ir)}')
            if ix:
                w(segments, f'  H+{h} XGBoost: avg={np.mean(ix):+.2f}% med={np.median(ix):+.2f}% pos={sum(1 for x in ix if x>0)}/{len(ix)}')

        w(segments, f'\n  按命格组 H+1:')
        by_label = defaultdict(list)
        for r in results_a: by_label[r['label']].append(r)
        for label, items in sorted(by_label.items(), key=lambda x: -len(x[1])):
            h1r = [r['h1_imp_r'] for r in items if r['h1_imp_r'] is not None]
            h1x = [r['h1_imp_x'] for r in items if r['h1_imp_x'] is not None]
            if h1r:
                line = f'    {label:<22} {len(items):<4}对 Ridge={np.mean(h1r):+.1f}%'
                if h1x: line += f' XGB={np.mean(h1x):+.1f}%'
                w(segments, line)

    return pair_count


def main():
    segments = []
    w(segments, '=' * 80)
    w(segments, '先天五行命格 × 波动率预测系统')
    w(segments, f'运行时间: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    w(segments, f'XGBoost: {"可用" if HAS_XGB else "不可用（安装: pip install xgboost）"}')
    w(segments, f'股票池: {len(STOCKS)} 只')
    w(segments, '=' * 80)

    # Phase 1: S₀ 命格计算
    w(segments, '\n[Phase 1] 计算先天五行命格 S₀...')
    s0_map, name_map, vec_groups = s0_analysis(segments)

    # Phase 2: 数据加载
    w(segments, '\n[Phase 2] 加载行情数据...')
    stock_data = load_or_download_data()

    # Phase 3: 超额波动率
    w(segments, '\n[Phase 3] 计算超额波动率（去沪深300 β）...')
    excess = calc_excess_vol(stock_data)
    w(segments, f'  有效: {len(excess)} 只')

    # Phase 4: 预测
    w(segments, '\n[Phase 4] 波动曲线预测...')
    pair_count = predict_analysis(segments, s0_map, name_map, excess)

    w(segments, f'\nDone! {pair_count} 配对完成')
    write_report(segments)
    print(f'\n✅ 报告已写入 {REPORT_PATH}')


if __name__ == '__main__':
    import sys, traceback
    try:
        if '--s0' in sys.argv and '--predict' not in sys.argv:
            segments = []
            s0_analysis(segments)
            write_report(segments)
            print(f'✅ S₀ 分析完成 → {S0_LOG_PATH}')
        elif '--predict' in sys.argv and '--s0' not in sys.argv:
            if not os.path.exists(JSON_PATH):
                print('❌ 请先运行 --s0 生成命格数据')
                sys.exit(1)
            with open(JSON_PATH, encoding='utf-8') as f:
                data = json.load(f)
            s0_map = {}
            name_map = {}
            for r in data['results']:
                sym = r['sym']
                s0_map[sym] = tuple(r['s0'])
                name_map[sym] = r['name']
            segments = []
            stock_data = load_or_download_data()
            excess = calc_excess_vol(stock_data)
            predict_analysis(segments, s0_map, name_map, excess)
            write_report(segments)
            print(f'✅ 预测完成 → {REPORT_PATH}')
        else:
            main()
    except Exception as e:
        print(f'\n❌ FATAL: {e}')
        traceback.print_exc()
