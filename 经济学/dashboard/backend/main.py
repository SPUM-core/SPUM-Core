"""AI 股票分析后端 — FastAPI

环境变量:
  OPENAI_API_KEY  — 设置后启用 LLM 对话分析
  OPENAI_BASE_URL — 非标准 API 地址 (可选)
"""
import json, os, math
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'vis_data.json')

app = FastAPI(title='五行命格 AI 分析')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

# ── Load data ──
_vis_data = None
def get_data():
    global _vis_data
    if _vis_data is None:
        with open(DATA_PATH, encoding='utf-8') as f:
            _vis_data = json.load(f)
    return _vis_data

# ── Models ──
class ChatRequest(BaseModel):
    sym: str
    message: str
    history: list[dict] = []

class ChatResponse(BaseModel):
    reply: str
    analysis: Optional[dict] = None

# ============================================================
#  Utility: build stock context
# ============================================================
def build_context(sym: str):
    data = get_data()
    stock = next((s for s in data['stocks'] if s['sym'] == sym), None)
    if not stock:
        raise HTTPException(404, 'Stock not found')

    vol = data['vol_data'].get(sym, {})
    vol_vals = vol.get('vol', [])
    pairs_target = [p for p in data['pairs'] if p['target'] == stock['name']]
    pairs_source = [p for p in data['pairs'] if p['source'] == stock['name']]
    group = next((g for g in data['groups'] if stock['name'] in g['members']), None)

    ctx = {
        'name': stock['name'],
        'sym': stock['sym'],
        'listing': stock['listing'],
        's0': stock['s0'],
        's0_normalized': stock['s0_normalized'],
        'label': stock['label'],
        'vol_stats': compute_vol_stats(vol_vals),
        'recent_vol': vol_vals[-60:] if len(vol_vals) >= 60 else vol_vals,
        'pairs_target': [
            {'source': p['source'], 'source_sym': p['source_sym'], 'h1_xgb': p['h1_xgb']}
            for p in pairs_target if p['h1_xgb'] is not None
        ],
        'pairs_source': [
            {'target': p['target'], 'target_sym': p['target_sym'], 'h1_xgb': p['h1_xgb']}
            for p in pairs_source if p['h1_xgb'] is not None
        ],
        'group_mates': group['members'] if group and len(group['members']) > 1 else [],
        'group_label': group['label'] if group else stock['label'],
    }
    return ctx

def compute_vol_stats(vals):
    if not vals:
        return {'mean': 0, 'std': 0, 'recent_mean': 0, 'min': 0, 'max': 0}
    recent = vals[-20:]
    return {
        'mean': round(float(np_mean(vals)), 6),
        'std': round(float(np_std(vals)), 6),
        'recent_mean': round(float(np_mean(recent)), 6),
        'min': round(float(min(vals)), 6),
        'max': round(float(max(vals)), 6),
    }

def np_mean(arr):
    return sum(arr) / len(arr) if arr else 0

def np_std(arr):
    if len(arr) < 2: return 0
    m = np_mean(arr)
    return math.sqrt(sum((x - m) ** 2 for x in arr) / (len(arr) - 1))

# ============================================================
#  Structured analysis (no LLM needed)
# ============================================================
def run_structured_analysis(ctx: dict) -> str:
    lines = []
    s = ctx['s0']
    wx = ['木', '火', '土', '金', '水']
    wx_colors = {'木':'🟢','火':'🔴','土':'🟡','金':'⚪','水':'🔵'}

    # S0 analysis
    lines.append(f"## {ctx['name']}（{ctx['sym']}）— {ctx['label']}")
    max_elem = max(range(5), key=lambda i: s[i])
    min_elem = min(range(5), key=lambda i: s[i])
    missing = [wx[i] for i, v in enumerate(s) if v == 0]
    strong = [wx[i] for i, v in enumerate(s) if v == max(s)]
    s0_desc = f"命格：{''.join(wx_colors[w] for w in wx)}\n"
    s0_desc += f"旺 {strong[0]}（{max(s)}分），"
    if missing:
        s0_desc += f"缺 {''.join(missing)}，五行{'不全' if len(missing)>=2 else '偏弱'}"
    else:
        s0_desc += f"五行俱全"
    lines.append(s0_desc)

    # Vol analysis
    vs = ctx['vol_stats']
    vol_level = '高' if vs['mean'] > 0.025 else '中' if vs['mean'] > 0.015 else '低'
    vol_trend = '上升' if vs['recent_mean'] > vs['mean'] else '下降' if vs['recent_mean'] < vs['mean'] else '平稳'
    lines.append(f"\n波动率：{vol_level}（均值 {vs['mean']*100:.2f}%），近20日 {vol_trend}（{vs['recent_mean']*100:.2f}%）")
    lines.append(f"波动区间 [{vs['min']*100:.2f}%, {vs['max']*100:.2f}%]")

    # Signal analysis
    if ctx['pairs_target']:
        best = max(ctx['pairs_target'], key=lambda p: p['h1_xgb'] or 0)
        avg_signal = np_mean([p['h1_xgb'] for p in ctx['pairs_target'] if p['h1_xgb']])
        lines.append(f"\n可预测性：{len(ctx['pairs_target'])}个信号源，平均提升 {avg_signal:.1f}%")
        lines.append(f"最强信号：{best['source']} → +{best['h1_xgb']:.1f}%")
    else:
        lines.append("\n可预测性：暂无配对数据")

    if ctx['pairs_source']:
        best_src = max(ctx['pairs_source'], key=lambda p: p['h1_xgb'] or 0)
        lines.append(f"作为信号源：可预测 {len(ctx['pairs_source'])}只股票，最佳 → {best_src['target']} +{best_src['h1_xgb']:.1f}%")

    # Group
    if ctx['group_mates']:
        lines.append(f"\n同命格组（{ctx['group_label']}）：{'、'.join(ctx['group_mates'])}")

    # S0 correlation with vol
    # 五行偏废 vs 波动
    imbalance = max(s) - min([v for v in s if v > 0] or [0])
    if imbalance >= 2:
        lines.append(f"\n五行偏废度：{imbalance}（偏废较大，波动易受外部环境影响）")
    else:
        lines.append(f"\n五行偏废度：{imbalance}（较为均衡，波动相对独立）")

    return '\n'.join(lines)

# ============================================================
#  LLM analysis (requires OPENAI_API_KEY)
# ============================================================
def build_llm_context(ctx: dict) -> str:
    vs = ctx['vol_stats']
    s0_str = ' | '.join(f'{wx}={v}' for wx, v in zip(['木','火','土','金','水'], ctx['s0']))
    pairs_str = '\n'.join(
        f"  - {p['source']} → {ctx['name']}: +{p['h1_xgb']:.1f}%"
        for p in sorted(ctx['pairs_target'], key=lambda x: -x['h1_xgb'])[:5]
    ) if ctx['pairs_target'] else '  无'
    return f"""## 股票：{ctx['name']}（{ctx['sym']}）
- 上市日期：{ctx['listing']}
- 五行命格 S₀：{ctx['label']}
  {s0_str}
- 同命格组：{', '.join(ctx['group_mates']) if ctx['group_mates'] else '无'}
- 超额波动率均值：{vs['mean']*100:.2f}% | 近20日：{vs['recent_mean']*100:.2f}%
- 波动标准差：{vs['std']*100:.2f}%
- 配对预测信号：
{pairs_str}"""

async def llm_chat(system: str, user: str, history: list[dict]) -> str:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        api_key=os.environ.get('OPENAI_API_KEY'),
        base_url=os.environ.get('OPENAI_BASE_URL'),
    )
    msgs = [{'role': 'system', 'content': system}]
    for h in history[-8:]:
        msgs.append({'role': h.get('role', 'user'), 'content': h.get('content', '')})
    msgs.append({'role': 'user', 'content': user})

    resp = await client.chat.completions.create(
        model=os.environ.get('OPENAI_MODEL', 'gpt-4o-mini'),
        messages=msgs,
        temperature=0.7,
        max_tokens=1000,
    )
    return resp.choices[0].message.content or ''

SYSTEM_PROMPT = """你是 SPUM 五行命格理论的股票分析师。你精通：
1. 先天五行命格 S₀（木火土金水）对股票波动特征的影响
2. SPUM 离散关系网络理论在金融时间序列中的应用
3. 同命格股票之间的波动关联性分析

分析原则：
- 基于用户提供的股票数据和预测结果给出客观分析
- 解释五行命格如何影响波动模式
- 指出信号源与标的之间的命格关联
- 用数据说话，避免模糊判断
- 回答简洁专业，使用中文"""

# ============================================================
#  Endpoints
# ============================================================
@app.get('/api/stock/{sym}')
def get_stock_info(sym: str):
    ctx = build_context(sym)
    return ctx

@app.post('/api/analyze')
def analyze_stock(req: ChatRequest):
    ctx = build_context(req.sym)
    # Always do structured analysis
    structured = run_structured_analysis(ctx)

    # If LLM available, use it for chat
    if os.environ.get('OPENAI_API_KEY'):
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            llm_ctx = build_llm_context(ctx)
            reply = loop.run_until_complete(llm_chat(
                SYSTEM_PROMPT,
                f"请分析以下股票，并回答用户的问题：{req.message}\n\n数据：\n{llm_ctx}",
                req.history,
            ))
            loop.close()
            return ChatResponse(reply=reply)
        except Exception as e:
            return ChatResponse(reply=structured + f'\n\n⚠ LLM 调用失败: {e}')
    else:
        # Structured analysis as reply
        return ChatResponse(reply=structured)

@app.get('/api/health')
def health():
    return {'status': 'ok', 'data_stocks': len(get_data()['stocks'])}

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT', 8765))
    uvicorn.run(app, host='0.0.0.0', port=port)
