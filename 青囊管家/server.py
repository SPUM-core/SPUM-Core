"""
青囊生活管家 — 本地 API 服务器（v3.1 · +音频闻诊）
==================================================
运行方式：python server.py
默认端口：8122
"""

import json
import os
import re
import uuid
import tempfile
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("qingnang")

# ── 配置 ──

PORT = int(os.environ.get("QINGNANG_PORT", "8122"))
DATA_DIR = Path(__file__).parent / "data"
PROMPT_FILE = Path(__file__).parent / "prompt.md"
KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
STATIC_DIR = Path(__file__).parent / "static"
TEMP_DIR = Path(__file__).parent / "temp_audio"

LLM_API_URL = os.environ.get(
    "LLM_API_URL",
    "https://api.deepseek.com/v1/chat/completions",
)
LLM_API_KEY = os.environ.get("LLM_API_KEY", "sk-352d395ed25645e293a7db69fa45e2a1")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")

# 备用模型名（如果 deepseek-chat 返回 401 自动切换）
_llm_model_used = LLM_MODEL
_llm_model_fallback = "deepseek-reasoner"

# ── 初始化 ──

DATA_DIR.mkdir(exist_ok=True)
KNOWLEDGE_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)
system_prompt = PROMPT_FILE.read_text(encoding="utf-8")

# 加载知识库文件（P0 必加载，P1/P2 按需）
_knowledge_cache = {}

def load_knowledge():
    """加载所有知识库文件到缓存"""
    if _knowledge_cache:
        return _knowledge_cache
    for f in sorted(KNOWLEDGE_DIR.glob("*.md")):
        _knowledge_cache[f.stem] = {
            "path": f.name,
            "content": f.read_text(encoding="utf-8"),
            "size": f.stat().st_size,
        }
    return _knowledge_cache

def build_knowledge_appendix() -> str:
    """构建知识附录（P0 文件全量注入，P1/P2 仅索引）"""
    cache = load_knowledge()
    appendix_parts = ["\n\n## 【知识库附录】"]
    p0_files = ["spum-黄帝内经", "spum-神农本草经", "五形-skill"]
    for name, info in cache.items():
        if name in p0_files and info["size"] < 80000:
            appendix_parts.append(f"\n### {info['path']}（已加载）\n{info['content']}")
    return "\n".join(appendix_parts)

# 系统提示 = prompt.md + P0 知识全量注入
_knowledge_appendix = build_knowledge_appendix()
system_prompt += _knowledge_appendix

app = FastAPI(title="青囊生活管家")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── 数据模型 ──

class ChatRequest(BaseModel):
    message: str
    account_id: str = ""


class ChatResponse(BaseModel):
    reply: str
    account_id: str


# ── 数据持久化 ──

def _account_path(account_id: str) -> Path:
    return DATA_DIR / f"account_{account_id}.json"


def _history_path(account_id: str) -> Path:
    return DATA_DIR / f"history_{account_id}.json"


def load_account(account_id: str) -> Optional[dict]:
    path = _account_path(account_id)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def save_account(account: dict):
    path = _account_path(account["account_id"])
    path.write_text(json.dumps(account, ensure_ascii=False, indent=2), encoding="utf-8")


def load_history(account_id: str) -> list:
    path = _history_path(account_id)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


def save_history(account_id: str, messages: list):
    path = _history_path(account_id)
    path.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_id(prefix="id") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


# ── LLM ──

import httpx


async def call_llm(messages: list[dict]) -> str:
    key = LLM_API_KEY
    print(f"[DEBUG] LLM_API_KEY prefix: {key[:12]}...  | URL: {LLM_API_URL}")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 2048,
    }
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            resp = await client.post(LLM_API_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"抱歉，调用 AI 引擎时出错：{str(e)}"


# ── [PROFILE] 块解析 ──

def parse_profile_block(text: str) -> tuple[str, dict, str]:
    """
    从 LLM 输出中提取 [PROFILE] 块。

    返回: (clean_text, members_dict, last_round)
      - clean_text: 去掉 [PROFILE] 块后的文本
      - members_dict: {key: {name, relation, birth, s, note}}
      - last_round: 上一轮摘要字符串

    [PROFILE] 格式：
        [PROFILE]
        self: 张莹莹|self|1987-07-24-15|0,1,2,0,-1|肾水不足+土壅
        宝宝: 宝宝|child|2019-03-15|1,0,0,-1,1|金↓易烦躁
        last_round: 用户说吃了百合好转
        [/PROFILE]
    """
    pattern = r'\[PROFILE\](.*?)\[/PROFILE\]'
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return text, {}, ""

    block = match.group(1).strip()
    clean_text = text[:match.start()] + text[match.end():]
    clean_text = clean_text.strip()

    members = {}
    last_round = ""

    for line in block.split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.startswith('last_round:'):
            last_round = line[len('last_round:'):].strip()
            continue
        if ':' not in line:
            continue

        key, value = line.split(':', 1)
        key = key.strip()
        value = value.strip()

        # 解析: name|relation|birth|s|note
        parts = value.split('|')
        member = {
            "name": parts[0] if len(parts) > 0 else key,
            "relation": parts[1] if len(parts) > 1 else "self",
            "birth": parts[2] if len(parts) > 2 else "",
            "s": parts[3] if len(parts) > 3 else "",
            "note": parts[4] if len(parts) > 4 else "",
        }
        members[key] = member

    return clean_text, members, last_round


# ── 上下文构建 ──

def build_context(account: dict, history: list) -> list[dict]:
    """
    构建发送给 LLM 的消息列表。

    - System prompt（含档案上下文）
    - 最近 3 轮历史
    - 当前用户输入
    """
    members = account.get("members", {})
    last_round = account.get("last_round", "")

    # 构建档案上下文
    context_lines = [system_prompt]

    if members:
        context_lines.append("\n\n## 当前家庭档案")
        for key, m in members.items():
            context_lines.append(f"{key}: {m['name']}|{m['relation']}|{m.get('birth','')}|{m.get('s','')}|{m.get('note','')}")

    if last_round:
        context_lines.append(f"\n\n## 上轮摘要\n{last_round}")

    messages = [{"role": "system", "content": "\n".join(context_lines)}]

    # 最近 3 轮历史（最多 6 条消息，只保留前 3 对 user/assistant）
    recent = history[-6:]
    for h in recent:
        messages.append(h)

    return messages


# ── 农历转换工具 ──

from zhdate import ZhDate


class LunarDateInput(BaseModel):
    year: int
    month: int
    day: int
    hour: int = 0


@app.get("/api/convert/lunar")
async def convert_lunar(year: int, month: int, day: int, hour: int = 0):
    """农历转公历"""
    try:
        lunar = ZhDate(year, month, day)
        solar = lunar.to_datetime()
        solar_str = solar.strftime("%Y年%m月%d日")
        if hour:
            solar_str += f" {hour}时"
        return {"solar": solar_str, "lunar": f"{year}年{month}月{day}日"}
    except Exception as e:
        return {"error": f"转换失败：{str(e)}"}


# ── 音频闻诊 ──

ALLOWED_AUDIO_EXTS = {".m4a", ".mp3", ".wav", ".mp4", ".aac", ".ogg"}

@app.post("/api/analyze/voice")
async def analyze_voice(file: UploadFile = File(...)):
    """
    音频闻诊分析。

    上传语音文件（.m4a/.mp3/.wav 等），返回声域分型 + 五音频段能量 + 闻诊 S 向量。

    对应青囊agent.md §4.6 音频数字信号分析协议。
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_AUDIO_EXTS:
        return {"error": f"不支持的文件格式: {ext}，支持: {', '.join(ALLOWED_AUDIO_EXTS)}"}

    # 保存到临时目录
    save_path = TEMP_DIR / f"{uuid.uuid4().hex}{ext}"
    try:
        content = await file.read()
        save_path.write_bytes(content)
        logger.info(f"音频文件已保存: {save_path} ({len(content)} bytes)")

        # 调用 voice_analyzer
        from voice_analyzer import analyze
        result = analyze(str(save_path))
        return result
    except ModuleNotFoundError:
        return {
            "error": "音频分析模块未安装",
            "detail": "请安装依赖: pip install librosa numpy imageio-ffmpeg soundfile",
        }
    except Exception as e:
        logger.error(f"音频分析失败: {e}")
        return {"error": f"音频分析失败: {str(e)}"}
    finally:
        # 清理临时文件
        if save_path.exists():
            try:
                save_path.unlink()
            except OSError:
                pass


# ── Web 界面 ──

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>青囊生活管家</h1><p>请先构建 static/index.html</p>")


@app.get("/api/prompt")
async def get_prompt():
    return {"prompt": system_prompt}


# ── 聊天 ──

@app.post("/api/chat")
async def chat(req: ChatRequest):
    account_id = req.account_id or generate_id("acc")

    # 加载或创建账户
    account = load_account(account_id)
    is_new = account is None
    if is_new:
        account = {
            "account_id": account_id,
            "members": {},           # {key: {name, relation, birth, s, note}}
            "last_round": "",        # 上轮摘要
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

    # 加载历史（已由覆盖机制限制为最近 3 轮）
    history = load_history(account_id)

    # 构建消息
    messages = build_context(account, history)
    messages.append({"role": "user", "content": req.message})

    # 调用 LLM
    reply = await call_llm(messages)

    # 解析 [PROFILE] 块
    clean_reply, new_members, new_last_round = parse_profile_block(reply)

    # 如果 LLM 返回了错误信息（不含 [PROFILE]），直接返回
    if not new_members and not new_last_round and reply.startswith("抱歉"):
        if not account_id:
            account_id = generate_id("acc")
        return ChatResponse(reply=reply, account_id=account_id)

    # 更新成员档案
    if new_members:
        for key, m in new_members.items():
            account["members"][key] = m

    # 更新 last_round（LLM 未提供时自动生成摘要）
    if new_last_round:
        account["last_round"] = new_last_round
    else:
        # 自动摘要：截取用户输入前 40 字
        summary = req.message.strip()[:40]
        if len(req.message.strip()) > 40:
            summary += "…"
        account["last_round"] = f"用户说: {summary}"

    # 保存历史（控制数量：只保留最近 3 轮 = 6 条消息）
    history.append({"role": "user", "content": req.message})
    history.append({"role": "assistant", "content": reply})  # 保存原始回复含 [PROFILE]
    if len(history) > 6:
        history = history[-6:]
    save_history(account_id, history)

    account["updated_at"] = datetime.now().isoformat()
    save_account(account)

    return ChatResponse(reply=clean_reply, account_id=account_id)


# ── 重置 ──

@app.post("/api/chat/reset/{account_id}")
async def reset_chat(account_id: str):
    """重置会话（不清除档案）"""
    path = _history_path(account_id)
    if path.exists():
        path.unlink()

    # 也清除 last_round，避免断档
    account = load_account(account_id)
    if account:
        account["last_round"] = ""
        save_account(account)

    return {"status": "reset"}


# ── 启动 ──

if __name__ == "__main__":
    print(f"[青囊] 生活管家 v3.1 · [PROFILE]协议 + 音频闻诊")
    print(f"  本地地址: http://localhost:{PORT}")
    print(f"  数据目录: {DATA_DIR}")
    print(f"  LLM: {LLM_MODEL}")
    if not LLM_API_KEY:
        print(f"  ⚠️ 未设置 LLM_API_KEY")
    print()
    uvicorn.run(app, host="0.0.0.0", port=PORT)
