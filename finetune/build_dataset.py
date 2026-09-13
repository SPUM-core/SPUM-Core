# -*- coding: utf-8 -*-
"""build_dataset.py — SPUM 本地模型微调数据集构建器（阶段一：数据）。

设计原则
--------
1. **零编造**：每条样本的 assistant 回答都来自仓库既有文本，片段级可溯源
   （样本带 `source` = `相对路径#L起-止`）。不生成知识库里没有的说法。
2. **确定性**：同一份仓库 → 同一份数据集（固定种子、稳定排序、哈希去重）。
3. **四类任务**（用户裁决）：
   - `concept_def`  概念释义      ：术语 → SPUM 范式定义。七个来源：nodes.txt /
     词汇表 / `knowledge.md` 释义表（§六 词汇、§八 感受词、§九 共识定义）/
     模块 `spum-*.md` 还原表（`| 概念 | SPUM 图论对应 |`）/ 模块 skill 核心范式 /
     反例库「✅ 正确」段 / faq「一句话结论」
   - `rewrite`      旧范式改写    ：反例库的「错误段落」→「为什么错 + 正确表述」
   - `detect`       自检判别      ：判定一段文字是否属于 SPUM 范式（正例取自语料，负例取自反例库）
   - `reduction`    跨域归约      ：领域对象 → ⟨P, ε⟩ 的图论表达（模块 `spum-*.md` 核心命题）
   - `code_skeleton` 代码骨架     ：函数语义说明 → 函数源码（openSPUM/src，ast 解析）
4. **防泄漏**：分层自检——答案层（同一归一化答案不得跨 train/val）与题面层
   （同一归一化问题不得跨 train/val）。`concept_def` 按**术语**分组切分：同一术语的
   多个释义来源是同一道题的多个参考答案，必须同侧。

用法
----
    python build_dataset.py                 # 构建 train/val + report
    python build_dataset.py --out data      # 指定输出目录
    python build_dataset.py --val-ratio 0.15
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]

# 统一 system 提示：训练与评测两侧必须用**同一份**，否则前后对比不成立。
SYSTEM_PROMPT = (
    "你是 SPUM（关系网络本体论）的推理助手。回答须守住四条底线："
    "① 关系先于实体——不把任何对象当作先于关系的独立实体；"
    "② 离散先于连续——不用连续流、无限小、终态可达一类隐喻；"
    "③ 用 SPUM 术语（空间粒子、边、度数 deg、σ、V⁺/V⁻、离散帧、晶子、五形）"
    "替代旧范式词汇（力、时空、概率云、点粒子、宇宙常数）；"
    "④ 不暗示无限增长，也不暗示宇宙有最终静止态。"
)

# 领域模块目录（`<dir>/**/spum-*.md`）——「跨域归约」与「自检判别」正例的语料来源。
MODULE_DIRS = ("社会学", "经济学", "语言学", "图论", "数学",
               "儒释道哲学", "五行", "物理学", "宇宙学")


def _iter_module_docs():
    """枚举领域模块文档（稳定排序，跳过草稿目录）。"""
    for name in MODULE_DIRS:
        base = ROOT / name
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("spum-*.md")):
            if "草稿" in path.parts:
                continue
            yield path


# ---------------------------------------------------------------- 工具

def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _lines(text: str) -> list[str]:
    return text.splitlines()


def _span(lines: list[str], start: int, end: int) -> str:
    """1-based 行号区间取文本（含端点），并去掉首尾空行。"""
    seg = lines[start - 1:end]
    while seg and not seg[0].strip():
        seg.pop(0)
    while seg and not seg[-1].strip():
        seg.pop()
    return "\n".join(seg).strip()


def _trim(text: str, limit: int = 1200) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    for sep in ("\n\n", "\n", "。", "；"):
        idx = cut.rfind(sep)
        if idx > limit * 0.6:
            return cut[: idx + len(sep)].strip()
    return cut.strip()


def _norm(text: str) -> str:
    """归一化用于去重：去空白、去掉 markdown 加粗标记。"""
    t = text.replace("**", "").replace("`", "")
    return re.sub(r"\s+", "", t)


def _clean_md_cell(cell: str) -> str:
    return cell.strip().strip("|").replace("**", "").strip()


def _parse_tables(text: str) -> list[tuple[list[str], list[list[str]], int, list[int]]]:
    """解析 markdown 表格 → [(headers, rows, header_ln, row_lns)]（行号 1-based）。

    `row_lns[i]` 对应 `rows[i]` 的源文件行号——样本要保持「片段级可溯源」，
    只给表头行号不够。
    """
    tables: list[tuple[list[str], list[list[str]], int, list[int]]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("|") and i + 1 < len(lines) \
                and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            headers = [c.strip() for c in line.strip().strip("|").split("|")]
            rows: list[list[str]] = []
            row_lns: list[int] = []
            j = i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                cells = [c.strip() for c in lines[j].strip().strip("|").split("|")]
                if len(cells) == len(headers):
                    rows.append(cells)
                    row_lns.append(j + 1)
                j += 1
            tables.append((headers, rows, i + 1, row_lns))
            i = j
        else:
            i += 1
    return tables


# ---------------------------------------------------------------- 样本容器

class Sample:
    __slots__ = ("task", "prompt", "answer", "source", "extra")

    def __init__(self, task: str, prompt: str, answer: str, source: str, **extra):
        self.task = task
        self.prompt = prompt.strip()
        self.answer = answer.strip()
        self.source = source
        self.extra = extra

    def to_record(self, system: bool = True) -> dict:
        messages = []
        if system:
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
        messages.append({"role": "user", "content": self.prompt})
        messages.append({"role": "assistant", "content": self.answer})
        rec = {"task": self.task, "source": self.source, "messages": messages}
        rec.update(self.extra)
        rec["id"] = _make_id(rec)
        return rec


def _make_id(rec: dict) -> str:
    h = hashlib.sha1((rec["task"] + "|" + rec["messages"][-2]["content"]
                      + "|" + rec["source"]).encode("utf-8")).hexdigest()[:12]
    return f"{rec['task']}-{h}"


# ---------------------------------------------------------------- T1 概念释义

PROMPT_T1 = [
    "用 SPUM 本体论解释「{term}」。",
    "在 SPUM 中，「{term}」指什么？",
    "什么是「{term}」？请按 SPUM 范式回答。",
]

# 「概念 → 定义」型表格的释义列关键字：列名不含这些词，说明该列不是定义列。
_DEF_HEADER_KEYWORDS = ("定义", "对应", "翻译", "含义", "内涵")
# 列名出现这些词，说明该表的语义方向与「概念→定义」不同（改名表/公式表），整表跳过。
_TABLE_EXCLUDE_KEYWORDS = ("原名", "新名称", "公式")
# 单份文档最多取多少行放进取值——防某一部长文档在池中压倒性占比。
_MAX_PER_DOC = 6
# 模块还原表总条数上限。实测：不设限时该来源 307 条，把 concept_def 顶到全集的 64%，
# 语料配比失衡会挤压 code_skeleton/rewrite。上限 + 跨文档轮转取值，既保底模块覆盖面，
# 又不让 concept_def 独占训练信号。
_MODULE_TOTAL_CAP = 150
# 模块还原表单行释义的最短长度：低于此值的单元格是"标签化速记"而非定义，
# 既训不出表述、也会把池子撑成其他任务的 5 倍（实测 71%）。
_MODULE_DESC_MIN = 16


def _valid_term(term: str) -> bool:
    """术语是否可作为提问对象（防表格里混进的编号行/空行/长句）。"""
    if not (2 <= len(term) <= 24):
        return False
    if any(ch in term for ch in "\n\t|"):
        return False
    if term.startswith(("---", "#", "-", "*", "：")):
        return False
    return bool(re.search(r"[\u4e00-\u9fffA-Za-z]", term))


def _def_column(headers: list[str]) -> int | None:
    """定位释义列：优先第一个含「定义/对应/翻译/含义/内涵」的列；退化到第 2 列（含 SPUM/图论/拓扑）。"""
    for i, h in enumerate(headers):
        if i == 0:
            continue
        if any(k in _clean_md_cell(h) for k in _DEF_HEADER_KEYWORDS):
            return i
    if len(headers) >= 2 and any(k in _clean_md_cell(headers[1])
                                 for k in ("SPUM", "图论", "拓扑")):
        return 1
    return None


def extract_concept_nodes() -> list[Sample]:
    """network/nodes.txt：`ID | 名 | 定义`。"""
    path = ROOT / "network" / "nodes.txt"
    lines = _lines(_read(path))
    out: list[Sample] = []
    for i, line in enumerate(lines, start=1):
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|", 2)]
        if len(parts) != 3:
            continue
        nid, term, definition = parts
        if not re.match(r"^N\d{3}$", nid) or not term or not definition:
            continue
        out.append(Sample("concept_def", PROMPT_T1[1].format(term=term),
                          definition, f"{_rel(path)}#L{i}", term=term, origin="nodes"))
    return out


def extract_concept_vocabulary() -> list[Sample]:
    """`.trae/rules/spum-vocabulary.md` 的 2 列表格与「概念|图论定义|公理来源」3 列表格。"""
    path = ROOT / ".trae" / "rules" / "spum-vocabulary.md"
    text = _read(path)
    out: list[Sample] = []
    for headers, rows, _hln, row_lns in _parse_tables(text):
        h = [x.strip() for x in headers]
        if len(h) < 2:
            continue
        head0 = _clean_md_cell(h[0])
        if head0 in {"你要表达…", "你要表达"}:
            continue
        if head0 not in {"概念", "**形**", "形"}:
            continue
        for r, ln in zip(rows, row_lns):
            term = _clean_md_cell(r[0])
            if not term or term.startswith("---"):
                continue
            desc = _clean_md_cell(r[1])
            if len(h) >= 3:
                extra = _clean_md_cell(r[2])
                if extra and extra not in {"---"} and "公理" in _clean_md_cell(h[2]):
                    desc = f"{desc}（公理来源：{extra}）"
            if len(desc) < 12:
                continue
            out.append(Sample("concept_def", PROMPT_T1[1].format(term=term), desc,
                              f"{_rel(path)}#L{ln}", term=term, origin="vocabulary"))
    return out


def extract_concept_knowledge() -> list[Sample]:
    """`knowledge.md` 的三处释义表：

    - §六 本体论词汇：`| 概念 | 定义 |`（基本构成 / 演化机制 / 结构属性 / 认知方法论）
    - §八 身体感受词汇：`| 感受词 | 拓扑映射 |`
    - §九 共识定义：`| 节点 | 标签 | 核心含义 |`

    这些表此前完全没被取用；而 `knowledge.md` 正是现有 concept_def 参考答案的
    权威来源——池子太小的直接原因就是"只取了它的一句话，没取它的表"。
    """
    path = ROOT / "knowledge.md"
    text = _read(path)
    rel = _rel(path)
    out: list[Sample] = []
    for headers, rows, _hln, row_lns in _parse_tables(text):
        if len(headers) < 2:
            continue
        joined = " ".join(_clean_md_cell(h) for h in headers)
        if any(k in joined for k in _TABLE_EXCLUDE_KEYWORDS):
            continue
        h0 = _clean_md_cell(headers[0])
        if h0 == "节点" and len(headers) >= 3:
            for r, ln in zip(rows, row_lns):
                nid, term, desc = (_clean_md_cell(x) for x in r[:3])
                if not re.match(r"^N\d{3}$", nid) or not _valid_term(term):
                    continue
                if len(desc) < 12:
                    continue
                out.append(Sample("concept_def", PROMPT_T1[1].format(term=term), desc,
                                  f"{rel}#L{ln}", term=term, origin="knowledge_nodes"))
            continue
        if h0 not in {"概念", "感受词", "**形**", "形"}:
            continue
        ci = _def_column(headers)
        if ci is None:
            continue
        for r, ln in zip(rows, row_lns):
            term, desc = _clean_md_cell(r[0]), _clean_md_cell(r[ci])
            if not _valid_term(term) or len(desc) < 12:
                continue
            out.append(Sample("concept_def", PROMPT_T1[1].format(term=term), desc,
                              f"{rel}#L{ln}", term=term, origin="knowledge_vocab"))
    return out


def extract_concept_module_tables() -> list[Sample]:
    """各模块 `spum-*.md` 的「领域概念 → SPUM 图论对应」表（含 `## [L0] 还原表`）。

    这是释义型文本的**大宗**来源：每份模块文档末尾的还原表逐条给出
    「概念 → 它的 ⟨P, ε⟩ 表达」，本身就是「术语 → 定义」样本。
    取值按「每文档限行数 + 跨文档轮转 + 总量上限」，保证模块覆盖面而不让配比失衡。
    """
    per_doc: list[list[Sample]] = []
    for path in _iter_module_docs():
        rel = _rel(path)
        doc_out: list[Sample] = []
        for headers, rows, _hln, row_lns in _parse_tables(_read(path)):
            if len(doc_out) >= _MAX_PER_DOC or len(headers) < 2:
                continue
            joined = " ".join(_clean_md_cell(h) for h in headers)
            if any(k in joined for k in _TABLE_EXCLUDE_KEYWORDS):
                continue
            h0 = _clean_md_cell(headers[0])
            if not (h0 == "概念" or h0.endswith("概念") or h0 in {"术语", "对象"}):
                continue
            ci = _def_column(headers)
            if ci is None:
                continue
            for r, ln in zip(rows, row_lns):
                if len(doc_out) >= _MAX_PER_DOC:
                    break
                term, desc = _clean_md_cell(r[0]), _clean_md_cell(r[ci])
                if not _valid_term(term) or len(desc) < _MODULE_DESC_MIN:
                    continue
                doc_out.append(Sample("concept_def", PROMPT_T1[1].format(term=term), desc,
                                      f"{rel}#L{ln}", term=term, origin="module_table"))
        if doc_out:
            per_doc.append(doc_out)
    # 轮转取值：每轮每份文档各取一条，直到上限。所有模块等权进入，
    # 不会因文档排序靠前就把后面的模块挤掉（顺序由 _iter_module_docs 的稳定排序保证）。
    out: list[Sample] = []
    for round_i in range(_MAX_PER_DOC):
        for doc_out in per_doc:
            if round_i < len(doc_out):
                out.append(doc_out[round_i])
                if len(out) >= _MODULE_TOTAL_CAP:
                    return out
    return out


def extract_concept_skill_axioms() -> list[Sample]:
    """各模块 skill.md 的「核心范式」条目：`N. **名称** (GT-xxx): 描述`。"""
    out: list[Sample] = []
    for path in sorted((ROOT / ".trae" / "skills").glob("*/skill.md")):
        lines = _lines(_read(path))
        for i, line in enumerate(lines, start=1):
            m = re.match(r"^\d+\.\s+\*\*(.+?)\*\*\s*(?:\([^)]*\))?\s*[:：]\s*(.+)$", line.strip())
            if not m:
                continue
            term, desc = m.group(1).strip(), m.group(2).strip()
            if len(desc) < 20:
                continue
            out.append(Sample("concept_def", PROMPT_T1[1].format(term=term), desc,
                              f"{_rel(path)}#L{i}", term=term, origin="skill_axiom"))
    return out


def extract_faq() -> list[Sample]:
    """faq/*.md：`# Q0xx：问题` + `> **一句话结论**：结论`。"""
    out: list[Sample] = []
    for path in sorted((ROOT / "faq").glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        lines = _lines(_read(path))
        title = ""
        start = 0
        for i, line in enumerate(lines):
            if line.startswith("#"):
                title = line.lstrip("#").strip()
                start = i
                break
        if not title:
            continue
        question = re.sub(r"^Q\d+\s*[:：]\s*", "", title).strip()
        # 「一句话结论」可能跨多行（后续以 `>` 开头的续行）
        concl, buf, in_concl = "", [], False
        for i in range(start, len(lines)):
            line = lines[i]
            if "**一句话结论**" in line:
                in_concl = True
                buf.append(line.split("**一句话结论**", 1)[1].lstrip("：: "))
                continue
            if in_concl:
                if line.strip().startswith(">") and "权威总纲" not in line:
                    buf.append(line.strip().lstrip("> ").strip())
                else:
                    break
        concl = " ".join(x for x in buf if x).strip()
        concl = concl.lstrip("：: ").strip()
        if len(concl) < 40:
            continue
        out.append(Sample("concept_def", f"{question}", concl,
                          f"{_rel(path)}", term=question, origin="faq"))
    return out


# ---------------------------------------------------------------- T2 旧范式（改写 / 判别）

MARKER_ERR = re.compile(r"^\*\*错误\*\*[：:]\s*(.*)$")
MARKER_WHY = re.compile(r"^\*\*为什么错\*\*[：:]\s*(.*)$")
MARKER_OK = re.compile(r"^\*\*正确\*\*[：:]\s*(.*)$")


def _load_antipattern_text() -> str:
    return _read(ROOT / ".trae" / "rules" / "spum-anti-pattern.md")


def parse_antipattern_bad() -> tuple[list[dict], list[dict]]:
    """解析反例库 → (模式块列表, 概念块列表)。每项含 title/错误/为什么错/正确/行号。"""
    path = ROOT / ".trae" / "rules" / "spum-anti-pattern.md"
    lines = _lines(_read(path))
    modes: list[dict] = []
    concepts: list[dict] = []
    cur: dict | None = None
    kind = ""
    field = ""

    def flush():
        nonlocal cur
        if cur and cur.get("错误") and cur.get("正确"):
            (modes if kind == "mode" else concepts).append(cur)
        cur = None

    for i, line in enumerate(lines, start=1):
        s = line.strip()
        m = re.match(r"^### 模式 (\d+)：(.+)$", s)
        if m:
            flush()
            cur = {"n": int(m.group(1)), "title": m.group(2).strip(), "line": i}
            kind, field = "mode", ""
            continue
        m = re.match(r"^### 概念 (\d+)：(.+)$", s)
        if m:
            flush()
            cur = {"n": int(m.group(1)), "title": m.group(2).strip(), "line": i}
            kind, field = "concept", ""
            continue
        if s.startswith("---"):
            continue
        if s.startswith("## ") and not s.startswith("### "):
            flush()
            continue
        if cur is None:
            continue
        m = MARKER_ERR.match(s)
        if m:
            cur.setdefault("错误", [])
            if m.group(1):
                cur["错误"].append(m.group(1))
            field = "错误"
            continue
        m = MARKER_WHY.match(s)
        if m:
            cur["为什么错"] = [m.group(1)] if m.group(1) else []
            field = "为什么错"
            continue
        m = MARKER_OK.match(s)
        if m:
            cur.setdefault("正确", [])
            if m.group(1):
                cur["正确"].append(m.group(1))
            field = "正确"
            continue
        if s.startswith("❌"):
            cur["错误"] = []
            field = "错误"
            rest = s.split("**", 2)[-1].strip()
            if rest and not rest.startswith("**"):
                cur["错误"].append(rest)
            continue
        if s.startswith("✅"):
            cur["正确"] = []
            field = "正确"
            rest = s.split("**", 2)[-1].strip()
            if rest and not rest.startswith("**"):
                cur["正确"].append(rest)
            continue
        if s.startswith("为什么错"):
            cur["为什么错"] = [s.split("：", 1)[-1].strip()] if "：" in s else [s]
            field = "为什么错"
            continue
        if not s:
            continue
        if field in {"错误", "正确", "为什么错"}:
            if field == "为什么错":
                cur.setdefault("为什么错", []).append(s)
            else:
                cur.setdefault(field, []).append(s)
    flush()

    def join(d: dict) -> None:
        for k in ("错误", "为什么错", "正确"):
            if k in d:
                d[k] = "\n".join(d[k]).strip()

    for d in modes + concepts:
        join(d)
    return modes, concepts


def extract_concept_antipattern_ok() -> list[Sample]:
    """反例库「概念 N」块的 «✅ 正确（SPUM 范式）» 段落——正是该概念的规范释义。

    这是全库**唯一**给出"反面 → 正面"对照的释义源，对"同义表述"最宽容：
    它本身就是一段完整的规范性论述，而非一句压缩定义。
    """
    path = ROOT / ".trae" / "rules" / "spum-anti-pattern.md"
    out: list[Sample] = []
    _modes, concepts = parse_antipattern_bad()
    for b in concepts:
        ok = b.get("正确", "")
        if len(ok) < 40:
            continue
        term = re.sub(r"[（(].*$", "", b["title"]).strip()
        if not _valid_term(term):
            continue
        out.append(Sample("concept_def", PROMPT_T1[1].format(term=term), ok,
                          f"{_rel(path)}#L{b['line']}", term=term,
                          origin="anti_pattern_ok"))
    return out


def extract_rewrite() -> list[Sample]:
    """T2a：错误段落 → 「为什么错 + 正确表述」。"""
    path = ROOT / ".trae" / "rules" / "spum-anti-pattern.md"
    out: list[Sample] = []
    modes, concepts = parse_antipattern_bad()
    for kind, blocks in (("模式", modes), ("概念", concepts)):
        for b in blocks:
            bad, why, ok = b.get("错误", ""), b.get("为什么错", ""), b.get("正确", "")
            if not (bad and ok):
                continue
            src = f"{_rel(path)}#L{b['line']}"
            answer = f"**为什么错**：{why}\n\n**正确表述**：{ok}" if why else f"**正确表述**：{ok}"
            prompt_variants = [
                f"下面这段表述违反了 SPUM 范式（{kind}{b['n']}：{b['title']}），请指出问题并改写：\n\n{bad}",
                f"把下面的旧范式表述改写为 SPUM 表述，并说明为什么原表述有问题：\n\n{bad}",
            ]
            for p in prompt_variants:
                out.append(Sample("rewrite", p, answer, src,
                                  ref=f"{kind}{b['n']}:{b['title']}", origin="anti_pattern"))
    return out


def extract_detect() -> list[Sample]:
    """T2b 自检判别：负例=反例库错误段落；正例=语料中的 SPUM 表述（模块核心命题）。"""
    path = ROOT / ".trae" / "rules" / "spum-anti-pattern.md"
    out: list[Sample] = []
    modes, concepts = parse_antipattern_bad()
    for kind, blocks in (("模式", modes), ("概念", concepts)):
        for b in blocks:
            bad, why = b.get("错误", ""), b.get("为什么错", "")
            if not bad:
                continue
            answer = (f"判定：**不符合** SPUM 范式。\n违反：{kind}{b['n']}「{b['title']}」。\n理由：{why}"
                      if why else f"判定：**不符合** SPUM 范式。\n违反：{kind}{b['n']}「{b['title']}」。")
            out.append(Sample("detect",
                              f"判断下面这段表述是否属于 SPUM 范式，并给出理由：\n\n{bad}",
                              answer, f"{_rel(path)}#L{b['line']}",
                              label="neg", origin="anti_pattern"))
    # 正例：模块核心命题（真正符合 SPUM 的文本）。理由取自该模块的「归约路径」原文，
    # 逐条不同——若所有正例共用同一句套话，去重后会坍缩成 1 条，判别任务就退化成"全判负例"。
    positive: list[tuple[str, str, str]] = []
    for path in _iter_module_docs():
        prop = _extract_core_proposition(path)
        if not prop or len(prop) <= 60:
            continue
        positive.append((prop, _rel(path), _extract_reduction_path(path)))
    rng = random.Random(20260913)
    rng.shuffle(positive)
    for prop, src, rpath in positive[:40]:
        chain = rpath.split("：", 1)[-1].strip() if "：" in rpath else ""
        reason = (f"该表述把对象还原为 ⟨P, ε⟩ 的拓扑表达——{chain}。"
                  if chain else
                  "该表述把对象还原为 ⟨P, ε⟩ 上的度数、σ、V⁺/V⁻ 或连通结构，"
                  "未引入先于关系的独立实体，也未依赖连续统假设。")
        out.append(Sample("detect",
                          f"判断下面这段表述是否属于 SPUM 范式，并给出理由：\n\n{prop}",
                          f"判定：**符合** SPUM 范式。\n理由：{reason}",
                          src, label="pos", origin="corpus"))
    return out


# ---------------------------------------------------------------- T3 跨域归约

def _extract_core_proposition(path: Path) -> str:
    """模块 `spum-*.md` 的 `## 核心命题` 段（到下一条 `---` / `## ` 为止）。"""
    lines = _lines(_read(path))
    for i, line in enumerate(lines):
        if line.strip().rstrip("：:") in {"## 核心命题", "## 核心命题（一句话）"} or \
                line.strip().startswith("## 核心命题"):
            buf = []
            for j in range(i + 1, len(lines)):
                s = lines[j].strip()
                if s.startswith("---") or (s.startswith("## ") and not s.startswith("### ")):
                    break
                buf.append(s)
            text = "\n".join(buf).strip()
            return _trim(text, 900)
    return ""


def _extract_reduction_path(path: Path) -> str:
    lines = _lines(_read(path))
    for line in lines[:12]:
        s = line.strip().lstrip("> ").strip()
        if s.startswith("归约路径"):
            return s
    return ""


def extract_reduction() -> list[Sample]:
    out: list[Sample] = []
    for path in _iter_module_docs():
        prop = _extract_core_proposition(path)
        if len(prop) < 60:
            continue
        lines = _lines(_read(path))
        title = lines[0].lstrip("#").strip() if lines else path.stem
        topic = title.split("×")[-1].strip() if "×" in title else title.replace("SPUM", "").strip()
        topic = topic.strip("「」《》 ")
        if not (2 <= len(topic) <= 24):
            continue
        rpath = _extract_reduction_path(path)
        if rpath:
            answer = f"{rpath}\n\n{prop}"
        else:
            answer = prop
        out.append(Sample("reduction",
                          f"用 SPUM 把「{topic}」还原为 ⟨P, ε⟩ 的图论表达。",
                          answer, f"{_rel(path)}", topic=topic, origin="module"))
    return out


# ---------------------------------------------------------------- T4 代码骨架

def extract_code_skeleton() -> list[Sample]:
    src_root = ROOT / "openSPUM" / "src"
    out: list[Sample] = []
    for path in sorted(src_root.rglob("*.py")):
        if any(k in part for part in path.parts for k in ("backup", "archive", "legacy")):
            continue
        text = _read(path)
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        lines = _lines(text)
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            doc = ast.get_docstring(node, clean=True)
            if not doc or len(doc) < 24:
                continue
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            end = node.end_lineno or node.lineno
            if not (4 <= end - start + 1 <= 70):
                continue
            code = _span(lines, start, end)
            prompt = (f"实现函数 `{node.name}`。语义说明：\n\n{_trim(doc, 700)}\n\n"
                      f"只输出该函数的源码（含中文注释），不要输出测试或 __main__。")
            out.append(Sample("code_skeleton", prompt, code,
                              f"{_rel(path)}#L{start}-{end}",
                              func=node.name, origin="openspum_src"))
    return out


# ---------------------------------------------------------------- 构建与切分

def dedupe(samples: list[Sample]) -> tuple[list[Sample], int]:
    """去掉 (prompt, answer) 完全相同的样本；保留同一答案的不同 prompt 变体。

    防泄漏由 `build()` 的分组切分负责（同答案的变体整体落到同一侧），不靠这里丢弃。
    """
    seen: set[str] = set()
    kept: list[Sample] = []
    dropped = 0
    for s in samples:
        key = s.task + "|" + _norm(s.prompt) + "|" + _norm(s.answer)
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        kept.append(s)
    return kept, dropped


def _group_key(s: Sample) -> str:
    """分组切分的「组」标识。

    concept_def 按**术语**分组——同一术语常有多个释义来源（nodes.txt / 词汇表 /
    knowledge 表 / 模块还原表 / 反例库 ✅ 段），它们是同一道题的多个参考答案，
    绝不能一 train 一 val：否则验证题已在训练集里以另一种措辞出现过。
    其余任务按归一化答案分组（同一答案的 prompt 变体整体落到同一侧）。
    """
    if s.task == "concept_def":
        term = (s.extra.get("term") or "").strip()
        if term:
            return "term::" + _norm(term)
    return "ans::" + _norm(s.answer)


def build(val_ratio: float, system: bool) -> dict:
    pools = {
        "concept_def": (extract_concept_nodes() + extract_concept_vocabulary()
                        + extract_concept_knowledge()
                        + extract_concept_module_tables()
                        + extract_concept_skill_axioms()
                        + extract_concept_antipattern_ok()
                        + extract_faq()),
        "rewrite": extract_rewrite(),
        "detect": extract_detect(),
        "reduction": extract_reduction(),
        "code_skeleton": extract_code_skeleton(),
    }
    records: list[dict] = []
    stats: dict[str, dict] = {}
    total_dropped = 0
    for task, samples in pools.items():
        before = len(samples)
        samples, dropped = dedupe(samples)
        total_dropped += dropped
        # 分组切分：整组落到同一侧。
        # 同一答案的多个 prompt 变体是正当增广，但绝不能一 train 一 val（那是泄漏）。
        # 种子用字符串（random 对 str 的 seed 走 sha512，跨进程稳定）——
        # 若用 `hash(task)` 会受 PYTHONHASHSEED 影响，破坏"同一仓库→同一数据集"。
        groups: dict[str, list[Sample]] = {}
        for s in samples:
            groups.setdefault(_group_key(s), []).append(s)
        keys = sorted(groups)
        random.Random(f"spum-split::{task}").shuffle(keys)
        n_val = max(1, int(len(keys) * val_ratio)) if len(keys) >= 6 else 0
        val_keys = set(keys[:n_val])
        n_train = n_val_samples = 0
        for key in keys:
            split = "val" if key in val_keys else "train"
            for s in groups[key]:
                rec = s.to_record(system=system)
                rec["split"] = split
                records.append(rec)
                if split == "val":
                    n_val_samples += 1
                else:
                    n_train += 1
        stats[task] = {"raw": before, "kept": len(samples), "dropped_dup": dropped,
                       "groups": len(keys),
                       "val": n_val_samples, "train": n_train,
                       "answer_len_avg": round(sum(len(x.answer) for x in samples)
                                               / max(1, len(samples))),
                       "origins": sorted({x.extra.get("origin", "?") for x in samples})}
    # 防泄漏自检（两层）：
    #   答案层——同一归一化答案不得跨 train/val；
    #   题面层——同一归一化问题不得跨 train/val（同题多参考时必须同侧）。
    answer_split: dict[str, str] = {}
    prompt_split: dict[str, str] = {}
    leaks = prompt_leaks = 0
    for r in records:
        for bucket, text, name in ((answer_split, r["messages"][-1]["content"], "ans"),
                                   (prompt_split, r["messages"][-2]["content"], "prompt")):
            key = _norm(text)
            prev = bucket.get(key)
            if prev is not None and prev != r["split"]:
                if name == "ans":
                    leaks += 1
                else:
                    prompt_leaks += 1
            bucket[key] = r["split"]
    return {"records": records, "stats": stats, "dropped_dup_total": total_dropped,
            "leaks": leaks, "prompt_leaks": prompt_leaks}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--no-system", action="store_true",
                    help="不写入 system 提示（默认写入）")
    args = ap.parse_args()

    out_dir = (Path(__file__).resolve().parent / args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    built = build(val_ratio=args.val_ratio, system=not args.no_system)
    records = built["records"]
    train = [r for r in records if r["split"] == "train"]
    val = [r for r in records if r["split"] == "val"]

    for name, rows in (("train.jsonl", train), ("val.jsonl", val)):
        with (out_dir / name).open("w", encoding="utf-8") as f:
            for r in sorted(rows, key=lambda x: (x["task"], x["id"])):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {"total": len(records), "train": len(train), "val": len(val),
              "val_ratio": args.val_ratio, "system_prompt": not args.no_system,
              "system_text": SYSTEM_PROMPT if not args.no_system else "",
              "dropped_dup_total": built["dropped_dup_total"],
              "leaks": built["leaks"], "prompt_leaks": built["prompt_leaks"],
              "tasks": built["stats"]}
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"== 数据集构建完成 → {out_dir} ==")
    print(f"{'task':<15}{'kept':>6}{'train':>7}{'val':>5}{'dup':>6}{'avg_len':>9}  origins")
    for task, st in report["tasks"].items():
        print(f"{task:<15}{st['kept']:>6}{st['train']:>7}{st['val']:>5}"
              f"{st['dropped_dup']:>6}{st['answer_len_avg']:>9}  {','.join(st['origins'])}")
    print(f"合计 train={len(train)} val={len(val)} 去重丢弃={built['dropped_dup_total']} "
          f"答案泄漏={built['leaks']} 题面泄漏={built['prompt_leaks']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
