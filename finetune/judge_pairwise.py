# -*- coding: utf-8 -*-
"""judge_pairwise.py — 用 LLM 裁判对 concept_def 做**成对偏好**判决。

为什么要另起一个判决器
----------------------
`char_f1` / `ref_cov` 在 concept_def 上被长度伪影主导（基座啰嗦 168 字 vs
微调精练 39 字），句向量余弦（`BAAI/bge-small-zh-v1.5`）实测逐术语
16 胜 / 2 平 / 14 负 ≈ 掷硬币——三个自动读数在该任务上均已被证伪。
释义的「意思对不对」需要一个能读参考、能比较的判决者，这就是本脚本。

三条防空转约束（沿用本项目「读数可证伪性纪律」）
------------------------------------------------
1. **正控**：以「该条参考原文」为 gold 对「另一术语的参考」为离题，
   裁判必须压倒性选 gold。正控不过 ⇒ 裁判本身不可用，本脚本**不下结论**，
   只如实报告。
2. **双向**：同一对候选正反各问一次。位置偏好由「两次是否指向同一侧」度量；
   最终胜负面取双向平均，消除 A/B 位置偏置。
3. **确定性**：`temperature=0`、`seed=0`，A/B 顺序由 `random.Random("judge::" + id)`
   固定——受 `PYTHONHASHSEED` 影响的 `hash()` 一律不用。同一输入必得同一判决。

用法
----
    python judge_pairwise.py --judge qwen3:14b --a hf-base-v2 --b hf-lora-v2 \
        --out eval/judge-v2.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
EVAL = HERE / "eval"
OLLAMA_CHAT = "http://localhost:11434/api/chat"

SYSTEM = (
    "你是严格的释义评审。只依据给出的参考释义判断，不依据你自己的领域知识。"
    "两个候选释义在措辞上不同不算错——只要**核心命题**与参考一致就算说对。"
    "若两者对核心命题的覆盖程度相当，判平局。\n"
    "只输出 JSON，不要任何解释性前后缀。\n"
    "/no_think"
)

USER_TMPL = """术语：{term}

参考释义（权威，以此为准）：
{refs}

候选 A：{a}

候选 B：{b}

请判断哪一个候选更接近参考释义的核心命题。
只输出 JSON：{{"winner": "A" 或 "B" 或 "tie", "reason": "不超过 25 字"}}"""


def ollama_json(model: str, messages: list[dict], num_predict: int,
                timeout: int) -> str:
    """带 `format=json` 的 ollama 调用——约束解码，省去大部分解析失败。"""
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_predict": num_predict, "seed": 0},
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA_CHAT, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return (body.get("message") or {}).get("content", "")


def parse_winner(text: str) -> str | None:
    """抽胜者。先按 JSON 解，失败则退化到正则——14B 偶尔会漏引号。"""
    try:
        obj = json.loads(text)
        w = str(obj.get("winner", "")).strip().lower()
        if w in {"a", "b", "tie"}:
            return w
    except Exception:
        pass
    m = re.search(r'"?winner"?\s*[:：]\s*"?\s*(tie|TIE|A|B|a|b)', text)
    if m:
        v = m.group(1).lower()
        return "tie" if v == "tie" else v
    return None


def load_reports(a_tag: str, b_tag: str) -> tuple[dict, dict]:
    a = json.loads((EVAL / f"{a_tag}.json").read_text(encoding="utf-8"))
    b = json.loads((EVAL / f"{b_tag}.json").read_text(encoding="utf-8"))
    return a, b


def refs_for(term: str, own_ref: str, index: dict[str, list[str]]) -> str:
    """参考文本块：本条参考在前，同术语的其他表述附后（对两侧同等放宽）。"""
    lines = [f"- {own_ref}"]
    for r in index.get(term, []):
        if r != own_ref:
            lines.append(f"- {r}")
    return "\n".join(lines)


def ask_judge(judge: str, term: str, refs: str, ca: str, cb: str,
              num_predict: int, timeout: int) -> tuple[str | None, str]:
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER_TMPL.format(
                term=term, refs=refs, a=ca, b=cb)}]
    raw = ollama_json(judge, msgs, num_predict, timeout)
    m = re.search(r'"reason"\s*[:：]\s*"([^"]*)"', raw)
    return parse_winner(raw), (m.group(1) if m else raw[:60])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="qwen3:14b", help="裁判模型（ollama 名）")
    ap.add_argument("--a", default="hf-base-v2", help="A 侧 eval 报告 tag（基座）")
    ap.add_argument("--b", default="hf-lora-v2", help="B 侧 eval 报告 tag（微调）")
    ap.add_argument("--out", default="judge-v2.json")
    ap.add_argument("--controls", type=int, default=20, help="正控条数")
    ap.add_argument("--limit", type=int, default=0, help="只判前 N 对（冒烟用）")
    ap.add_argument("--num-predict", type=int, default=200)
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    val = [json.loads(x) for x in
           (DATA / "val.jsonl").read_text(encoding="utf-8").splitlines()]
    train = [json.loads(x) for x in
             (DATA / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    index: dict[str, list[str]] = {}
    for r in train + val:
        if r["task"] == "concept_def":
            t = (r.get("term") or "").strip()
            if t:
                index.setdefault(t, [])
                if r["messages"][-1]["content"] not in index[t]:
                    index[t].append(r["messages"][-1]["content"])

    vmap = {r["id"]: r for r in val}
    ra, rb = load_reports(args.a, args.b)
    pairs = []
    for x, y in zip(ra["records"], rb["records"]):
        if x["task"] != "concept_def" or x.get("error") or y.get("error"):
            continue
        if not x["answer"].strip() or not y["answer"].strip():
            continue
        pairs.append((x["id"], vmap[x["id"]], x["answer"], y["answer"]))
    if args.limit:
        pairs = pairs[:args.limit]

    print(f"=== 正控：gold（参考原文） vs 离题（另一术语的参考） ===", flush=True)
    ctrl_ok = ctrl_n = 0
    for cid, rec, _xa, _yb in pairs[:args.controls]:
        term = (rec.get("term") or "").strip()
        gold = rec["messages"][-1]["content"]
        others = [r for r in val
                  if r["task"] == "concept_def" and r["id"] != cid
                  and (r.get("term") or "").strip() != term]
        pick = random.Random(f"ctrl::{cid}").choice(others)
        off = pick["messages"][-1]["content"]
        refs = refs_for(term, gold, index)
        swapped = random.Random(f"ctrlord::{cid}").random() < 0.5
        ca, cb = (off, gold) if swapped else (gold, off)
        w, _ = ask_judge(args.judge, term, refs, ca, cb,
                         args.num_predict, args.timeout)
        gold_side = "b" if swapped else "a"
        ctrl_n += 1
        if w == gold_side:
            ctrl_ok += 1
        print(f"  [{ctrl_n:>2}/{args.controls}] winner={w} 期望={gold_side} "
              f"{'OK' if w == gold_side else 'MISS'}  {term}", flush=True)
    acc = ctrl_ok / ctrl_n if ctrl_n else 0.0
    print(f"正控命中 {ctrl_ok}/{ctrl_n} = {acc:.2%}", flush=True)
    if acc < 0.9:
        print(f"[warn] 正控未达 90%——裁判不可用，以下判决只作参考、不作结论。",
              flush=True)

    print(f"\n=== 判决：{args.a} vs {args.b}（双向各一次） ===", flush=True)
    records = []
    tally = Counter()
    t0 = time.time()
    for i, (cid, rec, abase, alora) in enumerate(pairs, 1):
        term = (rec.get("term") or "").strip()
        refs = refs_for(term, rec["messages"][-1]["content"], index)
        # 正向：A=base B=lora；反向：A=lora B=base（消除位置偏置）
        fwd, _ = ask_judge(args.judge, term, refs, abase, alora,
                           args.num_predict, args.timeout)
        rev, _ = ask_judge(args.judge, term, refs, alora, abase,
                           args.num_predict, args.timeout)
        # 统一到 base/lora 坐标
        v1 = {"a": "base", "b": "lora", "tie": "tie"}.get(fwd)
        v2 = {"a": "lora", "b": "base", "tie": "tie"}.get(rev)
        if v1 == v2 and v1 is not None:
            verdict = v1
            consistent = True
        elif v1 is None or v2 is None:
            verdict = "unparsed"
            consistent = False
        else:
            verdict = "split"          # 两次判定矛盾 = 位置偏置或边界难例
            consistent = False
        tally[verdict] += 1
        records.append({"id": cid, "term": term, "fwd": fwd, "rev": rev,
                        "verdict": verdict, "consistent": consistent,
                        "base": abase, "lora": alora})
        print(f"[{i:>3}/{len(pairs)}] {term:<14} fwd={fwd} rev={rev} "
              f"=> {verdict}", flush=True)

    n = len(pairs)
    decided = tally["base"] + tally["lora"]
    # 位置偏置：两次都选同一个**位置**（B 或 A）说明裁判按位置作答而非按内容。
    # 若位置平均胜率恰为 50%，则原始胜率被位置偏置吃掉，不能当结论——必须同时报出。
    pos_b = sum(1 for r in records if r["fwd"] == "b" and r["rev"] == "b")
    pos_a = sum(1 for r in records if r["fwd"] == "a" and r["rev"] == "a")
    lora_calls = (sum(1 for r in records if r["fwd"] == "b")
                  + sum(1 for r in records if r["rev"] == "a"))
    calls = 2 * n
    summary = {
        "judge": args.judge, "a": args.a, "b": args.b, "n": n,
        "control": {"n": ctrl_n, "gold_win": ctrl_ok, "acc": round(acc, 4)},
        "control_pass": acc >= 0.9,
        "verdicts": dict(tally),
        "decided": decided,
        "lora_win_rate_decided": round(tally["lora"] / decided, 4) if decided else None,
        "position_consistent": round(
            sum(1 for r in records if r["consistent"]) / n, 4) if n else None,
        "position_bias": {"always_b": pos_b, "always_a": pos_a},
        "lora_pick_rate_position_averaged": round(lora_calls / calls, 4) if calls else None,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out = EVAL / args.out
    out.write_text(json.dumps({"summary": summary, "records": records},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n== 判决汇总（judge={args.judge} n={n}）==")
    print(f"正控 gold 命中：{ctrl_ok}/{ctrl_n} = {acc:.2%}"
          f"  {'通过' if summary['control_pass'] else '**未通过，判决不作结论**'}")
    print(f"逐条：{dict(tally)}")
    print(f"lora 胜率（仅计已决出，{decided} 条）：{summary['lora_win_rate_decided']}")
    print(f"双向一致率：{summary['position_consistent']}"
          f"   位置偏置 alwaysB={pos_b} alwaysA={pos_a}")
    print(f"位置平均 lora 命中率（全部 {calls} 次调用）："
          f"{summary['lora_pick_rate_position_averaged']}"
          f"   ← 若≈0.5，说明原始胜率被位置偏置吃掉，不可作结论")
    print(f"报告 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
