# -*- coding: utf-8 -*-
"""eval_baseline.py — SPUM 微调基线评测 harness（阶段一：基线）。

用途：把 `data/val.jsonl` 逐条送进本地 ollama 模型，按任务分项打分，
产出 `eval/<tag>.json` + 控制台摘要。先量出现役模型（未微调）的短板，
再决定是否值得投入 QLoRA 训练。

指标（全部可复算，无人工主观项）
--------------------------------
字面类（词面重合，快、无依赖）：
- `char_f1`   参考答 vs 模型答的字符 2-gram F1（去 markdown 标记/空白）
- `ref_cov`   参考答的字符 2-gram 被模型答覆盖的比例（纯召回，无长度惩罚）
- `len_ratio` 模型答 / 参考答 的字符 2-gram 总数之比（判「话多」还是「话少」）
- `term_hit`  参考答案里的 SPUM 术语在模型答中的命中率
- `banned`    模型答命中的旧范式词汇数（启发式子集，见 BANNED_TERMS）

语义类（句向量余弦，用于释义任务）：
- `emb_sim`      模型答 vs 该条参考答 的语义相似度
- `emb_sim_max`  模型答 vs **该术语全部释义参考** 的最大相似度（多参考）

任务专属：`detect` 抽判词与 `label` 比对算准确率；`code_skeleton` 算 `ast.parse`
通过率与函数名是否出现。

为什么要语义读数
----------------
`char_f1`/`ref_cov` 是**词面**读数。释义任务上它必然失真：模型答「度数为 1 的
节点是悬挂端」而参考写「度为 1 是悬挂边」，意思一致、词面几乎不重合 ⇒ 读出接近 0。
实测 `concept_def` 长期停在 `ref_cov≈0.09` 但人工看答得对，就是这一伪影。
`emb_sim` 用句向量（默认 `BAAI/bge-small-zh-v1.5`，CPU，无采样、可复算）把
「换一种说法说对了」与「说错了」分开；`emb_sim_max` 再消除「参考只有一种表述」
造成的惩罚。字面类读数一并保留，便于对照。

用法
----
    python eval_baseline.py --tag baseline-zeroshot --shots 0
    python eval_baseline.py --tag baseline-4shot --shots 4
    python eval_baseline.py --limit 3 --tag smoke
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
EVAL = HERE / "eval"
OLLAMA_CHAT = "http://localhost:11434/api/chat"

# 旧范式词汇的启发式子集：取自 `.trae/rules/spum-vocabulary.md` §一，
# 只保留多字、无歧义的词——「力」「场」「背景」「时空」这类单字/泛词在
# SPUM 合法语境中也会出现（如"背景混沌网络""连续统陷阱"），收进来只会误报。
BANNED_TERMS = [
    "时空弯曲", "弯曲时空", "时空流形", "流形", "引力子", "万有引力",
    "概率云", "概率波", "波函数", "波粒二象性", "叠加态", "观测坍缩",
    "点粒子", "大爆炸", "奇点", "宇宙常数", "热寂", "均匀流逝", "无限小",
]

# SPUM 应出现的术语（用于 term_hit）
SPUM_TERMS = [
    "⟨P", "ε", "σ", "deg", "V⁺", "V⁻", "节点", "边", "子图",
    "拓扑", "帧", "晶子", "度数", "五形", "归约", "空间粒子",
]


# ---------------------------------------------------------------- 基础工具

def _norm(text: str) -> str:
    """去 markdown 标记与空白，用于字符 n-gram 比对。"""
    t = re.sub(r"```[a-zA-Z]*|```", "", text)
    t = re.sub(r"[`*#>|\-\s]", "", t)
    return t


def _ngrams(text: str, n: int = 2) -> Counter:
    t = _norm(text)
    return Counter(t[i:i + n] for i in range(max(0, len(t) - n + 1)))


def char_f1(ref: str, hyp: str, n: int = 2) -> float:
    a, b = _ngrams(ref, n), _ngrams(hyp, n)
    if not a or not b:
        return 0.0
    inter = sum((a & b).values())
    if inter == 0:
        return 0.0
    p = inter / sum(b.values())
    r = inter / sum(a.values())
    return 2 * p * r / (p + r)


def term_hit(ref: str, hyp: str) -> float:
    want = [t for t in SPUM_TERMS if t in ref]
    if not want:
        return float("nan")
    return sum(1 for t in want if t in hyp) / len(want)


def banned_count(hyp: str) -> int:
    return sum(hyp.count(t) for t in BANNED_TERMS)


def _strip_fence(text: str) -> str:
    m = re.search(r"```[a-zA-Z]*\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def parse_detect(text: str) -> str | None:
    """抽判词。注意「不符合」包含「符合」，交替式按左优先匹配即可正确区分。"""
    m = re.search(r"(不符合|符合)", text.replace("**", ""))
    if not m:
        return None
    return "neg" if m.group(1) == "不符合" else "pos"


# ---------------------------------------------------------------- ollama

def ollama_chat(model: str, messages: list[dict], num_predict: int,
                temperature: float, timeout: int) -> str:
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict, "seed": 0},
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_CHAT, data=payload,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return (body.get("message") or {}).get("content", "")


class OllamaGenerator:
    def __init__(self, model: str, num_predict: int, temperature: float, timeout: int):
        self.model, self.num_predict = model, num_predict
        self.temperature, self.timeout = temperature, timeout

    def chat(self, messages: list[dict]) -> str:
        return ollama_chat(self.model, messages, self.num_predict,
                           self.temperature, self.timeout)


class HFGenerator:
    """本地 transformers 生成后端——训练与评测走同一份权重格式，可直接挂 LoRA 适配器。

    评测一律用 4-bit：7B 的 bf16 权重约 15GB，放不进 12GB（RTX 3060）。
    只要**对照的两次运行（挂/不挂适配器）量化方式一致**，A/B 结论就成立。
    """

    def __init__(self, model_path: str, adapter: str | None, num_predict: int):
        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  BitsAndBytesConfig)
        self.torch = torch
        self.num_predict = num_predict
        self.tok = AutoTokenizer.from_pretrained(model_path)
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb, device_map={"": 0},
            attn_implementation="sdpa")
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def chat(self, messages: list[dict]) -> str:
        text = self.tok.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt").to("cuda")
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.num_predict,
                                      do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True)


# ---------------------------------------------------------------- few-shot

def _trim(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "……"


def load_fewshot(train_rows: list[dict], task: str, shots: int,
                 max_chars: int = 600) -> str:
    if shots <= 0:
        return ""
    rows = sorted((r for r in train_rows if r["task"] == task),
                  key=lambda r: r["id"])[:shots]
    blocks = []
    for i, r in enumerate(rows, 1):
        u = r["messages"][-2]["content"]
        a = _trim(r["messages"][-1]["content"], max_chars)
        blocks.append(f"示例{i}\n用户：{u}\n助手：{a}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------- 多参考

def build_ref_index(rows: list[dict]) -> dict[str, list[str]]:
    """术语 → 该术语在数据集内的**全部释义参考**（多参考评分用）。

    索引建在 train + val 之上：这是标准的多参考评测做法（参考集不是答案标签，
    而是"该术语的已知表述集合"），模型贴近其中任一条即算说对。
    """
    idx: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        if r["task"] != "concept_def":
            continue
        term = (r.get("term") or "").strip()
        if term:
            idx[term].append(r["messages"][-1]["content"])
    return {k: sorted(set(v)) for k, v in idx.items()}


# ---------------------------------------------------------------- 语义读数

class Embedder:
    """句向量（默认 `BAAI/bge-small-zh-v1.5`，CPU）——释义任务的语义读数来源。

    BGE 系列取 [CLS] 位并做 L2 归一化即得句向量，无需 sentence-transformers。
    只在前向推理，无采样，同一输入必得同一读数。
    """

    def __init__(self, name: str, batch: int = 16):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.batch = batch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).eval()

    def encode(self, texts: list[str]):
        import torch
        vecs = []
        for i in range(0, len(texts), self.batch):
            enc = self.tok(texts[i:i + self.batch], padding=True, truncation=True,
                           max_length=512, return_tensors="pt")
            with self.torch.inference_mode():
                h = self.model(**enc).last_hidden_state[:, 0]
            vecs.append(torch.nn.functional.normalize(h, dim=-1))
        return torch.cat(vecs) if vecs else torch.zeros(0, 0)


def add_semantic(records: list[dict], val_rows: list[dict],
                 refs: dict[str, list[str]], embedder: Embedder) -> int:
    """给 concept_def 记录补 `emb_sim` / `emb_sim_max`（就地写入 metrics）。

    `records` 与 `val_rows` 逐位对齐（同一次循环生成，含出错项）。
    返回补充成功的条数；出错项与空答不参与。
    """
    pairs = [(rec, row) for rec, row in zip(records, val_rows)
             if rec["task"] == "concept_def" and not rec.get("error") and rec["answer"].strip()]
    if not pairs:
        return 0
    own = [row["messages"][-1]["content"] for _rec, row in pairs]
    term_refs = [refs.get((row.get("term") or "").strip(), []) for _rec, row in pairs]
    uniq = sorted({t for ts in (own + term_refs) for t in ts})
    ans_vec = embedder.encode([rec["answer"] for rec, _ in pairs])
    ref_vec = embedder.encode(uniq)
    pos = {t: i for i, t in enumerate(uniq)}
    for k, (rec, _row) in enumerate(pairs):
        a = ans_vec[k]
        sims = [float((a * ref_vec[pos[t]]).sum()) for t in term_refs[k]]
        rec["metrics"]["n_refs"] = len(sims)
        rec["metrics"]["emb_sim"] = round(float((a * ref_vec[pos[own[k]]]).sum()), 4)
        rec["metrics"]["emb_sim_max"] = round(max(sims), 4) if sims else None
    return len(pairs)


# ---------------------------------------------------------------- 打分

def score(rec: dict, answer: str) -> dict:
    task = rec["task"]
    ref = rec["messages"][-1]["content"]
    a, b = _ngrams(ref), _ngrams(answer)
    ta, tb = sum(a.values()), sum(b.values())
    inter = sum((a & b).values()) if (a and b) else 0
    # char_f1 对"话多"有惩罚——答得越长 P 越低，F1 越低。所以同时记录
    # ref_cov（参考内容被覆盖了多少，不含长度惩罚）与 len_ratio，避免把
    # "啰嗦但说对了"误判成"不会"，也避免把"短而空洞"误判成"会"。
    out = {"char_f1": round(char_f1(ref, answer), 4),
           "ref_cov": round(inter / ta, 4) if ta else None,
           "len_ratio": round(tb / ta, 2) if ta else None,
           "term_hit": term_hit(ref, answer),
           "banned": banned_count(answer)}
    if out["term_hit"] != out["term_hit"]:          # nan → None 便于 JSON
        out["term_hit"] = None
    if task == "detect":
        pred = parse_detect(answer)
        out["pred"] = pred
        out["correct"] = int(pred is not None and pred == rec.get("label"))
    elif task == "code_skeleton":
        code = _strip_fence(answer)
        try:
            ast.parse(code)
            out["ast_ok"] = 1
        except SyntaxError:
            out["ast_ok"] = 0
        out["def_ok"] = int(f"def {rec.get('func','')}(" in code)
    return out


def _mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _fmt(v) -> str:
    return "-" if v is None else str(v)


def summarize(records: list[dict]) -> dict:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r.get("error"):
            continue
        by_task[r["task"]].append(r["metrics"])
    summary: dict[str, dict] = {}
    for task, ms in sorted(by_task.items()):
        row = {"n": len(ms),
               "char_f1": _mean([m["char_f1"] for m in ms]),
               "ref_cov": _mean([m["ref_cov"] for m in ms]),
               "term_hit": _mean([m["term_hit"] for m in ms]),
               "banned_per_ans": _mean([float(m["banned"]) for m in ms]),
               "len_ratio": _mean([m["len_ratio"] for m in ms])}
        if task == "concept_def":
            row["emb_sim"] = _mean([m.get("emb_sim") for m in ms])
            row["emb_sim_max"] = _mean([m.get("emb_sim_max") for m in ms])
            row["n_refs_mean"] = _mean([float(m.get("n_refs") or 0) for m in ms])
        if task == "detect":
            row["accuracy"] = _mean([float(m["correct"]) for m in ms])
        if task == "code_skeleton":
            row["ast_ok_rate"] = _mean([float(m["ast_ok"]) for m in ms])
            row["def_ok_rate"] = _mean([float(m["def_ok"]) for m in ms])
        summary[task] = row
    allm = [m for ms in by_task.values() for m in ms]
    cd = [m for m in by_task.get("concept_def", [])]
    summary["_overall"] = {
        "n": len(allm),
        "char_f1": _mean([m["char_f1"] for m in allm]),
        "ref_cov": _mean([m["ref_cov"] for m in allm]),
        "emb_sim": _mean([m.get("emb_sim") for m in cd]) if cd else None,
        "emb_sim_max": _mean([m.get("emb_sim_max") for m in cd]) if cd else None,
        "term_hit": _mean([m["term_hit"] for m in allm]),
        "banned_per_ans": _mean([float(m["banned"]) for m in allm]),
        "len_ratio": _mean([m["len_ratio"] for m in allm]),
        "errors": sum(1 for r in records if r.get("error")),
    }
    return summary


# ---------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["ollama", "hf"], default="ollama")
    ap.add_argument("--model", default="qwen2.5-coder:14b",
                    help="ollama: 模型名；hf: 基座权重目录")
    ap.add_argument("--adapter", default="", help="hf 后端的 LoRA 适配器目录（可空）")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--shots", type=int, default=0, help="同任务 few-shot 示例数")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟用）")
    ap.add_argument("--num-predict", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--embed-model", default="BAAI/bge-small-zh-v1.5",
                    help="语义读数用句向量模型；置空则跳过语义指标")
    args = ap.parse_args()

    val = [json.loads(x) for x in (DATA / "val.jsonl").read_text(encoding="utf-8").splitlines()]
    train = [json.loads(x) for x in (DATA / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    if args.limit:
        val = val[:args.limit]
    refs = build_ref_index(train + val)
    multi = sum(len(v) for v in refs.values()) / max(1, len(refs))
    print(f"多参考索引：{len(refs)} 个术语 / {sum(len(v) for v in refs.values())} 条参考"
          f"（平均每术语 {multi:.2f} 条）", flush=True)

    demos = {t: load_fewshot(train, t, args.shots)
             for t in {r["task"] for r in val}}

    if args.backend == "hf":
        gen = HFGenerator(args.model, args.adapter or None, args.num_predict)
    else:
        gen = OllamaGenerator(args.model, args.num_predict, args.temperature, args.timeout)

    EVAL.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    t0 = time.time()
    for i, rec in enumerate(val, 1):
        user = rec["messages"][-2]["content"]
        if demos.get(rec["task"]):
            user = f"{demos[rec['task']]}\n\n现在请回答：\n{user}"
        messages = [rec["messages"][0], {"role": "user", "content": user}]
        answer, err, latency = "", None, 0.0
        for attempt in range(args.retries + 1):
            ts = time.time()
            try:
                answer = gen.chat(messages)
                latency = time.time() - ts
                err = None
                break
            except Exception as e:                  # HF 后端异常类型不可穷举，如实记账
                latency = time.time() - ts
                err = f"{type(e).__name__}: {e}"
        entry = {"id": rec["id"], "task": rec["task"], "split": rec["split"],
                 "source": rec["source"], "label": rec.get("label"),
                 "func": rec.get("func"), "latency": round(latency, 1),
                 "error": err, "answer": answer,
                 "metrics": {} if err else score(rec, answer)}
        records.append(entry)
        flag = "ERR" if err else f"f1={entry['metrics'].get('char_f1')}"
        print(f"[{i:>3}/{len(val)}] {rec['task']:<14} {flag:<14} {latency:>6.1f}s  {rec['id']}",
              flush=True)

    # 语义读数（concept_def 专用）。取不到句向量模型时如实跳过——不静默降级成 0，
    # 也不拿字面读数冒充语义结论。
    semantic_n = 0
    if args.embed_model:
        try:
            semantic_n = add_semantic(records, val, refs, Embedder(args.embed_model))
            print(f"语义读数：{semantic_n} 条 concept_def（{args.embed_model}）", flush=True)
        except Exception as e:
            print(f"[warn] 语义读数跳过（{type(e).__name__}: {e}）", flush=True)

    summary = summarize(records)
    report = {"tag": args.tag, "backend": args.backend, "model": args.model,
              "adapter": args.adapter, "shots": args.shots,
              "temperature": args.temperature, "num_predict": args.num_predict,
              "embed_model": args.embed_model, "semantic": semantic_n,
              "samples": len(val), "elapsed_sec": round(time.time() - t0, 1),
              "summary": summary, "records": records}
    out = EVAL / f"{args.tag}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n== {args.tag}（backend={args.backend} model={args.model} "
          f"adapter={args.adapter or '-'} shots={args.shots} "
          f"n={len(val)} 用时 {report['elapsed_sec']}s）==")
    print(f"{'task':<15}{'n':>4}{'char_f1':>9}{'ref_cov':>9}{'emb_sim':>9}{'emb_max':>9}"
          f"{'term_hit':>10}{'banned':>8}{'len×':>7}  备注")
    for task, row in summary.items():
        extra = ""
        if "accuracy" in row:
            extra = f"acc={row['accuracy']}"
        if "ast_ok_rate" in row:
            extra = f"ast={row['ast_ok_rate']} def={row['def_ok_rate']}"
        if "n_refs_mean" in row:
            extra = f"n_refs={row['n_refs_mean']}"
        if task == "_overall" and row.get("emb_sim") is not None:
            extra = "emb_sim/emb_max 列仅覆盖 concept_def"
        print(f"{task:<15}{row['n']:>4}{_fmt(row['char_f1']):>9}{_fmt(row['ref_cov']):>9}"
              f"{_fmt(row.get('emb_sim')):>9}{_fmt(row.get('emb_sim_max')):>9}"
              f"{_fmt(row['term_hit']):>10}{_fmt(row['banned_per_ans']):>8}"
              f"{_fmt(row['len_ratio']):>7}  {extra}")
    print(f"报告 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
