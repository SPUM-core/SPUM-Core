# -*- coding: utf-8 -*-
"""merge_lora.py — 把 QLoRA 适配器合并回基座，产出完整权重供 ollama 导入。

为什么在 CPU 上合并
------------------
基座 bf16 约 15GB，RTX 3060 只有 12GB。用 4-bit 加载虽省显存，但 peft 的
`merge_and_unload()` 会把 4-bit 层**反量化回 bf16** 再叠加 adapter —— 合并结果
仍是约 15GB，在 GPU 上必然 OOM。本机 32GB 内存，故用 `low_cpu_mem_usage`
分片加载 + CPU 合并，峰值约 16GB。

用法
----
    python merge_lora.py --base <基座目录> --adapter out/lora-spum --out out/merged-spum
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_base(path: str):
    """transformers v5 把 `torch_dtype` 改名成 `dtype`，这里两条都兼容。"""
    common = dict(device_map="cpu", low_cpu_mem_usage=True)
    try:
        return AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, **common)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, **common)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="基座权重目录")
    ap.add_argument("--adapter", required=True, help="LoRA 适配器目录")
    ap.add_argument("--out", required=True, help="合并输出目录")
    ap.add_argument("--max-shard", default="4GB")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)          # 合并结果必须整体重来，不能与旧分片混放
    out.mkdir(parents=True)

    t0 = time.time()
    print(f"[1/4] 加载基座（CPU, bf16）…", flush=True)
    tok = AutoTokenizer.from_pretrained(args.base)
    model = load_base(args.base)
    n_base = sum(p.numel() for p in model.parameters())
    print(f"      基座参数量 {n_base/1e9:.3f}B  用时 {time.time()-t0:.1f}s", flush=True)

    print(f"[2/4] 挂载适配器 {args.adapter} …", flush=True)
    model = PeftModel.from_pretrained(model, args.adapter)

    print(f"[3/4] 合并并卸载适配器 …", flush=True)
    model = model.merge_and_unload()
    n_merged = sum(p.numel() for p in model.parameters())
    assert n_merged == n_base, f"合并后参数量变了：{n_base} → {n_merged}"
    assert not any("lora_" in n for n, _ in model.named_parameters()), "仍有 lora_ 参数残留"

    print(f"[4/4] 写出到 {out} …", flush=True)
    model.save_pretrained(out, safe_serialization=True, max_shard_size=args.max_shard)
    tok.save_pretrained(out)

    shards = sorted(p.name for p in out.glob("*.safetensors"))
    size_gb = sum(p.stat().st_size for p in out.glob("*.safetensors")) / 1024**3
    report = {"base": args.base, "adapter": args.adapter, "out": str(out),
              "params": n_merged, "shards": shards, "weights_gb": round(size_gb, 2),
              "elapsed_sec": round(time.time() - t0, 1)}
    (out / "merge_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n合并完成：{len(shards)} 个分片，{size_gb:.2f}GB，用时 {time.time()-t0:.1f}s")
    print(f"报告 → {out/'merge_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
