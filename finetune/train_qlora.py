# -*- coding: utf-8 -*-
"""train_qlora.py — 在 SPUM 指令数据上做 QLoRA 微调（阶段二）。

链路：Qwen2.5-Coder-7B-Instruct（4-bit NF4） + LoRA 适配器 → SPUM 指令数据。
只训 assistant 段（prompt 段 label 置 -100），单卡 12GB（RTX 3060）可跑。

用法
----
    python train_qlora.py --smoke                 # 2 步冒烟，验显存与链路
    python train_qlora.py                         # 正式训练（3 epoch）
    python train_qlora.py --epochs 1 --out out/lora-e1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, Trainer, TrainingArguments)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
DEFAULT_BASE = str(Path.home() / "models" / "Qwen2.5-Coder-7B-Instruct")

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


def load_rows(name: str) -> list[dict]:
    path = HERE / "data" / name
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def build_examples(rows: list[dict], tok, max_len: int) -> tuple[list[dict], int]:
    """把 messages 转成 input_ids/labels；prompt 段 label 置 -100（只学回答）。

    返回 (样本, 因过长被丢弃的条数)。
    """
    out: list[dict] = []
    dropped = 0
    for r in rows:
        msgs = r["messages"]
        prompt_text = tok.apply_chat_template(msgs[:-1], tokenize=False,
                                              add_generation_prompt=True)
        full_text = tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=False)
        n_prompt = len(tok(prompt_text, add_special_tokens=False)["input_ids"])
        ids = tok(full_text, add_special_tokens=False)["input_ids"]
        if n_prompt >= max_len - 4:          # prompt 就吃满窗口，没有可学的内容
            dropped += 1
            continue
        ids = ids[:max_len]
        cut = min(n_prompt, len(ids))
        out.append({"input_ids": ids,
                    "labels": [-100] * cut + ids[cut:],
                    "attention_mask": [1] * len(ids)})
    return out, dropped


def make_collator(pad_id: int):
    def collate(batch: list[dict]) -> dict:
        width = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            pad = width - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append([1] * len(b["input_ids"]) + [0] * pad)
        return {"input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long)}
    return collate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--out", default=str(HERE / "out" / "lora-spum"))
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=1, help="单卡 batch size")
    ap.add_argument("--accum", type=int, default=8, help="梯度累积")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--logging-steps", type=int, default=5)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--smoke", action="store_true", help="只跑 2 步，验链路")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(args.base).exists():
        print(f"基座不存在：{args.base}")
        return 2

    print(f"加载 tokenizer ← {args.base}")
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    train_rows, val_rows = load_rows("train.jsonl"), load_rows("val.jsonl")
    train_ex, drop_tr = build_examples(train_rows, tok, args.max_len)
    val_ex, drop_va = build_examples(val_rows, tok, args.max_len)
    lens = sorted(len(x["input_ids"]) for x in train_ex)
    print(f"样本 train={len(train_ex)}(丢 {drop_tr}) val={len(val_ex)}(丢 {drop_va})；"
          f"token 长度 min/中位/max = {lens[0]}/{lens[len(lens)//2]}/{lens[-1]}")

    print("加载基座（4-bit NF4）…")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, device_map={"": 0},
        attn_implementation="sdpa")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM", target_modules=TARGET_MODULES))
    model.print_trainable_parameters()

    # transformers v5 的 TrainingArguments 已无 warmup_ratio，只收 warmup_steps。
    steps_per_epoch = max(1, len(train_ex) // (args.bs * args.accum))
    total_steps = 2 if args.smoke else int(steps_per_epoch * args.epochs)
    warmup_steps = 0 if args.smoke else max(1, int(total_steps * args.warmup_ratio))
    print(f"步数：每 epoch {steps_per_epoch}，共 {total_steps}，warmup {warmup_steps}")

    targs = TrainingArguments(
        output_dir=str(out_dir), per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=1, gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs, max_steps=2 if args.smoke else -1,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=warmup_steps,
        logging_steps=args.logging_steps, eval_strategy="epoch",
        save_strategy="epoch", save_total_limit=2, bf16=True,
        optim="paged_adamw_8bit", gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0, seed=20260913, data_seed=20260913,
        report_to=[], remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=targs,
                      train_dataset=Dataset.from_list(train_ex),
                      eval_dataset=Dataset.from_list(val_ex),
                      data_collator=make_collator(tok.pad_token_id),
                      processing_class=tok)

    t0 = time.time()
    trainer.train()
    print(f"训练用时 {round(time.time()-t0,1)}s")

    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    metrics = {"train_samples": len(train_ex), "val_samples": len(val_ex),
               "dropped": {"train": drop_tr, "val": drop_va},
               "epochs": args.epochs, "lr": args.lr, "bs": args.bs,
               "accum": args.accum, "lora_r": args.lora_r,
               "lora_alpha": args.lora_alpha, "max_len": args.max_len,
               "base": args.base, "elapsed_sec": round(time.time() - t0, 1),
               "log_history": trainer.state.log_history}
    (out_dir / "train_report.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"适配器 → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
