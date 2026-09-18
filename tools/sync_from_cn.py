# -*- coding: utf-8 -*-
"""sync_from_cn.py — Sync changes from the Chinese Gitee repo into this English GitHub repo.

Workflow:
  1. Pull latest in the CN repo (../spum-core-cn).
  2. Diff changed files since the last sync stamp.
  3. For each changed file:
     - Code/data files (.py, .json, .csv, .png, ...) -> copy as-is.
     - .md files listed in file_map.json -> translate via local ollama model.
     - .md files NOT in file_map -> copy as-is (untranslated) and log.
  4. Commit and push in this (EN) repo.

Usage:
    python sync_from_cn.py [--dry-run] [--no-translate] [--no-push]

Requires:
  - ollama running on http://localhost:11434 with model qwen2.5-coder:14b
  - ../spum-core-cn/ (a clone of the Gitee repo)
  - tools/file_map.json
  - GLOSSARY.md at repo root
  - translate_spec_docs.txt in %TEMP%/spum_translate/
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import time
import urllib.request

# --- paths ---
HERE = pathlib.Path(__file__).resolve().parent
EN_ROOT = HERE.parent                       # this repo root
CN_ROOT = EN_ROOT.parent / "spum-core-cn"  # sibling clone of Gitee
STAMP = HERE / ".sync_stamp"                # last synced CN commit SHA
FILE_MAP = HERE / "file_map.json"
GLOSSARY = EN_ROOT / "GLOSSARY.md"
SPEC_PATH = pathlib.Path(r"C:\Users\macotai\AppData\Local\Temp\spum_translate\translate_spec_docs.txt")

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "qwen2.5-coder:14b"

# file extensions that are copied, not translated
COPY_EXT = {".py", ".json", ".csv", ".txt", ".png", ".jpg", ".js", ".ts",
            ".cu", ".cuh", ".gguf", ".pt", ".safetensors", ".yml", ".yaml",
            ".html", ".css", ".sh", ".bat", ".ps1", ".gitignore", ".keep",
            ".toml", ".cfg", ".ini", ".mod", ".sum", ".md5"}

# --- translation helpers (adapted from translate_chunks2.py) ---

EXT_PATTERN = (".md", ".py", ".txt", ".json", ".png", ".csv", ".jinja",
               ".yml", ".exe", ".js", ".ts", ".cu", ".cuh", ".html", ".css")
BACKTICK = re.compile(r"`([^`\n]+)`")
LINKTGT = re.compile(r"(\]\()([^)\n]+)(\))")


def looks_like_path(s: str) -> bool:
    s = s.strip()
    if s.startswith(("http://", "https://", "file://", "mailto:")):
        return False
    if "/" in s or "\\" in s:
        return True
    return s.endswith(EXT_PATTERN)


def mask_paths(chunk: str):
    store: dict[str, str] = {}

    def key_for(orig: str) -> str:
        for k, v in store.items():
            if v == orig:
                return k
        k = "%%P%02d%%" % (len(store) + 1)
        store[k] = orig
        return k

    def repl_bt(m):
        inner = m.group(1)
        if looks_like_path(inner):
            return "`" + key_for(inner.strip()) + "`"
        return m.group(0)

    def repl_link(m):
        tgt = m.group(2).strip()
        if looks_like_path(tgt):
            return m.group(1) + key_for(tgt) + m.group(3)
        return m.group(0)

    out = BACKTICK.sub(repl_bt, chunk)
    out = LINKTGT.sub(repl_link, out)
    return out, store


def normalize_headings(chunk: str) -> str:
    ch = "一二三四五六七八九十"
    num = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
           "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}

    def conv(m):
        s = m.group(2)
        if all(c in ch for c in s):
            try:
                v = sum(num[c] for c in s) if len(s) == 1 else int("".join(str(num[c]) for c in s))
            except KeyError:
                return m.group(0)
            return "%s%d. " % (m.group(1), v)
        return m.group(0)

    return re.sub(r"^(#{2,4} )([一二三四五六七八九十]+)、", conv, chunk, flags=re.M)


def repair_trailing(src_chunk: str, out_text: str) -> str:
    tail = []
    for ln in reversed(src_chunk.splitlines()):
        if ln.strip() in ("", "---"):
            tail.insert(0, ln)
        else:
            break
    body = out_text.splitlines()
    while body and body[-1].strip() in ("", "---"):
        body.pop()
    return "\n".join(body + tail) + "\n"


MASK_NOTE = (
    "\n\n---\n\n# 占位符说明（最重要）\n\n"
    "片段里的 `%%Pnn%%` 是**受保护的原始文本占位符**，代表一段不能翻译的原文（多为文件路径）。\n"
    "- 必须**原样保留**在译文的对应位置上，一个字符都不要改。\n"
    "- **不得翻译、不得改写、不得删除、不得解引用**。`%%P01%%` 就是 `%%P01%%`。\n"
    "- 不要把它们换成中文、英文或任何其它写法。\n"
)


def split_chunks(text: str):
    lines = text.splitlines(keepends=True)
    starts = [i for i, ln in enumerate(lines) if re.match(r"^## ", ln)]
    if not starts:
        return [text]
    if starts[0] != 0:
        starts = [0] + starts[1:]
    bounds = starts + [len(lines)]
    return ["".join(lines[bounds[k]:bounds[k + 1]]) for k in range(len(starts))]


def call_ollama(prompt: str, num_predict: int = 4096) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": 20480,
            "num_predict": num_predict,
            "top_p": 0.9,
            "repeat_penalty": 1.05,
        },
    }
    req = urllib.request.Request(
        OLLAMA,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=7200) as r:
        body = json.loads(r.read().decode("utf-8"))
    return body.get("response", "")


def translate_md(cn_path: pathlib.Path) -> str | None:
    """Translate a Chinese .md file to English text via local model."""
    if not SPEC_PATH.exists():
        print("  [WARN] translation spec not found: %s" % SPEC_PATH)
        return None
    try:
        resp = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
        json.loads(resp.read())
    except Exception:
        print("  [WARN] ollama not running — skipping translation")
        return None

    spec = SPEC_PATH.read_text(encoding="utf-8")
    glossary = GLOSSARY.read_text(encoding="utf-8") if GLOSSARY.exists() else ""
    header = spec + "\n\n---\n\n# 术语表\n\n" + glossary

    text = cn_path.read_text(encoding="utf-8")
    chunks = split_chunks(text)
    out_parts = []

    for i, chunk in enumerate(chunks, start=1):
        raw = chunk
        masked, store = mask_paths(normalize_headings(chunk))
        prompt = header + MASK_NOTE + "\n\n---\n\n# 待翻译片段（只译这一块）\n\n" + masked
        t0 = time.time()
        draft = call_ollama(prompt)
        out = draft
        for k, v in store.items():
            out = out.replace(k, v)
        missing = [k for k in store if k not in draft]
        out = repair_trailing(raw, out)
        out_parts.append(out)
        print("  [chunk %02d] %d->%d lines, %d placeholders, %d lost, %.1fs"
              % (i, len(raw.splitlines()), len(out.splitlines()),
                 len(store), len(missing), time.time() - t0))
        if missing:
            print("         lost: %s" % ", ".join(missing))

    return "".join(out_parts)


# --- git helpers ---

def git(repo: pathlib.Path, *args) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "-c", "core.quotepath=false"] + list(args),
        capture_output=True, text=True, encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError("git %s failed: %s" % (" ".join(args), result.stderr.strip()))
    return result.stdout.strip()


# --- main ---

def main():
    parser = argparse.ArgumentParser(description="Sync CN->EN repo with translation")
    parser.add_argument("--dry-run", action="store_true", help="show what would be done, make no changes")
    parser.add_argument("--no-translate", action="store_true", help="copy .md files untranslated")
    parser.add_argument("--no-push", action="store_true", help="commit but don't push")
    args = parser.parse_args()

    # verify paths
    if not CN_ROOT.exists():
        print("ERROR: CN repo not found at %s" % CN_ROOT)
        print("Clone it first:  git clone <gitee-url> %s" % CN_ROOT)
        sys.exit(1)
    if not FILE_MAP.exists():
        print("ERROR: file_map.json not found at %s" % FILE_MAP)
        sys.exit(1)

    file_map = json.loads(FILE_MAP.read_text(encoding="utf-8")).get("map", {})

    # 1. pull CN (get remote changes)
    print("== Pulling CN repo ==")
    if not args.dry_run:
        git(CN_ROOT, "pull", "origin", "master")
    cn_head = git(CN_ROOT, "rev-parse", "HEAD")
    print("  CN HEAD: %s" % cn_head[:8])

    # 2. compare stamp against current HEAD
    if STAMP.exists():
        last_sha = STAMP.read_text(encoding="utf-8").strip()
    else:
        # first run: stamp = current HEAD, no changes to sync
        print("\n== First run: setting sync stamp to current CN HEAD ==")
        if not args.dry_run:
            STAMP.write_text(cn_head + "\n", encoding="utf-8")
        print("  Stamp set to %s" % cn_head[:8])
        print("  Future changes will be synced. Run again after editing CN files.")
        return

    if last_sha == cn_head:
        print("\n  Already synced (stamp matches HEAD).")
        return

    print("\n== Detecting changed files ==")
    diff = git(CN_ROOT, "diff", "--name-status", last_sha, cn_head)
    if not diff:
        print("  No changes detected.")
        return

    changes = []
    for line in diff.splitlines():
        parts = line.split("\t", 1)
        status = parts[0]
        path = parts[1] if len(parts) > 1 else ""
        if status.startswith("R"):  # rename: "R100\told\tnew"
            sub = path.split("\t")
            if len(sub) == 2:
                changes.append(("R", sub[0], sub[1]))
            else:
                changes.append(("A", path, path))
        elif status == "D":
            changes.append(("D", path, None))
        else:
            changes.append((status[0] if status else "A", path, path))

    print("  %d files changed:" % len(changes))
    for s, p, p2 in changes:
        if s == "D":
            print("    D  %s" % p)
        elif s == "R":
            print("    R  %s -> %s" % (p, p2))
        else:
            print("    %s  %s" % (s, p))

    if args.dry_run:
        print("\n[dry-run] no changes made.")
        return

    # 3. process each file
    print("\n== Processing files ==")
    untranslated = []
    for status, cn_rel, new_rel in changes:
        cn_path = CN_ROOT / cn_rel
        if status == "D":
            # file deleted in CN -> delete in EN too
            en_path = EN_ROOT / (file_map.get(cn_rel, cn_rel))
            if en_path.exists():
                en_path.unlink()
                print("  DEL %s" % en_path.relative_to(EN_ROOT))
            continue

        # determine EN path
        en_rel = file_map.get(cn_rel, cn_rel)
        en_path = EN_ROOT / en_rel

        ext = pathlib.Path(cn_rel).suffix.lower()

        if ext in COPY_EXT or ext == "":
            # copy as-is
            en_path.parent.mkdir(parents=True, exist_ok=True)
            en_path.write_bytes(cn_path.read_bytes())
            print("  COPY %s" % en_rel)
            continue

        if ext == ".md":
            if cn_rel in file_map and not args.no_translate:
                # translate
                print("  TRANSLATE %s -> %s" % (cn_rel, en_rel))
                result = translate_md(cn_path)
                if result is not None:
                    en_path.parent.mkdir(parents=True, exist_ok=True)
                    en_path.write_text(result, encoding="utf-8")
                else:
                    # fallback: copy untranslated
                    en_path.parent.mkdir(parents=True, exist_ok=True)
                    en_path.write_text(cn_path.read_text(encoding="utf-8"), encoding="utf-8")
                    untranslated.append(cn_rel)
            else:
                # copy untranslated
                en_path.parent.mkdir(parents=True, exist_ok=True)
                en_path.write_text(cn_path.read_text(encoding="utf-8"), encoding="utf-8")
                print("  COPY (untranslated) %s" % en_rel)
                untranslated.append(cn_rel)
        else:
            # unknown type, copy as-is
            en_path.parent.mkdir(parents=True, exist_ok=True)
            en_path.write_bytes(cn_path.read_bytes())
            print("  COPY %s" % en_rel)

    # 4. update stamp
    STAMP.write_text(cn_head + "\n", encoding="utf-8")
    print("\n  Sync stamp updated to %s" % cn_head[:8])

    if untranslated:
        print("\n  Untranslated files (need manual translation):")
        for f in untranslated:
            print("    %s" % f)

    # 5. commit and push in EN repo
    print("\n== Committing EN repo ==")
    git(EN_ROOT, "add", "-A")
    status = git(EN_ROOT, "status", "--porcelain")
    if not status:
        print("  No changes to commit.")
        return

    commit_msg = "Sync from CN repo (%s)\n\nChanges: %d files\n" % (cn_head[:8], len(changes))
    git(EN_ROOT, "-c", "user.name=空间里子宇宙模型",
        "-c", "user.email=332556447@qq.com",
        "commit", "-m", commit_msg)
    head = git(EN_ROOT, "rev-parse", "HEAD")
    print("  Committed: %s" % head[:8])

    if not args.no_push:
        print("\n== Pushing to GitHub ==")
        try:
            git(EN_ROOT, "push", "github", "master")
            print("  Pushed.")
        except RuntimeError as e:
            print("  Push failed: %s" % e)
            print("  Run manually: git push github master")


if __name__ == "__main__":
    main()
