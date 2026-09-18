# Local-LLM Co-Derivation (Collaborative Derivation of Ontic Models)

> Purpose: The code increments for each layer of OpenSPUM (L0 / L1 / L2 / experimental scripts) are uniformly advanced according to this mode.
> In one sentence: **The main agent writes the interface specification → the local model produces a draft → the main agent reviews and judges → rewrites and implements → bit-for-bit verification → frozen anchor.**
> The local model only acts as a "skeleton generator," **the correctness benchmark is always the CPU reference implementation**, not the local model.

---

## 0. Environment (Locally Tested Specifications)

| Item | Value |
|----|----|
| Portable Python | `C:\Users\macotai\python-sdk\python3.13.2\python.exe` (**not in PATH**, must write full path) |
| Ollama service | `http://localhost:11434` |
| Ollama binary | `C:\Users\macotai\AppData\Local\Programs\Ollama\ollama.exe` (**not in PATH**) |
| Code draft model | `qwen2.5-coder:14b` (tested; `temperature=0.2`) |
| Other available models | `qwen3:14b` (visible on server, not used in this workflow) |
| Main libraries | cupy 14.2.0 / numpy 2.5.3 / scipy 1.18.1 / torch 2.6.0+cu124 |
| CUDA | runtime 12090; RTX 3060 (cc 8.6) |

---

## 1. Five-Step Process

### Step 1 — Main agent writes interface specification (spec)

The specification is written as a **plain text file** (not committed to the repository, placed in a temporary directory), with six fixed sections:

| Section | Content | Highlights |
|--------|---------|----------|
| 0. Environment and Objective | Python path, library version, target file, what to move this increment | One sentence clarifies "which functions are added to which file" |
| 1. Hard Requirements | Prohibits element-wise Python loops; all indices are position-based; full GPU; only outputs function source code | Violations make it unusable, listed item by item |
| 2. Reusable Existing Tools | Signatures and semantics of existing functions | **Prevents model from rewriting existing components** |
| 3. Functions to Implement | Bit-for-bit semantics of each function + parameters + return + **Implementation steps (S1…Sn)** | Steps must be detailed to the "do this and it's correct" level |
| 4. Minimal Fixtures and Expected Values | 3–5 sets of manually computable fixtures + expected arrays | **No fixtures, drafts cannot self-validate** |
| 5. Delivery Requirements | Only outputs functions, Chinese comments, empty input boundaries | Clearly excludes `__main__` / test code |

File naming convention: `l0_gpu_spec{N}.txt` (N = increment number).

### Step 2 — Send to local model

Use **Python `urllib.request` to directly send JSON**, without PowerShell.

- Sending script: `l0_gpu_ask{N}.py`
- Output file: `l0_gpu_out{N}.py`

```python
import json, urllib.request

SPEC = r"C:\Users\macotai\AppData\Local\Temp\l0_gpu_spec3.txt"
OUT  = r"C:\Users\macotai\AppData\Local\Temp\l0_gpu_out3.py"

with open(SPEC, "r", encoding="utf-8") as f:
    prompt = f.read()
payload = {
    "model": "qwen2.5-coder:14b",
    "prompt": prompt,
    "stream": False,
    "options": {"temperature": 0.2, "num_ctx": 16384, "num_predict": 4096},
}
req = urllib.request.Request(
    "http://localhost:11434/api/generate",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=3600) as r:
    body = json.loads(r.read().decode("utf-8"))
text = body.get("response", "")
with open(OUT, "w", encoding="utf-8") as f:
    f.write(text)
print("DONE chars=%d  eval_count=%s" % (len(text), body.get("eval_count")))
```

### Step 3 — Main agent review (fixed checklist)

Go through in order, judge each item as "pass / fail":

1. **Skeleton completeness**: Are all functions required by the specification present? Are implementation steps (S1…Sn) missing?
2. **Undefined references** (most frequent hard bug): Every name used in the draft must be defined in the draft or in the "Reusable Tools" list?
3. **Element-wise loops**: Are there `for i in range(V)` / `for c in cands` loops iterating over nodes, edges, darts, or candidates? (O(1) fixed-round loops are allowed)
4. **Index semantics**: Are "original id" and "position index" mixed?
5. **Empty input boundaries**: Can empty arrays / `m=0` / empty mask return normally? (`.max()` on an empty array will throw an exception, guards needed)
6. **Framework details**: CuPy's `lexsort` **only accepts 2-D arrays, not tuples, last row is the primary key**; `searchsorted` returns int64, index before downcasting.
7. **Ambiguous expressions**: Semantically correct but deeply nested vectorized expressions → judged as "skeleton idea can be retained, rewrite body".

Review conclusions use only three types of wording: **Skeleton Correct** / **Skeleton Gap** / **Hard Bug**.

### Step 4 — Rewrite for implementation

- Start with the draft's **step-by-step skeleton** (step-by-step division is usually correct, directly reuse it);
- Add missing definitions and guards from the model;
- Replace deeply nested expressions with **equivalent but readable** versions;
- Chinese comments and section separators (`# --- S1 … ---`) match the style of this file.

### Step 5 — Verification (three-tiered progression, all passed to complete)

| Level | Method | Description |
|------|--------|-------------|
| ① Fixture | `verify_fixtures()` | Use manual expected values from specification §4, **run this first** — without it, don't run full-scale |
| ② Bit-for-bit | `verify_*_against_cpu()` | Compare ring-sequence dict, `nid`, event counts with CPU reference implementation; multiple configurations × multiple frames |
| ③ End-to-end | `verify_frame_path()` | GPU full path vs pure CPU, frame-by-frame `state_hash` consistency |
| Wrap-up | Full `selftest=1` + existing CLI regression | New capabilities must not break existing CLI |

**Test configuration matrix** (inherits the L0 criteria, reusable across other layers):
`seed × drive × cap × dmin × pairing × piercing` — each cell 4 frames, and **each cell simultaneously covers "with deletion / without deletion, with broken edge / without broken edge"**.

---

## 2. Collaborative Principles

1. **Local models only produce skeletons**, not criteria. The criteria are the CPU reference implementation (`l0_core.py` / `combinatorial_proto.py`).
2. **Specifications must include minimal fixtures**. The expected array that can be manually calculated is the only thing that can make a draft "self-validate"; it is also the first verification after the main agent is implemented.
3. **Bit-for-bit identical = rotation-system dict + `nid` + event count**, all three are indispensable. `state_hash` only covers the rotation system, **not `nid`**, so `nid` must be verified separately (otherwise, new id numbering drift will delay several frames before causing a crash).
4. **Conclusions are immediately frozen into "progress anchors"**, written into `docs/L0L1L2_Architecture.md` §7: list of new items, key simplifications, equivalence basis, verification readings, and parts still remaining in CPU.
5. **Do not presuppose that local models will fail**, but **assume that drafts cannot be run directly**—review and rewriting are fixed steps in the process, not exceptional handling.

---

## 3. Known failure modes in the draft of local models (measured)

| # | Pattern | Example | Handling |
|---|--------|---------|----------|
| 1 | **Reference to undefined variable** | `rebuild_rings` uses `no_alive` in S3, but it is never defined → direct `NameError` | Supplement the definition according to the specification (`no_alive = int(alive.sum())`) |
| 2 | **Vectorized writing detour** | ragged group indices written as nested `searchsorted(bincount(...))` | Keep the skeleton; replace the body with "template order `argsort` + group ok prefix sum − number of ok before this candidate" |
| 3 | **Fixture description error** | The expected value in version B of the spec's fixture is wrong, and the wording is not strict | The specification itself needs to be rechecked; a wrong fixture will make the model answer incorrectly without being noticeable |
| 4 | **Equivalent but redundant** | Using `arange.repeat` instead of `owner_of_dart(rot_ptr)` | Both are equivalent; can be retained |
| 5 | **Sending channel** | PowerShell `Invoke-RestMethod` + `ConvertTo-Json` escapes prompt failed, reported `cannot unmarshal object into Go struct field GenerateRequest.prompt of type string` | Switch to Python `urllib.request` (see step 2), don't mess with encoding |

---

## 4. Reuse Checklist (Directly Follow for Next Derivation)

- [ ] Create `l0_gpu_spec{N}.txt`: six subsections complete, containing 3–5 groups of manual fixtures
- [ ] Create `l0_gpu_ask{N}.py`: direct `urllib` dispatch, `temperature=0.2`
- [ ] Read `l0_gpu_out{N}.py`: pass 7-item review checklist, judgment: "skeleton correct / skeleton gap / hard bug"
- [ ] Rewrite implementation: supplement definitions and guards, replace convoluted writing, maintain subsection comment style
- [ ] Add `verify_fixtures()` (if fixtures exist) and `verify_*_against_cpu()`
- [ ] Hook into `selftest` + add a new CLI entry
- [ ] Run: fixtures → bit-for-bit (matrix) → end-to-end → full `selftest=1` → existing CLI regression
- [ ] Freeze progress anchor point into `docs/L0L1L2_Architecture.md` §7

---

## 5. Completed derivation records under this mode

| increment | target | specification | draft | result |
|----------|--------|---------------|-------|--------|
| Phase III First | Decision segment (`propose` + `priority`) on GPU | — | — | 7 structural invariants + bit-for-bit 24 groups × 4 frames passed |
| Phase III Second | Commit segment judgment (`resolve` + intersection + pruning marker) on GPU, 7 vectorized operators | `l0_gpu_spec2.txt` | `l0_gpu_out2.py` | Commit segment bit-for-bit 16 groups, end-to-end 24 groups passed; CLI `commit=1` |
| Phase III Third | Write operations (`cone_*` / `break_edge` / `remove_vertex`) synthesized into one-time CSR reconstruction | `l0_gpu_spec3.txt` | `l0_gpu_out3.py` (4845 characters / eval_count=1656) | Fixture 5 groups + write operations bit-for-bit 24 groups + end-to-end 24 groups passed; CLI `write=1`; L0 full-frame GPU closed-loop |

> Review conclusion for the third increment (as an example): **High-level skeleton 100% correct** (Steps S1–S9 complete, key packing / masking / ragged expansion ideas correct), **1 skeleton gap + 1 hard bug** (`no_alive` undefined), consistent with specification otherwise → retain skeleton, rewrite body for implementation.
