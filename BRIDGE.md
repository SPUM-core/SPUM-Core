# Bridge contract · spum-core ⇄ Qingmeng engine

> This file is the **human-readable** bridge contract. The machine-readable version is in the same directory [`bridge/manifest.json`](bridge/manifest.json).
> Counterpart file on the other side: `../qingmeng/BRIDGE.md`

## 1. Positioning

`qingmeng/` and `qingmeng-weapp/` were **split out of this repo** on 2026-09-14, becoming the independent project 「Qingmeng engine」.
This repo **no longer contains Qingmeng code**; the two codebases are zero-coupled and exchange **result artifacts** only through this contract.

| | spum-core (this repo) | Qingmeng engine (other side) |
|:---|:---|:---|
| Path | `.` | `../qingmeng` |
| Role | Source: defines the axioms, ontology, and derivation graph | Executor: **runs** the axioms into auditable reasoning frames |
| Artifacts | Axiom rules / node-edge graph / interface contract | Frame snapshots / S vector / consensus report / derivation chain |
| Entry point | `spum.SPUM` (`from spum import SPUM`) | `qingmeng.QingmengEngine` |
| Frontend | — | `weapp/` on the other side (Taro mini-app) |

## 2. Exchange directory

```
bridge/
├── manifest.json          # machine-readable contract (paths, manifest, verification method)
├── exchange/
│   ├── inbox/             # artifacts delivered by the other side (read-only on this side)
│   └── outbox/            # artifacts produced on this side, pending delivery
```

Artifacts do not enter git (`.gitignore` keeps directory-structure placeholders only). Transfer is **performed manually or by a script according to the manifest**; no automatic sync.

### Delivery procedure

1. Export on this side: write the artifact to `bridge/exchange/outbox/<artifact-id>.<ext>`;
2. Generate the sidecar: a `.manifest.json` with the same name, fields below;
3. Place it into `bridge/exchange/inbox/` on the other side (same path name);
4. The other side verifies against the sidecar `sha256` before use.

### Sidecar fields

```json
{
  "artifact": "s-vector",
  "producer": "qingmeng@0.1.0",
  "generated_at": "2026-09-14T10:00:00+08:00",
  "sha256": "<sha256 hex of payload>",
  "source_frames": 12,
  "notes": "optional"
}
```

## 3. Exports from this side (written for the other side)

| id | Repo path | Purpose |
|:---|:---|:---|
| `spum-core-rules` | `.trae/rules/spum-core.md` | Axioms and ontology baseline |
| `spum-evolution-rules` | `.trae/rules/spum-evolution.md` | Five-step frame / creation-annihilation / Imperfection Theorem |
| `spum-wuxing-rules` | `.trae/rules/spum-wuxing.md` | Five Forms → S vector definition |
| `spum-knowledge-core` | `knowledge-core.md` | Condensed authoritative baseline |
| `graph-nodes` | `network/nodes.txt` | 22 core node definitions |
| `graph-edges` | `network/edges.txt` | Derivation edges (derives_from / requires / refines…) |
| `qingmeng-guard` | `.trae/rules/spum-qingmeng-guard.md` | **Interface contract + 11 invariants** |

## 4. Imports into this side (reading the other side)

| id | Producing method | Content |
|:---|:---|:---|
| `frame-snapshot` | `QingmengEngine.summarize()` | Frame-structure summary (topological metrics) |
| `s-vector` | `QingmengEngine.emergence()` | Five-Form phase vector `S = (水,木,土,金,火)` |
| `consensus-report` | `QingmengEngine.check()` | Verification results for the 11 invariants (including violation details) |
| `reasoning-trace` | `QingmengEngine.reason(prompt)` | Derivation chain of the L2 frame reasoning loop |
| `domain-interpretation` | `QingmengEngine.interpret(domain)` | Topological signals → domain semantics (physics / sociology) |

## 5. Hard constraints

1. **No cross-repo import**: this repo must not `import qingmeng`, and the other side must not `import spum`.
   When the other side's data is needed, go through the artifact files in the exchange directory.
2. **The axiom layer is revised in this repo only**. If a `consensus-report` returned from the other side exposes an axiom problem,
   this repo changes the rule files (`.trae/rules/spum-*.md` + `knowledge*.md`), and the other side upgrades accordingly.
3. **Guardrail retained**: `.trae/rules/spum-qingmeng-guard.md` defines this repo's minimum interface contract with the Qingmeng engine
   (11 invariants); before modifying the other side's code, verification against that file is required.
4. **Artifacts must carry a sidecar**: an artifact without a `sha256` counts as undelivered.

## 6. Files on the other side

- `../qingmeng/BRIDGE.md`
- `../qingmeng/bridge/manifest.json`
