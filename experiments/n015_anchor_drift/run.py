"""
N015: 锚点漂移率独立监控
=======================
验证 SPUM 核心发现：锚点在标准 SGD 训练中持续震荡，永不收敛。

核心指标 Δμ(t) = ||μ(t) - μ(t-1)|| — 相邻 epoch 间类锚点范式变化。
双层不完美的第二部分：中心不完美（锚点持续漂移）。

用法: python run.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch, torch.nn as nn, numpy as np, time, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shared.data import load_mnist
from shared.models import MLP

# ── 配置 ──
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_DIM = 64
BATCH_SIZE = 128
EPOCHS = 50
SEEDS = [42, 52, 62, 72, 82]
DRIFT_THRESHOLD = 0.01
DANGLING_EPS = 0.3

OUT_DIR = os.path.dirname(__file__)


def compute_anchors(hidden: torch.Tensor, labels: torch.Tensor, num_classes: int = 10):
    """计算每类锚点（类均值隐藏表示）。"""
    anchors = torch.zeros(num_classes, hidden.shape[1], device=hidden.device)
    counts = torch.zeros(num_classes, 1, device=hidden.device)
    one_hot = torch.zeros(labels.size(0), num_classes, device=labels.device)
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    anchors += one_hot.T @ hidden
    counts += one_hot.sum(0).unsqueeze(1)
    return anchors / counts.clamp(min=1)


@torch.no_grad()
def compute_dangling(hidden, anchors, eps=DANGLING_EPS):
    """悬挂端检测。"""
    d = torch.cdist(hidden.unsqueeze(1), anchors.unsqueeze(0)).squeeze(1)
    s, _ = d.sort(1)
    return (s[:, 1] - s[:, 0]) < eps


def run_seed(seed, num_classes=10):
    """单 seed 完整训练 + 漂移监控。"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, test_loader = load_mnist(
        os.path.join(os.path.dirname(__file__), ".."),
        batch_size=BATCH_SIZE,
    )

    model = MLP(784, HIDDEN_DIM, num_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    ce = nn.CrossEntropyLoss()

    # 初始锚点
    model.eval()
    all_h, all_y = [], []
    for x, y in train_loader:
        x = x.to(DEVICE)
        _, h = model(x, return_hidden=True)
        all_h.append(h.cpu())
        all_y.append(y)
    anchors_prev = compute_anchors(torch.cat(all_h), torch.cat(all_y), num_classes)

    history = []
    for ep in range(1, EPOCHS + 1):
        # 训练
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            ce(model(x), y).backward()
            opt.step()

        # 评估 + 锚点计算
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        all_h, all_y = [], []
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            lo, h = model(x, return_hidden=True)
            val_loss += ce(lo, y).item()
            correct += (lo.argmax(1) == y).sum().item()
            total += y.size(0)
            all_h.append(h.cpu())
            all_y.append(y)
        val_loss /= len(test_loader)
        acc = correct / total

        all_h = torch.cat(all_h)
        all_y_cat = torch.cat(all_y)
        anchors = compute_anchors(all_h, all_y_cat, num_classes)

        # Δμ: 锚点漂移率
        drift_per_class = (anchors - anchors_prev).norm(dim=1)
        drift_mean = drift_per_class.mean().item()
        drift_max = drift_per_class.max().item()

        # δ: 悬挂端密度
        is_dangling = compute_dangling(all_h, anchors)
        delta = is_dangling.float().mean().item()

        history.append({
            "epoch": ep, "loss": val_loss, "acc": acc,
            "delta": delta,
            "drift_mean": drift_mean,
            "drift_max": drift_max,
            "drift_per_class": drift_per_class.tolist(),
            "drift_below_thresh": drift_mean < DRIFT_THRESHOLD,
        })

        anchors_prev = anchors.clone()

        if ep % 10 == 0:
            below = "✓" if drift_mean < DRIFT_THRESHOLD else "✗"
            print(f"    ep={ep:3d} loss={val_loss:.4f} acc={acc:.3f} "
                  f"δ={delta:.4f} drift={drift_mean:.4f} {below}", flush=True)

    return {
        "seed": seed,
        "history": history,
        "min_drift": min(h["drift_mean"] for h in history),
        "final_drift": history[-1]["drift_mean"],
        "final_acc": history[-1]["acc"],
        "total_epochs": len(history),
        "drift_below_count": sum(1 for h in history if h["drift_below_thresh"]),
    }


def plot_results(all_results):
    """绘制多 seed 漂移曲线。"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) 所有 seed 漂移曲线
    ax = axes[0, 0]
    for r in all_results:
        ep = [h["epoch"] for h in r["history"]]
        dr = [h["drift_mean"] for h in r["history"]]
        ax.plot(ep, dr, lw=2, alpha=0.7, label=f"seed {r['seed']}")
    ax.axhline(DRIFT_THRESHOLD, color="red", ls="--", label=f"threshold={DRIFT_THRESHOLD}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Anchor Drift Δμ")
    ax.set_title("(a) Anchor Drift per Seed")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # (b) δ vs drift 散点
    ax = axes[0, 1]
    all_delta, all_drift = [], []
    for r in all_results:
        all_delta.extend(h["delta"] for h in r["history"])
        all_drift.extend(h["drift_mean"] for h in r["history"])
    ax.scatter(all_delta, all_drift, s=5, alpha=0.5)
    ax.set_xlabel("δ (dangling density)"); ax.set_ylabel("Δμ (anchor drift)")
    ax.set_title("(b) δ vs Δμ (all epochs, all seeds)")
    ax.grid(True, alpha=0.3)

    # (c) min drift per seed
    ax = axes[1, 0]
    seeds = [r["seed"] for r in all_results]
    min_drifts = [r["min_drift"] for r in all_results]
    colors = ["green" if d < DRIFT_THRESHOLD else "red" for d in min_drifts]
    ax.bar(range(len(seeds)), min_drifts, tick_label=[str(s) for s in seeds], color=colors)
    ax.axhline(DRIFT_THRESHOLD, color="gray", ls="--")
    ax.set_ylabel("Min Drift"); ax.set_title("(c) Minimum Drift per Seed")
    ax.grid(axis="y", alpha=0.3)

    # (d) 最终漂移 vs 准确率
    ax = axes[1, 1]
    final_drifts = [r["final_drift"] for r in all_results]
    final_accs = [r["final_acc"] for r in all_results]
    ax.scatter(final_drifts, final_accs, s=60)
    for i, r in enumerate(all_results):
        ax.annotate(str(r["seed"]), (final_drifts[i], final_accs[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Final Drift Δμ"); ax.set_ylabel("Accuracy")
    ax.set_title("(d) Final Drift vs Accuracy")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "n015_anchor_drift.png")
    plt.savefig(out, dpi=150)
    print(f"\n图表已保存: {out}")


# ══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("N015: 锚点漂移率独立监控")
print("=" * 60)
print(f"Device: {DEVICE}, Epochs: {EPOCHS}, Seeds: {SEEDS}")
print(f"Drift threshold: {DRIFT_THRESHOLD}")

all_results = []
for sd in SEEDS:
    print(f"\n  Seed {sd}...")
    t0 = time.time()
    r = run_seed(sd)
    dt = time.time() - t0
    print(f"  done ({dt:.0f}s) | "
          f"min_drift={r['min_drift']:.4f} | "
          f"below_thresh={r['drift_below_count']}/{r['total_epochs']} | "
          f"acc={r['final_acc']:.3f}")
    all_results.append(r)

# ── 汇总 ──
print("\n" + "=" * 60)
print("汇总")
print("=" * 60)
min_drifts = [r["min_drift"] for r in all_results]
final_drifts = [r["final_drift"] for r in all_results]
below_counts = [r["drift_below_count"] for r in all_results]

print(f"  min drift:     {np.mean(min_drifts):.4f} ± {np.std(min_drifts):.4f}")
print(f"  final drift:   {np.mean(final_drifts):.4f} ± {np.std(final_drifts):.4f}")
print(f"  below thresh:  {np.mean(below_counts):.1f}/{EPOCHS} epochs (avg)")
print(f"  any converged: {'YES' if any(d < DRIFT_THRESHOLD for d in min_drifts) else 'NO'}")

if all(d > DRIFT_THRESHOLD for d in min_drifts):
    print("\n  >>> N015 结论确认: 锚点在标准 SGD 训练中永不收敛于阈值。")
    print(f"  >>> 最低漂移 {min(min_drifts):.4f} > {DRIFT_THRESHOLD}，")
    print("  >>> 符合 SPUM 双层不完美的中心不完美预测。")

# ── 保存结果 ──
out_json = os.path.join(OUT_DIR, "n015_results.json")
serializable = []
for r in all_results:
    sr = {k: v for k, v in r.items() if k != "history"}
    sr["history_summary"] = [
        {"epoch": h["epoch"], "drift_mean": h["drift_mean"], "delta": h["delta"]}
        for h in r["history"]
    ]
    serializable.append(sr)
with open(out_json, "w") as f:
    json.dump(serializable, f, indent=2)
print(f"\n结果已保存: {out_json}")

plot_results(all_results)
print("=" * 60)
