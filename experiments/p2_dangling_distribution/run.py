"""
P0 + P2 分析脚本
=================
在已有训练好的模型上分析：
  P0: 同层 Lipschitz（隐藏层扰动 → logits 变化）
  P2: 永久悬挂端标签分布

用法：python analyze_p0_p2.py [--load PATH]  （不传 --load 则先训练）
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from collections import defaultdict

import argparse, os, sys

# ── 配置 ──────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_DIM = 128
BATCH_SIZE = 128
DANGLING_EPS = 0.3
SEED = 42
CKPT_PATH = os.path.join(os.path.dirname(__file__), "model_dangling.pt")

torch.manual_seed(SEED)
np.random.seed(SEED)
rng = np.random.RandomState(SEED)

# ── 数据 ──────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(os.path.join(os.path.dirname(__file__), ".."),
                   train=True, download=True, transform=transform),
    batch_size=BATCH_SIZE, shuffle=True, pin_memory=True,
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(os.path.join(os.path.dirname(__file__), ".."),
                   train=False, download=True, transform=transform),
    batch_size=BATCH_SIZE, shuffle=False, pin_memory=True,
)

# ── 模型 ──────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, 10)

    def forward(self, x, return_hidden=False):
        x = x.view(x.size(0), -1)
        h = torch.relu(self.fc1(x))
        out = self.fc2(h)
        return (out, h) if return_hidden else out

# ── 辅助函数 ──────────────────────────────────────────────────────────
NUM_CLASSES = 10

@torch.no_grad()
def compute_dangling_info(hidden, anchors, eps=DANGLING_EPS):
    dists = torch.cdist(hidden.unsqueeze(1), anchors.unsqueeze(0)).squeeze(1)
    sorted_dists, _ = dists.sort(dim=1)
    dist_diff = sorted_dists[:, 1] - sorted_dists[:, 0]
    is_dangling = dist_diff < eps
    return is_dangling, dist_diff

# ══════════════════════════════════════════════════════════════════════
#  训练
# ══════════════════════════════════════════════════════════════════════
def train():
    print("=" * 60)
    print("训练 MLP on MNIST ...")
    model = MLP().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    ce_loss = nn.CrossEntropyLoss(reduction="none")

    # 初始化锚点
    model.eval()
    anchors = torch.zeros(NUM_CLASSES, HIDDEN_DIM, device=DEVICE)
    counts = torch.zeros(NUM_CLASSES, 1, device=DEVICE)
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            _, h = model(data, return_hidden=True)
            for c in range(NUM_CLASSES):
                mask = target == c
                if mask.any():
                    anchors[c] += h[mask].sum(dim=0)
                    counts[c] += mask.sum().item()
    anchors /= counts.clamp(min=1)

    EPOCHS_WARMUP = 5
    EPOCHS_EXP = 5
    for epoch in range(1, EPOCHS_WARMUP + EPOCHS_EXP + 1):
        use_dangling = epoch > EPOCHS_WARMUP
        model.train()
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            logits, hidden = model(data, return_hidden=True)
            is_dangling, _ = compute_dangling_info(hidden, anchors)

            losses = ce_loss(logits, target)
            if use_dangling:
                losses[is_dangling] *= 2.0
            loss = losses.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 动量更新锚点
            with torch.no_grad():
                for c in range(NUM_CLASSES):
                    mask = target == c
                    if mask.any():
                        anchors[c] = 0.9 * anchors[c] + 0.1 * hidden[mask].mean(dim=0)

        # 本 epoch 精度
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in train_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                logits = model(data)
                correct += (logits.argmax(1) == target).sum().item()
                total += target.size(0)
        phase = "[悬挂端加权]" if use_dangling else "[正常训练]"
        acc = correct / total
        # 测试集悬挂端密度
        dangling_count, total_count = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                _, h = model(data, return_hidden=True)
                is_d, _ = compute_dangling_info(h, anchors)
                dangling_count += is_d.sum().item()
                total_count += target.size(0)
        print(f"  Epoch {epoch:2d} {phase}  acc={acc:.4f}  δ_test={dangling_count/total_count:.4f}")

    # 保存
    torch.save({"model": model.state_dict(), "anchors": anchors.cpu()}, CKPT_PATH)
    print(f"  模型已保存: {CKPT_PATH}")
    return model, anchors

# ══════════════════════════════════════════════════════════════════════
#  P0: 同层 Lipschitz
# ══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def p0_same_layer_lipschitz(model, anchors):
    """
    对测试集每个样本，在隐藏层空间加相对噪声 δh，
    测量 logits 对隐藏层的敏感性：
      L(h) = ||fc2(h) - fc2(h + δh)|| / ||δh||
    不涉及 fc1 和输入空间，纯 L1 → L1 层内 Lipschitz。
    """
    print("\n" + "=" * 60)
    print("P0: 同层 Lipschitz 分析")
    print("=" * 60)

    fc2 = model.fc2  # 只用到这一层

    all_L = []
    all_is_dangling = []
    sigmas = [0.01, 0.02, 0.05]  # 多尺度噪声

    for sigma in sigmas:
        L_vals = []
        dangling_flags = []
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            # 取隐藏层激活
            flattened = data.view(data.size(0), -1)
            h = torch.relu(model.fc1(flattened))

            is_d, _ = compute_dangling_info(h, anchors)
            B = h.size(0)

            # 每个样本按自身范数缩放噪声
            h_norm = h.norm(dim=1, keepdim=True).clamp(min=1e-8)
            noise = torch.randn_like(h) * h_norm * sigma
            h_pert = h + noise

            # 过 fc2（不含 ReLU，保持线性映射）
            out_orig = fc2(h)
            out_pert = fc2(h_pert)

            delta_out = (out_orig - out_pert).norm(dim=1)
            delta_h = noise.norm(dim=1)
            L = delta_out / delta_h.clamp(min=1e-8)

            L_vals.append(L.cpu())
            dangling_flags.append(is_d.cpu())

        L_vals = torch.cat(L_vals).numpy()
        dangling_flags = torch.cat(dangling_flags).numpy()
        all_L.append((sigma, L_vals, dangling_flags))

    # ── 统计 + 画图 ──
    fig, axes = plt.subplots(1, len(sigmas), figsize=(5 * len(sigmas), 4))
    if len(sigmas) == 1:
        axes = [axes]

    for ax, (sigma, L, df) in zip(axes, all_L):
        L_dang = L[df == 1]
        L_non = L[df == 0]
        bp = ax.boxplot([L_non, L_dang], labels=["非悬挂端", "悬挂端"],
                        patch_artist=True)
        bp["boxes"][0].set_facecolor("steelblue")
        bp["boxes"][1].set_facecolor("coral")
        ax.set_title(f"σ = {sigma}")
        ax.set_ylabel("同层 Lipschitz L(h)")
        ax.grid(True, axis="y", alpha=0.3)

        # 数值输出
        print(f"\n  σ = {sigma}:")
        print(f"    非悬挂端: mean={L_non.mean():.4f}  median={np.median(L_non):.4f}  "
              f"n={len(L_non)}")
        print(f"    悬挂端:   mean={L_dang.mean():.4f}  median={np.median(L_dang):.4f}  "
              f"n={len(L_dang)}")
        if len(L_dang) > 0 and len(L_non) > 0:
            from scipy.stats import mannwhitneyu
            u_stat, p_val = mannwhitneyu(L_dang, L_non, alternative="greater")
            print(f"    MWU (悬挂端>非悬挂端): U={u_stat:.0f}  p={p_val:.6f}  "
                  + ("✅ 显著" if p_val < 0.05 else "❌ 不显著"))

    plt.suptitle("P0: 同层 Lipschitz (隐藏层扰动 → logits 敏感性)")
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "p0_same_layer_lipschitz.png")
    plt.savefig(out_path, dpi=150)
    print(f"\n  图已保存: {out_path}")

    # 用 σ=0.02 的 L 值做散点图 vs dist_diff
    _, _, L_best, df_best = None, None, None, None
    for sigma, L, df in all_L:
        if sigma == 0.02:
            L_best, df_best = L, df
            break
    if L_best is not None:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.scatter(np.arange(len(L_best)), L_best, c=df_best, cmap="coolwarm",
                    s=3, alpha=0.5)
        ax2.set_xlabel("样本序号")
        ax2.set_ylabel("L(h)")
        ax2.set_title("P0: 每个样本的同层 Lipschitz (σ=0.02, 红色=悬挂端)")
        out_path2 = os.path.join(os.path.dirname(__file__), "p0_lipschitz_scatter.png")
        plt.savefig(out_path2, dpi=150)
        print(f"  散点图已保存: {out_path2}")

# ══════════════════════════════════════════════════════════════════════
#  P2: 永久悬挂端标签分布
# ══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def p2_dangling_label_distribution(model, anchors):
    print("\n" + "=" * 60)
    print("P2: 永久悬挂端标签分布分析")
    print("=" * 60)

    # 收集测试集所有悬挂端信息
    all_labels = []
    all_preds = []
    all_dist_diff = []
    all_is_dangling = []

    for data, target in test_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        logits, hidden = model(data, return_hidden=True)
        pred = logits.argmax(dim=1)
        is_d, dd = compute_dangling_info(hidden, anchors)

        all_labels.append(target.cpu())
        all_preds.append(pred.cpu())
        all_dist_diff.append(dd.cpu())
        all_is_dangling.append(is_d.cpu())

    labels = torch.cat(all_labels).numpy()
    preds = torch.cat(all_preds).numpy()
    dist_diff = torch.cat(all_dist_diff).numpy()
    is_dangling = torch.cat(all_is_dangling).numpy()

    total = len(labels)
    n_dangling = is_dangling.sum()
    print(f"\n  测试集总样本: {total}")
    print(f"  其中悬挂端:  {n_dangling} ({n_dangling/total*100:.1f}%)")

    # ── 按真实标签统计悬挂端比例 ──
    print("\n  --- 按真实标签统计 ---")
    print(f"  {'类别':>4s}  {'总数':>5s}  {'悬挂端':>7s}  {'比例':>6s}  "
          f"{'预测正确':>8s}  {'悬挂且正确':>10s}")
    print(f"  {'-'*44}")
    for c in range(NUM_CLASSES):
        mask = labels == c
        n_c = mask.sum()
        d_c = is_dangling[mask].sum()
        correct_c = (preds[mask] == c).sum()
        d_and_correct = (is_dangling[mask] & (preds[mask] == c)).sum()
        print(f"  {c:4d}  {n_c:5d}  {d_c:7d}  {d_c/n_c*100:5.1f}%  "
              f"{correct_c:8d}  {d_and_correct:10d}")

    # ── 混淆对分析 ──
    print("\n  --- 悬挂端中常见的混淆对 ---")
    dangling_mask = is_dangling == 1
    true_labels = labels[dangling_mask]
    pred_labels = preds[dangling_mask]
    confusion_pairs = defaultdict(int)
    for t, p in zip(true_labels, pred_labels):
        if t != p:
            confusion_pairs[(t, p)] += 1
    sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: -x[1])
    print(f"  {'真实→预测':>10s}  {'数量':>6s}")
    print(f"  {'-'*18}")
    for (t, p), count in sorted_pairs[:10]:
        print(f"  {t}→{p}        {count:6d}")

    # ── 正确分类但仍悬挂的样本 ──
    correct_dangling = dangling_mask & (preds == labels)
    n_correct_dangling = correct_dangling.sum()
    print(f"\n  --- 正确分类但仍悬挂的样本 ---")
    print(f"  数量: {n_correct_dangling} ({n_correct_dangling/n_dangling*100:.1f}% 的悬挂端)")
    if n_correct_dangling > 0:
        print(f"  分布:")
        for c in range(NUM_CLASSES):
            mask = (labels == c) & correct_dangling
            if mask.any():
                # 这类样本的 dist_diff 分布
                dd_vals = dist_diff[mask]
                print(f"    类别 {c}: {mask.sum():5d}  "
                      f"dist_diff_mean={dd_vals.mean():.4f}  "
                      f"dist_diff_min={dd_vals.min():.4f}")

    # ── 散点图：dist_diff vs 隐藏层 PCA ──
    # 收集隐藏层用于 PCA
    all_h = []
    for data, target in test_loader:
        data = data.to(DEVICE)
        _, h = model(data, return_hidden=True)
        all_h.append(h.cpu())
    all_h = torch.cat(all_h).numpy()

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    h_pca = pca.fit_transform(all_h)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # 左：按是否悬挂着色
    sc1 = axes[0].scatter(h_pca[:, 0], h_pca[:, 1], c=is_dangling,
                          cmap="coolwarm", s=5, alpha=0.5)
    axes[0].set_title("P2: 悬挂端分布 (红色=悬挂端)")
    plt.colorbar(sc1, ax=axes[0])
    # 右：按 dist_diff 着色
    sc2 = axes[1].scatter(h_pca[:, 0], h_pca[:, 1], c=dist_diff,
                          cmap="viridis", s=5, alpha=0.5)
    axes[1].set_title("P2: dist_diff 分布 (亮=距离大)")
    plt.colorbar(sc2, ax=axes[1])
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "p2_dangling_distribution.png")
    plt.savefig(out_path, dpi=150)
    print(f"\n  分布图已保存: {out_path}")


# ══════════════════════════════════════════════════════════════════════
#  main
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, default=None,
                        help="从 checkpoint 加载模型，跳过训练")
    args = parser.parse_args()

    if args.load and os.path.exists(args.load):
        print(f"加载模型: {args.load}")
        model = MLP().to(DEVICE)
        ckpt = torch.load(args.load, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        anchors = ckpt["anchors"].to(DEVICE)
    else:
        model, anchors = train()

    model.eval()

    # P0
    p0_same_layer_lipschitz(model, anchors)

    # P2
    p2_dangling_label_distribution(model, anchors)

    print("\n" + "=" * 60)
    print("P0 + P2 分析完成")
    print("=" * 60)
