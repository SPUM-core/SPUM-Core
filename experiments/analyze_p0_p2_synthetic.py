"""
P0 + P2: 合成数据版本（无需下载 MNIST）
=========================================
用 sklearn 生成可控拓扑结构数据，
测试 SPUM 三条预测。
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

import os, argparse

# ── 配置 ──────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SAMPLES = 20000
N_FEATURES = 20
N_CLASSES = 4
N_INFORMATIVE = 12
N_REDUNDANT = 4
CLASS_SEP = 0.8          # 类间分离度（调小→更多悬挂端）
HIDDEN_DIM = 64
BATCH_SIZE = 128
EPOCHS = 30
DANGLING_EPS = 0.3
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── 生成合成数据 ────────────────────────────────────────────────────
print("=" * 60)
print("生成合成分类数据...")
print(f"  samples={N_SAMPLES}, features={N_FEATURES}, classes={N_CLASSES}")
print(f"  class_sep={CLASS_SEP}")

X, y = make_classification(
    n_samples=N_SAMPLES,
    n_features=N_FEATURES,
    n_informative=N_INFORMATIVE,
    n_redundant=N_REDUNDANT,
    n_classes=N_CLASSES,
    class_sep=CLASS_SEP,
    random_state=SEED,
)

X = StandardScaler().fit_transform(X).astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

train_dataset = torch.utils.data.TensorDataset(
    torch.from_numpy(X_train), torch.from_numpy(y_train).long())
test_dataset = torch.utils.data.TensorDataset(
    torch.from_numpy(X_test), torch.from_numpy(y_test).long())

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"  训练集: {len(train_dataset)}  测试集: {len(test_dataset)}")

# ── 模型 ──────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(N_FEATURES, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, N_CLASSES)

    def forward(self, x, return_hidden=False):
        h = torch.relu(self.fc1(x))
        out = self.fc2(h)
        return (out, h) if return_hidden else out

model = MLP().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
ce_loss = nn.CrossEntropyLoss(reduction="none")

# ── 锚点 ─────────────────────────────────────────────────────────────
@torch.no_grad()
def init_anchors(model, loader):
    anchors = torch.zeros(N_CLASSES, HIDDEN_DIM, device=DEVICE)
    counts = torch.zeros(N_CLASSES, 1, device=DEVICE)
    model.eval()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        _, h = model(x, return_hidden=True)
        for c in range(N_CLASSES):
            mask = y == c
            if mask.any():
                anchors[c] += h[mask].sum(dim=0)
                counts[c] += mask.sum().item()
    anchors /= counts.clamp(min=1)
    return anchors

@torch.no_grad()
def compute_dangling(hidden, anchors, eps=DANGLING_EPS):
    dists = torch.cdist(hidden.unsqueeze(1), anchors.unsqueeze(0)).squeeze(1)
    sorted_dists, _ = dists.sort(dim=1)
    dist_diff = sorted_dists[:, 1] - sorted_dists[:, 0]
    return dist_diff < eps, dist_diff

# ══════════════════════════════════════════════════════════════════════
#  训练
# ══════════════════════════════════════════════════════════════════════
print("\n训练中...")
model.train()
anchors = init_anchors(model, train_loader)

density_history = []
for epoch in range(1, EPOCHS + 1):
    use_dangling = epoch > EPOCHS // 2
    model.train()
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits, hidden = model(x, return_hidden=True)
        is_d, _ = compute_dangling(hidden, anchors)

        losses = ce_loss(logits, y)
        if use_dangling:
            losses[is_d] *= 2.0
        loss = losses.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            for c in range(N_CLASSES):
                mask = y == c
                if mask.any():
                    anchors[c] = 0.9 * anchors[c] + 0.1 * hidden[mask].mean(dim=0)

    # 测试集悬挂端密度
    model.eval()
    d_count, d_total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            _, h = model(x, return_hidden=True)
            is_d, _ = compute_dangling(h, anchors)
            d_count += is_d.sum().item()
            d_total += y.size(0)
    density = d_count / d_total
    density_history.append(density)
    phase = "[悬挂端加权]" if use_dangling else "[正常训练]"
    print(f"  Epoch {epoch:2d} {phase}  δ_test={density:.4f}")

# ══════════════════════════════════════════════════════════════════════
#  收集测试集数据
# ══════════════════════════════════════════════════════════════════════
model.eval()
all_hidden, all_labels, all_preds = [], [], []
all_dist_diff, all_is_dangling = [], []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits, h = model(x, return_hidden=True)
        pred = logits.argmax(1)
        is_d, dd = compute_dangling(h, anchors)
        all_hidden.append(h.cpu())
        all_labels.append(y.cpu())
        all_preds.append(pred.cpu())
        all_dist_diff.append(dd.cpu())
        all_is_dangling.append(is_d.cpu())

hidden = torch.cat(all_hidden).numpy()
labels = torch.cat(all_labels).numpy()
preds = torch.cat(all_preds).numpy()
dist_diff = torch.cat(all_dist_diff).numpy()
is_dangling = torch.cat(all_is_dangling).numpy()

print(f"\n测试集悬挂端比例: {is_dangling.mean():.4f}")

# ══════════════════════════════════════════════════════════════════════
#  P0: 同层 Lipschitz
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("P0: 同层梯度范数 ||∂L/∂h|| 分析")
print("(替代 Lipschitz: 线性层 Lipschitz 是常数，梯度范数才随样本变化)")
print("=" * 60)

ce_loss_fn = nn.CrossEntropyLoss()
sigmas = []  # 不再需要噪声尺度

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# ── 方法1: 梯度范数 ||∂L/∂h|| ──
grad_norms = []
dang_flags = []
for x, y in test_loader:
    x, y = x.to(DEVICE), y.to(DEVICE)
    h = torch.relu(model.fc1(x))
    h.requires_grad_(True)
    logits = model.fc2(h)
    loss = ce_loss_fn(logits, y)
    grad = torch.autograd.grad(loss, h, create_graph=False)[0]
    gn = grad.norm(dim=1).detach().cpu()
    is_d, _ = compute_dangling(h.detach(), anchors)
    grad_norms.append(gn)
    dang_flags.append(is_d.cpu())

grad_norms = torch.cat(grad_norms).numpy()
dang_flags = torch.cat(dang_flags).numpy()

gn_dang = grad_norms[dang_flags == 1]
gn_non = grad_norms[dang_flags == 0]

print(f"\n  梯度范数 ||∂L/∂h||:")
print(f"    非悬挂端: mean={gn_non.mean():.4f}  median={np.median(gn_non):.4f}  n={len(gn_non)}")
print(f"    悬挂端:   mean={gn_dang.mean():.4f}  median={np.median(gn_dang):.4f}  n={len(gn_dang)}")
if len(gn_dang) > 0 and len(gn_non) > 0:
    from scipy.stats import mannwhitneyu
    u_stat, p_val = mannwhitneyu(gn_dang, gn_non, alternative="greater")
    print(f"    MWU (悬挂端>非悬挂端): U={u_stat:.0f}  p={p_val:.6f}  "
          + ("✅ 显著" if p_val < 0.05 else "❌ 不显著"))

bp0 = axes[0].boxplot([gn_non, gn_dang], labels=["非悬挂端", "悬挂端"],
                      patch_artist=True)
bp0["boxes"][0].set_facecolor("steelblue")
bp0["boxes"][1].set_facecolor("coral")
axes[0].set_ylabel("||∂L/∂h||")
axes[0].set_title("(a) 梯度范数")
axes[0].grid(True, axis="y", alpha=0.3)

# ── 方法2: softmax 雅可比范数（重新 forward，避免 graph 冲突）──
jac_norms = []
jac_flags = []
for x, y in test_loader:
    x, y = x.to(DEVICE), y.to(DEVICE)
    h = torch.relu(model.fc1(x)).detach()
    h.requires_grad_(True)
    logits = model.fc2(h)
    probs = torch.softmax(logits, dim=1)
    # 用单次 backward 计算随机投影雅可比范数
    v = torch.randn(probs.size(1), device=DEVICE)
    v = v / v.norm()
    proj = probs @ v
    grad_h = torch.autograd.grad(proj.sum(), h, create_graph=False)[0]
    jn = grad_h.norm(dim=1).detach().cpu()
    is_d, _ = compute_dangling(h.detach(), anchors)
    jac_norms.append(jn)
    jac_flags.append(is_d.cpu())

jac_norms = torch.cat(jac_norms).numpy()
jac_flags = torch.cat(jac_flags).numpy()

jn_dang = jac_norms[jac_flags == 1]
jn_non = jac_norms[jac_flags == 0]

print(f"\n  softmax 雅可比范数 ||∂p/∂h||_F:")
print(f"    非悬挂端: mean={jn_non.mean():.4f}  median={np.median(jn_non):.4f}")
print(f"    悬挂端:   mean={jn_dang.mean():.4f}  median={np.median(jn_dang):.4f}")
if len(jn_dang) > 0 and len(jn_non) > 0:
    u2, p2 = mannwhitneyu(jn_dang, jn_non, alternative="greater")
    print(f"    MWU (悬挂端>非悬挂端): U={u2:.0f}  p={p2:.6f}  "
          + ("✅ 显著" if p2 < 0.05 else "❌ 不显著"))

bp1 = axes[1].boxplot([jn_non, jn_dang], labels=["非悬挂端", "悬挂端"],
                      patch_artist=True)
bp1["boxes"][0].set_facecolor("steelblue")
bp1["boxes"][1].set_facecolor("coral")
axes[1].set_ylabel("||∂p/∂h||_F")
axes[1].set_title("(b) Softmax 雅可比范数")
axes[1].grid(True, axis="y", alpha=0.3)

plt.suptitle("P0: 同层敏感性 (梯度范数 & 雅可比范数)")
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "p0_gradient_norm.png"), dpi=150)
print("\n  图已保存: p0_gradient_norm.png")

# ══════════════════════════════════════════════════════════════════════
#  P2: 永久悬挂端分析
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("P2: 悬挂端标签分布分析")
print("=" * 60)

n_dangling = is_dangling.sum()
print(f"\n  测试集总样本: {len(labels)}")
print(f"  其中悬挂端:  {n_dangling} ({n_dangling/len(labels)*100:.1f}%)")

# 按真实标签统计
print("\n  --- 按真实标签统计 ---")
print(f"  {'类别':>4s}  {'总数':>5s}  {'悬挂端':>7s}  {'比例':>6s}")
print(f"  {'-'*26}")
for c in range(N_CLASSES):
    mask = labels == c
    n_c = mask.sum()
    d_c = is_dangling[mask].sum()
    print(f"  {c:4d}  {n_c:5d}  {d_c:7d}  {d_c/n_c*100:5.1f}%")

# 混淆对
print("\n  --- 悬挂端中常见的混淆对 ---")
dang_mask = is_dangling == 1
true_l = labels[dang_mask]
pred_l = preds[dang_mask]
pairs = defaultdict(int)
for t, p in zip(true_l, pred_l):
    if t != p:
        pairs[(t, p)] += 1
sorted_pairs = sorted(pairs.items(), key=lambda x: -x[1])
print(f"  {'真实→预测':>10s}  {'数量':>6s}")
print(f"  {'-'*18}")
for (t, p), cnt in sorted_pairs[:10]:
    print(f"  {t}→{p}        {cnt:6d}")

# 正确但仍悬挂
correct_dangling = dang_mask & (preds == labels)
n_cd = correct_dangling.sum()
print(f"\n  --- 正确分类但仍悬挂 ---")
print(f"  数量: {n_cd} ({n_cd/n_dangling*100:.1f}% 的悬挂端)")
if n_cd > 0:
    for c in range(N_CLASSES):
        mask = (labels == c) & correct_dangling
        if mask.any():
            dd_vals = dist_diff[mask]
            print(f"    类别 {c}: {mask.sum():5d}  dist_diff={dd_vals.mean():.4f}")

# PCA 投影
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
h_pca = pca.fit_transform(hidden)

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
sc1 = axes2[0].scatter(h_pca[:, 0], h_pca[:, 1], c=is_dangling,
                       cmap="coolwarm", s=8, alpha=0.5)
axes2[0].set_title("P2: 悬挂端分布 (红色=悬挂端)")
plt.colorbar(sc1, ax=axes2[0])
sc2 = axes2[1].scatter(h_pca[:, 0], h_pca[:, 1], c=dist_diff,
                       cmap="viridis", s=8, alpha=0.5)
axes2[1].set_title("P2: dist_diff 分布 (亮=距离大)")
plt.colorbar(sc2, ax=axes2[1])
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "p2_dangling_distribution.png"), dpi=150)
print("\n  分布图已保存")

# ══════════════════════════════════════════════════════════════════════
#  三合一总图（复现原始实验 + P0/P2）
# ══════════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10))

# (a) 密度衰减
axes3[0, 0].plot(density_history, "b-", lw=2)
axes3[0, 0].axvline(EPOCHS // 2, color="gray", ls="--", label="悬挂端加权开始")
axes3[0, 0].set_xlabel("Epoch")
axes3[0, 0].set_ylabel("悬挂端密度 δ")
axes3[0, 0].set_title("(a) 悬挂端密度衰减")
axes3[0, 0].legend()
axes3[0, 0].grid(True, alpha=0.3)

# (b) 双对数
ax_log = axes3[0, 1]
ax_log.loglog(np.arange(1, len(density_history)+1),
              np.array(density_history) + 1e-8, "b-", lw=2)
ax_log.set_xlabel("Epoch (log)")
ax_log.set_ylabel("δ (log)")
ax_log.set_title("(b) 双对数 —— 幂律？")
ax_log.grid(True, alpha=0.3)

# (c) PCA
sc = axes3[1, 0].scatter(h_pca[:, 0], h_pca[:, 1], c=is_dangling,
                         cmap="coolwarm", s=5, alpha=0.5)
axes3[1, 0].set_title("(c) PCA 投影 (红=悬挂端)")
plt.colorbar(sc, ax=axes3[1, 0])

# (d) 梯度范数
gn_all, flag_all = [], []
ce_loss_fn = nn.CrossEntropyLoss()
for x, y in test_loader:
    x, y = x.to(DEVICE), y.to(DEVICE)
    h = torch.relu(model.fc1(x))
    h.requires_grad_(True)
    logits = model.fc2(h)
    loss = ce_loss_fn(logits, y)
    grad = torch.autograd.grad(loss, h, create_graph=False)[0]
    gn = grad.norm(dim=1).detach().cpu()
    is_d, _ = compute_dangling(h.detach(), anchors)
    gn_all.append(gn)
    flag_all.append(is_d.cpu())
gn_all = torch.cat(gn_all).numpy()
flag_all = torch.cat(flag_all).numpy()

axes3[1, 1].boxplot([gn_all[flag_all == 0], gn_all[flag_all == 1]],
                    labels=["非悬挂端", "悬挂端"], patch_artist=True)
axes3[1, 1].patches[0].set_facecolor("steelblue")
axes3[1, 1].patches[1].set_facecolor("coral")
axes3[1, 1].set_title("(d) 梯度范数 ||∂L/∂h||")
axes3[1, 1].grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "p0_p2_summary.png"), dpi=150)
print("\n  汇总图已保存: p0_p2_summary.png")
print("=" * 60)
print("P0 + P2 分析完成")
print("=" * 60)
