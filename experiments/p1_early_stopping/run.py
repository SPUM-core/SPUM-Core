"""N015: 锚点漂移率监控 — 验证锚点稳定性与 δ/loss 收敛的关系"""
import torch, torch.nn as nn, numpy as np, os, time, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

DEVICE = "cpu"
N_FEATURES, N_CLASSES = 784, 10
HIDDEN_DIM, BATCH_SIZE = 64, 128
EPOCHS, PATIENCE = 50, 5
DANGLING_EPS = 0.3
SEEDS = [42, 52, 62, 72, 82]

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(N_FEATURES, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, N_CLASSES)
    def forward(self, x, rh=False):
        h = torch.relu(self.fc1(x))
        return (self.fc2(h), h) if rh else self.fc2(h)

@torch.no_grad()
def dangling_info(hidden, anchors):
    d = torch.cdist(hidden.unsqueeze(1), anchors.unsqueeze(0)).squeeze(1)
    s, _ = d.sort(1)
    return (s[:,1]-s[:,0]) < DANGLING_EPS

@torch.no_grad()
def make_anchors(model, loader):
    a = torch.zeros(N_CLASSES, HIDDEN_DIM)
    c = torch.zeros(N_CLASSES, 1)
    model.eval()
    for x, y in loader:
        _, h = model(x, rh=True)
        # 一次性累加所有类
        one_hot = torch.zeros(y.size(0), N_CLASSES).scatter_(1, y.unsqueeze(1), 1)
        a += one_hot.T @ h
        c += one_hot.sum(0).unsqueeze(1)
    return a / c.clamp(min=1)

def run(seed, X, y):
    torch.manual_seed(seed); np.random.seed(seed)

    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=5000, random_state=seed)  # 减小验证集 10k→5k
    X_train, _, y_train, _ = train_test_split(
        X_train, y_train, train_size=5000, random_state=seed)

    tr = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        BATCH_SIZE, shuffle=True)
    va = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        BATCH_SIZE)

    model = MLP(); opt = torch.optim.Adam(model.parameters(), lr=0.005)
    ce = nn.CrossEntropyLoss()

    anchors_prev = make_anchors(model, tr).clone()

    best_l, best_d = 1e9, 1e9
    ep_l, ep_d = 0, 0
    wl, wd = 0, 0
    hl, hd, ha, hdrift = [], [], [], []
    all_h_list = []

    for ep in range(1, EPOCHS+1):
        model.train()
        for x, y in tr:
            opt.zero_grad(); ce(model(x), y).backward(); opt.step()
        model.eval()
        vl, ca, ct = 0.0, 0, 0
        all_h_list.clear()
        with torch.no_grad():
            for x, y in va:
                lo, h = model(x, rh=True)
                vl += ce(lo, y).item()
                ca += (lo.argmax(1) == y).sum().item()
                ct += y.size(0)
                all_h_list.append(h)
            vl /= len(va); acc = ca / ct
            anchors = make_anchors(model, tr)
            all_h = torch.cat(all_h_list)
            dd = dangling_info(all_h, anchors).float().mean().item()

        # ── N015: 锚点漂移率 ──
        drift_per_class = (anchors - anchors_prev).norm(dim=1)  # [N_CLASSES]
        drift_mean = drift_per_class.mean().item()
        drift_max = drift_per_class.max().item()
        anchors_prev = anchors.clone()

        hl.append(vl); hd.append(dd); ha.append(acc)
        hdrift.append({"mean": drift_mean, "max": drift_max,
                        "per_class": drift_per_class.tolist()})

        if vl < best_l: best_l = vl; ep_l = ep; wl = 0
        else: wl += 1
        if dd < best_d: best_d = dd; ep_d = ep; wd = 0
        else: wd += 1

        if wl >= PATIENCE and wd >= PATIENCE:
            break
        if ep % 10 == 0:
            print(f"      ep={ep:3d} loss={vl:.4f} δ={dd:.4f} "
                  f"drift={drift_mean:.4f}", flush=True)

    return {"seed": seed, "ep_l": ep_l, "ep_d": ep_d,
            "acc_l": ha[ep_l-1], "acc_d": ha[ep_d-1], "acc_final": ha[-1],
            "hl": hl, "hd": hd, "ha": ha, "hdrift": hdrift}

# ── 加载数据 ──
print("=" * 60)
print("N015: 锚点漂移率监控")
print("=" * 60)
data = np.load(os.path.join(os.path.dirname(__file__), "..", "shared", "mnist_data.npz"))
X_all, y_all = data["X"], data["y"]
print(f"MNIST 已加载: X={X_all.shape}, y={y_all.shape}")

results = []
for sd in SEEDS:
    print(f"  Seed {sd}...", end="", flush=True)
    t0 = time.time()
    r = run(sd, X_all, y_all)
    dt = time.time() - t0
    # 计算锚点漂移的一阶差异（epoch-to-epoch）
    drift_means = [d["mean"] for d in r["hdrift"]]
    drift_maxs = [d["max"] for d in r["hdrift"]]

    print(f" done ({dt:.0f}s)")
    print(f"    loss停@ep{r['ep_l']:2d}  δ停@ep{r['ep_d']:2d}")
    print(f"    漂移均值:  start={drift_means[0]:.4f} "
          f"atLossStop={drift_means[min(r['ep_l'], len(drift_means))-1]:.4f} "
          f"atDeltaStop={drift_means[min(r['ep_d'], len(drift_means))-1]:.4f} "
          f"final={drift_means[-1]:.4f}")
    print(f"    漂移最大值: start={drift_maxs[0]:.4f} final={drift_maxs[-1]:.4f}")
    results.append(r)

# ── 跨种子汇总 ──
print("\n--- 汇总 ---")
density_wins = sum(1 for r in results if r["ep_l"] > r["ep_d"])
loss_wins = sum(1 for r in results if r["ep_d"] > r["ep_l"])
print(f"δ 先停: {density_wins}/5   loss 先停: {loss_wins}/5")

# 漂移率在 loss/δ 停止时的平均状态
for stop_type, stop_ep_key in [("loss", "ep_l"), ("δ", "ep_d")]:
    drift_at_stop = []
    for r in results:
        ep = r[stop_ep_key]
        idx = min(ep, len(r["hdrift"])) - 1
        drift_at_stop.append(r["hdrift"][idx]["mean"])
    print(f"{stop_type}停止时 漂移均值: {np.mean(drift_at_stop):.4f}±{np.std(drift_at_stop):.4f}")
    print(f"{stop_type}停止时 漂移最大值: {np.mean([r['hdrift'][min(r[stop_ep_key],len(r['hdrift']))-1]['max'] for r in results]):.4f}")

# ── 画图 ──
r = results[-1]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

drift_means = [d["mean"] for d in r["hdrift"]]
drift_maxs = [d["max"] for d in r["hdrift"]]
ep = np.arange(1, len(r["hl"]) + 1)

# (a) Loss
axes[0,0].plot(ep, r["hl"], "r-", lw=2)
axes[0,0].axvline(r["ep_l"], color="red", ls="--", label=f"loss stop @{r['ep_l']}")
axes[0,0].set_title("(a) Val Loss"); axes[0,0].legend(); axes[0,0].grid()

# (b) δ
axes[0,1].plot(ep, r["hd"], "g-", lw=2)
axes[0,1].axvline(r["ep_d"], color="orange", ls="--", label=f"δ stop @{r['ep_d']}")
axes[0,1].set_title("(b) Dangling δ"); axes[0,1].legend(); axes[0,1].grid()

# (c) 锚点漂移均值 + 最大值
axes[0,2].plot(ep, drift_means, "b-", lw=2, label="drift mean")
axes[0,2].plot(ep, drift_maxs, "c-", lw=1.5, alpha=0.7, label="drift max")
axes[0,2].axvline(r["ep_l"], color="red", ls="--", alpha=0.5)
axes[0,2].axvline(r["ep_d"], color="orange", ls="--", alpha=0.5)
axes[0,2].set_title("(c) 锚点漂移率 Δμ"); axes[0,2].legend(); axes[0,2].grid()

# (d) 三者归一化对比
def norm(arr):
    arr = np.array(arr)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
axes[1,0].plot(ep, norm(r["hl"]), "r-", lw=2, label="loss (norm)")
axes[1,0].plot(ep, norm(r["hd"]), "g-", lw=2, label="δ (norm)")
axes[1,0].plot(ep, norm(drift_means), "b-", lw=2, label="drift (norm)")
axes[1,0].set_title("(d) Loss vs δ vs 漂移 (归一化)")
axes[1,0].legend(); axes[1,0].grid()

# (e) 跨种子停止时间对比
x = np.arange(len(results))
w = 0.25
ax = axes[1,1]
ax.bar(x-w, [r["ep_l"] for r in results], w, label="loss stop", color="red", alpha=0.7)
ax.bar(x, [r["ep_d"] for r in results], w, label="δ stop", color="orange", alpha=0.7)
# 漂移停止时间（漂移均值 < 0.01 的最早 epoch）
drift_stop_eps = []
for r in results:
    dm = [d["mean"] for d in r["hdrift"]]
    stop = next((i+1 for i, v in enumerate(dm) if v < 0.01), len(dm))
    drift_stop_eps.append(stop)
ax.bar(x+w, drift_stop_eps, w, label="drift<0.01", color="blue", alpha=0.5)
ax.set_xticks(x); ax.set_xticklabels([str(r["seed"]) for r in results])
ax.set_ylabel("Stop epoch"); ax.legend(); ax.grid(axis="y")

# (f) 相关性散点：漂移 vs δ
ax = axes[1,2]
all_drift = np.array([d["mean"] for r in results for d in r["hdrift"]])
all_delta = np.array([dd for r in results for dd in r["hd"]])
ax.scatter(all_delta, all_drift, s=5, alpha=0.5)
ax.set_xlabel("δ"); ax.set_ylabel("锚点漂移均值")
ax.set_title("(f) δ vs 漂移 (所有 epoch)")
ax.grid()

plt.tight_layout()
out_png = os.path.join(os.path.dirname(__file__), "p1_mnist_drift.png")
plt.savefig(out_png, dpi=150)
print(f"\n图已保存: {out_png}")

# ── 相关性分析 ──
print("\n--- 相关性分析 ---")
from scipy.stats import pearsonr, spearmanr
for i, r in enumerate(results):
    dm = np.array([d["mean"] for d in r["hdrift"]])
    dd = np.array(r["hd"])
    r_p, p_p = pearsonr(dm, dd)
    r_s, p_s = spearmanr(dm, dd)
    print(f"  Seed {r['seed']}: "
          f"Pearson r={r_p:.3f}(p={p_p:.4f})  "
          f"Spearman ρ={r_s:.3f}(p={p_s:.4f})")

# ── 结论 ──
print("\n--- N015 结论 ---")
# 检查漂移是否在 δ 停止时已经很小
drift_at_delta_stop = []
for r in results:
    idx = min(r["ep_d"], len(r["hdrift"])) - 1
    drift_at_delta_stop.append(r["hdrift"][idx]["mean"])
print(f"  δ 停止时锚点漂移均值: {np.mean(drift_at_delta_stop):.4f}")
print(f"  δ 停止时漂移≈0? {'是' if np.mean(drift_at_delta_stop) < 0.005 else '否'}")

# 漂移最早低于 0.01 的时间
first_drift_stable = [next((i+1 for i, r in enumerate(results[j]["hdrift"]) if r["mean"] < 0.01), EPOCHS) for j in range(len(results))]
print(f"  漂移首次<0.01的epoch: {first_drift_stable}")
print(f"  与δ停止epoch对比: {[r['ep_d'] for r in results]}")
print("=" * 60)
