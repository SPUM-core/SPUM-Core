import time, json, sys, numpy as np
import torch, torch.nn as nn
import warnings; warnings.filterwarnings("ignore")

log = open("f:/qingmeng_spum/qingmeng_engine/experiments/n016_out.txt", "w", 1)
def p(s):
    print(s, flush=True)
    log.write(s + "\n"); log.flush()

p("N016 DeepCNN on MNIST")

DEVICE = "cuda"
N_CLASS = 10
BATCH = 256
EPOCHS = 150
PATIENCE = 20
EPS = 0.3
DRIFT_TH = 0.01
ANCHOR_EVERY = 5

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3,1,1), nn.ReLU(),
            nn.Conv2d(32, 32, 3,1,1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3,1,1), nn.ReLU(),
            nn.Conv2d(64, 64, 3,1,1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3,1,1), nn.ReLU(),
            nn.Conv2d(128, 128, 3,1,1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.out = nn.Linear(128, N_CLASS)
        self.h = None
    def forward(self, x):
        x = self.conv(x)
        self.h = x.view(x.size(0), -1)
        return self.out(self.h)

p("Loading...")
data = np.load("f:/qingmeng_spum/qingmeng_engine/experiments/mnist_data.npz")
from sklearn.model_selection import train_test_split
X, y = data["X"].astype(np.float32), data["y"]
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=10000, random_state=42)
X_tr, _, y_tr, _ = train_test_split(X_tr, y_tr, train_size=5000, random_state=42)
p(f"Train: {len(X_tr)}, Val: {len(X_va)}")

def ld(X, y, sh=False):
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X.reshape(-1,1,28,28), dtype=torch.float32).contiguous(),
            torch.tensor(y, dtype=torch.long)),
        BATCH, shuffle=sh, num_workers=0)

tr_ld = ld(X_tr, y_tr, True)
va_ld = ld(X_va, y_va)

model = CNN().to(DEVICE)
p(f"Params: {sum(p.numel() for p in model.parameters())}")
opt = torch.optim.Adam(model.parameters(), lr=0.001)
ce = nn.CrossEntropyLoss()
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

@torch.no_grad()
def get_anchors(ld):
    a = torch.zeros(N_CLASS, 128, device=DEVICE)
    c = torch.zeros(N_CLASS, 1, device=DEVICE)
    model.eval()
    for bx, by in ld:
        model(bx.to(DEVICE))
        h = model.h
        yg = by.to(DEVICE)
        o = torch.zeros(yg.size(0), N_CLASS, device=DEVICE).scatter_(1, yg.unsqueeze(1), 1)
        a += o.T @ h
        c += o.sum(0).unsqueeze(1)
    return a / c.clamp(min=1)

p("Initial anchors...")
anch = get_anchors(tr_ld).clone()
p(f"Anchors: {anch.norm(dim=1)[:3].tolist()}")

hist = {"ep": [], "loss": [], "acc": [], "d": [], "drift": []}
bl = bd = float("inf")
el = ed = 0
wl = wd = 0

p("Training...")
t0 = time.time()
for ep in range(1, EPOCHS + 1):
    model.train()
    for bx, by in tr_ld:
        opt.zero_grad(); ce(model(bx.to(DEVICE)), by.to(DEVICE)).backward(); opt.step()
    sched.step()
    
    model.eval()
    vl = ca = ct = 0.0
    ah = []
    with torch.no_grad():
        for bx, by in va_ld:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            lo = model(bx)
            vl += ce(lo, by).item()
            ca += (lo.argmax(1) == by).sum().item()
            ct += by.size(0)
            ah.append(model.h)
        vl /= len(va_ld)
        ac = ca / ct
        ah = torch.cat(ah)
        d = torch.cdist(ah.unsqueeze(1), anch.unsqueeze(0)).squeeze(1)
        s, _ = d.sort(1)
        dd = float(((s[:,1] - s[:,0]) < EPS).float().mean().item())
    
    drift = None
    if ep % ANCHOR_EVERY == 0:
        new_a = get_anchors(tr_ld)
        drift = float((new_a - anch).norm(dim=1).mean().item())
        anch = new_a.clone()
    
    hist["ep"].append(ep)
    hist["loss"].append(vl)
    hist["acc"].append(ac)
    hist["d"].append(dd)
    hist["drift"].append(drift)
    
    if vl < bl: bl = vl; el = ep; wl = 0
    else: wl += 1
    if dd < bd: bd = dd; ed = ep; wd = 0
    else: wd += 1
    
    dr_s = f"{drift:.4f}" if drift is not None else "."
    
    # Check stability
    drifts = [d for d in hist["drift"] if d is not None]
    stable = len(drifts) >= 3 and all(d < DRIFT_TH for d in drifts[-3:])
    
    p(f"ep={ep:4d} loss={vl:.4f} acc={ac:.4f} d={dd:.4f} drift={dr_s}{' STABLE' if stable else ''}")
    
    if stable:
        p(f"*** SUSTAINED STABILITY @ ep={ep}!")
        break
    
    if wl >= PATIENCE and wd >= PATIENCE:
        p(f">> Early stop @ ep={ep}")
        break

elapsed = time.time() - t0
drifts = np.array([d for d in hist["drift"] if d is not None])
p(f"\nTime: {elapsed:.0f}s")
p(f"Loss stop: ep={el} | Delta stop: ep={ed} | Acc: {ac:.4f}")
p(f"Min drift: {float(drifts.min()):.4f}" if len(drifts) > 0 else "No drift data")
p(f"Drift<{DRIFT_TH}: {int((drifts < DRIFT_TH).sum())}/{len(drifts)}" if len(drifts) > 0 else "No drift data")

# Save
res = {"epochs": len(hist["ep"]), "ep_loss": el, "ep_delta": ed, "acc": ac,
       "min_drift": float(drifts.min()) if len(drifts) else None,
       "drift_below_thresh": int((drifts < DRIFT_TH).sum()) if len(drifts) else 0,
       "checks": len(drifts)}
json.dump(res, open("f:/qingmeng_spum/qingmeng_engine/experiments/n016_res.json", "w"))

# Plot
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 2, figsize=(12, 10))
ax[0,0].plot(hist["ep"], hist["loss"], "r-"); ax[0,0].axvline(el, ls="--"); ax[0,0].grid()
ax[0,1].plot(hist["ep"], hist["d"], "g-"); ax[0,1].axvline(ed, ls="--"); ax[0,1].grid()
dr_plot = [d for d in hist["drift"] if d is not None]
if dr_plot:
    xe = list(range(1, len(dr_plot) * ANCHOR_EVERY + 1, ANCHOR_EVERY))
    ax[1,0].plot(xe, dr_plot, "b-o"); ax[1,0].axhline(DRIFT_TH, c="gray", ls=":"); ax[1,0].grid()
ax[1,1].plot(hist["ep"], hist["acc"], "purple"); ax[1,1].grid()
plt.savefig("f:/qingmeng_spum/qingmeng_engine/experiments/n016_deepcnn_results.png", dpi=150)
p("Plot saved")

if len(drifts) >= 3 and all(d < DRIFT_TH for d in drifts[-3:]):
    p("CONCLUSION: Sustained stability achieved!")
elif len(drifts) > 0 and drifts.min() < DRIFT_TH:
    p("CONCLUSION: Drift briefly below threshold")
else:
    p(f"CONCLUSION: No stability (min_drift={drifts.min():.4f})")
p("DONE")
log.close()
