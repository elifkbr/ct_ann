import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings("ignore")
torch.manual_seed(42)

# 1. VERİ
df = pd.read_csv("sart_results.csv")
df = df.dropna(subset=["psnr", "ssim", "time_sec"])
print(f"Dataset: {len(df)} satır")

preset_dummies = pd.get_dummies(df["scan_preset"], prefix="preset")
X = np.hstack([df[["iteration", "relaxation"]].values, preset_dummies.values]).astype(np.float32)
y = df[["psnr", "ssim"]].values.astype(np.float32)
print(f"Giriş boyutu: {X.shape[1]}")

# 2. TRAIN / TEST
y_time = np.log1p(df["time_sec"].values).reshape(-1, 1).astype(np.float32)
X_tv, X_test, y_tv, y_test, yt_tv, yt_test = train_test_split(X, y, y_time, test_size=0.20, random_state=42)
print(f"Train+Val: {len(X_tv)} | Test: {len(X_test)}")

# 3. MİMARİLER
class SARTNet(nn.Module):
    def __init__(self, input_dim=5, output_dim=2, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),  nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32),   nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    def forward(self, x): return self.net(x)

class FlexNet(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, dropout):
        super().__init__()
        layers = []
        in_d = input_dim
        for h in hidden_layers:
            layers += [nn.Linear(in_d, h), nn.ReLU(), nn.Dropout(dropout)]
            in_d = h
        layers.append(nn.Linear(in_d, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

# 4. K-FOLD KARŞILAŞTIRMA
architectures = {
    "Küçük  (32-16)":     {"hidden": [32, 16],      "dropout": 0.1},
    "Orta   (64-32)":     {"hidden": [64, 32],      "dropout": 0.2},
    "Büyük  (128-64-32)": {"hidden": [128, 64, 32], "dropout": 0.2},
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
loss_fn = nn.MSELoss()

print("\n── Mimari Karşılaştırması (5-Fold CV) ──")
print(f"{'Mimari':<25} {'PSNR R²':>10} {'SSIM R²':>10} {'Ort R²':>10}")
print("-" * 55)

best_score = -999
best_arch_name = None

for name, cfg in architectures.items():
    r2p, r2s = [], []
    for tr_idx, val_idx in kf.split(X_tv):
        X_tr, X_val = X_tv[tr_idx], X_tv[val_idx]
        y_tr, y_val = y_tv[tr_idx], y_tv[val_idx]
        sx = StandardScaler(); sy = StandardScaler()
        Xtr = torch.FloatTensor(sx.fit_transform(X_tr))
        Xval = torch.FloatTensor(sx.transform(X_val))
        ytr = torch.FloatTensor(sy.fit_transform(y_tr))
        m = FlexNet(X.shape[1], cfg["hidden"], 2, cfg["dropout"])
        opt = optim.Adam(m.parameters(), lr=0.01, weight_decay=1e-4)
        m.train()
        for _ in range(500):
            opt.zero_grad(); loss_fn(m(Xtr), ytr).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            yp = sy.inverse_transform(m(Xval).numpy())
        r2p.append(r2_score(y_val[:, 0], yp[:, 0]))
        r2s.append(r2_score(y_val[:, 1], yp[:, 1]))
    ap, as_ = np.mean(r2p), np.mean(r2s)
    avg = (ap + as_) / 2
    print(f"{name:<25} {ap:>10.4f} {as_:>10.4f} {avg:>10.4f}")
    if avg > best_score:
        best_score = avg; best_arch_name = name.strip()

print(f"\n✓ En iyi mimari: {best_arch_name}")

# 5. SCALER + FİNAL MODEL
scaler_X = StandardScaler(); scaler_y = StandardScaler()
X_tv_s   = torch.FloatTensor(scaler_X.fit_transform(X_tv))
y_tv_s   = torch.FloatTensor(scaler_y.fit_transform(y_tv))
X_test_s = torch.FloatTensor(scaler_X.transform(X_test))

scaler_t = StandardScaler()
yt_tv_s  = torch.FloatTensor(scaler_t.fit_transform(yt_tv))

final_model = SARTNet(input_dim=X.shape[1], output_dim=2, dropout=0.2)
optimizer   = optim.Adam(final_model.parameters(), lr=0.01, weight_decay=1e-4)

train_losses, test_losses = [], []
print("\nFinal SARTNet eğitiliyor...")
for epoch in range(1, 1001):
    final_model.train()
    optimizer.zero_grad()
    loss = loss_fn(final_model(X_tv_s), y_tv_s)
    loss.backward(); optimizer.step()
    final_model.eval()
    with torch.no_grad():
        tl = loss_fn(final_model(X_test_s), torch.FloatTensor(scaler_y.transform(y_test)))
    train_losses.append(loss.item()); test_losses.append(tl.item())
    if epoch % 100 == 0:
        print(f"  Epoch [{epoch}/1000] Train: {loss.item():.6f}  Test: {tl.item():.6f}")

# Süre modeli
time_model_sk = __import__('sklearn.neural_network', fromlist=['MLPRegressor']).MLPRegressor(
    hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
    alpha=1e-4, max_iter=2000, early_stopping=True, validation_fraction=0.15, random_state=42
)
from sklearn.neural_network import MLPRegressor as MLPR
time_model_sk = MLPR(hidden_layer_sizes=(64,32), activation="relu", solver="adam",
                     alpha=1e-4, max_iter=2000, early_stopping=True, random_state=42)
time_model_sk.fit(scaler_X.transform(X_tv), scaler_t.transform(yt_tv).ravel())
print("Model eğitildi.")

# 6. TEST
final_model.eval()
with torch.no_grad():
    yp_s = final_model(X_test_s).numpy()
y_test_pred = scaler_y.inverse_transform(yp_s)
yt_test_pred = np.expm1(scaler_t.inverse_transform(
    time_model_sk.predict(scaler_X.transform(X_test)).reshape(-1,1)))
yt_test_real = np.expm1(yt_test)

r2_psnr = r2_score(y_test[:, 0], y_test_pred[:, 0])
r2_ssim = r2_score(y_test[:, 1], y_test_pred[:, 1])
r2_time = r2_score(yt_test_real, yt_test_pred)

print(f"\n── Test Seti Sonuçları ──")
print(f"PSNR  R² : {r2_psnr:.4f}")
print(f"SSIM  R² : {r2_ssim:.4f}")
print(f"Time  R² : {r2_time:.4f}")

# 7. OPTİMİZASYON
iter_range  = np.linspace(1, 100, 100)
relax_range = np.linspace(0.01, 1.0, 100)
ii, rr = np.meshgrid(iter_range, relax_range)
grid_base = np.column_stack([ii.ravel(), rr.ravel()])

presets_onehot = {
    "ge_scan":      [1, 0, 0],
    "hologic_scan": [0, 1, 0],
    "siemens_scan": [0, 0, 1],
}

print("\n── ANN Optimizasyon Sonuçları ──")
best_results = {}

final_model.eval()
for preset_name, onehot in presets_onehot.items():
    oh   = np.tile(onehot, (len(grid_base), 1))
    grid = np.hstack([grid_base, oh]).astype(np.float32)
    g_s  = torch.FloatTensor(scaler_X.transform(grid))
    with torch.no_grad():
        preds = scaler_y.inverse_transform(final_model(g_s).numpy())
    psnr_pred = preds[:, 0]
    ssim_pred = preds[:, 1]
    bp = np.argmax(psnr_pred)

    # Score: 0.45*PSNR + 0.45*SSIM - 0.10*time
    time_pred = np.expm1(scaler_t.inverse_transform(
        time_model_sk.predict(scaler_X.transform(grid)).reshape(-1,1)).ravel())
    psnr_n = (psnr_pred - psnr_pred.min()) / (psnr_pred.max() - psnr_pred.min() + 1e-9)
    ssim_n = (ssim_pred - ssim_pred.min()) / (ssim_pred.max() - ssim_pred.min() + 1e-9)
    time_n = (time_pred - time_pred.min()) / (time_pred.max() - time_pred.min() + 1e-9)
    score  = 0.45 * psnr_n + 0.45 * ssim_n - 0.10 * time_n
    bb = np.argmax(score)

    best_results[preset_name] = {
        "grid": grid_base, "psnr": psnr_pred, "ssim": ssim_pred,
        "best_psnr_idx": bp, "best_bal_idx": bb, "score": score,
    }
    print(f"\n{preset_name}:")
    print(f"  En yüksek PSNR : iter={grid_base[bp,0]:.1f}, relax={grid_base[bp,1]:.3f} → PSNR={psnr_pred[bp]:.4f}, SSIM={ssim_pred[bp]:.4f}")
    print(f"  En iyi denge   : iter={grid_base[bb,0]:.1f}, relax={grid_base[bb,1]:.3f} → PSNR={psnr_pred[bb]:.4f}, SSIM={ssim_pred[bb]:.4f}")

# 8. GÖRSELLEŞTİRME
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(train_losses, label="Eğitim Kaybı (Train Loss)", color="blue", linewidth=2)
ax.plot(test_losses,  label="Test Kaybı (Test Loss)",    color="orange", linestyle="--", linewidth=2)
ax.set_title("SARTNet: Epoch vs MSE Kayıp Eğrisi", fontsize=13, fontweight="bold")
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
ax.legend(); ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout(); plt.savefig("loss_curve_pytorch.png", dpi=150, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("SART Parametre Optimizasyonu — SARTNet (PyTorch)", fontsize=13, fontweight="bold")

for col, (preset_name, res) in enumerate(best_results.items()):
    g = res["grid"]; bp = res["best_psnr_idx"]; bb = res["best_bal_idx"]

    ax = axes[0, col]
    c = ax.contourf(iter_range, relax_range, res["psnr"].reshape(100,100), levels=20, cmap="viridis")
    plt.colorbar(c, ax=ax)
    pdf = df[df["scan_preset"] == preset_name]
    ax.scatter(pdf["iteration"], pdf["relaxation"], c="red", s=15, zorder=5, label="Ölçülen")
    ax.scatter(g[bp,0], g[bp,1], c="yellow", s=150, marker="*", zorder=6, label="Optimum")
    ax.set_title(f"PSNR — {preset_name}"); ax.set_xlabel("Iteration"); ax.set_ylabel("Relaxation")
    ax.legend(fontsize=7)

    ax = axes[1, col]
    c = ax.contourf(iter_range, relax_range, res["ssim"].reshape(100,100), levels=20, cmap="plasma")
    plt.colorbar(c, ax=ax)
    ax.scatter(pdf["iteration"], pdf["relaxation"], c="red", s=15, zorder=5, label="Ölçülen")
    ax.scatter(g[bs,0], g[bs,1], c="yellow", s=150, marker="*", zorder=6, label="Optimum")
    ax.set_title(f"SSIM — {preset_name}"); ax.set_xlabel("Iteration"); ax.set_ylabel("Relaxation")
    ax.legend(fontsize=7)

plt.tight_layout(); plt.savefig("ann_results_pytorch.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nGrafikler kaydedildi.")