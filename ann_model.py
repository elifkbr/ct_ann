import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")

# ===============================
# 1. DATA LOAD
# ===============================

df = pd.read_csv("sart_results.csv")
df = df.dropna(subset=["psnr", "ssim", "time_sec", "iteration", "relaxation", "scan_preset"])

print(f"Dataset: {len(df)} rows")
print("Presets:", df["scan_preset"].unique())

preset_dummies = pd.get_dummies(df["scan_preset"], prefix="preset")

X = np.hstack([
    df[["iteration", "relaxation"]].values,
    preset_dummies.values
])

y_quality = df[["psnr", "ssim"]].values
y_time = np.log1p(df["time_sec"].values).reshape(-1, 1)

print(f"Input size: {X.shape[1]}")
print("Output: PSNR + SSIM")
print("Time is modeled separately.")

# ===============================
# 2. TRAIN / TEST SPLIT
# ===============================

X_trainval, X_test, y_trainval, y_test, yt_trainval, yt_test = train_test_split(
    X, y_quality, y_time, test_size=0.20, random_state=42
)

# ===============================
# 3. ARCHITECTURE SELECTION
# ===============================

architectures = {
    "Small  (32-16)": (32, 16),
    "Medium (64-32)": (64, 32),
    "Large  (128-64-32)": (128, 64, 32)
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

best_arch = None
best_score = -999

print("\n--- 5-Fold Architecture Comparison ---")
print(f"{'Architecture':<22} {'PSNR R2':>10} {'SSIM R2':>10} {'Avg R2':>10}")
print("-" * 56)

for name, layers in architectures.items():
    psnr_scores = []
    ssim_scores = []

    for train_idx, val_idx in kf.split(X_trainval):
        X_tr, X_val = X_trainval[train_idx], X_trainval[val_idx]
        y_tr, y_val = y_trainval[train_idx], y_trainval[val_idx]

        sx = StandardScaler()
        sy = StandardScaler()

        X_tr_s = sx.fit_transform(X_tr)
        X_val_s = sx.transform(X_val)
        y_tr_s = sy.fit_transform(y_tr)

        model = MLPRegressor(
            hidden_layer_sizes=layers,
            activation="relu",
            solver="adam",
            alpha=1e-4,
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42
        )

        model.fit(X_tr_s, y_tr_s)

        pred = sy.inverse_transform(model.predict(X_val_s))

        psnr_scores.append(r2_score(y_val[:, 0], pred[:, 0]))
        ssim_scores.append(r2_score(y_val[:, 1], pred[:, 1]))

    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    avg_total = (avg_psnr + avg_ssim) / 2

    print(f"{name:<22} {avg_psnr:>10.4f} {avg_ssim:>10.4f} {avg_total:>10.4f}")

    if avg_total > best_score:
        best_score = avg_total
        best_arch = layers
        best_arch_name = name

print(f"\nBest architecture: {best_arch_name} -> {best_arch}")

# ===============================
# 4. FINAL QUALITY MODEL
# ===============================

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_trainval_s = scaler_X.fit_transform(X_trainval)
X_test_s = scaler_X.transform(X_test)

y_trainval_s = scaler_y.fit_transform(y_trainval)
y_test_s = scaler_y.transform(y_test)

quality_model = MLPRegressor(
    hidden_layer_sizes=best_arch,
    activation="relu",
    solver="adam",
    alpha=1e-4,
    max_iter=1,
    warm_start=True,
    random_state=42
)

train_losses = []
test_losses = []

best_test_loss = np.inf
patience = 80
patience_counter = 0
best_model_params = None

print("\nTraining final quality model...")

for epoch in range(1, 1001):
    quality_model.fit(X_trainval_s, y_trainval_s)

    train_pred_s = quality_model.predict(X_trainval_s)
    test_pred_s = quality_model.predict(X_test_s)

    train_loss = mean_squared_error(y_trainval_s, train_pred_s)
    test_loss = mean_squared_error(y_test_s, test_pred_s)

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        patience_counter = 0
        best_model_params = [coef.copy() for coef in quality_model.coefs_], [inter.copy() for inter in quality_model.intercepts_]
        best_epoch = epoch
    else:
        patience_counter += 1

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
        break

quality_model.coefs_, quality_model.intercepts_ = best_model_params

# ===============================
# 5. SEPARATE TIME MODEL
# ===============================

scaler_t = StandardScaler()
yt_trainval_s = scaler_t.fit_transform(yt_trainval)

time_model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    max_iter=2000,
    early_stopping=True,
    validation_fraction=0.15,
    random_state=42
)

time_model.fit(X_trainval_s, yt_trainval_s.ravel())

# ===============================
# 6. TEST RESULTS
# ===============================

quality_pred = scaler_y.inverse_transform(quality_model.predict(X_test_s))
time_pred_log = scaler_t.inverse_transform(time_model.predict(X_test_s).reshape(-1, 1))
time_pred = np.expm1(time_pred_log)
time_true = np.expm1(yt_test)

r2_psnr = r2_score(y_test[:, 0], quality_pred[:, 0])
r2_ssim = r2_score(y_test[:, 1], quality_pred[:, 1])
r2_time = r2_score(time_true, time_pred)

print("\n--- Final Test Results ---")
print(f"PSNR R2 : {r2_psnr:.4f}")
print(f"SSIM R2 : {r2_ssim:.4f}")
print(f"Time R2 : {r2_time:.4f}  (separate helper model)")

# ===============================
# 7. OPTIMIZATION GRID
# ===============================

iter_range = np.linspace(1, 100, 100)
relax_range = np.linspace(0.01, 1.0, 100)

ii, rr = np.meshgrid(iter_range, relax_range)
grid_base = np.column_stack([ii.ravel(), rr.ravel()])

preset_columns = list(preset_dummies.columns)

print("\n--- ANN Optimization Results ---")

best_results = {}

for preset in df["scan_preset"].unique():
    onehot_df = pd.DataFrame(columns=preset_columns)
    onehot_vector = np.zeros(len(preset_columns))

    target_col = f"preset_{preset}"
    if target_col in preset_columns:
        onehot_vector[preset_columns.index(target_col)] = 1
    else:
        print(f"Preset column not found for {preset}")
        continue

    oh = np.tile(onehot_vector, (len(grid_base), 1))
    grid = np.hstack([grid_base, oh])

    grid_s = scaler_X.transform(grid)

    pred_quality = scaler_y.inverse_transform(quality_model.predict(grid_s))
    pred_time_log = scaler_t.inverse_transform(time_model.predict(grid_s).reshape(-1, 1))
    pred_time = np.expm1(pred_time_log).ravel()

    psnr = pred_quality[:, 0]
    ssim = pred_quality[:, 1]

    # Normalization
    psnr_n = (psnr - psnr.min()) / (psnr.max() - psnr.min() + 1e-9)
    ssim_n = (ssim - ssim.min()) / (ssim.max() - ssim.min() + 1e-9)
    time_n = (pred_time - pred_time.min()) / (pred_time.max() - pred_time.min() + 1e-9)

    # Final cost function
    score = 0.45 * psnr_n + 0.45 * ssim_n - 0.10 * time_n

    best_quality_idx = np.argmax(0.5 * psnr_n + 0.5 * ssim_n)
    best_balance_idx = np.argmax(score)

    best_results[preset] = {
        "grid": grid_base,
        "psnr": psnr,
        "ssim": ssim,
        "time": pred_time,
        "score": score,
        "best_quality_idx": best_quality_idx,
        "best_balance_idx": best_balance_idx
    }

    print(f"\nPreset: {preset}")

    print(
        f"Best Quality -> "
        f"iter={grid_base[best_quality_idx, 0]:.1f}, "
        f"relax={grid_base[best_quality_idx, 1]:.3f}, "
        f"PSNR={psnr[best_quality_idx]:.4f}, "
        f"SSIM={ssim[best_quality_idx]:.4f}, "
        f"time={pred_time[best_quality_idx]:.2f}s"
    )

    print(
        f"Best Balance -> "
        f"iter={grid_base[best_balance_idx, 0]:.1f}, "
        f"relax={grid_base[best_balance_idx, 1]:.3f}, "
        f"PSNR={psnr[best_balance_idx]:.4f}, "
        f"SSIM={ssim[best_balance_idx]:.4f}, "
        f"time={pred_time[best_balance_idx]:.2f}s"
    )

# ===============================
# 8. PLOTS
# ===============================

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss", linestyle="--")
plt.axvline(best_epoch, linestyle=":", label=f"Best Epoch: {best_epoch}")
plt.title("Training and Test Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True, linestyle=":")
plt.tight_layout()
plt.savefig("loss_curve_final.png", dpi=150)
plt.show()

fig, axes = plt.subplots(3, len(best_results), figsize=(5 * len(best_results), 12))
fig.suptitle("ANN-Based SART Parameter Optimization", fontsize=14, fontweight="bold")

if len(best_results) == 1:
    axes = axes.reshape(3, 1)

for col, (preset, res) in enumerate(best_results.items()):
    g = res["grid"]
    bq = res["best_quality_idx"]
    bb = res["best_balance_idx"]

    measured = df[df["scan_preset"] == preset]

    psnr_surface = res["psnr"].reshape(100, 100)
    ssim_surface = res["ssim"].reshape(100, 100)
    score_surface = res["score"].reshape(100, 100)

    ax = axes[0, col]
    c = ax.contourf(iter_range, relax_range, psnr_surface, levels=20)
    plt.colorbar(c, ax=ax)
    ax.scatter(measured["iteration"], measured["relaxation"], s=12, label="Measured")
    ax.scatter(g[bq, 0], g[bq, 1], s=120, marker="*", label="Best Quality")
    ax.set_title(f"PSNR - {preset}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Relaxation")
    ax.legend(fontsize=7)

    ax = axes[1, col]
    c = ax.contourf(iter_range, relax_range, ssim_surface, levels=20)
    plt.colorbar(c, ax=ax)
    ax.scatter(measured["iteration"], measured["relaxation"], s=12, label="Measured")
    ax.scatter(g[bq, 0], g[bq, 1], s=120, marker="*", label="Best Quality")
    ax.set_title(f"SSIM - {preset}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Relaxation")
    ax.legend(fontsize=7)

    ax = axes[2, col]
    c = ax.contourf(iter_range, relax_range, score_surface, levels=20)
    plt.colorbar(c, ax=ax)
    ax.scatter(measured["iteration"], measured["relaxation"], s=12, label="Measured")
    ax.scatter(g[bb, 0], g[bb, 1], s=120, marker="*", label="Best Balance")
    ax.set_title(f"Quality/Time Score - {preset}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Relaxation")
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig("ann_optimization_final.png", dpi=150)
plt.show()

print("\nSaved figures:")
print("- loss_curve_final.png")
print("- ann_optimization_final.png")