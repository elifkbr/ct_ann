import os
import re
import numpy as np
import pandas as pd

from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity
)

# -----------------------------------
# DATA FOLDER
# -----------------------------------

DATA_DIR = "data"
LOG_FILE = "grid_run.log"

# -----------------------------------
# LOAD PHANTOM
# -----------------------------------

phantom_path = os.path.join(DATA_DIR, "phantom.npy")
phantom = np.load(phantom_path)

phantom_slice = phantom[phantom.shape[0] // 2].astype(np.float32)
phantom_slice = (phantom_slice - phantom_slice.min()) / (phantom_slice.max() - phantom_slice.min())

# -----------------------------------
# SÜRELER: grid_run.log'dan oku
# -----------------------------------

time_dict = {
    # Eski elle girilen siemens süreler (preset yok)
    "sart_i1_r0.05.npy": 2.816,
    "sart_i1_r0.15.npy": 2.686,
    "sart_i1_r0.50.npy": 2.803,
    "sart_i5_r0.05.npy": 14.070,
    "sart_i5_r0.15.npy": 13.145,
    "sart_i5_r0.50.npy": 13.726,
    "sart_i10_r0.05.npy": 27.644,
    "sart_i10_r0.15.npy": 27.058,
    "sart_i10_r0.50.npy": 26.945,
    "sart_i20_r0.05.npy": 56.809,
    "sart_i20_r0.15.npy": 53.663,
    "sart_i20_r0.50.npy": 53.397,
    "sart_i50_r0.05.npy": 138.085,
    "sart_i50_r0.15.npy": 140.634,
    "sart_i50_r0.50.npy": 139.672,
    "sart_i100_r0.05.npy": 275.246,
    "sart_i100_r0.15.npy": 276.965,
    "sart_i100_r0.50.npy": 283.051,
}

# grid_run.log'dan süreler ekle
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "r") as f:
        for line in f:
            match = re.match(r"[\d:]+ OK (sart_\S+\.npy) ([\d.]+)s", line.strip())
            if match:
                time_dict[match.group(1)] = float(match.group(2))

print(f"Toplam süre kaydı: {len(time_dict)}")

# -----------------------------------
# FIND ALL SART FILES
# -----------------------------------

files = [
    f for f in os.listdir(DATA_DIR)
    if f.startswith("sart_") and f.endswith(".npy")
]

print(f"Toplam dosya: {len(files)}")

results = []

for file in files:
    path = os.path.join(DATA_DIR, file)
    recon = np.load(path)

    recon_slice = recon[recon.shape[0] // 2].astype(np.float32)
    recon_slice = (recon_slice - recon_slice.min()) / (recon_slice.max() - recon_slice.min())

    mse  = mean_squared_error(phantom_slice, recon_slice)
    psnr = peak_signal_noise_ratio(phantom_slice, recon_slice, data_range=1.0)
    ssim = structural_similarity(phantom_slice, recon_slice, data_range=1.0)

    # Parametreleri çıkar
    match = re.search(r"i(\d+)_r([\d]+(?:\.[\d]+)?)(?:_([\w]+))?\.npy", file)
    iteration  = int(match.group(1))
    relaxation = float(match.group(2))
    preset     = match.group(3) if match.group(3) else "siemens_scan"

    results.append({
        "file":        file,
        "scan_preset": preset,
        "iteration":   iteration,
        "relaxation":  relaxation,
        "time_sec":    time_dict.get(file, None),
        "mse":         mse,
        "psnr":        psnr,
        "ssim":        ssim,
    })

# -----------------------------------
# CREATE DATAFRAME
# -----------------------------------

df = pd.DataFrame(results)
df = df.sort_values(by=["scan_preset", "mse"])
df.to_csv("sart_results.csv", index=False)

print("\nSART RESULTS:\n")
print(df.groupby("scan_preset")[["psnr", "ssim", "time_sec"]].mean().round(4))
print(f"\nToplam satır: {len(df)}")
print("Saved as sart_results.csv")