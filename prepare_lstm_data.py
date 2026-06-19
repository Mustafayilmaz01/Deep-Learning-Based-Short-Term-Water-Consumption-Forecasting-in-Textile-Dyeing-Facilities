#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════════════
PREPARE_LSTM_DATA.PY
Aylık Atıksu (W-Water) Zaman Serisi – Sliding-Window Feature Engineering
IEEE Makale / Tez Çalışması  |  Yeniden Üretilebilir Versiyon
═══════════════════════════════════════════════════════════════════════════════

Kronolojik split (ham ay sayısına göre, pencere kaymasından ÖNCE):
  Train      : Jan 2011 – Nov 2019  (108 ay, indeks  0–107)
  Validation : Dec 2019 – Nov 2020  ( 24 ay, indeks 108–131)   ← 2x büyütüldü
  Test       : Jan 2022 – Dec 2022  ( 12 ay, indeks 132–143)
  Toplam     : 144 ay

NOT:
  - Korelasyon YALNIZCA train verisi üzerinden hesaplanır (data leakage önlemi).
  - Tüm meta bilgiler data_lstm/meta.json dosyasına kaydedilir.
  - Sliding-window sonrası her split'in örnek sayısı meta.json'a yazılır.
═══════════════════════════════════════════════════════════════════════════════
"""

# ═══════════════════════════════════════════════════════════════════════════════
# 1. IMPORTS & CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Sabit Ayarlar ──────────────────────────────────────────────────────────────
INPUT_CSV  = os.path.join("data", "fabrika_clean.csv")
OUTPUT_DIR = "data_lstm"

YEAR_COL   = "Year"
MONTH_COL  = "Month"
TARGET_COL = "W-Water"

# Pencere ve ufuk (run_sktime'da W=12 nihai model olarak kullanılır)
WINDOW  = 12
HORIZON = 1

# Kronolojik 3'lü split – ham ay sayısına göre (144 ay toplam)
#   Train      : 108 ay → train_ratio = 108/144   (val 2x büyütüldü, DL stabilite için)
#   Validation :  24 ay → val_ratio   =  24/144
#   Test       :  12 ay → test_ratio  =  12/144
TRAIN_RATIO = 108 / 144   # 0.7500
VAL_RATIO   =  24 / 144   # 0.1667

# Kullanılacak özellik sütunları (0-tabanlı indeks, Year/Month hariç sonraki sütunlar)
# None → tüm sütunlar kullanılır
SELECTED_X_COLS = [9, 10, 17]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════
df = pd.read_csv(INPUT_CSV)

df["Date"] = pd.to_datetime(
    df[YEAR_COL].astype(str) + "-" +
    df[MONTH_COL].astype(str).str.zfill(2) + "-01"
)
df = df.sort_values("Date").reset_index(drop=True).set_index("Date")

print("Veri okundu")
print(f"  Zaman araligi : {df.index.min().strftime('%Y-%m')} -> {df.index.max().strftime('%Y-%m')}")
print(f"  Toplam ay     : {len(df)}")

n_total = len(df)
assert n_total == 144, f"Beklenen 144 ay, bulunan: {n_total}"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING – X / y AYIRMA
# ═══════════════════════════════════════════════════════════════════════════════
X_all = df.drop(columns=[YEAR_COL, MONTH_COL], errors="ignore")

if SELECTED_X_COLS is None:
    X_df = X_all.copy()
else:
    X_df = X_all.iloc[:, SELECTED_X_COLS].copy()

y_df          = df[[TARGET_COL]]
feature_names = X_df.columns.tolist()
num_features  = len(feature_names)

print("\nKullanilan X sutunlari:")
for i, col in enumerate(feature_names):
    print(f"  [{i}] {col}")
print(f"Hedef: {TARGET_COL}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TRAIN / VALIDATION / TEST SPLIT (ham indeksler)
# ═══════════════════════════════════════════════════════════════════════════════
train_end_idx = int(round(n_total * TRAIN_RATIO))                # 120
val_end_idx   = int(round(n_total * (TRAIN_RATIO + VAL_RATIO)))  # 132

df_train = df.iloc[:train_end_idx]    # Jan 2011 – Dec 2020
df_val   = df.iloc[train_end_idx:val_end_idx]   # Jan 2021 – Dec 2021
df_test  = df.iloc[val_end_idx:]                # Jan 2022 – Dec 2022

print(f"\nSplit (ham ay):")
print(f"  Train : {df_train.index.min().strftime('%Y-%m')} -> {df_train.index.max().strftime('%Y-%m')}  ({len(df_train)} ay)")
print(f"  Val   : {df_val.index.min().strftime('%Y-%m')} -> {df_val.index.max().strftime('%Y-%m')}  ({len(df_val)} ay)  [2x büyütüldü]")
print(f"  Test  : {df_test.index.min().strftime('%Y-%m')} -> {df_test.index.max().strftime('%Y-%m')}  ({len(df_test)} ay)")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. KORELASYON HEATMAP (SADECE TRAIN VERİSİ)
# ═══════════════════════════════════════════════════════════════════════════════
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"]  = ["Times New Roman"]

X_train_tmp = df_train.drop(columns=[YEAR_COL, MONTH_COL], errors="ignore")
if SELECTED_X_COLS is not None:
    X_train_tmp = X_train_tmp.iloc[:, SELECTED_X_COLS]

corr_df     = pd.concat([X_train_tmp, df_train[[TARGET_COL]]], axis=1)
corr_matrix = corr_df.corr()

fig, ax = plt.subplots(figsize=(10, 8))
im      = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
cbar    = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=10)

ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_yticks(range(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right", fontsize=10)
ax.set_yticklabels(corr_matrix.columns, fontsize=10)

train_start_str = df_train.index.min().strftime("%Y-%m")
train_end_str   = df_train.index.max().strftime("%Y-%m")
ax.set_title(
    f"Correlation Heatmap of W-Water\n"
    f"(Computed on Training Data Only: {train_start_str} – {train_end_str})",
    fontsize=13, fontweight="bold"
)

for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        val = corr_matrix.iloc[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=8, color="white" if abs(val) > 0.5 else "black")

plt.tight_layout()
pdf_path = os.path.join(OUTPUT_DIR, "correlation_heatmap.pdf")
plt.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
plt.close()
print(f"\nKorelasyon PDF kaydedildi (train-only): {pdf_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. SLIDING WINDOW PENCERELEME
# ═══════════════════════════════════════════════════════════════════════════════
X_list, y_list, ts_list = [], [], []
X_notna = ~X_df.isna()
y_notna = ~y_df.isna()

for i in range(n_total - WINDOW - HORIZON + 1):
    w_end      = i + WINDOW
    target_idx = i + WINDOW + HORIZON - 1
    if not X_notna.iloc[i:w_end].values.all():
        continue
    if not y_notna.iloc[target_idx].values.all():
        continue
    X_list.append(X_df.iloc[i:w_end].values)
    y_list.append(y_df.iloc[target_idx].values[0])
    ts_list.append([X_df.index[w_end - 1], X_df.index[target_idx]])

X_windows = np.array(X_list)
y_array   = np.array(y_list)

print(f"\nPencereleme tamamlandi")
print(f"  X_windows : {X_windows.shape}  (samples, window, features)")
print(f"  y         : {y_array.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. DOSYA KAYDETME
# ═══════════════════════════════════════════════════════════════════════════════
np.save(os.path.join(OUTPUT_DIR, "X_windows.npy"), X_windows)

pd.DataFrame(y_array, columns=[TARGET_COL]).to_csv(
    os.path.join(OUTPUT_DIR, "y.csv"), index=False)

ts_df = pd.DataFrame(ts_list, columns=["window_end_month", "target_month"])
ts_df.to_csv(os.path.join(OUTPUT_DIR, "timestamps.csv"), index=False)

# X düzleştirilmiş CSV (isteğe bağlı, analiz için)
csv_columns = [f"{f}_t-{WINDOW - 1 - t}" for t in range(WINDOW) for f in feature_names]
X_flat      = X_windows.reshape(X_windows.shape[0], -1)
pd.DataFrame(X_flat, columns=csv_columns).to_csv(
    os.path.join(OUTPUT_DIR, "X_windows_flat.csv"), index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. META.JSON KAYDETME
# ═══════════════════════════════════════════════════════════════════════════════
# Pencere kayması sonrası her split'in örnek sayısı:
#   train  : [0 .. train_end_idx) → pencere başlangıcından itibaren train_end_idx - WINDOW adet
#   val    : [train_end_idx .. val_end_idx) → val_end_idx - train_end_idx adet
#   test   : [val_end_idx ..) → n_total - val_end_idx adet
train_samples_w = int(train_end_idx - WINDOW)   # 120 - 12 = 108
val_samples_w   = int(val_end_idx   - train_end_idx)   # 132 - 120 = 12
test_samples_w  = int(n_total       - val_end_idx)     # 144 - 132 = 12

meta = {
    "input_csv":     INPUT_CSV,
    "target_column": TARGET_COL,
    "time_unit":     "monthly",
    "total_months":  int(n_total),
    "date_range":    [df.index.min().strftime("%Y-%m"),
                      df.index.max().strftime("%Y-%m")],

    "window_months":  WINDOW,
    "horizon_months": HORIZON,
    "n_samples":      int(X_windows.shape[0]),

    "n_features":         int(num_features),
    "selected_x_indices": SELECTED_X_COLS,
    "feature_columns":    feature_names,

    "train_ratio": float(TRAIN_RATIO),
    "val_ratio":   float(VAL_RATIO),
    "test_ratio":  float(1.0 - TRAIN_RATIO - VAL_RATIO),

    "train_end_index": int(train_end_idx),
    "val_end_index":   int(val_end_idx),

    "train_date_range": [
        df.index.min().strftime("%Y-%m"),
        df.iloc[train_end_idx - 1].name.strftime("%Y-%m"),
    ],
    "val_date_range": [
        df.iloc[train_end_idx].name.strftime("%Y-%m"),
        df.iloc[val_end_idx - 1].name.strftime("%Y-%m"),
    ],
    "test_date_range": [
        df.iloc[val_end_idx].name.strftime("%Y-%m"),
        df.index.max().strftime("%Y-%m"),
    ],

    "train_samples_after_windowing": train_samples_w,
    "val_samples_after_windowing":   val_samples_w,
    "test_samples_after_windowing":  test_samples_w,

    "correlation_heatmap_computed_on": "training_data_only",
    "correlation_heatmap_path":        pdf_path,
}

with open(os.path.join(OUTPUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print(f"\nmeta.json kaydedildi")
print(f"\nOZET:")
print(f"  Train : {meta['train_date_range'][0]} -> {meta['train_date_range'][1]}"
      f"  ({train_end_idx} ay | pencere sonrasi {train_samples_w} ornek)")
print(f"  Val   : {meta['val_date_range'][0]} -> {meta['val_date_range'][1]}"
      f"  ({val_end_idx - train_end_idx} ay | pencere sonrasi {val_samples_w} ornek)")
print(f"  Test  : {meta['test_date_range'][0]} -> {meta['test_date_range'][1]}"
      f"  ({n_total - val_end_idx} ay | pencere sonrasi {test_samples_w} ornek)")
print(f"  X     : {X_windows.shape}")
print(f"  y     : {y_array.shape}")
