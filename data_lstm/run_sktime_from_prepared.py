#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
W-Water Monthly Wastewater Forecasting — sktime Models
IEEE Academic Study | Leakage-Free Approach B

Split (post-windowing):  Train=108 | Validation=12 | Test=12
Final window: W=12  |  Metrics: RMSE, MAE, MAPE, MASE, RMSSE
"""

# ── Suppress TF/C++ noise before any imports ──────────────────────────────────
import os, sys, logging
os.environ.update({
    "TF_ENABLE_ONEDNN_OPTS": "0",
    "TF_DETERMINISTIC_OPS": "1",
    "TF_CUDNN_DETERMINISTIC": "1",
    "TF_CPP_MIN_LOG_LEVEL": "3",       # suppress C++ INFO/WARNING/ERROR logs
    "GRPC_VERBOSITY": "ERROR",
    "AUTOGRAPH_VERBOSITY": "0",
})
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

import json, warnings, time, random as _random
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED); _random.seed(RANDOM_SEED)
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
try:
    import tensorflow as _tf; _tf.random.set_seed(RANDOM_SEED)
    # Also suppress TF Python-level warnings
    _tf.get_logger().setLevel("ERROR")
except Exception:
    pass

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════
PREP_DIR   = "data_lstm"
OUTPUT_DIR = "results_sktime_wwater_ieee"

TRAIN_RATIO = 108 / 144   # 108 ay: Jan 2011 – Dec 2019
VAL_RATIO   =  24 / 144   # 24 ay:  Jan 2020 – Dec 2021
# Test       :  12 ay:  Jan 2022 – Dec 2022

FINAL_WINDOW           = 12          # selected from WINDOW_SIZE_CANDIDATES
WINDOW_SIZE_CANDIDATES = [6, 12, 18, 24, 36]
BATCH_SIZE_CANDIDATES  = [8, 16, 32, 64]

ES_PATIENCE   = 15
ES_MIN_DELTA  = 1e-4
ES_MAX_EPOCHS = 100
N_BOOTSTRAP   = 1000
CI_LEVEL      = 0.95
N_SEEDS       = 5
SEEDS         = [42, 123, 7, 99, 2024]
RUN_WALK_FORWARD = False
N_WF_FOLDS       = 6

for sub in ["", "hyperparameter_plots", "model_comparisons", "time_series_plots",
            "combined_plots", "training_curves", "baseline_results", "residual_tests"]:
    os.makedirs(os.path.join(OUTPUT_DIR, sub) if sub else OUTPUT_DIR, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# STYLE & METADATA
# ═════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "lines.linewidth": 1.5, "axes.linewidth": 0.8, "grid.linewidth": 0.5,
    "text.usetex": False,
})

COLORS = {
    "rocketregressor": "#1f77b4", "lstmfcnregressor": "#ff7f0e",
    "inceptiontime": "#d62728",   "naive": "#7f7f7f",
    "seasonalnaive": "#bcbd22",   "arima": "#17becf",
    "sarima": "#9467bd",          "randomforest": "#8c564b",
    "xgboost": "#e377c2",         "actual": "#000000",
}
LS = {  # line styles
    "rocketregressor": "-",  "lstmfcnregressor": "--", "inceptiontime": "-.",
    "naive": ":",            "seasonalnaive": "--",     "arima": "-.",
    "sarima": ":",           "randomforest": "--",      "xgboost": "-.", "actual": "-",
}
MK = {  # markers
    "rocketregressor": "o", "lstmfcnregressor": "s", "inceptiontime": "^",
    "naive": "x",           "seasonalnaive": "+",    "arima": "D",
    "sarima": "v",          "randomforest": "P",     "xgboost": "*",
}
DISPLAY = {
    "RocketRegressor": "ROCKET",      "LSTMFCNRegressor": "LSTM-FCN",
    "InceptionTime":   "InceptionTime", "Naive": "Naive",
    "SeasonalNaive":   "Seasonal Naive", "ARIMA": "ARIMA",
    "SARIMA":          "SARIMA",        "RandomForest": "Random Forest",
    "XGBoost":         "XGBoost",
}
MTYPE = {
    "RocketRegressor": "Kernel DL (no GPU)", "LSTMFCNRegressor": "Deep learning",
    "InceptionTime":   "Deep learning",      "Naive": "Statistical",
    "SeasonalNaive":   "Statistical",        "ARIMA": "Statistical",
    "SARIMA":          "Statistical",        "RandomForest": "ML ensemble",
    "XGBoost":         "ML ensemble",
}

def disp(n): return DISPLAY.get(n, n)
def mstyle(n):
    k = n.lower()
    return COLORS.get(k, "#1f77b4"), LS.get(k, "--"), MK.get(k, "o")
def fmt_ts(ts):
    try: return pd.Timestamp(ts).strftime("%Y-%m")
    except: return str(ts)[:7]
def save_fig(p): plt.tight_layout(); plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
def pj(*a): return os.path.join(*a)   # shorthand
def section(t): print(f"\n{'='*70}\n{t}\n{'='*70}")

# ═════════════════════════════════════════════════════════════════════════════
# METRICS
# ═════════════════════════════════════════════════════════════════════════════
def compute_metrics(y_true, y_pred, y_naive=None):
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100)
    mse  = rmse**2
    mase = rmsse = np.nan
    if y_naive is not None and len(y_naive) > 1:
        d = np.diff(y_naive)
        sm, sr = np.mean(np.abs(d)), np.sqrt(np.mean(d**2))
        if sm > 1e-8: mase  = float(mae  / sm)
        if sr > 1e-8: rmsse = float(rmse / sr)
    return mae, mse, rmse, mape, mase, rmsse

# ═════════════════════════════════════════════════════════════════════════════
# NORMALIZATION  (train-only fit — no leakage)
# ═════════════════════════════════════════════════════════════════════════════
def norm3d_fit(Xtr, Xte):
    _, c, _ = Xtr.shape
    Xtr_n, Xte_n, scs = np.zeros_like(Xtr), np.zeros_like(Xte), []
    for i in range(c):
        sc = MinMaxScaler()
        Xtr_n[:, i, :] = sc.fit_transform(Xtr[:, i, :])
        Xte_n[:, i, :] = sc.transform(Xte[:, i, :])
        scs.append(sc)
    return Xtr_n, Xte_n, scs

def norm3d_apply(X, scs):
    Xo = np.zeros_like(X)
    for i, sc in enumerate(scs): Xo[:, i, :] = sc.transform(X[:, i, :])
    return Xo

def norm3d(Xtr, Xte):
    a, b, _ = norm3d_fit(Xtr, Xte); return a, b

# ═════════════════════════════════════════════════════════════════════════════
# SPLITS
# ═════════════════════════════════════════════════════════════════════════════
def time_split(n, tr, vr=0.0):
    te = int(round(n * tr)); ve = int(round(n * (tr + vr)))
    return np.arange(0, te), np.arange(te, ve), np.arange(ve, n)

def split_from_meta(n, meta):
    ntr = meta.get("train_samples_after_windowing")
    nva = meta.get("val_samples_after_windowing")
    if ntr is None or nva is None: return time_split(n, TRAIN_RATIO, VAL_RATIO)
    ntr, nva = int(ntr), int(nva)
    return np.arange(0, ntr), np.arange(ntr, ntr+nva), np.arange(ntr+nva, n)

# ═════════════════════════════════════════════════════════════════════════════
# BOOTSTRAP CI
# ═════════════════════════════════════════════════════════════════════════════
def bootstrap_ci(y_true, y_pred, n=1000, ci=0.95):
    res = y_true - y_pred; a = 1 - ci
    rng = np.random.default_rng(RANDOM_SEED)
    boots = np.array([y_pred + rng.choice(res, size=len(y_pred), replace=True) for _ in range(n)])
    return np.percentile(boots, 100*a/2, axis=0), np.percentile(boots, 100*(1-a/2), axis=0)

# ═════════════════════════════════════════════════════════════════════════════
# RESIDUAL TEST
# ═════════════════════════════════════════════════════════════════════════════
def residual_tests(y_true, y_pred, name):
    res = y_true - y_pred; out = {"model": disp(name)}
    try:
        lb = acorr_ljungbox(res, lags=[5], return_df=True)
        out["ljung_box_stat_lag5"]   = float(lb["lb_stat"].values[0])
        out["ljung_box_pvalue_lag5"] = float(lb["lb_pvalue"].values[0])
        out["autocorrelation_ok"]    = bool(lb["lb_pvalue"].values[0] > 0.05)
    except Exception as e: out["ljung_box_error"] = str(e)
    return out

# ═════════════════════════════════════════════════════════════════════════════
# KERAS HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def _make_es(monitor="loss"):
    try:
        import tensorflow as tf
        return tf.keras.callbacks.EarlyStopping(
            monitor=monitor, patience=ES_PATIENCE, min_delta=ES_MIN_DELTA,
            restore_best_weights=True, verbose=0)
    except: return None

def _inner_keras(m):
    for a in ("model_", "network_", "clf_"):
        obj = getattr(m, a, None)
        if obj is None: continue
        for s in ("model_", "model", "network_"):
            km = getattr(obj, s, None)
            if km is not None and hasattr(km, "fit") and hasattr(km, "history"): return km
        if hasattr(obj, "fit") and hasattr(obj, "history"): return obj
    return None

def _get_history(m):
    cands = []
    h = getattr(m, "history_", None)
    if h is not None: cands.append(h.history if hasattr(h, "history") else h)
    for a in ("model_", "network_"):
        inner = getattr(m, a, None)
        if inner is not None:
            for s in ("history", "history_"):
                h = getattr(inner, s, None)
                if h is not None: cands.append(h.history if hasattr(h, "history") else h)
    for c in cands:
        if isinstance(c, dict) and c.get("loss"): return c
    return {}

def fit_capture(model, X, y, val_data=None):
    """
    sktime modelini fit eder, temiz bir Keras eğitimi yapar ve history döndürür.
    val_data verilirse:
      1. sktime.fit(X,y)  → Keras mimarisini ve optimizer'ı kur
      2. clone_model()    → tüm weights + optimizer momentumları tamamen sıfırlanır
      3. km.fit(val_data) → temiz train+val loss, early stopping ile
      4. set_weights()    → best weights orijinal modele aktarılır (predict doğru çalışır)
    """
    model.fit(X, y)
    if val_data is not None:
        km = _inner_keras(model)
        if km is not None:
            Xv, yv = val_data
            try:
                import tensorflow as tf
                tf.random.set_seed(RANDOM_SEED)
                # Modeli klonla: weights + optimizer momentumları tamamen sıfırlanır
                new_km = tf.keras.models.clone_model(km)
                # Optimizer'ı sıfırdan derle (eski momentumları siler)
                # Learning rate 1e-4: küçük veri setinde ezberlemeyi yavaşlatır
                opt_cfg = km.optimizer.get_config()
                opt_cfg["learning_rate"] = 5e-4
                new_km.compile(
                    optimizer=km.optimizer.__class__.from_config(opt_cfg),
                    loss=km.loss
                )
                # Temiz eğitim: validation_data dahil
                history = new_km.fit(
                    np.transpose(X, (0,2,1)), y,
                    epochs=getattr(model, "n_epochs", ES_MAX_EPOCHS),
                    batch_size=getattr(model, "batch_size", 32),
                    validation_data=(np.transpose(Xv, (0,2,1)), yv),
                    callbacks=[_make_es("val_loss")],
                    verbose=0
                )
                # Best weights'i orijinal modele aktar (predict doğru epoch'u kullanır)
                km.set_weights(new_km.get_weights())
                return history.history
            except Exception as e:
                print(f"  [WARN] val_loss fit failed: {e}")
        else:
            print(f"  [WARN] {type(model).__name__}: no Keras inner model – val_loss N/A")
    return _get_history(model)

# ═════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═════════════════════════════════════════════════════════════════════════════
def _xticks(ax, ts, n):
    if ts is not None and len(ts) == n:
        step = max(1, n//6); idx = list(range(0, n, step))
        ax.set_xticks(idx); ax.set_xticklabels([fmt_ts(ts[i]) for i in idx], rotation=30, ha="right", fontsize=8)

def plot_epoch_loss(hist, name, save_dir):
    if not hist or not hist.get("loss"): return
    loss = [float(v) for v in hist["loss"]]
    vloss = [float(v) for v in hist.get("val_loss", [])]
    ep = list(range(1, len(loss)+1))
    has_v = bool(vloss) and len(vloss)==len(loss)
    fig, ax = plt.subplots(figsize=(8, 4), facecolor="white"); ax.set_facecolor("#fafafa")
    ax.plot(ep, loss, color="#2166ac", linewidth=1.8, label="Train Loss")
    if has_v:
        bi = int(np.argmin(vloss))+1
        ax.plot(ep, vloss, color="#d6604d", linewidth=1.8, linestyle="--", label="Validation Loss")
        ax.axvline(bi, color="#2ca02c", linestyle=":", linewidth=1.4, label=f"Best Epoch={bi}")
        ax.scatter([bi], [vloss[bi-1]], color="#2ca02c", zorder=5, s=60)
    else:
        ax.text(0.98, 0.95, "Val Loss: N/A (final: train+val)",
                transform=ax.transAxes, fontsize=8, va="top", ha="right", color="#888", style="italic")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (MSE, normalized)")
    ax.set_title(f"Training & Validation Loss – {disp(name)}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); plt.tight_layout()
    p = pj(save_dir, f"{name}_epoch_loss_curve.png")
    plt.savefig(p, dpi=300, bbox_inches="tight", facecolor="white"); plt.close()
    print(f"  Loss curve: {p}")

def plot_training_curves_combined(histories, save_dir):
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="white"); ax.set_facecolor("#fafafa")
    plotted = False
    for name, hist in histories.items():
        if not hist or not hist.get("loss"): continue
        plotted = True; col, ls, _ = mstyle(name)
        ep = list(range(1, len(hist["loss"])+1))
        ax.plot(ep, [float(v) for v in hist["loss"]], linewidth=1.8, color=col, linestyle=ls,
                label=f"{disp(name)} – Train")
        vl = [float(v) for v in hist.get("val_loss", [])]
        if vl and len(vl)==len(ep):
            ax.plot(ep, vl, linewidth=1.5, color=col, linestyle=":", alpha=0.85, label=f"{disp(name)} – Val")
            ax.axvline(int(np.argmin(vl))+1, color=col, linestyle="--", linewidth=0.8, alpha=0.5)
    if not plotted: plt.close(); return
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (MSE, normalized)")
    ax.set_title("Training & Validation Loss – All DL Models"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout(); p = pj(save_dir, "all_dl_models_training_curves.png")
    plt.savefig(p, dpi=300, bbox_inches="tight", facecolor="white"); plt.close()
    print(f"  Combined loss curves: {p}")

def plot_full_series(y_tr, y_va, y_te, y_pred_val, y_pred_test, path, name,
                     ts_tr=None, ts_va=None, ts_te=None,
                     ci_lo_v=None, ci_hi_v=None, ci_lo=None, ci_hi=None):
    nm = disp(name); col, ls, _ = mstyle(name)
    ntr, nva, nte = len(y_tr), len(y_va), len(y_te)
    ntot = ntr + nva + nte
    xtr = np.arange(0, ntr); xva = np.arange(ntr, ntr+nva); xte = np.arange(ntr+nva, ntot)
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.axvspan(xtr[0],      xtr[-1]+0.5,  facecolor="#daeef3", alpha=0.35, label="Train region")
    ax.axvspan(xva[0]-0.5,  xva[-1]+0.5,  facecolor="#fff3cd", alpha=0.45, label="Validation region")
    ax.axvspan(xte[0]-0.5,  xte[-1]+0.5,  facecolor="#fce4ec", alpha=0.35, label="Test region")
    ax.plot(range(ntot), np.concatenate([y_tr, y_va, y_te]),
            color=COLORS["actual"], linewidth=1.6, label="Actual", zorder=5)
    if y_pred_val is not None and len(y_pred_val)==nva:
        ax.plot(xva, y_pred_val, color="#e67e00", linewidth=1.8, linestyle="--",
                label=f"{nm} Val Prediction", zorder=6)
        if ci_lo_v is not None: ax.fill_between(xva, ci_lo_v, ci_hi_v, alpha=0.12, color="#e67e00")
    ax.plot(xte, y_pred_test, color=col, linewidth=1.8, linestyle=ls,
            label=f"{nm} Test Prediction", zorder=7)
    if ci_lo is not None: ax.fill_between(xte, ci_lo, ci_hi, alpha=0.15, color=col, label="95% CI Test")
    if ts_tr is not None:
        try:
            all_ts = np.concatenate([ts_tr, ts_va, ts_te]); step = max(1, ntot//10)
            idx = list(range(0, ntot, step)); ax.set_xticks(idx)
            ax.set_xticklabels([fmt_ts(all_ts[i]) for i in idx], rotation=30, ha="right", fontsize=8)
        except: pass
    ax.axvline(xtr[-1]+0.5, color="#555", linewidth=0.9, linestyle=":")
    ax.axvline(xva[-1]+0.5, color="#555", linewidth=0.9, linestyle=":")
    ytop = ax.get_ylim()[1]
    for x, lbl, c in [(xtr[len(xtr)//2], "Train", "#1a5276"),
                      (xva[len(xva)//2], "Validation", "#7d6608"),
                      (xte[len(xte)//2], "Test", "#922b21")]:
        ax.text(x, ytop, lbl, ha="center", va="top", fontsize=8, color=c, fontweight="bold")
    ax.set_xlabel("Month"); ax.set_ylabel("W-Water (m³)")
    ax.set_title(f"W-Water – Full Series (Train/Val/Test) – {nm}", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=7.5, ncol=3, framealpha=0.9); ax.grid(True, alpha=0.25)
    save_fig(path)

def plot_forecast(y_true, y_pred, path, name, ts=None, ci_lo=None, ci_hi=None, label="Test"):
    nm, n = disp(name), len(y_true); col, ls, _ = mstyle(name); x = np.arange(n)
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(x, y_true, label="Actual", linewidth=1.5, color=COLORS["actual"])
    ax.plot(x, y_pred, label=f"{nm} Prediction", linewidth=1.5, linestyle=ls, color=col)
    if ci_lo is not None: ax.fill_between(x, ci_lo, ci_hi, alpha=0.15, color=col, label="95% CI")
    _xticks(ax, ts, n)
    ax.set_xlabel("Month"); ax.set_ylabel("W-Water (m³)")
    ax.set_title(f"W-Water – {label} Forecast – {nm}")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); save_fig(path)

def plot_parity(y_true, y_pred, path, name):
    nm = disp(name); col, _, _ = mstyle(name)
    mae = float(np.mean(np.abs(y_true-y_pred))); rmse = float(np.sqrt(np.mean((y_true-y_pred)**2)))
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.scatter(y_true, y_pred, s=8, alpha=0.6, edgecolors="none", color=col)
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.7)
    ax.text(0.05, 0.95, f"MAE={mae:.2f}\nRMSE={rmse:.2f}", transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.set_xlabel("Actual W-Water (m³)"); ax.set_ylabel("Predicted W-Water (m³)")
    ax.set_title(f"Parity – {nm}"); ax.grid(True, alpha=0.3); save_fig(path)

def plot_all_single(y_true, preds, path, ts=None):
    n = len(y_true); fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(np.arange(n), y_true, label="Actual", linewidth=2, color=COLORS["actual"], zorder=10)
    for nm, yp in preds.items():
        col, ls, _ = mstyle(nm)
        ax.plot(np.arange(n), yp, label=disp(nm), linewidth=1.5, linestyle=ls, color=col, alpha=0.7)
    _xticks(ax, ts, n); ax.set_xlabel("Month"); ax.set_ylabel("W-Water (m³)")
    ax.set_title("W-Water Test Predictions – All Models")
    ax.legend(fontsize=7, ncol=3); ax.grid(True, alpha=0.3); save_fig(path); print(f"  All-models single: {path}")

def plot_all_subplots(y_true, preds, path, ts=None, ci_dict=None):
    nm_list = list(preds.keys()); n = len(y_true)
    fig, axes = plt.subplots(len(nm_list), 1, figsize=(11, 2.5*len(nm_list)))
    if len(nm_list)==1: axes=[axes]
    for i, (nm, yp) in enumerate(preds.items()):
        ax = axes[i]; col, ls, _ = mstyle(nm)
        ax.plot(np.arange(n), y_true, label="Actual", linewidth=1.5, color=COLORS["actual"])
        ax.plot(np.arange(n), yp, label=disp(nm), linewidth=1.5, linestyle=ls, color=col)
        if ci_dict and nm in ci_dict:
            lo, hi = ci_dict[nm]; ax.fill_between(np.arange(n), lo, hi, alpha=0.15, color=col, label="95% CI")
        mae = float(np.mean(np.abs(y_true-yp))); mape = float(np.mean(np.abs((y_true-yp)/(np.abs(y_true)+1e-8)))*100)
        ax.text(0.02, 0.98, f"MAE={mae:.1f}, MAPE={mape:.2f}%", transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        ax.set_ylabel("W-Water (m³)"); ax.set_title(disp(nm)); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        if ts is not None and len(ts)==n:
            step=max(1,n//5); idx2=list(range(0,n,step)); ax.set_xticks(idx2)
            ax.set_xticklabels([fmt_ts(ts[j]) for j in idx2], rotation=30, ha="right", fontsize=7)
        if i==len(nm_list)-1: ax.set_xlabel("Month")
    save_fig(path); print(f"  All-models subplot: {path}")

def plot_error_boxplot(preds, y_true, path):
    labels=[disp(m) for m in preds]; errors=[np.abs(y_true-yp) for yp in preds.values()]
    fig, ax = plt.subplots(figsize=(max(8, len(labels)*1.2), 4))
    bp = ax.boxplot(errors, labels=labels, patch_artist=True, medianprops=dict(color="black", linewidth=2))
    for patch, nm in zip(bp["boxes"], preds): patch.set_facecolor(COLORS.get(nm.lower(), "#1f77b4")); patch.set_alpha(0.6)
    ax.set_ylabel("Absolute Error (m³)"); ax.set_title("Absolute Error Distribution – All Models")
    ax.grid(True, alpha=0.3, axis="y"); plt.xticks(rotation=30, ha="right")
    plt.tight_layout(); plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Error boxplot: {path}")

def plot_parity_grid(y_true, preds, path, ncols=3):
    nms = list(preds.keys()); n = len(nms)
    if n==0: return
    nc=min(ncols,n); nr=int(np.ceil(n/nc))
    fig, axes = plt.subplots(nr, nc, figsize=(3.5*nc, 3.5*nr), facecolor="white",
                             constrained_layout=True, squeeze=False)
    fig.suptitle("Parity Plots – All Models", fontsize=13, fontweight="bold")
    all_v = [y_true]+list(preds.values())
    gmin,gmax = min(v.min() for v in all_v), max(v.max() for v in all_v)
    pad=(gmax-gmin)*0.05; lims=[gmin-pad, gmax+pad]
    for i, nm in enumerate(nms):
        r,c = divmod(i,nc); ax=axes[r][c]; ax.set_facecolor("#fafafa")
        yp=np.asarray(preds[nm]); col,_,_=mstyle(nm)
        ax.scatter(y_true, yp, s=20, alpha=0.65, color=col, edgecolors="white", linewidths=0.3, zorder=3)
        ax.plot(lims, lims, "k--", linewidth=1, alpha=0.7, zorder=2)
        ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal", adjustable="box")
        mae=float(np.mean(np.abs(y_true-yp))); rmse=float(np.sqrt(np.mean((y_true-yp)**2)))
        ax.text(0.04,0.96,f"RMSE={rmse:.1f}\nMAE={mae:.1f}",transform=ax.transAxes,fontsize=8,va="top",
                bbox=dict(boxstyle="round,pad=0.3",facecolor="white",edgecolor="#cccccc",linewidth=0.7,alpha=0.9))
        ax.set_title(disp(nm), fontsize=10.5, fontweight="bold"); ax.grid(True, alpha=0.25); ax.tick_params(labelsize=7)
        if c==0: ax.set_ylabel("Predicted (m³)", fontsize=9)
        if r==nr-1: ax.set_xlabel("Actual (m³)", fontsize=9)
    for j in range(n, nr*nc):
        r,c=divmod(j,nc); axes[r][c].axis("off")
    plt.savefig(path, dpi=400, bbox_inches="tight", facecolor="white"); plt.close()
    print(f"  Parity grid: {path}")

def plot_hp_grid(tuning_res, name, path):
    PLABELS={"n_epochs":"Number of Epochs","batch_size":"Batch Size","lstm_size":"LSTM Units","num_kernels":"Num Kernels"}
    np_ = len(tuning_res)
    if np_==0: return
    nc=min(2,np_); nr=int(np.ceil(np_/nc))
    fig,axes=plt.subplots(nr,nc,figsize=(6.8*nc,4.3*nr),facecolor="white",constrained_layout=True,squeeze=False)
    fig.suptitle(f"Hyperparameter Sensitivity – {disp(name)}",fontsize=13,fontweight="bold")
    CR,CM="#2166ac","#d6604d"
    for i,(pn,data) in enumerate(tuning_res.items()):
        ax1=axes.flatten()[i]; ax1.set_facecolor("#fafafa")
        vals,rmses,maes=list(data["values"]),data["rmse"],data["mae"]
        xl=PLABELS.get(pn,pn); xp=np.arange(len(vals)); ax2=ax1.twinx()
        l1,=ax1.plot(xp,rmses,marker="o",linewidth=1.8,markersize=7,color=CR,markerfacecolor="white",
                     markeredgewidth=1.8,label="RMSE (Val)",zorder=4)
        l2,=ax2.plot(xp,maes,marker="s",linewidth=1.8,markersize=7,color=CM,linestyle="--",
                     markerfacecolor="white",markeredgewidth=1.8,label="MAE (Val)",zorder=4)
        ax1.set_xticks(xp); ax1.set_xticklabels([str(v) for v in vals],fontsize=8)
        ax1.set_xlabel(xl,fontsize=10); ax1.set_ylabel("Val RMSE (m³)",color=CR,fontsize=10)
        ax2.set_ylabel("Val MAE (m³)",color=CM,fontsize=10)
        ax1.tick_params(axis="y",labelcolor=CR,labelsize=8); ax2.tick_params(axis="y",labelcolor=CM,labelsize=8)
        ax1.grid(True,alpha=0.25,linestyle=":"); ax1.set_title(f"Effect of {xl}",fontsize=10,fontweight="bold",pad=10)
        if [v for v in rmses if not np.isnan(v)]:
            bi=int(np.nanargmin(rmses))
            ax1.axvline(xp[bi],color="#4dac26",linestyle=":",linewidth=1.3,zorder=1)
            ax1.plot(xp[bi],rmses[bi],marker="*",markersize=15,color="#4dac26",zorder=10)
            ax1.annotate(f"Optimal={vals[bi]}",xy=(xp[bi],rmses[bi]),xytext=(0.94,0.94),
                         textcoords="axes fraction",ha="right",va="top",fontsize=8,
                         bbox=dict(boxstyle="round,pad=0.3",facecolor="#ffffcc",edgecolor="#4dac26",alpha=0.95),
                         arrowprops=dict(arrowstyle="->",color="#4dac26",lw=1.0,alpha=0.85))
        ax1.legend([l1,l2],["RMSE (Val)","MAE (Val)"],loc="upper left",fontsize=8.5,framealpha=0.9)
    for j in range(np_,len(axes.flatten())): axes.flatten()[j].axis("off")
    plt.savefig(path,dpi=400,bbox_inches="tight",facecolor="white"); plt.close()
    print(f"  HP grid: {path}")

def plot_sweep(rmse_d, mae_d, param_vals, xlabel, title, path):
    fig,axes=plt.subplots(2,1,figsize=(7,6),facecolor="white",sharex=True)
    for ax,sweep,ylabel in zip(axes,[rmse_d,mae_d],["RMSE","MAE"]):
        ax.set_facecolor("#fafafa")
        for nm,vals in sweep.items():
            col,ls,mk=mstyle(nm)
            ax.plot(param_vals,vals,marker=mk,markersize=7,linewidth=1.8,color=col,linestyle=ls,
                    markerfacecolor="white",markeredgewidth=1.8,label=disp(nm),zorder=4)
        ax.set_ylabel(ylabel,fontsize=10); ax.legend(fontsize=8,loc="upper right"); ax.grid(True,alpha=0.25,linestyle=":")
    axes[-1].set_xlabel(xlabel,fontsize=10); fig.suptitle(title,fontsize=12,fontweight="bold")
    plt.tight_layout(); plt.savefig(path,dpi=300,bbox_inches="tight",facecolor="white"); plt.close()
    print(f"  Sweep: {path}")

def plot_metrics_bar(df, path):
    metrics=[m for m in ["MAE","RMSE","MAPE(%)","MASE","RMSSE"] if m in df.columns]
    fig,axes=plt.subplots(1,len(metrics),figsize=(4.5*len(metrics),4))
    if len(metrics)==1: axes=[axes]
    for ax,met in zip(axes,metrics):
        labels=[disp(m) for m in df["model"].values]; vals=df[met].values
        cols=[COLORS.get(m.lower(),"#1f77b4") for m in df["model"].values]
        bars=ax.bar(range(len(labels)),vals,color=cols,alpha=0.7,edgecolor="black",linewidth=0.5)
        for bar,v in zip(bars,vals):
            if not np.isnan(v): ax.text(bar.get_x()+bar.get_width()/2., bar.get_height(), f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels,rotation=45,ha="right",fontsize=8)
        ax.set_ylabel(met); ax.set_title(met); ax.grid(True,alpha=0.3,axis="y")
    save_fig(path)

def plot_comp_time(train_t, path, inf_t=None):
    models_=list(train_t.keys()); labels=[disp(m) for m in models_]
    tr=[train_t.get(m,0) for m in models_]
    has_inf=inf_t is not None and any(inf_t.get(m,0)>0 for m in models_)
    x=np.arange(len(models_)); w=0.38 if has_inf else 0.55
    fig,ax=plt.subplots(figsize=(max(9,len(models_)*1.3),4),facecolor="white")
    bars=ax.bar(x-(w/2 if has_inf else 0),tr,w,label="Train Time",color="#2166ac",alpha=0.75,edgecolor="black",linewidth=0.5)
    for bar,t in zip(bars,tr):
        if t>0: ax.text(bar.get_x()+bar.get_width()/2.,bar.get_height()+max(tr)*0.01,f"{t:.1f}s",ha="center",va="bottom",fontsize=7.5)
    if has_inf:
        iv=[inf_t.get(m,0) for m in models_]
        bi=ax.bar(x+w/2,iv,w,label="Infer. Time",color="#d6604d",alpha=0.75,edgecolor="black",linewidth=0.5)
        for bar,t in zip(bi,iv):
            if t>0: ax.text(bar.get_x()+bar.get_width()/2.,bar.get_height()+max(tr)*0.01,f"{t:.3f}s",ha="center",va="bottom",fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(labels,rotation=40,ha="right",fontsize=9)
    ax.set_ylabel("Time (s)"); ax.set_title("Computational Cost"); ax.legend(fontsize=9); ax.grid(True,alpha=0.3,axis="y")
    plt.tight_layout(); plt.savefig(path,dpi=300,bbox_inches="tight",facecolor="white"); plt.close()

def plot_seed_var(seed_res, path):
    mets=["MAE","RMSE","MAPE(%)"]; ml=list(seed_res.keys()); labels=[disp(m) for m in ml]
    nseed=max((len([v for v in seed_res[m].get("RMSE",[]) if not np.isnan(v)]) for m in ml),default=0)
    fig,axes=plt.subplots(1,len(mets),figsize=(5.4*len(mets),4.6),facecolor="white")
    if len(mets)==1: axes=[axes]
    x=np.arange(len(ml)); rng=np.random.default_rng(0)
    for ax,met in zip(axes,mets):
        means,stds,raw=[],[],[]
        for m in ml:
            v=[x for x in seed_res[m].get(met,[]) if not np.isnan(x)]
            raw.append(v); means.append(np.mean(v) if v else np.nan); stds.append(np.std(v) if v else np.nan)
        cols=[COLORS.get(m.lower(),"#1f77b4") for m in ml]
        ax.bar(x,means,yerr=stds,capsize=5,color=cols,alpha=0.55,edgecolor="black",linewidth=0.8,width=0.6,
               error_kw=dict(elinewidth=1.4,ecolor="#333333"),zorder=2)
        for xi,v,c in zip(x,raw,cols):
            if v: ax.scatter(xi+rng.uniform(-0.13,0.13,size=len(v)),v,s=36,color=c,edgecolors="black",linewidths=0.6,zorder=4)
        tops=[max(m+s,max(v) if v else m+s) if not np.isnan(m) else np.nan for m,s,v in zip(means,stds,raw)]
        ymax=np.nanmax(tops) if any(not np.isnan(t) for t in tops) else 1.0
        for xi,mn,sd,top in zip(x,means,stds,tops):
            if not np.isnan(mn):
                cv=(sd/mn*100) if abs(mn)>1e-9 else 0.0
                ax.text(xi,top+ymax*0.045,f"{mn:.1f}±{sd:.1f}\nCV={cv:.1f}%",ha="center",va="bottom",fontsize=7.5)
        ax.set_xticks(x); ax.set_xticklabels(labels,rotation=20,ha="right",fontsize=9)
        ax.set_ylabel(met,fontsize=10); ax.set_title(met,fontsize=11,fontweight="bold")
        ax.grid(True,alpha=0.3,axis="y",linestyle=":"); ax.set_facecolor("#fafafa"); ax.set_ylim(top=ymax*1.32)
    fig.suptitle(f"Multi-Seed Stability (n={nseed} seeds)",fontsize=13,fontweight="bold")
    plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig(path,dpi=400,bbox_inches="tight",facecolor="white"); plt.close()
    print(f"  Seed variance: {path}")

# ═════════════════════════════════════════════════════════════════════════════
# BASELINE MODELS
# ═════════════════════════════════════════════════════════════════════════════
def run_naive(ytr, yte):      return np.full(len(yte), ytr[-1])
def run_snaive(ytr, yte, s=12):
    """Rolling seasonal naive: her adımda geçmiş tüm seriyi kullanır."""
    history = list(ytr)
    preds = []
    for i in range(len(yte)):
        idx = len(history) - s
        preds.append(history[idx] if idx >= 0 else history[0])
        history.append(yte[i])
    return np.array(preds)

def run_arima(ytr, yte):
    from statsmodels.tsa.arima.model import ARIMA
    best_aic, best_ord = np.inf, (1,1,1)
    for p in range(3):
        for d in range(2):
            for q in range(3):
                try:
                    a=ARIMA(ytr,order=(p,d,q)).fit().aic
                    if a<best_aic: best_aic,best_ord=a,(p,d,q)
                except: pass
    print(f"    ARIMA best order (AIC): {best_ord}")
    hist,preds=list(ytr),[]
    for t in range(len(yte)):
        try: fc=ARIMA(hist,order=best_ord).fit().forecast(1)[0]
        except: fc=hist[-1]
        preds.append(fc); hist.append(yte[t])
    return np.array(preds)

def run_sarima(ytr, yte):
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        hist,preds=list(ytr),[]
        for t in range(len(yte)):
            try: fc=SARIMAX(hist,order=(1,1,1),seasonal_order=(1,1,0,12),
                            enforce_stationarity=False,enforce_invertibility=False).fit(disp=False).forecast(1)[0]
            except:
                try:
                    from statsmodels.tsa.arima.model import ARIMA
                    fc=ARIMA(hist,order=(1,1,1)).fit().forecast(1)[0]
                except: fc=hist[-1]
            preds.append(fc); hist.append(yte[t])
        return np.array(preds)
    except: return np.full(len(yte), np.nan)

def run_rf(Xtr, ytr, Xte, rs=42):
    from sklearn.ensemble import RandomForestRegressor
    # Small-sample regularization: max_depth + min_samples_leaf prevent overfitting
    # on limited training data (n=120). See Probst et al. (2019) for RF tuning guidance.
    m=RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=5,
        max_features=0.5,
        random_state=rs, n_jobs=-1
    )
    m.fit(Xtr.reshape(len(Xtr),-1),ytr); return m.predict(Xte.reshape(len(Xte),-1))

def run_xgb(Xtr, ytr, Xte, rs=42):
    try:
        import xgboost as xgb
        # Small-sample regularization: reduced depth + L1/L2 penalties
        # prevent memorization on limited training data (n=120).
        m=xgb.XGBRegressor(
            n_estimators=100, learning_rate=0.05,
            max_depth=2, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=2.0, reg_alpha=0.1,
            random_state=rs, verbosity=0
        )
        m.fit(Xtr.reshape(len(Xtr),-1),ytr); return m.predict(Xte.reshape(len(Xte),-1))
    except: return np.full(len(ytr),np.nan)

# ═════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETER TUNING
# Approach B: tune on val only, final fit on train+val (test never seen)
# ═════════════════════════════════════════════════════════════════════════════
param_grids = {
    "RocketRegressor":  {"num_kernels": [500, 1000, 2000, 5000, 10000]},
    "LSTMFCNRegressor": {"lstm_size": [4, 8],   "batch_size": [8, 16]},
    "InceptionTime":    {"batch_size": [8,16,32,64]},
}

def tune_model(cls, base_kw, grid, Xtr, ytr_sc, yscaler, name, Xva, yva_sc, Xtrva, ytrva_sc):
    section(f"HYPERPARAMETER TUNING: {disp(name)}")
    print(f"  train={len(Xtr)} | val={len(Xva)}  [TEST SET NOT SEEN]")
    yva = yscaler.inverse_transform(yva_sc.reshape(-1,1)).ravel()
    best_model, best_score, best_params, total_t, tune_res = None, np.inf, {}, 0.0, {}
    for pn, vals in grid.items():
        print(f"\n  {pn}: {vals}")
        rmse_l, mae_l = [], []
        for v in vals:
            kw = {**base_kw, pn: v}
            if "callbacks" in kw: kw["callbacks"]=[cb for cb in [_make_es()] if cb]
            try:
                # ── Her HP denemesi aynı başlangıç koşulunda değerlendirilsin ──
                # Böylece HP farkı ölçülür, rastlantısallık değil (reproducible tuning)
                try:
                    import tensorflow as _tf_tune
                    _tf_tune.random.set_seed(RANDOM_SEED)
                except Exception: pass
                np.random.seed(RANDOM_SEED)
                _random.seed(RANDOM_SEED)
                # ────────────────────────────────────────────────────────────────
                print(f"    {pn}={v} ...", end=" ", flush=True)
                m=cls(**kw); t0=time.time()
                h_t = fit_capture(m, Xtr, ytr_sc, val_data=(Xva, yva_sc))
                total_t += time.time()-t0
                ypv=yscaler.inverse_transform(m.predict(Xva).reshape(-1,1)).ravel()
                mae,_,rmse,_,_,_=compute_metrics(yva, ypv)
                rmse_l.append(rmse); mae_l.append(mae)
                ep=len(h_t.get("loss",[])) if h_t else "?"
                print(f"val_RMSE={rmse:.2f}  val_MAE={mae:.2f}  stopped@ep={ep}")
                if rmse < best_score:
                    best_score=rmse; best_params={pn:v}
                    kw2=dict(kw)
                    if "callbacks" in kw2: kw2["callbacks"]=[cb for cb in [_make_es()] if cb]
                    # ── Seed sabitlenir: final model Table III ile tutarlı olsun ──
                    try:
                        import tensorflow as _tf2
                        _tf2.random.set_seed(RANDOM_SEED)
                    except Exception: pass
                    np.random.seed(RANDOM_SEED)
                    _random.seed(RANDOM_SEED)
                    # ─────────────────────────────────────────────────────────────
                    t0=time.time(); bc=cls(**{**kw2,"random_state":RANDOM_SEED})
                    h_f = fit_capture(bc, Xtrva, ytrva_sc, val_data=None)
                    total_t += time.time()-t0
                    hp = h_t if (h_t and h_t.get("val_loss")) else h_f
                    if hp: bc._captured_history=hp
                    best_model=bc
            except Exception as e: print(f"ERROR: {e}"); rmse_l.append(np.nan); mae_l.append(np.nan)
        tune_res[pn]={"values":vals,"rmse":rmse_l,"mae":mae_l}
    print(f"\n  Best: {best_params}  val-RMSE: {best_score:.2f}")
    print(f"  Final model: train+val={len(Xtrva)} samples")
    return best_model, best_params, tune_res, total_t

def sweep_dl(cfg, pn, vals, Xtr, ytr_sc, yscaler, Xva, yva_sc):
    yva=yscaler.inverse_transform(yva_sc.reshape(-1,1)).ravel(); rd,md={},{}
    for nm,c in cfg.items():
        bkw={**c["base_kwargs"],**c.get("best_params",{})}; rl,ml=[],[]
        print(f"\n  {disp(nm)} | {pn}: {vals}")
        for v in vals:
            kw={**bkw,pn:v}
            if "callbacks" in kw: kw["callbacks"]=[cb for cb in [_make_es()] if cb]
            try:
                print(f"    {pn}={v} ...",end=" ",flush=True)
                m=c["class"](**kw); m.fit(Xtr,ytr_sc)
                yp=yscaler.inverse_transform(m.predict(Xva).reshape(-1,1)).ravel()
                mae,_,rmse,_,_,_=compute_metrics(yva,yp); rl.append(rmse); ml.append(mae); print(f"RMSE={rmse:.2f}")
            except Exception as e: print(f"ERROR:{e}"); rl.append(np.nan); ml.append(np.nan)
        rd[nm]=rl; md[nm]=ml
    return rd,md

def _sweep_window_core(m_or_fn, bkw, ws, rawX, rawy, train_r, is_bl=False):
    rl,ml=[],[]
    for w in ws:
        try:
            T=rawX.shape[2]
            Xw=(rawX[:,:,(T-w):] if w<=T else np.concatenate([np.zeros((rawX.shape[0],rawX.shape[1],w-T)),rawX],axis=2))
            ti,vi,_=time_split(Xw.shape[0],train_r,VAL_RATIO)
            ytr,yva=rawy[ti],rawy[vi]
            if is_bl: yp=m_or_fn(Xw[ti],ytr,Xw[vi])
            else:
                Xn,Xvn=norm3d(Xw[ti],Xw[vi]); sc=MinMaxScaler(); ytrs=sc.fit_transform(ytr.reshape(-1,1)).ravel()
                m=m_or_fn(**bkw); m.fit(Xn,ytrs); yp=sc.inverse_transform(m.predict(Xvn).reshape(-1,1)).ravel()
            mae,_,rmse,_,_,_=compute_metrics(yva,yp); rl.append(rmse); ml.append(mae); print(f"    w={w}: RMSE={rmse:.2f}")
        except Exception as e: print(f"    w={w}: ERROR {e}"); rl.append(np.nan); ml.append(np.nan)
    return rl,ml

def sweep_window(cfg, ws, rawX, rawy, train_r):
    rd,md={},{}
    for nm,c in cfg.items():
        bkw={**c["base_kwargs"],**c.get("best_params",{})}; print(f"\n  {disp(nm)} | window: {ws}")
        rd[nm],md[nm]=_sweep_window_core(c["class"],bkw,ws,rawX,rawy,train_r)
    return rd,md

def sweep_window_bl(ws, rawX, rawy, train_r):
    rd,md={},{}
    for nm,fn in [("RandomForest",run_rf),("XGBoost",run_xgb)]:
        print(f"\n  {disp(nm)} | window: {ws}")
        rd[nm],md[nm]=_sweep_window_core(fn,{},ws,rawX,rawy,train_r,is_bl=True)
    return rd,md

def count_params(model):
    for a in ("model_","model","network_","clf_"):
        obj=getattr(model,a,None)
        if obj is None: continue
        for s in ("model_","model","network_"):
            k=getattr(obj,s,None)
            if k and hasattr(k,"count_params"):
                try: return int(k.count_params())
                except: pass
        if hasattr(obj,"count_params"):
            try: return int(obj.count_params())
            except: pass
    return None

def _import_inception():
    for cn in ("InceptionTimeRegressor","InceptionTime"):
        try: return getattr(__import__("sktime.regression.deep_learning",fromlist=[cn]),cn)
        except ImportError: pass
    raise ImportError("InceptionTime not found. pip install sktime[deep-learning]")

# helper: save model results
def _save_model_results(name, ytr, yva, yte, yp_val, yp_test,
                        train_t, inf_t, ts_tr, ts_va, ts_te,
                        results, val_results, all_predictions, all_val_predictions,
                        all_ci, all_ci_val, outdir, is_baseline=False):
    sub = "baseline_results" if is_baseline else "time_series_plots"
    mc  = pj(outdir, "baseline_results" if is_baseline else "model_comparisons")

    ci_lo_v, ci_hi_v = bootstrap_ci(yva, yp_val, N_BOOTSTRAP, CI_LEVEL) if yp_val is not None else (None,None)
    ci_lo, ci_hi     = bootstrap_ci(yte, yp_test, N_BOOTSTRAP, CI_LEVEL)

    all_predictions[name]=yp_test; all_ci[name]=(ci_lo,ci_hi)
    if yp_val is not None:
        all_val_predictions[name]=yp_val; all_ci_val[name]=(ci_lo_v,ci_hi_v)

    mae_v,mse_v,rmse_v,mape_v,mase_v,rmsse_v = compute_metrics(yva,yp_val,ytr) if yp_val is not None else (np.nan,)*6
    mae,mse,rmse,mape,mase,rmsse = compute_metrics(yte,yp_test,np.concatenate([ytr,yva]))

    val_results.append({"model":name,"MAE":mae_v,"MSE":mse_v,"RMSE":rmse_v,"MAPE(%)":mape_v,
                        "MASE":mase_v,"RMSSE":rmsse_v,"Training_Time_s":train_t,"Inference_Time_s":inf_t})
    results.append(    {"model":name,"MAE":mae,  "MSE":mse,  "RMSE":rmse,  "MAPE(%)":mape,
                        "MASE":mase, "RMSSE":rmsse, "Training_Time_s":train_t,"Inference_Time_s":inf_t})

    if yp_val is not None:
        plot_forecast(yva, yp_val, pj(outdir,sub,f"{name}_val_timeseries.png"), name,
                      ts=ts_va, ci_lo=ci_lo_v, ci_hi=ci_hi_v, label="Validation")
        pd.DataFrame({"y_true":yva,"y_pred":yp_val,"residual":yva-yp_val,
                      "ci_lower":ci_lo_v,"ci_upper":ci_hi_v}).to_csv(pj(outdir,f"{name}_val_predictions.csv"),index=False)

    plot_parity(yte, yp_test, pj(mc,f"{name}_parity.png"), name)
    plot_forecast(yte, yp_test, pj(outdir,sub,f"{name}_test_timeseries.png"), name,
                  ts=ts_te, ci_lo=ci_lo, ci_hi=ci_hi, label="Test")
    plot_full_series(ytr, yva, yte, yp_val, yp_test, pj(outdir,sub,f"{name}_full_series.png"), name,
                     ts_tr=ts_tr, ts_va=ts_va, ts_te=ts_te,
                     ci_lo_v=ci_lo_v, ci_hi_v=ci_hi_v, ci_lo=ci_lo, ci_hi=ci_hi)

    rt=residual_tests(yte,yp_test,name)
    pd.DataFrame([rt]).to_csv(pj(outdir,"residual_tests",f"{name}_stat_tests.csv"),index=False)
    pd.DataFrame({"y_true":yte,"y_pred":yp_test,"residual":yte-yp_test,
                  "ci_lower":ci_lo,"ci_upper":ci_hi}).to_csv(pj(outdir,f"{name}_test_predictions.csv"),index=False)

    print(f"\n  {disp(name)} – VALIDATION: RMSE={rmse_v:.2f}  MAE={mae_v:.2f}  MAPE={mape_v:.4f}%  MASE={mase_v:.4f}  RMSSE={rmsse_v:.4f}")
    print(f"  {disp(name)} – TEST:       RMSE={rmse:.2f}  MAE={mae:.2f}  MAPE={mape:.4f}%  MASE={mase:.4f}  RMSSE={rmsse:.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN FLOW
# ═════════════════════════════════════════════════════════════════════════════
section("DATA LOADING")
X    = np.load(pj(PREP_DIR,"X_windows.npy"))
y    = pd.read_csv(pj(PREP_DIR,"y.csv"))["W-Water"].values.astype(float)
meta = json.load(open(pj(PREP_DIR,"meta.json"),encoding="utf-8"))
try: ts = pd.to_datetime(pd.read_csv(pj(PREP_DIR,"timestamps.csv"))["target_month"]).values
except: ts = None

X3d = np.transpose(X,(0,2,1)); n=X3d.shape[0]; assert len(y)==n
print(f"X3d: {X3d.shape}  |  y: {y.shape}  min={y.min():.0f}  max={y.max():.0f}")
print(f"Features: {meta.get('feature_columns','?')}  |  Final window: W={FINAL_WINDOW}")

section("TRAIN / VALIDATION / TEST SPLIT")
tri,vai,tei = split_from_meta(n,meta)
Xtr,Xva,Xte  = X3d[tri],X3d[vai],X3d[tei]
ytr,yva,yte  = y[tri],y[vai],y[tei]
ts_tr = ts[tri] if ts is not None else None
ts_va = ts[vai] if ts is not None else None
ts_te = ts[tei] if ts is not None else None
def dr(k): return f"{meta.get(k,['?','?'])[0]} – {meta.get(k,['?','?'])[1]}"
print(f"Train: {len(tri)} samples ({dr('train_date_range')})")
print(f"Val  : {len(vai)} samples ({dr('val_date_range')})")
print(f"Test : {len(tei)} samples ({dr('test_date_range')})")
print("Leakage-free: scaler fit on train only | HP tuning on val only | final fit on train+val")

section("NORMALIZATION")
Xtr_n, Xte_n, x_scs = norm3d_fit(Xtr,Xte)
Xva_n    = norm3d_apply(Xva,x_scs)
Xtrva    = np.concatenate([Xtr,Xva],axis=0)
Xtrva_n  = np.concatenate([Xtr_n,Xva_n],axis=0)
ysc      = MinMaxScaler()
ytr_sc   = ysc.fit_transform(ytr.reshape(-1,1)).ravel()
yva_sc   = ysc.transform(yva.reshape(-1,1)).ravel()
ytrva    = np.concatenate([ytr,yva])
ytrva_sc = ysc.transform(ytrva.reshape(-1,1)).ravel()

section("MODEL DEFINITIONS")
from sktime.regression.kernel_based import RocketRegressor
from sktime.regression.deep_learning import LSTMFCNRegressor
InceptionTimeRegressor = _import_inception()

models = {
    "RocketRegressor":  {"class":RocketRegressor,
                         "base_kwargs":dict(num_kernels=10000,random_state=RANDOM_SEED)},
    "LSTMFCNRegressor": {"class":LSTMFCNRegressor,
                         "base_kwargs":dict(n_epochs=ES_MAX_EPOCHS,batch_size=8,lstm_size=8,
                                           dropout=0.3,
                                           random_state=RANDOM_SEED,verbose=False,callbacks=[_make_es()])},
    "InceptionTime":    {"class":InceptionTimeRegressor,
                         "base_kwargs":dict(n_epochs=ES_MAX_EPOCHS,batch_size=16,
                                           random_state=RANDOM_SEED,verbose=False,callbacks=[_make_es()])},
}

section("TRAINING & HYPERPARAMETER SELECTION")
results,val_results,all_predictions,all_val_predictions = [],[],{},{}
train_times,infer_times,all_ci,all_ci_val = {},{},{},{}
all_histories,best_params_log,all_tune_res = {},{},{}
param_counts,seed_results = {},{nm:{"MAE":[],"RMSE":[],"MAPE(%)":[]} for nm in models}
tc=pj(OUTPUT_DIR,"training_curves")

for name,cfg in models.items():
    cls,bkw = cfg["class"],cfg["base_kwargs"]
    section(f"Model: {disp(name)}")

    model,best_p,tune_res,tr_t = tune_model(
        cls,bkw,param_grids[name],Xtr_n,ytr_sc,ysc,name,Xva_n,yva_sc,Xtrva_n,ytrva_sc)

    train_times[name]=tr_t; best_params_log[name]=best_p
    all_tune_res[name]=tune_res; models[name]["best_params"]=best_p
    plot_hp_grid(tune_res, name, pj(OUTPUT_DIR,"hyperparameter_plots",f"{name}_tuning.png"))

    h=getattr(model,"_captured_history",None) or _get_history(model)
    if h: all_histories[name]=h; plot_epoch_loss(h,name,tc)

    t0=time.time()
    yp_val_sc=model.predict(Xva_n); inf_v=time.time()-t0
    if np.isnan(yp_val_sc).any(): print("  [WARN] NaN val – skipped"); continue
    yp_val=ysc.inverse_transform(yp_val_sc.reshape(-1,1)).ravel()

    t0=time.time()
    yp_te_sc=model.predict(Xte_n); infer_times[name]=time.time()-t0
    if np.isnan(yp_te_sc).any(): print("  [WARN] NaN test – skipped"); continue
    yp_test=ysc.inverse_transform(yp_te_sc.reshape(-1,1)).ravel()

    pc=count_params(model)
    if pc: param_counts[name]=pc; print(f"  Trainable params: {pc:,}")

    _save_model_results(name,ytr,yva,yte,yp_val,yp_test,tr_t,infer_times[name],
                        ts_tr,ts_va,ts_te,results,val_results,all_predictions,
                        all_val_predictions,all_ci,all_ci_val,OUTPUT_DIR)

    if name in {"LSTMFCNRegressor","InceptionTime"} and N_SEEDS>1:
        bkw2={**bkw,**best_p}
        # RANDOM_SEED zaten Table III final modeli ile aynı seed.
        # SEEDS listesinde yoksa başa ekliyoruz ki tablolar tutarlı olsun.
        seeds_to_run = SEEDS if RANDOM_SEED in SEEDS else [RANDOM_SEED] + SEEDS
        print(f"\n  Multi-seed (best_params={best_p}, seeds={seeds_to_run})...")
        for seed in seeds_to_run:
            try:
                try: import tensorflow as tf2; tf2.random.set_seed(seed)
                except: pass
                np.random.seed(seed); _random.seed(seed)
                bkw2_copy = dict(bkw2)
                if "callbacks" in bkw2_copy:
                    bkw2_copy["callbacks"]=[cb for cb in [_make_es()] if cb]
                ms=cls(**{**bkw2_copy,"random_state":seed}); ms.fit(Xtrva_n,ytrva_sc)
                yps=ysc.inverse_transform(ms.predict(Xte_n).reshape(-1,1)).ravel()
                m_,_,r_,mp_,_,_=compute_metrics(yte,yps,ytrva)
                seed_results[name]["MAE"].append(m_); seed_results[name]["RMSE"].append(r_)
                seed_results[name]["MAPE(%)"].append(mp_)
                is_table3 = (seed == RANDOM_SEED)
                seed_results[name].setdefault("_rows",[]).append(
                    {"model":disp(name),"seed":seed,"RMSE":r_,"MAE":m_,"MAPE(%)":mp_,
                     "is_table3_seed": is_table3})
                marker = " <- Table III seed" if is_table3 else ""
                print(f"    seed={seed}: RMSE={r_:.2f}  MAE={m_:.2f}{marker}")
            except Exception as e: print(f"    seed={seed}: {e}")
    print(f"\n  {disp(name)} done.")

if all_histories: plot_training_curves_combined(all_histories,tc)

section("BASELINE MODELS")
bl_fns = {
    "Naive":        (lambda: run_naive(ytrva,yte),       lambda: run_naive(ytr,yva)),
    "SeasonalNaive":(lambda: run_snaive(ytrva,yte),      lambda: run_snaive(ytr,yva)),
    "ARIMA":        (lambda: run_arima(ytrva,yte),        lambda: run_arima(ytr,yva)),
    "SARIMA":       (lambda: run_sarima(ytrva,yte),       lambda: run_sarima(ytr,yva)),
    "RandomForest": (lambda: run_rf(Xtrva_n,ytrva,Xte_n),lambda: run_rf(Xtr_n,ytr,Xva_n)),
    "XGBoost":      (lambda: run_xgb(Xtrva_n,ytrva,Xte_n),lambda: run_xgb(Xtr_n,ytr,Xva_n)),
}
for bname,(test_fn,val_fn) in bl_fns.items():
    print(f"\n  {disp(bname)} ...")
    try:
        yp_v=val_fn()
        if np.isnan(yp_v).any(): yp_v=None
        t0=time.time(); yp_t=test_fn(); bt=time.time()-t0
        if np.isnan(yp_t).any(): print("  NaN – skipped"); continue
        train_times[bname]=bt; infer_times[bname]=0.0
        _save_model_results(bname,ytr,yva,yte,yp_v,yp_t,bt,0.0,
                            ts_tr,ts_va,ts_te,results,val_results,all_predictions,
                            all_val_predictions,all_ci,all_ci_val,OUTPUT_DIR,is_baseline=True)
        print(f"  {disp(bname)} done.")
    except Exception as e: print(f"  ERROR: {e}")

section("SWEEP PLOTS  [validation only]")
DL_CFG={k:v for k,v in models.items() if k in {"LSTMFCNRegressor","InceptionTime"}}
es_log={mn:len(h.get("loss",[])) for mn,h in all_histories.items()}
for mn,ep in es_log.items(): print(f"  Early stopping – {disp(mn)}: {ep} epochs")
if es_log: pd.DataFrame([{"model":disp(k),"epochs_trained":v} for k,v in es_log.items()]).to_csv(pj(tc,"early_stopping_log.csv"),index=False)

print("\n  Batch size sweep (DL)...")
bsr,bsm=sweep_dl(DL_CFG,"batch_size",BATCH_SIZE_CANDIDATES,Xtr_n,ytr_sc,ysc,Xva_n,yva_sc)
plot_sweep(bsr,bsm,BATCH_SIZE_CANDIDATES,"Batch Size","Batch Size Sensitivity – DL Models (Val)",pj(tc,"all_models_batch_rmse_mae.png"))

print("\n  Window size sweep (DL)...")
wrd,wmd=sweep_window(models,WINDOW_SIZE_CANDIDATES,X3d,y,TRAIN_RATIO)
print("\n  Window size sweep (RF, XGBoost)...")
wrb,wmb=sweep_window_bl(WINDOW_SIZE_CANDIDATES,X3d,y,TRAIN_RATIO)
plot_sweep({**wrd,**wrb},{**wmd,**wmb},WINDOW_SIZE_CANDIDATES,
           "Window Size (months)","Window Size Sensitivity – All Models (Val)",
           pj(tc,"all_models_window_rmse_mae.png"))

section("COMBINED PLOTS")
cp=pj(OUTPUT_DIR,"combined_plots")
if all_predictions:
    plot_all_single(yte,all_predictions,pj(cp,"all_models_test_single_plot.png"),ts=ts_te)
    plot_all_subplots(yte,all_predictions,pj(cp,"all_models_test_subplots_with_ci.png"),ts=ts_te,ci_dict=all_ci)
    plot_error_boxplot(all_predictions,yte,pj(cp,"all_models_error_boxplot.png"))
    plot_parity_grid(yte,all_predictions,pj(cp,"combined_parity_all_models.png"))

if param_counts:
    pd.DataFrame([{"model":disp(k),"trainable_parameters":v} for k,v in param_counts.items()]).to_csv(pj(OUTPUT_DIR,"model_parameter_counts.csv"),index=False)

dl_seed={k:v for k,v in seed_results.items() if any(len(l)>0 for l in v.values() if isinstance(l,list) and all(isinstance(x,(int,float)) for x in l))}
if dl_seed: plot_seed_var(dl_seed,pj(cp,"multi_seed_variance.png"))
if train_times: plot_comp_time(train_times,pj(cp,"computation_time_training.png"),inf_t=infer_times)

section("SAVE RESULTS")
res_df=pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
val_df=pd.DataFrame(val_results).sort_values("RMSE").reset_index(drop=True)
res_df.to_csv(pj(OUTPUT_DIR,"results_test.csv"),index=False); print("results_test.csv saved")
val_df.to_csv(pj(OUTPUT_DIR,"results_validation.csv"),index=False); print("results_validation.csv saved")

plot_metrics_bar(res_df,pj(cp,"all_models_test_metrics_comparison.png"))
plot_metrics_bar(val_df,pj(cp,"all_models_val_metrics_comparison.png"))

rt_files=[f for f in os.listdir(pj(OUTPUT_DIR,"residual_tests")) if f.endswith("_stat_tests.csv")]
if rt_files:
    pd.concat([pd.read_csv(pj(OUTPUT_DIR,"residual_tests",f)) for f in rt_files],ignore_index=True
              ).to_csv(pj(OUTPUT_DIR,"all_residual_tests.csv"),index=False)

# computation_times.csv (Model | Type | Train time | Infer. time)
def fmt_t(t): return "<0.001" if (not np.isnan(t) and t<0.001) else (f"{t:.3f}" if not np.isnan(t) else "")
pd.DataFrame([{"model":disp(m),"type":MTYPE.get(m,"—"),
               "training_time_s":fmt_t(train_times.get(m,np.nan)),
               "inference_time_s":fmt_t(infer_times.get(m,np.nan))}
              for m in sorted(set(list(train_times)+list(infer_times)))
              ]).to_csv(pj(OUTPUT_DIR,"computation_times.csv"),index=False); print("computation_times.csv saved")

# multi_seed_results.csv
seed_rows=[]
for nm,data in seed_results.items():
    rows=data.get("_rows",[])
    if not rows: continue
    mean_r=float(np.mean([r["RMSE"] for r in rows]))
    for r in rows: seed_rows.append({**r,"mean_RMSE":round(mean_r,2)})
if seed_rows: pd.DataFrame(seed_rows).to_csv(pj(OUTPUT_DIR,"multi_seed_results.csv"),index=False); print("multi_seed_results.csv saved")

json.dump(best_params_log,open(pj(OUTPUT_DIR,"best_hyperparameters.json"),"w"),indent=2)
json.dump({mn:{pn:{"values":[float(v) for v in d["values"]],
                   "rmse":[float(v) if not np.isnan(v) else None for v in d["rmse"]],
                   "mae": [float(v) if not np.isnan(v) else None for v in d["mae"]]}
               for pn,d in pd_.items()} for mn,pd_ in all_tune_res.items()},
          open(pj(OUTPUT_DIR,"hyperparameter_tuning_results.json"),"w"),indent=2)

def _fmv(v): return "N/A" if (isinstance(v,float) and np.isnan(v)) else f"{v:.3f}"
section("SUMMARY")
print(f"Final window: W={FINAL_WINDOW}  |  Seed: {RANDOM_SEED}  |  Output: {OUTPUT_DIR}/\n")
print("="*90)
print("VALIDATION PERFORMANCE (sorted by RMSE):")
print("="*90)
for i,(_,row) in enumerate(val_df.iterrows(),1):
    print(f"{i:2}. {disp(row['model']):<18} RMSE={row['RMSE']:.2f}  MAE={row['MAE']:.2f}  "
          f"MAPE={row['MAPE(%)']:.4f}%  MASE={_fmv(row['MASE'])}  RMSSE={_fmv(row['RMSSE'])}")
print("\n"+"="*90)
print("TEST PERFORMANCE (sorted by RMSE) — reported AFTER final model selection:")
print("="*90)
for i,(_,row) in enumerate(res_df.iterrows(),1):
    print(f"{i:2}. {disp(row['model']):<18} RMSE={row['RMSE']:.2f}  MAE={row['MAE']:.2f}  "
          f"MAPE={row['MAPE(%)']:.4f}%  MASE={_fmv(row['MASE'])}  RMSSE={_fmv(row['RMSSE'])}  "
          f"TrainT={row['Training_Time_s']:.2f}s")
