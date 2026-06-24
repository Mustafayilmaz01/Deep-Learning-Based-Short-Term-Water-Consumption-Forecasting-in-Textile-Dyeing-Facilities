#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plots.py — W-Water Forecasting: Tüm görselleştirme fonksiyonları
run.py tarafından import edilir.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Stil ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 10,
    "axes.labelsize": 10, "axes.titlesize": 11, "xtick.labelsize": 9,
    "ytick.labelsize": 9, "legend.fontsize": 9, "lines.linewidth": 1.5,
    "axes.linewidth": 0.8, "grid.linewidth": 0.5, "text.usetex": False,
})

COLORS = {
    "rocketregressor": "#1f77b4", "lstmfcnregressor": "#ff7f0e",
    "inceptiontime": "#d62728",   "naive": "#7f7f7f",
    "seasonalnaive": "#bcbd22",   "arima": "#17becf",
    "sarima": "#9467bd",          "randomforest": "#8c564b",
    "xgboost": "#e377c2",         "actual": "#000000",
}
LS = {
    "rocketregressor": "-",  "lstmfcnregressor": "--", "inceptiontime": "-.",
    "naive": ":",            "seasonalnaive": "--",     "arima": "-.",
    "sarima": ":",           "randomforest": "--",      "xgboost": "-.", "actual": "-",
}
MK = {
    "rocketregressor": "o", "lstmfcnregressor": "s", "inceptiontime": "^",
    "naive": "x",           "seasonalnaive": "+",    "arima": "D",
    "sarima": "v",          "randomforest": "P",     "xgboost": "*",
}
DISPLAY = {
    "RocketRegressor": "ROCKET",        "LSTMFCNRegressor": "LSTM-FCN",
    "InceptionTime":   "InceptionTime", "Naive": "Naive",
    "SeasonalNaive":   "Seasonal Naive","ARIMA": "ARIMA",
    "SARIMA":          "SARIMA",        "RandomForest": "Random Forest",
    "XGBoost":         "XGBoost",
}

# ── Yardımcılar ───────────────────────────────────────────────────────────────
def pj(*a):  return os.path.join(*a)
def disp(n): return DISPLAY.get(n, n)

def fmt_ts(ts):
    try:    return pd.Timestamp(ts).strftime("%Y-%m")
    except: return str(ts)[:7]

def save_fig(p):
    plt.tight_layout()
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()

def mstyle(n):
    k = n.lower()
    return COLORS.get(k, "#1f77b4"), LS.get(k, "--"), MK.get(k, "o")

def _xticks(ax, ts, n):
    if ts is not None and len(ts) == n:
        step = max(1, n // 6); idx = list(range(0, n, step))
        ax.set_xticks(idx)
        ax.set_xticklabels([fmt_ts(ts[i]) for i in idx], rotation=30, ha="right", fontsize=8)

# ── Eğitim Eğrileri ───────────────────────────────────────────────────────────
def plot_epoch_loss(hist, name, save_dir):
    if not hist or not hist.get("loss"): return
    loss  = [float(v) for v in hist["loss"]]
    vloss = [float(v) for v in hist.get("val_loss", [])]
    ep    = list(range(1, len(loss) + 1))
    has_v = bool(vloss) and len(vloss) == len(loss)
    fig, ax = plt.subplots(figsize=(8, 4), facecolor="white")
    ax.set_facecolor("#fafafa")
    ax.plot(ep, loss, color="#2166ac", linewidth=1.8, label="Train Loss")
    if has_v:
        bi = int(np.argmin(vloss)) + 1
        ax.plot(ep, vloss, color="#d6604d", linewidth=1.8, linestyle="--", label="Validation Loss")
        ax.axvline(bi, color="#2ca02c", linestyle=":", linewidth=1.4, label=f"Best Epoch={bi}")
        ax.scatter([bi], [vloss[bi-1]], color="#2ca02c", zorder=5, s=60)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (MSE, normalized)")
    ax.set_title(f"Training & Validation Loss – {disp(name)}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    p = pj(save_dir, f"{name}_epoch_loss_curve.png")
    plt.tight_layout()
    plt.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Loss curve: {p}")

def plot_training_curves_combined(histories, save_dir):
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="white")
    ax.set_facecolor("#fafafa"); plotted = False
    for name, hist in histories.items():
        if not hist or not hist.get("loss"): continue
        plotted = True
        col, ls, _ = mstyle(name)
        ep = list(range(1, len(hist["loss"]) + 1))
        ax.plot(ep, [float(v) for v in hist["loss"]], linewidth=1.8, color=col, linestyle=ls,
                label=f"{disp(name)} – Train")
        vl = [float(v) for v in hist.get("val_loss", [])]
        if vl and len(vl) == len(ep):
            ax.plot(ep, vl, linewidth=1.5, color=col, linestyle=":", alpha=0.85,
                    label=f"{disp(name)} – Val")
            ax.axvline(int(np.argmin(vl)) + 1, color=col, linestyle="--", linewidth=0.8, alpha=0.5)
    if not plotted: plt.close(); return
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (MSE, normalized)")
    ax.set_title("Training & Validation Loss – All DL Models")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    p = pj(save_dir, "all_dl_models_training_curves.png")
    plt.tight_layout()
    plt.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Combined loss curves: {p}")

# ── Zaman Serisi Grafikleri ───────────────────────────────────────────────────
def plot_full_series(y_tr, y_va, y_te, y_pred_val, y_pred_test, path, name,
                     ts_tr=None, ts_va=None, ts_te=None,
                     ci_lo_v=None, ci_hi_v=None, ci_lo=None, ci_hi=None):
    nm = disp(name); col, ls, _ = mstyle(name)
    ntr, nva, nte = len(y_tr), len(y_va), len(y_te); ntot = ntr + nva + nte
    xtr = np.arange(0, ntr); xva = np.arange(ntr, ntr+nva); xte = np.arange(ntr+nva, ntot)
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.axvspan(xtr[0],   xtr[-1]+0.5,  facecolor="#daeef3", alpha=0.35, label="Train region")
    ax.axvspan(xva[0]-0.5, xva[-1]+0.5, facecolor="#fff3cd", alpha=0.45, label="Validation region")
    ax.axvspan(xte[0]-0.5, xte[-1]+0.5, facecolor="#fce4ec", alpha=0.35, label="Test region")
    ax.plot(range(ntot), np.concatenate([y_tr, y_va, y_te]),
            color=COLORS["actual"], linewidth=1.6, label="Actual", zorder=5)
    if y_pred_val is not None and len(y_pred_val) == nva:
        ax.plot(xva, y_pred_val, color="#e67e00", linewidth=1.8, linestyle="--",
                label=f"{nm} Val Prediction", zorder=6)
        if ci_lo_v is not None:
            ax.fill_between(xva, ci_lo_v, ci_hi_v, alpha=0.12, color="#e67e00")
    ax.plot(xte, y_pred_test, color=col, linewidth=1.8, linestyle=ls,
            label=f"{nm} Test Prediction", zorder=7)
    if ci_lo is not None:
        ax.fill_between(xte, ci_lo, ci_hi, alpha=0.15, color=col, label="95% CI Test")
    if ts_tr is not None:
        try:
            all_ts = np.concatenate([ts_tr, ts_va, ts_te])
            step = max(1, ntot // 10); idx = list(range(0, ntot, step))
            ax.set_xticks(idx)
            ax.set_xticklabels([fmt_ts(all_ts[i]) for i in idx], rotation=30, ha="right", fontsize=8)
        except Exception: pass
    ax.axvline(xtr[-1]+0.5, color="#555", linewidth=0.9, linestyle=":")
    ax.axvline(xva[-1]+0.5, color="#555", linewidth=0.9, linestyle=":")
    ytop = ax.get_ylim()[1]
    for x, lbl, c in [(xtr[len(xtr)//2], "Train",      "#1a5276"),
                      (xva[len(xva)//2], "Validation",  "#7d6608"),
                      (xte[len(xte)//2], "Test",        "#922b21")]:
        ax.text(x, ytop, lbl, ha="center", va="top", fontsize=8, color=c, fontweight="bold")
    ax.set_xlabel("Month"); ax.set_ylabel("W-Water (m³)")
    ax.set_title(f"W-Water – Full Series – {nm}", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=7.5, ncol=3, framealpha=0.9)
    ax.grid(True, alpha=0.25); save_fig(path)

def plot_forecast(y_true, y_pred, path, name, ts=None, ci_lo=None, ci_hi=None, label="Test"):
    nm = disp(name); n = len(y_true); col, ls, _ = mstyle(name)
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(np.arange(n), y_true, label="Actual", linewidth=1.5, color=COLORS["actual"])
    ax.plot(np.arange(n), y_pred, label=f"{nm} Prediction", linewidth=1.5, linestyle=ls, color=col)
    if ci_lo is not None:
        ax.fill_between(np.arange(n), ci_lo, ci_hi, alpha=0.15, color=col, label="95% CI")
    _xticks(ax, ts, n)
    ax.set_xlabel("Month"); ax.set_ylabel("W-Water (m³)")
    ax.set_title(f"W-Water – {label} Forecast – {nm}")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); save_fig(path)

# ── Parity ────────────────────────────────────────────────────────────────────
def plot_parity(y_true, y_pred, path, name, precomputed_metrics=None):
    nm = disp(name); col, _, _ = mstyle(name)
    mae  = precomputed_metrics["MAE"]  if precomputed_metrics else float(np.mean(np.abs(y_true - y_pred)))
    rmse = precomputed_metrics["RMSE"] if precomputed_metrics else float(np.sqrt(np.mean((y_true - y_pred)**2)))
    pad  = (max(y_true.max(), y_pred.max()) - min(y_true.min(), y_pred.min())) * 0.05
    lims = [min(y_true.min(), y_pred.min()) - pad, max(y_true.max(), y_pred.max()) + pad]
    fig, ax = plt.subplots(figsize=(5, 5)); ax.set_facecolor("#fafafa")
    ax.scatter(y_true, y_pred, s=35, alpha=0.75, edgecolors="white", linewidths=0.4, color=col, zorder=3)
    ax.plot(lims, lims, "k--", linewidth=1.2, alpha=0.7, zorder=2)
    ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal", adjustable="box")
    ax.text(0.05, 0.95, f"MAE={mae:.2f}\nRMSE={rmse:.2f}", transform=ax.transAxes, fontsize=10,
            va="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.set_xlabel("Actual W-Water (m³)"); ax.set_ylabel("Predicted W-Water (m³)")
    ax.set_title(f"Parity – {nm}"); ax.grid(True, alpha=0.3); save_fig(path)

def plot_parity_grid(y_true, preds, path, ncols=3, metrics_dict=None):
    nms = list(preds.keys()); n = len(nms)
    if n == 0: return
    nc = min(ncols, n); nr = int(np.ceil(n / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(3.5*nc, 3.5*nr), facecolor="white",
                             constrained_layout=True, squeeze=False)
    fig.suptitle("Parity Plots – All Models", fontsize=13, fontweight="bold")
    all_v = [y_true] + list(preds.values())
    gmin = min(v.min() for v in all_v); gmax = max(v.max() for v in all_v)
    pad = (gmax - gmin) * 0.05; lims = [gmin - pad, gmax + pad]
    for i, nm in enumerate(nms):
        r, c = divmod(i, nc); ax = axes[r][c]; ax.set_facecolor("#fafafa")
        yp = np.asarray(preds[nm]); col, _, _ = mstyle(nm)
        ax.scatter(y_true, yp, s=20, alpha=0.65, color=col, edgecolors="white", linewidths=0.3, zorder=3)
        ax.plot(lims, lims, "k--", linewidth=1, alpha=0.7, zorder=2)
        ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal", adjustable="box")
        if metrics_dict and nm in metrics_dict:
            mae = metrics_dict[nm]["MAE"]; rmse = metrics_dict[nm]["RMSE"]
        else:
            mae  = float(np.mean(np.abs(y_true - yp)))
            rmse = float(np.sqrt(np.mean((y_true - yp)**2)))
        ax.text(0.04, 0.96, f"RMSE={rmse:.1f}\nMAE={mae:.1f}", transform=ax.transAxes,
                fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#cccccc", linewidth=0.7, alpha=0.9))
        ax.set_title(disp(nm), fontsize=10.5, fontweight="bold")
        ax.grid(True, alpha=0.25); ax.tick_params(labelsize=7)
        if c == 0: ax.set_ylabel("Predicted (m³)", fontsize=9)
        if r == nr-1: ax.set_xlabel("Actual (m³)", fontsize=9)
    for j in range(n, nr*nc):
        r, c = divmod(j, nc); axes[r][c].axis("off")
    plt.savefig(path, dpi=400, bbox_inches="tight", facecolor="white"); plt.close()

# ── Karşılaştırma Grafikleri ──────────────────────────────────────────────────
def plot_all_single(y_true, preds, path, ts=None):
    n = len(y_true); fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(np.arange(n), y_true, label="Actual", linewidth=2, color=COLORS["actual"], zorder=10)
    for nm, yp in preds.items():
        col, ls, _ = mstyle(nm)
        ax.plot(np.arange(n), yp, label=disp(nm), linewidth=1.5, linestyle=ls, color=col, alpha=0.7)
    _xticks(ax, ts, n)
    ax.set_xlabel("Month"); ax.set_ylabel("W-Water (m³)")
    ax.set_title("W-Water Test Predictions – All Models")
    ax.legend(fontsize=7, ncol=3); ax.grid(True, alpha=0.3); save_fig(path)

def plot_all_subplots(y_true, preds, path, ts=None, ci_dict=None, metrics_dict=None):
    nm_list = list(preds.keys()); n = len(y_true)
    fig, axes = plt.subplots(len(nm_list), 1, figsize=(11, 2.5 * len(nm_list)))
    if len(nm_list) == 1: axes = [axes]
    for i, (nm, yp) in enumerate(preds.items()):
        ax = axes[i]; col, ls, _ = mstyle(nm)
        ax.plot(np.arange(n), y_true, label="Actual", linewidth=1.5, color=COLORS["actual"])
        ax.plot(np.arange(n), yp, label=disp(nm), linewidth=1.5, linestyle=ls, color=col)
        if ci_dict and nm in ci_dict:
            lo, hi = ci_dict[nm]
            ax.fill_between(np.arange(n), lo, hi, alpha=0.15, color=col, label="95% CI")
        if metrics_dict and nm in metrics_dict:
            mae = metrics_dict[nm]["MAE"]; mape = metrics_dict[nm]["MAPE(%)"]
        else:
            mae  = float(np.mean(np.abs(y_true - yp)))
            mape = float(np.mean(np.abs((y_true - yp) / (np.abs(y_true) + 1e-8))) * 100)
        ax.text(0.02, 0.98, f"MAE={mae:.1f}, MAPE={mape:.2f}%", transform=ax.transAxes,
                fontsize=8, va="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        ax.set_ylabel("W-Water (m³)"); ax.set_title(disp(nm))
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        if ts is not None and len(ts) == n:
            step = max(1, n // 5); idx2 = list(range(0, n, step))
            ax.set_xticks(idx2)
            ax.set_xticklabels([fmt_ts(ts[j]) for j in idx2], rotation=30, ha="right", fontsize=7)
        if i == len(nm_list) - 1: ax.set_xlabel("Month")
    save_fig(path)

def plot_error_boxplot(preds, y_true, path):
    labels = [disp(m) for m in preds]
    errors = [np.abs(y_true - yp) for yp in preds.values()]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 4))
    bp = ax.boxplot(errors, labels=labels, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, nm in zip(bp["boxes"], preds):
        patch.set_facecolor(COLORS.get(nm.lower(), "#1f77b4")); patch.set_alpha(0.6)
    ax.set_ylabel("Absolute Error (m³)"); ax.set_title("Absolute Error Distribution – All Models")
    ax.grid(True, alpha=0.3, axis="y"); plt.xticks(rotation=30, ha="right")
    plt.tight_layout(); plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()

def plot_metrics_bar(df, path):
    metrics = [m for m in ["MAE", "RMSE", "MAPE(%)", "MASE", "RMSSE"] if m in df.columns]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4))
    if len(metrics) == 1: axes = [axes]
    for ax, met in zip(axes, metrics):
        labels = [disp(m) for m in df["model"].values]
        vals   = df[met].values
        cols   = [COLORS.get(m.lower(), "#1f77b4") for m in df["model"].values]
        bars   = ax.bar(range(len(labels)), vals, color=cols, alpha=0.7,
                        edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(met); ax.set_title(met); ax.grid(True, alpha=0.3, axis="y")
    save_fig(path)

# ── Hiperparametre & Sweep Grafikleri ─────────────────────────────────────────
def plot_hp_grid(tuning_res, name, path):
    PLABELS = {"n_epochs": "Number of Epochs", "batch_size": "Batch Size",
               "lstm_size": "LSTM Units",      "num_kernels": "Num Kernels",
               "n_filters": "Num Filters",     "depth": "Network Depth"}
    np_ = len(tuning_res)
    if np_ == 0: return
    nc = min(2, np_); nr = int(np.ceil(np_ / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(6.8*nc, 4.3*nr), facecolor="white",
                             constrained_layout=True, squeeze=False)
    fig.suptitle(f"Hyperparameter Sensitivity – {disp(name)}", fontsize=13, fontweight="bold")
    CR, CM = "#2166ac", "#d6604d"
    for i, (pn, data) in enumerate(tuning_res.items()):
        ax1 = axes.flatten()[i]; ax1.set_facecolor("#fafafa")
        vals = list(data["values"]); rmses = data["rmse"]; maes = data["mae"]
        xl = PLABELS.get(pn, pn); xp = np.arange(len(vals)); ax2 = ax1.twinx()
        l1, = ax1.plot(xp, rmses, marker="o", linewidth=1.8, markersize=7, color=CR,
                       markerfacecolor="white", markeredgewidth=1.8, label="RMSE (Val)", zorder=4)
        l2, = ax2.plot(xp, maes,  marker="s", linewidth=1.8, markersize=7, color=CM, linestyle="--",
                       markerfacecolor="white", markeredgewidth=1.8, label="MAE (Val)", zorder=4)
        ax1.set_xticks(xp); ax1.set_xticklabels([str(v) for v in vals], fontsize=8)
        ax1.set_xlabel(xl, fontsize=10)
        ax1.set_ylabel("Val RMSE (m³)", color=CR, fontsize=10)
        ax2.set_ylabel("Val MAE (m³)",  color=CM, fontsize=10)
        ax1.tick_params(axis="y", labelcolor=CR, labelsize=8)
        ax2.tick_params(axis="y", labelcolor=CM, labelsize=8)
        ax1.grid(True, alpha=0.25, linestyle=":")
        ax1.set_title(f"Effect of {xl}", fontsize=10, fontweight="bold", pad=10)
        if [v for v in rmses if not np.isnan(v)]:
            bi = int(np.nanargmin(rmses))
            ax1.axvline(xp[bi], color="#4dac26", linestyle=":", linewidth=1.3, zorder=1)
            ax1.plot(xp[bi], rmses[bi], marker="*", markersize=15, color="#4dac26", zorder=10)
            ax1.annotate(f"Optimal={vals[bi]}", xy=(xp[bi], rmses[bi]), xytext=(0.94, 0.94),
                         textcoords="axes fraction", ha="right", va="top", fontsize=8,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffcc",
                                   edgecolor="#4dac26", alpha=0.95),
                         arrowprops=dict(arrowstyle="->", color="#4dac26", lw=1.0, alpha=0.85))
        ax1.legend([l1, l2], ["RMSE (Val)", "MAE (Val)"], loc="upper left", fontsize=8.5, framealpha=0.9)
    for j in range(np_, len(axes.flatten())): axes.flatten()[j].axis("off")
    plt.savefig(path, dpi=400, bbox_inches="tight", facecolor="white"); plt.close()

def plot_sweep(rmse_d, mae_d, param_vals, xlabel, title, path, use_log=False):
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), facecolor="white", sharex=True)
    for ax, sweep, ylabel in zip(axes, [rmse_d, mae_d], ["RMSE", "MAE"]):
        ax.set_facecolor("#fafafa")
        for nm, vals in sweep.items():
            col, ls, mk = mstyle(nm)
            ax.plot(param_vals, vals, marker=mk, markersize=7, linewidth=1.8, color=col,
                    linestyle=ls, markerfacecolor="white", markeredgewidth=1.8,
                    label=disp(nm), zorder=4)
        if use_log: ax.set_yscale("log"); ax.set_ylabel(f"{ylabel} (log scale)", fontsize=10)
        else: ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.25, linestyle=":")
    axes[-1].set_xlabel(xlabel, fontsize=10)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white"); plt.close()

# ── Seed & Zaman ──────────────────────────────────────────────────────────────
def plot_seed_var(seed_res, path):
    mets = ["MAE", "RMSE", "MAPE(%)"]; ml = list(seed_res.keys())
    labels = [disp(m) for m in ml]
    nseed = max((len([v for v in seed_res[m].get("RMSE", []) if not np.isnan(v)])
                 for m in ml), default=0)
    fig, axes = plt.subplots(1, len(mets), figsize=(5.4*len(mets), 4.6), facecolor="white")
    if len(mets) == 1: axes = [axes]
    x = np.arange(len(ml)); rng = np.random.default_rng(0)
    for ax, met in zip(axes, mets):
        means, stds, raw = [], [], []
        for m in ml:
            v = [val for val in seed_res[m].get(met, []) if not np.isnan(val)]
            raw.append(v)
            means.append(np.mean(v) if v else np.nan)
            stds.append(np.std(v)  if v else np.nan)
        cols = [COLORS.get(m.lower(), "#1f77b4") for m in ml]
        ax.bar(x, means, yerr=stds, capsize=5, color=cols, alpha=0.55, edgecolor="black",
               linewidth=0.8, width=0.6, error_kw=dict(elinewidth=1.4, ecolor="#333333"), zorder=2)
        for xi, v, c in zip(x, raw, cols):
            if v:
                ax.scatter(xi + rng.uniform(-0.13, 0.13, size=len(v)), v, s=36,
                           color=c, edgecolors="black", linewidths=0.6, zorder=4)
        tops = [max(m+s, max(v) if v else m+s) if not np.isnan(m) else np.nan
                for m, s, v in zip(means, stds, raw)]
        ymax = np.nanmax(tops) if any(not np.isnan(t) for t in tops) else 1.0
        for xi, mn, sd, top in zip(x, means, stds, tops):
            if not np.isnan(mn):
                cv = (sd/mn*100) if abs(mn) > 1e-9 else 0.0
                ax.text(xi, top + ymax*0.045, f"{mn:.1f}±{sd:.1f}\nCV={cv:.1f}%",
                        ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(met, fontsize=10); ax.set_title(met, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y", linestyle=":"); ax.set_facecolor("#fafafa")
        ax.set_ylim(top=ymax * 1.32)
    fig.suptitle(f"Multi-Seed Stability (n={nseed} seeds)", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path, dpi=400, bbox_inches="tight", facecolor="white"); plt.close()

def plot_comp_time(train_t, path, inf_t=None):
    models_ = list(train_t.keys()); labels = [disp(m) for m in models_]
    tr = [train_t.get(m, 0) for m in models_]
    has_inf = inf_t is not None and any(inf_t.get(m, 0) > 0 for m in models_)
    x = np.arange(len(models_)); w = 0.38 if has_inf else 0.55
    fig, ax = plt.subplots(figsize=(max(9, len(models_)*1.3), 4), facecolor="white")
    bars = ax.bar(x - (w/2 if has_inf else 0), tr, w, label="Train Time",
                  color="#2166ac", alpha=0.75, edgecolor="black", linewidth=0.5)
    for bar, t in zip(bars, tr):
        if t > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(tr)*0.01,
                    f"{t:.1f}s", ha="center", va="bottom", fontsize=7.5)
    if has_inf:
        iv = [inf_t.get(m, 0) for m in models_]
        bi = ax.bar(x + w/2, iv, w, label="Infer. Time",
                    color="#d6604d", alpha=0.75, edgecolor="black", linewidth=0.5)
        for bar, t in zip(bi, iv):
            if t > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(tr)*0.01,
                        f"{t:.3f}s", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Time (s)"); ax.set_title("Computational Cost"); ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white"); plt.close()
