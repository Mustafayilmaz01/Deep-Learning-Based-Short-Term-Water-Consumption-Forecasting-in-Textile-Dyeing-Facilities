#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run.py — W-Water Monthly Wastewater Forecasting: Ana akış
Tüm grafik fonksiyonları plots.py'den import edilir.
"""

import os, json, logging, warnings, time, random as _random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.diagnostic import acorr_ljungbox

from plots import (
    pj, disp, mstyle,
    plot_epoch_loss, plot_training_curves_combined,
    plot_full_series, plot_forecast, plot_parity, plot_parity_grid,
    plot_all_single, plot_all_subplots, plot_error_boxplot,
    plot_hp_grid, plot_sweep, plot_metrics_bar,
    plot_seed_var, plot_comp_time,
)

# ── Env / Logging ─────────────────────────────────────────────────────────────
os.environ.update({
    "TF_ENABLE_ONEDNN_OPTS": "0", "TF_DETERMINISTIC_OPS": "1",
    "TF_CUDNN_DETERMINISTIC": "1", "TF_CPP_MIN_LOG_LEVEL": "3",
    "GRPC_VERBOSITY": "ERROR",     "AUTOGRAPH_VERBOSITY": "0",
})
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# ── Seeds ─────────────────────────────────────────────────────────────────────
RANDOM_SEED = 1
SEEDS = list(range(1, 11))

def set_all_seeds(seed):
    np.random.seed(seed); _random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf; tf.random.set_seed(seed)
    except Exception: pass

set_all_seeds(RANDOM_SEED)
try:
    import tensorflow as _tf; _tf.get_logger().setLevel("ERROR")
except Exception: pass

# ── Config ────────────────────────────────────────────────────────────────────
PREP_DIR    = "data_lstm"
OUTPUT_DIR  = "results_sktime_wwater_ieee"
TRAIN_RATIO, VAL_RATIO = 108 / 144, 24 / 144
FINAL_WINDOW            = 12
WINDOW_SIZE_CANDIDATES  = [6, 12, 18, 24, 36]
BATCH_SIZE_CANDIDATES   = [8, 16, 32, 64]
ES_PATIENCE   = 10
ES_MIN_DELTA  = 1e-5
ES_MAX_EPOCHS = 100
N_BOOTSTRAP, CI_LEVEL = 1000, 0.95

MTYPE = {
    "RocketRegressor": "Kernel DL (no GPU)", "LSTMFCNRegressor": "Deep learning",
    "InceptionTime":   "Deep learning",      "Naive": "Statistical",
    "SeasonalNaive":   "Statistical",        "ARIMA": "Statistical",
    "SARIMA":          "Statistical",        "RandomForest": "ML ensemble",
    "XGBoost":         "ML ensemble",
}

for sub in ["", "hyperparameter_plots", "model_comparisons", "time_series_plots",
            "combined_plots", "training_curves", "baseline_results", "residual_tests"]:
    os.makedirs(os.path.join(OUTPUT_DIR, sub) if sub else OUTPUT_DIR, exist_ok=True)

def section(t): print(f"\n{'='*70}\n{t}\n{'='*70}")
def _fmv(v):    return "N/A" if (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_naive=None):
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    mse  = float(np.mean((y_true - y_pred)**2))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100)
    mase = rmsse = np.nan
    if y_naive is not None and len(y_naive) > 1:
        d = np.diff(y_naive); sm = np.mean(np.abs(d)); sr = np.sqrt(np.mean(d**2))
        if sm > 1e-8: mase  = float(mae  / sm)
        if sr > 1e-8: rmsse = float(rmse / sr)
    return mae, mse, rmse, mape, mase, rmsse

def _nanmean(lst):
    v = [x for x in lst if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.nanmean(v)) if v else np.nan

def _nanstd(lst):
    v = [x for x in lst if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.std(v)) if v else np.nan

def seed_avg_metrics(preds_list, y_true, y_naive):
    rows = [compute_metrics(y_true, yp, y_naive) for yp in preds_list]
    keys = ["MAE", "MSE", "RMSE", "MAPE(%)", "MASE", "RMSSE"]
    return {k: _nanmean([r[i] for r in rows]) for i, k in enumerate(keys)}

# ── Normalization ─────────────────────────────────────────────────────────────
def norm3d_fit(Xtr, Xte):
    scs = []; Xtr_n, Xte_n = np.zeros_like(Xtr), np.zeros_like(Xte)
    for i in range(Xtr.shape[1]):
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
    Xtr_n, Xte_n, _ = norm3d_fit(Xtr, Xte); return Xtr_n, Xte_n

# ── Splits ────────────────────────────────────────────────────────────────────
def time_split(n, tr, vr=0.0):
    te = int(round(n * tr)); ve = int(round(n * (tr + vr)))
    return np.arange(0, te), np.arange(te, ve), np.arange(ve, n)

def split_from_meta(n, meta):
    ntr = meta.get("train_samples_after_windowing")
    nva = meta.get("val_samples_after_windowing")
    if ntr is None or nva is None: return time_split(n, TRAIN_RATIO, VAL_RATIO)
    ntr, nva = int(ntr), int(nva)
    return np.arange(0, ntr), np.arange(ntr, ntr+nva), np.arange(ntr+nva, n)

# ── Bootstrap CI & Residual Test ──────────────────────────────────────────────
def bootstrap_ci(y_true, y_pred, n=1000, ci=0.95):
    res = y_true - y_pred; a = 1 - ci; rng = np.random.default_rng(1)
    boots = np.array([y_pred + rng.choice(res, size=len(y_pred), replace=True) for _ in range(n)])
    return np.percentile(boots, 100*a/2, axis=0), np.percentile(boots, 100*(1-a/2), axis=0)

def residual_tests(y_true, y_pred, name):
    res = y_true - y_pred; out = {"model": disp(name)}
    try:
        lb = acorr_ljungbox(res, lags=[5], return_df=True)
        out["ljung_box_stat_lag5"]   = float(lb["lb_stat"].values[0])
        out["ljung_box_pvalue_lag5"] = float(lb["lb_pvalue"].values[0])
        out["autocorrelation_ok"]    = bool(lb["lb_pvalue"].values[0] > 0.05)
    except Exception as e: out["ljung_box_error"] = str(e)
    return out

# ── Keras Helpers ─────────────────────────────────────────────────────────────
def _make_es(monitor="loss"):
    try:
        import tensorflow as tf
        return tf.keras.callbacks.EarlyStopping(
            monitor=monitor, patience=ES_PATIENCE, min_delta=ES_MIN_DELTA,
            restore_best_weights=True, verbose=0)
    except Exception: return None

def _make_reduce_lr():
    try:
        import tensorflow as tf
        return tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=0)
    except Exception: return None

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
    for a in ("model_", "network_"):
        inner = getattr(m, a, None)
        if inner is None: continue
        for s in ("history", "history_"):
            h = getattr(inner, s, None)
            if h is not None:
                h = h.history if hasattr(h, "history") else h
                if isinstance(h, dict) and h.get("loss"): return h
    h = getattr(m, "history_", None)
    if h is not None:
        h = h.history if hasattr(h, "history") else h
        if isinstance(h, dict) and h.get("loss"): return h
    return {}

def fit_capture(model, X, y, val_data=None):
    """
    Tek eğitim noktası. Clone/refit yok.
    val_data verilirse: ES val_loss + ReduceLR aktif.
    val_data yoksa: ES train loss aktif.
    """
    if val_data is not None:
        Xv, yv = val_data
        cbs = [cb for cb in [_make_es("val_loss"), _make_reduce_lr()] if cb]
        try: model.callbacks = cbs
        except Exception: pass
        try:
            model.fit(X, y, validation_data=(Xv, yv))
            return _get_history(model)
        except TypeError: pass
        model.fit(X, y)
        km = _inner_keras(model)
        if km is not None:
            try:
                cbs2 = [cb for cb in [_make_es("val_loss"), _make_reduce_lr()] if cb]
                h = km.fit(
                    np.transpose(X, (0, 2, 1)), y,
                    epochs=getattr(model, "n_epochs", ES_MAX_EPOCHS),
                    batch_size=getattr(model, "batch_size", 32),
                    validation_data=(np.transpose(Xv, (0, 2, 1)), yv),
                    callbacks=cbs2, verbose=0,
                )
                return h.history
            except Exception as e:
                print(f"  [WARN] val_loss refit failed: {e}")
        else:
            print(f"  [WARN] {type(model).__name__}: no Keras inner model")
    else:
        cbs = [cb for cb in [_make_es("loss")] if cb]
        try: model.callbacks = cbs
        except Exception: pass
        model.fit(X, y)
    return _get_history(model)

# ── Baseline Models ───────────────────────────────────────────────────────────
def run_naive(ytr, yte):  return np.full(len(yte), ytr[-1])

def run_snaive(ytr, yte, s=12):
    history, preds = list(ytr), []
    for i in range(len(yte)):
        idx = len(history) - s
        preds.append(history[idx] if idx >= 0 else history[0])
        history.append(yte[i])
    return np.array(preds)

def run_arima(ytr, yte):
    from statsmodels.tsa.arima.model import ARIMA
    best_aic, best_ord = np.inf, (1, 1, 1)
    for p in range(3):
        for d in range(2):
            for q in range(3):
                try:
                    a = ARIMA(ytr, order=(p, d, q)).fit().aic
                    if a < best_aic: best_aic, best_ord = a, (p, d, q)
                except Exception: pass
    print(f"    ARIMA best order: {best_ord}")
    hist, preds = list(ytr), []
    for t in range(len(yte)):
        try:    fc = ARIMA(hist, order=best_ord).fit().forecast(1)[0]
        except: fc = hist[-1]
        preds.append(fc); hist.append(yte[t])
    return np.array(preds)

def run_sarima(ytr, yte):
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        from statsmodels.tsa.arima.model import ARIMA
        hist, preds = list(ytr), []
        for t in range(len(yte)):
            try:
                fc = SARIMAX(hist, order=(1,1,1), seasonal_order=(1,1,0,12),
                             enforce_stationarity=False,
                             enforce_invertibility=False).fit(disp=False).forecast(1)[0]
            except:
                try:    fc = ARIMA(hist, order=(1,1,1)).fit().forecast(1)[0]
                except: fc = hist[-1]
            preds.append(fc); hist.append(yte[t])
        return np.array(preds)
    except Exception: return np.full(len(yte), np.nan)

def run_rf(Xtr, ytr, Xte, rs=42):
    from sklearn.ensemble import RandomForestRegressor
    m = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=3,
                              max_features="sqrt", random_state=rs, n_jobs=-1)
    m.fit(Xtr.reshape(len(Xtr), -1), ytr)
    return m.predict(Xte.reshape(len(Xte), -1))

def run_xgb(Xtr, ytr, Xte, rs=42):
    try:
        import xgboost as xgb
        m = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                             reg_lambda=1.0, random_state=rs, verbosity=0)
        m.fit(Xtr.reshape(len(Xtr), -1), ytr)
        return m.predict(Xte.reshape(len(Xte), -1))
    except Exception: return np.full(len(ytr), np.nan)

# ── Model Utilities ───────────────────────────────────────────────────────────
def count_params(model):
    for a in ("model_", "model", "network_", "clf_"):
        obj = getattr(model, a, None)
        if obj is None: continue
        for s in ("model_", "model", "network_"):
            k = getattr(obj, s, None)
            if k and hasattr(k, "count_params"):
                try: return int(k.count_params())
                except Exception: pass
        if hasattr(obj, "count_params"):
            try: return int(obj.count_params())
            except Exception: pass
    return None

def _import_inception():
    for cn in ("InceptionTimeRegressor", "InceptionTime"):
        try: return getattr(__import__("sktime.regression.deep_learning", fromlist=[cn]), cn)
        except ImportError: pass
    raise ImportError("InceptionTime not found. pip install sktime[deep-learning]")

# ── Hyperparameter Tuning ─────────────────────────────────────────────────────
param_grids = {
    "RocketRegressor":  {"num_kernels": [500, 1000, 2000, 5000, 10000]},
    "LSTMFCNRegressor": {"lstm_size": [4, 8, 16], "batch_size": [8, 16]},
    "InceptionTime":    {"batch_size": [8, 16, 32], "n_filters": [16, 32, 64], "depth": [3, 6]},
}

def tune_model(cls, base_kw, grid, Xtr, ytr_sc, yscaler, name, Xva, yva_sc, Xtrva, ytrva_sc):
    section(f"HYPERPARAMETER TUNING: {disp(name)}")
    print(f"  train={len(Xtr)} | val={len(Xva)}  [TEST SET NOT SEEN]")
    yva = yscaler.inverse_transform(yva_sc.reshape(-1, 1)).ravel()
    best_model, best_score, best_params, total_t, tune_res = None, np.inf, {}, 0.0, {}
    for pn, vals in grid.items():
        print(f"\n  {pn}: {vals}"); rmse_l, mae_l = [], []
        for v in vals:
            kw = {k: w for k, w in {**base_kw, pn: v}.items() if k != "callbacks"}
            try:
                set_all_seeds(RANDOM_SEED)
                print(f"    {pn}={v} ...", end=" ", flush=True)
                m = cls(**kw); t0 = time.time()
                h_t = fit_capture(m, Xtr, ytr_sc, val_data=(Xva, yva_sc))
                total_t += time.time() - t0
                ypv = yscaler.inverse_transform(m.predict(Xva).reshape(-1, 1)).ravel()
                mae, _, rmse, _, _, _ = compute_metrics(yva, ypv)
                rmse_l.append(rmse); mae_l.append(mae)
                ep = len(h_t.get("loss", [])) if h_t else "?"
                print(f"val_RMSE={rmse:.2f}  val_MAE={mae:.2f}  stopped@ep={ep}")
                if rmse < best_score:
                    best_score = rmse; best_params = {pn: v}
                    set_all_seeds(RANDOM_SEED); t0 = time.time()
                    bc = cls(**{**{k: w for k, w in kw.items() if k != "callbacks"},
                                "random_state": RANDOM_SEED})
                    h_f = fit_capture(bc, Xtrva, ytrva_sc, val_data=None)
                    total_t += time.time() - t0
                    hp = h_t if (h_t and h_t.get("val_loss")) else h_f
                    if hp: bc._captured_history = hp
                    best_model = bc
            except Exception as e:
                print(f"ERROR: {e}"); rmse_l.append(np.nan); mae_l.append(np.nan)
        tune_res[pn] = {"values": vals, "rmse": rmse_l, "mae": mae_l}
    print(f"\n  Best: {best_params}  val-RMSE: {best_score:.2f}")
    print(f"  Final model: train+val={len(Xtrva)} samples")
    return best_model, best_params, tune_res, total_t

# ── Sweep (görsel amaçlı) ─────────────────────────────────────────────────────
def sweep_dl(cfg, pn, vals, Xtr, ytr_sc, yscaler, Xva, yva_sc):
    yva = yscaler.inverse_transform(yva_sc.reshape(-1, 1)).ravel()
    rd, md = {}, {}
    for nm, c in cfg.items():
        bkw = {k: v for k, v in {**c["base_kwargs"], **c.get("best_params", {})}.items()
               if k != "callbacks"}
        rl, ml = [], []; print(f"\n  {disp(nm)} | {pn}: {vals}")
        for v in vals:
            try:
                print(f"    {pn}={v} ...", end=" ", flush=True)
                m = c["class"](**{**bkw, pn: v})
                fit_capture(m, Xtr, ytr_sc)
                yp = yscaler.inverse_transform(m.predict(Xva).reshape(-1, 1)).ravel()
                mae, _, rmse, _, _, _ = compute_metrics(yva, yp)
                rl.append(rmse); ml.append(mae); print(f"RMSE={rmse:.2f}")
            except Exception as e:
                print(f"ERROR:{e}"); rl.append(np.nan); ml.append(np.nan)
        rd[nm] = rl; md[nm] = ml
    return rd, md

def _sweep_window_core(m_or_fn, bkw, ws, rawX, rawy, train_r, is_bl=False):
    rl, ml = [], []
    for w in ws:
        try:
            T = rawX.shape[2]
            Xw = (rawX[:, :, (T-w):] if w <= T
                  else np.concatenate([np.zeros((rawX.shape[0], rawX.shape[1], w-T)), rawX], axis=2))
            ti, vi, _ = time_split(Xw.shape[0], train_r, VAL_RATIO)
            ytr, yva = rawy[ti], rawy[vi]
            if is_bl:
                yp = m_or_fn(Xw[ti], ytr, Xw[vi])
            else:
                Xn, Xvn = norm3d(Xw[ti], Xw[vi])
                sc = MinMaxScaler()
                ytrs = sc.fit_transform(ytr.reshape(-1, 1)).ravel()
                m = m_or_fn(**{k: v for k, v in bkw.items() if k != "callbacks"})
                fit_capture(m, Xn, ytrs)
                yp = sc.inverse_transform(m.predict(Xvn).reshape(-1, 1)).ravel()
            mae, _, rmse, _, _, _ = compute_metrics(yva, yp)
            rl.append(rmse); ml.append(mae); print(f"    w={w}: RMSE={rmse:.2f}")
        except Exception as e:
            print(f"    w={w}: ERROR {e}"); rl.append(np.nan); ml.append(np.nan)
    return rl, ml

def sweep_window(cfg, ws, rawX, rawy, train_r):
    rd, md = {}, {}
    for nm, c in cfg.items():
        bkw = {**c["base_kwargs"], **c.get("best_params", {})}
        print(f"\n  {disp(nm)} | window: {ws}")
        rd[nm], md[nm] = _sweep_window_core(c["class"], bkw, ws, rawX, rawy, train_r)
    return rd, md

def sweep_window_bl(ws, rawX, rawy, train_r):
    rd, md = {}, {}
    for nm, fn in [("RandomForest", run_rf), ("XGBoost", run_xgb)]:
        print(f"\n  {disp(nm)} | window: {ws}")
        rd[nm], md[nm] = _sweep_window_core(fn, {}, ws, rawX, rawy, train_r, is_bl=True)
    return rd, md

# ── Sonuçları Kaydet ──────────────────────────────────────────────────────────
def _save_model_results(name, ytr, yva, yte, yp_val, yp_test,
                        train_t, inf_t, ts_tr, ts_va, ts_te,
                        results, val_results, all_predictions, all_val_predictions,
                        all_ci, all_ci_val, outdir, is_baseline=False,
                        precomputed_test_metrics=None, precomputed_val_metrics=None):
    sub = "baseline_results" if is_baseline else "time_series_plots"
    mc  = pj(outdir, "baseline_results" if is_baseline else "model_comparisons")

    ci_lo_v, ci_hi_v = bootstrap_ci(yva, yp_val, N_BOOTSTRAP, CI_LEVEL) if yp_val is not None else (None, None)
    ci_lo,   ci_hi   = bootstrap_ci(yte, yp_test, N_BOOTSTRAP, CI_LEVEL)

    all_predictions[name] = yp_test; all_ci[name] = (ci_lo, ci_hi)
    if yp_val is not None:
        all_val_predictions[name] = yp_val; all_ci_val[name] = (ci_lo_v, ci_hi_v)

    if precomputed_val_metrics is not None:
        mae_v, mse_v, rmse_v, mape_v, mase_v, rmsse_v = (
            precomputed_val_metrics["MAE"], precomputed_val_metrics["MSE"],
            precomputed_val_metrics["RMSE"], precomputed_val_metrics["MAPE(%)"],
            precomputed_val_metrics["MASE"], precomputed_val_metrics["RMSSE"])
    else:
        mae_v, mse_v, rmse_v, mape_v, mase_v, rmsse_v = (
            compute_metrics(yva, yp_val, ytr) if yp_val is not None else (np.nan,)*6)

    if precomputed_test_metrics is not None:
        mae, mse, rmse, mape, mase, rmsse = (
            precomputed_test_metrics["MAE"], precomputed_test_metrics["MSE"],
            precomputed_test_metrics["RMSE"], precomputed_test_metrics["MAPE(%)"],
            precomputed_test_metrics["MASE"], precomputed_test_metrics["RMSSE"])
    else:
        mae, mse, rmse, mape, mase, rmsse = compute_metrics(yte, yp_test, np.concatenate([ytr, yva]))

    val_results.append({"model": name, "MAE": mae_v, "MSE": mse_v, "RMSE": rmse_v,
                        "MAPE(%)": mape_v, "MASE": mase_v, "RMSSE": rmsse_v,
                        "Training_Time_s": train_t, "Inference_Time_s": inf_t})
    results.append({"model": name, "MAE": mae, "MSE": mse, "RMSE": rmse,
                    "MAPE(%)": mape, "MASE": mase, "RMSSE": rmsse,
                    "Training_Time_s": train_t, "Inference_Time_s": inf_t})

    if yp_val is not None:
        plot_forecast(yva, yp_val, pj(outdir, sub, f"{name}_val_timeseries.png"), name,
                      ts=ts_va, ci_lo=ci_lo_v, ci_hi=ci_hi_v, label="Validation")
        pd.DataFrame({"y_true": yva, "y_pred": yp_val, "residual": yva-yp_val,
                      "ci_lower": ci_lo_v, "ci_upper": ci_hi_v}
                     ).to_csv(pj(outdir, f"{name}_val_predictions.csv"), index=False)

    _pm = {"MAE": mae, "RMSE": rmse, "MAPE(%)": mape}
    plot_parity(yte, yp_test, pj(mc, f"{name}_parity.png"), name, precomputed_metrics=_pm)
    plot_forecast(yte, yp_test, pj(outdir, sub, f"{name}_test_timeseries.png"), name,
                  ts=ts_te, ci_lo=ci_lo, ci_hi=ci_hi, label="Test")
    plot_full_series(ytr, yva, yte, yp_val, yp_test, pj(outdir, sub, f"{name}_full_series.png"),
                     name, ts_tr=ts_tr, ts_va=ts_va, ts_te=ts_te,
                     ci_lo_v=ci_lo_v, ci_hi_v=ci_hi_v, ci_lo=ci_lo, ci_hi=ci_hi)

    rt = residual_tests(yte, yp_test, name)
    pd.DataFrame([rt]).to_csv(pj(outdir, "residual_tests", f"{name}_stat_tests.csv"), index=False)
    pd.DataFrame({"y_true": yte, "y_pred": yp_test, "residual": yte-yp_test,
                  "ci_lower": ci_lo, "ci_upper": ci_hi}
                 ).to_csv(pj(outdir, f"{name}_test_predictions.csv"), index=False)

    print(f"\n  {disp(name)} – VAL:  RMSE={rmse_v:.2f}  MAE={mae_v:.2f}  MAPE={mape_v:.4f}%")
    print(f"  {disp(name)} – TEST: RMSE={rmse:.2f}  MAE={mae:.2f}  MAPE={mape:.4f}%")

# ═════════════════════════════════════════════════════════════════════════════
# MAIN FLOW
# ═════════════════════════════════════════════════════════════════════════════
section("DATA LOADING")
X    = np.load(pj(PREP_DIR, "X_windows.npy"))
y    = pd.read_csv(pj(PREP_DIR, "y.csv"))["W-Water"].values.astype(float)
meta = json.load(open(pj(PREP_DIR, "meta.json"), encoding="utf-8"))
try:    ts = pd.to_datetime(pd.read_csv(pj(PREP_DIR, "timestamps.csv"))["target_month"]).values
except: ts = None

X3d = np.transpose(X, (0, 2, 1)); n = X3d.shape[0]
assert len(y) == n
print(f"X3d: {X3d.shape}  |  y: {y.shape}  min={y.min():.0f}  max={y.max():.0f}")
print(f"Features: {meta.get('feature_columns','?')}  |  Final window: W={FINAL_WINDOW}")

section("TRAIN / VALIDATION / TEST SPLIT")
tri, vai, tei = split_from_meta(n, meta)
Xtr, Xva, Xte = X3d[tri], X3d[vai], X3d[tei]
ytr, yva, yte  = y[tri],  y[vai],  y[tei]
ts_tr = ts[tri] if ts is not None else None
ts_va = ts[vai] if ts is not None else None
ts_te = ts[tei] if ts is not None else None

def dr(k): return f"{meta.get(k,['?','?'])[0]} – {meta.get(k,['?','?'])[1]}"
print(f"Train: {len(tri)} samples ({dr('train_date_range')})")
print(f"Val  : {len(vai)} samples ({dr('val_date_range')})")
print(f"Test : {len(tei)} samples ({dr('test_date_range')})")

section("NORMALIZATION")
Xtr_n, Xte_n, x_scs = norm3d_fit(Xtr, Xte)
Xva_n   = norm3d_apply(Xva, x_scs)
Xtrva   = np.concatenate([Xtr, Xva],    axis=0)
Xtrva_n = np.concatenate([Xtr_n, Xva_n], axis=0)
ysc      = MinMaxScaler()
ytr_sc   = ysc.fit_transform(ytr.reshape(-1, 1)).ravel()
yva_sc   = ysc.transform(yva.reshape(-1, 1)).ravel()
ytrva    = np.concatenate([ytr, yva])
ytrva_sc = ysc.transform(ytrva.reshape(-1, 1)).ravel()

section("MODEL DEFINITIONS")
from sktime.regression.kernel_based import RocketRegressor
from sktime.regression.deep_learning import LSTMFCNRegressor
InceptionTimeRegressor = _import_inception()

models = {
    "RocketRegressor": {
        "class": RocketRegressor,
        "base_kwargs": dict(num_kernels=10000, random_state=RANDOM_SEED),
    },
    "LSTMFCNRegressor": {
        "class": LSTMFCNRegressor,
        "base_kwargs": dict(n_epochs=ES_MAX_EPOCHS, batch_size=8, lstm_size=8,
                            dropout=0.3, random_state=RANDOM_SEED, verbose=False),
    },
    "InceptionTime": {
        "class": InceptionTimeRegressor,
        "base_kwargs": dict(n_epochs=ES_MAX_EPOCHS, batch_size=16, n_filters=32, depth=6,
                            use_residual=True, use_bottleneck=True,
                            random_state=RANDOM_SEED, verbose=False),
    },
}

section("TRAINING & HYPERPARAMETER SELECTION  [Multi-seed]")
results, val_results             = [], []
all_predictions, all_val_predictions = {}, {}
all_ci, all_ci_val               = {}, {}
train_times, infer_times         = {}, {}
all_histories, best_params_log, all_tune_res = {}, {}, {}
param_counts = {}
seed_results = {nm: {"MAE": [], "RMSE": [], "MAPE(%)": []} for nm in models}
tc = pj(OUTPUT_DIR, "training_curves")

for name, cfg in models.items():
    cls = cfg["class"]; bkw = cfg["base_kwargs"]
    section(f"Model: {disp(name)}")

    model, best_p, tune_res, tr_t = tune_model(
        cls, bkw, param_grids[name], Xtr_n, ytr_sc, ysc, name,
        Xva_n, yva_sc, Xtrva_n, ytrva_sc)

    train_times[name] = tr_t; best_params_log[name] = best_p
    all_tune_res[name] = tune_res; models[name]["best_params"] = best_p
    plot_hp_grid(tune_res, name, pj(OUTPUT_DIR, "hyperparameter_plots", f"{name}_tuning.png"))

    h = getattr(model, "_captured_history", None) or _get_history(model)
    if h: all_histories[name] = h; plot_epoch_loss(h, name, tc)

    pc = count_params(model)
    if pc: param_counts[name] = pc; print(f"  Trainable params: {pc:,}")

    bkw2 = {k: v for k, v in {**bkw, **best_p}.items() if k != "callbacks"}
    print(f"\n  Multi-seed run (best_params={best_p}, seeds={SEEDS})...")
    test_preds_list, val_preds_list = [], []

    for seed in SEEDS:
        try:
            set_all_seeds(seed); kw_s = {**bkw2, "random_state": seed}

            ms_val = cls(**kw_s)
            fit_capture(ms_val, Xtr_n, ytr_sc, val_data=(Xva_n, yva_sc))
            yp_va_sc = ms_val.predict(Xva_n)
            if not np.isnan(yp_va_sc).any():
                val_preds_list.append(ysc.inverse_transform(yp_va_sc.reshape(-1, 1)).ravel())

            ms_test = cls(**kw_s)
            fit_capture(ms_test, Xtrva_n, ytrva_sc, val_data=None)
            yp_te_sc = ms_test.predict(Xte_n)
            if np.isnan(yp_te_sc).any():
                print(f"    seed={seed}: NaN test – atlandı"); continue

            yp_t_s = ysc.inverse_transform(yp_te_sc.reshape(-1, 1)).ravel()
            test_preds_list.append(yp_t_s)
            m_, mse_, r_, mp_, mase_, rmsse_ = compute_metrics(yte, yp_t_s, ytrva)
            seed_results[name]["MAE"].append(m_)
            seed_results[name]["RMSE"].append(r_)
            seed_results[name]["MAPE(%)"].append(mp_)
            seed_results[name].setdefault("_rows", []).append({
                "model": disp(name), "seed": seed,
                "RMSE": r_, "MAE": m_, "MAPE(%)": mp_, "MASE": mase_, "RMSSE": rmsse_})
            print(f"    seed={seed}: RMSE={r_:.2f}  MAE={m_:.2f}  MAPE={mp_:.4f}%")
        except Exception as e:
            print(f"    seed={seed}: HATA – {e}")

    if not test_preds_list:
        print(f"  [WARN] {disp(name)}: hiçbir seed başarılı olmadı, atlandı."); continue

    yp_test_avg = np.mean(test_preds_list, axis=0)
    yp_val_avg  = np.mean(val_preds_list,  axis=0) if val_preds_list else None
    infer_times[name] = 0.0

    te_metrics = seed_avg_metrics(test_preds_list, yte, ytrva)
    te_metrics["MSE"] = float(np.mean((yte - yp_test_avg)**2))
    val_metrics = seed_avg_metrics(val_preds_list, yva, ytr) if val_preds_list else None

    sr = seed_results[name]
    print(f"  {disp(name)} – SEED ORT: RMSE={_nanmean(sr['RMSE']):.2f}  "
          f"MAE={_nanmean(sr['MAE']):.2f}  MAPE={_nanmean(sr['MAPE(%)']):.4f}%  "
          f"(N={len(test_preds_list)} seed)")

    _save_model_results(name, ytr, yva, yte, yp_val_avg, yp_test_avg,
                        tr_t, infer_times[name], ts_tr, ts_va, ts_te,
                        results, val_results, all_predictions, all_val_predictions,
                        all_ci, all_ci_val, OUTPUT_DIR,
                        precomputed_test_metrics=te_metrics,
                        precomputed_val_metrics=val_metrics)
    print(f"\n  {disp(name)} done  ({len(test_preds_list)} seed).")

if all_histories: plot_training_curves_combined(all_histories, tc)

# ── Baseline: Deterministik ───────────────────────────────────────────────────
section("BASELINE MODELS")
det_bl_fns = {
    "Naive":         (lambda: run_naive(ytrva, yte),  lambda: run_naive(ytr, yva)),
    "SeasonalNaive": (lambda: run_snaive(ytrva, yte), lambda: run_snaive(ytr, yva)),
    "ARIMA":         (lambda: run_arima(ytrva, yte),  lambda: run_arima(ytr, yva)),
    "SARIMA":        (lambda: run_sarima(ytrva, yte), lambda: run_sarima(ytr, yva)),
}
for bname, (test_fn, val_fn) in det_bl_fns.items():
    print(f"\n  {disp(bname)} ...")
    try:
        yp_v = val_fn()
        if np.isnan(yp_v).any(): yp_v = None
        t0 = time.time(); yp_t = test_fn(); bt = time.time() - t0
        if np.isnan(yp_t).any(): print("  NaN – atlandı"); continue
        train_times[bname] = bt; infer_times[bname] = 0.0
        _save_model_results(bname, ytr, yva, yte, yp_v, yp_t, bt, 0.0,
                            ts_tr, ts_va, ts_te, results, val_results,
                            all_predictions, all_val_predictions,
                            all_ci, all_ci_val, OUTPUT_DIR, is_baseline=True)
        print(f"  {disp(bname)} done.")
    except Exception as e: print(f"  ERROR: {e}")

# ── Baseline: Stokastik ───────────────────────────────────────────────────────
stoch_bl = {
    "RandomForest": {"test_fn": lambda rs: run_rf(Xtrva_n, ytrva, Xte_n, rs=rs),
                     "val_fn":  lambda rs: run_rf(Xtr_n,   ytr,   Xva_n, rs=rs)},
    "XGBoost":      {"test_fn": lambda rs: run_xgb(Xtrva_n, ytrva, Xte_n, rs=rs),
                     "val_fn":  lambda rs: run_xgb(Xtr_n,   ytr,   Xva_n, rs=rs)},
}
for bname, fns in stoch_bl.items():
    print(f"\n  {disp(bname)} – multi-seed (seeds={SEEDS}) ...")
    bl_te, bl_va = [], []; t_total = 0.0
    for seed in SEEDS:
        try:
            set_all_seeds(seed)
            yp_v_s = fns["val_fn"](rs=seed)
            t0 = time.time(); yp_t_s = fns["test_fn"](rs=seed); t_total += time.time() - t0
            if not np.isnan(yp_t_s).any():
                bl_te.append(yp_t_s)
                m_, mse_, r_, mp_, mas_, rms_ = compute_metrics(yte, yp_t_s, ytrva)
                seed_results.setdefault(bname, {"MAE": [], "RMSE": [], "MAPE(%)": []})
                seed_results[bname]["MAE"].append(m_)
                seed_results[bname]["RMSE"].append(r_)
                seed_results[bname]["MAPE(%)"].append(mp_)
                seed_results[bname].setdefault("_rows", []).append({
                    "model": disp(bname), "seed": seed,
                    "RMSE": r_, "MAE": m_, "MAPE(%)": mp_, "MASE": mas_, "RMSSE": rms_})
                print(f"    seed={seed}: RMSE={r_:.2f}  MAE={m_:.2f}  MAPE={mp_:.4f}%")
            if not np.isnan(yp_v_s).any(): bl_va.append(yp_v_s)
        except Exception as e: print(f"    seed={seed}: HATA – {e}")

    if not bl_te: print(f"  [WARN] {disp(bname)}: hiçbir seed başarılı olmadı."); continue

    yp_t_avg = np.mean(bl_te, axis=0)
    yp_v_avg = np.mean(bl_va, axis=0) if bl_va else None
    train_times[bname] = t_total; infer_times[bname] = 0.0

    bl_te_metrics = seed_avg_metrics(bl_te, yte, ytrva)
    bl_te_metrics["MSE"] = float(np.mean((yte - yp_t_avg)**2))
    bl_val_metrics = seed_avg_metrics(bl_va, yva, ytr) if bl_va else None

    _save_model_results(bname, ytr, yva, yte, yp_v_avg, yp_t_avg, t_total, 0.0,
                        ts_tr, ts_va, ts_te, results, val_results,
                        all_predictions, all_val_predictions,
                        all_ci, all_ci_val, OUTPUT_DIR, is_baseline=True,
                        precomputed_test_metrics=bl_te_metrics,
                        precomputed_val_metrics=bl_val_metrics)
    print(f"  {disp(bname)} done  ({len(bl_te)} seed).")

# ── Sweep Plots ───────────────────────────────────────────────────────────────
section("SWEEP PLOTS  [validation only]")
DL_CFG = {k: v for k, v in models.items() if k in {"LSTMFCNRegressor", "InceptionTime"}}
es_log = {mn: len(h.get("loss", [])) for mn, h in all_histories.items()}
for mn, ep in es_log.items(): print(f"  Early stopping – {disp(mn)}: {ep} epochs")
if es_log:
    pd.DataFrame([{"model": disp(k), "epochs_trained": v} for k, v in es_log.items()]
                 ).to_csv(pj(tc, "early_stopping_log.csv"), index=False)

print("\n  Batch size sweep (DL)...")
bsr, bsm = sweep_dl(DL_CFG, "batch_size", BATCH_SIZE_CANDIDATES, Xtr_n, ytr_sc, ysc, Xva_n, yva_sc)
plot_sweep(bsr, bsm, BATCH_SIZE_CANDIDATES, "Batch Size",
           "Batch Size Sensitivity – DL Models (Val)", pj(tc, "all_models_batch_rmse_mae.png"))

print("\n  Window size sweep (DL + ROCKET)...")
wrd, wmd = sweep_window(models, WINDOW_SIZE_CANDIDATES, X3d, y, TRAIN_RATIO)
print("\n  Window size sweep (RF, XGBoost)...")
wrb, wmb = sweep_window_bl(WINDOW_SIZE_CANDIDATES, X3d, y, TRAIN_RATIO)
plot_sweep({**wrd, **wrb}, {**wmd, **wmb}, WINDOW_SIZE_CANDIDATES,
           "Window Size (months)", "Window Size Sensitivity – All Models (Val)",
           pj(tc, "all_models_window_rmse_mae.png"), use_log=True)

# ── Combined Plots ────────────────────────────────────────────────────────────
section("COMBINED PLOTS")
cp = pj(OUTPUT_DIR, "combined_plots")
if all_predictions:
    _mdict = {r["model"]: {"MAE": r["MAE"], "RMSE": r["RMSE"], "MAPE(%)": r["MAPE(%)"]}
              for r in results}
    plot_all_single(yte, all_predictions, pj(cp, "all_models_test_single_plot.png"), ts=ts_te)
    plot_all_subplots(yte, all_predictions, pj(cp, "all_models_test_subplots_with_ci.png"),
                      ts=ts_te, ci_dict=all_ci, metrics_dict=_mdict)
    plot_error_boxplot(all_predictions, yte, pj(cp, "all_models_error_boxplot.png"))
    plot_parity_grid(yte, all_predictions, pj(cp, "combined_parity_all_models.png"),
                     metrics_dict=_mdict)

if param_counts:
    pd.DataFrame([{"model": disp(k), "trainable_parameters": v}
                  for k, v in param_counts.items()]
                 ).to_csv(pj(OUTPUT_DIR, "model_parameter_counts.csv"), index=False)

dl_seed = {k: v for k, v in seed_results.items()
           if any(len(l) > 0 for l in v.values()
                  if isinstance(l, list) and all(isinstance(x, (int, float)) for x in l))}
if dl_seed: plot_seed_var(dl_seed, pj(cp, "multi_seed_variance.png"))
if train_times: plot_comp_time(train_times, pj(cp, "computation_time_training.png"), inf_t=infer_times)

# ── Sonuçları Kaydet ──────────────────────────────────────────────────────────
section("SAVE RESULTS")
res_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
val_df = pd.DataFrame(val_results).sort_values("RMSE").reset_index(drop=True)
res_df.to_csv(pj(OUTPUT_DIR, "results_test.csv"),       index=False); print("results_test.csv saved")
val_df.to_csv(pj(OUTPUT_DIR, "results_validation.csv"), index=False); print("results_validation.csv saved")
plot_metrics_bar(res_df, pj(cp, "all_models_test_metrics_comparison.png"))
plot_metrics_bar(val_df, pj(cp, "all_models_val_metrics_comparison.png"))

rt_files = [f for f in os.listdir(pj(OUTPUT_DIR, "residual_tests")) if f.endswith("_stat_tests.csv")]
if rt_files:
    pd.concat([pd.read_csv(pj(OUTPUT_DIR, "residual_tests", f)) for f in rt_files], ignore_index=True
              ).to_csv(pj(OUTPUT_DIR, "all_residual_tests.csv"), index=False)

def fmt_t(t): return "<0.001" if (not np.isnan(t) and t < 0.001) else (f"{t:.3f}" if not np.isnan(t) else "")
pd.DataFrame([{"model": disp(m), "type": MTYPE.get(m, "—"),
               "training_time_s":  fmt_t(train_times.get(m, np.nan)),
               "inference_time_s": fmt_t(infer_times.get(m, np.nan))}
              for m in sorted(set(list(train_times) + list(infer_times)))
              ]).to_csv(pj(OUTPUT_DIR, "computation_times.csv"), index=False)
print("computation_times.csv saved")

# ── multi_seed_results.csv ────────────────────────────────────────────────────
seed_rows = []
for nm, data in seed_results.items():
    rows = data.get("_rows", [])
    if not rows: continue
    def _col(key): return [r.get(key, np.nan) for r in rows]
    stats = {k: (_nanmean(_col(k)), _nanstd(_col(k))) for k in ["RMSE","MAE","MAPE(%)","MASE","RMSSE"]}
    for r in rows:
        seed_rows.append({**r,
            **{f"mean_{k}": round(stats[k][0], 4) if not np.isnan(stats[k][0]) else np.nan for k in stats},
            **{f"std_{k}":  round(stats[k][1], 4) if not np.isnan(stats[k][1]) else np.nan
               for k in ["RMSE","MAE","MAPE(%)"]}})
    seed_rows.append({"model": disp(nm), "seed": "MEAN",
        **{k: round(stats[k][0], 4) if not np.isnan(stats[k][0]) else np.nan for k in stats},
        **{f"mean_{k}": round(stats[k][0], 4) if not np.isnan(stats[k][0]) else np.nan for k in stats},
        **{f"std_{k}":  round(stats[k][1], 4) if not np.isnan(stats[k][1]) else np.nan
           for k in ["RMSE","MAE","MAPE(%)"]}})
if seed_rows:
    pd.DataFrame(seed_rows).to_csv(pj(OUTPUT_DIR, "multi_seed_results.csv"), index=False)
    print("multi_seed_results.csv saved")

json.dump(best_params_log, open(pj(OUTPUT_DIR, "best_hyperparameters.json"), "w"), indent=2)
json.dump({mn: {pn: {"values": [float(v) for v in d["values"]],
                     "rmse":   [float(v) if not np.isnan(v) else None for v in d["rmse"]],
                     "mae":    [float(v) if not np.isnan(v) else None for v in d["mae"]]}
                for pn, d in pd_.items()}
           for mn, pd_ in all_tune_res.items()},
          open(pj(OUTPUT_DIR, "hyperparameter_tuning_results.json"), "w"), indent=2)

# ── Özet ─────────────────────────────────────────────────────────────────────
section("SUMMARY")
print(f"Final window: W={FINAL_WINDOW}  |  Seeds: {SEEDS}  |  Output: {OUTPUT_DIR}/")
print(f"Metrikler {len(SEEDS)} seed ortalaması üzerinden hesaplanmıştır.\n")

print("=" * 90 + "\nVALIDATION PERFORMANCE (sorted by RMSE):\n" + "=" * 90)
for i, (_, row) in enumerate(val_df.iterrows(), 1):
    print(f"{i:2}. {disp(row['model']):<18} RMSE={row['RMSE']:.2f}  MAE={row['MAE']:.2f}  "
          f"MAPE={row['MAPE(%)']:.4f}%  MASE={_fmv(row['MASE'])}  RMSSE={_fmv(row['RMSSE'])}")

print("\n" + "=" * 90 + "\nTEST PERFORMANCE (sorted by RMSE):\n" + "=" * 90)
for i, (_, row) in enumerate(res_df.iterrows(), 1):
    print(f"{i:2}. {disp(row['model']):<18} RMSE={row['RMSE']:.2f}  MAE={row['MAE']:.2f}  "
          f"MAPE={row['MAPE(%)']:.4f}%  MASE={_fmv(row['MASE'])}  RMSSE={_fmv(row['RMSSE'])}  "
          f"TrainT={row['Training_Time_s']:.2f}s")

print("\n" + "=" * 110)
print(f"MULTI-SEED ORTALAMA (seeds={SEEDS}):")
print("=" * 110)
print(f"{'Model':<20} {'Mean RMSE':>10} {'Std RMSE':>9} {'Mean MAE':>10} "
      f"{'Mean MAPE(%)':>13} {'Mean MASE':>10} {'Mean RMSSE':>11} {'N':>4}")
print("-" * 110)
for nm, data in seed_results.items():
    rows = data.get("_rows", [])
    if not rows: continue
    def _c(key): return [r.get(key,np.nan) for r in rows if not np.isnan(r.get(key,np.nan))]
    rv = _c("RMSE")
    if not rv: continue
    def _f(lst): return f"{np.mean(lst):>10.4f}" if lst else f"{'N/A':>10}"
    print(f"{disp(nm):<20} {np.mean(rv):>10.2f} {np.std(rv):>9.2f} "
          f"{np.mean(_c('MAE')) if _c('MAE') else float('nan'):>10.2f} "
          f"{np.mean(_c('MAPE(%)')) if _c('MAPE(%)') else float('nan'):>13.4f}"
          f"{_f(_c('MASE'))}{_f(_c('RMSSE'))} {len(rv):>4}")
