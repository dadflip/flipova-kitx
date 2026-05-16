import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PAL  = ["#6366f1","#10b981","#f59e0b","#ef4444","#3b82f6","#8b5cf6","#ec4899"]
_GRAY = "#64748b"
_BG   = "#f8fafc"
_GRID = "#e2e8f0"
_MAX_LC_SAMPLES = 5_000

def is_valid(arr) -> bool:
    if arr is None: return False
    if isinstance(arr, pd.DataFrame) and arr.empty: return False
    if isinstance(arr, pd.Series)    and arr.empty: return False
    if isinstance(arr, np.ndarray)   and arr.size == 0: return False
    return True

def is_inference_mode(splits: dict) -> bool:
    return not is_valid(splits.get("y_test")) and is_valid(splits.get("X_test"))

def resolve_eval_data(splits, predictions, model_name):
    pred = (predictions or {}).get(model_name, {})
    Xv = pred.get("X_val"); yv = pred.get("y_val")
    if not is_valid(Xv) or not is_valid(yv):
        Xv = splits.get("X_test"); yv = splits.get("y_test")
    return Xv, yv

def fig(w=10, h=5, title=None):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(_BG); ax.set_facecolor(_BG)
    ax.grid(color=_GRID, linewidth=0.8, zorder=0)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color(_GRID)
    if title: ax.set_title(title, fontsize=11, fontweight="bold", color="#1e293b", pad=10)
    return fig, ax

def multi_fig(rows, cols, w=14, h=4):
    fig, axes = plt.subplots(rows, cols, figsize=(w, h*rows))
    fig.patch.set_facecolor(_BG)
    axes_flat = np.array(axes).flatten() if rows*cols > 1 else [axes]
    for ax in axes_flat:
        ax.set_facecolor(_BG); ax.grid(color=_GRID, linewidth=0.8, zorder=0)
        ax.spines[["top","right"]].set_visible(False); ax.spines[["left","bottom"]].set_color(_GRID)
    return fig, axes_flat

def compute_metrics(model, X_eval, y_eval, task, subtask, cfg_metrics) -> dict:
    from sklearn import metrics as skm
    results = {}
    if task == "classification":
        y_pred = model.predict(X_eval)
        sub_cfg = cfg_metrics.get("classification", {}).get(subtask, cfg_metrics.get("classification", {}).get("binary", []))
        for m in sub_cfg:
            try:
                fn = getattr(skm, m["func"])
                kwargs = m.get("kwargs", {})
                if m["func"] in ("roc_auc_score", "log_loss"):
                    if hasattr(model, "predict_proba"):
                        y_prob = model.predict_proba(X_eval)
                        val = fn(y_eval, y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob, **kwargs)
                    else: continue
                else:
                    val = fn(y_eval, y_pred, **kwargs)
                results[m["name"]] = val
            except Exception as e:
                results[m["name"]] = f"ERR: {e}"
    elif task == "regression":
        y_pred = model.predict(X_eval)
        for m in cfg_metrics.get("regression", []):
            try:
                fn = getattr(skm, m["func"])
                val = fn(y_eval, y_pred)
                if m.get("post") == "np.sqrt": val = np.sqrt(val)
                results[m["name"]] = val
            except Exception as e:
                results[m["name"]] = f"ERR: {e}"
    return results

def plot_confusion_matrix(model, X_eval, y_eval, model_name, ax=None):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import seaborn as sns
    y_pred = model.predict(X_eval)
    cm = confusion_matrix(y_eval, y_pred)
    if ax is None:
        f, ax = fig(5, 4, f"Confusion Matrix — {model_name}")
        
    if cm.shape == (2, 2):
        group_names = ['True Negative (TN)', 'False Positive (FP)', 'False Negative (FN)', 'True Positive (TP)']
        group_counts = [f"{value:0.0f}" for value in cm.flatten()]
        labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
        labels = np.asarray(labels).reshape(2, 2)
        
        sns.heatmap(cm, annot=labels, fmt='', cmap="Blues", cbar=False, ax=ax,
                    annot_kws={"size": 9, "fontweight": "bold"}, 
                    linecolor="white", linewidths=1)
        ax.set_xlabel('Predicted label', color=_GRAY)
        ax.set_ylabel('True label', color=_GRAY)
        
        if hasattr(model, 'classes_'):
            classes = model.classes_
            ax.set_xticklabels(classes)
            ax.set_yticklabels(classes)
    else:
        disp = ConfusionMatrixDisplay(cm, display_labels=getattr(model, 'classes_', None))
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=10, fontweight="bold", color="#1e293b")
    ax.set_facecolor(_BG)
    return ax.figure

def plot_roc_curves(models_dict, X_eval, y_eval):
    from sklearn.metrics import roc_curve, auc
    f, ax = fig(7, 5, "ROC Curves")
    for i, (name, model) in enumerate(models_dict.items()):
        try:
            if not hasattr(model, "predict_proba"): continue
            y_prob = model.predict_proba(X_eval)
            scores = y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob.max(axis=1)
            fpr, tpr, _ = roc_curve(y_eval, scores)
            ax.plot(fpr, tpr, color=_PAL[i % len(_PAL)], lw=2, label=f"{name} (AUC={auc(fpr,tpr):.3f})")
        except Exception: pass
    ax.plot([0,1],[0,1],"--",color=_GRAY,lw=1); ax.set_xlabel("FPR",color=_GRAY); ax.set_ylabel("TPR",color=_GRAY)
    ax.legend(fontsize=9); ax.set_xlim([0,1]); ax.set_ylim([0,1.02]); plt.tight_layout(); return f

def plot_residuals(model, X_eval, y_eval, model_name):
    y_pred = model.predict(X_eval); y_arr = np.array(y_eval); residuals = y_arr - y_pred
    f, axes = multi_fig(1, 3, w=14, h=4)
    axes[0].scatter(y_pred, residuals, alpha=0.5, color=_PAL[0], s=20)
    axes[0].axhline(0, color=_PAL[3], lw=1.5, ls="--"); axes[0].set_xlabel("Predicted",color=_GRAY); axes[0].set_ylabel("Residuals",color=_GRAY); axes[0].set_title(f"Residuals vs Predicted — {model_name}",fontsize=10,fontweight="bold",color="#1e293b")
    _mn = min(y_arr.min(), y_pred.min()); _mx = max(y_arr.max(), y_pred.max())
    axes[1].scatter(y_arr, y_pred, alpha=0.5, color=_PAL[1], s=20); axes[1].plot([_mn,_mx],[_mn,_mx],"--",color=_PAL[3],lw=1.5); axes[1].set_xlabel("Actual",color=_GRAY); axes[1].set_ylabel("Predicted",color=_GRAY); axes[1].set_title("Actual vs Predicted",fontsize=10,fontweight="bold",color="#1e293b")
    axes[2].hist(residuals, bins=30, color=_PAL[2], edgecolor="white", alpha=0.85); axes[2].set_xlabel("Residual",color=_GRAY); axes[2].set_ylabel("Count",color=_GRAY); axes[2].set_title("Residual Distribution",fontsize=10,fontweight="bold",color="#1e293b")
    plt.tight_layout(); return f

def plot_feature_importance(model, feature_names, model_name, top_n=20, X=None, y=None):
    importance = None
    if hasattr(model, "feature_importances_"): importance = model.feature_importances_
    elif hasattr(model, "coef_") and getattr(model, "coef_", None) is not None:
        coef = model.coef_
        if len(coef) > 0:
            importance = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    
    if importance is None and X is not None and y is not None:
        try:
            from sklearn.inspection import permutation_importance
            r = permutation_importance(model, X, y, n_repeats=3, random_state=42)
            importance = r.importances_mean
        except Exception:
            pass

    if importance is None: return None
    idx = np.argsort(importance)[-top_n:]
    f, ax = fig(8, max(4, len(idx)*0.35), f"Feature Importance — {model_name}")
    colors = [_PAL[0] if v > np.median(importance[idx]) else _PAL[2] for v in importance[idx]]
    ax.barh(np.array(feature_names)[idx], importance[idx], color=colors, edgecolor="white", height=0.7)
    ax.set_xlabel("Importance", color=_GRAY); ax.tick_params(labelsize=8); plt.tight_layout(); return f

def plot_learning_curve(model, X_train, y_train, model_name, task, max_samples=_MAX_LC_SAMPLES, cv=3, n_points=6):
    from sklearn.model_selection import learning_curve
    scoring = "accuracy" if task == "classification" else "r2"
    X_lc, y_lc = X_train, y_train; n = len(X_lc)
    if n > max_samples:
        rng = np.random.RandomState(42); idx = rng.choice(n, max_samples, replace=False)
        X_lc = X_lc.iloc[idx] if hasattr(X_lc, "iloc") else X_lc[idx]
        y_lc = y_lc.iloc[idx] if hasattr(y_lc, "iloc") else y_lc[idx]
    try:
        train_sz, train_sc, val_sc = learning_curve(model, X_lc, y_lc, cv=cv, n_jobs=1, scoring=scoring, train_sizes=np.linspace(0.1, 1.0, n_points))
    except Exception: return None
    f, ax = fig(8, 4.5, f"Learning Curve — {model_name}")
    ax.fill_between(train_sz, train_sc.mean(1)-train_sc.std(1), train_sc.mean(1)+train_sc.std(1), alpha=0.15, color=_PAL[0])
    ax.fill_between(train_sz, val_sc.mean(1)-val_sc.std(1), val_sc.mean(1)+val_sc.std(1), alpha=0.15, color=_PAL[1])
    ax.plot(train_sz, train_sc.mean(1), "o-", color=_PAL[0], lw=2, label=f"Train ({train_sc.mean(1)[-1]:.3f})")
    ax.plot(train_sz, val_sc.mean(1), "s-", color=_PAL[1], lw=2, label=f"Val CV ({val_sc.mean(1)[-1]:.3f})")
    ax.set_xlabel("Training examples", color=_GRAY); ax.set_ylabel(scoring.upper(), color=_GRAY)
    ax.legend(fontsize=9); ax.set_ylim(bottom=max(0, ax.get_ylim()[0])); plt.tight_layout(); return f

def plot_metric_comparison(all_metrics, metric_name):
    names = list(all_metrics.keys())
    pairs = [(n, all_metrics[n].get(metric_name)) for n in names if isinstance(all_metrics[n].get(metric_name), (int, float))]
    if not pairs: return None
    names, vals = zip(*pairs)
    f, ax = fig(7, 3.5, f"Model Comparison — {metric_name}")
    bars = ax.bar(names, vals, color=_PAL[:len(vals)], edgecolor="white", width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f"{val:.4f}", ha="center", va="bottom", fontsize=9, color="#1e293b")
    ax.set_ylabel(metric_name, color=_GRAY); ax.set_ylim(0, min(1.1, max(vals)*1.15))
    plt.xticks(rotation=20, ha="right", fontsize=9); plt.tight_layout(); return f

