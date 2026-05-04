import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_COLORS = {"before": "#60a5fa", "after": "#34d399", "text": "#1e293b",
           "muted": "#64748b", "border": "#e2e8f0", "warning": "#f59e0b",
           "danger": "#ef4444", "success": "#10b981"}

def bar_chart(ax, series, color: str, title: str) -> None:
    vc = series.value_counts().sort_index()
    ax.bar([str(l) for l in vc.index], vc.values, color=color, alpha=0.8, width=0.6)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=6, color=_COLORS["text"])
    ax.set_xlabel(series.name, fontsize=8, color=_COLORS["muted"])
    ax.set_ylabel("Count", fontsize=8, color=_COLORS["muted"])
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", color=_COLORS["border"], linestyle="--", linewidth=0.3, alpha=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor(_COLORS["border"])

def simulate_balance(y_series, method: str) -> pd.Series:
    if method == "none":
        return y_series.copy()
    counts = y_series.value_counts()
    if method == "undersample":
        min_size = counts.min()
        return pd.concat([y_series[y_series == cls].sample(min_size, random_state=42) for cls in counts.index])
    max_size = counts.max()
    return pd.concat([y_series[y_series == cls].sample(max_size, replace=True, random_state=42) for cls in counts.index])

def apply_balancing(X_train: pd.DataFrame, y_train: pd.Series, method: str):
    balance_applied = False
    
    y_train_before = y_train.copy()
    
    try:
        if method == "smote":
            from imblearn.over_sampling import SMOTE
            sampler = SMOTE(random_state=42)
        elif method == "oversample":
            from imblearn.over_sampling import RandomOverSampler
            sampler = RandomOverSampler(random_state=42)
        elif method == "undersample":
            from imblearn.under_sampling import RandomUnderSampler
            sampler = RandomUnderSampler(random_state=42)
        X_train, y_train = sampler.fit_resample(X_train, y_train)
        balance_applied = True
    except ImportError:
        if method == "undersample":
            min_size = y_train_before.value_counts().min()
            df_comb = pd.concat([X_train, y_train_before], axis=1)
            df_res  = df_comb.groupby(y_train.name, group_keys=False).apply(lambda x: x.sample(min_size, random_state=42)).reset_index(drop=True)
        else:
            max_size = y_train_before.value_counts().max()
            df_comb = pd.concat([X_train, y_train_before], axis=1)
            df_res  = df_comb.groupby(y_train.name, group_keys=False).apply(lambda x: x.sample(max_size, replace=True, random_state=42)).reset_index(drop=True)
        X_train = df_res.drop(columns=[y_train.name])
        y_train = df_res[y_train.name]
        balance_applied = True
    
    return X_train, y_train, balance_applied
