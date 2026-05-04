import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_timeseries_fig(df: pd.DataFrame, col: str, time_col: str, p_type: str, window: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#f8fafc"); ax.set_facecolor("#f8fafc")
    s = df[col] if time_col == "Index" else df.set_index(time_col)[col]
    s = s.dropna()
    x, y = s.index, s.values
    if window > 1:
        y = s.rolling(window).mean().values
    if p_type == "Line":
        ax.plot(x, y, color="#3b82f6", label=f"{col} (window={window})")
        ax.legend()
    elif p_type == "Scatter":
        ax.scatter(x, y, color="#8b5cf6", alpha=0.5, s=10)
    elif p_type == "Area":
        ax.fill_between(x, y, color="#10b981", alpha=0.3)
        ax.plot(x, y, color="#10b981")
    elif p_type == "Box (par mois/année)":
        if not pd.api.types.is_datetime64_any_dtype(s.index):
            ax.text(0.5, 0.5, "Index doit être datetime", ha="center")
            return fig
        sns.boxplot(x=s.index.month, y=y, palette="Set2", ax=ax)
        ax.set_xlabel("Month")
    elif p_type == "Autocorrélation":
        lag = window
        ax.scatter(y[:-lag], y[lag:], alpha=0.5)
        ax.set_xlabel("y(t)"); ax.set_ylabel(f"y(t+{lag})")
        ax.set_title(f"Autocorrelation (lag={lag})")
    
    if p_type != "Autocorrélation":
        ax.set_title(f"Time Series: {col}")
        ax.set_xlabel("Time"); ax.set_ylabel(col)
    
    plt.tight_layout()
    return fig

def get_ts_seasonal_decompose(df: pd.DataFrame, col: str, time_col: str, period: int = 12):
    s = df[col] if time_col == "Index" else df.set_index(time_col)[col]
    s = s.dropna()
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        res = seasonal_decompose(s, period=period, extrapolate_trend='freq')
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        fig.patch.set_facecolor("#f8fafc")
        res.observed.plot(ax=axes[0], legend=False, title='Observed')
        res.trend.plot(ax=axes[1], legend=False, title='Trend')
        res.seasonal.plot(ax=axes[2], legend=False, title='Seasonal')
        res.resid.plot(ax=axes[3], style='o', legend=False, title='Residual')
        plt.tight_layout()
        return fig
    except ImportError:
        fig, ax = plt.subplots(figsize=(8,4))
        ax.text(0.5, 0.5, "pip install statsmodels requis pour la décomposition.", ha="center")
        return fig

def get_ts_acf_pacf(df: pd.DataFrame, col: str, time_col: str, lags: int = 40):
    s = df[col] if time_col == "Index" else df.set_index(time_col)[col]
    s = s.dropna()
    n_lags = min(lags, max(1, len(s) // 2 - 1))
    try:
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#f8fafc")
        plot_acf(s, lags=n_lags, ax=axes[0])
        plot_pacf(s, lags=n_lags, ax=axes[1])
        plt.tight_layout()
        return fig
    except ImportError:
        # Fallback manual ACF
        fig, ax = plt.subplots(figsize=(8, 4))
        from pandas.plotting import autocorrelation_plot
        autocorrelation_plot(s, ax=ax)
        ax.set_title("Autocorrelation Plot (fallback)")
        return fig

def get_ts_rolling_stats(df: pd.DataFrame, col: str, time_col: str, window: int = 12):
    s = df[col] if time_col == "Index" else df.set_index(time_col)[col]
    s = s.dropna()
    rolmean = s.rolling(window=window).mean()
    rolstd = s.rolling(window=window).std()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#f8fafc")
    ax.plot(s, color='blue', label='Original', alpha=0.5)
    ax.plot(rolmean, color='red', label='Rolling Mean')
    ax.plot(rolstd, color='black', label='Rolling Std')
    ax.legend(loc='best')
    ax.set_title(f'Rolling Mean & Standard Deviation (window={window})')
    plt.tight_layout()
    return fig

def get_ts_stationarity(df: pd.DataFrame, col: str, time_col: str):
    s = df[col] if time_col == "Index" else df.set_index(time_col)[col]
    s = s.dropna()
    res_dict = {}
    try:
        from statsmodels.tsa.stattools import adfuller
        dftest = adfuller(s, autolag='AIC')
        res_dict = {
            "Test Statistic": dftest[0],
            "p-value": dftest[1],
            "Lags Used": dftest[2],
            "Observations": dftest[3],
        }
        for key, value in dftest[4].items():
            res_dict[f'Critical Value ({key})'] = value
    except ImportError:
        res_dict = {"Error": "statsmodels package is required for Augmented Dickey-Fuller test"}
    return res_dict


