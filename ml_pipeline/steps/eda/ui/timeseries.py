import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ml_pipeline.styles import styles
from ml_pipeline.steps.eda.logic.timeseries import (
    get_timeseries_fig, get_ts_seasonal_decompose,
    get_ts_acf_pacf, get_ts_rolling_stats, get_ts_stationarity
)

def build_timeseries_ui(eda_ui, df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        eda_ui.dynamic_ui.children = [widgets.HTML("<div style='padding:12px;'>Time series non-DataFrame.</div>")]
        return
        
    num_cols = list(df.select_dtypes(include=np.number).columns)
    all_cols = list(df.columns)
    
    # Global selectors for the TS
    ts_col      = widgets.Dropdown(options=num_cols, description="Measurement:")
    ts_time_col = widgets.Dropdown(options=["Index"] + all_cols, value="Index", description="Time Col:")
    global_selectors = widgets.HBox([ts_time_col, ts_col], layout=widgets.Layout(margin="0 0 10px 0"))

    # Tool 1: Basics
    eda_cfg = getattr(eda_ui.state, "config", {}).get("eda", {})
    ts_type     = widgets.Dropdown(options=["Line", "Scatter", "Area", "Box (par mois/année)", "Autocorrélation"],
                                         value="Line", description="Plot Type:")
    ts_window   = widgets.IntSlider(value=1, min=1, max=365, description="Window:")
    ts_btn      = widgets.Button(description="Plot", button_style=styles.BTN_PRIMARY)
    ts_out      = widgets.Output()
    
    def _plot_timeseries(b):
        with ts_out:
            clear_output(wait=True)
            if not ts_col.value: return
            fig = get_timeseries_fig(df, ts_col.value, ts_time_col.value, ts_type.value, ts_window.value)
            display(fig)
            plt.close(fig)
    ts_btn.on_click(_plot_timeseries)
    tab1 = widgets.VBox([widgets.HBox([ts_type, ts_window, ts_btn]), ts_out])

    # Tool 2: Seasonal Decompose
    period_in = widgets.IntText(value=12, description="Period:")
    seas_btn  = widgets.Button(description="Decompose", button_style=styles.BTN_PRIMARY)
    seas_out  = widgets.Output()
    def _plot_seas(b):
        with seas_out:
            clear_output(wait=True)
            if not ts_col.value: return
            fig = get_ts_seasonal_decompose(df, ts_col.value, ts_time_col.value, period_in.value)
            display(fig)
            plt.close(fig)
    seas_btn.on_click(_plot_seas)
    tab2 = widgets.VBox([widgets.HBox([period_in, seas_btn]), seas_out])
    
    # Tool 3: ACF / PACF
    lags_in = widgets.IntText(value=40, description="Lags:")
    acf_btn  = widgets.Button(description="Plot ACF/PACF", button_style=styles.BTN_PRIMARY)
    acf_out  = widgets.Output()
    def _plot_acf(b):
        with acf_out:
            clear_output(wait=True)
            if not ts_col.value: return
            fig = get_ts_acf_pacf(df, ts_col.value, ts_time_col.value, lags=lags_in.value)
            display(fig)
            plt.close(fig)
    acf_btn.on_click(_plot_acf)
    tab3 = widgets.VBox([widgets.HBox([lags_in, acf_btn]), acf_out])
    
    # Tool 4: Rolling Stats
    roll_win = widgets.IntText(value=12, description="Window:")
    roll_btn = widgets.Button(description="Plot Rolling", button_style=styles.BTN_PRIMARY)
    roll_out = widgets.Output()
    def _plot_roll(b):
        with roll_out:
            clear_output(wait=True)
            if not ts_col.value: return
            fig = get_ts_rolling_stats(df, ts_col.value, ts_time_col.value, window=roll_win.value)
            display(fig)
            plt.close(fig)
    roll_btn.on_click(_plot_roll)
    tab4 = widgets.VBox([widgets.HBox([roll_win, roll_btn]), roll_out])
    
    # Tool 5: Stationarity (ADF)
    adf_btn = widgets.Button(description="Test Stationarity (ADF)", button_style=styles.BTN_PRIMARY)
    adf_out = widgets.Output()
    def _calc_adf(b):
        with adf_out:
            clear_output(wait=True)
            if not ts_col.value: return
            res = get_ts_stationarity(df, ts_col.value, ts_time_col.value)
            if "Error" in res:
                print(res["Error"])
            else:
                display(pd.DataFrame([res]).T.rename(columns={0: "Value"}))
    adf_btn.on_click(_calc_adf)
    tab5 = widgets.VBox([adf_btn, adf_out])
    
    ts_tabs = widgets.Tab(children=[tab1, tab2, tab3, tab4, tab5])
    for i, title in enumerate(["Basic Plots", "Decomposition", "ACF / PACF", "Rolling Stats", "Stationarity"]):
        ts_tabs.set_title(i, title)

    eda_ui.dynamic_ui.children = [widgets.VBox([global_selectors, ts_tabs])]
