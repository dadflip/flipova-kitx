import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
import numpy as np

from ml_pipeline.styles import styles
from ml_pipeline.steps.eda.logic.tabular import EDAVisualizerUtils
from .tabular_uni_bi import _build_uni_tab, _build_bi_tab
from .tabular_multi import _build_multi_tab
from .tabular_compare import _build_compare_tab

_SEP = widgets.HTML("<div style='height:1px;background:#e5e7eb;margin:10px 0;'></div>")

def build_tabular_ui(eda_ui, df: pd.DataFrame) -> None:
    eda_ui.tabs = widgets.Tab()
    cols = list(df.columns)

    # ── Tab 0 : Quality & Stats 
    eda_ui.out_recap = widgets.Output()
    enc_cfg = getattr(eda_ui.state, "config", {}).get("encoding", {})
    tabular_cfg = enc_cfg.get("tabular", enc_cfg)
    tabular_types = list(tabular_cfg.keys()) if isinstance(tabular_cfg, dict) else ["numeric", "categorical", "datetime", "binary", "id_like", "text"]
    
    eda_ui.type_col_dd  = widgets.Dropdown(options=cols, description="Column:", layout=styles.LAYOUT_DD)
    eda_ui.type_kind_dd = widgets.Dropdown(options=tabular_types, description="-> type:", layout=styles.LAYOUT_AUTO)
    eda_ui.type_btn     = widgets.Button(description="Update", button_style=styles.BTN_WARNING, layout=styles.LAYOUT_BTN_STD)
    eda_ui.type_msg     = widgets.HTML("")

    def _update_kind_dd(change):
        if eda_ui.type_col_dd.value and eda_ui.current_ds in eda_ui.meta:
            curr = eda_ui.meta[eda_ui.current_ds].get(eda_ui.type_col_dd.value, {}).get("kind", "")
            if curr in eda_ui.type_kind_dd.options:
                eda_ui.type_kind_dd.value = curr

    eda_ui.type_col_dd.observe(_update_kind_dd, names="value")
    if cols:
        _update_kind_dd(None)

    def _on_type_update(b):
        col = eda_ui.type_col_dd.value
        new_kind = eda_ui.type_kind_dd.value
        orig_key = eda_ui.current_ds.split(" ", 1)[1] if " " in eda_ui.current_ds else eda_ui.current_ds
        
        eda_ui.state.meta[orig_key][col]["kind"] = new_kind
        eda_ui.meta[eda_ui.current_ds][col]["kind"] = new_kind
        if hasattr(eda_ui.state, "log_step"):
            eda_ui.state.log_step("EDA", "Type Override", {"dataset": orig_key, "column": col, "new_kind": new_kind})
        
        eda_ui.type_msg.value = f"<span style='color:#059669;font-size:0.85em;'>[OK] '{col}' → {new_kind}</span>"
        _fill_recap_tab(eda_ui, df)

    eda_ui.type_btn.on_click(_on_type_update)
    eda_ui.btn_missing = widgets.Button(description="Plot Missing", button_style=styles.BTN_INFO, layout=styles.LAYOUT_BTN_STD)
    eda_ui.btn_missing.on_click(lambda b: _plot_missing(eda_ui, df))
    
    help_recap = styles.help_box("<b>Metadata:</b> Type detection, missing values, statistics.", "#3b82f6")
    type_row   = widgets.HBox([eda_ui.type_col_dd, eda_ui.type_kind_dd, eda_ui.type_btn, eda_ui.type_msg])
    tab_recap  = widgets.VBox([help_recap, type_row, _SEP, eda_ui.out_recap, eda_ui.btn_missing])

    # ── Tab 1 : Target 
    target_val = "(None)"
    bc = getattr(eda_ui.state, "business_context", {})
    if bc and bc.get("target") in cols:
        target_val = bc["target"]
    eda_ui.target_dd = widgets.Dropdown(options=["(None)"] + cols, value=target_val, description="Target:")
    eda_ui.target_feature_dd = widgets.Dropdown(options=cols, description="Feature:")
    eda_ui.target_btn = widgets.Button(description="Analyze", button_style=styles.BTN_PRIMARY)
    eda_ui.target_out = widgets.Output()

    def _on_target_change(change):
        if change["new"] and change["new"] != "(None)":
            if not hasattr(eda_ui.state, "business_context"):
                eda_ui.state.business_context = {}
            eda_ui.state.business_context["target"] = change["new"]
    
    eda_ui.target_dd.observe(_on_target_change, names="value")
    eda_ui.target_btn.on_click(lambda b: _plot_target_analysis(eda_ui, df))
    
    tab_target = widgets.VBox([
        styles.help_box("<b>Target Selection:</b> Choose the target for supervised learning analysis.", "#ec4899"),
        widgets.HBox([eda_ui.target_dd, eda_ui.target_feature_dd, eda_ui.target_btn]),
        eda_ui.target_out
    ])

    tab_uni = _build_uni_tab(eda_ui, df, cols)
    tab_bi = _build_bi_tab(eda_ui, df, cols)
    tab_multi = _build_multi_tab(eda_ui, df, cols)
    tab_compare, tabular_keys = _build_compare_tab(eda_ui)

    # ── Assemblage Simple pour les autres tabs (univarié, bivarié, temps, etc.) 
    tabs_list = [tab_recap, tab_target, tab_uni, tab_bi, tab_multi]
    titles = ["Quality & Stats", "Target Analysis", "Univariate", "Bivariate", "Multivariate"]
    
    if len(tabular_keys) > 1:
        tabs_list.append(tab_compare)
        titles.append("Compare Sets")
        
    eda_ui.tabs.children = tabs_list
    for i, title in enumerate(titles):
        eda_ui.tabs.set_title(i, title)
    
    eda_ui.dynamic_ui.children = [eda_ui.tabs]
    _fill_recap_tab(eda_ui, df)

def _fill_recap_tab(eda_ui, df: pd.DataFrame) -> None:
    from IPython.display import display, HTML
    with eda_ui.out_recap:
        clear_output()
        meta = eda_ui.meta.get(eda_ui.current_ds, {})
        rows = [{"Variable": c, "Type": m.get("kind", ""),
                 "Missing": f"{m.get('missing', 0)}",
                 "Unique": m.get("n_unique", 0), "Dtype": m.get("dtype", "")}
                for c, m in meta.items()]
        display(HTML("<b style='color:#374151;font-size:0.9em;'>Column Metadata</b>"))
        display(pd.DataFrame(rows).set_index("Variable"))
        
def _plot_missing(eda_ui, df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    with eda_ui.out_recap:
        missing = df.isna().sum()
        missing = missing[missing > 0]
        if missing.empty:
            print("[OK] No missing values found.")
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        missing.sort_values(ascending=False).plot(kind="bar", color="#ef4444", ax=ax)
        ax.set_title("Missing Values per Feature")
        plt.tight_layout()
        display(fig)
        eda_ui.dashboard.add(fig, "Missing Values")
        plt.close(fig)

def _plot_target_analysis(eda_ui, df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import warnings
    with eda_ui.target_out:
        clear_output(wait=True)
        target = eda_ui.target_dd.value
        feature = eda_ui.target_feature_dd.value
        if target == "(None)" or not feature or target == feature:
            print("[INFO] Invalid selection.")
            return
            
        t_kind = eda_ui.meta[eda_ui.current_ds].get(target, {}).get("kind", "categorical")
        f_kind = eda_ui.meta[eda_ui.current_ds].get(feature, {}).get("kind", "categorical")
        
        if t_kind in ("categorical", "binary") and f_kind in ("numeric", "timeseries"):
            p_type = "box"
        elif t_kind in ("numeric", "timeseries") and f_kind in ("categorical", "binary"):
            p_type = "box"
        elif t_kind in ("numeric", "timeseries") and f_kind in ("numeric", "timeseries"):
            p_type = "scatter"
        else:
            p_type = "stacked_bar"
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = EDAVisualizerUtils.plot_bivariate(
                df, feature, target, f_kind, t_kind, plot_type=p_type,
                hue=target if t_kind in ("categorical", "binary") else None,
                alpha=0.7, palette="Set2")
        display(fig)
        eda_ui.dashboard.add(fig, f"Target: {feature} vs {target}")
        plt.close(fig)
