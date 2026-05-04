import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ml_pipeline.styles import styles

def _build_compare_tab(eda_ui):
    tab_compare = widgets.VBox([])
    tabular_keys = [k for k, v in eda_ui.all_datasets.items() if isinstance(v, pd.DataFrame)]
    if len(tabular_keys) > 1:
        eda_ui.comp_ds1 = widgets.Dropdown(options=tabular_keys, value=eda_ui.current_ds, description="DS 1:", layout=styles.LAYOUT_DD)
        eda_ui.comp_ds2 = widgets.Dropdown(
            options=tabular_keys,
            value=tabular_keys[1] if tabular_keys[0] == eda_ui.current_ds else tabular_keys[0],
            description="DS 2:", layout=styles.LAYOUT_DD)
        eda_ui.comp_col = widgets.Dropdown(options=[], description="Col:", layout=styles.LAYOUT_DD)

        def _update_comp_cols(*args):
            df1 = eda_ui.all_datasets[eda_ui.comp_ds1.value]
            df2 = eda_ui.all_datasets[eda_ui.comp_ds2.value]
            shared = [c for c in df1.columns if c in df2.columns]
            eda_ui.comp_col.options = shared
            if shared and eda_ui.comp_col.value not in shared:
                eda_ui.comp_col.value = shared[0]

        eda_ui.comp_ds1.observe(_update_comp_cols, names="value")
        eda_ui.comp_ds2.observe(_update_comp_cols, names="value")
        _update_comp_cols()
        
        eda_ui.comp_btn      = widgets.Button(description="Compare Drift", button_style=styles.BTN_PRIMARY, layout=styles.LAYOUT_BTN_STD)
        eda_ui.comp_save_btn = widgets.Button(description="Save to Dashboard", button_style="info", layout=styles.LAYOUT_BTN_STD)
        eda_ui.comp_out      = widgets.Output()
        eda_ui._last_comp_fig = None

        def _plot_comparison(b):
            with eda_ui.comp_out:
                clear_output(wait=True)
                col = eda_ui.comp_col.value
                if not col: return
                df1_name, df2_name = eda_ui.comp_ds1.value, eda_ui.comp_ds2.value
                df1 = eda_ui.all_datasets[df1_name][col].dropna()
                df2 = eda_ui.all_datasets[df2_name][col].dropna()
                is_num = pd.api.types.is_numeric_dtype(df1) and pd.api.types.is_numeric_dtype(df2)
                fig, ax = plt.subplots(figsize=(10, 6))
                fig.patch.set_facecolor("#f8fafc"); ax.set_facecolor("#f8fafc")
                if is_num:
                    sns.kdeplot(df1, fill=True, label=df1_name, color="#3b82f6", alpha=0.5, ax=ax)
                    sns.kdeplot(df2, fill=True, label=df2_name, color="#ef4444", alpha=0.5, ax=ax)
                    ax.set_title(f"Density Drift for '{col}': {df1_name} vs {df2_name}")
                    ax.set_ylabel("Density")
                else:
                    v1 = df1.value_counts(normalize=True).rename(df1_name)
                    v2 = df2.value_counts(normalize=True).rename(df2_name)
                    comp = pd.concat([v1, v2], axis=1).fillna(0).head(15)
                    comp.plot(kind="bar", ax=ax, color=["#3b82f6", "#ef4444"])
                    ax.set_title(f"Category Distribution Drift for '{col}'")
                    ax.set_ylabel("Proportion")
                ax.set_xlabel(col); ax.legend()
                plt.tight_layout()
                display(fig); plt.close(fig)
                eda_ui._last_comp_fig = fig

        def _save_comp(b):
            if eda_ui._last_comp_fig:
                eda_ui.dashboard.add(eda_ui._last_comp_fig, f"Drift: {eda_ui.comp_col.value} — {eda_ui.comp_ds1.value} vs {eda_ui.comp_ds2.value}")

        eda_ui.comp_btn.on_click(_plot_comparison)
        eda_ui.comp_save_btn.on_click(_save_comp)
        
        help_comp = styles.help_box("Compare les distributions entre deux datasets (ex. Train vs Test). Détecte le <b>Data Drift</b>.", "#ef4444")
        tab_compare.children = [
            help_comp,
            widgets.HBox([eda_ui.comp_ds1, eda_ui.comp_ds2, eda_ui.comp_col, eda_ui.comp_btn, eda_ui.comp_save_btn],
                          layout=widgets.Layout(gap="10px", flex_wrap="wrap", align_items="center")),
            eda_ui.comp_out]
            
    return tab_compare, tabular_keys
