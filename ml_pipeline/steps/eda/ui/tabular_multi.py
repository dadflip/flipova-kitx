import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from ml_pipeline.styles import styles

def _build_multi_tab(eda_ui, df, cols):
    eda_cfg = getattr(eda_ui.state, "config", {}).get("eda", {})
    eda_ui.multi_type = widgets.Dropdown(
        options=eda_cfg.get("multivariate", ["Correlation Matrix", "Pairplot"]),
        value="Correlation Matrix", description="Analysis:", layout=styles.LAYOUT_DD)
    eda_ui.multi_hue  = widgets.Dropdown(options=["None"] + cols, value="None", description="Hue:", layout=styles.LAYOUT_AUTO)
    eda_ui.multi_corr = widgets.Dropdown(
        options=eda_cfg.get("correlation_methods", ["pearson", "spearman", "kendall"]),
        value="pearson", description="Method:", layout=styles.LAYOUT_AUTO)
        
    meta = eda_ui.meta[eda_ui.current_ds]
    available_kinds = set()
    eda_ui.multi_col_boxes = {}
    for c in cols:
        k = meta[c]["kind"]
        available_kinds.add(k)
        eda_ui.multi_col_boxes[c] = widgets.Checkbox(
            value=bool(pd.api.types.is_numeric_dtype(df[c])),
            description=f"{c} [{k}]", style={"description_width": "initial"},
            layout=widgets.Layout(width="auto", margin="1px 4px"))
            
    btn_layout = widgets.Layout(width="auto", height="26px", margin="2px")
    select_btns = []
    for k in sorted(available_kinds):
        btn = widgets.Button(description=f"All {k}", button_style="info", layout=btn_layout)
        def _make_selector(kind):
            def _select(*a):
                for cn, cb in eda_ui.multi_col_boxes.items():
                    cb.value = (meta[cn]["kind"] == kind)
            return _select
        btn.on_click(_make_selector(k))
        select_btns.append(btn)
        
    btn_none = widgets.Button(description="Clear all", button_style="warning", layout=btn_layout)
    btn_none.on_click(lambda *a: [setattr(cb, "value", False) for cb in eda_ui.multi_col_boxes.values()])
    select_btns.append(btn_none)
    
    col_boxes_grid = widgets.HBox(list(eda_ui.multi_col_boxes.values()), layout=widgets.Layout(flex_wrap="wrap", gap="4px"))
    multi_cols_container = widgets.VBox([
        widgets.HTML("<div style='font-size:0.82em;font-weight:600;color:#6b7280;margin-bottom:4px;'>COLONNES</div>"),
        widgets.HBox(select_btns, layout=widgets.Layout(flex_wrap="wrap", margin="4px 0 8px 0", gap="4px")),
        col_boxes_grid],
        layout=widgets.Layout(border="1px solid #e5e7eb", border_radius="6px", padding="10px", margin="8px 0"))
        
    eda_ui.multi_btn      = widgets.Button(description="Generate Plot", button_style=styles.BTN_PRIMARY, layout=widgets.Layout(width="160px", height="32px"))
    eda_ui.multi_save_btn = widgets.Button(description="Save to Dashboard", button_style="info", layout=styles.LAYOUT_BTN_STD)
    eda_ui.multi_out      = widgets.Output()
    eda_ui._last_multi_fig = None

    def _plot_multivariate(b):
        with eda_ui.multi_out:
            clear_output(wait=True)
            m_type = eda_ui.multi_type.value
            sel_cols = [c for c, cb in eda_ui.multi_col_boxes.items() if cb.value]
            if not sel_cols:
                print("[INFO] Sélectionnez au moins une colonne.")
                return
            fig = None
            if m_type == "Correlation Matrix":
                num_cols = [c for c in sel_cols if pd.api.types.is_numeric_dtype(df[c])]
                if len(num_cols) < 2:
                    print(f"[INFO] Besoin d'au moins 2 colonnes numériques (obtenu {len(num_cols)}).")
                    return
                corr = df[num_cols].corr(method=eda_ui.multi_corr.value)
                fig, ax = plt.subplots(figsize=(min(max(len(num_cols) * 0.8, 8), 16),
                                                min(max(len(num_cols) * 0.6, 6), 12)))
                fig.patch.set_facecolor("#f8fafc")
                sns.heatmap(corr, annot=len(num_cols) <= 12, cmap="coolwarm",
                            fmt=".2f", center=0, square=True, ax=ax)
                ax.set_title(f"{eda_ui.multi_corr.value.capitalize()} Correlation Matrix")
                plt.tight_layout()
                display(fig); plt.close(fig)
            elif m_type == "Pairplot":
                if len(sel_cols) > 10:
                    print("[WARNING] Plus de 10 colonnes — utilisation des 10 premières.")
                    sel_cols = sel_cols[:10]
                h = None if eda_ui.multi_hue.value == "None" else eda_ui.multi_hue.value
                cols_to_plot = sel_cols + [h] if (h and h not in sel_cols) else sel_cols
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    g = sns.pairplot(df[cols_to_plot].dropna(), hue=h, palette="Set2")
                    fig = g.fig
                    display(fig); plt.close(fig)
            eda_ui._last_multi_fig = fig

    def _save_multi(b):
        if eda_ui._last_multi_fig:
            eda_ui.dashboard.add(eda_ui._last_multi_fig, f"Multivariate: {eda_ui.multi_type.value}")

    eda_ui.multi_btn.on_click(_plot_multivariate)
    eda_ui.multi_save_btn.on_click(_save_multi)
    
    help_multi = styles.help_box(
        "<b>Correlation Matrix:</b> corrélations pairées.<br>"
        "<b>Pairplot:</b> scatter + histogrammes (lent avec &gt;10 colonnes).", "#f59e0b")
        
    return widgets.VBox([
        help_multi,
        widgets.HBox([eda_ui.multi_type, eda_ui.multi_hue, eda_ui.multi_corr],
                      layout=widgets.Layout(align_items="flex-end", gap="10px", margin="0 0 4px 0", flex_wrap="wrap")),
        multi_cols_container,
        widgets.HBox([eda_ui.multi_btn, eda_ui.multi_save_btn],
                      layout=widgets.Layout(align_items="center", gap="8px", margin="4px 0 0 0")),
        eda_ui.multi_out])
