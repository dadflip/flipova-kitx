import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from ml_pipeline.styles import styles
from ml_pipeline.steps.eda.logic.tabular import EDAVisualizerUtils

def _build_uni_tab(eda_ui, df, cols):
    eda_cfg = getattr(eda_ui.state, "config", {}).get("eda", {})
    eda_ui.uni_col  = widgets.Dropdown(options=cols, description="Variable:", layout=styles.LAYOUT_DD)
    eda_ui.uni_hue  = widgets.Dropdown(options=["None"] + cols, value="None", description="Hue:", layout=styles.LAYOUT_AUTO)
    eda_ui.uni_type = widgets.Dropdown(options=eda_cfg.get("univariate_plots", ["auto", "hist", "kde", "box", "violin", "bar", "pie"]),
                                      value="auto", description="Plot:", layout=styles.LAYOUT_AUTO)
    eda_ui.uni_bins = widgets.IntSlider(value=30, min=5, max=100, description="Bins/Top N:",
                                      layout=widgets.Layout(width="220px"),
                                      style={"description_width": "initial"})
    eda_ui.uni_kde  = widgets.Checkbox(value=True, description="KDE", layout=styles.LAYOUT_AUTO)
    eda_ui.uni_log  = widgets.Checkbox(value=False, description="Log scale", layout=widgets.Layout(width="105px"))
    eda_ui.uni_pal  = widgets.Dropdown(options=eda_cfg.get("palettes", ["Set2", "Set1", "viridis", "plasma", "coolwarm"]),
                                      value="Set2", description="Palette:", layout=styles.LAYOUT_AUTO)
    eda_ui.uni_btn      = widgets.Button(description="Plot", button_style=styles.BTN_PRIMARY, layout=styles.LAYOUT_BTN_STD)
    eda_ui.uni_save_btn = widgets.Button(description="Save to Dashboard", button_style="info", layout=styles.LAYOUT_BTN_STD)
    eda_ui.uni_out      = widgets.Output()
    eda_ui._last_uni_fig = None

    def _plot_uni(b):
        with eda_ui.uni_out:
            clear_output(wait=True)
            col    = eda_ui.uni_col.value
            kind   = eda_ui.meta[eda_ui.current_ds][col]["kind"]
            p_type = None if eda_ui.uni_type.value == "auto" else eda_ui.uni_type.value
            hue    = None if eda_ui.uni_hue.value == "None" else eda_ui.uni_hue.value
            fig = EDAVisualizerUtils.plot_univariate(
                df, col, kind, plot_type=p_type, bins=eda_ui.uni_bins.value,
                kde=eda_ui.uni_kde.value, log_scale=eda_ui.uni_log.value,
                hue=hue, palette=eda_ui.uni_pal.value)
            eda_ui._last_uni_fig = fig
            display(fig); plt.close(fig)

    def _save_uni(b):
        if eda_ui._last_uni_fig:
            title = f"Univariate: {eda_ui.uni_col.value}" + (f" by {eda_ui.uni_hue.value}" if eda_ui.uni_hue.value != 'None' else "")
            eda_ui.dashboard.add(eda_ui._last_uni_fig, title)

    eda_ui.uni_btn.on_click(_plot_uni)
    eda_ui.uni_save_btn.on_click(_save_uni)
    
    help_uni = styles.help_box(
        "<b>Numérique:</b> Histogramme / KDE / Box / Violin.<br>"
        "<b>Catégoriel:</b> Bar ou Pie — fréquences.<br>"
        "<b>Hue:</b> découpe par variable.", "#8b5cf6")
        
    return widgets.VBox([
        help_uni,
        widgets.HBox([eda_ui.uni_col, eda_ui.uni_hue, eda_ui.uni_type],
                      layout=widgets.Layout(align_items="flex-end", gap="10px")),
        widgets.HBox([eda_ui.uni_bins, eda_ui.uni_kde, eda_ui.uni_log, eda_ui.uni_pal],
                      layout=widgets.Layout(align_items="center", gap="10px", margin="6px 0 0 0")),
        widgets.HBox([eda_ui.uni_btn, eda_ui.uni_save_btn],
                      layout=widgets.Layout(align_items="center", gap="8px", margin="8px 0 0 0")),
        eda_ui.uni_out])

def _build_bi_tab(eda_ui, df, cols):
    eda_cfg = getattr(eda_ui.state, "config", {}).get("eda", {})
    eda_ui.bi_x     = widgets.Dropdown(options=cols, description="X:", layout=styles.LAYOUT_DD)
    eda_ui.bi_y     = widgets.Dropdown(options=cols, description="Y:", layout=styles.LAYOUT_DD)
    eda_ui.bi_type  = widgets.Dropdown(options=eda_cfg.get("bivariate_plots", ["auto", "scatter", "hexbin", "hist2d", "kde", "box", "violin", "strip", "swarm", "heatmap", "stacked_bar", "pie"]),
                                      value="auto", description="Plot:", layout=styles.LAYOUT_AUTO)
    eda_ui.bi_hue   = widgets.Dropdown(options=["None"] + cols, value="None", description="Hue:", layout=styles.LAYOUT_AUTO)
    eda_ui.bi_alpha = widgets.FloatSlider(value=0.6, min=0.1, max=1.0, step=0.1, description="Alpha:",
                                         layout=widgets.Layout(width="220px"),
                                         style={"description_width": "initial"}, readout_format=".1f")
    eda_ui.bi_pal   = widgets.Dropdown(options=eda_cfg.get("palettes", ["Set2", "Set1", "viridis"]),
                                      value="Set2", description="Palette:", layout=styles.LAYOUT_AUTO)
    eda_ui.bi_btn      = widgets.Button(description="Plot", button_style=styles.BTN_PRIMARY, layout=styles.LAYOUT_BTN_STD)
    eda_ui.bi_save_btn = widgets.Button(description="Save to Dashboard", button_style="info", layout=styles.LAYOUT_BTN_STD)
    eda_ui.bi_out      = widgets.Output()
    eda_ui._last_bi_fig = None

    def _plot_bi(b):
        with eda_ui.bi_out:
            clear_output(wait=True)
            x_col, y_col = eda_ui.bi_x.value, eda_ui.bi_y.value
            if x_col == y_col:
                print("[INFO] Sélectionnez deux variables différentes.")
                return
            x_kind = eda_ui.meta[eda_ui.current_ds][x_col]["kind"]
            y_kind = eda_ui.meta[eda_ui.current_ds][y_col]["kind"]
            p_type = None if eda_ui.bi_type.value == "auto" else eda_ui.bi_type.value
            h      = None if eda_ui.bi_hue.value == "None" else eda_ui.bi_hue.value
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fig = EDAVisualizerUtils.plot_bivariate(
                    df, x_col, y_col, x_kind, y_kind,
                    plot_type=p_type, hue=h,
                    alpha=eda_ui.bi_alpha.value, palette=eda_ui.bi_pal.value)
            eda_ui._last_bi_fig = fig
            display(fig); plt.close(fig)
            
    def _save_bi(b):
        if eda_ui._last_bi_fig:
            title = f"Bivariate: {eda_ui.bi_y.value} vs {eda_ui.bi_x.value}" + (f" by {eda_ui.bi_hue.value}" if eda_ui.bi_hue.value != 'None' else "")
            eda_ui.dashboard.add(eda_ui._last_bi_fig, title)

    eda_ui.bi_btn.on_click(_plot_bi)
    eda_ui.bi_save_btn.on_click(_save_bi)
    
    help_bi = styles.help_box(
        "<b>Num × Num:</b> Scatter, Hexbin, Hist2D, KDE.<br>"
        "<b>Num × Cat:</b> Box, Violin, Strip, Swarm.<br>"
        "<b>Cat × Cat:</b> Heatmap ou Stacked Bar.", "#10b981")
        
    return widgets.VBox([
        help_bi,
        widgets.HBox([eda_ui.bi_x, eda_ui.bi_y, eda_ui.bi_type],
                      layout=widgets.Layout(align_items="flex-end", gap="10px")),
        widgets.HBox([eda_ui.bi_hue, eda_ui.bi_pal, eda_ui.bi_alpha],
                      layout=widgets.Layout(align_items="flex-end", gap="10px", margin="6px 0 0 0")),
        widgets.HBox([eda_ui.bi_btn, eda_ui.bi_save_btn],
                      layout=widgets.Layout(align_items="center", gap="8px", margin="8px 0 0 0")),
        eda_ui.bi_out])
