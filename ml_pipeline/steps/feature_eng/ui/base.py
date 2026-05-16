import math
import traceback
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from ml_pipeline.styles import styles

from ..logic.operations import (
    apply_math, apply_condition, run_formula, 
    apply_text, apply_date, apply_binning,
    create_viz_fig, create_dashboard_fig
)

_TAB_CSS_INJECTED = False

def _inject_tab_css() -> None:
    global _TAB_CSS_INJECTED
    if _TAB_CSS_INJECTED:
        return
    _TAB_CSS_INJECTED = True
    display(HTML("""<style>
    .fe-tabs .jupyter-widgets-tab-bar {
        display:flex !important; flex-wrap:wrap !important;
        border-bottom:2px solid #e2e8f0 !important; gap:4px !important; 
        padding: 4px 4px 0 4px !important; background: #f1f5f9 !important;
        border-radius: 8px 8px 0 0 !important;
    }
    .fe-tabs .jupyter-widgets-tab-bar .p-TabBar-tab {
        flex-shrink:0 !important; white-space:nowrap !important;
        font-size:0.82em !important; padding:8px 16px !important;
        border-radius:8px 8px 0 0 !important; border:1px solid transparent !important;
        background:none !important; color:#64748b !important; font-weight:500 !important; 
        transition: all 0.2s ease;
    }
    .fe-tabs .jupyter-widgets-tab-bar .p-TabBar-tab:hover {
        background: rgba(255,255,255,0.5) !important; color: #334155 !important;
    }
    .fe-tabs .jupyter-widgets-tab-bar .p-TabBar-tab.p-mod-current {
        background:#ffffff !important; color:#6366f1 !important;
        border-color:#e2e8f0 !important; border-bottom-color:#ffffff !important;
        font-weight:700 !important; box-shadow: 0 -2px 5px rgba(0,0,0,0.02) !important; }
    
    .fe-section-title {
        font-size: 0.9em; font-weight: 700; color: #1e293b; margin: 15px 0 8px 0;
        display: flex; align-items: center; gap: 8px;
    }
    .fe-control-group {
        background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; margin-bottom: 10px;
    }
    
    /* Enhanced Table Styles */
    .rendered_html table { border-collapse: collapse; border: none; font-variant-numeric: tabular-nums; width: 100%; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .rendered_html th { background: #f1f5f9 !important; color: #475569 !important; font-weight: 600 !important; border: 1px solid #e2e8f0 !important; padding: 10px !important; }
    .rendered_html td { border: 1px solid #f1f5f9 !important; padding: 8px 12px !important; font-size: 0.9em !important; }
    .rendered_html tr:nth-child(even) { background: #f8fafc; }
    .rendered_html tr:hover { background: #f1f5f9 !important; }
    </style>"""))

class TabularFeatureEngUI:
    """Interface Feature Engineering Numérique/Tabulaire."""

    def __init__(self, state):
        self.state = state
        if not self.state.config:
            self.ui = styles.error_msg("[ERROR] Configuration non chargée.")
            return
            
        # Get tabular dict recursively or not, depending on where it's stored.
        # Check either data_raw dict, or if data loader returned nested dicts (like folders)
        tabular_ds = {}
        for k, v in self.state.data_raw.items():
            if isinstance(v, pd.DataFrame):
                tabular_ds[k] = v
            elif isinstance(v, dict):
                # could be local folder...
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, pd.DataFrame):
                        tabular_ds[f"{k} - {sub_k}"] = sub_v
                        
        self.tabular_datasets = tabular_ds
        if not self.tabular_datasets:
            self.ui = styles.error_msg("Aucun dataset tabulaire disponible.")
            return
            
        self.current_ds = list(self.tabular_datasets.keys())[0]
        _inject_tab_css()
        self._build_ui()

    def _get_df(self) -> pd.DataFrame:
        return self.tabular_datasets[self.current_ds]

    def _propagate_to_state(self) -> None:
        ds = self.current_ds
        df = self.tabular_datasets[ds]
        
        # Determine actual key for nested dicts
        target_dict = self.state.data_raw
        key = ds
        # Simplify propagation: just put it in data_raw flat or try to map it back if needed
        # (For now, we'll assign it to data_raw directly, which handles both normal and "folder - file" naming loosely)
        if " - " in ds and ds not in target_dict:
            parent, child = ds.split(" - ", 1)
            if parent in target_dict and isinstance(target_dict[parent], dict):
                target_dict[parent][child] = df
            else:
                target_dict[ds] = df
        else:
            target_dict[ds] = df
            
        if hasattr(self.state, "data_cleaned"):
            key2 = ds
            if " - " in ds and ds not in self.state.data_cleaned:
                parent, child = ds.split(" - ", 1)
                if parent in self.state.data_cleaned and isinstance(self.state.data_cleaned[parent], dict):
                    self.state.data_cleaned[parent][child] = df.copy()
                else:
                    self.state.data_cleaned[ds] = df.copy()
            else:
                self.state.data_cleaned[ds] = df.copy()
                
        if not hasattr(self.state, "meta"):
            self.state.meta = {}
        if ds not in self.state.meta:
            self.state.meta[ds] = {}
            
        meta_ds = self.state.meta[ds]
        for col in list(meta_ds.keys()):
            if col not in df.columns:
                del meta_ds[col]
        for col in df.columns:
            if col not in meta_ds:
                s = df[col]; n_unq = s.nunique()
                if pd.api.types.is_datetime64_any_dtype(s): kind = "datetime"
                elif pd.api.types.is_bool_dtype(s) or n_unq == 2: kind = "binary"
                elif pd.api.types.is_numeric_dtype(s):
                    kind = "id_like" if n_unq / max(len(s), 1) > 0.95 else "numeric"
                else:
                    kind = "categorical" if n_unq < 100 else "text"
                meta_ds[col] = {"kind": kind}

    def _sync_targets(self, op_fn, op_label: str) -> None:
        if not self.sync_check.value:
            return
        targets = list(self.sync_datasets.value)
        lines = []
        for ds_name in targets:
            if ds_name == self.current_ds:
                continue
            df_target = self.tabular_datasets.get(ds_name)
            if df_target is None:
                lines.append(f"<span style='color:#ef4444;'>⚠ '{ds_name}' introuvable.</span>")
                continue
            try:
                df_result = op_fn(df_target.copy())
                if df_result is not None:
                    self.tabular_datasets[ds_name] = df_result
                    
                    # Store back safely
                    if " - " in ds_name and ds_name not in self.state.data_raw:
                        parent, child = ds_name.split(" - ", 1)
                        if parent in self.state.data_raw and isinstance(self.state.data_raw[parent], dict):
                            self.state.data_raw[parent][child] = df_result
                        else:
                            self.state.data_raw[ds_name] = df_result
                    else:
                        self.state.data_raw[ds_name] = df_result
                        
                    if hasattr(self.state, "data_cleaned"):
                        if " - " in ds_name and ds_name not in self.state.data_cleaned:
                            parent, child = ds_name.split(" - ", 1)
                            if parent in self.state.data_cleaned and isinstance(self.state.data_cleaned[parent], dict):
                                self.state.data_cleaned[parent][child] = df_result.copy()
                            else:
                                self.state.data_cleaned[ds_name] = df_result.copy()
                        else:
                            self.state.data_cleaned[ds_name] = df_result.copy()
                            
                    lines.append(f"<span style='color:#10b981;'>✓ '{ds_name}' — {op_label}</span>")
                else:
                    lines.append(f"<span style='color:#94a3b8;'>– '{ds_name}' — aucune modification.</span>")
            except Exception as e:
                lines.append(f"<span style='color:#ef4444;'>⚠ '{ds_name}' — {e}</span>")
        if lines:
            with self.sync_status:
                clear_output(wait=True)
                display(HTML("<div style='font-size:0.79em;padding:4px 0;'><b style='color:#6366f1;'>Sync :</b> "
                              + "  ·  ".join(lines) + "</div>"))

    def _build_ui(self) -> None:
        header  = widgets.HTML(styles.card_html("Feature Engineering", "Advanced Variable Laboratory", ""))
        top_bar = widgets.HBox([header], layout=widgets.Layout(
            align_items="center", margin="0 0 12px 0",
            padding="0 0 10px 0", border_bottom="2px solid #ede9fe"))
        self.ds_dd = widgets.Dropdown(options=list(self.tabular_datasets.keys()),
                                       value=self.current_ds, description="Dataset:",
                                       layout=styles.LAYOUT_DD_LONG)
        self.ds_dd.observe(self._on_ds_change, names="value")
        other_ds = [k for k in self.tabular_datasets if k != self.current_ds]
        self.sync_datasets = widgets.SelectMultiple(
            options=other_ds, value=other_ds, description="Sync vers :",
            layout=widgets.Layout(width="280px", height=f"{max(36, min(120, len(other_ds)*24))}px"),
            disabled=not bool(other_ds))
        self.sync_check = widgets.Checkbox(value=bool(other_ds),
                                            description="Répliquer sur d'autres datasets",
                                            layout=widgets.Layout(width="280px"),
                                            disabled=not bool(other_ds))
        self.sync_check.observe(lambda c: setattr(self.sync_datasets, "disabled", not c["new"]), names="value")
        self.sync_status = widgets.Output()
        self._sync_box = widgets.VBox([
            widgets.HBox([self.sync_check, self.sync_datasets],
                          layout=widgets.Layout(align_items="flex-start", gap="12px")),
            self.sync_status],
            layout=widgets.Layout(padding="8px 10px", margin="6px 0",
                                   border="1px solid #e0e7ff", border_radius="6px",
                                   background_color="#f5f3ff"))
        self.tabs = widgets.Tab()
        self._build_preview_tab()
        self._build_math_tab()
        self._build_condition_tab()
        self._build_formula_tab()
        self._build_text_tab()
        self._build_date_tab()
        self._build_binning_tab()
        self._build_viz_tab()
        self._build_dashboard_tab()
        self._build_manage_tab()
        self.tabs.children = [self.tab_preview, self.tab_math, self.tab_condition,
                               self.tab_formula, self.tab_text, self.tab_date,
                               self.tab_binning, self.tab_viz, self.tab_dashboard, self.tab_manage]
        for i, t in enumerate(["Data Preview", "Math & Logic", "Conditions", "Custom Formula",
                                "Text Operations", "Date / Time", "Binning",
                                "Visualization", "Target Dashboard", "Manage Columns"]):
            self.tabs.set_title(i, t)
        self.tabs.add_class("fe-tabs")
        self.ui = widgets.VBox(
            [top_bar, self.ds_dd, self._sync_box,
             widgets.HTML("<hr style='border:1px solid #f1f5f9;margin:10px 0;'>"),
             self.tabs],
            layout=widgets.Layout(width="100%", max_width="1000px",
                                   border="1px solid #e5e7eb", padding="18px",
                                   border_radius="10px", background_color="#ffffff"))

    def _refresh_columns(self) -> None:
        cols = list(self._get_df().columns)
        for w in (self.math_col1, self.text_col, self.date_col, self.bin_col,
                  self.viz_x, self.viz_y, self.dash_target, self.cond_col):
            w.options = cols
        self.math_col2.options = ["(None)", "(Constant)"] + cols
        self.viz_hue.options   = ["(None)"] + cols
        self.dash_target.options = ["(None)"] + cols
        self.dash_features.options = cols
        self.cond_then_col.options = ["(Constant)"] + cols
        self.cond_else_col.options = ["(Constant)"] + cols
        self._refresh_formula_col_list()
        if hasattr(self, "manage_col"):
            self.manage_col.options = cols
        self._refresh_preview_col_selector()

    def _on_ds_change(self, change) -> None:
        if change["new"]:
            self.current_ds = change["new"]
            other_ds = [k for k in self.tabular_datasets if k != self.current_ds]
            self.sync_datasets.options = other_ds
            self.sync_datasets.value   = other_ds
            self.sync_check.disabled   = not bool(other_ds)
            self.sync_datasets.disabled = not (bool(other_ds) and self.sync_check.value)
            self._refresh_columns()
            self.viz_out.clear_output()
            if hasattr(self, "dash_out"):
                self.dash_out.clear_output()
            self._render_preview()

    def _notify(self, out_widget, msg: str, is_error: bool = False) -> None:
        color = "#ef4444" if is_error else "#10b981"
        tag   = "[ERROR]" if is_error else "[OK]"
        with out_widget:
            clear_output(wait=True)
            display(HTML(f"<div style='color:{color};font-weight:bold;font-size:0.85em;'>{tag} {msg}</div>"))

    def _section_title(self, text: str, icon: str = "Step", color: str = "#6366f1") -> widgets.HTML:
        return widgets.HTML(f"<div class='fe-section-title' style='color:{color};'><span>{icon}</span> {text}</div>")

    # ── Preview tab ───────────────────────────────────────────────────────────
    def _build_preview_tab(self) -> None:
        df = self._get_df(); cols = list(df.columns)
        self.preview_rows = widgets.IntSlider(value=20, min=5, max=500, step=5, description="Lignes:", layout=widgets.Layout(width="280px"))
        self.preview_search = widgets.Text(placeholder='Filtre: col == "val" or age > 20', description="Filtrer:", layout=widgets.Layout(width="400px"))
        self.preview_col_select = widgets.SelectMultiple(options=cols, value=cols[:min(len(cols), 30)], description="Colonnes:", layout=widgets.Layout(width="280px", height="120px"))
        self.preview_col_all_btn = widgets.Button(description="Toutes", button_style="info", layout=widgets.Layout(width="80px", height="28px"))
        self.preview_col_new_btn = widgets.Button(description="Nouvelles", layout=widgets.Layout(width="100px", height="28px"))
        self.preview_highlight_new = widgets.Checkbox(value=True, description="Surbrillance (New)", layout=widgets.Layout(width="160px"))
        self.preview_show_stats = widgets.Checkbox(value=False, description="Stats rapides", layout=widgets.Layout(width="130px"))
        self.preview_sort_col = widgets.Dropdown(options=["(none)"] + cols, value="(none)", description="Trier par:", layout=widgets.Layout(width="220px"))
        self.preview_sort_asc = widgets.Checkbox(value=True, description="Ascendant", layout=widgets.Layout(width="100px"))
        self.preview_refresh_btn = widgets.Button(description="Actualiser", button_style="primary", layout=widgets.Layout(width="120px", height="34px"))
        self.preview_out = widgets.Output()
        
        self._preview_original_cols = set(cols)
        self.preview_col_all_btn.on_click(lambda _: setattr(self.preview_col_select, "value", list(self.preview_col_select.options)))
        self.preview_col_new_btn.on_click(lambda _: setattr(self.preview_col_select, "value",
            [c for c in self._get_df().columns if c not in self._preview_original_cols] or list(self.preview_col_select.options)))
        self.preview_refresh_btn.on_click(lambda _: self._render_preview())
        
        for w in (self.preview_rows, self.preview_search, self.preview_sort_col, self.preview_sort_asc,
                  self.preview_highlight_new, self.preview_show_stats, self.preview_col_select):
            w.observe(lambda c: self._render_preview(), names="value")
            
        self.tab_preview = widgets.VBox([
            styles.help_box("<b>Exploration Interactive</b> — Visualisez le dataset avec tri, filtres et indicateurs.", "#0ea5e9"),
            self._section_title("Controles de la Vue", "View", "#0ea5e9"),
            widgets.VBox([
                widgets.HBox([self.preview_rows, self.preview_sort_col, self.preview_sort_asc], layout=widgets.Layout(gap="10px", margin="0 0 10px 0")),
                widgets.HBox([self.preview_search], layout=widgets.Layout(margin="0 0 10px 0")),
                widgets.HBox([
                    self.preview_col_select, 
                    widgets.VBox([self.preview_col_all_btn, self.preview_col_new_btn], layout=widgets.Layout(gap="5px", margin="0 0 0 10px"))
                ], layout=widgets.Layout(align_items="flex-start", margin="0 0 10px 0")),
                widgets.HBox([self.preview_highlight_new, self.preview_show_stats, self.preview_refresh_btn], layout=widgets.Layout(align_items="center", gap="15px")),
            ], layout=widgets.Layout(padding="15px", border="1px solid #e2e8f0", border_radius="8px", background_color="#f8fafc")),
            self.preview_out], layout=widgets.Layout(padding="12px"))
        self._render_preview()

    def _refresh_preview_col_selector(self) -> None:
        df = self._get_df(); cols = list(df.columns)
        current_val = list(self.preview_col_select.value)
        self.preview_col_select.options = cols
        still_valid = [c for c in current_val if c in cols]
        new_cols = [c for c in cols if c not in self._preview_original_cols]
        merged = still_valid + [c for c in new_cols if c not in still_valid]
        self.preview_col_select.value = merged if merged else cols[:min(len(cols), 30)]
        self.preview_sort_col.options = ["(none)"] + cols

    def _render_preview(self) -> None:
        df = self._get_df().copy()
        selected_cols = list(self.preview_col_select.value) or list(df.columns)
        filter_expr = self.preview_search.value.strip()
        filter_error = None
        if filter_expr:
            try: df = df.query(filter_expr)
            except Exception:
                try: df = df[df.eval(filter_expr)]
                except Exception as e: filter_error = str(e)
        sort_col = self.preview_sort_col.value
        if sort_col and sort_col != "(none)" and sort_col in df.columns:
            df = df.sort_values(sort_col, ascending=self.preview_sort_asc.value)
        n_rows = self.preview_rows.value
        view = df[selected_cols].head(n_rows)
        new_cols = set(self._get_df().columns) - self._preview_original_cols
        with self.preview_out:
            clear_output(wait=True)
            meta_parts = [f"<b>{len(df):,}</b> rows", f"<b>{len(self._get_df().columns)}</b> cols",
                          f"<b>{len(new_cols)}</b> new"]
            if filter_error: meta_parts.append(f"<span style='color:#ef4444;'>filter error: {filter_error}</span>")
            display(HTML("<div style='font-size:0.82em;color:#64748b;margin-bottom:8px;padding:6px 10px;background:#f8fafc;border-radius:6px;border:1px solid #e2e8f0;'>" + "  ·  ".join(meta_parts) + "</div>"))
            ts = [{"selector": "thead th", "props": [("background-color","#f1f5f9"),("font-size","0.8em"),("padding","6px 10px")]},
                  {"selector": "td", "props": [("font-size","0.82em"),("padding","4px 10px"),("max-width","200px"),("overflow","hidden"),("text-overflow","ellipsis"),("white-space","nowrap")]}]
            if self.preview_highlight_new.value and new_cols:
                def _hl(col):
                    return ["background-color:#fef9c3;font-weight:600;" if col.name in new_cols else "" for _ in col]
                display(view.style.apply(_hl, axis=0).set_table_styles(ts).format(precision=4, na_rep="—"))
            else:
                display(view.style.set_table_styles(ts).format(precision=4, na_rep="—"))
            if len(df) > n_rows:
                display(HTML(f"<div style='font-size:0.78em;color:#94a3b8;margin-top:8px;text-align:right;'>Showing {n_rows} of {len(df):,} rows</div>"))

    # ── Math tab ──────────────────────────────────────────────────────────────
    def _build_math_tab(self) -> None:
        cols = list(self._get_df().columns)
        math_ops = self.state.config.get("feature_engineering", {}).get("math_operations", ["+","-","*","/","log(A)","exp(A)","sqrt(A)","A^2","Abs(A)","Modulo"])
        self.math_col1  = widgets.Dropdown(options=cols, description="Col A:", layout=widgets.Layout(width="280px"))
        self.math_op    = widgets.Dropdown(options=math_ops, value=math_ops[0], description="Opération:", layout=widgets.Layout(width="180px"))
        self.math_col2  = widgets.Dropdown(options=["(None)","(Constant)"]+cols, description="Col B:", layout=widgets.Layout(width="280px"))
        self.math_const = widgets.FloatText(value=0, description="Valeur:", layout=widgets.Layout(width="150px", display="none"))
        self.math_new_col = widgets.Text(description="Résultat:", placeholder="nom_colonne", layout=widgets.Layout(width="280px"))
        self.math_btn   = widgets.Button(description="Exécuter", button_style=styles.BTN_PRIMARY, layout=widgets.Layout(width="140px", height="34px"))
        self.math_out   = widgets.Output()
        
        def _on_op(c):
            hide = c["new"] in ("log(A)","exp(A)","sqrt(A)","A^2","Abs(A)")
            self.math_col2.layout.display = "none" if hide else "flex"
            self.math_const.layout.display = "none" if hide else ("flex" if self.math_col2.value == "(Constant)" else "none")
            
        self.math_op.observe(_on_op, names="value")
        self.math_col2.observe(lambda c: setattr(self.math_const.layout, "display", "flex" if c["new"] == "(Constant)" else "none"), names="value")
        self.math_btn.on_click(self._apply_math)
        
        self.tab_math = widgets.VBox([
            styles.help_box("<b>Math et Logique</b> — Calculs arithmetiques simples entre colonnes numeriques.", "#8b5cf6"),
            self._section_title("Configuration du calcul", "Math", "#8b5cf6"),
            widgets.VBox([
                widgets.HBox([self.math_col1, self.math_op, self.math_col2, self.math_const], layout=widgets.Layout(gap="10px", margin="0 0 10px 0")),
                widgets.HBox([self.math_new_col, self.math_btn], layout=widgets.Layout(gap="10px"))
            ], layout=widgets.Layout(padding="15px", border="1px solid #e2e8f0", border_radius="8px", background_color="#f8fafc")),
            self.math_out], layout=widgets.Layout(padding="12px"))

    def _apply_math(self, _) -> None:
        df = self._get_df(); col1 = self.math_col1.value; col2 = self.math_col2.value
        op = self.math_op.value; new_name = self.math_new_col.value or f"{col1}_new"
        const_val = self.math_const.value
        try:
            res_df, final_name = apply_math(df, col1, col2, op, const_val, new_name)
            self.tabular_datasets[self.current_ds] = res_df
            self.state.log_step("Feature Eng", "Math applied", {"op": op, "new_col": final_name})
            self._propagate_to_state()
            
            def _sync(df_t):
                df_res, _ = apply_math(df_t, col1, col2, op, const_val, new_name)
                return df_res
            self._sync_targets(_sync, f"Math {op} → '{final_name}'")
            self._refresh_columns(); self._notify(self.math_out, f"Created '{final_name}'"); self._render_preview()
        except Exception as e:
            self._notify(self.math_out, str(e), True)

    # ── Condition tab ─────────────────────────────────────────────────────────
    def _build_condition_tab(self) -> None:
        cols = list(self._get_df().columns)
        ops_list = ["==","!=",">",">=","<","<=","isin","not isin","is null","is not null","contains (str)","startswith","endswith"]
        self.cond_col      = widgets.Dropdown(options=cols, description="Si colonne:", layout=widgets.Layout(width="250px"))
        self.cond_op       = widgets.Dropdown(options=ops_list, description="Est:", layout=widgets.Layout(width="180px"))
        self.cond_val      = widgets.Text(description="Valeur:", placeholder="ex: -1 ou v1,v2", layout=widgets.Layout(width="220px"))
        self.cond_then_col = widgets.Dropdown(options=["(Constant)"]+cols, value="(Constant)", description="ALORS:", layout=widgets.Layout(width="250px"))
        self.cond_then_val = widgets.Text(value="1", description="Valeur:", layout=widgets.Layout(width="150px"))
        self.cond_else_col = widgets.Dropdown(options=["(Constant)"]+cols, value="(Constant)", description="SINON:", layout=widgets.Layout(width="250px"))
        self.cond_else_val = widgets.Text(value="0", description="Valeur:", layout=widgets.Layout(width="150px"))
        self.cond_combine  = widgets.Dropdown(options=["AND","OR"], value="AND", description="Logique:", layout=widgets.Layout(width="140px"))
        
        self.cond_extra_rows: list = []
        self.cond_extra_box = widgets.VBox([])
        add_btn = widgets.Button(description="+ Condition", button_style="info", layout=widgets.Layout(width="130px"))
        rem_btn = widgets.Button(description="- Retirer", button_style="warning", layout=widgets.Layout(width="130px"))
        
        def _add(_):
            r = (widgets.Dropdown(options=cols, layout=widgets.Layout(width="250px")),
                 widgets.Dropdown(options=ops_list, layout=widgets.Layout(width="180px")),
                 widgets.Text(placeholder="valeur", layout=widgets.Layout(width="220px")))
            self.cond_extra_rows.append(r)
            self.cond_extra_box.children = [widgets.HBox(list(r), layout=widgets.Layout(margin="5px 0")) for r in self.cond_extra_rows]
        
        def _rem(_):
            if self.cond_extra_rows:
                self.cond_extra_rows.pop()
                self.cond_extra_box.children = [widgets.HBox(list(r), layout=widgets.Layout(margin="5px 0")) for r in self.cond_extra_rows]
        
        add_btn.on_click(_add); rem_btn.on_click(_rem)
        self.cond_map_text = widgets.Textarea(placeholder="jan:1, feb:2\n(laisser vide pour utiliser ALORS/SINON)", description="Valeurs Auto:", layout=widgets.Layout(width="460px", height="80px"))
        self.cond_new_col  = widgets.Text(description="Nom final:", placeholder="nouvelle_col", layout=widgets.Layout(width="250px"))
        self.cond_btn      = widgets.Button(description="Appliquer Logique", button_style=styles.BTN_PRIMARY, layout=widgets.Layout(width="180px", height="34px"))
        self.cond_out      = widgets.Output()
        self.cond_btn.on_click(self._apply_condition)
        
        self.tab_condition = widgets.VBox([
            styles.help_box("<b>Filtres et Conditions</b> — Creez des flags binaires ou des mappings conditionnels.", "#f97316"),
            self._section_title("Definition du filtre", "Filter", "#f97316"),
            widgets.VBox([
                widgets.HBox([self.cond_col, self.cond_op, self.cond_val, self.cond_combine], layout=widgets.Layout(gap="10px", margin="0 0 10px 0")),
                self.cond_extra_box,
                widgets.HBox([add_btn, rem_btn], layout=widgets.Layout(gap="10px", margin="5px 0 15px 0")),
            ], layout=widgets.Layout(padding="15px", border="1px solid #e2e8f0", border_radius="8px", background_color="#fffaf5")),
            self._section_title("Resultat (Output)", "Out", "#f97316"),
            widgets.VBox([
                widgets.HBox([self.cond_then_col, self.cond_then_val, self.cond_else_col, self.cond_else_val], layout=widgets.Layout(gap="10px", margin="0 0 10px 0")),
                self.cond_map_text,
                widgets.HBox([self.cond_new_col, self.cond_btn], layout=widgets.Layout(gap="10px", margin="10px 0 0 0"))
            ], layout=widgets.Layout(padding="15px", border="1px solid #e2e8f0", border_radius="8px", background_color="#fffaf5")),
            self.cond_out],
            layout=widgets.Layout(padding="12px"))

    def _apply_condition(self, _) -> None:
        df = self._get_df().copy(); new_name = self.cond_new_col.value or "new_flag"
        extra_rows_vals = [(r[0].value, r[1].value, r[2].value) for r in self.cond_extra_rows]
        try:
            res_df, final_name = apply_condition(
                df, self.cond_col.value, self.cond_op.value, self.cond_val.value, self.cond_combine.value,
                extra_rows_vals, self.cond_then_col.value, self.cond_then_val.value, self.cond_else_col.value,
                self.cond_else_val.value, self.cond_map_text.value, new_name
            )
            self.tabular_datasets[self.current_ds] = res_df
            self.state.log_step("Feature Eng", "Condition applied", {"new_col": final_name})
            self._propagate_to_state(); self._refresh_columns()
            self._notify(self.cond_out, f"Created '{final_name}'"); self._render_preview()
        except Exception as e:
            self._notify(self.cond_out, traceback.format_exc(), True)

    # ── Formula tab ───────────────────────────────────────────────────────────
    def _build_formula_tab(self) -> None:
        cols = list(self._get_df().columns)
        self._formula_col_html = widgets.HTML(self._formula_col_list_html(cols))
        SNIPPETS = {
            "Binary flag":                "df['new_col'] = (df['col'] != -1).astype(int)",
            "Ratio A / B (safe)":         "df['ratio'] = df['A'] / df['B'].replace(0, np.nan)",
            "Zscore normalize":           "df['zscore'] = (df['col'] - df['col'].mean()) / df['col'].std()",
            "Min-max normalize":          "df['norm'] = (df['col'] - df['col'].min()) / (df['col'].max() - df['col'].min())",
            "Log1p transform":            "df['log_val'] = np.log1p(df['col'])",
            "Interaction A x B":          "df['inter'] = df['A'] * df['B']",
            "Conditional value":          "df['val'] = np.where(df['col'] > 0, df['col'] * 2, 0)",
            "Map values":                 "df['mapped'] = df['col'].map({'yes': 1, 'no': 0})",
            "Extract year":               "df['year'] = pd.to_datetime(df['col']).dt.year",
            "Replace Data (filter)":      "df = df[df['Age'] > 18].copy()",
        }
        self._snippets = SNIPPETS
        self.formula_snippet = widgets.Dropdown(options=["-- choisir un snippet --"]+list(SNIPPETS.keys()), description="Snippets:", layout=widgets.Layout(width="380px"))
        self.formula_insert_btn = widgets.Button(description="Insérer", button_style="info", layout=widgets.Layout(width="100px", height="30px"))
        self.formula_editor = widgets.Textarea(
            placeholder="# Saisissez votre code Python.\n# Utilisez df['col'] = ... pour créer des colonnes.\n# 'raw_dataset' et 'all_datasets' sont disponibles.",
            layout=widgets.Layout(width="100%", height="280px", font_family="monospace"))
        
        self.formula_preview_btn = widgets.Button(description="Aperçu (10 lignes)", layout=widgets.Layout(width="180px", height="36px"))
        self.formula_apply_btn   = widgets.Button(description="Appliquer au Dataset", button_style=styles.BTN_PRIMARY, layout=widgets.Layout(width="180px", height="36px"))
        self.formula_out = widgets.Output()
        
        def _insert(_):
            key = self.formula_snippet.value
            if key in SNIPPETS:
                cur = self.formula_editor.value
                self.formula_editor.value = cur + ("\n" if cur and not cur.endswith("\n") else "") + SNIPPETS[key] + "\n"
        
        self.formula_insert_btn.on_click(_insert)
        self.formula_preview_btn.on_click(lambda _: self._apply_formula(preview=True))
        self.formula_apply_btn.on_click(lambda _: self._apply_formula(preview=False))
        
        self.tab_formula = widgets.VBox([
            styles.help_box("<b>Editeur Python (Power User)</b> — Manipulation directe du DataFrame <code>df</code>.", "#0ea5e9"),
            self._section_title("Code Python Custom", "Python", "#0ea5e9"),
            widgets.VBox([
                widgets.HTML("<b style='font-size:0.82em;color:#6b7280;'>Colonnes disponibles :</b>"),
                self._formula_col_html,
                widgets.HBox([self.formula_snippet, self.formula_insert_btn], layout=widgets.Layout(align_items="center", gap="10px", margin="10px 0")),
                self.formula_editor,
                widgets.HBox([self.formula_preview_btn, self.formula_apply_btn], layout=widgets.Layout(gap="15px", margin="15px 0 0 0")),
            ], layout=widgets.Layout(padding="15px", border="1px solid #e2e8f0", border_radius="8px", background_color="#f0f9ff")),
            self.formula_out], layout=widgets.Layout(padding="12px"))

    def _formula_col_list_html(self, cols: list) -> str:
        pills = "".join(f"<span style='display:inline-block;background:#ede9fe;color:#5b21b6;border-radius:4px;padding:2px 8px;margin:2px 3px;font-size:0.78em;font-family:monospace;'>{c}</span>" for c in cols)
        return f"<div style='line-height:2;margin-bottom:6px;'>{pills}</div>"

    def _refresh_formula_col_list(self) -> None:
        self._formula_col_html.value = self._formula_col_list_html(list(self._get_df().columns))

    def _apply_formula(self, preview: bool = False) -> None:
        df = self._get_df().copy(); code = self.formula_editor.value.strip()
        if not code:
            self._notify(self.formula_out, "Formula editor is empty.", True); return
            
        try:
            # Get the actual raw object if it exists (e.g. if the tabular df was derived from a Graph)
            raw_obj = self.state.data_raw.get(self.current_ds)
            all_ds  = self.state.data_raw
            
            res_df, new_or_mod = run_formula(df, code, raw_dataset=raw_obj, all_datasets=all_ds)
            
            if not new_or_mod:
                self._notify(self.formula_out, "No new/modified columns detected.", True); return
            
            if "__replaced__" in new_or_mod:
                if preview:
                    with self.formula_out:
                        clear_output(wait=True)
                        display(HTML(f"<b style='color:#0ea5e9;'>Preview — Full Dataset Replaced :</b>"))
                        display(res_df.head(10))
                else:
                    self.tabular_datasets[self.current_ds] = res_df
                    self.state.log_step("Feature Eng", "Custom Code applied (DataFrame replaced)", {})
                    self._propagate_to_state(); self._refresh_columns()
                    self._notify(self.formula_out, f"Applied (Dataset replaced) — {len(res_df)} rows, {len(res_df.columns)} cols")
                    self._render_preview()
            else:
                if preview:
                    with self.formula_out:
                        clear_output(wait=True)
                        display(HTML(f"<b style='color:#0ea5e9;'>Preview — {len(new_or_mod)} column(s) :</b>"))
                        display(pd.DataFrame({k: v for k, v in new_or_mod.items()}).head(10))
                else:
                    self.tabular_datasets[self.current_ds] = res_df
                    self.state.log_step("Feature Eng", "Custom Formula applied", {"columns": list(new_or_mod.keys())})
                    self._propagate_to_state(); self._refresh_columns()
                    self._notify(self.formula_out, f"Applied {len(new_or_mod)} column(s) : {', '.join(new_or_mod)}")
                    self._render_preview()
        except Exception:
            with self.formula_out:
                clear_output(wait=True)
                display(HTML(f"<pre style='color:#ef4444;font-size:0.82em;'>{traceback.format_exc()}</pre>"))

    # ── Text tab ──────────────────────────────────────────────────────────────
    def _build_text_tab(self) -> None:
        cols = list(self._get_df().columns)
        text_ops = self.state.config.get("feature_engineering", {}).get("text_operations", ["Lowercase","Uppercase","Length","Extract Regex","Replace","Split & Keep N"])
        self.text_col     = widgets.Dropdown(options=cols, description="Colonne:", layout=widgets.Layout(width="280px"))
        self.text_op      = widgets.Dropdown(options=text_ops, value=text_ops[0], description="Opération:", layout=widgets.Layout(width="220px"))
        self.text_arg1    = widgets.Text(description="Args (Regex/N):", layout=widgets.Layout(width="280px", display="none"))
        self.text_arg2    = widgets.Text(description="Args (Rep/Sep):", layout=widgets.Layout(width="280px", display="none"))
        self.text_new_col = widgets.Text(description="Nouveau Nom:", placeholder="resultat_texte", layout=widgets.Layout(width="280px"))
        self.text_btn     = widgets.Button(description="Appliquer Texte", button_style=styles.BTN_PRIMARY, layout=widgets.Layout(width="180px", height="34px"))
        self.text_out     = widgets.Output()
        
        def _on_op(c):
            hide = c["new"] in ("Lowercase","Uppercase","Length")
            self.text_arg1.layout.display = "none" if hide else "flex"
            self.text_arg2.layout.display = "none" if hide or c["new"] == "Extract Regex" else "flex"
            
        self.text_op.observe(_on_op, names="value")
        self.text_btn.on_click(self._apply_text)
        
        self.tab_text = widgets.VBox([
            styles.help_box("<b>Manipulation de Texte</b> — Nettoyage, extraction ou transformation de chaines.", "#06b6d4"),
            self._section_title("Configuration Textuelle", "Text", "#06b6d4"),
            widgets.VBox([
                widgets.HBox([self.text_col, self.text_op, self.text_arg1, self.text_arg2], layout=widgets.Layout(gap="10px", margin="0 0 10px 0")),
                widgets.HBox([self.text_new_col, self.text_btn], layout=widgets.Layout(gap="10px"))
            ], layout=widgets.Layout(padding="15px", border="1px solid #e2e8f0", border_radius="8px", background_color="#f0f9ff")),
            self.text_out],
            layout=widgets.Layout(padding="12px"))

    def _apply_text(self, _) -> None:
        df = self._get_df(); col = self.text_col.value; op = self.text_op.value
        new_name = self.text_new_col.value or f"{col}_txt"
        try:
            res_df, final_name = apply_text(df, col, op, self.text_arg1.value, self.text_arg2.value, new_name)
            self.tabular_datasets[self.current_ds] = res_df
            self.state.log_step("Feature Eng", "Text Op applied", {"op": op, "new_col": final_name})
            self._propagate_to_state(); self._refresh_columns()
            self._notify(self.text_out, f"Created '{final_name}'"); self._render_preview()
        except Exception as e:
            self._notify(self.text_out, str(e), True)

    # ── Date tab ──────────────────────────────────────────────────────────────
    def _build_date_tab(self) -> None:
        cols = list(self._get_df().columns)
        date_ops = self.state.config.get("feature_engineering", {}).get("date_operations", ["Year","Month","Day","DayOfWeek","Hour","Minute","IsWeekend"])
        self.date_col     = widgets.Dropdown(options=cols, description="Colonne Date:", layout=widgets.Layout(width="300px"))
        self.date_extract = widgets.SelectMultiple(options=date_ops, value=date_ops[:2], description="Extraire:", layout=widgets.Layout(width="250px", height="120px"))
        self.date_btn     = widgets.Button(description="Extraire Features", button_style=styles.BTN_PRIMARY, layout=widgets.Layout(width="180px", height="34px"))
        self.date_out     = widgets.Output()
        self.date_btn.on_click(self._apply_date)
        
        self.tab_date = widgets.VBox([
            styles.help_box("<b>Dates et Temps</b> — Decomposez des dates en features exploitables pour le ML.", "#f59e0b"),
            self._section_title("Configuration Temporelle", "Date", "#f59e0b"),
            widgets.VBox([
                widgets.HBox([self.date_col, self.date_extract], layout=widgets.Layout(gap="15px", margin="0 0 10px 0")),
                self.date_btn,
            ], layout=widgets.Layout(padding="15px", border="1px solid #e2e8f0", border_radius="8px", background_color="#fffbeb")),
            self.date_out],
            layout=widgets.Layout(padding="12px"))

    def _apply_date(self, _) -> None:
        df = self._get_df(); col = self.date_col.value; features = self.date_extract.value
        try:
            res_df, created = apply_date(df, col, features)
            self.tabular_datasets[self.current_ds] = res_df
            self.state.log_step("Feature Eng", "Date Features extracted", {"col": col, "features": created})
            self._propagate_to_state(); self._refresh_columns()
            self._notify(self.date_out, f"Created {len(created)} columns"); self._render_preview()
        except Exception as e:
            self._notify(self.date_out, str(e), True)

    # ── Binning tab ───────────────────────────────────────────────────────────
    def _build_binning_tab(self) -> None:
        cols = list(self._get_df().columns)
        bin_ops = self.state.config.get("feature_engineering", {}).get("binning_strategies", ["Equal Width (Cut)","Equal Frequency (Qcut)","Custom Edges"])
        self.bin_col     = widgets.Dropdown(options=cols, description="Colonne:", layout=widgets.Layout(width="280px"))
        self.bin_method  = widgets.Dropdown(options=bin_ops, value=bin_ops[0], description="Méthode:", layout=widgets.Layout(width="240px"))
        self.bin_bins    = widgets.Text(value="5", description="Bins/Edges:", layout=widgets.Layout(width="220px"))
        self.bin_labels  = widgets.Checkbox(value=False, description="Labels numériques", layout=widgets.Layout(width="200px"))
        self.bin_new_col = widgets.Text(description="Nom final:", placeholder="binned_col", layout=widgets.Layout(width="280px"))
        self.bin_btn     = widgets.Button(description="Appliquer Binning", button_style=styles.BTN_PRIMARY, layout=widgets.Layout(width="180px", height="34px"))
        self.bin_out     = widgets.Output()
        self.bin_btn.on_click(self._apply_binning)
        
        self.tab_binning = widgets.VBox([
            styles.help_box("<b>Binning (Discretisation)</b> — Transformez des variables continues en categories.", "#10b981"),
            self._section_title("Configuration du Binning", "Chart", "#10b981"),
            widgets.VBox([
                widgets.HBox([self.bin_col, self.bin_method, self.bin_bins], layout=widgets.Layout(gap="10px", margin="0 0 10px 0")),
                widgets.HBox([self.bin_labels, self.bin_new_col, self.bin_btn], layout=widgets.Layout(gap="10px"))
            ], layout=widgets.Layout(padding="15px", border="1px solid #e2e8f0", border_radius="8px", background_color="#f0fdf4")),
            self.bin_out],
            layout=widgets.Layout(padding="12px"))

    def _apply_binning(self, _) -> None:
        df = self._get_df(); col = self.bin_col.value; method = self.bin_method.value
        bins_val = self.bin_bins.value; new_name = self.bin_new_col.value or f"{col}_bin"
        try:
            res_df, final_name = apply_binning(df, col, method, bins_val, self.bin_labels.value, new_name)
            self.tabular_datasets[self.current_ds] = res_df
            self.state.log_step("Feature Eng", "Binning applied", {"method": method, "new_col": final_name})
            self._propagate_to_state(); self._refresh_columns()
            self._notify(self.bin_out, f"Created '{final_name}'"); self._render_preview()
        except Exception as e:
            self._notify(self.bin_out, str(e), True)

    # ── Viz tab ───────────────────────────────────────────────────────────────
    def _build_viz_tab(self) -> None:
        cols = list(self._get_df().columns)
        viz_ops = self.state.config.get("feature_engineering", {}).get("viz_types", ["auto","scatter","line","bar","box","violin","hist","kde"])
        self.viz_x    = widgets.Dropdown(options=cols, description="Axe X:", layout=widgets.Layout(width="240px"))
        self.viz_y    = widgets.Dropdown(options=cols, description="Axe Y:", layout=widgets.Layout(width="240px"))
        self.viz_hue  = widgets.Dropdown(options=["(None)"]+cols, description="Couleur:", layout=widgets.Layout(width="240px"))
        self.viz_kind = widgets.Dropdown(options=viz_ops, value="auto", description="Type:", layout=widgets.Layout(width="180px"))
        self.viz_btn  = widgets.Button(description="Générer Graphe", button_style="info", layout=widgets.Layout(width="150px", height="34px"))
        self.viz_out  = widgets.Output()
        self.viz_btn.on_click(self._apply_viz)
        
        self.tab_viz = widgets.VBox([
            styles.help_box("<b>Visualisation Rapide</b> — Verifiez l'impact de vos transformations en un clin d'oeil.", "#6366f1"),
            self._section_title("Configuration Graphique", "Viz", "#6366f1"),
            widgets.VBox([
                widgets.HBox([self.viz_x, self.viz_y, self.viz_hue, self.viz_kind, self.viz_btn], 
                              layout=widgets.Layout(gap="10px", align_items="center")),
            ], layout=widgets.Layout(padding="15px", border="1px solid #e2e8f0", border_radius="8px", background_color="#f5f3ff")),
            self.viz_out], layout=widgets.Layout(padding="12px"))

    def _apply_viz(self, _) -> None:
        df = self._get_df(); x = self.viz_x.value; y = self.viz_y.value
        hue = None if self.viz_hue.value == "(None)" else self.viz_hue.value
        kind = self.viz_kind.value
        with self.viz_out:
            clear_output(wait=True)
            try:
                fig = create_viz_fig(df, x, y, hue, kind)
                display(fig)
                plt.close(fig)
            except Exception as e:
                print(f"[ERROR] {e}")

    # ── Dashboard tab ─────────────────────────────────────────────────────────
    def _build_dashboard_tab(self) -> None:
        cols = list(self._get_df().columns)
        target_val = "(None)"
        if hasattr(self.state, "business_context") and self.state.business_context.get("target") in cols:
            target_val = self.state.business_context["target"]
        self.dash_target   = widgets.Dropdown(options=["(None)"]+cols, value=target_val, description="Cible (Y):", layout=widgets.Layout(width="280px"))
        self.dash_features = widgets.SelectMultiple(options=cols, description="Features:", layout=widgets.Layout(width="280px", height="150px"))
        self.dash_btn      = widgets.Button(description="Lancer l'Analyse", button_style=styles.BTN_PRIMARY, layout=widgets.Layout(width="180px", height="34px"))
        self.dash_out      = widgets.Output()
        self.dash_btn.on_click(self._apply_dashboard)
        
        self.tab_dashboard = widgets.VBox([
            styles.help_box("<b>Analyse de Correlation</b> — Comparez vos features a la target pour evaluer leur pertinence.", "#be185d"),
            self._section_title("Dashboard de Performance", "Target", "#be185d"),
            widgets.VBox([
                widgets.HBox([self.dash_target, self.dash_features, self.dash_btn], 
                              layout=widgets.Layout(gap="20px", align_items="flex-start")),
            ], layout=widgets.Layout(padding="15px", border="1px solid #e2e8f0", border_radius="8px", background_color="#fff1f2")),
            self.dash_out], layout=widgets.Layout(padding="12px"))

    def _apply_dashboard(self, _) -> None:
        df = self._get_df(); target = self.dash_target.value; features = list(self.dash_features.value)[:4]
        if target == "(None)" or not features:
            self._notify(self.dash_out, "Sélectionnez une target et au moins une feature.", True); return
        with self.dash_out:
            clear_output(wait=True)
            try:
                fig = create_dashboard_fig(df, target, features)
                display(fig)
                plt.close(fig)
            except Exception as e:
                print(f"Error plotting dashboard: {e}")

    # ── Manage tab ────────────────────────────────────────────────────────────
    def _build_manage_tab(self) -> None:
        cols = list(self._get_df().columns)
        enc_cfg = self.state.config.get("encoding", {})
        tabular_types = list(enc_cfg.get("tabular", enc_cfg).keys()) or ["numeric","categorical","binary","datetime","text","id_like"]
        manage_ops = self.state.config.get("feature_engineering", {}).get("manage_actions", ["Set Type (Meta)","Duplicate","Delete"])
        self.manage_col      = widgets.Dropdown(options=cols, description="Colonne:", layout=widgets.Layout(width="280px"))
        self.manage_action   = widgets.Dropdown(options=manage_ops, value=manage_ops[0], description="Action:", layout=widgets.Layout(width="220px"))
        self.manage_type     = widgets.Dropdown(options=tabular_types, value=tabular_types[0], description="Nouveau Type:", layout=widgets.Layout(width="220px"))
        self.manage_new_name = widgets.Text(description="Nom Duplicate:", placeholder="nom_copie", layout=widgets.Layout(width="220px", display="none"))
        self.manage_btn      = widgets.Button(description="Exécuter Action", button_style="warning", layout=widgets.Layout(width="180px", height="34px"))
        self.manage_out      = widgets.Output()
        
        def _on_action(c):
            self.manage_new_name.layout.display = "flex" if c["new"] == "Duplicate" else "none"
            self.manage_type.layout.display = "flex" if c["new"] == "Set Type (Meta)" else "none"
            
        self.manage_action.observe(_on_action, names="value")
        self.manage_btn.on_click(self._apply_manage)
        
        self.tab_manage = widgets.VBox([
            styles.help_box("<b>Inventaire des Colonnes</b> — Gerez les types, dupliquez ou supprimez des colonnes.", "#eab308"),
            self._section_title("Maintenance du Dataset", "Setup", "#eab308"),
            widgets.VBox([
                widgets.HBox([self.manage_col, self.manage_action], layout=widgets.Layout(gap="10px", margin="0 0 10px 0")),
                widgets.HBox([self.manage_type, self.manage_new_name], layout=widgets.Layout(gap="10px", margin="0 0 10px 0")),
                self.manage_btn
            ], layout=widgets.Layout(padding="15px", border="1px solid #e2e8f0", border_radius="8px", background_color="#fefce8")),
            self.manage_out],
            layout=widgets.Layout(padding="12px"))

    def _apply_manage(self, _) -> None:
        df = self._get_df(); col = self.manage_col.value; action = self.manage_action.value
        try:
            if action == "Delete":
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)
                    self.tabular_datasets[self.current_ds] = df
                    self.state.log_step("Feature Eng", "Column deleted", {"col": col})
                    self._propagate_to_state(); self._refresh_columns()
                    self._notify(self.manage_out, f"Deleted '{col}'"); self._render_preview()
            elif action == "Drop & Keep for Submission":
                if col in df.columns:
                    if not hasattr(self.state, "aside_features"):
                        self.state.aside_features = {}
                    self.state.aside_features[col] = df[col].copy()
                    df.drop(columns=[col], inplace=True)
                    self.tabular_datasets[self.current_ds] = df
                    self.state.log_step("Feature Eng", "Column stored aside", {"col": col})
                    self._propagate_to_state(); self._refresh_columns()
                    self._notify(self.manage_out, f"Dropped & stored '{col}'"); self._render_preview()
            elif action == "Duplicate":
                new_name = self.manage_new_name.value or f"{col}_copy"
                if new_name in df.columns:
                    self._notify(self.manage_out, f"'{new_name}' already exists.", True); return
                df[new_name] = df[col].copy()
                self.tabular_datasets[self.current_ds] = df
                self.state.log_step("Feature Eng", "Column duplicated", {"col": col, "new_col": new_name})
                self._propagate_to_state(); self._refresh_columns()
                self._notify(self.manage_out, f"Duplicated '{col}' → '{new_name}'"); self._render_preview()
            elif action == "Set Type (Meta)":
                new_type = self.manage_type.value; ds = self.current_ds
                if not hasattr(self.state, "meta"): self.state.meta = {}
                if ds not in self.state.meta: self.state.meta[ds] = {}
                if col not in self.state.meta[ds]: self.state.meta[ds][col] = {}
                self.state.meta[ds][col]["kind"] = new_type
                self.state.log_step("Feature Eng", "Type overridden", {"col": col, "new_type": new_type})
                self._notify(self.manage_out, f"Set type of '{col}' → '{new_type}'")
        except Exception as e:
            self._notify(self.manage_out, str(e), True)

class FeatureEngUI:
    """Interface de routage pour Feature Engineering selon le type de dataset."""
    def __init__(self, state):
        self.state = state
        self.all_datasets = {}
        for k, v in self.state.data_raw.items():
            self.all_datasets[k] = v
        for k, v in getattr(self.state, "data_cleaned", {}).items():
            self.all_datasets[f"[CLN] {k}"] = v
            
        self._build_ui()

    def _build_guide(self) -> widgets.Accordion:
        guide_html = """
        <div style='font-size:14px; line-height:1.6; color:#374151;'>
            <h4>Guide de l'Éditeur Python (Custom Code)</h4>
            <p>Cet éditeur permet de manipuler vos données avec la puissance de Python (Pandas, Numpy, Math).</p>
            <ul>
                <li><b>Variables disponibles :</b>
                    <ul>
                        <li><code>df</code> : Le DataFrame actuel. Utilisez <code>df['colonne']</code> pour lire ou écrire.</li>
                        <li><code>raw_dataset</code> : L'objet d'origine (ex: Graphe NetworkX, Ontologie RDFLib) si disponible.</li>
                        <li><code>all_datasets</code> : Un dictionnaire de TOUS les datasets chargés (ex: <code>all_datasets['Iris']</code>).</li>
                    </ul>
                </li>
                <li><b>Exemples :</b>
                    <pre style='background:#f1f5f9; padding:5px;'># Exemple 1: Nouvelle colonne basée sur calcul
df['Total'] = (df['col_a'] + df['col_b']) * 1.5

# Exemple 2: Remplacement complet (filtrage)
df = df[df['Age'] > 18].copy()

# Exemple 3: Utilisation de propriétés d'un graphe
if raw_dataset is not None:
    df['node_degree'] = [raw_dataset.degree(n) for n in df['node_id']]</pre>
                </li>
            </ul>
        </div>
        """
        out = widgets.Output()
        with out: display(HTML(guide_html))
        acc = widgets.Accordion(children=[out], selected_index=None)
        acc.set_title(0, "💡 Guide: Comment utiliser l'éditeur Python & Objets d'origine")
        return acc

    def _build_ui(self):
        # Existing code...
        if not self.all_datasets:
            self.ui = styles.error_msg("Aucun dataset disponible pour Feature Engineering.")
            return

        self.ds_selector = widgets.Dropdown(
            options=list(self.all_datasets.keys()),
            description="Dataset:", layout=widgets.Layout(width="360px"))
        self.ds_selector.observe(self.on_ds_change, names="value")
        self.current_ds = self.ds_selector.value
        
        header = widgets.HTML(styles.card_html("Feature Engineering", "Advanced Variable Laboratory", ""))
        guide_acc = self._build_guide()
        top_bar = widgets.HBox(
            [header, widgets.HTML("<div style='flex:1'></div>"), self.ds_selector],
            layout=widgets.Layout(align_items="center", justify_content="space-between",
                                   margin="0 0 12px 0", padding="0 0 10px 0",
                                   border_bottom="2px solid #ede9fe"))
        
        self.dynamic_ui = widgets.VBox([])
        self.ui = widgets.VBox(
            [top_bar, guide_acc, self.dynamic_ui],
            layout=widgets.Layout(width="100%", max_width="1000px", border="1px solid #e5e7eb",
                                   padding="18px", border_radius="10px", background_color="#ffffff"))
        self.on_ds_change(None)

    def on_ds_change(self, change):
        if change:
            self.current_ds = change["new"]
        if not self.current_ds:
            return
            
        data = self.all_datasets[self.current_ds]
        orig_key = self.current_ds.split(" ", 1)[1] if " " in self.current_ds else self.current_ds
        ds_type = self.state.data_types.get(orig_key, "tabular")
        
        if ds_type in ("tabular", "csv", "sklearn", "excel", "timeseries"):
            # Use TabularFeatureEngUI
            tabular_ui = TabularFeatureEngUI(self.state)
            # Remove top bars of tabular inside to avoid double top bar
            tabular_ui.ui.children = tabular_ui.ui.children[1:]
            self.dynamic_ui.children = [tabular_ui.ui]
        elif ds_type == "image":
            from .image import build_image_ui
            build_image_ui(self, data)
        elif ds_type == "text":
            from .text import build_text_ui
            build_text_ui(self, data)
        elif ds_type in ("graph", "ontology"):
            from .graph import build_graph_ui
            build_graph_ui(self, data)
        else:
            self.dynamic_ui.children = [widgets.HTML(f"<div style='padding:20px;'><b style='color:#ef4444;'>Warning:</b> Feature Engineering for '{ds_type}' is not yet implemented natively in this tab. The tabular engineer applies to dataframes only.</div>")]

