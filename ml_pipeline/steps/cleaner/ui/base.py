import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from ml_pipeline.styles import styles
from ..logic.operations import auto_suggest_missing, execute_cleaning_logic

class TabularCleaner:
    """Interface de nettoyage : valeurs manquantes, nulls, doublons (TABULAR)."""

    def __init__(self, state):
        self.state = state
        if not hasattr(state, "config") or not state.config:
            self.ui = styles.error_msg("[ERROR] Configuration non chargée.")
            return
        self.config = state.config.get("cleaning", {})
        self.missing_options = [(opt["label"], opt["value"])
                                for opt in self.config.get("missing", [])]
        if not any(v == "none" for _, v in self.missing_options):
            self.missing_options.insert(0, ("Do nothing", "none"))
            
        # Flatten nested data_raw for tabular datasets
        tabular_ds = {}
        for k, v in self.state.data_raw.items():
            if isinstance(v, pd.DataFrame):
                tabular_ds[k] = v
            elif isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, pd.DataFrame):
                        tabular_ds[f"{k} - {sub_k}"] = sub_v
                        
        self.original_datasets = {k: v.copy() for k, v in tabular_ds.items()}
        self.current_datasets = {k: v.copy() for k, v in self.original_datasets.items()}
        self.current_ds: str | None = None
        self.row_widgets: dict = {}
        if not hasattr(state, "meta"):
            state.meta = {}
        self.meta = state.meta
        self._sync_metadata()
        self._build_ui()

    def _sync_metadata(self) -> None:
        for ds_name, df in self.original_datasets.items():
            if ds_name not in self.meta:
                self.meta[ds_name] = {}
            for col in df.columns:
                if col not in self.meta[ds_name]:
                    s = df[col]; n_unq = s.nunique()
                    if pd.api.types.is_datetime64_any_dtype(s): kind = "datetime"
                    elif pd.api.types.is_bool_dtype(s) or n_unq == 2: kind = "binary"
                    elif pd.api.types.is_numeric_dtype(s):
                        kind = "id_like" if n_unq / max(len(s), 1) > 0.95 else "numeric"
                    else:
                        kind = "categorical" if n_unq < 100 else "text"
                    self.meta[ds_name][col] = {"kind": kind}

    def _build_ui(self) -> None:
        if not self.original_datasets:
            self.ui = styles.error_msg("Aucune donnée tabulaire disponible pour le nettoyage.")
            # Put non-pandas objects straight into data_cleaned
            for k, v in self.state.data_raw.items():
                if not isinstance(v, pd.DataFrame) and not isinstance(v, dict):
                    self.state.data_cleaned[k] = v
                elif isinstance(v, dict):
                    if k not in self.state.data_cleaned:
                        self.state.data_cleaned[k] = {}
                    for sk, sv in v.items():
                        if not isinstance(sv, pd.DataFrame):
                            self.state.data_cleaned[k][sk] = sv
            return
            
        header  = widgets.HTML(styles.card_html("Clean", "Gestion des valeurs manquantes", ""))
        top_bar = widgets.HBox([header], layout=widgets.Layout(
            align_items="center", margin="0 0 12px 0",
            padding="0 0 10px 0", border_bottom="2px solid #ede9fe"))
            
        self.ds_selector = widgets.Dropdown(
            options=list(self.original_datasets.keys()),
            description="Dataset:", layout=styles.LAYOUT_DD_LONG)
        self.ds_selector.observe(self.on_ds_change, names="value")
        
        if list(self.original_datasets.keys()):
            self.current_ds = list(self.original_datasets.keys())[0]
            
        help_text = styles.help_box(
            "<b>Suggestions automatiques</b> basées sur le taux de manquants :<br>"
            "<ul><li>>50% → Supprimer colonne</li>"
            "<li><5% → Supprimer lignes</li>"
            "<li>Numérique → Médiane</li>"
            "<li>Catégoriel → Mode</li></ul>",
            "#10b981")
            
        self.table_container = widgets.VBox(layout=widgets.Layout(
            width="100%", border="1px solid #e2e8f0", border_radius="6px",
            padding="10px", background_color="#f8fafc", margin="0 0 15px 0"))
            
        self.btn_apply = widgets.Button(description="Execute Cleaning",
                                         button_style=styles.BTN_PRIMARY,
                                         layout=styles.LAYOUT_BTN_LARGE)
        self.btn_apply.on_click(self._execute_cleaning)
        
        self.btn_reset = widgets.Button(description="Reset to Raw",
                                         button_style=styles.BTN_WARNING,
                                         layout=styles.LAYOUT_BTN_STD)
        self.btn_reset.on_click(self._reset_cleaning)
        
        self.out_logs = widgets.Output()
        self.ui = widgets.VBox(
            [top_bar, self.ds_selector, help_text, self.table_container,
             widgets.HBox([self.btn_apply, self.btn_reset],
                           layout=widgets.Layout(gap="15px")),
             self.out_logs],
            layout=widgets.Layout(width="100%", max_width="1000px",
                                   border="1px solid #e5e7eb", padding="18px",
                                   border_radius="10px", background_color="#ffffff"))
        self.on_ds_change(None)

    def on_ds_change(self, change) -> None:
        if change and change["new"]:
            self.current_ds = change["new"]
        self._build_table()

    def _build_table(self) -> None:
        if not self.current_ds:
            return
        df = self.original_datasets[self.current_ds]
        ds_meta = self.meta.get(self.current_ds, {})
        headers = widgets.HBox([
            widgets.HTML("<div style='width:220px;font-weight:bold;color:#475569;'>Column</div>"),
            widgets.HTML("<div style='width:100px;font-weight:bold;color:#475569;text-align:right;padding-right:15px;'>% Missing</div>"),
            widgets.HTML("<div style='width:220px;font-weight:bold;color:#475569;'>Null representations</div>"),
            widgets.HTML("<div style='width:250px;font-weight:bold;color:#475569;'>Missing Action</div>"),
        ], layout=widgets.Layout(border_bottom="2px solid #cbd5e1", padding="0 0 5px 0", margin="0 0 10px 0"))
        rows = [headers]
        self.row_widgets = {}
        for col in df.columns:
            col_meta = ds_meta.get(col, {})
            pct_miss = col_meta.get("pct_miss", df[col].isna().mean() * 100)
            is_num   = col_meta.get("kind", "") in ("numeric",) or pd.api.types.is_numeric_dtype(df[col])
            sug_miss = auto_suggest_missing(col, df, col_meta)
            col_opts = [(lbl, val) for lbl, val in self.missing_options
                        if is_num or val not in ("mean", "median", "interpolate_linear",
                                                  "interpolate_time", "knn", "mice", "zero")]
            if not any(sug_miss == v for _, v in col_opts):
                sug_miss = "none"
            color = "#ef4444" if pct_miss > 20 else "#f59e0b" if pct_miss > 0 else "#22c55e"
            lbl_col  = widgets.HTML(f"<div style='width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding-top:4px;' title='{col}'><b>{col}</b> <span style='color:#94a3b8;font-size:0.8em;'>[{col_meta.get('kind','?')}]</span></div>")
            lbl_miss = widgets.HTML(f"<div style='width:100px;text-align:right;padding-right:15px;color:{color};padding-top:4px;'>{pct_miss:.1f}%</div>")
            txt_nulls = widgets.Text(placeholder="e.g. -1, unknown, ?", layout=widgets.Layout(width="200px", margin="0 20px 0 0"))
            dd_m = widgets.Dropdown(options=col_opts, value=sug_miss, layout=widgets.Layout(width="240px"))
            if pct_miss == 0:
                dd_m.disabled = True

            def _on_nulls(change, c=col, d=dd_m, l=lbl_miss, opts=col_opts, s=df[col].copy(), m=col_meta):
                null_str = change["new"].strip()
                temp = s.copy()
                if null_str:
                    reps = [r.strip() for r in null_str.split(",") if r.strip()]
                    to_rep = []
                    for r in reps:
                        to_rep.append(r)
                        try: to_rep.append(int(r))
                        except ValueError: pass
                        try: to_rep.append(float(r))
                        except ValueError: pass
                    temp = temp.replace(to_rep, np.nan)
                new_pct = temp.isna().mean() * 100
                col2 = "#ef4444" if new_pct > 20 else "#f59e0b" if new_pct > 0 else "#22c55e"
                l.value = f"<div style='width:100px;text-align:right;padding-right:15px;color:{col2};padding-top:4px;'>{new_pct:.1f}%</div>"
                d.disabled = new_pct == 0
                if new_pct == 0:
                    d.value = "none"

            txt_nulls.observe(_on_nulls, names="value")
            self.row_widgets[col] = {"is_num": is_num, "missing": dd_m, "null_reps": txt_nulls}
            rows.append(widgets.HBox([lbl_col, lbl_miss, txt_nulls, dd_m],
                                      layout=widgets.Layout(padding="4px 0", border_bottom="1px solid #f1f5f9")))
        self.table_container.children = rows

    def _reset_cleaning(self, b) -> None:
        with self.out_logs:
            clear_output()
            if self.current_ds:
                self.current_datasets[self.current_ds] = self.original_datasets[self.current_ds].copy()
                self._build_table()
                print(f"[INFO] '{self.current_ds}' restauré à l'état original.")

    def _execute_cleaning(self, b) -> None:
        with self.out_logs:
            clear_output()
            params = {}
            ops = 0
            row_widgets_info = {}
            
            for col, wgts in self.row_widgets.items():
                m_act    = wgts["missing"].value
                null_str = wgts["null_reps"].value.strip()
                is_num   = wgts["is_num"]
                row_widgets_info[col] = {"is_num": is_num}
                if m_act == "none" and not null_str:
                    continue
                params[col] = {"missing": m_act, "null_reps": null_str}
                ops += 1

            df_new, fitted_states = execute_cleaning_logic(self.original_datasets[self.current_ds], params, row_widgets_info)
            self.current_datasets[self.current_ds] = df_new
            
            # Store back safely to data_cleaned
            ds_name = self.current_ds
            if " - " in ds_name and ds_name not in self.state.data_cleaned:
                parent, child = ds_name.split(" - ", 1)
                if parent not in self.state.data_cleaned:
                    self.state.data_cleaned[parent] = {}
                self.state.data_cleaned[parent][child] = df_new
            else:
                self.state.data_cleaned[ds_name] = df_new
                
            self.state.log_step("Data Cleaning", "Cleaning Applied",
                                 {"dataset": self.current_ds, "operations": params, "fitted_states": fitted_states})
            display(styles.info_msg(
                f"<b>{ops}</b> opération(s) appliquée(s) sur '{self.current_ds}'.<br>"
                f"Original : {self.original_datasets[self.current_ds].shape} → "
                f"Nettoyé : {df_new.shape}"))

class AdvancedCleaner:
    """Routage du nettoyage selon le type."""
    def __init__(self, state):
        self.state = state
        self.all_datasets = {}
        # Ensure data_cleaned mirrors data_raw initially for non-tabular
        for k, v in self.state.data_raw.items():
            self.all_datasets[k] = v
            
        self._build_ui()

    def _build_ui(self):
        if not self.all_datasets:
            self.ui = styles.error_msg("Aucun dataset disponible pour le Nettoyage.")
            return

        self.ds_selector = widgets.Dropdown(
            options=list(self.all_datasets.keys()),
            description="Dataset:", layout=widgets.Layout(width="360px"))
        self.ds_selector.observe(self.on_ds_change, names="value")
        self.current_ds = self.ds_selector.value
        
        header = widgets.HTML(styles.card_html("Clean", "Nettoyage des Données", ""))
        top_bar = widgets.HBox(
            [header, widgets.HTML("<div style='flex:1'></div>"), self.ds_selector],
            layout=widgets.Layout(align_items="center", justify_content="space-between",
                                   margin="0 0 12px 0", padding="0 0 10px 0",
                                   border_bottom="2px solid #ede9fe"))
        
        self.dynamic_ui = widgets.VBox([])
        self.ui = widgets.VBox(
            [top_bar, self.dynamic_ui],
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
            tabular_ui = TabularCleaner(self.state)
            if hasattr(tabular_ui, "ui") and tabular_ui.ui.children:
                tabular_ui.ui.children = tabular_ui.ui.children[1:]
                self.dynamic_ui.children = [tabular_ui.ui]
            else:
                 self.dynamic_ui.children = [widgets.HTML("Aucune donnée tabulaire à nettoyer.")]
        elif ds_type == "text":
            from .text import build_text_ui
            build_text_ui(self, data)
        elif ds_type == "image":
            from .image import build_image_ui
            build_image_ui(self, data)
        else:
            self.state.data_cleaned[self.current_ds] = data
            self.dynamic_ui.children = [widgets.HTML(f"<div style='padding:20px;'><b style='color:#10b981;'>Info:</b> Le dataset '{self.current_ds}' ({ds_type}) n'a pas besoin de nettoyage particulier via l'UI et a été transféré tel quel dans state.data_cleaned.</div>")]


