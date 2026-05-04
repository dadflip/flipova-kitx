import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ml_pipeline.styles import styles

from ..logic.operations import calc_outliers, apply_outliers, apply_encoding, build_sklearn_pipeline_code

class UltimateEncoder:
    """Interface d'encodage des variables + gestion des outliers."""

    def __init__(self, state):
        self.state = state
        if not hasattr(state, "config") or not state.config:
            self.ui = styles.error_msg("Configuration non chargée.")
            return
        self.config = state.config.get("encoding", {})
        self.all_datasets = getattr(state, "data_cleaned", {})
        if not hasattr(state, "meta"):
            state.meta = {}
        self.meta = state.meta
        cleaning_config = state.config.get("cleaning", {})
        self.outlier_options = [(opt["label"], opt["value"])
                                for opt in cleaning_config.get("outliers", [])]
        if not any(v == "none" for _, v in self.outlier_options):
            self.outlier_options.insert(0, ("Do nothing", "none"))
            
        # Flatten any nested dictionaries in data_cleaned (e.g. from local folders)
        tabular_ds = {}
        non_tab_ds = {}
        for k, v in self.all_datasets.items():
            if isinstance(v, pd.DataFrame):
                tabular_ds[k] = v
            elif isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, pd.DataFrame):
                        tabular_ds[f"{k} - {sub_k}"] = sub_v
                    else:
                        non_tab_ds[f"{k} - {sub_k}"] = sub_v
            else:
                non_tab_ds[k] = v
                
        self.datasets = {k: v.copy() for k, v in tabular_ds.items()}
        self.non_tabular = non_tab_ds
        
        self._sync_metadata()
        self._build_ui()

    def _sync_metadata(self) -> None:
        for ds_name, df in self.datasets.items():
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

    def _get_encoded_df(self) -> pd.DataFrame | None:
        if not getattr(self, "current_ds", None):
            return None
        df = self.datasets[self.current_ds].copy()
        if not hasattr(self, "_tab_enc_widgets"):
            return df
            
        enc_params = {}
        for col, wd in self._tab_enc_widgets.items():
            enc_params[col] = {
                "enc_value": wd["dd"].value,
                "kind": wd["kind"]
            }
            
        df_enc, _ = apply_encoding(df, enc_params, self.config)
        return df_enc

    def _build_ui(self) -> None:
        tabs_children = []
        if self.datasets:
            self.ds_selector = widgets.Dropdown(options=list(self.datasets.keys()), description="Dataset:")
            self.outlier_timing = widgets.Dropdown(
                options=["Before Encoding", "After Encoding"], value="Before Encoding",
                description="Outliers:", layout=widgets.Layout(width="350px"))
            help_text = styles.help_box(
                "<b>Encoding & Outliers</b> — configurez l'encodage par colonne et le timing des outliers.<br>"
                "<b>Before Encoding</b> (recommandé) : les outliers sont traités avant la mise à l'échelle.<br>"
                "<b>After Encoding</b> : utile si l'encodage génère de nouvelles métriques numériques.",
                "#10b981")
            selectors = widgets.HBox([self.ds_selector, self.outlier_timing],
                                      layout=widgets.Layout(margin="0 0 10px 0", gap="20px"))
            self.ds_selector.observe(self._on_tab_ds_change, names="value")
            self.outlier_timing.observe(self._build_outliers_table, names="value")
            self.current_ds = self.ds_selector.value
            self.enc_container     = widgets.VBox()
            self.outlier_container = widgets.VBox()
            self.outlier_plot_out  = widgets.Output()
            self.tab_out           = widgets.Output()
            btn_apply = widgets.Button(description="Execute Pipeline",
                                        button_style=styles.BTN_PRIMARY,
                                        layout=styles.LAYOUT_BTN_LARGE)
            btn_apply.on_click(self._apply_tabular)
            self._on_tab_ds_change(None)
            tabs_children.append(widgets.VBox([selectors, help_text, self.enc_container,
                                               self.outlier_container, self.outlier_plot_out,
                                               widgets.HBox([btn_apply], layout=widgets.Layout(margin="20px 0 10px 0")),
                                               self.tab_out]))
        else:
            tabs_children.append(widgets.HTML("<div style='padding:16px;'>Aucune donnée tabulaire nettoyée.</div>"))
            for k, v in self.non_tabular.items():
                if " - " in k and k not in self.state.data_encoded:
                    parent, child = k.split(" - ", 1)
                    if parent not in self.state.data_encoded:
                        self.state.data_encoded[parent] = {}
                    self.state.data_encoded[parent][child] = v
                else:    
                    self.state.data_encoded[k] = v

        # Onglet Non-Tabular avec gestion des ontologies
        non_tabular_content = [widgets.HTML("<div style='padding:8px;font-weight:600;color:#374151;'>Données non-tabulaires détectées:</div>")]

        if self.non_tabular:
            for name, data in self.non_tabular.items():
                data_type = self._detect_non_tabular_type(data)
                if data_type == "ontology":
                    # Ontology specific options
                    onto_opts = widgets.Dropdown(
                        options=[
                            ("Pass-through (conserver tel quel)", "passthrough"),
                            ("Extraire classes → DataFrame", "extract_classes"),
                            ("Extraire propriétés → DataFrame", "extract_props"),
                            ("Convertir NetworkX → graphe", "to_networkx"),
                            ("Vectoriser Node2Vec", "node2vec"),
                            ("Supprimer", "drop"),
                        ],
                        value="passthrough",
                        description=f"{name}:",
                        layout=widgets.Layout(width="500px")
                    )
                    non_tabular_content.append(widgets.HBox([
                        widgets.HTML(f"<span style='color:#6d28d9;font-weight:500;'>🧠 Ontologie:</span>"),
                        onto_opts
                    ], layout=widgets.Layout(padding="4px 0")))
                elif data_type == "graph":
                    non_tabular_content.append(widgets.HBox([
                        widgets.HTML(f"<span style='color:#059669;font-weight:500;'>📊 Graphe:</span> {name}")
                    ], layout=widgets.Layout(padding="4px 0")))
                elif data_type == "image":
                    non_tabular_content.append(widgets.HBox([
                        widgets.HTML(f"<span style='color:#3b82f6;font-weight:500;'>🖼️ Image:</span> {name}")
                    ], layout=widgets.Layout(padding="4px 0")))
                else:
                    non_tabular_content.append(widgets.HBox([
                        widgets.HTML(f"<span style='color:#64748b;font-weight:500;'>📦 Autre:</span> {name}")
                    ], layout=widgets.Layout(padding="4px 0")))

            # Bouton appliquer pour non-tabulaire
            btn_apply_non_tab = widgets.Button(
                description="Appliquer Encoding Non-Tabular",
                button_style="info",
                layout=widgets.Layout(margin="10px 0"))
            btn_apply_non_tab.on_click(self._apply_non_tabular)
            non_tabular_content.append(btn_apply_non_tab)
            self.non_tab_out = widgets.Output()
            non_tabular_content.append(self.non_tab_out)
        else:
            non_tabular_content.append(widgets.HTML("<div style='padding:16px;color:#64748b;'>Aucune donnée non-tabulaire.</div>"))

        tabs_children.append(widgets.VBox(non_tabular_content, layout=widgets.Layout(padding="10px")))
        tabs = widgets.Tab(children=tabs_children)
        tabs.set_title(0, "Tabular"); tabs.set_title(1, "Non-Tabular")
        header  = widgets.HTML(styles.card_html("Encode", "Encoding & Outliers", ""))
        top_bar = widgets.HBox([header], layout=widgets.Layout(
            align_items="center", margin="0 0 12px 0",
            padding="0 0 10px 0", border_bottom="2px solid #ede9fe"))
        self.ui = widgets.VBox(
            [top_bar, tabs],
            layout=widgets.Layout(width="100%", max_width="1000px",
                                   border="1px solid #e5e7eb", padding="18px",
                                   border_radius="10px", background_color="#ffffff"))

    def _on_tab_ds_change(self, change) -> None:
        if change:
            self.current_ds = change["new"]
        if not self.current_ds:
            return
        self._build_enc_table()
        self._build_outliers_table()

    def _build_enc_table(self) -> None:
        df = self.datasets[self.current_ds]
        self._tab_enc_widgets = {}
        headers = widgets.HBox([
            widgets.HTML("<div style='width:220px;font-weight:bold;color:#475569;'>Column</div>"),
            widgets.HTML("<div style='width:250px;font-weight:bold;color:#475569;'>Encoding Action</div>"),
        ], layout=widgets.Layout(border_bottom="2px solid #cbd5e1", padding="0 0 5px 0", margin="0 0 10px 0"))
        rows = [headers]
        for col in df.columns:
            kind = self.meta.get(self.current_ds, {}).get(col, {}).get("kind", "categorical")
            tabular_config = self.config.get("tabular", self.config)
            options_config = tabular_config.get(kind, [])
            opts = [(o["label"], o["value"]) for o in options_config]
            if not opts:
                opts = [("Passthrough", "none"), ("Drop column", "drop")]
            if not any(o[1] == "none" for o in opts):
                opts.insert(0, ("Passthrough", "none"))
            if not any(o[1] == "drop" for o in opts):
                opts.append(("Drop column", "drop"))
            defaults = {"id_like": "drop", "categorical": next((o[1] for o in opts if "onehot" in o[1]), opts[0][1]),
                        "numeric": next((o[1] for o in opts if o[1] in ("std","minmax","robust")), opts[0][1]),
                        "datetime": next((o[1] for o in opts if "extract" in o[1] or "epoch" in o[1]), opts[0][1]),
                        "binary": next((o[1] for o in opts if o[1] in ("label","bool_map")), opts[0][1])}
            default_val = defaults.get(kind, "none")
            if not any(o[1] == default_val for o in opts):
                default_val = opts[0][1]
            lbl_col = widgets.HTML(f"<div style='width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding-top:4px;' title='{col}'><b>{col}</b> <span style='color:#94a3b8;font-size:0.8em;'>[{kind}]</span></div>")
            enc_dd = widgets.Dropdown(options=opts, value=default_val, layout=widgets.Layout(width="240px"))
            self._tab_enc_widgets[col] = {"dd": enc_dd, "kind": kind}
            rows.append(widgets.HBox([lbl_col, enc_dd], layout=widgets.Layout(padding="4px 0", border_bottom="1px solid #f1f5f9")))
        self.enc_container.children = [widgets.HTML("<h4 style='color:#3b82f6;'>Encoding Rules</h4>")] + rows

    def _build_outliers_table(self, change=None) -> None:
        if not getattr(self, "current_ds", None):
            return
        is_after = self.outlier_timing.value == "After Encoding"
        df = self._get_encoded_df() if is_after else self.datasets[self.current_ds]
        self._tab_outlier_widgets = {}
        headers = widgets.HBox([
            widgets.HTML("<div style='width:220px;font-weight:bold;color:#475569;'>Column</div>"),
            widgets.HTML("<div style='width:200px;font-weight:bold;color:#475569;'>Outlier Action</div>"),
            widgets.HTML("<div style='width:150px;font-weight:bold;color:#475569;'>Detected</div>"),
            widgets.HTML("<div style='width:150px;font-weight:bold;color:#475569;'>Create Indicator</div>"),
        ], layout=widgets.Layout(border_bottom="2px solid #cbd5e1", padding="0 0 5px 0", margin="0 0 10px 0"))
        rows = [headers]
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            lbl_col  = widgets.HTML(f"<div style='width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding-top:4px;'><b>{col}</b></div>")
            out_dd   = widgets.Dropdown(options=self.outlier_options, value="none", layout=widgets.Layout(width="180px", margin="0 20px 0 0"))
            lbl_ratio = widgets.HTML("<div style='width:150px;color:#64748b;padding-top:4px;'>-</div>")
            flag_cb  = widgets.Checkbox(value=False, description="Significatif", indent=False, layout=styles.LAYOUT_BTN_STD, disabled=True)
            def _update_ratio(change, df_col=df[col].copy(), col_name=col, lbl=lbl_ratio, cb=flag_cb):
                if change["new"] == "none":
                    lbl.value = "<div style='width:150px;color:#64748b;padding-top:4px;'>-</div>"
                    cb.disabled = True; cb.value = False
                    with self.outlier_plot_out:
                        from IPython.display import clear_output
                        clear_output()
                    return
                cb.disabled = False
                o_act = change["new"]
                n_out, t_out = calc_outliers(df_col, o_act)
                pct = (n_out/t_out)*100 if t_out > 0 else 0
                color = "#ef4444" if pct > 5 else "#f59e0b" if pct > 1 else "#10b981"
                lbl.value = f"<div style='width:150px;color:{color};padding-top:4px;'>{n_out}/{t_out} ({pct:.1f}%)</div>"
                
                with self.outlier_plot_out:
                    from IPython.display import clear_output
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    clear_output(wait=True)
                    df_after = df_col.copy()
                    if o_act == "clip_iqr":
                        q1, q3 = df_after.quantile(0.25), df_after.quantile(0.75)
                        iqr = q3 - q1
                        df_after = df_after.clip(lower=q1-1.5*iqr, upper=q3+1.5*iqr)
                    elif o_act == "drop_zscore":
                        std = df_after.std()
                        if std > 0:
                            df_after = df_after[(((df_after - df_after.mean()) / std).abs() <= 3)]
                    
                    fig, axs = plt.subplots(1, 2, figsize=(10, 3))
                    fig.patch.set_facecolor('#ffffff')
                    sns.histplot(df_after, color="#10b981", ax=axs[0], kde=True, label="After", alpha=0.9)
                    sns.histplot(df_col, color="#ef4444", ax=axs[0], kde=False, label="Removed/Clipped", alpha=0.3)
                    axs[0].set_title(f"Distribution: {col_name}")
                    axs[0].legend()
                    
                    df_plot = pd.DataFrame({
                        "Value": list(df_col.dropna()) + list(df_after.dropna()),
                        "State": ["Avant"]*len(df_col.dropna()) + ["Après"]*len(df_after.dropna())
                    })
                    sns.boxplot(data=df_plot, x="State", y="Value", hue="State", palette={"Avant":"#ef4444", "Après":"#10b981"}, ax=axs[1], legend=False)
                    axs[1].set_title(f"Boxplot: {col_name}")
                    
                    plt.tight_layout()
                    display(fig)
                    plt.close(fig)
            out_dd.observe(_update_ratio, names="value")
            self._tab_outlier_widgets[col] = {"outlier_dd": out_dd, "flag_cb": flag_cb}
            rows.append(widgets.HBox([lbl_col, out_dd, lbl_ratio, flag_cb],
                                      layout=widgets.Layout(padding="4px 0", border_bottom="1px solid #f1f5f9", align_items="center")))
        step_title = "Outliers Rules (Applied After Encoding)" if is_after else "Outliers Rules (Applied Before Encoding)"
        self.outlier_container.children = [widgets.HTML(f"<h4 style='color:#eab308;margin-top:20px;'>{step_title}</h4>")] + rows

    def _apply_tabular(self, b) -> None:
        with self.tab_out:
            clear_output()
            if not getattr(self, "current_ds", None):
                return
            df = self.datasets[self.current_ds].copy()
            timing = self.outlier_timing.value
            
            outlier_params = {
                col: {"o_act": wd["outlier_dd"].value, "flag_it": wd["flag_cb"].value}
                for col, wd in self._tab_outlier_widgets.items()
            }
            
            enc_params = {
                col: {"enc_value": wd["dd"].value, "kind": wd["kind"]}
                for col, wd in self._tab_enc_widgets.items()
            }
            
            if timing == "Before Encoding":
                df = apply_outliers(df, outlier_params)
                
            df, _ = apply_encoding(df, enc_params, self.config)
            
            if timing == "After Encoding":
                df = apply_outliers(df, outlier_params)
                
            # Restoring into state safely handling nested structure
            ds_name = self.current_ds
            if " - " in ds_name and ds_name not in self.state.data_encoded:
                parent, child = ds_name.split(" - ", 1)
                if parent not in self.state.data_encoded:
                    self.state.data_encoded[parent] = {}
                self.state.data_encoded[parent][child] = df
            else:
                self.state.data_encoded[ds_name] = df
                
            self.state.log_step("Data Encoding", "Tabular Encoded",
                                 {"dataset": self.current_ds, 
                                  "outliers_timing": timing,
                                  "params": enc_params,
                                  "outliers": outlier_params,
                                  "fitted_encoders": _})
            display(styles.info_msg(
                f"Encodage appliqué sur '{self.current_ds}'.<br>"
                f"Original : {self.datasets[self.current_ds].shape} → Final : {df.shape}"))
                
            code = build_sklearn_pipeline_code(enc_params, self.config)
            if code:
                display(widgets.HTML("<div style='margin-top:16px;font-weight:bold;color:#334155;'>Sklearn Pipeline Code:</div>"))
                display(widgets.HTML(f"<pre style='background:#f8fafc;padding:12px;border:1px solid #e2e8f0;border-radius:6px;font-size:12px;max-width:800px;overflow-x:auto;'>{code}</pre>"))


    def _detect_non_tabular_type(self, data) -> str:
        """Détecte le type de données non-tabulaires."""
        if hasattr(data, "triples") and hasattr(data, "objects") and hasattr(data, "subjects"):
            return "ontology"
        if hasattr(data, "nodes") and hasattr(data, "edges"):
            return "graph"
        if hasattr(data, "mode") and hasattr(data, "size") and hasattr(data, "convert"):
            return "image"
        return "unknown"

    def _apply_non_tabular(self, b) -> None:
        """Applique l'encoding aux données non-tabulaires (ontologies)."""
        with self.non_tab_out:
            clear_output()
            if not self.non_tabular:
                display(styles.info_msg("Aucune donnée non-tabulaire à encoder."))
                return

            for name, data in self.non_tabular.items():
                data_type = self._detect_non_tabular_type(data)
                
                # Handling hierarchy restoring for non-tabular
                if " - " in name and name not in self.state.data_encoded:
                    parent, child = name.split(" - ", 1)
                    if parent not in self.state.data_encoded:
                        self.state.data_encoded[parent] = {}
                    
                    if data_type == "ontology":
                        self.state.data_encoded[parent][child] = data
                        display(styles.success_msg(f"Ontologie '{name}' : pass-through appliqué."))
                    elif data_type == "graph":
                        self.state.data_encoded[parent][child] = data
                        display(styles.success_msg(f"Graphe '{name}' : pass-through."))
                    elif data_type == "image":
                        self.state.data_encoded[parent][child] = data
                        display(styles.success_msg(f"Image '{name}' : pass-through."))
                    else:
                        self.state.data_encoded[parent][child] = data
                        display(styles.info_msg(f"Donnée '{name}' : pass-through (type inconnu)."))
                else:
                    if data_type == "ontology":
                        self.state.data_encoded[name] = data
                        display(styles.success_msg(f"Ontologie '{name}' : pass-through appliqué."))
                    elif data_type == "graph":
                        self.state.data_encoded[name] = data
                        display(styles.success_msg(f"Graphe '{name}' : pass-through."))
                    elif data_type == "image":
                        self.state.data_encoded[name] = data
                        display(styles.success_msg(f"Image '{name}' : pass-through."))
                    else:
                        self.state.data_encoded[name] = data
                        display(styles.info_msg(f"Donnée '{name}' : pass-through (type inconnu)."))
