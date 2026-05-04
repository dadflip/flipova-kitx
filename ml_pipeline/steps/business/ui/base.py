"""Étape 2 — Contexte métier et domaine ML (BusinessEditorUI)."""
from __future__ import annotations
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

from ml_pipeline.styles import styles
from ml_pipeline.steps.business.logic.context import validate_business_context

class BusinessEditorUI:
    """Définition du contexte métier et de la tâche ML."""

    def __init__(self, state):
        self.state = state
        self._build_ui()

    def _build_ui(self) -> None:
        header  = widgets.HTML(styles.card_html("Business Context", "Contexte & Définition du problème", ""))
        top_bar = widgets.HBox([header], layout=widgets.Layout(
            align_items="center", margin="0 0 12px 0",
            padding="0 0 10px 0", border_bottom="2px solid #ede9fe"))

        # Résumé des données chargées
        data_summary = []
        
        # Helper to format data summary items
        def _format_item(name, data, dtype):
            if isinstance(data, pd.DataFrame):
                cols = len(data.columns)
                rows = len(data)
                col_names = ", ".join(list(data.columns)[:5]) + ("..." if cols > 5 else "")
                return (f"<li><b>{name}</b>: Tabular ({rows:,} rows, {cols} cols) "
                        f"<br><span style='font-size:0.85em;color:#6b7280;'>Cols: {col_names}</span></li>")
            elif dtype == "ontology" or "rdflib" in type(data).__module__:
                num_triples = len(data) if hasattr(data, "__len__") else "?"
                return f"<li><b>{name}</b>: Ontology ({num_triples} triples)</li>"
            elif dtype in ["graph", "neo4j"] or "networkx" in type(data).__module__:
                nodes = getattr(data, "number_of_nodes", lambda: "?")() if callable(getattr(data, "number_of_nodes", None)) else "?"
                edges = getattr(data, "number_of_edges", lambda: "?")() if callable(getattr(data, "number_of_edges", None)) else "?"
                return f"<li><b>{name}</b>: Graph ({nodes} nodes, {edges} edges)</li>"
            elif dtype in ["image", "video"]:
                size_info = getattr(data, "shape", "?")
                return f"<li><b>{name}</b>: {dtype.capitalize()} <span style='font-size:0.85em;color:#6b7280;'>{size_info}</span></li>"
            else:
                return f"<li><b>{name}</b>: {dtype.capitalize() if isinstance(dtype, str) else type(data).__name__}</li>"
        
        seen_names = set()
        
        if hasattr(self.state, "raw_data_dict"):
            for name, data in self.state.raw_data_dict.items():
                seen_names.add(name)
                dtype = self.state.data_types.get(name, "tabular" if isinstance(data, pd.DataFrame) else "unknown")
                data_summary.append(_format_item(name, data, dtype))
                    
        if hasattr(self.state, "ontologies") and self.state.ontologies:
            for name, g in self.state.ontologies.items():
                if name not in seen_names:
                    seen_names.add(name)
                    data_summary.append(f"<li><b>{name}</b>: Ontology ({len(g)} triples)</li>")
                
        if hasattr(self.state, "data_types"):
            for name, dtype in self.state.data_types.items():
                if name not in seen_names:
                    data = self.state.data_raw.get(name) if hasattr(self.state, "data_raw") else None
                    if data is not None:
                        data_summary.append(_format_item(name, data, dtype))
                    else:
                        data_summary.append(f"<li><b>{name}</b>: {dtype.capitalize()} (Not present in data_raw)</li>")

        data_hint = (
            f"<div style='margin-top:8px;'><b>Datasets chargés :</b>"
            f"<ul style='margin-top:4px;padding-left:20px;'>"
            f"{''.join(data_summary) if data_summary else '<li><i>Aucune donnée chargée.</i></li>'}"
            f"</ul></div>"
        )
        description_box = widgets.HTML(
            f"<div style='padding:16px;margin-bottom:16px;border:1px solid #e2e8f0;"
            f"border-radius:8px;background:#f8fafc;font-family:sans-serif;color:#334155;'>"
            f"<p style='margin-top:0;font-weight:600;'>Configurez votre projet ML</p>"
            f"{data_hint}</div>"
        )

        self.project_name = widgets.Text(description="Titre projet :", placeholder="ex. Predictive Maintenance",
                                          layout=styles.LAYOUT_W95, style={"description_width": "160px"})
        self.problem_ta   = widgets.Textarea(description="Problème métier :", placeholder="Décrivez le problème métier en détail...",
                                              layout=widgets.Layout(width="95%", height="80px"),
                                              style={"description_width": "160px"})
        self.impact_ta    = widgets.Textarea(description="Impact & ROI :", placeholder="ROI, usage en production, intégration prévue...",
                                              layout=widgets.Layout(width="95%", height="80px"),
                                              style={"description_width": "160px"})
        self.latency_req  = widgets.Dropdown(
            description="Latence cible :",
            options=["Real-time (<10ms)", "Online (<100ms)", "Batch (seconds+)", "No rigid constraint"],
            value="No rigid constraint", layout=styles.LAYOUT_W95, style={"description_width": "160px"})
        self.interpretability_req = widgets.Checkbox(
            description=" Interprétabilité requise (modèles explicables uniquement : Linear, Trees, etc.)",
            value=False, layout=styles.LAYOUT_W95, style={"description_width": "initial"}, indent=False)

        # Handle config gracefully if domains is missing
        domain_cfg = self.state.config.get("domains", {}).get("supported", []) if getattr(self.state, "config", None) else []
        domain_options = [(d["label"], d["value"]) for d in domain_cfg]
        if not domain_options:
            domain_options = [("Classification (Binary)", "classification_binary"), ("No domains in config", "none")]
            
        self.domain_label_map = {d["value"]: d["label"] for d in domain_cfg}
        
        default_domain = self._guess_default_domain(domain_options)
        
        self.domain_dd = widgets.Dropdown(options=domain_options, value=default_domain, description="Domaine ML :",
                                           layout=styles.LAYOUT_W95, style={"description_width": "160px"})
        self.dynamic_settings = widgets.VBox([])
        self.dyn_widgets: dict = {}

        self.btn_save = widgets.Button(description="Valider le contexte", button_style=styles.BTN_PRIMARY,
                                        icon="check", layout=widgets.Layout(width="max-content",
                                                                              padding="4px 20px", margin="10px 0 0 0"))
        self.out_msg = widgets.Output()

        self.domain_dd.observe(self._on_domain_change, names="value")
        self.btn_save.on_click(self._on_save)
        self._on_domain_change({"new": self.domain_dd.value})

        form_box = widgets.VBox([
            widgets.HTML("<b style='color:#334155;'>1. Définition métier</b>"),
            self.project_name, self.problem_ta, self.impact_ta,
            self.latency_req, self.interpretability_req,
            widgets.HTML("<hr style='border:1px solid #e2e8f0;margin:15px 0;'>"),
            widgets.HTML("<b style='color:#334155;'>2. Tâche Machine Learning</b>"),
            self.domain_dd, self.dynamic_settings,
            widgets.HTML("<hr style='border:1px solid #e2e8f0;margin:15px 0;'>"),
            self.btn_save,
        ], layout=widgets.Layout(padding="20px", border="1px solid #f1f5f9",
                                  background_color="#ffffff", border_radius="8px", gap="5px"))

        self.ui = widgets.VBox(
            [top_bar, description_box, form_box, self.out_msg],
            layout=widgets.Layout(width="100%", max_width="1000px", border="1px solid #e5e7eb",
                                   padding="18px", border_radius="10px", background_color="#ffffff")
        )

    def _guess_default_domain(self, domain_options: list) -> str:
        dtypes = list(self.state.data_types.values())
        if "ontology" in dtypes: return "ontology"
        if "neo4j" in dtypes or "graph" in dtypes: return "graph"
        if "image" in dtypes or "video" in dtypes: return "computer_vision"
        if "timeseries" in dtypes: return "timeseries"
        if "text" in dtypes or "web" in dtypes: return "nlp"
        
        # Return first valid domain if no specific match
        return domain_options[0][1] if domain_options else "none"

    def _get_tabular_columns(self) -> list[str]:
        cols = ["(None)"]
        if hasattr(self.state, "raw_data_dict"):
            for v in self.state.raw_data_dict.values():
                if isinstance(v, pd.DataFrame):
                    cols.extend([str(c) for c in v.columns])
        if hasattr(self.state, "data_raw"):
            for v in self.state.data_raw.values():
                if isinstance(v, pd.DataFrame):
                    cols.extend([str(c) for c in v.columns])
        return list(dict.fromkeys(cols))

    def _on_domain_change(self, change) -> None:
        domain   = change["new"]
        children = []
        self.dyn_widgets = {}
        cols = self._get_tabular_columns()

        def _dd(key, desc, opts, val=None):
            w = widgets.Dropdown(options=opts, value=val or (opts[0] if opts else None),
                                  description=desc, layout=styles.LAYOUT_W95,
                                  style={"description_width": "160px"})
            self.dyn_widgets[key] = w
            return w

        def _combo(key, desc, placeholder=""):
            w = widgets.Combobox(options=cols, description=desc, placeholder=placeholder,
                                  layout=styles.LAYOUT_W95, style={"description_width": "160px"})
            self.dyn_widgets[key] = w
            return w

        def _text(key, desc, placeholder=""):
            w = widgets.Text(description=desc, placeholder=placeholder,
                              layout=styles.LAYOUT_W95, style={"description_width": "160px"})
            self.dyn_widgets[key] = w
            return w

        def _int(key, desc, val=3):
            w = widgets.IntText(value=val, description=desc,
                                 layout=styles.LAYOUT_W95, style={"description_width": "160px"})
            self.dyn_widgets[key] = w
            return w

        # Handle config gracefully
        tasks_by_domain = self.state.config.get("domains", {}).get("tasks_by_domain", {}) if getattr(self.state, "config", None) else {}

        if domain == "classification_binary":
            children = [_combo("target", "Variable cible :", "Variable à prédire"),
                        _text("pos_label", "Classe positive :", "ex. 1 ou 'Yes'"),
                        _dd("metric", "Métrique :", ["F1-Score", "ROC-AUC", "Accuracy", "Precision", "Recall"]),
                        _text("features_exclude", "Exclure cols :", "Colonnes séparées par virgule")]
        elif domain == "classification_multiclass":
            children = [_combo("target", "Variable cible :", "Variable à prédire"),
                        _dd("metric", "Métrique :", ["Macro F1", "Micro F1", "Weighted F1", "Accuracy"]),
                        _text("features_exclude", "Exclure cols :", "Colonnes séparées par virgule")]
        elif domain == "regression_continuous":
            children = [_combo("target", "Variable cible :", "Variable à prédire"),
                        _dd("metric", "Métrique :", ["RMSE", "MAE", "R2", "MAPE"]),
                        _text("features_exclude", "Exclure cols :", "Colonnes séparées par virgule")]
        elif domain == "clustering":
            children = [_int("expected_clusters", "Clusters attendus :", 3),
                        _dd("metric", "Métrique :", ["Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz Index"]),
                        _text("features_exclude", "Exclure cols :", "Colonnes à ignorer (ex. IDs)")]
        elif domain == "ontology":
            onto_cfg = tasks_by_domain.get("ontology", {})
            children = [_dd("onto_task", "Tâche :", onto_cfg.get("tasks", ["Consistency Checking", "Knowledge Graph Completion", "Entity Linking", "Semantic Reasoning"])),
                        _dd("inference_level", "Inférence :", onto_cfg.get("inference", ["RDFS", "OWL-DL", "Custom Rule System"]))]
        elif domain == "nlp":
            nlp_cfg = tasks_by_domain.get("nlp", {})
            children = [_dd("nlp_task", "Tâche NLP :", nlp_cfg.get("tasks", ["Text Classification", "Sentiment Analysis", "NER", "Summarization"])),
                        _combo("text_col", "Colonne texte :", "Variable texte principale"),
                        _combo("target", "Variable cible :", "(Optionnel) pour NLP supervisé")]
        elif domain == "computer_vision":
            cv_cfg = tasks_by_domain.get("computer_vision", {})
            children = [_dd("cv_task", "Tâche CV :", cv_cfg.get("tasks", ["Image Classification", "Object Detection", "Image Segmentation"])),
                        _text("img_shape", "Taille cible :", "ex. 224,224")]
        elif domain == "graph":
            graph_cfg = tasks_by_domain.get("graph", {})
            children = [_dd("graph_task", "Tâche Graph :", graph_cfg.get("tasks", ["Node Classification", "Link Prediction", "Community Detection", "Graph Classification"])),
                        _combo("node_target", "Cible nœud :", "(Optionnel) Variable à prédire pour nœuds"),
                        _combo("edge_target", "Cible arête :", "(Optionnel) Variable à prédire pour arêtes")]
        elif domain == "timeseries":
            children = [_combo("ts_target", "Cible :", "Variable à prévoir"),
                        _combo("ts_time_col", "Colonne temps :", "Variable datetime"),
                        _int("ts_horizon", "Horizon prévision :", 7),
                        _dd("metric", "Métrique :", ["RMSE", "MAE", "MAPE", "sMAPE"])]
        else:
            # Fallback for unrecognized domains
            children = [_text("custom_task", "Type de tâche :", "Définir manuellement la tâche..."),
                        _text("custom_metric", "Métrique principale :", "ex. RMSE, F1-Score...")]

        self.dynamic_settings.children = children

    def _on_save(self, btn) -> None:
        with self.out_msg:
            clear_output()
            dyn_params = {k: w.value for k, w in self.dyn_widgets.items()}
            
            context_data = validate_business_context(
                project_name=self.project_name.value,
                domain=self.domain_dd.value,
                problem=self.problem_ta.value,
                impact=self.impact_ta.value,
                latency_req=self.latency_req.value,
                interpretability=self.interpretability_req.value,
                dyn_params=dyn_params
            )
            
            self.state.business_context = context_data
            self.state.log_step("Business Context", "Context Defined", self.state.business_context)

            domain_lbl = self.domain_label_map.get(self.domain_dd.value, self.domain_dd.value)
            
            # Format custom dyn params elegantly
            dyn_html = ""
            for k, v in dyn_params.items():
                if v:
                    dyn_html += f"<div style='margin-bottom:8px;'><span style='color:#64748b;font-size:0.85em;text-transform:uppercase;letter-spacing:0.05em;'>{k}</span><br><span style='color:#1e293b;font-weight:500;'>{v}</span></div>"

            display(HTML(f'''
            <div style='margin-top:20px; border-left:4px solid #10b981; border-radius:0 8px 8px 0; background-color:#ffffff; box-shadow:0 2px 5px rgba(0,0,0,0.05); padding:20px; font-family:sans-serif;'>
                <div style='display:flex; align-items:center; margin-bottom:20px; border-bottom:1px solid #f1f5f9; padding-bottom:15px;'>
                    <div style='background-color:#10b981; color:#fff; width:28px; height:28px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:bold; margin-right:12px;'>✓</div>
                    <h3 style='margin:0; color:#064e3b; font-size:1.2em;'>Contexte Métier Validé</h3>
                </div>
                
                <div style='display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-bottom:20px;'>
                    <div style='background:#f8fafc; padding:15px; border-radius:8px;'>
                        <h4 style='margin:0 0 10px 0; color:#334155; font-size:0.95em; border-bottom:1px solid #e2e8f0; padding-bottom:6px;'>Projet & Objectifs</h4>
                        <div style='margin-bottom:10px;'><span style='color:#64748b;font-size:0.85em;'>TITRE</span><br><span style='color:#1e293b;font-weight:600;font-size:1.1em;'>{self.project_name.value or "Non spécifié"}</span></div>
                        <div style='margin-bottom:10px;'><span style='color:#64748b;font-size:0.85em;'>PROBLÈME</span><br><span style='color:#334155;'>{self.problem_ta.value or "Non spécifié"}</span></div>
                        <div><span style='color:#64748b;font-size:0.85em;'>IMPACT ATTENDU</span><br><span style='color:#334155;'>{self.impact_ta.value or "Non spécifié"}</span></div>
                    </div>
                    
                    <div style='background:#f8fafc; padding:15px; border-radius:8px;'>
                        <h4 style='margin:0 0 10px 0; color:#334155; font-size:0.95em; border-bottom:1px solid #e2e8f0; padding-bottom:6px;'>Spécifications ML</h4>
                        <div style='margin-bottom:10px;'><span style='color:#64748b;font-size:0.85em;'>DOMAINE</span><br><span style='background:#dbeafe;color:#1e40af;padding:2px 8px;border-radius:12px;font-size:0.9em;font-weight:600;'>{domain_lbl}</span></div>
                        <div style='margin-bottom:10px;'><span style='color:#64748b;font-size:0.85em;'>CONTRAINTES</span><br>
                            <span style='display:inline-block;background:#f1f5f9;border:1px solid #e2e8f0;padding:2px 6px;border-radius:4px;font-size:0.85em;margin-right:6px;'>⏱️ {self.latency_req.value}</span>
                            {"<span style='display:inline-block;background:#f1f5f9;border:1px solid #e2e8f0;padding:2px 6px;border-radius:4px;font-size:0.85em;color:#b45309;'>🔍 White-box</span>" if self.interpretability_req.value else ""}
                        </div>
                        <div style='margin-top:10px;'>
                            <h5 style='margin:10px 0 6px 0; color:#64748b; font-size:0.85em;'>PARAMÈTRES TECHNIQUES</h5>
                            {dyn_html or "<span style='color:#94a3b8;font-style:italic;'>Aucun paramètre</span>"}
                        </div>
                    </div>
                </div>
            </div>
            '''))

def runner(state) -> BusinessEditorUI:
    editor = BusinessEditorUI(state)
    display(editor.ui)
    return editor
