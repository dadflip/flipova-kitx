import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import pandas as pd
import numpy as np

from ml_pipeline.styles import styles
from ml_pipeline.steps.eda.ui.dashboard import PlotDashboard
from ml_pipeline.steps.eda.logic.tabular import EDAVisualizerUtils, infer_types

_SEP = widgets.HTML("<div style='height:1px;background:#e5e7eb;margin:10px 0;'></div>")

class UltimateEDA:
    """Interface EDA complète."""

    def __init__(self, state):
        self.state = state
        if not hasattr(self.state, "eda_dashboard"):
            self.state.eda_dashboard = []
        self.dashboard = PlotDashboard(state=self.state)
        
        self.all_datasets: dict = {}
        for k, v in self.state.data_raw.items():
            self.all_datasets[f"[RAW] {k}"] = v
        for k, v in getattr(self.state, "data_cleaned", {}).items():
            self.all_datasets[f"[CLN] {k}"] = v
        for k, v in getattr(self.state, "data_encoded", {}).items():
            self.all_datasets[f"[ENC] {k}"] = v
            
        self.meta: dict = {}
        if not hasattr(self.state, "meta"):
            self.state.meta = {}
            
        for k, v in self.all_datasets.items():
            if isinstance(v, pd.DataFrame):
                orig_key = k.split(" ", 1)[1] if " " in k else k
                if orig_key not in self.state.meta:
                    self.state.meta[orig_key] = {}
                enc_cfg = getattr(self.state, "config", {}).get("encoding", {})
                tabular_cfg = enc_cfg.get("tabular", enc_cfg)
                tabular_types = list(tabular_cfg.keys()) if hasattr(self.state, "config") else None
                inferred = infer_types(v, available_types=tabular_types)
                
                for col_name, col_info in inferred.items():
                    if col_name not in self.state.meta[orig_key]:
                        self.state.meta[orig_key][col_name] = col_info
                    else:
                        user_kind = self.state.meta[orig_key][col_name].get("kind", col_info["kind"])
                        self.state.meta[orig_key][col_name].update(col_info)
                        self.state.meta[orig_key][col_name]["kind"] = user_kind
                self.meta[k] = self.state.meta[orig_key]
                
        self.state.visualizers = EDAVisualizerUtils()
        self.current_ds: str | None = None
        self._build_ui()

    def reset_state(self) -> None:
        for attr in ("data_raw", "data_cleaned", "data_encoded", "data_splits",
                     "data_types", "meta", "business_context", "models", "history",
                     "eda_dashboard", "config"):
            setattr(self.state, attr, {} if attr != "history" and attr != "eda_dashboard" else [])
        self.dashboard._entries.clear()
        self.dashboard._update_label()
        with self.dashboard._grid_out:
            clear_output(wait=True)
        self._reset_msg.value = (
            "<div style='color:#b45309;background:#fef3c7;border-left:4px solid #f59e0b;"
            "padding:8px 12px;font-size:0.85em;border-radius:4px;margin-top:6px;'>"
            "<b>All state has been reset.</b> Reload your data and config to continue.</div>"
        )

    def _build_reset_bar(self) -> widgets.HBox:
        reset_btn = widgets.Button(
            description="⟳ Reset All State — WILL DELETE ALL LOADED DATASETS",
            button_style="danger", layout=widgets.Layout(width="auto", height="32px"))
        self._reset_msg = widgets.HTML("")
        reset_btn.on_click(lambda b: self.reset_state())
        return widgets.HBox(
            [reset_btn, self._reset_msg],
            layout=widgets.Layout(align_items="center", gap="10px", padding="6px 12px",
                                   border="1px solid #fecaca", border_radius="8px",
                                   background_color="#fff5f5", margin="0 0 12px 0"))

    def _build_guide(self) -> widgets.Accordion:
        guide_html = """
        <div style='font-size:14px; line-height:1.6; color:#374151;'>
            <h4>Guide EDA (Exploratory Data Analysis)</h4>
            <ul>
                <li><b>Tabulaire :</b> Explorez les statistiques univariées (distribution), bivariées (relations entre deux variables), l'Analyse en Composantes Principales (PCA) et la matrice de corrélation.</li>
                <li><b>Séries Temporelles :</b> Affichez vos mesures chronologiques, calculez les tendances par fenêtre glissante, décomposez (saisonnalité), analysez l'autocorrélation (ACF/PACF) et testez la stationnarité (ADF).</li>
                <li><b>Images :</b> Visualisez les histogrammes de couleurs, inspectez les canaux RGB, appliquez des filtres (flou, contours Sobel), et extrayez les couleurs dominantes par clustering KMeans.</li>
                <li><b>Texte :</b> Générez des nuages de mots, analysez les N-grammes les plus fréquents, évaluez le sentiment global (polarité) et étudiez la complexité du texte (longueur des phrases/mots).</li>
                <li><b>Graphes & Neo4j :</b> Affichez la typologie, la distribution des nœuds/relations, testez les réseaux via Cypher, détectez les communautés et calculez les métriques de centralité.</li>
                <li><b>Ontologies :</b> Explorez la hiérarchie des classes, vérifiez les Domain/Range des propriétés, parcourez les triplets via des requêtes directes.</li>
                <li><b>Vidéos :</b> Jouez et extrayez des frames, identifiez les coupes de scènes, créez une chronologie couleur globale et observez l'historique de mouvement (MHI).</li>
                <li><b>Pages Web :</b> Inspectez les sélecteurs DOM, compteurs tags/headers, et analysez les métadonnées de référencement incluses.</li>
            </ul>
        </div>
        """
        out = widgets.Output()
        with out:
            display(HTML(guide_html))
        acc = widgets.Accordion(children=[out], selected_index=None)
        acc.set_title(0, "Guide explicatif des Outils EDA")
        return acc

    def _build_ui(self) -> None:
        if not getattr(self.state, "config", None):
            self.ui = styles.error_msg("[ERROR] Configuration non chargée. Exécutez d'abord l'étape Config.")
            return
            
        if not self.all_datasets:
            self.ui = widgets.HTML(
                "<div style='padding:12px;color:#b91c1c;'>"
                "[WARNING] Aucun dataset disponible. Chargez des données d'abord.</div>")
            return
            
        self.ds_selector = widgets.Dropdown(
            options=list(self.all_datasets.keys()),
            description="Dataset:", layout=widgets.Layout(width="360px"))
        self.ds_selector.observe(self.on_ds_change, names="value")
        self.current_ds = self.ds_selector.value
        
        header = widgets.HTML(styles.card_html("EDA", "Exploratory Data Analysis", ""))
        top_bar = widgets.HBox(
            [header, widgets.HTML("<div style='flex:1'></div>"), self.ds_selector],
            layout=widgets.Layout(align_items="center", justify_content="space-between",
                                   margin="0 0 12px 0", padding="0 0 10px 0",
                                   border_bottom="2px solid #ede9fe"))
        reset_bar = self._build_reset_bar()
        guide_acc = self._build_guide()
        self.dynamic_ui = widgets.VBox([])
        
        self.ui = widgets.VBox(
            [top_bar, guide_acc, reset_bar, self.dynamic_ui, self.dashboard.widget],
            layout=widgets.Layout(width="100%", max_width="1000px", border="1px solid #e5e7eb",
                                   padding="18px", border_radius="10px", background_color="#ffffff"))
        self.on_ds_change(None)

    def on_ds_change(self, change) -> None:
        if change:
            self.current_ds = change["new"]
        if not self.current_ds:
            return
            
        data = self.all_datasets[self.current_ds]
        orig_key = self.current_ds.split(" ", 1)[1] if " " in self.current_ds else self.current_ds
        ds_type = self.state.data_types.get(orig_key, "tabular")
        
        # --- Gestion des Dictionnaires (Dossier Local) ---
        if isinstance(data, dict) and not isinstance(data, pd.DataFrame) and ds_type not in ("json", "web", "neo4j"):
            # Cas spécial : Ontologie
            if ds_type == "ontology":
                import rdflib
                merged_g = rdflib.Graph()
                for g in data.values():
                    if isinstance(g, rdflib.Graph):
                        merged_g += g
                self._dispatch_type(ds_type, merged_g)
            else:
                self._build_folder_ui(ds_type, data)
        else:
            self._dispatch_type(ds_type, data)

    def _build_folder_ui(self, ds_type, data_dict):
        # UI pour explorer un dossier (sélection d'un fichier à explorer)
        keys = list(data_dict.keys())
        if not keys:
            self.dynamic_ui.children = [widgets.HTML("<i style='color:red;'>Dossier vide.</i>")]
            return
            
        dropdown = widgets.Dropdown(options=keys, description="Fichier:", layout=widgets.Layout(width="300px"))
        container = widgets.VBox([])
        
        def _on_file_select(change):
            selected = change["new"] if change else dropdown.value
            file_data = data_dict[selected]
            
            old_dynamic = self.dynamic_ui
            self.dynamic_ui = container
            self._dispatch_type(ds_type, file_data)
            self.dynamic_ui = old_dynamic

        dropdown.observe(_on_file_select, names="value")
        _on_file_select(None)
        
        self.dynamic_ui.children = [
            widgets.HTML(f"<div style='margin-bottom:10px; font-weight:bold;'>Mode Dossier - Exploration d'un fichier à la fois ({len(keys)} éléments)</div>"),
            dropdown,
            container
        ]

    def _dispatch_type(self, ds_type, data):
        if ds_type in ("tabular", "csv", "sklearn", "clipboard", "excel") or (isinstance(data, pd.DataFrame) and ds_type != "timeseries"):
            from .tabular import build_tabular_ui
            build_tabular_ui(self, data)
        elif ds_type == "timeseries":
            from .timeseries import build_timeseries_ui
            build_timeseries_ui(self, data)
        elif ds_type == "image":
            from .image import build_image_ui
            build_image_ui(self, data)
        elif ds_type == "text":
            from .text import build_text_ui
            build_text_ui(self, data)
        elif ds_type == "graph":
            from .graph import build_graph_ui
            build_graph_ui(self, data)
        elif ds_type == "ontology":
            from .ontology import build_ontology_ui
            build_ontology_ui(self, data)
        elif ds_type == "video":
            from .video import build_video_ui
            build_video_ui(self, data)
        elif ds_type == "neo4j":
            from .neo4j_eda import build_neo4j_ui
            build_neo4j_ui(self, data)
        elif ds_type == "web":
            from .web import build_web_ui
            build_web_ui(self, data)
        else:
            self.dynamic_ui.children = [widgets.HTML(
                f"<div style='padding:12px;'>[INFO] Dataset ({ds_type}). "
                f"Visualisation limitée.</div>")]

def runner(state) -> UltimateEDA:
    if not hasattr(state, "eda_dashboard"):
        state.eda_dashboard = []
    eda = UltimateEDA(state)
    display(eda.ui)
    return eda
