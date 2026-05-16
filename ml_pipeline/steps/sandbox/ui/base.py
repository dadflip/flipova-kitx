import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from ml_pipeline.styles import styles

_PAL  = ["#6366f1","#10b981","#f59e0b","#ef4444","#3b82f6","#8b5cf6","#ec4899"]

def _section(title, color="#6366f1") -> widgets.HTML:
    return widgets.HTML(f"<div style='display:flex;align-items:center;gap:10px;margin:18px 0 10px 0;'>"
                        f"<div style='width:4px;height:20px;background:{color};border-radius:2px;'></div>"
                        f"<span style='font-size:0.95em;font-weight:700;color:#1e293b;'>{title}</span></div>")

def _warn(msg) -> widgets.HTML:
    return widgets.HTML(f"<div style='color:#92400e;background:#fffbeb;border-left:4px solid #f59e0b;padding:8px 12px;font-size:0.85em;border-radius:4px;'>{msg}</div>")


class SandboxUI:
    def __init__(self, state):
        self.state = state
        self.splits = getattr(state, "data_splits", {})
        self.models = getattr(state, "models", {})
        if not self.models:
            self.ui = styles.error_msg("Aucun modèle trouvé. Exécutez d'abord la cellule Modeling.")
            return
        
        self.X_train = self._get_X_train()
        if self.X_train is None:
            self.ui = styles.error_msg("Données X_train introuvables. Impossible de créer le bac à sable.")
            return

        self._build_ui()

    def _get_X_train(self):
        # Pour le bac à sable, utiliser le dataset brut, car le pipeline preprocess dynamiquement
        raw_datasets = getattr(self.state, "data_raw", {})
        if raw_datasets:
            # We take the first one or the one matching current context
            # A simple heuristic: if 'Train' exists (from auto-split), use it. Or first one.
            if 'Train' in raw_datasets:
                return raw_datasets['Train']
            return list(raw_datasets.values())[0]
            
        pred = getattr(self.state, "predictions", {})
        if not pred:
            return self.splits.get("X_train")
        first = list(pred.values())[0] if pred else {}
        return first.get("X_train", self.splits.get("X_train"))

    def _build_form(self, df):
        self.inputs = {}
        items = []
        for col in df.columns:
            # We determine the type
            if pd.api.types.is_numeric_dtype(df[col]):
                med = df[col].median()
                w = widgets.FloatText(value=float(med), description=col, style={'description_width': 'initial'}, layout=widgets.Layout(width='320px'))
            else:
                top = df[col].mode()[0] if not df[col].empty else ""
                w = widgets.Text(value=str(top), description=col, style={'description_width': 'initial'}, layout=widgets.Layout(width='320px'))
            self.inputs[col] = w
            items.append(w)
        
        return widgets.GridBox(items, layout=widgets.Layout(grid_template_columns="repeat(auto-fill, minmax(340px, 1fr))", grid_gap="12px", margin="16px 0", background_color="#f8fafc", padding="16px", border_radius="8px", border="1px solid #e2e8f0"))

    def _best_model_name(self) -> str:
        if hasattr(self.state, "best_model_name") and self.state.best_model_name in self.models:
            return self.state.best_model_name
        return list(self.models.keys())[0]

    def _build_ui(self) -> None:
        header = widgets.HTML(styles.card_html("Sandbox & Explainability", "Simulez une prédiction avec des features unitaires et affichez SHAP", ""))
        top_bar = widgets.HBox([header], layout=widgets.Layout(align_items="center", margin="0 0 12px 0", padding="0 0 10px 0", border_bottom="2px solid #ede9fe"))
        
        best_name = self._best_model_name()
        self.dd_model = widgets.Dropdown(options=list(self.models.keys()), value=best_name, description="Modèle pour prédiction:", style={"description_width": "initial"}, layout=widgets.Layout(width="400px"))
        
        load_random_btn = widgets.Button(description="Remplir avec valeur aléatoire (X_test)", icon="random", button_style="info", layout=widgets.Layout(width="300px"))
        load_random_btn.on_click(self._load_random)
        
        self.int_max_display_shap = widgets.BoundedIntText(value=10, min=3, max=100, step=1, description="Max SHAP vars:", style={"description_width": "initial"}, layout=widgets.Layout(width="180px"))
        
        self.form_box = self._build_form(self.X_train)
        
        self.btn_predict = widgets.Button(description="Prédire & Expliquer (SHAP/LIME)", button_style=styles.BTN_PRIMARY, layout=widgets.Layout(width="280px"))
        self.btn_predict.on_click(self._run_predict_explain)
        
        self.output = widgets.Output()
        
        self.ui = widgets.VBox([
            top_bar,
            styles.help_box("<b>Bac à sable :</b> Entrez manuellement les caractéristiques (features encodées telles qu'attendues par le modèle) pour générer une prédiction à la volée. L'explicabilité (SHAP) montrera l'impact de chaque variable.", "#6366f1"),
            widgets.HBox([self.dd_model, load_random_btn, self.int_max_display_shap], layout=widgets.Layout(gap="16px", align_items="center", margin="0 0 16px 0")),
            widgets.HTML("<b style='color:#374151;'>Caractéristiques (Features) :</b>"),
            self.form_box,
            self.btn_predict,
            self.output
        ], layout=widgets.Layout(width="100%", max_width="1100px", border="1px solid #e5e7eb", padding="18px", border_radius="10px", background_color="#ffffff"))

    def _load_random(self, b):
        raw_datasets = getattr(self.state, "data_raw", {})
        X_test_raw = raw_datasets.get('Test', self.X_train)
        if len(X_test_raw) > 0:
            idx = np.random.randint(0, len(X_test_raw))
            row = X_test_raw.iloc[idx]
            for col, widget in self.inputs.items():
                if col in row:
                    widget.value = row[col]

    def _run_predict_explain(self, b):
        with self.output:
            clear_output(wait=True)
            model_name = self.dd_model.value
            model = self.models.get(model_name)
            
            # Reconstruction du DataFrame singleton
            row_dict = {col: w.value for col, w in self.inputs.items()}
            X_sing = pd.DataFrame([row_dict])
            
            # Conversion type (si besoin sur brut)
            for col in self.X_train.columns:
                if col in X_sing:
                    X_sing[col] = X_sing[col].astype(self.X_train[col].dtype)
                    
            from ml_pipeline.utils.preprocessing import apply_preprocessing
            X_preproc = apply_preprocessing(X_sing, self.state.history)
            
            # Align features with model expecting columns
            if hasattr(model, 'feature_names_in_'):
                expected_cols = list(model.feature_names_in_)
                for c in expected_cols:
                    if c not in X_preproc.columns:
                        X_preproc[c] = 0
                X_preproc = X_preproc[expected_cols]
            else:
                # Fallback to the splits columns if available
                if "X_train" in self.splits:
                    model_cols = self.splits["X_train"].columns
                    for c in model_cols:
                        if c not in X_preproc.columns:
                            X_preproc[c] = 0
                    X_preproc = X_preproc[list(model_cols)]

            # Prediction
            try:
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_preproc)
                    pred = np.argmax(proba, axis=1) if proba.shape[1] > 2 else (proba[:, 1] >= 0.5).astype(int)
                    res_html = f"<div style='font-size:1.6em;color:#1e293b;font-weight:900;'>Prédiction: {pred[0]}</div>"
                    for i, p in enumerate(proba[0]):
                        res_html += f"<div style='color:#64748b;font-size:0.9em;'>Classe {i} (Probabilité) : {p:.4f}</div>"
                else:
                    pred = model.predict(X_preproc)
                    res_html = f"<div style='font-size:1.6em;color:#1e293b;font-weight:900;'>Prédiction (Régression): {pred[0]:.4f}</div>"
            except Exception as e:
                display(_warn(f"Erreur modèle : {e}"))
                return
            
            display(_section(f"Résultat Inférence : {model_name}", "#10b981"))
            display(HTML(f"<div style='background:#f0fdf4;border:1px solid #bbf7d0;padding:16px;border-radius:8px;'>{res_html}</div>"))
            
            # SHAP
            display(_section("Explicabilité (SHAP)", "#8b5cf6"))
            try:
                import shap
            except ImportError:
                display(_warn("Le module 'shap' n'est pas installé. Exécutez l'Installation (S00) du groupe 'explainability'."))
                return
            
            try:
                display(HTML("<div style='color:#64748b;font-size:0.85em;margin-bottom:12px;'>Génération des valeurs SHAP pour cette instance. Note : L'affichage peut prendre quelques secondes.</div>"))
                fig = plt.figure(figsize=(10, 4))
                # Attempt to use TreeExplainer first (XGBoost, Random Forest, etc)
                explainer = None
                shap_values = None
                
                # Check if it's a tree-based model
                is_tree = any(name in str(type(model)).lower() for name in ["forest", "tree", "xgb", "lgbm", "catboost"])
                
                if is_tree:
                    try:
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer(X_preproc)
                    except:
                        pass
                
                if explainer is None:
                    # Fallback to KernelExplainer or just standard Explainer
                    # We need background data for KernelExplainer
                    bg_X_train = self.splits.get("X_train", self.X_train)
                    bg_data = shap.maskers.Independent(bg_X_train.sample(min(100, len(bg_X_train)), random_state=42))
                    explainer = shap.Explainer(model.predict, bg_data) # use model.predict for general models
                    shap_values = explainer(X_preproc)
                
                # Plot waterfall for the singular prediction
                # If multiclass, shap_values might be a list or have 3 dimensions
                if len(shap_values.shape) == 3:
                     # Multiclass: explain the predicted class
                     pred_class = int(pred[0])
                     shap.plots.waterfall(shap_values[0, :, pred_class], max_display=self.int_max_display_shap.value, show=False)
                else:
                     shap.plots.waterfall(shap_values[0], max_display=self.int_max_display_shap.value, show=False)
                     
                plt.tight_layout()
                display(fig)
                plt.close(fig)
                
                display(HTML("<div style='font-size:0.85em;color:#475569;margin-top:8px;'><b>Comment lire le graphe SHAP :</b><br><li>En rouge/rose : Les features qui augmentent la prédiction ou poussent vers la classe positive (ou classe de destination).</li><li>En bleu : Les features qui diminuent la prédiction ou poussent vers la classe négative.</li><li>La valeur <b>f(x)</b> est la prédiction (log-odds pour certains modèles de classification, ou la valeur directe pour régression).</li><li>La valeur <b>E[f(x)]</b> est la prédiction moyenne sur l'ensemble de données d'entraînement.</li></div>"))
                
                self.state.log_step("Sandbox", "Prediction simulated with SHAP", {"model": model_name, "prediction": float(pred[0]) if not hasattr(pred[0], "item") else pred[0].item()})
            except Exception as e:
                display(_warn(f"Erreur détaillée lors de la génération avec SHAP : {e}"))
                plt.close('all')
