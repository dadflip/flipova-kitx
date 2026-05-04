import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import os
from ml_pipeline.styles import styles
from ml_pipeline.steps.deploy.logic.operations import build_deployment_package

def _section(title, color="#6366f1") -> widgets.HTML:
    return widgets.HTML(f"<div style='display:flex;align-items:center;gap:10px;margin:18px 0 10px 0;'>"
                        f"<div style='width:4px;height:20px;background:{color};border-radius:2px;'></div>"
                        f"<span style='font-size:0.95em;font-weight:700;color:#1e293b;'>{title}</span></div>")

class DeploymentUI:
    def __init__(self, state):
        self.state = state
        self._build_ui()

    def _build_ui(self):
        header = widgets.HTML(styles.card_html("Packaging & Déploiement API", "Génération automatique d'un backend REST FastAPI avec inférence, XAI, et système de ré-entraînement.", ""))
        top_bar = widgets.HBox([header], layout=widgets.Layout(align_items="center", margin="0 0 12px 0", padding="0 0 10px 0", border_bottom="2px solid #ede9fe"))
        
        # 1. Audit XAI Options
        self.chk_xai = widgets.Checkbox(value=True, description="USE_EXPLAINABILITY", style={"description_width": "initial"})
        self.dd_xai_method = widgets.Dropdown(options=["shap", "lime", "both"], value="shap", description="Méthode XAI:", style={"description_width": "initial"})
        
        # 2. Settings Retraining
        self.chk_retrain = widgets.Checkbox(value=True, description="Activer Endpoint de Ré-entraînement", style={"description_width": "initial"})
        
        box_config = widgets.VBox([
            _section("Configuration de l'API & Options", "#10b981"),
            widgets.HBox([self.chk_xai, self.dd_xai_method]),
            self.chk_retrain,
        ], layout=widgets.Layout(padding="12px", border="1px solid #e5e7eb", border_radius="8px", margin="0 0 16px 0", background_color="#f8fafc"))
        
        # 3. Action Buttons
        self.btn_gen_zip = widgets.Button(description="📦 Générer le package (.zip)", button_style=styles.BTN_SUCCESS, layout=widgets.Layout(width="250px"))
        self.btn_docker = widgets.Button(description="🐳 Docker build & instructions", button_style="info", layout=widgets.Layout(width="250px"))
        self.btn_cloud = widgets.Button(description="☁️ Deploy to Cloud (Aide)", button_style="primary", layout=widgets.Layout(width="250px"))
        
        self.btn_gen_zip.on_click(self._on_gen_zip)
        self.btn_docker.on_click(self._on_docker)
        
        box_actions = widgets.VBox([
            _section("Exécution Rapide", "#3b82f6"),
            widgets.HBox([self.btn_gen_zip, self.btn_docker, self.btn_cloud], layout=widgets.Layout(gap="16px")),
        ], layout=widgets.Layout(padding="12px", border="1px solid #e5e7eb", border_radius="8px", margin="0 0 16px 0", background_color="#f8fafc"))

        self.output = widgets.Output()

        self.ui = widgets.VBox([
            top_bar,
            styles.help_box("<b>Déploiement Complet (API Ready) :</b> Générez un package contenant FastAPI, Scikit-Learn Pipeline, SHAP/LIME, et la logique Docker pour pousser votre modèle en production sans effort.", "#6366f1"),
            box_config,
            box_actions,
            self.output
        ], layout=widgets.Layout(width="100%", max_width="1100px", border="1px solid #e5e7eb", padding="18px", border_radius="10px", background_color="#ffffff"))

    def _on_gen_zip(self, b):
        with self.output:
            clear_output(wait=True)
            display(widgets.HTML("<div style='color:#3b82f6;'>🚀 Génération du package en cours...</div>"))
            # Custom Logic based on UI toggles could be injected into state or passed directly
            is_xai = self.chk_xai.value
            method_xai = self.dd_xai_method.value
            has_retrain = self.chk_retrain.value
            zip_path = build_deployment_package(self.state, "ml_deployment_package.zip", is_xai, method_xai, has_retrain)
            display(widgets.HTML(f"<div style='color:#10b981; font-weight:bold;'>✅ Package généré : <a href='{zip_path}' download='{zip_path}'>📥 Télécharger {zip_path}</a></div>"))
            display(widgets.HTML("<div>Contenu du package :<br>- <b>api/main.py</b> : Serveur FastAPI<br>- <b>api/predictor.py</b> : Logique de prétraitement et d'inférence brute<br>- <b>Dockerfile</b> : Environnement isolé prêt à être lancé</div>"))

    def _on_docker(self, b):
        with self.output:
            clear_output(wait=True)
            display(widgets.HTML(\"\"\"
            <div style='background:#1e293b; color:#fbbf24; padding:16px; border-radius:8px; font-family:monospace;'>
                $ unzip ml_deployment_package.zip -d my_ml_app<br>
                $ cd my_ml_app<br>
                $ docker build -t my_fastapi_model .<br>
                $ docker run -p 8000:8000 my_fastapi_model<br>
                <br>
                <span style='color:#94a3b8;'># Accédez à http://localhost:8000/docs pour tester l'API (Swagger UI)</span>
            </div>
            \"\"\"))
