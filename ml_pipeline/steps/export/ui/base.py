import os
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from ml_pipeline.styles import styles

from ..logic.operations import export_python_script, export_models, generate_report

class ReportGenerator:
    """Génère le script Python, exporte les modèles et produit un rapport HTML."""

    def __init__(self, state):
        self.state = state
        self._build_ui()

    def _build_ui(self) -> None:
        header  = widgets.HTML(styles.card_html("Export", "Pipeline Artifact Exporter", ""))
        top_bar = widgets.HBox([header], layout=widgets.Layout(
            align_items="center", margin="0 0 12px 0",
            padding="0 0 10px 0", border_bottom="2px solid #ede9fe"))
        self.btn_export = widgets.Button(description="Generate & Export",
                                          button_style="success",
                                          layout=widgets.Layout(width="200px"))
        self.btn_export.on_click(self._on_export)
        self.output = widgets.Output()
        self.ui = widgets.VBox([
            top_bar,
            styles.help_box(
                "<b>Export Artifacts :</b> génère un script Python depuis l'historique du pipeline, "
                "exporte les modèles entraînés dans <code>trained_models.pkl</code> et produit un rapport HTML.",
                "#10b981"),
            self.btn_export,
            self.output,
        ], layout=widgets.Layout(width="100%", max_width="1000px",
                                  border="1px solid #e5e7eb", padding="18px",
                                  border_radius="10px", background_color="#ffffff"))

    def _on_export(self, b) -> None:
        with self.output:
            clear_output()
            self.generate_all()

    def generate_all(self) -> None:
        export_python_script(self.state, os.path.join(os.getcwd(), "exported_pipeline.py"))
        export_models(self.state, os.path.join(os.getcwd(), "trained_models.pkl"))
        generate_report(self.state, os.path.join(os.getcwd(), "execution_report.html"))
        display(styles.success_msg(
            "[SUCCESS] Export complet : <b>exported_pipeline.py</b>, "
            "<b>trained_models.pkl</b>, <b>execution_report.html</b>."))
