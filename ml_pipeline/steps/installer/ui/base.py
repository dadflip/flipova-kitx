import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

from ml_pipeline.styles import styles
from ml_pipeline.steps.installer.logic.packages import check_package, install_packages

class InstallerUI:
    def __init__(self, state):
        self.state = state
        self.config = getattr(self.state, "config", {})
        self._build_ui()

    def _build_guide(self) -> widgets.Accordion:
        guide_html = """
        <div style='font-size:14px; line-height:1.6; color:#374151;'>
            <h4>Guide d'Installation</h4>
            <p>Ce module gère les dépendances de votre environnement. Vous pouvez installer des paquets spécifiques selon vos besoins :</p>
            <ul>
                <li><b>Basiques (Data Science) :</b> Pandas, NumPy, Scikit-learn, etc. Utiles pour les données tabulaires.</li>
                <li><b>Vision (Images/Vidéos) :</b> OpenCV, Pillow. Utiles pour analyser les médias visuels.</li>
                <li><b>Séries Temporelles :</b> Statsmodels. Requis pour décomposition ou tests de stationnarité (ADF).</li>
                <li><b>Graphes & Neo4j :</b> NetworkX, Neo4j driver. Requis pour visualiser et requêter des réseaux de noeuds.</li>
                <li><b>Web & NLP :</b> BeautifulSoup, TextBlob, WordCloud. Indispensables pour scrapper du web ou parser du texte.</li>
                <li><b>Ontologies :</b> RDFLib, Owlready2. Destinés à la fouille de triplets et métadonnées complexes.</li>
            </ul>
        </div>
        """
        out = widgets.Output()
        with out:
            display(HTML(guide_html))
        acc = widgets.Accordion(children=[out], selected_index=None)
        acc.set_title(0, "Guide explicatif de l'installateur")
        return acc

    def _build_ui(self):
        env_cfg = self.config.get("environment", {})
        groups = env_cfg.get("packages", {}).get("groups", [])
        
        self.group_vars = {}
        checkboxes = []
        
        for g in groups:
            is_checked = g.get("default", False)
            chk = widgets.Checkbox(value=is_checked, description=g.get("label", g.get("id")), layout=widgets.Layout(width="auto"))
            self.group_vars[g["id"]] = {"chk": chk, "pkgs": g.get("packages", []), "check_names": g.get("check", [])}
            checkboxes.append(chk)
            
        self.btn_install = widgets.Button(description="Installer les packages sélectionnés", button_style=styles.BTN_PRIMARY)
        self.out = widgets.Output()
        
        self.btn_install.on_click(self._on_install)
        
        header = widgets.HTML(styles.card_html("Installer", "Gestion des dépendances", "Sélectionnez les bibliothèques à installer dans votre espace de travail Jupyter."))
        cb_grid = widgets.GridBox(checkboxes, layout=widgets.Layout(grid_template_columns="repeat(2, 1fr)", gap="10px"))
        
        guide_acc = self._build_guide()
        
        self.ui = widgets.VBox([header, guide_acc, cb_grid, self.btn_install, self.out])
        
    def _on_install(self, b):
        with self.out:
            clear_output(wait=True)
            print("Vérification des packages...")
            to_install = set()
            for gid, info in self.group_vars.items():
                if info["chk"].value:
                    for i, check_name in enumerate(info["check_names"]):
                        if not check_package(check_name):
                            if i < len(info["pkgs"]):
                                to_install.add(info["pkgs"][i])
            if not to_install:
                print("[OK] Tous les packages requis sont déjà installés.")
            else:
                print(f"Installation de : {', '.join(to_install)} ...")
                success, msg = install_packages(list(to_install))
                if success:
                    print("[OK] Installation terminée !")
                else:
                    print(f"[ERREUR] Erreur d'installation : {msg}")

def build_installer_ui(state):
    app = InstallerUI(state)
    display(app.ui)
    return app
