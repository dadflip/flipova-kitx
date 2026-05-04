import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

from ml_pipeline.styles import styles
from ml_pipeline.steps.installer.logic.packages import check_package, install_packages, uninstall_packages

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
        rows = []
        
        for g in groups:
            is_checked = g.get("default", False)
            chk = widgets.Checkbox(value=is_checked, description=g.get("label", g.get("id")), layout=widgets.Layout(width="300px"))
            status = widgets.HTML("<span style='color:gray;'>[?]</span>")
            
            def create_check_cb(group_info, st_label):
                def _check(b):
                    installed = True
                    for check_name in group_info["check_names"]:
                        if not check_package(check_name):
                            installed = False
                            break
                    if installed:
                        st_label.value = "<span style='color:green; font-weight:bold;'>[Installé]</span>"
                    else:
                        st_label.value = "<span style='color:red; font-weight:bold;'>[Non Installé]</span>"
                return _check
            
            btn_chk = widgets.Button(description="Vérifier", button_style='info', layout=widgets.Layout(width="80px"))
            
            self.group_vars[g["id"]] = {"chk": chk, "pkgs": g.get("packages", []), "check_names": g.get("check", []), "status": status, "btn_chk": btn_chk}
            btn_chk.on_click(create_check_cb(self.group_vars[g["id"]], status))
            
            row = widgets.HBox([chk, status, btn_chk])
            rows.append(row)
            
        self.btn_install = widgets.Button(description="Installer séléctionnés", button_style=styles.BTN_PRIMARY)
        self.btn_uninstall = widgets.Button(description="Désinstaller séléctionnés", button_style='danger')
        self.btn_select_all = widgets.Button(description="Tout sélectionner")
        self.btn_deselect_all = widgets.Button(description="Tout désélectionner")
        self.btn_check_all = widgets.Button(description="Vérifier (Global)", button_style='warning')
        
        btn_bar1 = widgets.HBox([self.btn_select_all, self.btn_deselect_all, self.btn_check_all])
        btn_bar2 = widgets.HBox([self.btn_install, self.btn_uninstall])
        
        self.out = widgets.Output()
        
        self.btn_install.on_click(self._on_install)
        self.btn_uninstall.on_click(self._on_uninstall)
        self.btn_select_all.on_click(self._on_select_all)
        self.btn_deselect_all.on_click(self._on_deselect_all)
        self.btn_check_all.on_click(self._on_check_all)
        
        header = widgets.HTML(styles.card_html("Installer", "Gestion des dépendances", "Sélectionnez les bibliothèques à installer ou désinstaller."))
        cb_grid = widgets.GridBox(rows, layout=widgets.Layout(grid_template_columns="repeat(2, 1fr)", gap="10px"))
        
        guide_acc = self._build_guide()
        
        self.ui = widgets.VBox([header, guide_acc, btn_bar1, cb_grid, btn_bar2, self.out])
        
    def _on_select_all(self, b):
        for info in self.group_vars.values():
            info["chk"].value = True
            
    def _on_deselect_all(self, b):
        for info in self.group_vars.values():
            info["chk"].value = False
            
    def _on_check_all(self, b):
        for info in self.group_vars.values():
            info["btn_chk"].click()

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
                                # prevent duplicates but preserve order as best as possible
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
            # Refresh status
            self._on_check_all(None)

    def _on_uninstall(self, b):
        with self.out:
            clear_output(wait=True)
            print("Désinstallation des packages...")
            to_uninstall = set()
            for gid, info in self.group_vars.items():
                if info["chk"].value:
                    for pkg in info["pkgs"]:
                        to_uninstall.add(pkg)
            if not to_uninstall:
                print("Aucun package sélectionné.")
            else:
                print(f"Désinstallation de : {', '.join(to_uninstall)} ...")
                success, msg = uninstall_packages(list(to_uninstall))
                if success:
                    print("[OK] Désinstallation terminée !")
                else:
                    print(f"[ERREUR] Erreur de désinstallation : {msg}")
            # Refresh status
            self._on_check_all(None)

def build_installer_ui(state):
    app = InstallerUI(state)
    display(app.ui)
    return app
