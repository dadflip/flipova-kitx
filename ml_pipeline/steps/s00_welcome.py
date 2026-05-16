"""s00_welcome — Page d'accueil du notebook Flipova KitX."""
import ipywidgets as widgets
from IPython.display import display
from ml_pipeline.styles import styles

class WelcomeStep:
    def __init__(self, state):
        self.state = state
        self._build_ui()

    def _build_ui(self):
        # Apply global styles
        styles.apply_globals()
        
        # Display the majestic header
        header = styles.flipova_header(version="v2.1.4 (LTS)", username="Flipova Kit Designer")
        
        # Quick access cards
        content = """
        Bienvenue dans votre environnement de travail Flipova. 
        Ce notebook est structuré en étapes séquentielles pour vous guider du chargement des données au déploiement.
        <br/><br/>
        <b>Prochaines étapes recommandées :</b>
        <ul>
            <li>Vérifiez la configuration dans <code>default.toml</code></li>
            <li>Lancez l'étape <b>S01_Loading</b> pour importer vos datasets</li>
            <li>Explorez vos données avec <b>S03_EDA</b></li>
        </ul>
        """
        intro_card = widgets.HTML(styles.card_html("Prise en main", "Démarrage Rapide", content))
        
        # System health / State summary
        state_info = f"""
        L'état du pipeline est actuellement <b>{"VIERGE" if not self.state.history else "ACTIF"}</b>.
        <br/>
        - Datasets chargés : {len(self.state.data_raw)}
        - Transformations effectuées : {len(self.state.history)}
        - Modèles en mémoire : {len(self.state.trained_models)}
        """
        state_card = widgets.HTML(styles.card_html("État du Système", "Diagnostic", state_info))
        
        self.ui = widgets.VBox([
            header,
            widgets.HBox([intro_card, state_card], layout=widgets.Layout(gap="20px"))
        ])
        
    def display(self):
        display(self.ui)
