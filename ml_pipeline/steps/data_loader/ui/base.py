import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import pandas as pd
import io

from ml_pipeline.styles import styles
from ml_pipeline.steps.data_loader.logic.loaders import load_data

class DataLoaderUI:
    def __init__(self, state):
        self.state = state
        self.config = getattr(self.state, "config", {}).get("loading", {})
        self.adv_configs = self.config.get("adv_configs", {})
        self.dynamic_opts = widgets.VBox([])
        self.opt_widgets = {}
        self._build_ui()
        self._update_adv_opts()

    def _build_guide(self) -> widgets.Accordion:
        guide_html = """
        <div style='font-size:14px; line-height:1.6; color:#374151;'>
            <h4>Guide du Chargement de Données</h4>
            <p>Ce module vous permet d'ingérer différents types de données. La vue s'adapte dynamiquement selon le format sélectionné.</p>
            <ul>
                <li><b>Fichiers Tabulaires (CSV, TSV, Excel, JSON) :</b> Définissez le séparateur, l'encodage, les lignes à sauter, ou la colonne d'index. Options pour échantillonner (<i>sample_frac</i>).</li>
                <li><b>Séries Temporelles :</b> Idem que tabulaire, mais pensez à cocher "Parser les dates auto" et à sélectionner la bonne colonne d'index.</li>
                <li><b>Images :</b> Vous pouvez forcer la conversion RGB ou redimensionner lors du chargement.</li>
                <li><b>Neo4j :</b> Renseignez l'URI (ex: <i>bolt://localhost:7687</i>), l'utilisateur et le mot de passe pour vous connecter à une base distante.</li>
                <li><b>Web :</b> Renseignez une URL, la page sera scrappée et chargée en mémoire (HTML brut). Dans l'EDA, vous pourrez afficher son DOM ou ses textes.</li>
            </ul>
        </div>
        """
        out = widgets.Output()
        with out:
            display(HTML(guide_html))
        acc = widgets.Accordion(children=[out], selected_index=None)
        acc.set_title(0, "Guide explicatif des Outils de Chargement")
        return acc

    def _build_ui(self):
        modes = self.config.get("modes", [{"label": "Single", "value": "single"}])
        mode_options = [(m.get("label", m["value"]), m["value"]) for m in modes]
        
        self.mode_dd = widgets.Dropdown(options=mode_options, description="Mode Loading:")
        
        types = self.config.get("supported_types", {"CSV": "csv"})
        type_options = [(k, v) for k, v in types.items()]
        self.type_dd = widgets.Dropdown(options=type_options, description="Format:", layout=widgets.Layout(width="auto"))
        
        self.type_dd.observe(lambda change: self._update_adv_opts(), names='value')
        
        self.path_in = widgets.Text(description="Path/URI:", placeholder="chemin local ou URL", layout=widgets.Layout(width="400px"))
        self.upload = widgets.FileUpload(accept="", multiple=False, description="Upload File", layout=widgets.Layout(width="200px"))
        self.btn_load = widgets.Button(description="Load Data", button_style=styles.BTN_PRIMARY)
        self.out = widgets.Output()
        
        self.btn_load.on_click(self._on_load)
        
        header = widgets.HTML(styles.card_html("DataLoader", "Chargement des données avancées", "Configurez les paramètres spécifiques à votre type de données ci-dessous pour un parsing optimal."))
        guide_acc = self._build_guide()
        
        self.ui = widgets.VBox([
            header,
            guide_acc,
            widgets.HBox([self.mode_dd, self.type_dd]),
            widgets.HBox([self.path_in, widgets.HTML("<b style='margin:0 10px; line-height:30px;'>OU</b>"), self.upload]),
            widgets.HTML("<hr style='border:1px solid #e5e7eb; margin: 10px 0;'>"),
            widgets.HTML("<b style='color:#374151;'>Options Avancées:</b>"),
            self.dynamic_opts,
            widgets.HTML("<hr style='border:1px solid #e5e7eb; margin: 10px 0;'>"),
            self.btn_load,
            self.out
        ], layout=widgets.Layout(padding="15px", border="1px solid #e5e7eb", border_radius="10px"))

    def _update_adv_opts(self):
        ds_type = self.type_dd.value
        opts = self.adv_configs.get(ds_type, [])
        self.opt_widgets = {}
        
        boxes = []
        for opt in opts:
            w = None
            if opt["type"] == "text":
                w = widgets.Text(description=opt["id"], value=str(opt.get("value", "")), placeholder=opt.get("placeholder", ""))
            elif opt["type"] == "dropdown":
                if "options" in opt:
                    w = widgets.Dropdown(options=opt["options"], value=opt["value"], description=opt["id"])
                else:
                    # fallback
                    w = widgets.Text(description=opt["id"], value=str(opt.get("value", "")))
            elif opt["type"] == "checkbox":
                w = widgets.Checkbox(value=opt.get("value", False), description=opt["id"])
            elif opt["type"] == "floatslider":
                w = widgets.FloatSlider(value=opt.get("value", 1.0), min=opt.get("min", 0.0), max=opt.get("max", 1.0), description=opt["id"])
                
            if w:
                help_text = widgets.HTML(f"<i style='font-size:0.8em; color:gray; margin-left:10px;'>{opt.get('help', '')}</i>")
                self.opt_widgets[opt["id"]] = w
                boxes.append(widgets.HBox([w, help_text]))
                
        if not boxes:
            boxes.append(widgets.HTML("<i style='color:gray;'>Aucune option avancée disponible pour ce format.</i>"))
            
        self.dynamic_opts.children = boxes

    def _on_load(self, b):
        with self.out:
            clear_output(wait=True)
            mode = self.mode_dd.value
            ds_type = self.type_dd.value
            
            adv_options = {k: w.value for k, w in self.opt_widgets.items()}
            
            src_type = None
            src_content = None
            
            if self.upload.value:
                src_type = "upload"
                # ipywidgets 8 returns dict or tuple of dict
                try:
                    uploaded_file = self.upload.value[0] if isinstance(self.upload.value, tuple) else list(self.upload.value.values())[0]
                    src_content = uploaded_file['content']
                    msg_src = f"fichier uploadé ({uploaded_file.get('name', 'inconnu')})"
                except Exception as e:
                    msg_src = "upload (erreur parsing widget)"
                    # Fallback for old versions
                    try:
                        keys = list(self.upload.value.keys())
                        src_content = self.upload.value[keys[0]]['content']
                    except:
                        pass
            elif self.path_in.value:
                val = self.path_in.value
                src_content = val
                msg_src = val
                if val.startswith("http"):
                    src_type = "url"
                else:
                    src_type = "local"
            else:
                print("Veuillez renseigner un chemin, une URL ou uploader un fichier.")
                return
                
            print(f"Chargement ({ds_type}) depuis {msg_src}...")
            
            try:
                data = load_data(ds_type, src_type, src_content, adv_options)
            except Exception as e:
                print(f"[ERREUR] Erreur de chargement: {e}")
                return
                
            if data is not None:
                if not hasattr(self.state, "data_raw"): self.state.data_raw = {}
                if not hasattr(self.state, "data_types"): self.state.data_types = {}
                
                ds_name = f"Dataset_{mode}"
                self.state.data_raw[ds_name] = data
                self.state.data_types[ds_name] = ds_type
                
                print(f"[OK] Chargé avec succès en tant que '{ds_name}' (Format: {ds_type})")
                
                # Preview
                try:
                    display(HTML("<b>Aperçu :</b>"))
                    if isinstance(data, pd.DataFrame):
                        display(data.head(5))
                    elif ds_type == "image":
                        display(data)
                    else:
                        print(f"Type: {type(data)}")
                except:
                    pass
                    
                self.state.log_step("loading", "dataset_loaded", {"name": ds_name, "type": ds_type})

def build_dataloader_ui(state):
    app = DataLoaderUI(state)
    display(app.ui)
    return app
