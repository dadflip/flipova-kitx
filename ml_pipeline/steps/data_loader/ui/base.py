import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import pandas as pd
import io

from ml_pipeline.styles import styles
from ml_pipeline.steps.data_loader.logic.loaders import load_data

class DataSourceBlock:
    def __init__(self, name, type_options, adv_configs_map, is_removable=False, on_remove=None):
        self.name = name
        self.adv_configs_map = adv_configs_map
        self.is_removable = is_removable
        self.on_remove = on_remove
        
        self.name_in = widgets.Text(description="Nom:", value=name, layout=widgets.Layout(width="200px"))
        self.type_dd = widgets.Dropdown(options=type_options, description="Format:", layout=widgets.Layout(width="200px"))
        self.path_in = widgets.Text(description="Path/URI:", placeholder="chemin, url, etc.", layout=widgets.Layout(flex="1", min_width="250px"))
        self.upload = widgets.FileUpload(accept="", multiple=False, description="Upload", layout=widgets.Layout(width="120px"))
        
        self.btn_preview = widgets.Button(description="Preview", button_style="info", layout=widgets.Layout(width="100px"))
        self.btn_preview.on_click(self._on_preview)
        
        if self.is_removable:
            self.btn_remove = widgets.Button(icon="minus", button_style="danger", layout=widgets.Layout(width="40px"))
            self.btn_remove.on_click(lambda b: self.on_remove(self) if self.on_remove else None)
        else:
            self.btn_remove = widgets.HTML(layout=widgets.Layout(width="40px")) # Spacer
            
        self.dynamic_opts = widgets.VBox([])
        self.opt_widgets = {}
        
        self.out_preview = widgets.Output()
        
        self.type_dd.observe(self._update_adv_opts, names='value')
        self._update_adv_opts()
        
        self.ui = widgets.VBox([
            widgets.HBox([
                self.btn_remove,
                self.name_in,
                self.type_dd,
                self.path_in,
                widgets.HTML("<b style='line-height:30px; margin:0 5px;'>OU</b>"),
                self.upload,
                self.btn_preview
            ], layout=widgets.Layout(align_items="center", flex_wrap="wrap", grid_gap="5px")),
            self.dynamic_opts,
            self.out_preview
        ], layout=widgets.Layout(border="1px solid #e5e7eb", padding="10px", margin="10px 0", border_radius="8px", background_color="#fcfcfc"))

    def _update_adv_opts(self, change=None):
        ds_type = self.type_dd.value
        opts = self.adv_configs_map.get(ds_type, [])
        self.opt_widgets = {}
        
        boxes = []
        for opt in opts:
            w = None
            lbl = opt.get("description", opt.get("label", opt["id"]))
            if opt["type"] == "text":
                w = widgets.Text(description=lbl, value=str(opt.get("value", "")), placeholder=opt.get("placeholder", ""), layout=widgets.Layout(width="250px"))
            elif opt["type"] == "dropdown":
                if "options" in opt:
                    w = widgets.Dropdown(options=opt["options"], value=opt.get("value"), description=lbl, layout=widgets.Layout(width="250px"))
                else:
                    w = widgets.Text(description=lbl, value=str(opt.get("value", "")), layout=widgets.Layout(width="250px"))
            elif opt["type"] == "checkbox":
                w = widgets.Checkbox(value=opt.get("value", False), description=lbl, layout=widgets.Layout(width="250px"))
            elif opt["type"] == "floatslider":
                w = widgets.FloatSlider(value=opt.get("value", 1.0), min=opt.get("min", 0.0), max=opt.get("max", 1.0), description=lbl, layout=widgets.Layout(width="300px"))
                
            if w:
                help_text = widgets.HTML(f"<i style='font-size:0.8em; color:gray; line-height:28px;'>{opt.get('help', '')}</i>")
                self.opt_widgets[opt["id"]] = w
                boxes.append(widgets.HBox([w, help_text], layout=widgets.Layout(margin="0 5px")))
                
        if not boxes:
            boxes.append(widgets.HTML("<i style='color:gray; font-size: 0.9em;'>Aucune option avancée.</i>"))
            
        self.dynamic_opts.children = [widgets.HBox(boxes, layout=widgets.Layout(flex_wrap="wrap", padding="5px 0 0 50px"))]

    def _get_data(self):
        ds_type = self.type_dd.value
        adv_options = {k: w.value for k, w in self.opt_widgets.items()}
        src_type = None
        src_content = None
        
        if self.upload.value:
            src_type = "upload"
            try:
                uploaded_file = self.upload.value[0] if isinstance(self.upload.value, tuple) else list(self.upload.value.values())[0]
                src_content = uploaded_file['content']
            except Exception:
                keys = list(self.upload.value.keys())
                src_content = self.upload.value[keys[0]]['content']
        elif self.path_in.value:
            val = self.path_in.value
            src_content = val
            if val.startswith("http"): src_type = "url"
            else: src_type = "local"
        else:
            return None, "Aucune source spécifiée (path/upload)."
            
        try:
            data = load_data(ds_type, src_type, src_content, adv_options)
            return data, None
        except Exception as e:
            return None, str(e)

    def _on_preview(self, b):
        with self.out_preview:
            clear_output(wait=True)
            print("Chargement en cours pour preview...")
            data, err = self._get_data()
            clear_output(wait=True)
            if err:
                print(f"[ERREUR] {err}")
                return
            
            ds_type = self.type_dd.value
            display(HTML(f"<b>Aperçu pour '{self.name_in.value}' (Type: {ds_type}) :</b>"))
            if isinstance(data, pd.DataFrame):
                display(data.head(5))
                display(HTML(f"<i>Total rows: {len(data)}, Total columns: {len(data.columns)}</i>"))
            elif ds_type == "image":
                display(data) # PIL image
            elif ds_type == "graph":
                print(f"NetworkX Graph: {data}")
            elif ds_type in ("json", "web", "neo4j"):
                import json
                try:
                    print(json.dumps(data, indent=2)[:500] + "\n...")
                except:
                    print(data)
            else:
                print(data)


class DataLoaderUI:
    def __init__(self, state):
        self.state = state
        self.config = getattr(self.state, "config", {}).get("loading", {})
        self.adv_configs = self.config.get("adv_configs", {})
        self.mode_configs = self.config.get("mode_configs", {})
        
        self.mode_opts_container = widgets.VBox([])
        self.mode_opt_widgets = {}
        
        self.blocks = []
        self.blocks_container = widgets.VBox([])
        
        self.out_global = widgets.Output()
        
        self._build_ui()
        self._on_mode_change()

    def _build_guide(self) -> widgets.Accordion:
        guide_html = """
        <div style='font-size:14px; line-height:1.6; color:#374151;'>
            <h4>Guide du Chargement de Données</h4>
            <p>Sélectionnez le mode de chargement selon le nombre de sources nécessaires. Vous pouvez ajouter/retirer des sources avec + et -.</p>
            <ul>
                <li><b>Fichiers Tabulaires :</b> Définissez le séparateur, l'index, encodage...</li>
                <li><b>Preview :</b> Prévisualisez chaque dataset un par un sans l'enregistrer dans l'état global.</li>
                <li><b>Load All :</b> Charge toutes les sources et les injecte dans le pipeline ML.</li>
            </ul>
        </div>
        """
        out = widgets.Output()
        with out:
            display(HTML(guide_html))
        acc = widgets.Accordion(children=[out], selected_index=None)
        acc.set_title(0, "Guide explicatif des Outils de Chargement (Cliquez pour ouvrir)")
        return acc

    def _build_ui(self):
        modes = self.config.get("modes", [{"label": "Single", "value": "single"}])
        mode_options = [(m.get("label", m["value"]), m["value"]) for m in modes]
        
        self.mode_dd = widgets.Dropdown(options=mode_options, description="Mode Loading:", layout=widgets.Layout(width="250px"))
        self.mode_dd.observe(self._on_mode_change, names='value')
        
        self.btn_add_source = widgets.Button(icon="plus", description=" Ajouter Source", button_style="success", layout=widgets.Layout(width="150px"))
        self.btn_add_source.on_click(self._on_add_source)
        
        self.btn_load_all = widgets.Button(description="Load All into State", button_style=styles.BTN_PRIMARY, layout=widgets.Layout(width="250px", height="40px"))
        self.btn_load_all.on_click(self._on_load_all)
        
        header = widgets.HTML(styles.card_html("DataLoader", "Chargement des données & Multimodal", "Configurez vos sources de données et prévisualisez-les indépendamment."))
        guide_acc = self._build_guide()
        
        self.ui = widgets.VBox([
            header,
            guide_acc,
            widgets.HBox([self.mode_dd, self.btn_add_source], layout=widgets.Layout(margin="10px 0", gap="15px")),
            self.mode_opts_container,
            widgets.HTML("<hr style='border:1px solid #e5e7eb; margin: 15px 0;'>"),
            self.blocks_container,
            widgets.HTML("<hr style='border:1px solid #e5e7eb; margin: 15px 0;'>"),
            widgets.HBox([self.btn_load_all], layout=widgets.Layout(justify_content="center")),
            self.out_global
        ], layout=styles.LAYOUT_SECTION)

    def _get_types_options(self):
        types = self.config.get("supported_types", {"CSV": "csv"})
        return [(k, v) for k, v in types.items()]

    def _update_mode_opts(self):
        mode_val = self.mode_dd.value
        opts = self.mode_configs.get(mode_val, [])
        self.mode_opt_widgets = {}
        
        boxes = []
        for opt in opts:
            w = None
            lbl = opt.get("label", opt["id"])
            if opt["type"] == "text":
                w = widgets.Text(description=lbl, value=str(opt.get("value", "")), placeholder=opt.get("placeholder", ""))
            elif opt["type"] == "dropdown":
                if "options" in opt:
                    w = widgets.Dropdown(options=opt["options"], value=opt.get("value"), description=lbl)
                else:
                    w = widgets.Text(description=lbl, value=str(opt.get("value", "")))
            elif opt["type"] == "checkbox":
                w = widgets.Checkbox(value=opt.get("value", False), description=lbl)
            elif opt["type"] == "floatslider":
                w = widgets.FloatSlider(value=opt.get("value", 1.0), min=opt.get("min", 0.0), max=opt.get("max", 1.0), description=lbl)
                
            if w:
                help_text = widgets.HTML(f"<i style='font-size:0.8em; color:gray; line-height:28px;'>{opt.get('help', '')}</i>")
                self.mode_opt_widgets[opt["id"]] = w
                boxes.append(widgets.HBox([w, help_text]))
                
        if boxes:
            self.mode_opts_container.children = [widgets.HTML("<b>Mode Config:</b>"), widgets.HBox(boxes, layout=widgets.Layout(flex_wrap="wrap", grid_gap="10px"))]
            self.mode_opts_container.layout.display = "block"
        else:
            self.mode_opts_container.children = []
            self.mode_opts_container.layout.display = "none"

    def _on_mode_change(self, change=None):
        mode_val = self.mode_dd.value
        modes = self.config.get("modes", [])
        slots = ["Data"]
        for m in modes:
            if m["value"] == mode_val:
                slots = m.get("slots", ["Data"])
                break
                
        self._update_mode_opts()
                
        self.blocks = []
        for s in slots:
            b = DataSourceBlock(s, self._get_types_options(), self.adv_configs, is_removable=False)
            self.blocks.append(b)
            
        self._update_blocks_ui()
        
        # We can allow adding blocks to any mode technically, but specifically multi/custom
        if mode_val in ("multi_source", "custom", "train_val_test", "train_test"):
            self.btn_add_source.layout.display = "block"
        else:
            self.btn_add_source.layout.display = "none"

    def _on_add_source(self, b):
        new_name = f"Source {len(self.blocks) + 1}"
        blk = DataSourceBlock(new_name, self._get_types_options(), self.adv_configs, is_removable=True, on_remove=self._remove_source)
        self.blocks.append(blk)
        self._update_blocks_ui()

    def _remove_source(self, blk):
        if blk in self.blocks:
            self.blocks.remove(blk)
            self._update_blocks_ui()

    def _update_blocks_ui(self):
        self.blocks_container.children = [b.ui for b in self.blocks]

    def _on_load_all(self, btn):
        with self.out_global:
            clear_output(wait=True)
            print("Chargement en cours...")
            
            if not hasattr(self.state, "data_raw"): self.state.data_raw = {}
            if not hasattr(self.state, "data_types"): self.state.data_types = {}
            
            # Retrieve mode config values
            mode_conf = {k: w.value for k, w in self.mode_opt_widgets.items()}
            
            loaded_count = 0
            loaded_names = []
            for b in self.blocks:
                name = b.name_in.value
                ds_type = b.type_dd.value
                
                # Check if configured
                if not b.upload.value and not b.path_in.value:
                    if len(self.blocks) == 1:
                        print(f"⚠️ Source '{name}' vide, ignorée.")
                    continue
                
                print(f"-> Chargement '{name}'...")
                data, err = b._get_data()
                
                if err:
                    print(f"❌ Erreur pour '{name}' : {err}")
                else:
                    self.state.data_raw[name] = data
                    self.state.data_types[name] = ds_type
                    loaded_names.append(name)
                    print(f"✅ Succès: '{name}' chargé (Format: {ds_type})")
                    loaded_count += 1
            
            if loaded_count > 0:
                print(f"🎉 Terminé! {loaded_count} dataset(s) chargés dans le state global.")
                # We could potentially split datasets here if `auto_split` is active
                mode_val = self.mode_dd.value
                if mode_val == "single" and mode_conf.get("auto_split") in ("Train/Test", "Train/Val/Test") and loaded_count == 1:
                    df = self.state.data_raw.get(loaded_names[0])
                    if isinstance(df, pd.DataFrame):
                        from sklearn.model_selection import train_test_split
                        print("==> Application de l'auto split...")
                        try:
                            stratify_col = mode_conf.get("stratify", "")
                            stratify = df[stratify_col] if stratify_col and stratify_col in df.columns else None
                            
                            ts = mode_conf.get("test_size", 0.2)
                            vs = mode_conf.get("val_size", 0.1)
                            
                            if mode_conf["auto_split"] == "Train/Test":
                                tr, te = train_test_split(df, test_size=ts, stratify=stratify, random_state=42)
                                self.state.data_raw["Train"] = tr
                                self.state.data_types["Train"] = ds_type
                                self.state.data_raw["Test"] = te
                                self.state.data_types["Test"] = ds_type
                                print("✅ Train/Test créé avec succès.")
                                
                            elif mode_conf["auto_split"] == "Train/Val/Test":
                                # train vs temp
                                tr, temp = train_test_split(df, test_size=(ts + vs), stratify=stratify, random_state=42)
                                # validation vs test
                                temp_strat = temp[stratify_col] if stratify is not None else None
                                val, te = train_test_split(temp, test_size=ts/(ts+vs), stratify=temp_strat, random_state=42)
                                
                                self.state.data_raw["Train"] = tr
                                self.state.data_types["Train"] = ds_type
                                self.state.data_raw["Validation"] = val
                                self.state.data_types["Validation"] = ds_type
                                self.state.data_raw["Test"] = te
                                self.state.data_types["Test"] = ds_type
                                print("✅ Train/Val/Test créé avec succès.")
                                
                        except Exception as e:
                            print(f"⚠️ Erreur lors de l'auto_split: {e}")
                    else:
                        print("⚠️ Auto split ignoré: les données ne sont pas des DataFrames tabulaires.")
                        
                self.state.log_step("loading", "all_datasets_loaded", {"count": loaded_count, "mode": mode_val})

def build_dataloader_ui(state):
    app = DataLoaderUI(state)
    display(app.ui)
    return app

