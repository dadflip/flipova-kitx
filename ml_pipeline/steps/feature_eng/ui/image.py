import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from PIL import Image

def build_image_ui(parent_ui, img_data):
    out = widgets.Output()
    
    btn_resize = widgets.Button(description="Resize (256x256)", button_style="info")
    btn_grayscale = widgets.Button(description="Convert Grayscale", button_style="info")
    
    def _on_resize(_):
        with out:
            clear_output()
            try:
                new_img = img_data.resize((256, 256))
                parent_ui.state.data_raw[parent_ui.current_ds] = new_img
                display(HTML("<b style='color:green;'>Image resized to 256x256.</b>"))
            except Exception as e:
                display(HTML(f"<b style='color:red;'>Error: {e}</b>"))
                
    def _on_grayscale(_):
        with out:
            clear_output()
            try:
                new_img = img_data.convert("L")
                parent_ui.state.data_raw[parent_ui.current_ds] = new_img
                display(HTML("<b style='color:green;'>Image converted to grayscale.</b>"))
            except Exception as e:
                display(HTML(f"<b style='color:red;'>Error: {e}</b>"))
                
    def _on_extract_tabular(_):
        with out:
            clear_output()
            try:
                import numpy as np
                import pandas as pd
                # Flatten image into a feature vector (e.g., color histograms)
                if img_data.mode != "RGB":
                    img = img_data.convert("RGB")
                else:
                    img = img_data
                arr = np.array(img)
                # Compute 8-bin histogram per channel
                features = []
                cols = []
                for i, color in enumerate(("R", "G", "B")):
                    hist, _ = np.histogram(arr[:, :, i].ravel(), bins=8, range=[0, 256])
                    features.extend(hist)
                    cols.extend([f"hist_{color}_{j}" for j in range(8)])
                
                df = pd.DataFrame([features], columns=cols)
                new_ds_name = f"{parent_ui.current_ds} - Hist_Features"
                parent_ui.state.data_raw[new_ds_name] = df
                parent_ui.state.data_types[new_ds_name] = "tabular"
                display(HTML(f"<b style='color:green;'>Extracted 24 Color Histogram Features. Saved as '{new_ds_name}' (tabular).</b>"))
            except Exception as e:
                display(HTML(f"<b style='color:red;'>Error: {e}</b>"))
                
    btn_resize.on_click(_on_resize)
    btn_grayscale.on_click(_on_grayscale)
    btn_extract = widgets.Button(description="Extract Features (Tabular)", button_style="success", layout=widgets.Layout(width="250px"))
    btn_extract.on_click(_on_extract_tabular)
    
    parent_ui.dynamic_ui.children = [
        widgets.HTML("<h3>Image Feature Engineering</h3>"),
        widgets.HBox([btn_resize, btn_grayscale, btn_extract]),
        out
    ]
