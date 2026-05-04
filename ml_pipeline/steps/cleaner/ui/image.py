import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

def build_image_ui(parent_ui, img_data):
    out = widgets.Output()
    
    btn_clean = widgets.Button(description="Normalize & Save to Cleaned", button_style="info", layout=widgets.Layout(width="280px"))
    
    def _on_clean(_):
        with out:
            clear_output()
            try:
                # Basic mock normalisation for image (usually resizing, convert RGB)
                if hasattr(img_data, "convert"):
                    cleaned = img_data.convert("RGB")
                else:
                    cleaned = img_data
                parent_ui.state.data_cleaned[parent_ui.current_ds] = cleaned
                display(HTML("<b style='color:green;'>Image converted to RGB and saved to data_cleaned.</b>"))
            except Exception as e:
                display(HTML(f"<b style='color:red;'>Error: {e}</b>"))
                
    btn_clean.on_click(_on_clean)
    
    parent_ui.dynamic_ui.children = [
        widgets.HTML("<h3>Image Cleaning</h3>"),
        btn_clean,
        out
    ]
