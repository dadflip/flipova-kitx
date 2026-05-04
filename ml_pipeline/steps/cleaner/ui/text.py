import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

def build_text_ui(parent_ui, text_data):
    out = widgets.Output()
    
    if isinstance(text_data, str):
        text_data = [text_data]
        
    btn_clean = widgets.Button(description="Basic Text Clean (Lower, Strip)", button_style="info", layout=widgets.Layout(width="250px"))
    
    def _on_clean(_):
        with out:
            clear_output()
            try:
                cleaned = [str(t).lower().strip() for t in text_data]
                parent_ui.state.data_cleaned[parent_ui.current_ds] = cleaned
                display(HTML("<b style='color:green;'>Text cleaned and saved to data_cleaned.</b>"))
            except Exception as e:
                display(HTML(f"<b style='color:red;'>Error: {e}</b>"))
                
    btn_clean.on_click(_on_clean)
    
    parent_ui.dynamic_ui.children = [
        widgets.HTML("<h3>Text Cleaning</h3>"),
        btn_clean,
        out
    ]
