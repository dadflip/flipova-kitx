import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import matplotlib.pyplot as plt
from ml_pipeline.steps.eda.logic.image import (
    get_image_info, get_image_preview_b64, get_image_color_histogram_fig,
    get_image_channels_fig, get_image_edges_fig, get_image_dominant_colors_fig,
    get_image_filters_fig
)
from ml_pipeline.styles import styles

def build_image_ui(eda_ui, img) -> None:
    # Tool 1: Basics & Histogram
    out_basic = widgets.Output()
    def _show_basic():
        with out_basic:
            clear_output(wait=True)
            info = get_image_info(img)
            print(f"Format: {info['format']}  |  Size: {info['size']}  |  Mode: {info['mode']}")
            b64 = get_image_preview_b64(img)
            if b64:
                display(HTML(f"<img src='data:image/jpeg;base64,{b64}' style='max-width:300px; margin-bottom:10px;'/>"))
            if info['mode'] == "RGB" or info['mode'] == "RGBA":
                fig = get_image_color_histogram_fig(img)
                display(fig)
                plt.close(fig)
    _show_basic()
    
    # Tool 2: RGB Channels
    out_channels = widgets.Output()
    btn_channels = widgets.Button(description="Extract Channels", button_style=styles.BTN_PRIMARY)
    def _show_channels(b):
        with out_channels:
            clear_output(wait=True)
            fig = get_image_channels_fig(img)
            display(fig)
            plt.close(fig)
    btn_channels.on_click(_show_channels)
    
    # Tool 3: Edge Detection
    out_edges = widgets.Output()
    btn_edges = widgets.Button(description="Detect Edges (Sobel)", button_style=styles.BTN_PRIMARY)
    def _show_edges(b):
        with out_edges:
            clear_output(wait=True)
            fig = get_image_edges_fig(img)
            display(fig)
            plt.close(fig)
    btn_edges.on_click(_show_edges)
    
    # Tool 4: Dominant Colors
    out_colors = widgets.Output()
    slider_k = widgets.IntSlider(value=5, min=2, max=15, description="K:")
    btn_colors = widgets.Button(description="Find Dominant Colors", button_style=styles.BTN_PRIMARY)
    def _show_colors(b):
        with out_colors:
            clear_output(wait=True)
            fig = get_image_dominant_colors_fig(img, k=slider_k.value)
            display(fig)
            plt.close(fig)
    btn_colors.on_click(_show_colors)
    
    # Tool 5: Filters
    out_filters = widgets.Output()
    btn_filters = widgets.Button(description="Apply Filters", button_style=styles.BTN_PRIMARY)
    def _show_filters(b):
        with out_filters:
            clear_output(wait=True)
            fig = get_image_filters_fig(img)
            display(fig)
            plt.close(fig)
    btn_filters.on_click(_show_filters)
    
    tabs = widgets.Tab(children=[
        out_basic,
        widgets.VBox([btn_channels, out_channels]),
        widgets.VBox([btn_edges, out_edges]),
        widgets.VBox([widgets.HBox([slider_k, btn_colors]), out_colors]),
        widgets.VBox([btn_filters, out_filters])
    ])
    
    for i, t in enumerate(["Recap & Hist", "RGB Channels", "Edge Detection", "Dominant Colors", "Filters"]):
        tabs.set_title(i, t)
        
    eda_ui.dynamic_ui.children = [tabs]
