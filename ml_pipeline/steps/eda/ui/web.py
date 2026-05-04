import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import matplotlib.pyplot as plt
import pandas as pd

from ml_pipeline.steps.eda.logic.web import (
    get_web_meta, get_web_links_fig, get_web_dom_distribution_fig,
    get_web_wordcloud_fig, get_web_text_stats
)
from ml_pipeline.styles import styles

def build_web_ui(eda_ui, data):
    html_content = ""
    if isinstance(data, dict):
        html_content = data.get("html", "")
    elif isinstance(data, str):
        html_content = data
    
    # Tool 1: Meta
    out_meta = widgets.Output()
    def _show_meta():
        with out_meta:
            clear_output(wait=True)
            meta = get_web_meta(html_content)
            print("Title:", meta.get("Title", "N/A"))
            display(HTML("<b style='color:#374151;'>Meta Tags</b>"))
            df = pd.DataFrame(list(meta.get("Meta Tags", {}).items()), columns=["Key", "Content"])
            display(df)
    _show_meta()
    
    # Tool 2: Links 
    out_links = widgets.Output()
    btn_links = widgets.Button(description="Analyze Links", button_style=styles.BTN_PRIMARY)
    def _links(b):
        with out_links:
            clear_output(wait=True)
            fig = get_web_links_fig(html_content)
            if fig:
                display(fig)
                plt.close(fig)
    btn_links.on_click(_links)
    
    # Tool 3: DOM Distribution
    out_dom = widgets.Output()
    btn_dom = widgets.Button(description="Analyze DOM Tags", button_style=styles.BTN_PRIMARY)
    def _dom(b):
        with out_dom:
            clear_output(wait=True)
            fig = get_web_dom_distribution_fig(html_content)
            if fig:
                display(fig)
                plt.close(fig)
    btn_dom.on_click(_dom)
    
    # Tool 4: Wordcloud
    out_wc = widgets.Output()
    btn_wc = widgets.Button(description="Generate WordCloud", button_style=styles.BTN_PRIMARY)
    def _wc(b):
        with out_wc:
            clear_output(wait=True)
            fig = get_web_wordcloud_fig(html_content)
            if fig:
                display(fig)
                plt.close(fig)
    btn_wc.on_click(_wc)
    
    # Tool 5: Content Text Stats
    out_stats = widgets.Output()
    btn_stats = widgets.Button(description="Content Stats", button_style=styles.BTN_PRIMARY)
    def _stats(b):
        with out_stats:
            clear_output(wait=True)
            stats = get_web_text_stats(html_content)
            display(pd.DataFrame([stats]).T.rename(columns={0: "Value"}))
    btn_stats.on_click(_stats)
    
    tabs = widgets.Tab(children=[
        out_meta,
        widgets.VBox([btn_links, out_links]),
        widgets.VBox([btn_dom, out_dom]),
        widgets.VBox([btn_wc, out_wc]),
        widgets.VBox([btn_stats, out_stats])
    ])
    
    for i, t in enumerate(["Meta Info", "Links Analysis", "DOM Structure", "Word Cloud", "Content Stats"]):
        tabs.set_title(i, t)
        
    eda_ui.dynamic_ui.children = [tabs]
