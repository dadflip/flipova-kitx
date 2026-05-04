import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import matplotlib.pyplot as plt

from ml_pipeline.steps.eda.logic.neo4j_eda import (
    get_neo4j_schema, get_neo4j_nodes_distribution_fig,
    get_neo4j_edges_distribution_fig, get_neo4j_sample_graph_fig,
    run_neo4j_cypher
)
from ml_pipeline.styles import styles

def build_neo4j_ui(eda_ui, data):
    
    # Tool 1: Schema
    out_schema = widgets.Output()
    def _show_schema():
        with out_schema:
            clear_output(wait=True)
            schema = get_neo4j_schema(data)
            display(HTML("<b style='color:#374151;'>Database Schema Overview</b>"))
            for k, v in schema.items():
                print(f"{k}: {v}")
    _show_schema()

    # Tool 2: Nodes Distribution
    out_nodes = widgets.Output()
    btn_nodes = widgets.Button(description="Node Label Stats", button_style=styles.BTN_PRIMARY)
    def _nodes(b):
        with out_nodes:
            clear_output(wait=True)
            fig = get_neo4j_nodes_distribution_fig(data)
            if fig:
                display(fig)
                plt.close(fig)
    btn_nodes.on_click(_nodes)
    
    # Tool 3: Edges Distribution
    out_edges = widgets.Output()
    btn_edges = widgets.Button(description="Rel Type Stats", button_style=styles.BTN_PRIMARY)
    def _edges(b):
        with out_edges:
            clear_output(wait=True)
            fig = get_neo4j_edges_distribution_fig(data)
            if fig:
                display(fig)
                plt.close(fig)
    btn_edges.on_click(_edges)
    
    # Tool 4: Subgraph Sample
    out_graph = widgets.Output()
    slider_limit = widgets.IntSlider(value=50, min=10, max=500, description="Limit:")
    btn_graph = widgets.Button(description="Draw Subgraph", button_style=styles.BTN_PRIMARY)
    def _graph(b):
        with out_graph:
            clear_output(wait=True)
            fig = get_neo4j_sample_graph_fig(data, limit=slider_limit.value)
            if fig:
                display(fig)
                plt.close(fig)
    btn_graph.on_click(_graph)
    
    # Tool 5: Cypher Sandbox
    eda_cfg = getattr(eda_ui.state, "config", {}).get("eda", {})
    doc_url = eda_cfg.get("neo4j_doc_url", "https://neo4j.com/docs/cypher-manual/current/")
    
    out_cypher = widgets.Output()
    txt_cypher = widgets.Textarea(value="MATCH (n) RETURN n LIMIT 5", layout=widgets.Layout(width="80%", height="100px"))
    btn_cypher = widgets.Button(description="Run Cypher", button_style=styles.BTN_PRIMARY)
    
    help_text = widgets.HTML(f"<div>Type your custom Cypher command below. <a href='{doc_url}' target='_blank'>[Neo4j Cypher Docs]</a></div>")
    
    def _cypher(b):
        with out_cypher:
            clear_output(wait=True)
            df = run_neo4j_cypher(data, txt_cypher.value)
            display(df)
    btn_cypher.on_click(_cypher)
    
    tabs = widgets.Tab(children=[
        out_schema,
        widgets.VBox([btn_nodes, out_nodes]),
        widgets.VBox([btn_edges, out_edges]),
        widgets.VBox([widgets.HBox([slider_limit, btn_graph]), out_graph]),
        widgets.VBox([help_text, txt_cypher, btn_cypher, out_cypher])
    ])
    
    for i, t in enumerate(["Schema", "Node Distribution", "Edge Distribution", "Sample Graph", "Cypher Sandbox"]):
        tabs.set_title(i, t)
        
    eda_ui.dynamic_ui.children = [tabs]
