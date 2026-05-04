import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import pandas as pd

def build_graph_ui(parent_ui, graph_data):
    out = widgets.Output()
    
    btn_extract = widgets.Button(description="Extract Topology Features (Tabular)", button_style="success", layout=widgets.Layout(width="280px"))
    
    def _on_extract(_):
        with out:
            clear_output()
            try:
                import networkx as nx
                # Extract degree, centrality, pagerank
                degrees = dict(graph_data.degree())
                centrality = nx.betweenness_centrality(graph_data)
                pagerank = nx.pagerank(graph_data)
                
                df = pd.DataFrame([
                    {"node": n, "degree": degrees.get(n, 0), "centrality": centrality.get(n, 0), "pagerank": pagerank.get(n, 0)}
                    for n in graph_data.nodes()
                ])
                
                new_ds_name = f"{parent_ui.current_ds} - Nodes"
                parent_ui.state.data_raw[new_ds_name] = df
                parent_ui.state.data_types[new_ds_name] = "tabular"
                display(HTML(f"<b style='color:green;'>Extracted topological features for {len(df)} nodes. Saved as '{new_ds_name}' (tabular).</b>"))
            except Exception as e:
                display(HTML(f"<b style='color:red;'>Error: {e}</b>"))
                
    btn_extract.on_click(_on_extract)
    
    parent_ui.dynamic_ui.children = [
        widgets.HTML("<h3>Graph Feature Engineering</h3>"),
        widgets.HTML("<p>Extract network topology parameters as a node-level tabular dataset.</p>"),
        btn_extract,
        out
    ]
