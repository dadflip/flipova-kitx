import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import pandas as pd
import matplotlib.pyplot as plt

from ml_pipeline.styles import styles
from ml_pipeline.steps.eda.logic.graph import (
    get_graph_stats, get_graph_stats_fig, get_graph_viz_fig,
    get_graph_centrality_fig, get_graph_communities_fig,
    get_graph_assortativity, get_graph_kcore_fig
)

def build_graph_ui(eda_ui, G) -> None:
    stats = get_graph_stats(G)
    if not stats:
        eda_ui.dynamic_ui.children = [widgets.HTML(
            "<div style='padding:12px;color:#b91c1c;'>networkx non disponible.</div>")]
        return

    n_nodes = stats["n_nodes"]
    
    # Tool 1 : Global Stats
    out_stats = widgets.Output()
    def _show_stats():
        with out_stats:
            clear_output(wait=True)
            summary = [
                {"Métrique": "Nœuds",              "Valeur": stats["n_nodes"]},
                {"Métrique": "Arêtes",              "Valeur": stats["n_edges"]},
                {"Métrique": "Dirigé",              "Valeur": str(stats["is_dir"])},
                {"Métrique": "Densité",             "Valeur": f"{stats['density']:.6f}"},
                {"Métrique": "Degré moyen",         "Valeur": f"{stats['avg_degree']:.2f}"},
                {"Métrique": "Degré max",           "Valeur": stats["max_degree"]},
                {"Métrique": "Composantes connexes","Valeur": stats["components"]},
            ]
            display(HTML("<b style='color:#374151;font-size:0.9em;'>Résumé du graphe</b>"))
            display(pd.DataFrame(summary).set_index("Métrique"))

            fig = get_graph_stats_fig(stats["degrees"], stats["top_nodes"])
            display(fig)
            plt.close(fig)
    _show_stats()

    # Tool 2 : Visualisation
    out_viz = widgets.Output()
    density_slider = widgets.IntSlider(
        value=min(100, n_nodes), min=10, max=min(500, n_nodes), step=10,
        description="Nœuds max:")
    layout_dd = widgets.Dropdown(
        options=["spring", "kamada_kawai", "circular", "shell", "spectral"],
        value="spring", description="Layout:")
    plot_btn  = widgets.Button(description="Générer Graphe", button_style=styles.BTN_PRIMARY)
    def _plot_viz(b):
        with out_viz:
            clear_output(wait=True)
            fig = get_graph_viz_fig(G, density_slider.value, layout_dd.value)
            display(fig)
            plt.close(fig)
    plot_btn.on_click(_plot_viz)
    
    # Tool 3 : Centrality
    out_cent = widgets.Output()
    btn_cent = widgets.Button(description="Compute Centrality", button_style=styles.BTN_PRIMARY)
    def _plot_cent(b):
        with out_cent:
            clear_output(wait=True)
            fig = get_graph_centrality_fig(G)
            display(fig)
            plt.close(fig)
    btn_cent.on_click(_plot_cent)
    
    # Tool 4 : Communities
    out_comm = widgets.Output()
    btn_comm = widgets.Button(description="Detect Communities", button_style=styles.BTN_PRIMARY)
    def _plot_comm(b):
        with out_comm:
            clear_output(wait=True)
            fig = get_graph_communities_fig(G)
            display(fig)
            plt.close(fig)
    btn_comm.on_click(_plot_comm)
    
    # Tool 5 : Assortativity & K-Core
    out_adv = widgets.Output()
    btn_adv = widgets.Button(description="Advanced Analysis", button_style=styles.BTN_PRIMARY)
    def _plot_adv(b):
        with out_adv:
            clear_output(wait=True)
            res = get_graph_assortativity(G)
            display(pd.DataFrame([res]).T.rename(columns={0: "Value"}))
            fig = get_graph_kcore_fig(G)
            display(fig)
            plt.close(fig)
    btn_adv.on_click(_plot_adv)

    tabs = widgets.Tab(children=[
        out_stats,
        widgets.VBox([widgets.HBox([density_slider, layout_dd]), plot_btn, out_viz]),
        widgets.VBox([btn_cent, out_cent]),
        widgets.VBox([btn_comm, out_comm]),
        widgets.VBox([btn_adv, out_adv])
    ])
    
    for i, t in enumerate(["Global Stats", "Viz & Topology", "Centrality", "Communities", "Assortativity/K-Core"]):
        tabs.set_title(i, t)
        
    eda_ui.dynamic_ui.children = [tabs]
