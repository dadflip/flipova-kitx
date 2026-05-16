import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def _to_networkx(graph_data):
    # Convert RDFLib to NetworkX if necessary
    import networkx as nx
    if hasattr(graph_data, "subjects"): # Simple check for RDFLib
        G = nx.DiGraph()
        for s, p, o in graph_data:
            G.add_edge(str(s), str(o), label=str(p))
        return G
    return graph_data

def build_graph_ui(parent_ui, graph_data):
    out = widgets.Output()
    
    # 1. Extraction Topology (Nodes)
    btn_extract_nodes = widgets.Button(description="Features Nœuds (Topologie)", button_style="info", layout=widgets.Layout(width="280px"))
    
    # 2. Extraction Recommendation / Link Prediction (Edges)
    btn_extract_edges = widgets.Button(description="Recommendation (Features liens)", button_style="success", layout=widgets.Layout(width="280px"))

    # 3. Visualisation riche
    btn_viz = widgets.Button(description="Visualiser le graphe", button_style="primary", layout=widgets.Layout(width="200px"))
    layout_opts = widgets.Dropdown(options=["spring", "kamada_kawai", "circular", "shell"], value="spring", description="Layout:", layout=widgets.Layout(width="200px"))
    max_nodes = widgets.IntText(value=100, description="Max nœuds:", layout=widgets.Layout(width="150px"))
    
    def _on_extract_nodes(_):
        with out:
            clear_output()
            try:
                import networkx as nx
                G = _to_networkx(graph_data)
                display(HTML("<b style='color:#3b82f6;'>Extraction des features des nœuds en cours...</b>"))
                degrees = dict(G.degree())
                # centrality calculations can be slow or fail on disconnected directed graphs
                if G.is_directed():
                    G_un = G.to_undirected()
                else:
                    G_un = G
                centrality = nx.betweenness_centrality(G_un)
                pagerank = nx.pagerank(G)
                closeness = nx.closeness_centrality(G)
                
                df = pd.DataFrame([
                    {
                        "node": str(n),
                        "degree": degrees.get(n, 0),
                        "betweenness_centrality": centrality.get(n, 0),
                        "pagerank": pagerank.get(n, 0),
                        "closeness_centrality": closeness.get(n, 0)
                    }
                    for n in G.nodes()
                ])
                
                new_ds_name = f"{parent_ui.current_ds} - Nodes"
                parent_ui.state.data_raw[new_ds_name] = df
                parent_ui.state.data_types[new_ds_name] = "tabular"
                display(HTML(f"<b style='color:#10b981;'>✓ Feature nœuds extraites ({len(df)} lignes). Sauvegardé sous '{new_ds_name}'.</b>"))
            except Exception as e:
                display(HTML(f"<b style='color:#ef4444;'>Erreur extraction nœuds: {e}</b>"))
                
    def _on_extract_edges(_):
        with out:
            clear_output()
            try:
                import networkx as nx
                G = _to_networkx(graph_data)
                if G.is_directed():
                    display(HTML("<div style='color:#f59e0b;'>Conversion du graphe orienté en non-orienté pour Jaccard/Adamic-Adar.</div>"))
                    G_un = G.to_undirected()
                else:
                    G_un = G
                    
                display(HTML("<b style='color:#3b82f6;'>Génération des features de paires pour la recommandation...</b>"))
                
                # Jaccard
                preds_jaccard = nx.jaccard_coefficient(G_un)
                dict_jaccard = { (u, v): p for u, v, p in preds_jaccard }
                
                # Adamic Adar
                preds_adamic = nx.adamic_adar_index(G_un)
                dict_adamic = { (u, v): p for u, v, p in preds_adamic }
                
                # Preferential Attachment
                preds_pref = nx.preferential_attachment(G_un)
                dict_pref = { (u, v): p for u, v, p in preds_pref }

                edges_data = []
                for u, v in G.edges():
                    pair = (u,v) if (u,v) in dict_jaccard else (v,u)
                    
                    edge_dict = {
                        "node_1": str(u),
                        "node_2": str(v),
                        "is_connected": 1,
                        "jaccard_coef": dict_jaccard.get(pair, 0),
                        "adamic_adar": dict_adamic.get(pair, 0),
                        "pref_attach": dict_pref.get(pair, 0)
                    }
                    
                    if G.is_directed():
                        edge_data_attr = G.get_edge_data(u, v)
                        if edge_data_attr and 'label' in edge_data_attr:
                            edge_dict["relation"] = edge_data_attr['label']
                            
                    edges_data.append(edge_dict)
                    
                df = pd.DataFrame(edges_data)
                
                new_ds_name = f"{parent_ui.current_ds} - Edges (RecSys)"
                parent_ui.state.data_raw[new_ds_name] = df
                parent_ui.state.data_types[new_ds_name] = "tabular"
                display(HTML(f"<b style='color:#10b981;'>✓ Features d'arêtes extraites ({len(df)} lignes). Utile pour recommandations. Sauvegardé sous '{new_ds_name}'.</b>"))
            except Exception as e:
                display(HTML(f"<b style='color:#ef4444;'>Erreur extraction recommandations: {e}</b>"))
                
    def _on_viz(_):
        with out:
            clear_output()
            try:
                import networkx as nx
                G = _to_networkx(graph_data)
                n_max = max_nodes.value
                lay = layout_opts.value
                
                sub_nodes = list(G.nodes())[:n_max]
                H = G.subgraph(sub_nodes)
                
                fig, ax = plt.subplots(figsize=(12, 8))
                fig.patch.set_facecolor("#ffffff")
                
                display(HTML(f"<b style='color:#374151;'>Affichage du sous-graphe ({H.number_of_nodes()} nœuds, {H.number_of_edges()} liens)...</b>"))
                
                if lay == "kamada_kawai": pos = nx.kamada_kawai_layout(H)
                elif lay == "circular": pos = nx.circular_layout(H)
                elif lay == "shell": pos = nx.shell_layout(H)
                else: pos = nx.spring_layout(H, seed=42)
                    
                deg = dict(H.degree())
                if not deg:
                    max_d = 1
                else:
                    max_d = max(deg.values()) if max(deg.values()) > 0 else 1
                
                node_sizes  = [100 + 900 * (deg.get(n, 0) / max_d) for n in H.nodes()]
                node_colors = [deg.get(n, 0) for n in H.nodes()]
                
                nc = nx.draw_networkx_nodes(H, pos, node_size=node_sizes,
                                             node_color=node_colors, cmap=plt.cm.coolwarm,
                                             alpha=0.9, ax=ax, edgecolors="white", linewidths=1.5)
                
                plt.colorbar(nc, ax=ax, label="Degré", shrink=0.7)
                nx.draw_networkx_edges(H, pos, edge_color="#94a3b8", alpha=0.6,
                                       arrows=G.is_directed(), arrowsize=12, ax=ax)
                
                # Text labels for nodes
                def shorten(s, length=15):
                    s = str(s)
                    # if ontology URI, take the last part
                    if "#" in s: s = s.split("#")[-1]
                    elif "/" in s: s = s.split("/")[-1]
                    return s[:length] + ("..." if len(s) > length else "")
                    
                labels = {n: shorten(n) for n in H.nodes()}
                for node, (x, y) in pos.items():
                    ax.text(x, y, labels[node], fontsize=8, ha='center', va='center',
                            color="black", fontweight="bold",
                            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.2"))
                
                # Edge labels (if exist and not too many nodes to avoid clutter)
                if n_max <= 50:
                    edge_labels = {}
                    for u, v, d in H.edges(data=True):
                        if 'label' in d:
                            edge_labels[(u, v)] = shorten(d['label'], 10)
                    if edge_labels:
                        nx.draw_networkx_edge_labels(H, pos, edge_labels=edge_labels, font_size=7, ax=ax, font_color="#475569")
                
                ax.set_title(f"Graphe — {lay}", fontsize=12, fontweight="bold", color="#1e293b")
                ax.axis("off")
                plt.tight_layout()
                display(fig); plt.close(fig)
            except Exception as e:
                display(HTML(f"<b style='color:#ef4444;'>Erreur viz: {e}</b>"))

    btn_extract_nodes.on_click(_on_extract_nodes)
    btn_extract_edges.on_click(_on_extract_edges)
    btn_viz.on_click(_on_viz)
    
    parent_ui.dynamic_ui.children = [
        widgets.HTML("<div style='background:#f8fafc; padding:15px; border-radius:8px; border:1px solid #e2e8f0;'>"
                     "<h3 style='margin-top:0;color:#1e293b;'>Graphe & Ontologie — Feature Engineering</h3>"
                     "<p style='color:#475569;'>Extrayez des caractéristiques structurelles pour l'apprentissage automatique ou préparez les données pour un problème de <b>Recommandation</b>.</p>"
                     "</div>"),
        widgets.HTML("<br><b>1. Extraction de caractéristiques (Features)</b>"),
        widgets.HBox([btn_extract_nodes, btn_extract_edges], layout=widgets.Layout(gap="10px", margin="10px 0")),
        widgets.HTML("<br><b>2. Visualisation Avancée</b>"),
        widgets.HBox([layout_opts, max_nodes, btn_viz], layout=widgets.Layout(gap="10px", margin="10px 0")),
        out
    ]

