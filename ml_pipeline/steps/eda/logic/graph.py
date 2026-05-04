import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_graph_stats(G) -> dict:
    try:
        import networkx as nx
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        is_dir  = G.is_directed()
        degrees = [d for _, d in G.degree()]
        return {
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "is_dir": is_dir,
            "density": nx.density(G),
            "avg_degree": sum(degrees) / max(n_nodes, 1),
            "max_degree": max(degrees) if degrees else 0,
            "components": nx.number_connected_components(G.to_undirected()),
            "degrees": degrees,
            "top_nodes": sorted(G.degree(), key=lambda x: -x[1])[:15]
        }
    except ImportError:
        return {}

def get_graph_stats_fig(degrees: list, top_nodes: list) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#f8fafc")
    sns.histplot(degrees, bins=30, kde=True, color="#10b981", ax=axes[0])
    axes[0].set_title("Distribution des degrés")
    axes[0].set_xlabel("Degré")
    axes[0].set_facecolor("#f8fafc")
    
    if top_nodes:
        axes[1].barh([str(n) for n, _ in top_nodes[::-1]],
                      [d for _, d in top_nodes[::-1]], color="#3b82f6", alpha=0.8)
    axes[1].set_title("Top 15 nœuds par degré")
    axes[1].set_xlabel("Degré")
    axes[1].set_facecolor("#f8fafc")
    plt.tight_layout()
    return fig

def get_graph_viz_fig(G, n_max: int, lay: str) -> plt.Figure:
    import networkx as nx
    sub_nodes = list(G.nodes())[:n_max]
    H = G.subgraph(sub_nodes)
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#f8fafc"); ax.set_facecolor("#f8fafc")
    
    try:
        if lay == "kamada_kawai":
            pos = nx.kamada_kawai_layout(H)
        elif lay == "circular":
            pos = nx.circular_layout(H)
        elif lay == "shell":
            pos = nx.shell_layout(H)
        elif lay == "spectral":
            pos = nx.spectral_layout(H)
        else:
            pos = nx.spring_layout(H, k=1.5, seed=42)
    except Exception:
        pos = nx.spring_layout(H, k=1.5, seed=42)
        
    deg = dict(H.degree())
    max_d = max(deg.values()) if deg else 1
    node_sizes  = [200 + 800 * (deg.get(n, 0) / max_d) for n in H.nodes()]
    node_colors = [deg.get(n, 0) for n in H.nodes()]
    
    nc = nx.draw_networkx_nodes(H, pos, node_size=node_sizes,
                                 node_color=node_colors, cmap=plt.cm.viridis,
                                 alpha=0.85, ax=ax)
    plt.colorbar(nc, ax=ax, label="Degré", shrink=0.6)
    nx.draw_networkx_edges(H, pos, edge_color="#cbd5e1", alpha=0.5,
                           arrows=G.is_directed(), arrowsize=10, ax=ax)
    if n_max <= 50:
        nx.draw_networkx_labels(H, pos, font_size=7, font_color="white",
                                font_weight="bold", ax=ax)
    
    ax.set_title(
        f"Graphe — {H.number_of_nodes()} nœuds · {H.number_of_edges()} arêtes "
        f"(/{G.number_of_nodes()} nœuds total) · layout={lay}",
        fontsize=9, color="#374151")
    ax.axis("off")
    plt.tight_layout()
    return fig

def get_graph_centrality_fig(G, n_max=10):
    try:
        import networkx as nx
        # Use a sub-sample if graph is huge
        if G.number_of_nodes() > 1000:
            H = G.subgraph(list(G.nodes())[:1000])
        else:
            H = G
            
        deg_cent = nx.degree_centrality(H)
        bet_cent = nx.betweenness_centrality(H)
        clo_cent = nx.closeness_centrality(H)
        
        def _get_top(metric_dict):
            return sorted(metric_dict.items(), key=lambda x: -x[1])[:n_max]
            
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.patch.set_facecolor("#f8fafc")
        
        for k, (ax, title, data) in enumerate([
            (axes[0], "Degree Centrality", _get_top(deg_cent)),
            (axes[1], "Betweenness Centrality", _get_top(bet_cent)),
            (axes[2], "Closeness Centrality", _get_top(clo_cent))
        ]):
            labels = [str(k) for k, v in data][::-1]
            vals = [v for k, v in data][::-1]
            ax.barh(labels, vals, color=sns.color_palette("muted")[k])
            ax.set_title(title)
            
        plt.tight_layout()
        return fig
    except Exception as e:
        fig, ax = plt.subplots(figsize=(8,4))
        ax.text(0.5, 0.5, f"Erreur Centrality: {e}", ha="center")
        return fig

def get_graph_communities_fig(G):
    try:
        import networkx as nx
        if nx.is_directed(G):
            H = G.to_undirected()
        else:
            H = G
            
        if H.number_of_nodes() > 1000:
            H = H.subgraph(list(H.nodes())[:1000])
            
        from networkx.algorithms.community import greedy_modularity_communities
        communities = greedy_modularity_communities(H)
        
        pos = nx.spring_layout(H)
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor("#f8fafc")
        ax.set_facecolor("#f8fafc")
        
        colors = sns.color_palette("hls", n_colors=len(communities))
        for i, com in enumerate(communities):
            idx = list(com)
            nx.draw_networkx_nodes(H, pos, nodelist=idx, node_color=[colors[i]], 
                                   node_size=100, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(H, pos, edge_color="#cbd5e1", alpha=0.3, ax=ax)
        ax.set_title(f"Community Detection (Found {len(communities)} communities)")
        ax.axis('off')
        plt.tight_layout()
        return fig
    except Exception as e:
        fig, ax = plt.subplots(figsize=(8,4))
        ax.text(0.5, 0.5, f"Erreur ou trop large: {e}", ha="center")
        return fig

def get_graph_assortativity(G):
    try:
        import networkx as nx
        return {
            "Assortativity Degree": nx.degree_assortativity_coefficient(G),
            "Average Clustering": nx.average_clustering(G) if not G.is_directed() else "N/A (Directed)"
        }
    except Exception as e:
        return {"Error": str(e)}

def get_graph_kcore_fig(G):
    try:
        import networkx as nx
        H = G.to_undirected() if nx.is_directed(G) else G
        H.remove_edges_from(nx.selfloop_edges(H))
        core_numbers = nx.core_number(H)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#f8fafc")
        sns.histplot(list(core_numbers.values()), bins=range(0, max(core_numbers.values())+2),
                     kde=False, color="purple", ax=ax, discrete=True)
        ax.set_title("K-Core Decomposition (Node Core Numbers)")
        ax.set_xlabel("K-Core Level")
        ax.set_ylabel("Count of Nodes")
        plt.tight_layout()
        return fig
    except Exception as e:
        fig, ax = plt.subplots(figsize=(8,4))
        ax.text(0.5, 0.5, f"Erreur K-Core: {e}", ha="center")
        return fig
