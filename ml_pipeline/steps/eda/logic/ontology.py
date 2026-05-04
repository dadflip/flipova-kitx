import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def _short(uri):
    s = str(uri)
    return s.split("#")[-1] if "#" in s else s.rsplit("/", 1)[-1]

def _ns(uri):
    s = str(uri)
    if "#" in s:
        return s.rsplit("#", 1)[0] + "#"
    parts = s.rsplit("/", 1)
    return parts[0] + "/" if len(parts) > 1 else s

def get_ontology_stats(g) -> dict:
    try:
        from rdflib.namespace import OWL, RDF, RDFS
    except ImportError:
        return {}

    classes = list(g.subjects(RDF.type, OWL.Class))
    props_obj = list(g.subjects(RDF.type, OWL.ObjectProperty))
    props_dt = list(g.subjects(RDF.type, OWL.DatatypeProperty))
    props_ann = list(g.subjects(RDF.type, OWL.AnnotationProperty))
    inds = list(g.subjects(RDF.type, OWL.NamedIndividual))
    
    ns_counter = Counter()
    for s, p, o in g:
        ns_counter[_ns(s)] += 1
        ns_counter[_ns(p)] += 1

    return {
        "n_triples": len(g),
        "classes": classes,
        "props_obj": props_obj,
        "props_dt": props_dt,
        "props_ann": props_ann,
        "inds": inds,
        "top_ns": ns_counter.most_common(20)
    }

def get_ontology_graph_fig(g, max_edges: int, lay: str, allowed_relations: list) -> plt.Figure | None:
    try:
        import networkx as nx
        from rdflib.namespace import RDF, RDFS
    except ImportError:
        return None

    G = nx.DiGraph()
    allowed = set(allowed_relations)
    edges_added = 0
    for s, p, o in g:
        if edges_added >= max_edges:
            break
        ps = _short(p)
        if ps in allowed:
            ss, os_ = _short(s), _short(o)
            if ss and os_ and ss != os_:
                G.add_edge(ss, os_, label=ps)
                edges_added += 1

    if G.number_of_nodes() == 0:
        return None

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#f8fafc"); ax.set_facecolor("#f8fafc")
    try:
        if lay == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        elif lay == "circular":
            pos = nx.circular_layout(G)
        elif lay == "shell":
            pos = nx.shell_layout(G)
        else:
            pos = nx.spring_layout(G, k=2.0, seed=42)
    except Exception:
        pos = nx.spring_layout(G, k=2.0, seed=42)

    node_colors = []
    for node in G.nodes():
        if G.in_degree(node) == 0:
            node_colors.append("#3b82f6")
        elif G.out_degree(node) == 0:
            node_colors.append("#10b981")
        else:
            node_colors.append("#8b5cf6")
            
    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=500, alpha=0.88, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, font_color="white",
                            font_weight="bold", ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="#94a3b8",
                           arrows=True, arrowsize=14,
                           connectionstyle="arc3,rad=0.1", ax=ax)
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                  font_size=6, font_color="#475569", ax=ax)
                                  
    ax.set_title(
        f"Graphe ontologique — {G.number_of_nodes()} nœuds · {G.number_of_edges()} arêtes "
        f"· layout={lay}",
        fontsize=9, color="#374151")
    ax.axis("off")
    
    from matplotlib.patches import Patch
    legend = [Patch(color="#3b82f6", label="Racines"),
              Patch(color="#8b5cf6", label="Intermédiaires"),
              Patch(color="#10b981", label="Feuilles")]
    ax.legend(handles=legend, loc="lower right", fontsize=7,
              framealpha=0.8, edgecolor="#e5e7eb")
    plt.tight_layout()
    return fig

def get_ontology_hierarchy_fig(g, max_hier: int) -> plt.Figure | None:
    try:
        from rdflib.namespace import RDFS
        import networkx as nx
    except ImportError:
        return None

    H = nx.DiGraph()
    for s, p, o in g.triples((None, RDFS.subClassOf, None)):
        ss, os_ = _short(s), _short(o)
        if ss and os_ and ss != os_:
            H.add_edge(os_, ss) 
        if H.number_of_nodes() >= max_hier:
            break
            
    if H.number_of_nodes() == 0:
        return None
        
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        pos = graphviz_layout(H, prog="dot")
    except Exception:
        pos = nx.spring_layout(H, k=2.5, seed=42)
        
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#f8fafc"); ax.set_facecolor("#f8fafc")
    roots = [n for n in H.nodes() if H.in_degree(n) == 0]
    leaves = [n for n in H.nodes() if H.out_degree(n) == 0]
    node_colors = [
        "#3b82f6" if n in roots else
        "#10b981" if n in leaves else "#8b5cf6"
        for n in H.nodes()
    ]
    nx.draw_networkx_nodes(H, pos, node_color=node_colors,
                           node_size=450, alpha=0.88, ax=ax)
    nx.draw_networkx_labels(H, pos, font_size=7, font_color="white",
                            font_weight="bold", ax=ax)
    nx.draw_networkx_edges(H, pos, edge_color="#94a3b8",
                           arrows=True, arrowsize=12, ax=ax)
    ax.set_title(
        f"Hiérarchie de classes (subClassOf) — {H.number_of_nodes()} classes",
        fontsize=9, color="#374151")
    ax.axis("off")
    from matplotlib.patches import Patch
    legend = [Patch(color="#3b82f6", label="Racines (Thing)"),
              Patch(color="#8b5cf6", label="Intermédiaires"),
              Patch(color="#10b981", label="Feuilles")]
    ax.legend(handles=legend, loc="lower right", fontsize=7, framealpha=0.8)
    plt.tight_layout()
    return fig

def get_namespaces_fig(top_ns: list) -> plt.Figure | None:
    if not top_ns:
        return None
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#f8fafc"); ax.set_facecolor("#f8fafc")
    labels = [ns.split("/")[-2] if ns.endswith("/") else ns.split("#")[0].rsplit("/", 1)[-1]
              for ns, _ in top_ns[:15]]
    values = [cnt for _, cnt in top_ns[:15]]
    ax.barh(labels[::-1], values[::-1], color="#3b82f6", alpha=0.8)
    ax.set_title("Top namespaces par occurrences", fontsize=9)
    ax.set_xlabel("Occurrences")
    plt.tight_layout()
    return fig

def get_ontology_domain_range(g) -> list:
    try:
        from rdflib.namespace import RDFS
    except ImportError:
        return []
    
    props = {}
    for s, p, o in g.triples((None, RDFS.domain, None)):
        ps = _short(s)
        if ps not in props: props[ps] = {"domain": set(), "range": set()}
        props[ps]["domain"].add(_short(o))
        
    for s, p, o in g.triples((None, RDFS.range, None)):
        ps = _short(s)
        if ps not in props: props[ps] = {"domain": set(), "range": set()}
        props[ps]["range"].add(_short(o))
        
    return [
        {"Property": p, "Domain": ", ".join(d["domain"]), "Range": ", ".join(d["range"])}
        for p, d in props.items()
    ]

