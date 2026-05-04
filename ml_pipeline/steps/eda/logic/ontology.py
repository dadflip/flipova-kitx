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
    if not hasattr(g, "subjects"):
        return {}
        
    try:
        from rdflib.namespace import OWL, RDF, RDFS
    except ImportError:
        return {}

    try:
        classes = list(g.subjects(RDF.type, OWL.Class))
    except TypeError:
        return {}
        
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
    if not hasattr(g, "__iter__"): return None
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

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor("#f8fafc"); ax.set_facecolor("#f8fafc")
    try:
        if lay == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        elif lay == "circular":
            pos = nx.circular_layout(G)
        elif lay == "shell":
            pos = nx.shell_layout(G)
        else:
            pos = nx.spring_layout(G, k=2.5, seed=42)
    except Exception:
        pos = nx.spring_layout(G, k=2.5, seed=42)

    # Do not draw networkx defaults, construct bounded labels
    nx.draw_networkx_edges(G, pos, edge_color="#94a3b8",
                           arrows=True, arrowsize=14,
                           connectionstyle="arc3,rad=0.1", ax=ax, node_size=1000)
    
    for node in G.nodes():
        if G.in_degree(node) == 0:
            facecolor = "#bae6fd" # light blue
        elif G.out_degree(node) == 0:
            facecolor = "#a7f3d0" # light green
        else:
            facecolor = "#ddd6fe" # light purple
            
        ax.text(pos[node][0], pos[node][1], str(node), 
                fontsize=8, color="black", 
                ha="center", va="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", facecolor=facecolor, edgecolor="gray", alpha=0.9))

    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                  font_size=7, font_color="#475569", ax=ax)
                                  
    ax.set_title(
        f"Graphe ontologique (complet) — {G.number_of_nodes()} nœuds · {G.number_of_edges()} arêtes "
        f"· layout={lay}",
        color="#334155", fontsize=11, weight="bold"
    )
    plt.axis('off')
    
    from matplotlib.patches import Patch
    legend = [Patch(color="#bae6fd", label="Racines"),
              Patch(color="#ddd6fe", label="Intermédiaires"),
              Patch(color="#a7f3d0", label="Feuilles")]
    ax.legend(handles=legend, loc="lower right", fontsize=8,
              framealpha=0.9, edgecolor="#e5e7eb")
    plt.tight_layout()
    return fig

def get_ontology_hierarchy_fig(g, max_hier: int) -> plt.Figure | None:
    if not hasattr(g, "triples"): return None
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
        
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor("#f8fafc"); ax.set_facecolor("#f8fafc")
    roots = [n for n in H.nodes() if H.in_degree(n) == 0]
    leaves = [n for n in H.nodes() if H.out_degree(n) == 0]
    
    nx.draw_networkx_edges(H, pos, edge_color="#94a3b8",
                           arrows=True, arrowsize=14, ax=ax, node_size=1000)
    
    for node in H.nodes():
        if node in roots:
            facecolor = "#bae6fd" # light blue
        elif node in leaves:
            facecolor = "#a7f3d0" # light green
        else:
            facecolor = "#ddd6fe" # light purple
            
        ax.text(pos[node][0], pos[node][1], str(node), 
                fontsize=8, color="black", 
                ha="center", va="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", facecolor=facecolor, edgecolor="gray", alpha=0.9))

    ax.set_title(
        f"Hiérarchie de classes (subClassOf) — {H.number_of_nodes()} classes",
        color="#334155", fontsize=11, weight="bold"
    )
    ax.axis("off")
    from matplotlib.patches import Patch
    legend = [Patch(color="#bae6fd", label="Racines"),
              Patch(color="#ddd6fe", label="Intermédiaires"),
              Patch(color="#a7f3d0", label="Feuilles")]
    ax.legend(handles=legend, loc="lower right", fontsize=8, framealpha=0.9, edgecolor="#e5e7eb")
    plt.tight_layout()
    return fig

def get_ontology_schema_fig(g) -> plt.Figure | None:
    if not hasattr(g, "triples"): return None
    try:
        from rdflib.namespace import RDFS
        import networkx as nx
    except ImportError:
        return None

    S = nx.DiGraph()
    prop_edges = {}
    for s, _, o in g.triples((None, RDFS.domain, None)):
        prop_edges[_short(s)] = {'domain': _short(o)}
    for s, _, o in g.triples((None, RDFS.range, None)):
        if _short(s) not in prop_edges:
            prop_edges[_short(s)] = {}
        prop_edges[_short(s)]['range'] = _short(o)
        
    for p, dr in prop_edges.items():
        if 'domain' in dr and 'range' in dr:
            d, r = dr['domain'], dr['range']
            if d and r:
                S.add_edge(d, r, label=p)
    
    if S.number_of_nodes() == 0:
        return None

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor("#f8fafc"); ax.set_facecolor("#f8fafc")
    try:
        pos = nx.spring_layout(S, k=3.0, seed=42)
    except:
        return None

    nx.draw_networkx_edges(S, pos, edge_color="#94a3b8",
                           arrows=True, arrowsize=14, ax=ax, node_size=1000)
    
    for node in S.nodes():
        facecolor = "#fef08a" # yellow
        ax.text(pos[node][0], pos[node][1], str(node), 
                fontsize=9, color="black", 
                ha="center", va="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", facecolor=facecolor, edgecolor="gray", alpha=0.9))

    edge_labels = nx.get_edge_attributes(S, "label")
    nx.draw_networkx_edge_labels(S, pos, edge_labels=edge_labels,
                                  font_size=8, font_color="#475569", ax=ax)
                                  
    ax.set_title(
        f"Schéma Récapitulatif (Domaines et Ranges)",
        color="#334155", fontsize=11, weight="bold"
    )
    plt.axis('off')
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
    if not hasattr(g, "triples"): return []
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

