import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import pandas as pd
import matplotlib.pyplot as plt

from ml_pipeline.styles import styles
from ml_pipeline.steps.eda.logic.ontology import (
    get_ontology_stats, get_ontology_graph_fig, 
    get_ontology_hierarchy_fig, get_namespaces_fig, _short,
    get_ontology_domain_range, get_ontology_schema_fig
)

def build_ontology_ui(eda_ui, g) -> None:
    eda_cfg = getattr(eda_ui.state, "config", {}).get("eda", {})
    onto_cfg = eda_cfg.get("ontology", {})
    stats = get_ontology_stats(g)
    
    if not stats:
        eda_ui.dynamic_ui.children = [widgets.HTML(
            "<div style='padding:12px;color:#b91c1c;'>rdflib non disponible.</div>")]
        return

    # ── Tab 0 : Stats 
    stats_out = widgets.Output()
    with stats_out:
        summary_rows = [
            {"Élément": "Triples totaux",        "Nombre": stats["n_triples"]},
            {"Élément": "Classes OWL",            "Nombre": len(stats["classes"])},
            {"Élément": "Object Properties",      "Nombre": len(stats["props_obj"])},
            {"Élément": "Datatype Properties",    "Nombre": len(stats["props_dt"])},
            {"Élément": "Annotation Properties",  "Nombre": len(stats["props_ann"])},
            {"Élément": "Named Individuals",      "Nombre": len(stats["inds"])},
            {"Élément": "Namespaces distincts",   "Nombre": len(stats["top_ns"])},
        ]
        display(HTML("<b style='color:#374151;font-size:0.9em;'>Résumé de l'ontologie</b>"))
        display(pd.DataFrame(summary_rows).set_index("Élément"))

    # ── Tab 1 : Graphe 
    graph_out = widgets.Output()
    default_density = int(onto_cfg.get("default_density", 40))
    max_density     = int(onto_cfg.get("max_density", 200))
    density_slider  = widgets.IntSlider(
        value=default_density, min=5, max=max_density, step=5,
        description="Densité (arêtes):",
        layout=widgets.Layout(width="380px"))
    layout_dd = widgets.Dropdown(
        options=onto_cfg.get("graph_layouts", ["spring", "kamada_kawai", "circular", "shell"]),
        value="spring", description="Layout:",
        layout=widgets.Layout(width="200px"))
    rel_filter = widgets.SelectMultiple(
        options=onto_cfg.get("relation_types",
                              ["subClassOf", "domain", "range", "type"]),
        value=onto_cfg.get("relation_types",
                            ["subClassOf", "domain", "range", "type"]),
        description="Relations:",
        rows=4, layout=widgets.Layout(width="260px"))
    plot_btn  = widgets.Button(description="Générer graphe", button_style=styles.BTN_PRIMARY)
    save_btn  = widgets.Button(description="Save Dashboard", button_style="info")
    eda_ui._last_onto_fig = None

    def _plot_onto_graph(_=None):
        with graph_out:
            clear_output(wait=True)
            fig = get_ontology_graph_fig(g, density_slider.value, layout_dd.value, list(rel_filter.value))
            if fig:
                display(fig)
                eda_ui._last_onto_fig = fig
                plt.close(fig)
            else:
                print("Aucune relation trouvée avec les filtres actuels.")

    plot_btn.on_click(_plot_onto_graph)
    save_btn.on_click(lambda b: eda_ui.dashboard.add(
        eda_ui._last_onto_fig, f"Ontologie ({density_slider.value} arêtes)") if eda_ui._last_onto_fig else None)

    graph_tab = widgets.VBox([
        styles.help_box("Génère un graphe des triplets filtrés.", "#8b5cf6"),
        widgets.HBox([density_slider, layout_dd]),
        widgets.HBox([rel_filter]),
        widgets.HBox([plot_btn, save_btn]),
        graph_out])

    # ── Tab 2 : Triplets 
    triplets_out = widgets.Output()
    pred_filter  = widgets.Dropdown(
        options=["(tous)"] + sorted({_short(p) for _, p, _ in g}),
        value="(tous)", description="Prédicat:")
    subj_filter  = widgets.Text(description="Sujet:")
    filter_btn   = widgets.Button(description="Filtrer", button_style=styles.BTN_INFO)
    
    def _show_triplets(_=None):
        with triplets_out:
            clear_output(wait=True)
            pred_sel = pred_filter.value
            subj_sel = subj_filter.value.strip().lower()
            rows = []
            for s, p, o in g:
                if pred_sel != "(tous)" and _short(p) != pred_sel: continue
                if subj_sel and subj_sel not in _short(s).lower(): continue
                rows.append({"Subject": _short(s), "Predicate": _short(p), "Object": _short(o)})
                if len(rows) >= 200: break
            display(pd.DataFrame(rows) if rows else HTML("<i>Aucun résultat</i>"))

    filter_btn.on_click(_show_triplets)
    triplets_tab = widgets.VBox([widgets.HBox([pred_filter, subj_filter, filter_btn]), triplets_out])
    
    # ── Tab 3 : NS et Hié (Split into 2)
    ns_out = widgets.Output()
    with ns_out:
        fig = get_namespaces_fig(stats["top_ns"])
        if fig:
            display(fig)
            plt.close(fig)

    hier_out = widgets.Output()
    hier_btn = widgets.Button(description="Générer hiérarchie", button_style=styles.BTN_PRIMARY)
    
    def _build_hierarchy(_=None):
        with hier_out:
            clear_output(wait=True)
            fig = get_ontology_hierarchy_fig(g, 60)
            if fig:
                display(fig)
                plt.close(fig)
            else:
                print("Pas de hiérarchie.")
                
    hier_btn.on_click(_build_hierarchy)
    hier_tab = widgets.VBox([hier_btn, hier_out])

    # ── Tab 5 : Schema (Domain & Range)
    schema_out = widgets.Output()
    schema_btn = widgets.Button(description="Afficher Schéma (Domaines / Ranges)", button_style=styles.BTN_PRIMARY)
    def _build_schema(_=None):
        with schema_out:
            clear_output(wait=True)
            res = get_ontology_domain_range(g)
            
            fig = get_ontology_schema_fig(g)
            if fig:
                display(fig)
                plt.close(fig)
            else:
                print("Pas de relations domaine/range pour le schéma visuel.")
                
            if res:
                display(HTML("<b style='color:#374151;font-size:0.9em;margin-top:10px;display:block;'>Tableau récapitulatif</b>"))
                display(pd.DataFrame(res).set_index("Property"))

    schema_btn.on_click(_build_schema)
    schema_tab = widgets.VBox([
        styles.help_box("Affiche la structure abstraite de l'ontologie (classes et leurs relations) basée sur RDFS.domain et RDFS.range.", "#8b5cf6"),
        schema_btn, schema_out
    ])

    # ── Assemblage 
    onto_tabs = widgets.Tab(children=[stats_out, graph_tab, triplets_tab, hier_tab, schema_tab])
    for i, title in enumerate(["Stats & Overview", "Interactive Graph", "Triplets Query", "Class Hierarchy", "Ontology Schema"]):
        onto_tabs.set_title(i, title)

    eda_ui.dynamic_ui.children = [onto_tabs]
