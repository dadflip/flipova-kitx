import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def get_neo4j_schema(driver_or_data):
    """Retrieve basic schema from a neo4j driver or return dict if mock."""
    try:
        if hasattr(driver_or_data, "session"):
            with driver_or_data.session() as session:
                node_labels = session.run("CALL db.labels()").value()
                rel_types = session.run("CALL db.relationshipTypes()").value()
            return {"Node Labels": node_labels, "Relationship Types": rel_types}
        elif isinstance(driver_or_data, dict):
            return driver_or_data.get("schema", {"Note": "No schema in mock data"})
        return {"Error": "Invalid driver object"}
    except Exception as e:
        return {"Error": str(e)}

def get_neo4j_nodes_distribution_fig(driver_or_data):
    try:
        data_counts = {}
        if hasattr(driver_or_data, "session"):
            with driver_or_data.session() as session:
                labels = session.run("CALL db.labels()").value()
                for label in labels:
                    count = session.run(f"MATCH (n:`{label}`) RETURN count(n)").value()[0]
                    data_counts[label] = count
        elif isinstance(driver_or_data, dict):
            data_counts = driver_or_data.get("nodes", {})
            
        if not data_counts: return None
        
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor("#f8fafc")
        ax.pie(list(data_counts.values()), labels=list(data_counts.keys()), autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        ax.set_title("Node Label Distribution")
        plt.tight_layout()
        return fig
    except Exception as e:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, str(e), ha="center")
        return fig

def get_neo4j_edges_distribution_fig(driver_or_data):
    try:
        data_counts = {}
        if hasattr(driver_or_data, "session"):
            with driver_or_data.session() as session:
                rel_types = session.run("CALL db.relationshipTypes()").value()
                for r in rel_types:
                    count = session.run(f"MATCH ()-[rel:`{r}`]->() RETURN count(rel)").value()[0]
                    data_counts[r] = count
        elif isinstance(driver_or_data, dict):
            data_counts = driver_or_data.get("edges", {})
            
        if not data_counts: return None
        
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#f8fafc")
        sns.barplot(x=list(data_counts.keys()), y=list(data_counts.values()), palette="viridis", ax=ax)
        ax.set_title("Relationship Type Distribution")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig
    except Exception as e:
        return None

def get_neo4j_sample_graph_fig(driver_or_data, limit=50):
    try:
        import networkx as nx
        G = nx.DiGraph()
        
        if hasattr(driver_or_data, "session"):
            with driver_or_data.session() as session:
                result = session.run(f"MATCH (n)-[r]->(m) RETURN n, r, m LIMIT {limit}")
                for record in result:
                    n = record["n"]
                    m = record["m"]
                    r = record["r"]
                    G.add_node(n.id, labels=list(n.labels))
                    G.add_node(m.id, labels=list(m.labels))
                    G.add_edge(n.id, m.id, type=r.type)
        else:
            # Fallback mock graph
            G = nx.fast_gnp_random_graph(20, 0.2, directed=True)
            
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor("#f8fafc")
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=False, node_color='teal', node_size=100, edge_color='gray', ax=ax)
        ax.set_title("Neo4j Sample Subgraph")
        plt.tight_layout()
        return fig
    except Exception as e:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, str(e), ha="center")
        return fig

def run_neo4j_cypher(driver_or_data, query):
    try:
        if hasattr(driver_or_data, "session"):
            with driver_or_data.session() as session:
                result = session.run(query)
                return pd.DataFrame([dict(record) for record in result])
        return pd.DataFrame([{"Error": "Not connected to live Neo4j database"}])
    except Exception as e:
        return pd.DataFrame([{"Error": str(e)}])
