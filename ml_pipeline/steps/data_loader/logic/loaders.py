import pandas as pd
import numpy as np
import io
import requests

def load_data(ds_type, source_type, path_or_content, adv_options):
    """
    Load data based on type and advanced options.
    source_type: 'local', 'url', 'upload'
    path_or_content: internal string path, URL, or bytes from upload
    adv_options: dict of parsed advanced options
    """
    try:
        if ds_type in ("csv", "timeseries"):
            sep = adv_options.get("sep", ",")
            if sep == "": sep = ","
            header = adv_options.get("header", "infer")
            if header == "None": header = None
            elif header.isdigit(): header = int(header)
            
            kwargs = {"sep": sep, "header": header}
            
            if "enc" in adv_options and adv_options["enc"]: kwargs["encoding"] = adv_options["enc"]
            if "skiprows" in adv_options and adv_options["skiprows"].isdigit(): kwargs["skiprows"] = int(adv_options["skiprows"])
            if "nrows" in adv_options and adv_options["nrows"].isdigit(): kwargs["nrows"] = int(adv_options["nrows"])
            if "index_col" in adv_options and adv_options["index_col"]: kwargs["index_col"] = adv_options["index_col"]
            if "parse_dates" in adv_options and adv_options["parse_dates"] is True: kwargs["parse_dates"] = True
                
            if source_type == "upload":
                df = pd.read_csv(io.BytesIO(path_or_content), **kwargs)
            else:
                df = pd.read_csv(path_or_content, **kwargs)
                
            # Sample Frac
            frac = adv_options.get("sample_frac", 1.0)
            if float(frac) < 1.0:
                df = df.sample(frac=float(frac), random_state=42)
            return df
            
        elif ds_type == "json":
            if source_type == "upload":
                return pd.read_json(io.BytesIO(path_or_content))
            return pd.read_json(path_or_content)
            
        elif ds_type == "excel":
            sheet = adv_options.get("sheet_name", 0)
            if str(sheet).isdigit(): sheet = int(sheet)
            kwargs = {"sheet_name": sheet}
            if "skiprows" in adv_options and adv_options["skiprows"].isdigit(): kwargs["skiprows"] = int(adv_options["skiprows"])
            if source_type == "upload":
                return pd.read_excel(io.BytesIO(path_or_content), **kwargs)
            return pd.read_excel(path_or_content, **kwargs)
            
        elif ds_type == "image":
            from PIL import Image
            if source_type == "upload":
                img = Image.open(io.BytesIO(path_or_content))
            elif source_type == "url":
                res = requests.get(path_or_content, stream=True)
                img = Image.open(res.raw)
            else:
                img = Image.open(path_or_content)
                
            # advanced options like convert RGB
            if adv_options.get("convert_rgb") is True and img.mode != "RGB":
                img = img.convert("RGB")
            
            # advanced options like resize
            resize = adv_options.get("resize", "")
            if resize and "," in resize:
                w, h = map(int, resize.split(","))
                img = img.resize((w, h))
            return img
            
        elif ds_type == "video":
            return path_or_content # Videos are not loaded in memory, return path/url
            
        elif ds_type == "web":
            if source_type == "url":
                res = requests.get(path_or_content)
                html = res.text
            elif source_type == "upload":
                html = path_or_content.decode("utf-8")
            else:
                with open(path_or_content, "r", encoding="utf-8") as f:
                    html = f.read()
            return {"html": html}
            
        elif ds_type == "neo4j":
            try:
                uri = adv_options.get("neo4j_uri", path_or_content)
                user = adv_options.get("neo4j_user", "neo4j")
                pwd = adv_options.get("neo4j_password", "")
                
                if uri and str(uri).startswith("bolt") or str(uri).startswith("neo4j"):
                    from neo4j import GraphDatabase
                    driver = GraphDatabase.driver(uri, auth=(user, pwd))
                    return driver
                else:
                    return {"schema": {"Node Labels": ["MockNode"], "Relationship Types": ["MOCK_REL"]}, "Note": "Not a neo4j URI"}
            except Exception as e:
                return {"Error": str(e)}
                
        elif ds_type == "ontology":
            import rdflib
            g = rdflib.Graph()
            
            # Pour l'upload, on a soit path (si local/url), soit les octets direct
            if source_type == "upload" and isinstance(path_or_content, bytes):
                g.parse(data=path_or_content.decode("utf-8", errors="replace"), format="xml")
            else:
                g.parse(path_or_content)
            return g
                
        elif ds_type == "graph":
            import networkx as nx
            if adv_options.get("graph_format", "graphml") == "graphml":
                if source_type == "upload":
                    return nx.read_graphml(io.BytesIO(path_or_content))
                return nx.read_graphml(path_or_content)
            else:
                if source_type == "upload":
                    return nx.read_gml(io.BytesIO(path_or_content))
                return nx.read_gml(path_or_content)
                
        elif ds_type == "text":
            if source_type == "upload":
                return path_or_content.decode("utf-8")
            elif source_type == "url":
                return requests.get(path_or_content).text
            else:
                with open(path_or_content, "r", encoding="utf-8") as f:
                    return f.read()
                    
        else:
            return f"Mock data for {ds_type}"
    except Exception as e:
        raise Exception(f"Erreur de chargement ({ds_type}): {e}")
