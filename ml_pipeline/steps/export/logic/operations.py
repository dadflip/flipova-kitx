import json
import os

def export_python_script(state, output_path: str = "exported_pipeline.py") -> None:
    script = [
        "# Auto-generated ML Pipeline Script",
        "import pandas as pd",
        "import numpy as np",
        "import joblib",
        "from sklearn.model_selection import train_test_split",
        "",
    ]
    for item in state.history:
        script.append(f"# --- Step: {item['step']} | Action: {item['action']} ---")
        script.append(f"# Parameters used: {json.dumps(item['details'], default=str)}")
        script.append("")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(script))
    print(f"Script exporté → {output_path}")

def export_models(state, path: str = "trained_models.pkl") -> None:
    if state.models:
        import joblib
        joblib.dump(state.models, path)
        print(f"Modèles exportés → {path}")
    else:
        print("Aucun modèle à exporter.")

def generate_report(state, path: str = "execution_report.html") -> None:
    html = ["<html><head><meta charset='utf-8'><title>ML Report</title>",
            "<style>body{font-family:sans-serif;max-width:900px;margin:40px auto;color:#1e293b;}",
            "h1{color:#6d28d9;}h2{color:#374151;border-bottom:1px solid #e2e8f0;padding-bottom:6px;}",
            "li{margin-bottom:4px;}pre{background:#f8fafc;padding:10px;border-radius:6px;font-size:0.85em;}</style>",
            "</head><body>",
            "<h1>Machine Learning Execution Report</h1>"]
    if state.business_context:
        html.append("<h2>Business Context</h2><ul>")
        for k, v in state.business_context.items():
            html.append(f"<li><b>{k}</b>: {v}</li>")
        html.append("</ul>")
    if state.models:
        html.append("<h2>Trained Models</h2><ul>")
        for name in state.models:
            html.append(f"<li>{name}</li>")
        html.append("</ul>")
    html.append("<h2>Execution History</h2><ul>")
    for step in state.history:
        html.append(f"<li><b>Step: {step['step']}</b> — {step['action']}<br>"
                    f"<pre>{json.dumps(step['details'], default=str, indent=2)}</pre></li>")
    html.append("</ul></body></html>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print(f"Rapport exporté → {path}")
