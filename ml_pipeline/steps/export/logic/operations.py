import json
import os

def get_python_script_string(state) -> str:
    script = [
        "\"\"\"",
        "Auto-generated Script using ml_pipeline package",
        "",
        "Ce script reproduit les actions que vous avez effectuées via l'UI",
        "de manière programmatique en utilisant l'API locale ml_pipeline.",
        "\"\"\"",
        "import pandas as pd",
        "import numpy as np",
        "import joblib",
        "from ml_pipeline.engine.state import State",
        "",
        "# Initialisation de l'état",
        "state = State(config_path='ml_pipeline/default.toml')",
        ""
    ]
    
    # Analyze history to reconstruct python code using the operations
    for item in state.history:
        step_name = item['step']
        action = item['action']
        details = item['details']
        
        script.append(f"\n# --- Step: {step_name} | Action: {action} ---")
        
        if step_name == "loading" and action == "all_datasets_loaded":
            script.append("# Chargement des datasets simulé (Remplacer par vos propres chargements)")
            script.append("# Note: Vous devrez fournir dynamiquement via pd.read_csv(...)")
            script.append("datasets_dict = {}")
            script.append(f"state.datasets = datasets_dict")
        
        elif step_name == "Feature Eng" and action == "Math applied":
            script.append("from ml_pipeline.steps.feature_eng.logic.operations import apply_math_op")
            script.append(f"try:")
            script.append(f"    # details: {details}")
            script.append(f"    # state.datasets['{details.get('dataset', 'none')}'] = apply_math_op(...)")
            script.append(f"    pass")
            script.append(f"except Exception as e: print(e)")
            
        elif step_name == "Data Cleaning":
            script.append("from ml_pipeline.steps.cleaner.logic.operations import clean_data")
            script.append(f"# Nettoyage avec les actions enregistrées")
            for ds, acts in details.items():
                script.append(f"actions_{ds.replace(' ', '_')} = {acts}")
                script.append(f"state.datasets['{ds}'] = clean_data(state.datasets['{ds}'], actions_{ds.replace(' ', '_')})")

        elif step_name == "Dataset Config" and "Train/Test Selection & Balance" in action:
            script.append("from ml_pipeline.steps.balancing.logic.operations import do_split")
            script.append(f"# Splits: {details}")
            script.append(f"state.splits = do_split(state.datasets, {details}, state.config)")
            
        elif step_name == "Modeling" and "Models Trained" in action:
            script.append("from ml_pipeline.steps.modeling.logic.operations import train_models")
            model_list = details.get("models_selected", [])
            script.append(f"models_to_train = {model_list}")
            script.append("task_type = 'classification' # or 'regression' (déduire de split_strategy)")
            script.append(f"trained_models = train_models(state.splits, models_to_train, task_type, state.config)")
            script.append("state.models = trained_models")
        
        else:
            # Fallback
            script.append(f"# Configuration/Détails de l'historique enregistrés :")
            script.append(f"# {json.dumps(details, default=str)}")
            script.append(f"pass")

    script.append("\n# --- Fin du pipeline généré ---")
    script.append("# Vous pouvez désormais utiliser state.models pour vos prédictions.")
    script.append("")
        
    return "\n".join(script)

def export_python_script(state, output_path: str = "exported_pipeline.py") -> None:
    script_content = get_python_script_string(state)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(script_content)
    print(f"Script exporté → {output_path}")

def export_models(state, path: str = "trained_models.pkl") -> None:
    if state.models:
        import joblib
        joblib.dump(state.models, path)
        print(f"Modèles exportés → {path}")
    else:
        print("Aucun modèle à exporter.")

def generate_report(state, path: str = "execution_report.html") -> None:
    history_json = json.dumps(state.history, default=str).replace("</", "<\\/")
    meta_json = json.dumps(state.meta if hasattr(state, "meta") else {}, default=str).replace("</", "<\\/")
    script_str = get_python_script_string(state).replace("'", "\\'").replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\r", "")
    
    html = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "<title>ML Pipeline - Full Execution Report</title>",
        "<script src='https://cdn.tailwindcss.com'></script>",
        "<script defer src='https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js'></script>",
        "<!-- PrismJS for code formatting -->",
        "<link href='https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css' rel='stylesheet' />",
        "<script src='https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js'></script>",
        "<script src='https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js'></script>",
        "<script src='https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-json.min.js'></script>",
        "<style>",
        "  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');",
        "  body { font-family: 'Inter', sans-serif; background-color: #f8fafc; color: #0f172a; }",
        "  [x-cloak] { display: none !important; }",
        "  pre { white-space: pre-wrap; word-wrap: break-word; border-radius: 0.5rem !important; }",
        "  .json-key { color: #8b5cf6; } .json-string { color: #10b981; } .json-number { color: #f59e0b; } .json-boolean { color: #ec4899; }",
        "</style>",
        "</head>",
        f"<body class='antialiased min-h-screen flex flex-col' x-data='reportData()'>",
        
        "<!-- Navigation -->",
        "<nav class='bg-white border-b border-slate-200 sticky top-0 z-10'>",
        "  <div class='max-w-7xl mx-auto px-4 sm:px-6 lg:px-8'>",
        "    <div class='flex justify-between h-16'>",
        "      <div class='flex items-center'>",
        "        <span class='text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 to-purple-600 mr-8'>ML Pipeline Report</span>",
        "        <div class='flex space-x-1'>",
        "          <template x-for='t in tabs'>",
        "            <button @click=\"tab = t.id\" :class=\"tab === t.id ? 'bg-indigo-50 text-indigo-700' : 'text-slate-600 hover:bg-slate-50'\" class='px-3 py-2 rounded-md text-sm font-medium transition-colors' x-text='t.label'></button>",
        "          </template>",
        "        </div>",
        "      </div>",
        "    </div>",
        "  </div>",
        "</nav>",
        
        "<!-- Main Content -->",
        "<main class='flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full' x-cloak>",
        
        "  <!-- TAB: Dashboard -->",
        "  <div x-show=\"tab === 'dashboard'\" class='space-y-8'>",
        "    <div class='grid grid-cols-1 md:grid-cols-3 gap-6'>",
    ]
    
    # Business context
    ctx = state.business_context if state.business_context else {}
    target = ctx.get("target_variable", "N/A")
    domain = ctx.get("domain", "N/A")
    total_steps = len(state.history)
    model_count = len(state.models) if state.models else 0

    html.extend([
        f"      <div class='bg-white rounded-2xl p-6 shadow-sm border border-slate-100 flex flex-col justify-center items-center'>",
        f"        <div class='text-sm uppercase tracking-wider text-slate-400 font-bold mb-1'>Target Variable</div>",
        f"        <div class='text-3xl font-bold text-slate-800 text-center'>{target}</div>",
        f"        <div class='text-xs text-slate-400 mt-2 bg-slate-100 px-3 py-1 rounded-full border border-slate-200'>Domain: {domain}</div>",
        f"      </div>",
        f"      <div class='bg-white rounded-2xl p-6 shadow-sm border border-slate-100 flex flex-col justify-center items-center'>",
        f"        <div class='text-sm uppercase tracking-wider text-slate-400 font-bold mb-1'>Trained Models</div>",
        f"        <div class='text-4xl font-black text-indigo-600'>{model_count}</div>",
        f"      </div>",
        f"      <div class='bg-white rounded-2xl p-6 shadow-sm border border-slate-100 flex flex-col justify-center items-center'>",
        f"        <div class='text-sm uppercase tracking-wider text-slate-400 font-bold mb-1'>Pipeline Steps</div>",
        f"        <div class='text-4xl font-black text-emerald-500'>{total_steps}</div>",
        f"      </div>",
        "    </div>",
    ])
    
    # Check for metrics
    metrics_step = None
    for item in state.history:
        if item["step"] == "Evaluation" and "Metrics Computed" in item["action"]:
            metrics_step = item
    
    if metrics_step and "metrics" in metrics_step["details"]:
        metrics_dict = metrics_step["details"]["metrics"]
        html.append("    <div class='mt-10'>")
        html.append("      <h2 class='text-2xl font-bold text-slate-800 mb-6'>Model Performance Overview</h2>")
        html.append("      <div class='grid grid-cols-1 md:grid-cols-2 gap-6'>")
        
        all_metrics_keys = set()
        valid_models = {}
        for m_name, m_vals in metrics_dict.items():
            valid_m = {k: v for k, v in m_vals.items() if isinstance(v, (int, float))}
            valid_models[m_name] = valid_m
            all_metrics_keys.update(valid_m.keys())
            
        colors = ["bg-indigo-500", "bg-emerald-500", "bg-sky-500", "bg-amber-500", "bg-rose-500", "bg-purple-500"]
        
        for idx, metric in enumerate(sorted(all_metrics_keys)):
            html.append(f"        <div class='bg-white rounded-2xl p-6 shadow-sm border border-slate-100' x-data=\"{{ sortAsc: false }}\">")
            html.append(f"          <h3 class='text-lg font-semibold text-slate-700 mb-4 flex justify-between'>{metric}</h3>")
            html.append("          <div class='space-y-4'>")
            
            max_v = max([m_vals[metric] for m_vals in valid_models.values() if metric in m_vals], default=1.0)
            if max_v <= 0: max_v = 1.0
            
            for m_idx, (m_name, m_vals) in enumerate(sorted(valid_models.items(), key=lambda item: item[1].get(metric, 0) if metric in item[1] else 0, reverse=True)):
                if metric in m_vals:
                    val = m_vals[metric]
                    pct = max(0, min(100, (val / max_v) * 100)) if max_v != 0 else 0
                    color = colors[m_idx % len(colors)]
                    html.append("            <div>")
                    html.append(f"              <div class='flex justify-between text-sm mb-1'><span class='font-medium text-slate-600'>{m_name}</span><span class='text-slate-500 font-mono'>{val:.4f}</span></div>")
                    html.append(f"              <div class='w-full bg-slate-100 rounded-full h-2.5 overflow-hidden'><div class='{color} h-2.5 rounded-full' style='width: {pct}%'></div></div>")
                    html.append("            </div>")
            html.append("          </div>")
            html.append("        </div>")
            
        html.append("      </div>")
        html.append("    </div>")
    html.append("  </div>")
    
    # TAB: Data & Schema
    html.append("  <!-- TAB: Data & Schema -->")
    html.append("  <div x-show=\"tab === 'data'\" class='space-y-8 max-w-5xl mx-auto'>")
    html.append("    <div class='flex justify-between items-center mb-6'>")
    html.append("        <h2 class='text-2xl font-bold text-slate-800'>Datasets & Schemas</h2>")
    html.append("        <a href='eda_dashboard.html' target='_blank' class='bg-emerald-50 text-emerald-700 hover:bg-emerald-100 px-4 py-2 rounded-lg font-semibold text-sm transition-colors border border-emerald-200 flex items-center'><svg class='w-4 h-4 mr-2' fill='none' stroke='currentColor' viewBox='0 0 24 24'><path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z'></path></svg> Open Detailed EDA Dashboard</a>")
    html.append("    </div>")
    html.append("    <div x-show='Object.keys(metaData).length === 0' class='text-center py-10 bg-white rounded-2xl border border-slate-200 text-slate-500'>No datasets loaded in the pipeline metadata.</div>")
    html.append("    <template x-for='(cols, dsName) in metaData'>")
    html.append("      <div class='bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden mb-8'>")
    html.append("        <div class='bg-slate-50 px-6 py-4 border-b border-slate-100 flex items-center justify-between'>")
    html.append("          <h3 class='text-lg font-bold text-slate-800 flex items-center'><svg class='w-5 h-5 text-indigo-500 mr-2' fill='none' stroke='currentColor' viewBox='0 0 24 24'><path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4'></path></svg><span x-text=\"dsName\"></span></h3>")
    html.append("          <span class='text-xs font-semibold px-3 py-1 bg-white border border-slate-200 rounded-full text-slate-500' x-text=\"Object.keys(cols).length + ' Columns'\"></span>")
    html.append("        </div>")
    html.append("        <div class='overflow-x-auto'>")
    html.append("          <table class='w-full text-sm text-left text-slate-600'>")
    html.append("            <thead class='text-xs text-slate-500 uppercase bg-slate-50 border-b border-slate-100'>")
    html.append("              <tr><th class='px-6 py-3 font-semibold'>Column Name</th><th class='px-6 py-3 font-semibold'>Type</th><th class='px-6 py-3 font-semibold'>Missing (%)</th><th class='px-6 py-3 font-semibold'>Zeroes (%)</th></tr>")
    html.append("            </thead>")
    html.append("            <tbody>")
    html.append("              <template x-for='(info, col) in cols'>")
    html.append("                <tr class='border-b border-slate-50 hover:bg-slate-50/50 transition-colors'>")
    html.append("                  <td class='px-6 py-3 font-medium text-slate-900 border-r border-slate-50/50' x-text='col'></td>")
    html.append("                  <td class='px-6 py-3'>")
    html.append("                    <span class='px-2 py-1 rounded text-xs font-medium border' :class=\"info.kind === 'cat' ? 'bg-purple-50 text-purple-700 border-purple-200' : (info.kind === 'num' ? 'bg-blue-50 text-blue-700 border-blue-200' : 'bg-slate-100 text-slate-700 border-slate-200')\" x-text='info.kind || info.type'></span>")
    html.append("                  </td>")
    html.append("                  <td class='px-6 py-3'>")
    html.append("                    <div class='flex items-center'><span class='w-10 text-right mr-2' x-text='(info.missing_pct || 0).toFixed(1) + \"%\"'></span><div class='w-24 bg-slate-100 rounded-full h-1.5'><div class='bg-rose-400 h-1.5 rounded-full' :style=\"'width: ' + (info.missing_pct || 0) + '%'\"></div></div></div>")
    html.append("                  </td>")
    html.append("                  <td class='px-6 py-3'>")
    html.append("                    <div class='flex items-center'><span class='w-10 text-right mr-2' x-text='(info.zero_pct || 0).toFixed(1) + \"%\"'></span><div class='w-24 bg-slate-100 rounded-full h-1.5'><div class='bg-amber-400 h-1.5 rounded-full' :style=\"'width: ' + (info.zero_pct || 0) + '%'\"></div></div></div>")
    html.append("                  </td>")
    html.append("                </tr>")
    html.append("              </template>")
    html.append("            </tbody>")
    html.append("          </table>")
    html.append("        </div>")
    html.append("      </div>")
    html.append("    </template>")
    html.append("  </div>")
    
    # TAB: Code Export & Production
    html.append("  <!-- TAB: Python Code -->")
    html.append("  <div x-show=\"tab === 'code'\" class='max-w-5xl mx-auto'>")
    html.append("    <div class='flex justify-between items-center mb-6'>")
    html.append("      <div>")
    html.append("        <h2 class='text-2xl font-bold text-slate-800'>Pipeline Source Code</h2>")
    html.append("        <p class='text-slate-500 text-sm mt-1'>Reproductibilité programmatique avec le package ml_pipeline.</p>")
    html.append("      </div>")
    html.append("      <button @click='copyCode()' class='bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-lg font-medium shadow-sm transition-all focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500'>Copy to Clipboard</button>")
    html.append("    </div>")
    html.append("    <div class='bg-[#1d1f21] rounded-2xl shadow-xl overflow-hidden border border-slate-700/50'>")
    html.append("      <div class='flex items-center px-4 py-3 bg-[#2d2f31] border-b border-[#1d1f21]'>")
    html.append("        <div class='flex space-x-2 mr-4'><div class='w-3 h-3 rounded-full bg-rose-500'></div><div class='w-3 h-3 rounded-full bg-amber-500'></div><div class='w-3 h-3 rounded-full bg-emerald-500'></div></div>")
    html.append("        <span class='text-xs font-mono text-slate-400'>pipeline_export.py</span>")
    html.append("      </div>")
    html.append("      <pre class='p-6 text-sm overflow-auto max-h-[70vh] m-0'><code class='language-python' x-text='pythonCode'></code></pre>")
    html.append("    </div>")
    html.append("    <div class='mt-10 bg-white p-8 rounded-2xl shadow-sm border border-slate-200'>")
    html.append("      <h3 class='text-xl font-bold text-slate-800 mb-4'>Guide d'intégration API / Production</h3>")
    html.append("      <div class='prose prose-slate max-w-none text-sm'>")
    html.append("        <p>Pour intégrer et tester le modèle en mode 'Inférence', suivez ces étapes :</p>")
    html.append("        <ol class='list-decimal pl-5 space-y-2 mt-4 mb-4'>")
    html.append("           <li>Enproduction pure sans le package <code>ml_pipeline</code>, exportez uniquement les objets scikit-learn via <code>joblib.dump</code>.</li>")
    html.append("           <li>Les nouvelles données reçues (via API REST par exemple) doivent subir les mêmes transformations que vues dans le dashboard (Nettoyage, Encodage) avant d'appeler <code>predict()</code>. L'architecture optimale est de packager ces étapes dans un Pipeline scikit-learn ou d'utiliser le script <code>pipeline_export.py</code> comme base backend.</li>")
    html.append("        </ol>")
    html.append("        <pre class='language-python rounded-xl shadow-sm border border-slate-200'><code>import joblib\nimport pandas as pd\n\n# --- EXEMPLE TEST INFERENCE API ---\n\n# 1. Charger les derniers modèles exportés par l'UI\nmodels = joblib.load('trained_models.pkl')\nbest_model_name = list(models.keys())[0]\nbest_model = models[best_model_name]\n\n# 2. Préparer un payload JSON (doit avoir les mêmes colonnes avant Nettoyage/Encodage)\n# NOTE: Remplacez l'appel ci-dessous par la logique Python équivalente aux étapes vues\npayload = pd.DataFrame([{\n    'temperature': 42.5,\n    'category': 'A',\n    'is_active': True\n}])\n\n# => (Insérez ici les transformations, ex: encodage, imputation, vues dans le script exporté)\npayload_transformed = payload # (Placeholder)\n\n# 3. Prédiction \npredictions = best_model.predict(payload_transformed)\nprint(f\"Prédiction par {best_model_name}:\", predictions)</code></pre>")
    html.append("      </div>")
    html.append("    </div>")
    html.append("  </div>")

    # TAB: History List (Enhanced JSON)
    html.append("  <!-- TAB: History List -->")
    html.append("  <div x-show=\"tab === 'history'\" class='space-y-6 max-w-5xl mx-auto'>")
    html.append("    <h2 class='text-2xl font-bold text-slate-800 mb-6'>Execution Log (Raw)</h2>")
    
    html.append("    <template x-for='(step, idx) in steps'>")
    html.append("      <div class='bg-white rounded-xl shadow-sm border border-slate-200/80 mb-6 overflow-hidden'>")
    html.append("        <div class='px-6 py-4 border-b border-slate-100 bg-slate-50 flex flex-col md:flex-row md:items-center justify-between gap-4'>")
    html.append("          <div class='flex items-center space-x-4'>")
    html.append("            <span class='flex items-center justify-center w-10 h-10 rounded-xl bg-indigo-100 text-indigo-700 font-bold text-sm border border-indigo-200/50' x-text='idx + 1'></span>")
    html.append("            <div>")
    html.append("              <h3 class='text-lg font-bold text-slate-800 leading-tight' x-text='step.step'></h3>")
    html.append("              <p class='text-sm text-slate-500 font-medium' x-text='step.action'></p>")
    html.append("            </div>")
    html.append("          </div>")
    html.append("        </div>")
    html.append("        <div class='p-6'>")
    html.append("          <div class='bg-[#1e1e1e] rounded-xl overflow-hidden shadow-inner'>")
    html.append("            <div class='px-4 py-2 bg-[#2d2d2d] flex justify-between items-center'><span class='text-xs font-mono text-slate-400'>JSON Payload</span></div>")
    html.append("            <pre class='p-4 text-xs font-mono m-0 overflow-x-auto text-slate-300'><code class='language-json' x-html='highlightJson(step.details)'></code></pre>")
    html.append("          </div>")
    html.append("        </div>")
    html.append("      </div>")
    html.append("    </template>")
    html.append("  </div>")
    
    # TAB: Slideshow
    html.append("  <!-- TAB: Slideshow -->")
    html.append("  <div x-show=\"tab === 'slideshow'\" class='h-[80vh] flex flex-col items-center justify-center py-2'>")
    html.append("    <div class='w-full max-w-5xl bg-white shadow-xl rounded-2xl overflow-hidden border border-slate-200/80 h-full flex flex-col relative'>")
    
    html.append("      <!-- Slides Content -->")
    html.append("      <div class='flex-1 p-10 flex flex-col justify-center relative overflow-y-auto bg-slate-50/50'>")
    html.append("        <template x-for='(slide, index) in steps' :key='index'>")
    html.append("          <div x-show='currentSlide === index' x-transition.opacity.duration.300ms class='absolute inset-0 p-12 flex flex-col'>")
    html.append("            <div class='text-indigo-600 font-bold tracking-widest text-sm uppercase mb-4 py-1 px-3 bg-indigo-50 border border-indigo-100 rounded-lg inline-flex w-max shadow-sm' x-text=\"'Step ' + (index + 1) + ' / ' + steps.length\"></div>")
    html.append("            <h2 class='text-4xl font-extrabold text-slate-800 mb-3 tracking-tight' x-text='slide.step'></h2>")
    html.append("            <div class='text-2xl font-medium text-slate-500 mb-8 pb-6 border-b border-slate-200' x-text='slide.action'></div>")
    html.append("            <div class='flex-1 min-h-0 overflow-y-auto bg-[#1e1e1e] rounded-xl shadow-inner border border-slate-800'>")
    html.append("               <div class='px-4 py-3 bg-[#2d2d2d] border-b border-[#111111] sticky top-0'><span class='text-xs font-mono text-slate-400'>payload.json</span></div>")
    html.append("               <pre class='p-6 text-sm font-mono text-slate-300 m-0'><code class='language-json' x-html='highlightJson(slide.details)'></code></pre>")
    html.append("            </div>")
    html.append("          </div>")
    html.append("        </template>")
    html.append("        <div x-show='steps.length === 0' class='text-center text-slate-400'>No history steps recorded.</div>")
    html.append("      </div>")
    
    html.append("      <!-- Controls -->")
    html.append("      <div class='bg-white px-8 py-5 flex justify-between items-center border-t border-slate-200'>")
    html.append("        <button @click='prevSlide()' :disabled='currentSlide === 0' class='px-6 py-2.5 rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed bg-white border border-slate-300 text-slate-700 hover:bg-slate-50 focus:ring-2 focus:ring-offset-2 focus:ring-slate-200'>Previous</button>")
    html.append("        <div class='flex space-x-1.5 overflow-x-auto max-w-md px-2'>")
    html.append("          <template x-for='(s, i) in steps' :key='i'>")
    html.append("             <button @click='currentSlide = i; scrollToTop()' :class=\"currentSlide === i ? 'bg-indigo-600 w-8' : 'bg-slate-200 hover:bg-slate-300 w-3'\" class='h-3 rounded-full transition-all duration-300 flex-shrink-0' :title='s.step'></button>")
    html.append("          </template>")
    html.append("        </div>")
    html.append("        <button @click='nextSlide()' :disabled='currentSlide === steps.length - 1' class='px-6 py-2.5 rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed bg-indigo-600 border border-transparent text-white hover:bg-indigo-700 shadow-sm focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500'>Next Step</button>")
    html.append("      </div>")
    html.append("    </div>")
    html.append("  </div>")
    
    html.append("</main>")
    
    # Init PrismJS properly after Alpine render if needed
    html.append("<script>")
    html.append("function reportData() {")
    html.append(f"  const historyData = {history_json};")
    html.append(f"  const metadataJson = {meta_json};")
    html.append(f"  const pythonSrc = \"{script_str}\";")
    html.append("  return {")
    html.append("    tab: 'dashboard',")
    html.append("    tabs: [")
    html.append("      { id: 'dashboard', label: 'Dashboard' },")
    html.append("      { id: 'data', label: 'Data & Schemas' },")
    html.append("      { id: 'code', label: 'Python Source & API' },")
    html.append("      { id: 'history', label: 'Execution Log' },")
    html.append("      { id: 'slideshow', label: 'Slideshow View' }")
    html.append("    ],")
    html.append("    steps: historyData,")
    html.append("    metaData: metadataJson,")
    html.append("    pythonCode: pythonSrc,")
    html.append("    currentSlide: 0,")
    html.append("    init() {")
    html.append("        this.$watch('tab', value => {")
    html.append("            if (value === 'code') setTimeout(() => Prism.highlightAll(), 50);")
    html.append("        });")
    html.append("    },")
    html.append("    nextSlide() { if (this.currentSlide < this.steps.length - 1) this.currentSlide++; this.scrollToTop(); },")
    html.append("    prevSlide() { if (this.currentSlide > 0) this.currentSlide--; this.scrollToTop(); },")
    html.append("    scrollToTop() { window.scrollTo({top:0, behavior: 'smooth'}); },")
    html.append("    copyCode() {")
    html.append("        navigator.clipboard.writeText(this.pythonCode);")
    html.append("        alert('Code copied to clipboard!');")
    html.append("    },")
    html.append("    highlightJson(obj) {")
    html.append("      try {")
    html.append("         const str = JSON.stringify(obj, null, 2);")
    html.append("         if (!window.Prism) return str;")
    html.append("         return Prism.highlight(str, Prism.languages.json, 'json');")
    html.append("      } catch(e) { return String(obj); }")
    html.append("    }")
    html.append("  }")
    html.append("}")
    html.append("</script>")
    
    html.append("</body></html>")
    
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print(f"Rapport interactif généré → {path}")

