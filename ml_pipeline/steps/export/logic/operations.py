import json
import os

def export_python_script(state, output_path: str = "exported_pipeline.py") -> None:
    script = [
        "\"\"\"",
        "Auto-generated Scikit-Learn Pipeline Script",
        "",
        "Note méthodologique importante :",
        "Dans l'interface interactive (notebook), un 'raccourci' visuel est utilisé où le",
        "nettoyage et l'encodage sont parfois appliqués au dataset complet avant séparation (Data Leakage),",
        "ou appliqués via un double encodage sur Train/Test séparés. De plus, l'étape s07 arrive après ",
        "et sert principalement à gérer le déséquilibre de classe (Balancing).",
        "",
        "Pour aller plus loin et packager le code de façon robuste, ce script génère des ",
        "Sklearn Pipelines qui respectent la méthodologie standard :",
        "1. L'apprentissage (.fit()) se fait strictement sur le Train (dictionnaires, stats, etc.).",
        "2. La projection (.transform()) s'applique ensuite fidèlement sur le Test.",
        "\"\"\"",
        "import pandas as pd",
        "import numpy as np",
        "import joblib",
        "from sklearn.model_selection import train_test_split",
        "from sklearn.pipeline import Pipeline",
        "from sklearn.compose import ColumnTransformer",
        ""
    ]
    
    # Imports dynamiques selon configuration
    imports_needed = set()
    estimators = {}
    
    try:
        from ml_pipeline.steps.encoder.logic.operations import build_sklearn_pipeline_code
        # Try to find encoding step
        for h in state.history:
            if h["step"] == "Data Encoding" and "Tabular Encoded" in h["action"]:
                if "params" in h["details"]:
                    enc_code = build_sklearn_pipeline_code(h["details"]["params"], state.config)
                    if "ColumnTransformer" in enc_code:
                        script.append("# --- Transformation Pipeline ---")
                        script.append(enc_code)
                        script.append("pipeline = Pipeline(steps=[('preprocessor', preprocessor)])\n")
    except:
        pass
        
    for item in state.history:
        script.append(f"# --- Step: {item['step']} | Action: {item['action']} ---")
        script.append(f"# {json.dumps(item['details'], default=str)}")
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
    history_json = json.dumps(state.history, default=str).replace("</", "<\\/")
    
    html = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "<title>ML Pipeline - Execution Report</title>",
        "<script src='https://cdn.tailwindcss.com'></script>",
        "<script defer src='https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js'></script>",
        "<style>",
        "  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');",
        "  body { font-family: 'Inter', sans-serif; background-color: #f8fafc; color: #0f172a; }",
        "  [x-cloak] { display: none !important; }",
        "  pre { white-space: pre-wrap; word-wrap: break-word; }",
        "</style>",
        "</head>",
        f"<body class='antialiased min-h-screen flex flex-col' x-data='reportData()'>",
        
        "<!-- Navigation -->",
        "<nav class='bg-white border-b border-slate-200 sticky top-0 z-10'>",
        "  <div class='max-w-7xl mx-auto px-4 sm:px-6 lg:px-8'>",
        "    <div class='flex justify-between h-16'>",
        "      <div class='flex'>",
        "        <div class='flex-shrink-0 flex items-center'>",
        "          <span class='text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 to-purple-600'>ML Pipeline</span>",
        "        </div>",
        "        <div class='ml-8 flex space-x-8'>",
        "          <button @click=\"tab = 'dashboard'\" :class=\"tab === 'dashboard' ? 'border-indigo-600 text-slate-900' : 'border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300'\" class='inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium transition-colors duration-200'>Dashboard</button>",
        "          <button @click=\"tab = 'history'\" :class=\"tab === 'history' ? 'border-indigo-600 text-slate-900' : 'border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300'\" class='inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium transition-colors duration-200'>Execution List</button>",
        "          <button @click=\"tab = 'slideshow'\" :class=\"tab === 'slideshow' ? 'border-indigo-600 text-slate-900' : 'border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300'\" class='inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium transition-colors duration-200'>Slideshow View</button>",
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
        f"        <div class='text-xs text-slate-400 mt-2 bg-slate-100 px-3 py-1 rounded-full'>Domain: {domain}</div>",
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
        html.append("    <h2 class='text-2xl font-bold text-slate-800 mt-10 mb-6'>Model Performance (Bar Charts)</h2>")
        html.append("    <div class='grid grid-cols-1 md:grid-cols-2 gap-6'>")
        
        # We will parse the metrics out and build a dynamic visual for it.
        # Find all available metrics
        all_metrics_keys = set()
        valid_models = {}
        for m_name, m_vals in metrics_dict.items():
            valid_m = {k: v for k, v in m_vals.items() if isinstance(v, (int, float))}
            valid_models[m_name] = valid_m
            all_metrics_keys.update(valid_m.keys())
            
        colors = ["bg-indigo-500", "bg-emerald-500", "bg-sky-500", "bg-amber-500", "bg-rose-500", "bg-purple-500"]
        
        for idx, metric in enumerate(sorted(all_metrics_keys)):
            html.append(f"      <div class='bg-white rounded-2xl p-6 shadow-sm border border-slate-100'>")
            html.append(f"        <h3 class='text-lg font-semibold text-slate-700 mb-4'>{metric}</h3>")
            html.append("        <div class='space-y-4'>")
            
            # find max to scale bars
            max_v = max([m_vals[metric] for m_vals in valid_models.values() if metric in m_vals], default=1.0)
            if max_v <= 0: max_v = 1.0
            
            for m_idx, (m_name, m_vals) in enumerate(valid_models.items()):
                if metric in m_vals:
                    val = m_vals[metric]
                    pct = max(0, min(100, (val / max_v) * 100)) if max_v != 0 else 0
                    color = colors[m_idx % len(colors)]
                    html.append("          <div>")
                    html.append(f"            <div class='flex justify-between text-sm mb-1'><span class='font-medium text-slate-600'>{m_name}</span><span class='text-slate-500'>{val:.4f}</span></div>")
                    html.append(f"            <div class='w-full bg-slate-100 rounded-full h-2.5'><div class='{color} h-2.5 rounded-full' style='width: {pct}%'></div></div>")
                    html.append("          </div>")
            html.append("        </div>")
            html.append("      </div>")
            
        html.append("    </div>")

    html.append("  </div>")
    
    # TAB: History
    html.append("  <!-- TAB: History List -->")
    html.append("  <div x-show=\"tab === 'history'\" class='space-y-6 max-w-4xl mx-auto'>")
    html.append("    <h2 class='text-2xl font-bold text-slate-800 mb-6'>Full Execution Log</h2>")
    
    for idx, step in enumerate(state.history):
        step_name = step.get('step', 'Unknown Step')
        action = step.get('action', '')
        details_str = json.dumps(step.get('details', {}), default=str, indent=2)
        html.append(f"    <div class='bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden'>")
        html.append(f"      <div class='px-6 py-4 border-b border-slate-100 bg-slate-50 flex justify-between items-center'>")
        html.append(f"        <div class='flex items-center space-x-3'>")
        html.append(f"          <span class='flex items-center justify-center w-8 h-8 rounded-full bg-indigo-100 text-indigo-700 font-bold text-sm'>{idx+1}</span>")
        html.append(f"          <h3 class='text-lg font-semibold text-slate-800'>{step_name}</h3>")
        html.append(f"        </div>")
        html.append(f"        <span class='text-sm font-medium text-slate-500 bg-white px-3 py-1 rounded-full border border-slate-200'>{action}</span>")
        html.append(f"      </div>")
        html.append(f"      <div class='p-6'>")
        html.append(f"        <pre class='text-xs text-slate-600 bg-slate-50 p-4 rounded-lg overflow-x-auto border border-slate-100'>{details_str}</pre>")
        html.append(f"      </div>")
        html.append(f"    </div>")
    html.append("  </div>")
    
    # TAB: Slideshow
    html.append("  <!-- TAB: Slideshow -->")
    html.append("  <div x-show=\"tab === 'slideshow'\" class='h-full flex flex-col items-center justify-center py-10'>")
    html.append("    <div class='w-full max-w-4xl bg-white shadow-xl rounded-2xl overflow-hidden border border-slate-200 aspect-video flex flex-col relative'>")
    
    html.append("      <!-- Slides Content -->")
    html.append("      <div class='flex-1 p-10 flex flex-col justify-center relative overflow-y-auto'>")
    html.append("        <template x-for='(slide, index) in steps' :key='index'>")
    html.append("          <div x-show='currentSlide === index' x-transition.opacity.duration.300ms class='absolute inset-0 p-12 flex flex-col'>")
    html.append("            <div class='text-indigo-500 font-bold tracking-widest text-sm uppercase mb-4' x-text=\"'Step ' + (index + 1) + ' of ' + steps.length\"></div>")
    html.append("            <h2 class='text-4xl font-black text-slate-800 mb-2' x-text='slide.step'></h2>")
    html.append("            <div class='text-2xl font-medium text-slate-500 mb-8 pb-6 border-b border-slate-100' x-text='slide.action'></div>")
    html.append("            <div class='flex-1 min-h-0 overflow-y-auto'>")
    html.append("               <pre class='text-sm bg-slate-50 p-6 rounded-xl border border-slate-100 text-slate-700 h-full' x-html='formatJson(slide.details)'></pre>")
    html.append("            </div>")
    html.append("          </div>")
    html.append("        </template>")
    html.append("        <div x-show='steps.length === 0' class='text-center text-slate-400'>No history steps recorded.</div>")
    html.append("      </div>")
    
    html.append("      <!-- Controls -->")
    html.append("      <div class='bg-slate-50 px-8 py-4 flex justify-between items-center border-t border-slate-100'>")
    html.append("        <button @click='prevSlide()' :disabled='currentSlide === 0' class='px-6 py-2 rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed bg-white border border-slate-300 text-slate-700 hover:bg-slate-50 hover:text-indigo-600'>Previous</button>")
    html.append("        <div class='flex space-x-1'>")
    html.append("          <template x-for='(s, i) in steps' :key='i'>")
    html.append("             <button @click='currentSlide = i' :class=\"currentSlide === i ? 'bg-indigo-600 w-6' : 'bg-slate-300 hover:bg-slate-400 w-2'\" class='h-2 rounded-full transition-all duration-300'></button>")
    html.append("          </template>")
    html.append("        </div>")
    html.append("        <button @click='nextSlide()' :disabled='currentSlide === steps.length - 1' class='px-6 py-2 rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed bg-indigo-600 border border-transparent text-white hover:bg-indigo-700 shadow-sm'>Next</button>")
    html.append("      </div>")
    html.append("    </div>")
    html.append("  </div>")
    
    html.append("</main>")
    
    # Alpine Data
    html.append("<script>")
    html.append("function reportData() {")
    html.append(f"  const historyData = {history_json};")
    html.append("  return {")
    html.append("    tab: 'dashboard',")
    html.append("    steps: historyData,")
    html.append("    currentSlide: 0,")
    html.append("    nextSlide() { if (this.currentSlide < this.steps.length - 1) this.currentSlide++; },")
    html.append("    prevSlide() { if (this.currentSlide > 0) this.currentSlide--; },")
    html.append("    formatJson(obj) {")
    html.append("      try { return JSON.stringify(obj, null, 2); } catch(e) { return String(obj); }")
    html.append("    }")
    html.append("  }")
    html.append("}")
    html.append("</script>")
    
    html.append("</body></html>")
    
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print(f"Rapport interactif avec Dashboard & Diaporama exporté → {path}")

