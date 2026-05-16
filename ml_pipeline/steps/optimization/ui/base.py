import pathlib
import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from ml_pipeline.styles import styles

from ..logic.operations import (
    is_valid, is_inference_mode, resolve_eval_data,
    serialize_search_space, parse_search_space,
    default_search_space, scoring_for_task
)

_PAL  = ["#6366f1","#10b981","#f59e0b","#ef4444","#3b82f6","#8b5cf6","#ec4899"]
_GRAY = "#64748b"; _BG = "#f8fafc"; _GRID = "#e2e8f0"

def _section(title, color="#6366f1") -> widgets.HTML:
    return widgets.HTML(f"<div style='display:flex;align-items:center;gap:10px;margin:18px 0 10px 0;'>"
                        f"<div style='width:4px;height:20px;background:{color};border-radius:2px;'></div>"
                        f"<span style='font-size:0.95em;font-weight:700;color:#1e293b;'>{title}</span></div>")

def _warn(msg) -> widgets.HTML:
    return widgets.HTML(f"<div style='color:#92400e;background:#fffbeb;border-left:4px solid #f59e0b;padding:8px 12px;font-size:0.85em;border-radius:4px;'>{msg}</div>")

def _info(msg) -> widgets.HTML:
    return widgets.HTML(f"<div style='color:#1e40af;background:#eff6ff;border-left:4px solid #3b82f6;padding:8px 12px;font-size:0.85em;border-radius:4px;'>{msg}</div>")

def _success(msg) -> widgets.HTML:
    return widgets.HTML(f"<div style='color:#065f46;background:#ecfdf5;border-left:4px solid #10b981;padding:8px 12px;font-size:0.85em;border-radius:4px;'>{msg}</div>")

def _metric_card(name, value, color="#6366f1", subtitle="") -> str:
    return (f"<div style='background:#ffffff;border:1px solid #e2e8f0;border-top:3px solid {color};"
            f"border-radius:6px;padding:12px 16px;text-align:center;min-width:130px;'>"
            f"<div style='font-size:0.7em;text-transform:uppercase;letter-spacing:0.08em;color:#94a3b8;margin-bottom:4px;'>{name}</div>"
            f"<div style='font-size:1.4em;font-weight:700;color:#1e293b;'>{value}</div>"
            f"<div style='font-size:0.72em;color:#94a3b8;margin-top:2px;'>{subtitle}</div></div>")


class OptimizationUI:
    """Interface d'optimisation des hyperparamètres et sélection du meilleur modèle."""

    def __init__(self, state):
        self.state       = state
        self.splits      = getattr(state, "data_splits", {})
        self.models      = getattr(state, "models", {})
        self.predictions = getattr(state, "predictions", {})
        self.config      = getattr(state, "config", {})
        if not self.models:
            self.ui = styles.error_msg("Aucun modèle trouvé. Exécutez d'abord la cellule Modeling."); return
        self._build_ui()

    def _detect_task(self) -> tuple[str, str]:
        y = self.splits.get("y_train"); prob, sub = "classification", "binary"
        if is_valid(y):
            n = y.nunique() if hasattr(y, "nunique") else len(np.unique(y))
            if n > 20: prob, sub = "regression", "continuous"
            elif n > 2: prob, sub = "classification", "multiclass"
        return prob, sub

    def _get_train_data(self, model_name):
        pred = self.predictions.get(model_name, {})
        return pred.get("X_train", self.splits.get("X_train")), pred.get("y_train", self.splits.get("y_train"))

    def _build_ui(self) -> None:
        self.task, self.subtask = self._detect_task()
        scoring_default = scoring_for_task(self.task, self.subtask)
        header  = widgets.HTML(styles.card_html("Optimisation & Sélection", "Hyperparameter Tuning — Meilleur Modèle", ""))
        top_bar = widgets.HBox([header], layout=widgets.Layout(align_items="center", margin="0 0 12px 0", padding="0 0 10px 0", border_bottom="2px solid #ede9fe"))
        self.dd_model = widgets.Dropdown(options=list(self.models.keys()), description="Modèle :", style={"description_width": "initial"}, layout=widgets.Layout(width="380px"))
        self.dd_model.observe(self._on_model_change, names="value")
        self.dd_method = widgets.Dropdown(options=[
            ("RandomizedSearchCV (Rapide & Équilibré)", "randomized"), 
            ("GridSearchCV (Exhaustif)", "grid"),
            ("Optuna (Bayesian — Recommandé, très fin)", "optuna")
        ], value="optuna", description="Méthode :", style={"description_width": "initial"}, layout=widgets.Layout(width="420px"))
        self.int_n_iter = widgets.IntSlider(value=30, min=5, max=500, step=5, description="Itérations/Essais :", style={"description_width": "initial"}, layout=widgets.Layout(width="380px"))
        self.int_cv = widgets.IntSlider(value=5, min=2, max=10, step=1, description="CV folds :", style={"description_width": "initial"}, layout=widgets.Layout(width="300px"))
        self.dd_scoring = widgets.Dropdown(options=["roc_auc","f1","f1_macro","f1_weighted","accuracy","precision","recall","r2","neg_mean_absolute_error","neg_mean_squared_error","neg_root_mean_squared_error"], value=scoring_default, description="Scoring :", style={"description_width": "initial"}, layout=widgets.Layout(width="300px"))
        self.chk_refit = widgets.Checkbox(value=True, description="Refit sur toutes les données (X_train complet)", layout=widgets.Layout(width="380px"))
        
        self.view_mode = widgets.RadioButtons(options=["Vue Formulaire", "Vue JSON"], value="Vue Formulaire", layout=widgets.Layout(width="180px"))
        self.search_space_editor = widgets.Textarea(placeholder="{\n    'param_name': [val1, val2, val3]\n}", layout=widgets.Layout(width="100%", height="240px", display="none"))
        self.form_container = widgets.VBox(layout=widgets.Layout(width="100%", border="1px solid #e2e8f0", padding="10px", border_radius="6px"))
        
        # We fill after creating the widgets
        self._fill_search_space(self.dd_model.value)
        
        self.view_mode.observe(self._on_view_mode_change, names="value")
        self.btn_reset_space = widgets.Button(description="Réinitialiser espace", layout=widgets.Layout(width="180px", height="28px"))
        self.btn_reset_space.on_click(lambda _: self._fill_search_space(self.dd_model.value))
        self.chk_compare_all = widgets.Checkbox(value=True, description="Comparer tous les modèles après optimisation", layout=widgets.Layout(width="400px"))
        
        self.btn_run = widgets.Button(description="Lancer l'optimisation", button_style=styles.BTN_PRIMARY, icon="cogs", layout=styles.LAYOUT_BTN_LARGE)
        self.btn_run.on_click(self._run_optimization)
        self.dd_best_manual = widgets.Dropdown(options=list(self.models.keys()), description="Définir comme meilleur :", style={"description_width": "initial"}, layout=widgets.Layout(width="320px"))
        self.btn_set_best = widgets.Button(description="Définir best_model", button_style="warning", layout=widgets.Layout(width="200px"))
        self.btn_set_best.on_click(self._set_best_manual)
        self.output = widgets.Output()
        self.dd_method.observe(lambda c: setattr(self.int_n_iter.layout, "display", "flex" if c["new"] in ["randomized", "optuna"] else "none"), names="value")
        
        editor_box = widgets.VBox([
            widgets.HBox([self.view_mode, self.btn_reset_space], layout=widgets.Layout(justify_content="space-between", align_items="center", margin="0 0 8px 0")),
            self.form_container,
            self.search_space_editor
        ])
        
        self.ui = widgets.VBox([
            top_bar,
            styles.help_box("<b>Optimisation des hyperparamètres</b><br><li><b>Optuna</b> : (Recommandé) Smart, s'adapte à l'espace, évite les tests inutiles.</li><li><b>RandomizedSearchCV</b> : Aléatoire, limite le temps pour l'entraînement.</li><li><b>GridSearchCV</b> : Exhaustif (peut être infiniment long).</li>", "#6366f1"),
            widgets.HTML("<b style='color:#374151;font-size:0.9em;'>Modèle à optimiser</b>"), self.dd_model,
            widgets.HTML("<hr style='border:1px solid #f1f5f9;margin:8px 0;'>"),
            widgets.HTML("<b style='color:#374151;font-size:0.9em;'>Méthode de recherche</b>"),
            widgets.HBox([self.dd_method, self.int_n_iter], layout=widgets.Layout(gap="16px", align_items="center")),
            widgets.HBox([self.int_cv, self.dd_scoring], layout=widgets.Layout(gap="16px", align_items="center", margin="6px 0")),
            self.chk_refit,
            widgets.HTML("<hr style='border:1px solid #f1f5f9;margin:8px 0;'>"),
            widgets.HTML("<b style='color:#374151;font-size:0.9em;'>Espace de recherche</b>"),
            editor_box,
            widgets.HTML("<hr style='border:1px solid #f1f5f9;margin:8px 0;'>"),
            self.chk_compare_all, self.btn_run,
            widgets.HTML("<hr style='border:1px solid #f1f5f9;margin:12px 0;'>"),
            widgets.HTML("<b style='color:#374151;font-size:0.9em;'>Sélection manuelle du meilleur modèle</b>"),
            widgets.HBox([self.dd_best_manual, self.btn_set_best], layout=widgets.Layout(gap="12px", align_items="center")),
            self.output,
        ], layout=widgets.Layout(width="100%", max_width="1000px", border="1px solid #e5e7eb", padding="18px", border_radius="10px", background_color="#ffffff"))

    def _on_view_mode_change(self, change):
        is_form = change["new"] == "Vue Formulaire"
        self.form_container.layout.display = "flex" if is_form else "none"
        self.search_space_editor.layout.display = "none" if is_form else "flex"
        # If switching back to form, try to read from JSON
        if is_form:
            try:
                space = parse_search_space(self.search_space_editor.value)
                self._build_form(space)
            except Exception as e:
                self.form_container.children = [widgets.HTML(f"<div style='color:#ef4444;'>JSON invalide. Corrigez-le avant de revenir à la vue Formulaire.<br><small>{e}</small></div>")]

    def _on_model_change(self, change) -> None:
        if change["new"]: self._fill_search_space(change["new"])

    def _fill_search_space(self, model_name: str) -> None:
        model = self.models.get(model_name)
        if model is None: return
        
        space = self.config.get("optimization", {}).get("search_spaces", {}).get(model.__class__.__name__)
        if not space:
            space = default_search_space(model.__class__.__name__)
            
        if space:
            self.search_space_editor.value = serialize_search_space(space)
            self._build_form(space)
        else:
            self.search_space_editor.value = "{\n}"
            self.form_container.children = [widgets.HTML("<div style='color:#64748b;'>Aucun espace prédéfini.</div>")]

    def _build_form(self, space_dict):
        rows = []
        import json
        for k, v in space_dict.items():
            lbl = widgets.HTML(f"<b style='min-width:160px;display:inline-block;color:#334155;'>{k}</b>")
            
            if isinstance(v, dict) and v.get("dist") == "randint":
                w_type = widgets.HTML("<span style='color:#10b981;min-width:80px;display:inline-block;font-size:0.85em;background:#d1fae5;padding:2px 8px;border-radius:12px;text-align:center;'>Entier</span>")
                w_min = widgets.IntText(value=v.get("low", 0), description="min:", layout=widgets.Layout(width="110px"), style={"description_width": "30px"})
                w_max = widgets.IntText(value=v.get("high", 10), description="max:", layout=widgets.Layout(width="110px"), style={"description_width": "35px"})
                
                def on_change(change, key=k, wmin=w_min, wmax=w_max):
                    self._update_json_from_form_range(key, "randint", wmin.value, wmax.value)
                    
                w_min.observe(on_change, names="value"); w_max.observe(on_change, names="value")
                row = widgets.HBox([lbl, w_type, w_min, w_max], layout=widgets.Layout(margin="4px 0", align_items="center", gap="10px"))
            
            elif isinstance(v, dict) and v.get("dist") == "uniform":
                w_type = widgets.HTML("<span style='color:#3b82f6;min-width:80px;display:inline-block;font-size:0.85em;background:#dbeafe;padding:2px 8px;border-radius:12px;text-align:center;'>Décimal</span>")
                w_min = widgets.FloatText(value=v.get("loc", 0.0), description="min:", layout=widgets.Layout(width="110px"), style={"description_width": "30px"})
                w_max = widgets.FloatText(value=v.get("loc", 0.0) + v.get("scale", 1.0), description="max:", layout=widgets.Layout(width="110px"), style={"description_width": "35px"})
                
                def on_change(change, key=k, wmin=w_min, wmax=w_max):
                    self._update_json_from_form_range(key, "uniform", wmin.value, wmax.value)
                    
                w_min.observe(on_change, names="value"); w_max.observe(on_change, names="value")
                row = widgets.HBox([lbl, w_type, w_min, w_max], layout=widgets.Layout(margin="4px 0", align_items="center", gap="10px"))
            
            else:
                w_type = widgets.HTML("<span style='color:#f59e0b;min-width:80px;display:inline-block;font-size:0.85em;background:#fef3c7;padding:2px 8px;border-radius:12px;text-align:center;'>Liste/Fixe</span>")
                txt_val = json.dumps(v)
                w_list = widgets.Text(value=txt_val, layout=widgets.Layout(width="300px"))
                
                def on_change(change, key=k, wlist=w_list):
                    self._update_json_from_form_list(key, wlist.value)
                    
                w_list.observe(on_change, names="value")
                row = widgets.HBox([lbl, w_type, w_list], layout=widgets.Layout(margin="4px 0", align_items="center", gap="10px"))
                
            rows.append(row)
        if not rows:
            rows = [widgets.HTML("<div style='color:#64748b;'>Aucun paramètre configuré. Utilisez l'éditeur JSON pour ajouter des clés.</div>")]
        self.form_container.children = rows

    def _update_json_from_form_range(self, key, dist, val_min, val_max):
        try:
            space = parse_search_space(self.search_space_editor.value)
            if dist == "randint": space[key] = {"dist": "randint", "low": val_min, "high": val_max}
            else: space[key] = {"dist": "uniform", "loc": val_min, "scale": val_max - val_min}
            self.search_space_editor.value = serialize_search_space(space)
        except: pass

    def _update_json_from_form_list(self, key, text_val):
        try:
            import json
            txt = text_val.strip()
            if not txt: return
            txt = txt.replace("'", '"')
            parsed_list = json.loads(txt)
            space = parse_search_space(self.search_space_editor.value)
            space[key] = parsed_list
            self.search_space_editor.value = serialize_search_space(space)
        except: pass

    def _run_optimization(self, b) -> None:
        with self.output:
            clear_output(wait=True)
            model_name = self.dd_model.value; model = self.models.get(model_name)
            if model is None: display(_warn(f"Modèle '{model_name}' introuvable.")); return
            X_tr, y_tr = self._get_train_data(model_name)
            if not is_valid(X_tr) or not is_valid(y_tr): display(_warn("X_train / y_train introuvables.")); return
            try: param_dist = parse_search_space(self.search_space_editor.value)
            except ValueError as e: display(_warn(str(e))); return
            if not param_dist: display(_warn("L'espace de recherche est vide.")); return
            method = self.dd_method.value; n_iter = self.int_n_iter.value
            cv_fold = self.int_cv.value; scoring = self.dd_scoring.value
            display(_info(f"Lancement <b>{method}</b> sur <b>{model_name}</b> | scoring={scoring} | CV={cv_fold} folds"))
            from sklearn.model_selection import StratifiedKFold, KFold, RandomizedSearchCV, GridSearchCV
            from sklearn.base import clone
            import scipy.stats
            
            def to_scipy_space(json_dist):
                res = {}
                for k, v in json_dist.items():
                    if isinstance(v, dict) and "dist" in v:
                        if v["dist"] == "randint": res[k] = scipy.stats.randint(v.get("low", 0), v.get("high", 100))
                        elif v["dist"] == "uniform": res[k] = scipy.stats.uniform(v.get("loc", 0.0), v.get("scale", 1.0))
                        else: res[k] = v
                    else:
                        res[k] = v
                return res

            cv_splitter = (StratifiedKFold(n_splits=cv_fold, shuffle=True, random_state=42)
                           if self.task == "classification"
                           else KFold(n_splits=cv_fold, shuffle=True, random_state=42))
            base_model = clone(model)
            try:
                if method == "optuna":
                    try:
                        import optuna
                    except ImportError:
                        display(_warn("Le package 'optuna' n'est pas installé. Veuillez l'installer via l'étape S00 ou utiliser RandomizedSearchCV."))
                        return
                    from sklearn.model_selection import cross_val_score
                    optuna.logging.set_verbosity(optuna.logging.WARNING)

                    def objective(trial):
                        params = {}
                        for k, v in param_dist.items():
                            if isinstance(v, dict) and v.get("dist") == "randint":
                                params[k] = trial.suggest_int(k, v.get("low", 0), v.get("high", 100) - 1)
                            elif isinstance(v, dict) and v.get("dist") == "uniform":
                                loc = v.get("loc", 0.0)
                                scale = v.get("scale", 1.0)
                                params[k] = trial.suggest_float(k, loc, loc + scale)
                            elif isinstance(v, list):
                                params[k] = trial.suggest_categorical(k, v)
                            else:
                                params[k] = v
                        cloned = clone(base_model).set_params(**params)
                        # On Optuna On cherche à maximiser le score retourné par cross_val_score.
                        # cross_val_score gère neg_mean_squared_error pour que "plus c'est élevé, mieux c'est".
                        score = cross_val_score(cloned, X_tr, y_tr, cv=cv_splitter, scoring=scoring, n_jobs=-1)
                        if np.isnan(score.mean()):
                            raise optuna.TrialPruned()
                        return score.mean()

                    study = optuna.create_study(direction="maximize")
                    study.optimize(objective, n_trials=n_iter, n_jobs=1, show_progress_bar=False)
                    best_score = study.best_value
                    best_params = study.best_params
                    best_model = clone(base_model).set_params(**best_params)
                    # Optuna doesn't give us searcher, so we make a dummy one for the plot
                    class DummySearcher:
                        pass
                    searcher = DummySearcher()
                    
                    df_trials = study.trials_dataframe()
                    if "value" in df_trials.columns:
                        searcher.cv_results_ = {
                            "mean_test_score": df_trials["value"].values,
                            "std_test_score": np.zeros(len(df_trials)),
                            "rank_test_score": df_trials["value"].rank(ascending=False, method="min").values.astype(int)
                        }
                    else:
                        searcher.cv_results_ = {}
                        
                elif method == "randomized":
                    scipy_dist = to_scipy_space(param_dist)
                    searcher = RandomizedSearchCV(base_model, scipy_dist, n_iter=n_iter, scoring=scoring, cv=cv_splitter, n_jobs=-1, verbose=0, random_state=42, refit=True)
                    searcher.fit(X_tr, y_tr)
                    best_score = searcher.best_score_; best_params = searcher.best_params_; best_model = searcher.best_estimator_
                else:
                    # GridSearchCV expects only lists
                    grid_dist = {k: v for k, v in param_dist.items() if isinstance(v, list)}
                    searcher = GridSearchCV(base_model, grid_dist, scoring=scoring, cv=cv_splitter, n_jobs=-1, verbose=0, refit=True)
                    searcher.fit(X_tr, y_tr)
                    best_score = searcher.best_score_; best_params = searcher.best_params_; best_model = searcher.best_estimator_
            except Exception as e: display(_warn(f"Erreur durant la recherche : {e}")); return
            
            display(_section(f"Résultats — {model_name}", "#6366f1"))
            orig_score = self._cv_score_current(model, X_tr, y_tr, scoring, cv_splitter)
            gain = best_score - orig_score
            cards = (_metric_card("Score CV original", f"{orig_score:.4f}", _PAL[2], scoring) +
                     _metric_card("Score CV optimisé", f"{best_score:.4f}", _PAL[1], scoring) +
                     _metric_card("Gain", f"{'+' if gain >= 0 else ''}{gain:.4f}", "#10b981" if gain >= 0 else "#ef4444", "optimisé − original"))
            display(HTML(f"<div style='display:flex;flex-wrap:wrap;gap:10px;margin:10px 0;'>{cards}</div>"))
            params_html = "".join(f"<div style='margin:2px 0;font-size:0.83em;'><span style='color:#6d28d9;font-weight:600;'>{k}</span> = {v}</div>" for k, v in best_params.items())
            display(HTML(f"<b style='font-size:0.85em;color:#374151;'>Meilleurs paramètres :</b><div style='background:#f8fafc;padding:10px;border-radius:6px;border:1px solid #e2e8f0;'>{params_html}</div>"))
            if self.chk_refit.value:
                display(_info("Refit sur X_train complet…"))
                try: best_model.fit(X_tr, y_tr); display(_success("Refit terminé."))
                except Exception as e: display(_warn(f"Erreur refit : {e}"))
            optimized_name = f"{model_name} (optimisé)"
            self.state.models[optimized_name] = best_model
            self.state.best_model = best_model; self.state.best_model_name = optimized_name
            self.dd_best_manual.options = list(self.state.models.keys()); self.dd_best_manual.value = optimized_name
            Xv, yv = resolve_eval_data(self.splits, self.predictions, model_name)
            if is_valid(Xv) and is_valid(yv):
                try:
                    pred_entry = self.predictions.get(model_name, {}).copy()
                    pred_entry.update({"model": optimized_name, "y_pred_val": best_model.predict(Xv), "feature_columns": list(X_tr.columns)})
                    self.state.predictions[optimized_name] = pred_entry
                except Exception: pass
            import joblib
            models_dir = pathlib.Path("models"); models_dir.mkdir(exist_ok=True)
            safe = optimized_name.replace(" ","_").replace("/","_").replace("(","").replace(")","")
            path = models_dir / f"{safe}.pkl"; joblib.dump(best_model, path)
            display(_success(f"Modèle optimisé sauvegardé → <b>{path}</b><br>Stocké dans <code>state.models['{optimized_name}']</code> et <code>state.best_model</code>."))
            self.state.log_step("Optimization", "Hyperparameter Tuning", {"model": model_name, "method": method, "best_score": best_score, "best_params": str(best_params)})
            self._plot_search_results(searcher, model_name, scoring)
            if self.chk_compare_all.value: self._compare_all_models(X_tr, y_tr, scoring, cv_splitter)

    def _cv_score_current(self, model, X_tr, y_tr, scoring, cv) -> float:
        from sklearn.model_selection import cross_val_score
        from sklearn.base import clone
        try: return cross_val_score(clone(model), X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=-1).mean()
        except Exception: return 0.0

    def _plot_search_results(self, searcher, model_name: str, scoring: str) -> None:
        try:
            results = searcher.cv_results_; means = results["mean_test_score"]; stds = results["std_test_score"]; ranks = results["rank_test_score"]
            sorted_idx = np.argsort(means)[::-1]; top_n = min(20, len(means)); idx = sorted_idx[:top_n]
            fig, ax = plt.subplots(figsize=(10, 4)); fig.patch.set_facecolor(_BG); ax.set_facecolor(_BG)
            ax.grid(color=_GRID, linewidth=0.8, zorder=0); ax.spines[["top","right"]].set_visible(False); ax.spines[["left","bottom"]].set_color(_GRID)
            colors = [_PAL[1] if i == 0 else _PAL[0] for i in range(top_n)]
            bars = ax.barh(range(top_n), means[idx][::-1], xerr=stds[idx][::-1], color=colors[::-1], edgecolor="white", height=0.6, capsize=4)
            ax.set_yticks(range(top_n)); ax.set_yticklabels([f"Config #{ranks[i]}" for i in idx[::-1]], fontsize=8)
            ax.set_xlabel(scoring, color=_GRAY); ax.set_title(f"Top {top_n} configurations — {model_name}", fontsize=11, fontweight="bold", color="#1e293b")
            for bar, mean in zip(bars, means[idx][::-1]):
                ax.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2, f"{mean:.4f}", va="center", fontsize=8, color="#1e293b")
            plt.tight_layout()
            display(fig)
            plt.close(fig)
        except Exception: pass

    def _compare_all_models(self, X_tr, y_tr, scoring: str, cv) -> None:
        from sklearn.model_selection import cross_val_score
        from sklearn.base import clone
        display(_section("Comparaison globale de tous les modèles", "#ec4899"))
        display(_info("Calcul des scores CV en cours…"))
        scores_dict = {}
        for name, model in self.state.models.items():
            try: scores_dict[name] = cross_val_score(clone(model), X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=-1)
            except Exception as e: scores_dict[name] = None; display(_warn(f"{name} : erreur CV — {e}"))
        rows = [{"Modèle": name, f"{scoring} mean": round(sc.mean(),4), f"{scoring} std": round(sc.std(),4), "Min": round(sc.min(),4), "Max": round(sc.max(),4)} for name, sc in scores_dict.items() if sc is not None]
        if rows:
            df = pd.DataFrame(rows).set_index("Modèle").sort_values(f"{scoring} mean", ascending=False)
            best_row = df.index[0]
            display(df.style.highlight_max(subset=[f"{scoring} mean"], color="#d1fae5").set_properties(**{"font-size": "0.85em"}).format(precision=4))
            if best_row in self.state.models:
                self.state.best_model = self.state.models[best_row]; self.state.best_model_name = best_row
                self.dd_best_manual.options = list(self.state.models.keys()); self.dd_best_manual.value = best_row
                display(_success(f"Meilleur modèle sélectionné automatiquement : <b>{best_row}</b> ({scoring}={df.loc[best_row, f'{scoring} mean']:.4f})"))

    def _set_best_manual(self, b) -> None:
        with self.output:
            clear_output(wait=True)
            chosen = self.dd_best_manual.value
            if chosen not in self.state.models: display(_warn(f"'{chosen}' introuvable dans state.models.")); return
            self.state.best_model = self.state.models[chosen]; self.state.best_model_name = chosen
            display(_success(f"<code>state.best_model</code> défini sur <b>{chosen}</b>."))
            self.state.log_step("Optimization", "Best Model Set Manually", {"model": chosen})
