import pandas as pd
import numpy as np

def is_valid(arr) -> bool:
    if arr is None: return False
    if isinstance(arr, pd.DataFrame) and arr.empty: return False
    if isinstance(arr, pd.Series)    and arr.empty: return False
    if isinstance(arr, np.ndarray)   and arr.size == 0: return False
    return True

def is_inference_mode(splits: dict) -> bool:
    return not is_valid(splits.get("y_test")) and is_valid(splits.get("X_test"))

def resolve_eval_data(splits, predictions, model_name):
    pred = (predictions or {}).get(model_name, {})
    Xv = pred.get("X_val"); yv = pred.get("y_val")
    if not is_valid(Xv) or not is_valid(yv):
        Xv = splits.get("X_test"); yv = splits.get("y_test")
    return Xv, yv

def serialize_search_space(space: dict) -> str:
    lines = ["{\n"]
    for k, v in space.items():
        cls = type(v).__name__
        if hasattr(v, "a") and hasattr(v, "b") and cls in ("randint_frozen", "randint"):
            serialized = f"randint({v.a}, {v.b})"
        elif hasattr(v, "args") and cls in ("uniform_frozen", "uniform"):
            loc, scale = v.args if v.args else (v.kwds.get("loc", 0), v.kwds.get("scale", 1))
            serialized = f"uniform({loc}, {scale})"
        elif hasattr(v, "dist") and hasattr(v, "args"):
            serialized = f"{v.dist.name}({', '.join(repr(a) for a in v.args)})"
        else:
            serialized = repr(v)
        lines.append(f"    '{k}': {serialized},\n")
    lines.append("}")
    return "".join(lines)

def parse_search_space(code: str) -> dict:
    from scipy.stats import randint, uniform
    code = code.strip()
    if not code or all(l.strip().startswith("#") for l in code.splitlines() if l.strip()):
        return {}
    try:
        ns = {"randint": randint, "uniform": uniform, "np": np, "None": None, "True": True, "False": False}
        result = eval(code, {"__builtins__": {}}, ns)
        if not isinstance(result, dict):
            raise ValueError("L'espace de recherche doit être un dict Python { ... }")
        return result
    except SyntaxError as e:
        raise ValueError(f"Syntaxe invalide à la ligne {e.lineno} : {e.msg}")
    except Exception as e:
        raise ValueError(f"Erreur d'évaluation : {e}")

def default_search_space(model_class_name: str) -> dict:
    from scipy.stats import randint, uniform
    spaces = {
        "LogisticRegression":          {"C": uniform(0.01, 10), "solver": ["lbfgs","saga"], "max_iter": [500,1000,2000]},
        "RandomForestClassifier":      {"n_estimators": randint(50,400), "max_depth": [None,5,10,15,20], "min_samples_split": randint(2,15), "min_samples_leaf": randint(1,8), "max_features": ["sqrt","log2",0.5]},
        "RandomForestRegressor":       {"n_estimators": randint(50,400), "max_depth": [None,5,10,15,20], "min_samples_split": randint(2,15), "min_samples_leaf": randint(1,8)},
        "GradientBoostingClassifier":  {"n_estimators": randint(50,300), "learning_rate": uniform(0.01,0.3), "max_depth": randint(2,8), "subsample": uniform(0.6,0.4)},
        "GradientBoostingRegressor":   {"n_estimators": randint(50,300), "learning_rate": uniform(0.01,0.3), "max_depth": randint(2,8)},
        "XGBClassifier":               {"n_estimators": randint(50,400), "learning_rate": uniform(0.01,0.3), "max_depth": randint(3,10), "subsample": uniform(0.5,0.5), "colsample_bytree": uniform(0.5,0.5)},
        "XGBRegressor":                {"n_estimators": randint(50,400), "learning_rate": uniform(0.01,0.3), "max_depth": randint(3,10)},
        "LGBMClassifier":              {"n_estimators": randint(50,400), "learning_rate": uniform(0.01,0.3), "num_leaves": randint(15,128), "subsample": uniform(0.5,0.5)},
        "LGBMRegressor":               {"n_estimators": randint(50,400), "learning_rate": uniform(0.01,0.3), "num_leaves": randint(15,128)},
        "CatBoostClassifier":          {"iterations": randint(50,400), "learning_rate": uniform(0.01,0.3), "depth": randint(3,10)},
        "Ridge":                       {"alpha": uniform(0.001,100), "solver": ["auto","svd","cholesky","lsqr"]},
        "Lasso":                       {"alpha": uniform(0.001,10), "max_iter": [1000,2000,5000]},
        "SVC":                         {"C": uniform(0.01,100), "kernel": ["rbf","poly","sigmoid"], "gamma": ["scale","auto"]},
    }
    return spaces.get(model_class_name, {})

def scoring_for_task(task: str, subtask: str) -> str:
    if task == "classification":
        return "roc_auc" if subtask == "binary" else "f1_macro"
    return "r2"
