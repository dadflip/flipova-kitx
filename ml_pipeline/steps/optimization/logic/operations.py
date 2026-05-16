import pandas as pd
import numpy as np
import json

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
    return json.dumps(space, indent=4)

def parse_search_space(code: str) -> dict:
    from scipy.stats import randint, uniform
    code = code.strip()
    if not code:
        return {}
        
    try:
        # User may put python-like none/null, trying to json load
        # Let's clean up a bit if they used python dict string by mistake
        code = code.replace("'", '"')
        code = code.replace("None", "null")
        code = code.replace("True", "true")
        code = code.replace("False", "false")
        
        # Remove comments
        lines = [l for l in code.splitlines() if not l.strip().startswith("#")]
        code = "\n".join(lines)
        
        result = json.loads(code)
        
        def replace_nulls(obj):
            if isinstance(obj, dict):
                return {k: replace_nulls(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_nulls(v) for v in obj]
            elif isinstance(obj, str) and obj == "null":
                return None
            return obj
            
        result = replace_nulls(result)
        
        if not isinstance(result, dict):
            raise ValueError("L'espace de recherche doit être un dictionnaire JSON { ... }")
            
        return result
    except json.JSONDecodeError as e:
        raise ValueError(f"Format JSON invalide : {e}")
    except Exception as e:
        raise ValueError(f"Erreur d'évaluation : {e}")

def default_search_space(model_class_name: str) -> dict:
    spaces = {
        "LogisticRegression":          {"C": {"dist": "uniform", "loc": 0.01, "scale": 10}, "solver": ["lbfgs","saga"], "max_iter": [500,1000,2000]},
        "RandomForestClassifier":      {"n_estimators": {"dist":"randint","low":50,"high":400}, "max_depth": [None,5,10,15,20], "min_samples_split": {"dist":"randint","low":2,"high":15}, "min_samples_leaf": {"dist":"randint","low":1,"high":8}, "max_features": ["sqrt","log2",0.5]},
        "RandomForestRegressor":       {"n_estimators": {"dist":"randint","low":50,"high":400}, "max_depth": [None,5,10,15,20], "min_samples_split": {"dist":"randint","low":2,"high":15}, "min_samples_leaf": {"dist":"randint","low":1,"high":8}},
        "GradientBoostingClassifier":  {"n_estimators": {"dist":"randint","low":50,"high":300}, "learning_rate": {"dist":"uniform","loc":0.01,"scale":0.3}, "max_depth": {"dist":"randint","low":2,"high":8}, "subsample": {"dist":"uniform","loc":0.6,"scale":0.4}},
        "GradientBoostingRegressor":   {"n_estimators": {"dist":"randint","low":50,"high":300}, "learning_rate": {"dist":"uniform","loc":0.01,"scale":0.3}, "max_depth": {"dist":"randint","low":2,"high":8}},
        "XGBClassifier":               {"n_estimators": {"dist":"randint","low":50,"high":400}, "learning_rate": {"dist":"uniform","loc":0.01,"scale":0.3}, "max_depth": {"dist":"randint","low":3,"high":10}, "subsample": {"dist":"uniform","loc":0.5,"scale":0.5}, "colsample_bytree": {"dist":"uniform","loc":0.5,"scale":0.5}},
        "XGBRegressor":                {"n_estimators": {"dist":"randint","low":50,"high":400}, "learning_rate": {"dist":"uniform","loc":0.01,"scale":0.3}, "max_depth": {"dist":"randint","low":3,"high":10}},
        "LGBMClassifier":              {"n_estimators": {"dist":"randint","low":50,"high":400}, "learning_rate": {"dist":"uniform","loc":0.01,"scale":0.3}, "num_leaves": {"dist":"randint","low":15,"high":128}, "subsample": {"dist":"uniform","loc":0.5,"scale":0.5}},
        "LGBMRegressor":               {"n_estimators": {"dist":"randint","low":50,"high":400}, "learning_rate": {"dist":"uniform","loc":0.01,"scale":0.3}, "num_leaves": {"dist":"randint","low":15,"high":128}},
        "CatBoostClassifier":          {"iterations": {"dist":"randint","low":50,"high":400}, "learning_rate": {"dist":"uniform","loc":0.01,"scale":0.3}, "depth": {"dist":"randint","low":3,"high":10}},
        "Ridge":                       {"alpha": {"dist":"uniform","loc":0.001,"scale":100}, "solver": ["auto","svd","cholesky","lsqr"]},
        "Lasso":                       {"alpha": {"dist":"uniform","loc":0.001,"scale":10}, "max_iter": [1000,2000,5000]},
        "SVC":                         {"C": {"dist":"uniform","loc":0.01,"scale":100}, "kernel": ["rbf","poly","sigmoid"], "gamma": ["scale","auto"]},
    }
    return spaces.get(model_class_name, {})

def scoring_for_task(task: str, subtask: str) -> str:
    if task == "classification":
        return "roc_auc" if subtask == "binary" else "f1_macro"
    return "r2"
