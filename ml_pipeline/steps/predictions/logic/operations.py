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

def align_columns(X_ref: pd.DataFrame, X_target: pd.DataFrame, fill_value=0) -> tuple:
    X_out = X_target.copy(); ref_cols = list(X_ref.columns); tgt_set = set(X_out.columns); ref_set = set(ref_cols)
    missing = ref_set - tgt_set; extra = tgt_set - ref_set
    for c in missing:
        fill = fill_value[c] if isinstance(fill_value, pd.Series) and c in fill_value else (fill_value if not isinstance(fill_value, pd.Series) else 0)
        X_out[c] = fill
    if extra: X_out.drop(columns=list(extra), inplace=True)
    X_out = X_out[ref_cols]
    parts = []
    if missing: parts.append(f"<b style='color:#f59e0b;'>{len(missing)} col(s) manquantes → remplies :</b> {', '.join(sorted(missing))}")
    if extra:   parts.append(f"<b style='color:#64748b;'>{len(extra)} col(s) en trop → supprimées :</b> {', '.join(sorted(extra))}")
    if not parts: parts.append("<span style='color:#10b981;'>Colonnes identiques à X_train.</span>")
    return X_out, "<br>".join(parts)

def resolve_feature_columns(state, model_name):
    pred = getattr(state, "predictions", {}).get(model_name, {})
    if pred.get("feature_columns"): return pred["feature_columns"]
    tm = getattr(state, "trained_models", {}).get(model_name, {})
    if tm.get("feature_columns"): return tm["feature_columns"]
    if hasattr(state, "feature_columns") and state.feature_columns: return state.feature_columns
    X_tr = getattr(state, "data_splits", {}).get("X_train")
    if is_valid(X_tr) and hasattr(X_tr, "columns"): return list(X_tr.columns)
    return None
