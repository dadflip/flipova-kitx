import pandas as pd
import importlib

def dynamic_import(import_string: str):
    if not import_string:
        return None
    try:
        parts = import_string.split(".")
        module = importlib.import_module(".".join(parts[:-1]))
        return getattr(module, parts[-1])
    except Exception as e:
        print(f"ImportError ({import_string}): {e}")
        return None

def is_inference_mode(splits: dict) -> bool:
    y_test = splits.get("y_test")
    X_test = splits.get("X_test")
    if y_test is None and X_test is not None:
        return True
    if isinstance(y_test, pd.Series) and y_test.empty:
        return True
    return False

def align_columns(X_ref: pd.DataFrame, X_target: pd.DataFrame, fill_value=0) -> pd.DataFrame:
    """Aligne X_target sur les colonnes de X_ref."""
    X_out = X_target.copy()
    ref_cols = list(X_ref.columns)
    tgt_set  = set(X_out.columns)
    for c in ref_cols:
        if c not in tgt_set:
            X_out[c] = fill_value
    extra = [c for c in X_out.columns if c not in ref_cols]
    if extra:
        X_out.drop(columns=extra, inplace=True)
    return X_out[ref_cols]

def align_report(X_ref: pd.DataFrame, X_target: pd.DataFrame, label: str = "X_test") -> str:
    ref_cols = set(X_ref.columns); tgt_cols = set(X_target.columns)
    missing = ref_cols - tgt_cols; extra = tgt_cols - ref_cols
    lines = []
    if missing: 
        lines.append(f"<b style='color:#f59e0b;'>{label} — {len(missing)} col(s) manquante(s) → remplies avec 0 :</b> " + ", ".join(sorted(missing)))
    if extra:   
        lines.append(f"<b style='color:#64748b;'>{label} — {len(extra)} col(s) en trop → supprimées :</b> " + ", ".join(sorted(extra)))
    if not lines: 
        lines.append(f"<span style='color:#10b981;'>{label} — colonnes identiques à X_train.</span>")
    return "<br>".join(lines)
