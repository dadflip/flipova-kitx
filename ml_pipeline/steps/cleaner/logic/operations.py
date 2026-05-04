import pandas as pd
import numpy as np

def auto_suggest_missing(col: str, df: pd.DataFrame, meta_info: dict) -> str:
    pct = meta_info.get("pct_miss", df[col].isna().mean() * 100)
    is_num = meta_info.get("kind") in ("numeric", "timeseries") or pd.api.types.is_numeric_dtype(df[col])
    if pct == 0: return "none"
    if pct > 50: return "drop_cols"
    if pct < 5:  return "drop_rows"
    return "median" if is_num else "mode"

def execute_cleaning_logic(df_original: pd.DataFrame, params: dict, row_widgets_info: dict) -> tuple[pd.DataFrame, dict]:
    df_new = df_original.copy()
    fitted_states = {}
    for col, param in params.items():
        m_act = param["missing"]
        null_str = param["null_reps"]
        is_num = row_widgets_info[col]["is_num"]
        
        if m_act == "none" and not null_str:
            continue
            
        if null_str:
            reps = [r.strip() for r in null_str.split(",") if r.strip()]
            to_rep = []
            for r in reps:
                to_rep.append(r)
                try: to_rep.append(int(r))
                except ValueError: pass
                try: to_rep.append(float(r))
                except ValueError: pass
            df_new[col] = df_new[col].replace(to_rep, np.nan)
            
        if m_act == "drop_cols":   
            df_new.drop(columns=[col], inplace=True)
            fitted_states[col] = {"action": "drop_cols"}
        elif m_act == "drop_rows": 
            df_new.dropna(subset=[col], inplace=True)
            fitted_states[col] = {"action": "drop_rows"}
        elif m_act == "mean" and is_num: 
            val = float(df_new[col].mean())
            df_new[col] = df_new[col].fillna(val)
            fitted_states[col] = {"action": "fill", "value": val}
        elif m_act == "median" and is_num: 
            val = float(df_new[col].median())
            df_new[col] = df_new[col].fillna(val)
            fitted_states[col] = {"action": "fill", "value": val}
        elif m_act == "mode":
            modes = df_new[col].mode()
            if not modes.empty: 
                val = modes.iloc[0]
                df_new[col] = df_new[col].fillna(val)
                fitted_states[col] = {"action": "fill", "value": val}
        elif m_act == "zero":  
            val = 0 if is_num else "0"
            df_new[col] = df_new[col].fillna(val)
            fitted_states[col] = {"action": "fill", "value": val}
        elif m_act == "ffill": 
            df_new[col] = df_new[col].ffill()
            fitted_states[col] = {"action": "ffill"}
        elif m_act == "bfill": 
            df_new[col] = df_new[col].bfill()
            fitted_states[col] = {"action": "bfill"}
            
    return df_new, fitted_states
