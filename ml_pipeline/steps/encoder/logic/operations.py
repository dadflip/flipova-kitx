import pandas as pd
import numpy as np

def calc_outliers(df_col, o_act: str) -> tuple[int, int]:
    if not pd.api.types.is_numeric_dtype(df_col):
        return 0, len(df_col)
    s = df_col.dropna()
    if len(s) == 0:
        return 0, 0
    if o_act == "clip_iqr":
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        return int(((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum()), len(s)
    if o_act == "drop_zscore":
        std = s.std()
        if std > 0:
            return int((((s - s.mean()) / std).abs() > 3).sum()), len(s)
    return 0, len(s)

def apply_outliers(df: pd.DataFrame, outlier_params: dict) -> pd.DataFrame:
    df_new = df.copy()
    for col, param in outlier_params.items():
        o_act = param["o_act"]
        flag_it = param["flag_it"]
        if o_act == "none" or col not in df_new.columns:
            continue
        if flag_it:
            if o_act == "clip_iqr":
                q1, q3 = df_new[col].quantile(0.25), df_new[col].quantile(0.75)
                iqr = q3 - q1
                df_new[f"{col}_is_outlier"] = ((df_new[col] < q1-1.5*iqr) | (df_new[col] > q3+1.5*iqr)).astype(int)
            elif o_act == "drop_zscore":
                std = df_new[col].std()
                if std > 0:
                    df_new[f"{col}_is_outlier"] = (((df_new[col] - df_new[col].mean()) / std).abs() > 3).astype(int)
        if o_act == "clip_iqr":
            q1, q3 = df_new[col].quantile(0.25), df_new[col].quantile(0.75)
            iqr = q3 - q1
            df_new[col] = df_new[col].clip(lower=q1-1.5*iqr, upper=q3+1.5*iqr)
        elif o_act == "drop_zscore":
            std = df_new[col].std()
            if std > 0:
                df_new = df_new[(((df_new[col] - df_new[col].mean()) / std).abs() <= 3)]
    return df_new

def apply_encoding(df: pd.DataFrame, enc_params: dict, config: dict) -> pd.DataFrame:
    df_new = df.copy()
    for col, param in enc_params.items():
        enc_value = param["enc_value"]
        kind = param["kind"]
        options_config = config.get("tabular", {}).get(kind, [])
        opt_info = next((o for o in options_config if o["value"] == enc_value), None)
        
        if col not in df_new.columns:
            continue
            
        if enc_value == "drop":
            df_new.drop(columns=[col], inplace=True)
        elif opt_info and "code" in opt_info and opt_info["code"]:
            loc_env = {"df": df_new, "col": col, "params": opt_info.get("params", {})}
            try:
                exec(opt_info["code"], globals(), loc_env)
                df_new = loc_env["df"]
            except Exception as e:
                print(f"[Error] Encoding {enc_value} on {col}: {e}")
                
    return df_new
