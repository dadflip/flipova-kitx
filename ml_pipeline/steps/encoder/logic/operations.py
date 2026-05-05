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

def apply_encoding(df: pd.DataFrame, enc_params: dict, config: dict, existing_encoders: dict = None) -> tuple[pd.DataFrame, dict]:
    df_new = df.copy()
    if existing_encoders is None:
        existing_encoders = {}
    fitted_encoders = {}
    for col, param in enc_params.items():
        enc_value = param["enc_value"]
        kind = param["kind"]
        options_config = config.get("tabular", {}).get(kind, [])
        opt_info = next((o for o in options_config if o["value"] == enc_value), None)
        
        if col not in df_new.columns:
            continue
            
        if enc_value == "drop":
            df_new.drop(columns=[col], inplace=True)
            fitted_encoders[col] = {"action": "drop"}
        elif opt_info and "code" in opt_info and opt_info["code"]:
            loc_env = {"pd": pd, "np": np, "df": df_new, "col": col, "params": opt_info.get("params", {})}
            try:
                exec(opt_info["code"], globals(), loc_env)
                df_new = loc_env["df"]
                fitted_encoders[col] = {"action": "code", "code": opt_info["code"], "params": opt_info.get("params", {})}
            except Exception as e:
                print(f"[Error] Encoding {enc_value} on {col}: {e}")
        elif opt_info and "module" in opt_info and "class_name" in opt_info:
            try:
                import importlib
                mod = importlib.import_module(opt_info["module"])
                EncoderClass = getattr(mod, opt_info["class_name"])
                
                raw_params = opt_info.get("params", {})
                clean_params = {}
                for pk, pv in raw_params.items():
                    if pv == "null": clean_params[pk] = None
                    elif pv == "false": clean_params[pk] = False
                    elif pv == "true": clean_params[pk] = True
                    else: clean_params[pk] = pv
                        
                encoder = existing_encoders.get(col, {}).get("encoder", None)
                is_fitted = True
                if encoder is None:
                    encoder = EncoderClass(**clean_params)
                    is_fitted = False
                
                # Using [[col]] since sklearn expects 2D
                # Note: label encoder from sklearn expects 1D, so we try/except
                if opt_info["class_name"] == "LabelEncoder":
                    if is_fitted and hasattr(encoder, "transform"):
                        df_new[col] = encoder.transform(df_new[col])
                    else:
                        df_new[col] = encoder.fit_transform(df_new[col])
                else:
                    if is_fitted and hasattr(encoder, "transform"):
                        transformed = encoder.transform(df_new[[col]])
                    else:
                        transformed = encoder.fit_transform(df_new[[col]])
                    
                    if hasattr(transformed, "toarray"):
                        transformed = transformed.toarray()
                    
                    if isinstance(transformed, pd.DataFrame):
                        # category_encoders typically return DataFrames
                        transformed.index = df_new.index
                        df_new = pd.concat([df_new.drop(columns=[col]), transformed], axis=1)
                    elif len(transformed.shape) > 1 and transformed.shape[1] > 1:
                        feat_names = [f"{col}_{i}" for i in range(transformed.shape[1])]
                        if hasattr(encoder, "get_feature_names_out"):
                            try:
                                feat_names = encoder.get_feature_names_out([col])
                            except: pass
                        df_trans = pd.DataFrame(transformed, columns=feat_names, index=df_new.index)
                        df_new = pd.concat([df_new.drop(columns=[col]), df_trans], axis=1)
                    else:
                        if len(transformed.shape) > 1:
                            df_new[col] = transformed[:, 0]
                        else:
                            df_new[col] = transformed
                            
                fitted_encoders[col] = {"action": "sklearn", "encoder": encoder}
            except Exception as e:
                print(f"[Error] Encoding {enc_value} on {col} with module {opt_info['module']}: {e}")
                
    return df_new, fitted_encoders

def build_sklearn_pipeline_code(enc_params: dict, config: dict) -> str:
    imports = set()
    transformers = []
    
    for col, param in enc_params.items():
        enc_value = param["enc_value"]
        kind = param["kind"]
        if enc_value in ("none", "drop"):
            continue
            
        options_config = config.get("tabular", {}).get(kind, [])
        opt_info = next((o for o in options_config if o["value"] == enc_value), None)
        
        if opt_info and "module" in opt_info and "class_name" in opt_info:
            imports.add(f"from {opt_info['module']} import {opt_info['class_name']}")
            
            raw_params = opt_info.get("params", {})
            param_strs = []
            for pk, pv in raw_params.items():
                if pv == "null": param_strs.append(f"{pk}=None")
                elif pv == "false": param_strs.append(f"{pk}=False")
                elif pv == "true": param_strs.append(f"{pk}=True")
                elif isinstance(pv, str): param_strs.append(f"{pk}='{pv}'")
                else: param_strs.append(f"{pk}={pv}")
                    
            p_str = ", ".join(param_strs)
            transformers.append(f"    ('{enc_value}_{col}', {opt_info['class_name']}({p_str}), ['{col}'])")
            
    if not transformers:
        return ""
        
    imports.add("from sklearn.compose import ColumnTransformer")
    
    code = "\n".join(sorted(imports)) + "\n\n"
    code += "preprocessor = ColumnTransformer(\n"
    code += "    transformers=[\n"
    code += ",\n".join(transformers) + "\n"
    code += "    ],\n"
    code += "    remainder='passthrough'\n"
    code += ")\n"
    
    return code


