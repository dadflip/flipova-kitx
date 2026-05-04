import numpy as np
import pandas as pd

def apply_preprocessing(df: pd.DataFrame, history: list) -> pd.DataFrame:
    df_new = df.copy()
    
    fitted_cleaners = {}
    fitted_encoders = {}
    for item in history:
        if item["step"] == "Data Cleaning":
            if "fitted_states" in item["details"]:
                fitted_cleaners.update(item["details"]["fitted_states"])
        if item["step"] == "Data Encoding":
            if "fitted_encoders" in item["details"]:
                fitted_encoders.update(item["details"]["fitted_encoders"])
                
    # Apply cleaners
    for col, state in fitted_cleaners.items():
        if col not in df_new.columns and state['action'] != 'drop_cols':
            df_new[col] = np.nan
        act = state['action']
        if act == 'drop_cols':
            if col in df_new.columns:
                df_new.drop(columns=[col], inplace=True)
        elif act == 'fill':
            df_new[col] = df_new[col].fillna(state['value'])
        elif act == 'ffill':
            df_new[col] = df_new[col].ffill()
        elif act == 'bfill':
            df_new[col] = df_new[col].bfill()
            
    # Apply encoders
    for col, enc_info in fitted_encoders.items():
        if col not in df_new.columns:
            continue
        act = enc_info['action']
        if act == 'drop':
            df_new.drop(columns=[col], inplace=True)
        elif act == 'code':
            loc_env = {"pd": pd, "np": np, "df": df_new, "col": col, "params": enc_info.get("params", {})}
            try:
                exec(enc_info["code"], globals(), loc_env)
                df_new = loc_env["df"]
            except Exception as e:
                print(f"Error executing custom code for {col}: {e}")
        elif act == 'sklearn':
            encoder = enc_info['encoder']
            class_name = type(encoder).__name__
            try:
                if class_name == "LabelEncoder":
                    try:
                        df_new[col] = encoder.transform(df_new[col])
                    except ValueError:
                        df_new[col] = -1
                else:
                    transformed = encoder.transform(df_new[[col]])
                    if hasattr(transformed, "toarray"):
                        transformed = transformed.toarray()
                    
                    if isinstance(transformed, pd.DataFrame):
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
            except Exception as e:
                print(f"Error applying {class_name} on {col}: {e}")
                
    return df_new
