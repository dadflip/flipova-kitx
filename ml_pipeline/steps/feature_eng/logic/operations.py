import pandas as pd
import numpy as np
import math
import traceback
import matplotlib.pyplot as plt
import seaborn as sns

def apply_math(df, col1, col2_str, op, const_val, new_name):
    df = df.copy()
    v1 = pd.to_numeric(df[col1], errors="coerce")
    if op in ("log(A)","exp(A)","sqrt(A)","A^2","Abs(A)"):
        res = {"log(A)": np.log1p(v1), "exp(A)": np.exp(v1), "sqrt(A)": np.sqrt(np.maximum(v1,0)), "A^2": v1**2, "Abs(A)": np.abs(v1)}[op]
    else:
        v2 = const_val if col2_str == "(Constant)" else pd.to_numeric(df[col2_str], errors="coerce")
        if op == "/":
            v2 = v2.replace(0, np.nan) if hasattr(v2, "replace") else v2
            res = v1 / v2
        else:
            res = {"+": v1+v2, "-": v1-v2, "*": v1*v2, "A^B/C": v1**v2, "Modulo": v1%v2}[op]
    df[new_name] = res
    return df, new_name

def parse_cond_value(raw: str):
    raw = raw.strip()
    try: return int(raw)
    except ValueError: pass
    try: return float(raw)
    except ValueError: pass
    return raw

def build_mask(df, col, op, raw_val):
    s = df[col]
    if op == "is null":     return s.isna()
    if op == "is not null": return s.notna()
    val = parse_cond_value(raw_val)
    if op == "==":  return s == val
    if op == "!=":  return s != val
    if op == ">":   return pd.to_numeric(s, errors="coerce") > float(val)
    if op == ">=":  return pd.to_numeric(s, errors="coerce") >= float(val)
    if op == "<":   return pd.to_numeric(s, errors="coerce") < float(val)
    if op == "<=":  return pd.to_numeric(s, errors="coerce") <= float(val)
    if op == "isin":     return s.astype(str).isin([v.strip() for v in raw_val.split(",")])
    if op == "not isin": return ~s.astype(str).isin([v.strip() for v in raw_val.split(",")])
    if op == "contains (str)": return s.astype(str).str.contains(raw_val, na=False)
    if op == "startswith":     return s.astype(str).str.startswith(raw_val, na=False)
    if op == "endswith":       return s.astype(str).str.endswith(raw_val, na=False)
    raise ValueError(f"Unknown operator: {op}")

def resolve_value(df, col_dd_val, const_txt_val):
    return parse_cond_value(const_txt_val) if col_dd_val == "(Constant)" else df[col_dd_val]

def apply_condition(df, base_col, base_op, base_val, combine, extra_rows, then_col, then_val_txt, else_col, else_val_txt, map_raw, new_name):
    df = df.copy()
    map_raw = map_raw.strip()
    if map_raw:
        mapping = {}
        for pair in map_raw.replace("\n", ",").split(","):
            if ":" not in pair: continue
            k, v = pair.split(":", 1)
            mapping[parse_cond_value(k.strip())] = parse_cond_value(v.strip())
        df[new_name] = df[base_col].map(mapping)
    else:
        mask = build_mask(df, base_col, base_op, base_val)
        for (r_col, r_op, r_val) in extra_rows:
            extra = build_mask(df, r_col, r_op, r_val)
            mask = (mask & extra) if combine == "AND" else (mask | extra)
        then_val = resolve_value(df, then_col, then_val_txt)
        else_val = resolve_value(df, else_col, else_val_txt)
        df[new_name] = np.where(mask, then_val, else_val)
    return df, new_name

def run_formula(df, code, raw_dataset=None, all_datasets=None):
    df_in = df.copy()
    ns = {
        "np": np, "pd": pd, "math": math, "df": df_in, 
        "raw_dataset": raw_dataset, "all_datasets": all_datasets
    }
    exec(compile(code, "<formula>", "exec"), ns)
    
    # CASE 1: The entire 'df' variable was replaced (e.g. df = pd.read_csv(...))
    if "df" in ns and ns["df"] is not df_in and isinstance(ns["df"], pd.DataFrame):
        return ns["df"], {"__replaced__": True}
        
    # CASE 2: Columns were added/modified in the 'df' object object
    new_or_mod = {}
    
    # Detect modifications in the 'df' object itself (standard Pandas way)
    for col in df_in.columns:
        if col not in df.columns or not df_in[col].equals(df[col]):
            new_or_mod[col] = df_in[col]

    return df_in, new_or_mod

def apply_text(df, col, op, arg1, arg2, new_name):
    df = df.copy()
    s = df[col].astype(str)
    res = {"Lowercase": s.str.lower(), "Uppercase": s.str.upper(), "Length": s.str.len(),
           "Extract Regex": s.str.extract(f"({arg1})", expand=False),
           "Replace": s.str.replace(arg1, arg2, regex=True)}.get(op)
    if op == "Split & Keep N":
        res = s.str.split(arg2).str[int(arg1)]
    if res is None: raise ValueError(f"Unknown op: {op}")
    df[new_name] = res
    return df, new_name

def apply_date(df, col, features):
    df = df.copy()
    s = pd.to_datetime(df[col], errors="coerce")
    created = []
    mapping = {"Year": s.dt.year, "Month": s.dt.month, "Day": s.dt.day,
               "DayOfWeek": s.dt.dayofweek, "Hour": s.dt.hour, "Minute": s.dt.minute,
               "IsWeekend": (s.dt.dayofweek >= 5).astype(int)}
    for feat in features:
        if feat in mapping:
            name = f"{col}_{feat}"
            df[name] = mapping[feat]
            created.append(name)
    return df, created

def apply_binning(df, col, method, bins_val, labels, new_name):
    df = df.copy()
    s = pd.to_numeric(df[col], errors="coerce")
    lbls = False if labels else None
    
    if method == "Equal Width (Cut)":       res = pd.cut(s, bins=int(bins_val), labels=lbls)
    elif method == "Equal Frequency (Qcut)": res = pd.qcut(s, q=int(bins_val), labels=lbls, duplicates="drop")
    elif method == "Custom Edges":
        edges = [float(x.strip()) for x in bins_val.split(",")]
        res = pd.cut(s, bins=edges, labels=lbls)
    df[new_name] = res
    return df, new_name

def create_viz_fig(df, x, y, hue, kind):
    fig, ax = plt.subplots(figsize=(9, 5))
    if kind == "auto":
        kind = "scatter" if pd.api.types.is_numeric_dtype(df[x]) and pd.api.types.is_numeric_dtype(df[y]) else ("box" if pd.api.types.is_numeric_dtype(df[y]) else "bar")
    try:
        {"scatter": lambda: sns.scatterplot(data=df, x=x, y=y, hue=hue, alpha=0.7, ax=ax),
         "line":    lambda: sns.lineplot(data=df, x=x, y=y, hue=hue, ax=ax),
         "bar":     lambda: sns.barplot(data=df, x=x, y=y, hue=hue, ax=ax),
         "box":     lambda: sns.boxplot(data=df, x=x, y=y, hue=hue, ax=ax),
         "violin":  lambda: sns.violinplot(data=df, x=x, y=y, hue=hue, ax=ax),
         "hist":    lambda: sns.histplot(data=df, x=x, hue=hue, kde=True, ax=ax),
         "kde":     lambda: sns.kdeplot(data=df, x=x, hue=hue, fill=True, ax=ax)}.get(kind, lambda: None)()
        ax.set_title(f"{kind.capitalize()} : {y} vs {x}")
        fig.tight_layout()
    except Exception as e:
        plt.close(fig)
        raise e
    return fig

def create_dashboard_fig(df, target, features):
    fig, axes = plt.subplots(1, len(features), figsize=(len(features)*6, 5))
    if len(features) == 1: axes = [axes]
    
    target_vals = df[target].dropna().unique()
    is_binary = len(target_vals) == 2
    rate_series = (df[target] == sorted(target_vals)[-1]).astype(int) if is_binary else pd.to_numeric(df[target], errors="coerce")
    overall_rate = rate_series.mean()
    
    for ax, feat in zip(axes, features):
        is_num = pd.api.types.is_numeric_dtype(df[feat]) and df[feat].nunique() >= 20
        if is_num:
            if is_binary: sns.histplot(data=df, x=feat, hue=target, multiple="layer", alpha=0.5, ax=ax)
            else: sns.scatterplot(data=df, x=feat, y=target, ax=ax, alpha=0.5)
            ax.set_title(f"{feat} vs {target}")
        else:
            df_t = pd.DataFrame({"f": df[feat].astype(str), "r": rate_series}).dropna()
            if df[feat].nunique() > 30:
                top = df_t["f"].value_counts().nlargest(15).index
                df_t = df_t[df_t["f"].isin(top)]
            rate_by_cat = df_t.groupby("f")["r"].mean().sort_values()
            ax.barh(range(len(rate_by_cat)), rate_by_cat.values, color="#a78bfa")
            ax.set_yticks(range(len(rate_by_cat)))
            ax.set_yticklabels(rate_by_cat.index)
            if overall_rate is not None and not np.isnan(overall_rate):
                ax.axvline(overall_rate, color="#f28859", linestyle="--", label="Overall mean")
                ax.legend()
            ax.set_title(f"Target Rate by {feat}")
            
    fig.tight_layout()
    return fig
