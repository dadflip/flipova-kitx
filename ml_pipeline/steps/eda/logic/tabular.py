import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def infer_types(df: pd.DataFrame, available_types: list | None = None) -> dict:
    if not available_types:
        available_types = ["numeric", "categorical", "datetime", "binary", "id_like", "text"]
    col_meta = {}
    for col in df.columns:
        s, n_unique, n_total = df[col], df[col].nunique(), max(len(df[col].dropna()), 1)
        kind = "unknown"
        if pd.api.types.is_datetime64_any_dtype(s) and "datetime" in available_types:
            kind = "datetime"
        elif (pd.api.types.is_bool_dtype(s) or n_unique == 2) and "binary" in available_types:
            kind = "binary"
        elif pd.api.types.is_numeric_dtype(s):
            if "id_like" in available_types and n_unique / n_total > 0.95 and n_unique > 50:
                kind = "id_like"
            elif "numeric" in available_types:
                kind = "numeric"
        else:
            avg_len = s.dropna().astype(str).str.len().mean() if not s.dropna().empty else 0
            if "text" in available_types and n_unique / n_total > 0.90 and avg_len > 20:
                kind = "text"
            elif "id_like" in available_types and n_unique / n_total > 0.95 and n_unique > 50:
                kind = "id_like"
            elif "categorical" in available_types:
                kind = "categorical"
        if kind == "unknown":
            kind = available_types[0] if available_types else "categorical"
        col_meta[col] = {
            "kind": kind, "n_unique": n_unique,
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "pct_miss": float(s.isna().mean() * 100),
        }
    return col_meta

class EDAVisualizerUtils:
    """Utilitaires de visualisation univariée et bivariée."""

    @staticmethod
    def plot_univariate(df, col, kind=None, plot_type=None, bins=30, kde=True,
                        log_scale=False, color="#3b82f6", hue=None, palette="Set2"):
        is_num = pd.api.types.is_numeric_dtype(df[col])
        if kind is None:
            kind = "numeric" if is_num else "categorical"
        plot_as_numeric = kind in ("numeric", "timeseries") or (
            kind not in ("categorical", "binary", "id_like", "text") and is_num)
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#f8fafc"); ax.set_facecolor("#f8fafc")
        hue_data = df[hue] if hue and hue in df.columns else None
        if plot_as_numeric:
            plot_type = plot_type or "hist"
            data = df[col].dropna()
            if hue_data is not None:
                groups = df[[col, hue]].dropna()
                unique_vals = groups[hue].unique()
                pal = sns.color_palette(palette, len(unique_vals))
                for i, val in enumerate(unique_vals):
                    subset = groups[groups[hue] == val][col]
                    if plot_type == "hist":
                        sns.histplot(subset, kde=kde, bins=bins, color=pal[i],
                                     label=str(val), alpha=0.6, ax=ax, log_scale=log_scale)
                    elif plot_type == "kde":
                        sns.kdeplot(subset, fill=True, color=pal[i],
                                    label=str(val), alpha=0.5, ax=ax, log_scale=log_scale)
                    elif plot_type in ("box", "violin"):
                        plot_df = df[[col, hue]].dropna().copy()
                        plot_df[hue] = plot_df[hue].astype(str)
                        if plot_type == "box":
                            sns.boxplot(data=plot_df, x=hue, y=col, palette=palette, ax=ax)
                        else:
                            sns.violinplot(data=plot_df, x=hue, y=col, palette=palette, ax=ax)
                        if log_scale:
                            ax.set_yscale("log")
                        break
                ax.legend(title=hue, fontsize=9)
            else:
                if plot_type == "hist":
                    sns.histplot(data, kde=kde, bins=bins, color=color, log_scale=log_scale, ax=ax)
                elif plot_type == "kde":
                    sns.kdeplot(data, fill=True, color=color, log_scale=log_scale, ax=ax)
                elif plot_type == "box":
                    sns.boxplot(y=data, color=color, ax=ax)
                    if log_scale: ax.set_yscale("log")
                elif plot_type == "violin":
                    sns.violinplot(y=data, color=color, ax=ax)
                    if log_scale: ax.set_yscale("log")
            ax.set_title(f"Distribution de {col}" + (f" par {hue}" if hue_data is not None else ""))
        else:
            plot_type = plot_type or "bar"
            if hue_data is not None:
                plot_df = df[[col, hue]].dropna().copy()
                plot_df[col] = plot_df[col].astype(str)
                plot_df[hue] = plot_df[hue].astype(str)
                top_cats = plot_df[col].value_counts().nlargest(bins).index
                plot_df = plot_df[plot_df[col].isin(top_cats)]
                if plot_type == "bar":
                    tbl = pd.crosstab(plot_df[col], plot_df[hue])
                    tbl.plot(kind="bar", ax=ax, colormap=palette, edgecolor="white")
                    ax.legend(title=hue, fontsize=9, bbox_to_anchor=(1.01, 1), loc="upper left")
                    ax.set_title(f"Top {len(top_cats)} catégories pour {col} par {hue}")
                elif plot_type == "pie":
                    tbl = pd.crosstab(plot_df[col], plot_df[hue], normalize="index")
                    tbl.plot(kind="bar", stacked=True, ax=ax, colormap=palette)
                    ax.legend(title=hue, fontsize=9, bbox_to_anchor=(1.01, 1), loc="upper left")
                    ax.set_title(f"Proportion de {hue} dans {col}")
            else:
                val_counts = df[col].value_counts().head(bins)
                if plot_type == "bar":
                    sns.barplot(y=val_counts.index.astype(str), x=val_counts.values,
                                palette="viridis", ax=ax)
                    ax.set_title(f"Top {len(val_counts)} catégories pour {col}")
                elif plot_type == "pie":
                    ax.axis("off")
                    fig.clear()
                    ax2 = fig.add_subplot(111)
                    ax2.pie(val_counts.values, labels=val_counts.index,
                            autopct="%1.1f%%", colors=sns.color_palette("viridis", len(val_counts)))
                    ax2.set_title(f"Top {len(val_counts)} catégories pour {col}")
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_bivariate(df, x_col, y_col, x_kind=None, y_kind=None,
                       plot_type=None, hue=None, alpha=0.5, palette="Set2"):
        num_dtypes = (pd.api.types.is_numeric_dtype(df[x_col]),
                      pd.api.types.is_numeric_dtype(df[y_col]))
        x_kind = x_kind or ("numeric" if num_dtypes[0] else "categorical")
        y_kind = y_kind or ("numeric" if num_dtypes[1] else "categorical")
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#f8fafc"); ax.set_facecolor("#f8fafc")

        def _is_num_kind(k, is_n):
            return k in ("numeric", "timeseries") or (
                k not in ("categorical", "binary", "id_like", "text") and is_n)

        x_is_n = _is_num_kind(x_kind, num_dtypes[0])
        y_is_n = _is_num_kind(y_kind, num_dtypes[1])
        hue_data = df[hue] if hue and hue in df.columns else None

        if x_is_n and y_is_n:
            plot_type = plot_type or "scatter"
            if plot_type == "scatter":
                sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_data,
                                alpha=alpha, palette=palette if hue else None, ax=ax)
            elif plot_type == "hexbin":
                ax.hexbin(x=df[x_col], y=df[y_col], gridsize=30, cmap="Blues", mincnt=1)
                fig.colorbar(ax.collections[0], ax=ax, label="count")
            elif plot_type == "hist2d":
                sns.histplot(data=df, x=x_col, y=y_col, bins=30,
                             pthresh=.1, cmap="mako", cbar=True, ax=ax)
            elif plot_type == "kde":
                sns.kdeplot(data=df, x=x_col, y=y_col, hue=hue_data,
                            fill=True, alpha=alpha, palette=palette, ax=ax)
        elif (x_is_n and not y_is_n) or (not x_is_n and y_is_n):
            plot_type = plot_type or "box"
            c_col = x_col if not x_is_n else y_col
            top_cats = df[c_col].value_counts().nlargest(15).index
            plot_df  = df[df[c_col].isin(top_cats)].copy()
            plot_df[c_col] = plot_df[c_col].astype(str)
            if hue and hue in plot_df.columns:
                plot_df[hue] = plot_df[hue].astype(str)
            if plot_type == "box":
                sns.boxplot(data=plot_df, x=x_col, y=y_col, hue=hue, palette=palette, ax=ax)
            elif plot_type == "violin":
                sns.violinplot(data=plot_df, x=x_col, y=y_col, hue=hue, palette=palette, ax=ax)
            elif plot_type == "strip":
                sns.stripplot(data=plot_df, x=x_col, y=y_col, hue=hue,
                              alpha=alpha, palette=palette, dodge=bool(hue), ax=ax)
            elif plot_type == "swarm":
                if len(plot_df) > 1000:
                    plot_df = plot_df.sample(1000, random_state=42)
                sns.swarmplot(data=plot_df, x=x_col, y=y_col,
                              hue=hue, palette=palette, dodge=bool(hue), ax=ax)
        else:
            plot_type = plot_type or "heatmap"
            top_x = df[x_col].value_counts().nlargest(15).index
            top_y = df[y_col].value_counts().nlargest(15).index
            plot_df = df[df[x_col].isin(top_x) & df[y_col].isin(top_y)]
            if plot_type == "heatmap":
                tbl = pd.crosstab(plot_df[y_col], plot_df[x_col])
                sns.heatmap(tbl, annot=True, fmt="d", cmap="Blues", ax=ax)
            elif plot_type == "stacked_bar":
                tbl = pd.crosstab(plot_df[x_col], plot_df[y_col], normalize="index")
                tbl.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
                ax.legend(title=y_col, bbox_to_anchor=(1.05, 1), loc="upper left")
            elif plot_type == "pie":
                plt.close(fig)
                tbl = pd.crosstab(plot_df[y_col], plot_df[x_col])
                n_pies = len(tbl.index)
                cols_n = min(n_pies, 3)
                rows_n = (n_pies - 1) // cols_n + 1
                fig, axes = plt.subplots(rows_n, cols_n, figsize=(cols_n * 4, rows_n * 4))
                fig.patch.set_facecolor("#f8fafc")
                axes = np.array(axes).reshape(-1)
                for i, (idx, row) in enumerate(tbl.iterrows()):
                    if i < len(axes):
                        axes[i].pie(row.values, labels=row.index, autopct="%1.1f%%",
                                    colors=sns.color_palette("viridis", len(row)))
                        axes[i].set_title(f"{y_col} = {idx}")
                for j in range(i + 1, len(axes)):
                    axes[j].axis("off")

        if plot_type != "pie":
            ax.set_title(f"{y_col} vs {x_col}")
        plt.tight_layout()
        return fig
