"""PipelineStyles — styles et templates HTML partagés."""
from __future__ import annotations
import ipywidgets as widgets
from IPython.display import display, HTML

class PipelineStyles:
    """Styles UI unifiés pour tout le pipeline ML."""

    # ── Layouts ───────────────────────────────────────────────────────────────
    LAYOUT_DD        = widgets.Layout(width="300px")
    LAYOUT_DD_LONG   = widgets.Layout(width="400px")
    LAYOUT_TEXT      = widgets.Layout(width="300px")
    LAYOUT_BTN_STD   = widgets.Layout(width="auto") # Changé à auto
    LAYOUT_BTN_LARGE = widgets.Layout(width="auto") # Changé à auto
    LAYOUT_BOX       = widgets.Layout(padding="12px", border="1px solid #e2e8f0", border_radius="8px", margin="12px 0")
    LAYOUT_ROW       = widgets.Layout(align_items="center", gap="12px", margin="8px 0")
    LAYOUT_SECTION   = widgets.Layout(border="1px solid #cbd5e1", border_radius="8px", padding="10px", margin="10px 0")
    LAYOUT_W95       = widgets.Layout(width="95%")
    LAYOUT_AUTO      = widgets.Layout(width="auto")

    # ── Button styles ─────────────────────────────────────────────────────────
    BTN_PRIMARY = "primary"
    BTN_INFO    = "info"
    BTN_SUCCESS = "success"
    BTN_WARNING = "warning"
    BTN_DANGER  = "danger"

    # ── CSS global ────────────────────────────────────────────────────────────
    CSS_GLOBALS = """
    <style>
        .pipeline-card { 
            border:1px solid #e2e8f0; 
            border-radius:12px; 
            background:#ffffff;
            padding:20px; 
            margin-bottom:16px; 
            box-shadow:0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06); 
            font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        .pipeline-title { 
            font-size:1.25em; 
            font-weight:700; 
            color:#0f172a;
            margin-bottom:12px; 
            display:flex; 
            align-items:center; 
            gap:10px; 
        }
        .pipeline-badge { 
            background:#e0e7ff; 
            color:#3730a3; 
            padding:4px 10px;
            border-radius:9999px; 
            font-size:0.75em; 
            font-weight:600; 
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .pipeline-kv-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:16px; margin-bottom:16px; }
        .pipeline-kv-cell { background:#f8fafc; padding:12px 16px; border-radius:8px; border:1px solid #e2e8f0; transition: all 0.2s ease;}
        .pipeline-kv-cell:hover { box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); border-color: #cbd5e1; }
        .kv-key { font-size:0.75em; text-transform:uppercase; color:#64748b; font-weight:700; letter-spacing:0.5px; margin-bottom:4px; }
        .kv-val { font-size:1.1em; font-weight:600; color:#0f172a; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
        .pipeline-section-title { font-size:0.85em; font-weight:800; color:#475569; text-transform:uppercase; margin:20px 0 12px 0; border-bottom:2px solid #f1f5f9; padding-bottom:6px; letter-spacing: 0.05em; }
        .pipeline-dtype-grid { display:flex; flex-wrap:wrap; gap:8px; }
        .pipeline-dtype-pill { background:#f1f5f9; border:1px solid #cbd5e1; padding:4px 10px; border-radius:6px; font-size:0.85em; font-weight:500; color:#334155; }
        
        .success-box { padding:14px; background:#f0fdf4; border-left:4px solid #22c55e; margin-bottom:12px; border-radius:6px; color:#15803d; font-weight:500; }
        .warning-box { padding:14px; background:#fffbeb; border-left:4px solid #f59e0b; margin-bottom:12px; border-radius:6px; color:#b45309; font-weight:500; }
        .error-box   { padding:14px; background:#fef2f2; border-left:4px solid #ef4444; margin-bottom:12px; border-radius:6px; color:#b91c1c; font-weight:500; }
        .info-box    { padding:14px; background:#eff6ff; border-left:4px solid #3b82f6; margin-bottom:12px; border-radius:6px; color:#1d4ed8; font-weight:500; }
        
        /* Modernize ipywidgets buttons */
        button.jupyter-button, .jupyter-widgets.jupyter-button {
            width: auto !important;
            max-width: none !important;
            min-width: 120px !important;
            padding: 8px 16px !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
            border: 1px solid transparent !important;
        }
        
        button.jupyter-button.mod-primary {
            background-color: #2563eb !important;
            color: white !important;
            box-shadow: 0 1px 3px rgba(37, 99, 235, 0.2) !important;
        }
        button.jupyter-button.mod-primary:hover {
            background-color: #1d4ed8 !important;
            box-shadow: 0 4px 6px rgba(37, 99, 235, 0.3) !important;
        }
        
        /* Override specifically some other buttons styles to match */
        button.jupyter-button:not(.mod-primary) {
            background-color: #f1f5f9 !important;
            color: #334155 !important;
            border-color: #cbd5e1 !important;
        }
        button.jupyter-button:not(.mod-primary):hover {
            background-color: #e2e8f0 !important;
            border-color: #94a3b8 !important;
        }
    </style>
    """

    @classmethod
    def apply_globals(cls) -> None:
        display(HTML(cls.CSS_GLOBALS))

    @classmethod
    def card_html(cls, title: str, subtitle: str, content: str) -> str:
        return (
            f"<div class='pipeline-card'>"
            f"<div class='pipeline-title'>{title} "
            f"<span class='pipeline-badge'>{subtitle}</span></div>"
            f"{content}</div>"
        )

    @classmethod
    def success_msg(cls, msg: str) -> widgets.HTML:
        return widgets.HTML(f"<div class='success-box'>{msg}</div>")

    @classmethod
    def error_msg(cls, msg: str) -> widgets.HTML:
        return widgets.HTML(f"<div class='error-box'>{msg}</div>")

    @classmethod
    def warning_msg(cls, msg: str) -> widgets.HTML:
        return widgets.HTML(f"<div class='warning-box'>{msg}</div>")

    @classmethod
    def info_msg(cls, msg: str) -> widgets.HTML:
        return widgets.HTML(f"<div class='info-box'>{msg}</div>")

    @classmethod
    def help_box(cls, content: str, color: str) -> widgets.Accordion:
        """Boîte d'aide repliable (fermée par défaut)."""
        inner = widgets.HTML(
            f"<div style='padding:10px 12px; background:#f8fafc; "
            f"border-left:4px solid {color}; font-size:0.85em; "
            f"color:#475569; line-height:1.6;'>{content}</div>"
        )
        acc = widgets.Accordion(children=[inner])
        acc.set_title(0, "Guide")
        acc.selected_index = None
        acc.layout = widgets.Layout(margin="0 0 12px 0")
        return acc

# Instance globale partagée
styles = PipelineStyles()
