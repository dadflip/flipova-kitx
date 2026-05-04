"""PipelineStyles — styles et templates HTML partagés."""
from __future__ import annotations
import ipywidgets as widgets
from IPython.display import display, HTML

class PipelineStyles:
    """Styles UI unifiés pour tout le pipeline ML."""

    # ── Layouts ───────────────────────────────────────────────────────────────
    @property
    def LAYOUT_DD(self): return widgets.Layout(width="300px")
    
    @property
    def LAYOUT_DD_LONG(self): return widgets.Layout(width="400px")
    
    @property
    def LAYOUT_TEXT(self): return widgets.Layout(width="300px")
    
    @property
    def LAYOUT_BTN_STD(self): return widgets.Layout(width="auto")

    @property
    def LAYOUT_BTN_LARGE(self): return widgets.Layout(width="auto")

    @property
    def LAYOUT_BOX(self): return widgets.Layout(padding="12px", border="1px solid #e2e8f0", border_radius="8px", margin="12px 0")
    
    @property
    def LAYOUT_ROW(self): return widgets.Layout(align_items="center", gap="12px", margin="8px 0")
    
    @property
    def LAYOUT_SECTION(self): return widgets.Layout(border="1px solid #cbd5e1", border_radius="8px", padding="10px", margin="10px 0")
    
    @property
    def LAYOUT_W95(self): return widgets.Layout(width="95%")
    
    @property
    def LAYOUT_AUTO(self): return widgets.Layout(width="auto")

    # ── Button styles ─────────────────────────────────────────────────────────
    BTN_PRIMARY = "primary"
    BTN_INFO    = "info"
    BTN_SUCCESS = "success"
    BTN_WARNING = "warning"
    BTN_DANGER  = "danger"

    # ── CSS global ────────────────────────────────────────────────────────────
    CSS_GLOBALS = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        .pipeline-card { 
            border: 1px solid #f1f5f9; 
            border-radius: 16px; 
            background: #ffffff;
            padding: 24px; 
            margin-bottom: 24px; 
            box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05); 
            font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: #334155;
        }
        .pipeline-title { 
            font-size: 1.5em; 
            font-weight: 700; 
            color: #0f172a;
            margin-bottom: 16px; 
            display: flex; 
            align-items: center; 
            gap: 12px; 
            letter-spacing: -0.02em;
        }
        .pipeline-badge { 
            background: #f8fafc; 
            color: #475569; 
            padding: 4px 12px;
            border-radius: 9999px; 
            font-size: 0.65em; 
            font-weight: 600; 
            text-transform: uppercase;
            letter-spacing: 0.06em;
            border: 1px solid #e2e8f0;
        }
        .pipeline-kv-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 20px; }
        .pipeline-kv-cell { background: #ffffff; padding: 16px; border-radius: 12px; border: 1px solid #f1f5f9; transition: all 0.2s ease; box-shadow: 0 1px 3px rgba(15,23,42,0.02);}
        .pipeline-kv-cell:hover { box-shadow: 0 4px 12px -2px rgba(15,23,42,0.05); border-color: #e2e8f0; transform: translateY(-1px); }
        .kv-key { font-size: 0.7em; text-transform: uppercase; color: #64748b; font-weight: 600; letter-spacing: 0.05em; margin-bottom: 6px; }
        .kv-val { font-size: 1.15em; font-weight: 600; color: #0f172a; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .pipeline-section-title { font-size: 0.8em; font-weight: 700; color: #64748b; text-transform: uppercase; margin: 24px 0 12px 0; border-bottom: 1px solid #f1f5f9; padding-bottom: 8px; letter-spacing: 0.06em; }
        .pipeline-dtype-grid { display: flex; flex-wrap: wrap; gap: 8px; }
        .pipeline-dtype-pill { background: #f8fafc; border: 1px solid #e2e8f0; padding: 4px 12px; border-radius: 8px; font-size: 0.8em; font-weight: 500; color: #475569; }
        
        .success-box { padding: 16px; background: #f0fdf4; border: 1px solid #bbf7d0; margin-bottom: 16px; border-radius: 12px; color: #166534; font-weight: 500; font-size: 0.9em; box-shadow: 0 1px 2px rgba(0,0,0,0.02); }
        .warning-box { padding: 16px; background: #fffbeb; border: 1px solid #fde68a; margin-bottom: 16px; border-radius: 12px; color: #92400e; font-weight: 500; font-size: 0.9em; box-shadow: 0 1px 2px rgba(0,0,0,0.02);  }
        .error-box   { padding: 16px; background: #fef2f2; border: 1px solid #fecaca; margin-bottom: 16px; border-radius: 12px; color: #991b1b; font-weight: 500; font-size: 0.9em; box-shadow: 0 1px 2px rgba(0,0,0,0.02);  }
        .info-box    { padding: 16px; background: #f8fafc; border: 1px solid #e2e8f0; margin-bottom: 16px; border-radius: 12px; color: #334155; font-weight: 500; font-size: 0.9em; box-shadow: 0 1px 2px rgba(0,0,0,0.02);  }
        
        /* Modernize ipywidgets buttons */
        button.jupyter-button, .jupyter-widgets.jupyter-button {
            width: auto !important;
            max-width: none !important;
            min-width: 120px !important;
            padding: 10px 20px !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            font-size: 0.9em !important;
            letter-spacing: 0.02em !important;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
            border: 1px solid transparent !important;
            font-family: inherit !important;
        }
        
        button.jupyter-button.mod-primary {
            background-color: #0f172a !important;
            color: #ffffff !important;
            box-shadow: 0 2px 4px rgba(15, 23, 42, 0.1) !important;
        }
        button.jupyter-button.mod-primary:hover {
            background-color: #334155 !important;
            box-shadow: 0 4px 8px rgba(15, 23, 42, 0.15) !important;
            transform: translateY(-1px) !important;
        }
        
        /* Override specifically some other buttons styles to match */
        button.jupyter-button:not(.mod-primary) {
            background-color: #ffffff !important;
            color: #334155 !important;
            border-color: #e2e8f0 !important;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.02) !important;
        }
        button.jupyter-button:not(.mod-primary):hover {
            background-color: #f8fafc !important;
            border-color: #cbd5e1 !important;
            transform: translateY(-1px) !important;
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
