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
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
        
        :root {
            --flipova-primary: #6366f1;
            --flipova-secondary: #a855f7;
            --flipova-accent: #f472b6;
            --flipova-bg: #f8fafc;
            --flipova-text: #0f172a;
            --flipova-text-muted: #64748b;
        }

        .flipova-hero {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            border-radius: 24px;
            padding: 48px;
            margin-bottom: 32px;
            position: relative;
            overflow: hidden;
            color: white;
            font-family: 'Plus Jakarta Sans', sans-serif;
            box-shadow: 0 20px 40px -12px rgba(15, 23, 42, 0.3);
        }

        .flipova-hero::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle at 50% 50%, rgba(99, 102, 241, 0.15), transparent 50%);
            animation: rotate 20s linear infinite;
        }

        @keyframes rotate {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }

        .flipova-logo {
            font-size: 2.5em;
            font-weight: 800;
            letter-spacing: -0.04em;
            margin-bottom: 8px;
            background: linear-gradient(to right, #818cf8, #c084fc, #f472b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .flipova-tagline {
            font-size: 1.1em;
            color: #94a3b8;
            font-weight: 500;
            max-width: 600px;
            line-height: 1.6;
            margin-bottom: 24px;
        }

        .flipova-stats {
            display: flex;
            gap: 32px;
            margin-top: 32px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            padding-top: 24px;
        }

        .flipova-stat-item {
            display: flex;
            flex-direction: column;
        }

        .stat-label {
            font-size: 0.7em;
            text-transform: uppercase;
            color: #64748b;
            font-weight: 700;
            letter-spacing: 0.1em;
        }

        .stat-value {
            font-size: 1.25em;
            font-weight: 700;
            color: #f8fafc;
        }

        .pipeline-card { 
            border: 1px solid #f1f5f9; 
            border-radius: 16px; 
            background: #ffffff;
            padding: 24px; 
            margin-bottom: 24px; 
            box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05); 
            font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: #334155;
            transition: transform 0.2s ease;
        }
        
        .pipeline-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 24px -4px rgba(15, 23, 42, 0.08);
        }

        .pipeline-title { 
            font-size: 1.5em; 
            font-weight: 700; 
            color: #0f172a;
            margin-bottom: 16px; 
            display: flex; 
            align-items: center; 
            justify-content: space-between;
            gap: 12px; 
            letter-spacing: -0.02em;
        }
        
        .pipeline-badge { 
            background: #f1f5f9; 
            color: #475569; 
            padding: 6px 14px;
            border-radius: 12px; 
            font-size: 0.65em; 
            font-weight: 700; 
            text-transform: uppercase;
            letter-spacing: 0.06em;
            border: 1px solid #e2e8f0;
        }

        /* Glassmorphism for specific UI elements */
        .glass-panel {
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 16px;
        }

        button.jupyter-button {
            border-radius: 12px !important;
            padding: 12px 24px !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
        }
    </style>
    """

    @classmethod
    def apply_globals(cls) -> None:
        display(HTML(cls.CSS_GLOBALS))

    @classmethod
    def flipova_header(cls, version: str = "v0.9.0", username: str = "Flipova Designer") -> widgets.HTML:
        """En-tête majestueux pour le notebook Flipova KitX."""
        html = f"""
        <div class="flipova-hero">
            <div class="flipova-logo">
                Flipova KitX
                <span style="font-size: 0.3em; background: rgba(99, 102, 241, 0.2); padding: 4px 10px; border-radius: 6px; color: #818cf8; vertical-align: middle; border: 1px solid rgba(99, 102, 241, 0.3);">ENTERPRISE AI</span>
            </div>
            <div class="flipova-tagline">
                Plateforme modulaire de Machine Learning pour l'exploration, le feature engineering 
                et le déploiement de modèles prédictifs haute performance.
            </div>
            
            <div class="flipova-stats">
                <div class="flipova-stat-item">
                    <span class="stat-label">Version</span>
                    <span class="stat-value">{version}</span>
                </div>
                <div class="flipova-stat-item">
                    <span class="stat-label">Statut</span>
                    <span class="stat-value"><span style="color: #10b981;">●</span> Prêt</span>
                </div>
                <div class="flipova-stat-item">
                    <span class="stat-label">Opérateur</span>
                    <span class="stat-value">{username}</span>
                </div>
            </div>
        </div>
        """
        return widgets.HTML(html)

    @classmethod
    def card_html(cls, title: str, subtitle: str, content: str) -> str:
        return (
            f"<div class='pipeline-card'>"
            f"<div class='pipeline-title'><div>{title}</div> "
            f"<span class='pipeline-badge'>{subtitle}</span></div>"
            f"<div style='color: #475569; line-height: 1.5;'>{content}</div></div>"
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
