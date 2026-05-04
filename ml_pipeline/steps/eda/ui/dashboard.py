import io
import base64
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=96, facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

class PlotDashboard:
    """Dashboard de plots persistant entre les reruns."""

    def __init__(self, state=None):
        self._state   = state
        self._entries = list(state.eda_dashboard) if (state and hasattr(state, "eda_dashboard")) else []
        self._visible = False
        self._build_widget()

    def _build_widget(self) -> None:
        self.toggle_btn = widgets.Button(
            description=f"Plot Dashboard ({len(self._entries)})",
            button_style="info", layout=widgets.Layout(width="220px", height="32px"))
        self.clear_btn  = widgets.Button(description="Clear all", button_style="warning",
                                          layout=widgets.Layout(width="100px", height="32px"))
        self.export_btn = widgets.Button(description="Export HTML", button_style="primary",
                                          layout=widgets.Layout(width="120px", height="32px"))
        self.toggle_btn.on_click(self._toggle)
        self.clear_btn.on_click(self._clear)
        self.export_btn.on_click(self._export_html)
        self._header = widgets.HBox(
            [self.toggle_btn, self.clear_btn, self.export_btn],
            layout=widgets.Layout(align_items="center", gap="8px", padding="8px 12px",
                                   border="1px solid #e2e8f0", border_radius="8px", margin="12px 0 0 0"))
        self._grid_out = widgets.Output()
        self._body = widgets.VBox([self._grid_out], layout=widgets.Layout(
            display="none", border="1px solid #e2e8f0", border_radius="8px",
            padding="12px", margin="4px 0 0 0", background_color="#ffffff"))
        self.widget = widgets.VBox([self._header, self._body])

    def _toggle(self, b=None) -> None:
        self._visible = not self._visible
        self._body.layout.display = "flex" if self._visible else "none"
        arrow = "v" if self._visible else ">"
        self.toggle_btn.description = f"{arrow} Plot Dashboard ({len(self._entries)})"
        if self._visible:
            self._render_grid()

    def _clear(self, b=None) -> None:
        self._entries.clear()
        self._persist()
        self._update_label()
        with self._grid_out: 
            clear_output(wait=True)

    def _update_label(self) -> None:
        arrow = "v" if self._visible else ">"
        self.toggle_btn.description = f"{arrow} Plot Dashboard ({len(self._entries)})"

    def _persist(self) -> None:
        if self._state is not None:
            self._state.eda_dashboard = list(self._entries)

    def add(self, fig, title: str = "") -> None:
        b64 = _fig_to_b64(fig)
        self._entries.append({"b64": b64, "title": title})
        self._persist()
        self._update_label()
        if self._visible:
            self._render_grid()

    def _render_grid(self) -> None:
        with self._grid_out:
            clear_output(wait=True)
            if not self._entries:
                display(HTML("<div style='color:#94a3b8;font-size:0.85em;padding:8px;'>No plots saved yet.</div>"))
                return
            cards = ""
            for i, entry in enumerate(self._entries):
                label = entry["title"] or f"Plot {i+1}"
                cards += (
                    f"<div style='border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;"
                    f"background:#fff;flex:1 1 420px;min-width:320px;max-width:580px;'>"
                    f"<div style='background:#f8fafc;padding:6px 12px;border-bottom:1px solid #e2e8f0;"
                    f"font-size:0.78em;font-weight:600;color:#475569;display:flex;"
                    f"justify-content:space-between;align-items:center;'>"
                    f"<span>{label}</span><span style='color:#94a3b8;font-weight:400;'>#{i+1}</span></div>"
                    f"<div style='padding:6px;'>"
                    f"<img src='data:image/png;base64,{entry['b64']}' style='width:100%;height:auto;display:block;'/>"
                    f"</div></div>"
                )
            display(HTML(f"<div style='display:flex;flex-wrap:wrap;gap:12px;padding:4px 0;'>{cards}</div>"))

    def _export_html(self, b=None) -> None:
        if not self._entries:
            return
        cards = ""
        for i, entry in enumerate(self._entries):
            label = entry["title"] or f"Plot {i+1}"
            cards += (
                f"<div class='card'>"
                f"<div class='card-header'><span>{label}</span><span class='idx'>#{i+1}</span></div>"
                f"<img src='data:image/png;base64,{entry['b64']}'/></div>"
            )
        html = (
            "<!DOCTYPE html><html><head><meta charset='utf-8'><title>EDA Dashboard</title>"
            "<style>body{font-family:sans-serif;background:#f1f5f9;margin:0;padding:20px}"
            "h1{color:#1e293b;font-size:1.1em;font-weight:600;margin-bottom:16px}"
            ".grid{display:flex;flex-wrap:wrap;gap:16px}"
            ".card{background:#fff;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;flex:1 1 420px}"
            ".card-header{background:#f8fafc;padding:6px 12px;border-bottom:1px solid #e2e8f0;"
            "font-size:0.78em;font-weight:600;color:#475569;display:flex;justify-content:space-between}"
            ".idx{color:#94a3b8;font-weight:400}img{width:100%;height:auto;display:block}"
            "</style></head><body>"
            f"<h1>EDA Plot Dashboard — {len(self._entries)} plots</h1>"
            f"<div class='grid'>{cards}</div></body></html>"
        )
        path = "eda_dashboard.html"
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        display(HTML(
            f"<div style='color:#065f46;background:#d1fae5;border-left:4px solid #10b981;"
            f"padding:8px 12px;font-size:0.85em;border-radius:4px;margin-top:6px;'>"
            f"Dashboard exporté → <b>{path}</b> ({len(self._entries)} plots)</div>"
        ))
