import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import matplotlib.pyplot as plt
import pandas as pd

from ml_pipeline.steps.eda.logic.video import (
    get_video_info, get_video_frame_fig, get_video_scene_cuts_fig,
    get_video_color_timeline_fig, get_video_motion_history_fig
)
from ml_pipeline.styles import styles

def build_video_ui(eda_ui, video_source):
    if not isinstance(video_source, str):
        # Could be some other object, assume we need a valid filepath to video
        try:
            video_source = str(video_source)
        except:
            return

    # Tool 1: Info & Player
    out_info = widgets.Output()
    info = get_video_info(video_source)
    def _show_info():
        with out_info:
            clear_output(wait=True)
            display(HTML("<b style='color:#374151;font-size:0.9em;'>Video Info</b>"))
            display(pd.DataFrame([info]).T.rename(columns={0: "Value"}))
            # Display basic HTML5 video player if local
            display(HTML(f'''
            <video width="400" controls>
              <source src="{video_source}" type="video/mp4">
              Your browser does not support HTML video.
            </video>
            '''))
    _show_info()
    
    # Tool 2: Frame Extractor
    out_frame = widgets.Output()
    time_slider = widgets.FloatSlider(value=0, min=0, max=info.get("Duration (s)", 60) if isinstance(info, dict) and "Duration (s)" in info else 60, step=0.5, description="Time (s):")
    btn_frame = widgets.Button(description="Extract Frame", button_style=styles.BTN_PRIMARY)
    def _extract(b):
        with out_frame:
            clear_output(wait=True)
            fig = get_video_frame_fig(video_source, time_slider.value)
            display(fig)
            plt.close(fig)
    btn_frame.on_click(_extract)
    
    # Tool 3: Scene Cuts
    out_cuts = widgets.Output()
    btn_cuts = widgets.Button(description="Analyze Scene Cuts", button_style=styles.BTN_PRIMARY)
    def _cuts(b):
        with out_cuts:
            clear_output(wait=True)
            fig = get_video_scene_cuts_fig(video_source)
            if fig:
                display(fig)
                plt.close(fig)
    btn_cuts.on_click(_cuts)
    
    # Tool 4: Color Timeline
    out_color = widgets.Output()
    btn_color = widgets.Button(description="Generate Color Timeline", button_style=styles.BTN_PRIMARY)
    def _color(b):
        with out_color:
            clear_output(wait=True)
            fig = get_video_color_timeline_fig(video_source)
            if fig:
                display(fig)
                plt.close(fig)
    btn_color.on_click(_color)
    
    # Tool 5: Motion History
    out_motion = widgets.Output()
    btn_motion = widgets.Button(description="Generate Motion History", button_style=styles.BTN_PRIMARY)
    def _motion(b):
        with out_motion:
            clear_output(wait=True)
            fig = get_video_motion_history_fig(video_source)
            if fig:
                display(fig)
                plt.close(fig)
    btn_motion.on_click(_motion)
    
    tabs = widgets.Tab(children=[
        out_info,
        widgets.VBox([widgets.HBox([time_slider, btn_frame]), out_frame]),
        widgets.VBox([btn_cuts, out_cuts]),
        widgets.VBox([btn_color, out_color]),
        widgets.VBox([btn_motion, out_motion])
    ])
    
    for i, t in enumerate(["Video Info", "Frame Extractor", "Scene Cuts", "Color Timeline", "Motion History"]):
        tabs.set_title(i, t)
        
    eda_ui.dynamic_ui.children = [tabs]
