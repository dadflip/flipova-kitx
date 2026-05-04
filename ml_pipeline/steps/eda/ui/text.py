import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import matplotlib.pyplot as plt
import pandas as pd
from ml_pipeline.steps.eda.logic.text import (
    get_text_stats, get_text_top_words_fig, get_text_wordcloud_fig,
    get_text_ngrams_fig, get_text_sentiment, get_text_complexity_fig
)
from ml_pipeline.styles import styles

def build_text_ui(eda_ui, text: str) -> None:
    # Tool 1: Basics
    out_basic = widgets.Output()
    def _show_basic():
        with out_basic:
            clear_output()
            stats = get_text_stats(text)
            print(f"Length: {stats['length']} chars | Words: {stats['word_count']} | Sentences: {stats['sentence_count']}\n")
            if stats['top_words']:
                fig = get_text_top_words_fig(stats['top_words'])
                display(fig)
                plt.close(fig)
            print("--- Preview ---")
            print(text[:1000] + ("\n... [TRUNCATED]" if len(text) > 1000 else ""))
    _show_basic()

    # Tool 2: Word Cloud
    out_wc = widgets.Output()
    btn_wc = widgets.Button(description="Generate Word Cloud", button_style=styles.BTN_PRIMARY)
    def _show_wc(b):
        with out_wc:
            clear_output(wait=True)
            fig = get_text_wordcloud_fig(text)
            display(fig)
            plt.close(fig)
    btn_wc.on_click(_show_wc)
    
    # Tool 3: N-Grams
    out_ngrams = widgets.Output()
    n_slider = widgets.IntSlider(value=2, min=2, max=5, description="N-Gram:")
    btn_ngrams = widgets.Button(description="Plot N-Grams", button_style=styles.BTN_PRIMARY)
    def _show_ngrams(b):
        with out_ngrams:
            clear_output(wait=True)
            fig = get_text_ngrams_fig(text, n=n_slider.value)
            display(fig)
            plt.close(fig)
    btn_ngrams.on_click(_show_ngrams)
    
    # Tool 4: Sentiment Profile
    out_sentiment = widgets.Output()
    btn_sentiment = widgets.Button(description="Analyze Sentiment", button_style=styles.BTN_PRIMARY)
    def _show_sentiment(b):
        with out_sentiment:
            clear_output(wait=True)
            res = get_text_sentiment(text)
            display(pd.DataFrame([res]).T.rename(columns={0: "Value"}))
    btn_sentiment.on_click(_show_sentiment)
    
    # Tool 5: Readability & Complexity
    out_complex = widgets.Output()
    btn_complex = widgets.Button(description="Analyze Complexity", button_style=styles.BTN_PRIMARY)
    def _show_complex(b):
        with out_complex:
            clear_output(wait=True)
            fig = get_text_complexity_fig(text)
            display(fig)
            plt.close(fig)
    btn_complex.on_click(_show_complex)

    tabs = widgets.Tab(children=[
        out_basic,
        widgets.VBox([btn_wc, out_wc]),
        widgets.VBox([widgets.HBox([n_slider, btn_ngrams]), out_ngrams]),
        widgets.VBox([btn_sentiment, out_sentiment]),
        widgets.VBox([btn_complex, out_complex])
    ])
    
    for i, t in enumerate(["Basic Stats", "Word Cloud", "N-Grams", "Sentiment", "Complexity"]):
        tabs.set_title(i, t)
        
    eda_ui.dynamic_ui.children = [tabs]
