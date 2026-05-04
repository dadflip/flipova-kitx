import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def build_text_ui(parent_ui, text_data):
    if not isinstance(text_data, list) and not isinstance(text_data, pd.Series):
        try:
            # Assuming it's a list or we can make it a list
            text_data = [str(text_data)]
        except:
            pass

    out = widgets.Output()
    
    btn_tfidf = widgets.Button(description="TF-IDF Features", button_style="info")
    btn_bow = widgets.Button(description="Bag of Words", button_style="info")
    
    def apply_vectorizer(vect_class, name):
        with out:
            clear_output()
            try:
                vect = vect_class(max_features=100)
                mat = vect.fit_transform(text_data)
                df = pd.DataFrame(mat.toarray(), columns=vect.get_feature_names_out())
                new_ds_name = f"{parent_ui.current_ds} - {name}"
                parent_ui.state.data_raw[new_ds_name] = df
                parent_ui.state.data_types[new_ds_name] = "tabular"
                display(HTML(f"<b style='color:green;'>Extracted 100 {name} features. Saved as '{new_ds_name}'.</b>"))
            except Exception as e:
                display(HTML(f"<b style='color:red;'>Error: {e}</b>"))
                
    btn_tfidf.on_click(lambda _: apply_vectorizer(TfidfVectorizer, "TFIDF"))
    btn_bow.on_click(lambda _: apply_vectorizer(CountVectorizer, "BoW"))
    
    parent_ui.dynamic_ui.children = [
        widgets.HTML("<h3>Text Feature Engineering</h3>"),
        widgets.HTML("<p>Extract text features into a tabular dataset format.</p>"),
        widgets.HBox([btn_tfidf, btn_bow]),
        out
    ]
