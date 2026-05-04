"""Étape 4 — Feature Engineering (FeatureEngUI)."""
from .feature_eng.ui.base import FeatureEngUI

def runner(state) -> FeatureEngUI:
    fe = FeatureEngUI(state)
    from IPython.display import display
    if hasattr(fe, "ui"):
        display(fe.ui)
    return fe

__all__ = ["FeatureEngUI", "runner"]
