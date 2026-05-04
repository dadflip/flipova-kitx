"""Étape 11 — Prédictions & Export (PredictionsUI)."""

from .predictions.ui.base import PredictionsUI
from IPython.display import display

def runner(state) -> PredictionsUI:
    ui = PredictionsUI(state)
    if hasattr(ui, "ui"):
        display(ui.ui)
    return ui

__all__ = ["PredictionsUI", "runner"]
