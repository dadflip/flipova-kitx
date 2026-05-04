"""Étape 8 — Modélisation (ModelingUI)."""

from .modeling.ui.base import ModelingUI
from IPython.display import display
import traceback

def runner(state) -> ModelingUI:
    m = ModelingUI(state)
    if hasattr(m, "ui"):
        display(m.ui)
    return m

__all__ = ["ModelingUI", "runner"]
