"""Étape 7 — Dataset Balancing (SplitBalancingUI)."""

from .balancing.ui.base import SplitBalancingUI
from IPython.display import display
import traceback

def runner(state) -> SplitBalancingUI:
    ui = SplitBalancingUI(state)
    if hasattr(ui, "ui"):
        display(ui.ui)
    return ui

__all__ = ["SplitBalancingUI", "runner"]
