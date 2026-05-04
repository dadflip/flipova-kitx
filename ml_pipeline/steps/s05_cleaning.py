"""Étape 5 — Nettoyage des données (AdvancedCleaner)."""

from .cleaner.ui.base import AdvancedCleaner

def runner(state) -> AdvancedCleaner:
    cleaner = AdvancedCleaner(state)
    from IPython.display import display
    if hasattr(cleaner, "ui"):
        display(cleaner.ui)
    return cleaner

__all__ = ["AdvancedCleaner", "runner"]
