"""Étape 12 — Export & Rapport (ReportGenerator)."""

from .export.ui.base import ReportGenerator
from IPython.display import display

def runner(state) -> ReportGenerator:
    exporter = ReportGenerator(state)
    if hasattr(exporter, "ui"):
        display(exporter.ui)
    return exporter

__all__ = ["ReportGenerator", "runner"]
