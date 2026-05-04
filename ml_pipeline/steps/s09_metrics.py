"""Étape 9 — Évaluation (EvaluationUI)."""

from .evaluation.ui.base import EvaluationUI
from IPython.display import display
import traceback

def runner(state) -> EvaluationUI:
    ev = EvaluationUI(state)
    if hasattr(ev, "ui"):
        display(ev.ui)
    return ev

__all__ = ["EvaluationUI", "runner"]
