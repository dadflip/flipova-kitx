"""Étape 10 — Optimisation des hyperparamètres (OptimizationUI)."""

from .optimization.ui.base import OptimizationUI
from IPython.display import display
import traceback

def runner(state) -> OptimizationUI:
    opt = OptimizationUI(state)
    if hasattr(opt, "ui"):
        display(opt.ui)
    return opt

__all__ = ["OptimizationUI", "runner"]
