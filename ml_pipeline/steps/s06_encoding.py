"""Étape 6 — Encodage & Outliers (UltimateEncoder)."""

from .encoder.ui.base import UltimateEncoder
from IPython.display import display

def runner(state) -> UltimateEncoder:
    encoder = UltimateEncoder(state)
    if hasattr(encoder, "ui"):
        display(encoder.ui)
    return encoder

__all__ = ["UltimateEncoder", "runner"]
