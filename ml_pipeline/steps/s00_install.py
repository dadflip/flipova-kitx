"""Étape 0 — Installation des packages (Point d'entrée)."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('absl').setLevel(logging.ERROR)

from .installer.ui.base import InstallerUI

__all__ = ["InstallerUI"]
