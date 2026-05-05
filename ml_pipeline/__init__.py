"""Initialisation du package ml_pipeline."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('absl').setLevel(logging.ERROR)

from .state import PipelineState
from .styles import styles
from .config_loader import load_config

__all__ = ["PipelineState", "styles", "load_config"]
