# flipova-kitx

![GitHub Banner](https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6)

A dynamic, interactive Machine Learning toolkit designed for Jupyter Notebooks and Google Colab. The pipeline provides a modular, step-by-step UI built with `ipywidgets` for managing your entire ML workflow directly within your notebooks.

## Quickstart

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dadflip/flipova-kitx/blob/main/notebook.ipynb)

Explore the toolkit immediately by opening the provided notebook in Google Colab.

## Features

- **Interactive UI Components**: Beautiful, interactive cells built with `ipywidgets` for configurations, data loading, EDA, and business objectives.
- **Dynamic Dependency Management**: Install only the libraries you specify right from the UI (Data Science, Vision, Neo4j, Graphs, NLP, etc.).
- **Flexible Data Loading**: Supports local tabular files, timeseries, images, videos, ontology graphs, Cypher Neo4j querying, and more.
- **Config-Driven**: Uses TOML files to dynamically render configuration settings, parameters, and form choices without altering code.

## Installation

Install the package directly from GitHub:

```bash
pip install git+https://github.com/dadflip/flipova-kitx.git
```

## Usage

In any Jupyter Notebook (or Google Colab environment), run the following to initialize the ML Pipeline UI:

```python
import traceback
from IPython.display import display
from ml_pipeline.state import PipelineState

# Initialize pipeline state and config
state = PipelineState("ml_pipeline/default.toml")

# Start Step 0: Installations
try:
    from ml_pipeline.steps.installer.ui.base import InstallerUI
    s00 = InstallerUI(state)
    display(s00.ui)
except Exception:
    traceback.print_exc()
```

Look at the `notebook.ipynb` file in this repository to see a full structure featuring EDA, context definitions, and data loading steps.
