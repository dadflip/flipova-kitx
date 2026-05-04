from ml_pipeline.steps.sandbox.ui.base import SandboxUI

def run(state_obj):
    ui = SandboxUI(state_obj)
    return ui.ui
