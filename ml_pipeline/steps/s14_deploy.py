from ml_pipeline.steps.deploy.ui.base import DeploymentUI

def run(state_obj):
    ui = DeploymentUI(state_obj)
    return ui.ui
