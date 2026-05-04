import os
import json
import zipfile
import shutil
import joblib
from ml_pipeline.steps.export.logic.operations import get_python_script_string

def build_deployment_package(state, output_zip="deployment_package.zip", is_xai=True, method_xai="shap", has_retrain=True):
    base_dir = "deployment_build"
    api_dir = os.path.join(base_dir, "api")
    models_dir = os.path.join(base_dir, "models")
    
    os.makedirs(api_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Extract fitted states and encoders from history
    fitted_cleaners = {}
    fitted_encoders = {}
    for item in state.history:
        if item["step"] == "Data Cleaning":
            # Can be multiple datasets, we assume the last applied one is the main one
            if "fitted_states" in item["details"]:
                fitted_cleaners.update(item["details"]["fitted_states"])
        if item["step"] == "Data Encoding":
            if "fitted_encoders" in item["details"]:
                fitted_encoders.update(item["details"]["fitted_encoders"])
                
    # 1. Export Models & Preprocessors
    models = getattr(state, "models", {})
    if models:
        joblib.dump(models, os.path.join(models_dir, "trained_models.pkl"))
    with open(os.path.join(models_dir, "fitted_cleaners.json"), "w") as f:
        json.dump(fitted_cleaners, f)
    joblib.dump(fitted_encoders, os.path.join(models_dir, "fitted_encoders.pkl"))
        
    # 2. Export Standalone Predictor
    predictor_script = """import joblib
import json
import os
import pandas as pd
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(__file__), '../models')
try:
    models = joblib.load(os.path.join(MODELS_DIR, 'trained_models.pkl'))
except:
    models = {}
    
try:
    with open(os.path.join(MODELS_DIR, 'fitted_cleaners.json'), 'r') as f:
        fitted_cleaners = json.load(f)
except:
    fitted_cleaners = {}
    
try:
    fitted_encoders = joblib.load(os.path.join(MODELS_DIR, 'fitted_encoders.pkl'))
except:
    fitted_encoders = {}

def apply_preprocessing(df):
    df_new = df.copy()
    
    # Apply cleaners
    for col, state in fitted_cleaners.items():
        if col not in df_new.columns and state['action'] != 'drop_cols':
            df_new[col] = np.nan
        act = state['action']
        if act == 'drop_cols':
            if col in df_new.columns:
                df_new.drop(columns=[col], inplace=True)
        elif act == 'fill':
            df_new[col] = df_new[col].fillna(state['value'])
        elif act == 'ffill':
            df_new[col] = df_new[col].ffill()
        elif act == 'bfill':
            df_new[col] = df_new[col].bfill()
            
    # Apply encoders
    for col, enc_info in fitted_encoders.items():
        if col not in df_new.columns:
            continue
        act = enc_info['action']
        if act == 'drop':
            df_new.drop(columns=[col], inplace=True)
        elif act == 'code':
            loc_env = {"pd": pd, "np": np, "df": df_new, "col": col, "params": enc_info.get("params", {})}
            try:
                exec(enc_info["code"], globals(), loc_env)
                df_new = loc_env["df"]
            except Exception as e:
                print(f"Error executing custom code for {col}: {e}")
        elif act == 'sklearn':
            encoder = enc_info['encoder']
            class_name = type(encoder).__name__
            try:
                if class_name == "LabelEncoder":
                    # LabelEncoder expects 1D but unseen labels might throw error, handle gracefully
                    try:
                        df_new[col] = encoder.transform(df_new[col])
                    except ValueError:
                        df_new[col] = -1 # Unknown class
                else:
                    transformed = encoder.transform(df_new[[col]])
                    if hasattr(transformed, "toarray"):
                        transformed = transformed.toarray()
                    
                    if isinstance(transformed, pd.DataFrame):
                        transformed.index = df_new.index
                        df_new = pd.concat([df_new.drop(columns=[col]), transformed], axis=1)
                    elif len(transformed.shape) > 1 and transformed.shape[1] > 1:
                        feat_names = [f"{col}_{i}" for i in range(transformed.shape[1])]
                        if hasattr(encoder, "get_feature_names_out"):
                            try:
                                feat_names = encoder.get_feature_names_out([col])
                            except: pass
                        df_trans = pd.DataFrame(transformed, columns=feat_names, index=df_new.index)
                        df_new = pd.concat([df_new.drop(columns=[col]), df_trans], axis=1)
                    else:
                        if len(transformed.shape) > 1:
                            df_new[col] = transformed[:, 0]
                        else:
                            df_new[col] = transformed
            except Exception as e:
                print(f"Error applying {class_name} on {col}: {e}")
                
    return df_new

def predict(raw_data_dict, model_name=None):
    if not models:
        return {'error': 'No models loaded'}
    model = models.get(model_name) if model_name else list(models.values())[0]
    if not model:
        return {'error': f'Model {model_name} not found'}
        
    df = pd.DataFrame([raw_data_dict])
    df_preprocessed = apply_preprocessing(df)
    
    try:
        # Align features with what the model expects if possible
        if hasattr(model, 'feature_names_in_'):
            expected_cols = list(model.feature_names_in_)
            for c in expected_cols:
                if c not in df_preprocessed.columns:
                    df_preprocessed[c] = 0
            df_preprocessed = df_preprocessed[expected_cols]
            
        pred = model.predict(df_preprocessed)
        return {'prediction': int(pred[0]) if hasattr(pred[0], 'item') else float(pred[0])}
    except Exception as e:
        return {'error': str(e)}
"""
    with open(os.path.join(api_dir, "predictor.py"), "w", encoding="utf-8") as f:
        f.write(predictor_script)

    # 3. Create FastAPI App (main.py)
    import_retrain = "from .retraining import append_feedback, trigger_retraining" if has_retrain else ""
    import_explain = "from .explainer import explain_prediction" if is_xai else ""
    
    retrain_routes = """
@app.post("/feedback")
def submit_feedback(req: FeedbackRequest, background_tasks: BackgroundTasks):
    append_feedback(req.data, req.true_label)
    background_tasks.add_task(trigger_retraining)
    return {"status": "feedback logged"}

@app.post("/retrain")
def retrain_model():
    return trigger_retraining(force=True)
""" if has_retrain else ""

    explain_logic = """
    if req.explain and "error" not in res:
        res["explanation"] = explain_prediction(req.data, model_name)
""" if is_xai else ""

    main_py = f"""from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional
import os
from .predictor import predict
{import_explain}
{import_retrain}

app = FastAPI(title="ML Pipeline API", version="1.0.0")

class PredictRequest(BaseModel):
    data: Dict[str, Any]
    explain: bool = {is_xai}

class FeedbackRequest(BaseModel):
    data: Dict[str, Any]
    true_label: Any

@app.get("/")
def health_check():
    return {{"status": "ok", "models": "loaded"}}

@app.post("/predict/{{model_name}}")
def make_prediction(model_name: str, req: PredictRequest):
    res = predict(req.data, model_name)
{explain_logic}
    return res
{retrain_routes}
"""
    with open(os.path.join(api_dir, "main.py"), "w", encoding="utf-8") as f:
        f.write(main_py)

    # 4. Explainer
    explainer_py = f"""import {method_xai}
# Mock explainer file, to be customized based on final models
def explain_prediction(data, model_name=None):
    return {{"{method_xai}_values": "Mocked {method_xai} values", "top_features": ["feature_1"]}}
"""
    with open(os.path.join(api_dir, "explainer.py"), "w", encoding="utf-8") as f:
        f.write(explainer_py)

    # 5. Retraining
    retraining_py = """import json
import os

FEEDBACK_FILE = "feedbacks.jsonl"

def append_feedback(data, label):
    with open(FEEDBACK_FILE, "a") as f:
        f.write(json.dumps({"data": data, "label": label}) + "\\n")

def trigger_retraining(force=False):
    # Logic to retrain model if enough feedback is collected
    if force:
        return {"status": "retraining started"}
    return {"status": "not enough data"}
"""
    with open(os.path.join(api_dir, "retraining.py"), "w", encoding="utf-8") as f:
        f.write(retraining_py)

    # 6. Dockerfile
    dockerfile = """FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY api /app/api
COPY models /app/models
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
    with open(os.path.join(base_dir, "Dockerfile"), "w", encoding="utf-8") as f:
        f.write(dockerfile)

    # 7. Requirements
    reqs = """fastapi
uvicorn
scikit-learn
pandas
numpy
joblib
shap
lime
pydantic
python-multipart
"""
    if any('category_encoders' in str(v.get('encoder', '')) for v in fitted_encoders.values()):
        reqs += "category_encoders\n"
        
    with open(os.path.join(base_dir, "requirements.txt"), "w", encoding="utf-8") as f:
        f.write(reqs)

    # 8. README
    readme = """# ML Pipeline Production Package
This package contains a ready-to-deploy FastAPI application for your trained models.

## Local run
`pip install -r requirements.txt`
`uvicorn api.main:app --reload`

## Docker
`docker build -t ml-api .`
`docker run -p 8000:8000 ml-api`
"""
    with open(os.path.join(base_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)

    # ZIP everything
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, base_dir)
                zipf.write(file_path, arcname)

    # Cleanup
    shutil.rmtree(base_dir, ignore_errors=True)
    return output_zip
