
import os
import torch
from dotenv import dotenv_values
from typing import Any, Dict

try:
    from app.model_fun.inference import loadModel, loadDevice
except ImportError as e:
    print(f"Error importing model_fun dependencies: {e}")
    raise

config = dotenv_values(".env")

GPU_AVAILABLE = torch.cuda.is_available() and config.get("GPU", "False").lower() in ('true', '1', 't')
# Puntiamo alla cartella dei modelli, non al singolo file
SIXCLASS_MODELS_DIR = config.get("SIXCLASS_MODELS_DIR", "models/detection_models/6class")
ONEVSALL_MODEL_DIR = config.get("ONEVSALL_MODEL_DIR", "models/detection_models/1vsall")
CLASS_NAMES = ['O. exaltata', 'O. garganica', 'O. incubacea', 'O. majellensis', 'O. sphegodes', 'O. sphegodes_Palena']

def load_resources() -> Dict[str, Any]:
    """
    Carica il device, un dizionario di modelli 6-Class e i modelli 1-vs-All.
    """
    device = loadDevice(forceCpu=not GPU_AVAILABLE)
    
    # Ora usiamo un dizionario per gestire più modelli 6-class
    six_class_models = {}
    onevall_models = []
    
    print(f"--- SERVER STARTUP: Loading models... ---", flush=True)

    # 1. Load ALL 6-Class Models from directory
    try:
        if not os.path.exists(SIXCLASS_MODELS_DIR):
            raise FileNotFoundError(f"6-Class models directory not found at {SIXCLASS_MODELS_DIR}")
        
        # Estensioni supportate
        valid_ext = ('.pt', '.pth')
        model_files = [f for f in os.listdir(SIXCLASS_MODELS_DIR) if f.lower().endswith(valid_ext)]

        if not model_files:
            print("Warning: No model files found in 6class directory.", flush=True)

        for model_file in model_files:
            path = os.path.join(SIXCLASS_MODELS_DIR, model_file)
            # Carichiamo ogni modello e lo associamo al suo nome file
            six_class_models[model_file] = loadModel(path, len(CLASS_NAMES), device)
            print(f"Success: 6-Class Model '{model_file}' loaded.", flush=True)
            
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load 6-Class Models. {e}", flush=True)
        raise RuntimeError(f"Failed to load 6-Class Models: {e}")

    # 2. Load 1-vs-All Models
    try:
        if not os.path.exists(ONEVSALL_MODEL_DIR):
            print(f"Warning: 1-vs-All directory not found at {ONEVSALL_MODEL_DIR}", flush=True)
        else:
            loaded_ovr = []
            for class_name in CLASS_NAMES:
                model_file = os.path.join(ONEVSALL_MODEL_DIR, class_name, 'model.pt')
                if not os.path.exists(model_file):
                    # Fallback opzionale o errore se manca un pezzo dell'ensemble
                    raise FileNotFoundError(f"Missing 1-vs-All model for: {class_name}")
                
                ovr_model = loadModel(model_file, 2, device)
                loaded_ovr.append(ovr_model)
            
            onevall_models = loaded_ovr
            print("Success: 1-vs-All Models loaded.", flush=True)
    except Exception as e:
        print(f"Error: Failed to load 1-vs-All models. {e}", flush=True)
        raise RuntimeError(f"Failed to load 1-vs-All models: {e}")

    return {
        "device": device,
        "models_6class": six_class_models, # Restituisce il dizionario
        "onevall_models": onevall_models,
        "class_names": CLASS_NAMES
    }