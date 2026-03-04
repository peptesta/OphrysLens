from typing import Tuple, Optional, Any
import torch
from app.model_fun.inference import getValues6ClassModel, getValues1vsAllModel

CLASS_NAMES = ['O. exaltata', 'O. garganica', 'O. incubacea', 'O. majellensis', 'O. sphegodes', 'O. sphegodes_Palena']

# --- SPECIFIC INFERENCE FUNCTIONS ---

def predict_6class(model, tensor: torch.Tensor, device) -> Tuple[int, float, Any, Optional[str]]:
    """
    Esegue la predizione utilizzando il modello standard a 6 classi.
    """
    if tensor is None:
        return -1, 0.0, None, "No image tensor provided."
    
    try:
        idx, conf, probs = getValues6ClassModel(model, tensor, device)
        print(f"DEBUG: Standard 6-Class Result -> Class: {idx}, Conf: {conf:.4f}", flush=True)
        return idx, conf, probs, None
    except Exception as e:
        print(f"DEBUG: 6-Class Inference Error: {str(e)}", flush=True)
        return -1, 0.0, None, f"Inference Error: {str(e)}"


def predict_1vsall(onevall_models, tensor: torch.Tensor, device) -> Tuple[int, float, Any, Optional[str]]:
    """
    Esegue la predizione utilizzando la strategia dei modelli One-Vs-All.
    """
    if tensor is None:
        return -1, 0.0, None, "No image tensor provided."
    
    try:
        idx, conf, probs = getValues1vsAllModel(onevall_models, tensor, device)
        print(f"DEBUG: 1vsAll Model Result -> Class: {idx}, Conf: {conf:.4f}", flush=True)
        
        if idx == -1:
            return -1, 0.0, probs, "No class predicted with sufficient confidence."
        
        return idx, conf, probs, None
    except Exception as e:
        print(f"DEBUG: 1vsAll Inference Error: {str(e)}", flush=True)
        return -1, 0.0, None, f"Inference Error: {str(e)}"