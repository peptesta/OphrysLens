import io
import base64
from dotenv import dotenv_values
from typing import Optional
import torch
import numpy as np
import matplotlib
import gc # Garbage collection
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients, Occlusion
from captum.attr import visualization as viz
from PIL import Image

SLIDING_WINDOW_SIZE = 20
STRIDE = 20

config = dotenv_values(".env")
MEAN = [float(x) for x in config.get("MEAN", "0.5364 0.5518 0.3866").split()]
STD = [float(x) for x in config.get("STD", "0.2045 0.2296 0.2025").split()]

def fig_to_base64(fig):
    buf = io.BytesIO()
    # --- OTTIMIZZAZIONE 2: DPI bassi ---
    fig.savefig(buf, format="JPEG", bbox_inches='tight', pad_inches=0, dpi=72)
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    # --- OTTIMIZZAZIONE 3: Pulizia violenta della memoria ---
    plt.close(fig)
    plt.close('all')
    gc.collect() # Forza il garbage collector di Python
    return img_str

def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean

def get_integrated_gradients_b64(model, input_tensor, target_label, mean, std):
    try:
        model.eval()
        ig = IntegratedGradients(model)
        
        # --- Parametri Memory-Friendly ---
        # n_steps=10: Molto meno preciso ma 5 volte più leggero del default (50)
        # internal_batch_size=1: Elabora una copia alla volta. Lento ma non crasha.
        attributions = ig.attribute(
            input_tensor, 
            target=target_label, 
            n_steps=10, 
            internal_batch_size=1 
        )
        
        original_image = denormalize(input_tensor.squeeze(0), mean, std).detach().cpu().permute(1, 2, 0).numpy()
        attribution_map = attributions.squeeze(0).cpu().permute(1, 2, 0).detach().numpy()

        fig, ax = plt.subplots(figsize=(4, 8))
        viz.visualize_image_attr(
            attribution_map,
            original_image,
            method="blended_heat_map",
            sign="absolute_value",
            show_colorbar=True,
            title="IG",
            plt_fig_axis=(fig, ax),
            use_pyplot=False
        )
        
        # Pulizia tensori
        del attributions
        del ig
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        return fig_to_base64(fig)
    except Exception as e:
        print(f"Errore IG: {e}")
        return None

def get_occlusion_b64(model, input_tensor, target_label, mean, std):
    try:
        model.eval()
        occlusion = Occlusion(model)

        # Increase strides for speed! 
        # If image is 512x256, a stride of 25 reduces passes by 4x vs stride of 10.
        attributions = occlusion.attribute(
            input_tensor,
            strides=(3, 25, 25), 
            target=target_label,
            sliding_window_shapes=(3, 30, 30),
            baselines=0,
            perturbations_per_eval=2 # Slightly faster if you have enough RAM
        )

        original_image = denormalize(input_tensor.squeeze(0), mean, std).detach().cpu().permute(1, 2, 0).numpy()
        attribution_map = attributions.squeeze(0).cpu().permute(1, 2, 0).detach().numpy()

        fig, ax = plt.subplots(figsize=(4, 8))
        viz.visualize_image_attr(
            attribution_map,
            original_image,
            method="blended_heat_map",
            sign="positive",
            show_colorbar=True,
            title="Occlusion",
            plt_fig_axis=(fig, ax),
            use_pyplot=False
        )

        # Pulizia tensori
        del attributions
        del occlusion
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        return fig_to_base64(fig)
    except Exception as e:
        print(f"Errore Occlusion: {e}")
        return None

def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    # Save as JPEG to keep it light
    image.save(buffered, format="JPEG") 
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def base64_to_image(base64_string):
    """Decodes a Base64 string into a PIL Image."""
    if not base64_string:
        return None
    try:
        return Image.open(io.BytesIO(base64.b64decode(base64_string)))
    except Exception:
        return None

def generate_explanation(model, tensor: torch.Tensor, target_idx: int, method: str) -> Optional[str]:
    if method == "none" or target_idx == -1 or tensor is None:
        return None

    try:
        # Crucial: Disable gradients for both methods to save RAM/Time 
        # (IG needs them internally but handles its own logic, Occlusion definitely doesn't)
        with torch.no_grad(): 
            if method == 'occlusion':
                return get_occlusion_b64(model, tensor, target_idx, MEAN, STD)
            
        if method == 'integrated_gradients':
            # IG requires gradients, so call it outside the no_grad block
            return get_integrated_gradients_b64(model, tensor, target_idx, MEAN, STD)
            
    except Exception as e:
        print(f"XAI Generation Warning: {e}", flush=True)
        return None