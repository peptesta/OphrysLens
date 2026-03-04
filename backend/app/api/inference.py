# --- IMPORTS ---
import io
import os
from flask import request, jsonify, Blueprint
from PIL import Image
from dotenv import dotenv_values

# --- LOCAL IMPORTS ---
from app import model_state                                 
from app.model_fun.preprocess_data import getTransforms     
from app.model_fun.inference_handler import predict_6class, predict_1vsall
from app.model_fun.explainability_fun import (
    generate_explanation,                                              
    image_to_base64                                                               
)

# Tentativo di importazione del modulo di cropping
try:
    from app.cropping_fun.fasterrcnn_crop import crop
    HAS_EXTERNAL_CROP = True
except ImportError:
    HAS_EXTERNAL_CROP = False

inference_bp = Blueprint('inference', __name__)
config = dotenv_values(".env")

WIDTH = int(config.get("WIDTH", 256))
HEIGHT = int(config.get("HEIGHT", 512))
MEAN = [float(x) for x in config.get("MEAN", '0.5414286851882935 0.5396731495857239 0.3529253602027893').split()]
STD = [float(x) for x in config.get("STD", '0.2102500945329666 0.23136012256145477 0.19928686320781708').split()]

def get_processed_tensor(image_file, transform_pipeline, device):
    """Helper per gestire il caricamento e l'eventuale crop dell'immagine"""
    use_crop = request.form.get('use_crop', 'false').lower() == 'true'
    image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
    
    image_to_process = image
    crop_executed = False

    if use_crop and HAS_EXTERNAL_CROP:
        try:
            cropped_img, _, _ = crop(image.copy())
            if cropped_img is not None:
                image_to_process = cropped_img
                crop_executed = True
        except Exception as e:
            print(f"Crop failed, using original: {e}")

    tensor = transform_pipeline(image_to_process).unsqueeze(0).to(device)
    return tensor, image, image_to_process, crop_executed

@inference_bp.route('/inference/6class', methods=['POST'])
def run_6class_inference():
    # Otteniamo il nome del modello dal frontend
    selected_model_name = request.form.get('model_name')
    model, device = model_state.get_6class_model_by_name(selected_model_name)
    _, _, class_names = model_state.get_1vsall_resources()
    
    if model is None: return jsonify({'error': 'Selected 6class model not found or loaded'}), 500
    
    try:
        image_file = request.files.get('image')
        if not image_file: return jsonify({'error': 'No image provided'}), 400
        
        transform_pipeline = getTransforms(WIDTH, HEIGHT, True, MEAN, STD)
        tensor, _, cropped_img, did_crop = get_processed_tensor(image_file, transform_pipeline, device)
        
        idx, conf, probs, err = predict_6class(model, tensor, device)
        
        return jsonify({
            'success': True,
            'model_type': '6class',
            'model_name_used': selected_model_name or "default",
            'crop_applied': did_crop,
            'predicted_class': class_names[idx] if idx != -1 else "Unknown",
            'confidence': conf,
            'all_classes_probs': probs,
            'image_cropped': image_to_base64(cropped_img) if did_crop else None,
            'error': err
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@inference_bp.route('/inference/1vsall', methods=['POST'])
def run_1vsall_inference():
    # La strategia 1vsall usa l'ensemble fisso
    onevall_models, device, class_names = model_state.get_1vsall_resources()
    if not onevall_models: return jsonify({'error': '1vsAll models not loaded'}), 500
    
    try:
        image_file = request.files.get('image')
        if not image_file: return jsonify({'error': 'No image provided'}), 400
            
        transform_pipeline = getTransforms(WIDTH, HEIGHT, True, MEAN, STD)
        tensor, _, cropped_img, did_crop = get_processed_tensor(image_file, transform_pipeline, device)
        
        idx, conf, probs, err = predict_1vsall(onevall_models, tensor, device)
        
        return jsonify({
            'success': True,
            'model_type': '1vsall',
            'crop_applied': did_crop,
            'predicted_class': class_names[idx] if idx != -1 else "Unknown",
            'confidence': conf,
            'all_classes_probs': probs,
            'image_cropped': image_to_base64(cropped_img) if did_crop else None,
            'error': err
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@inference_bp.route('/inference/generate_occlusion', methods=['POST'])
def run_occlusion_endpoint():
    selected_model_name = request.form.get('model_name')
    model, device = model_state.get_6class_model_by_name(selected_model_name)
    _, _, class_names = model_state.get_1vsall_resources()
    
    if model is None: return jsonify({'error': 'Base model not loaded'}), 500
    
    try:
        image_file = request.files.get('image')
        if not image_file: return jsonify({'error': 'No image provided'}), 400
            
        transform_pipeline = getTransforms(WIDTH, HEIGHT, True, MEAN, STD)
        tensor, original_img, processed_img, did_crop = get_processed_tensor(image_file, transform_pipeline, device)
        
        idx, conf, _, _ = predict_6class(model, tensor, device)
        if idx == -1: return jsonify({'error': 'Prediction failed'}), 400

        explanation_base64 = generate_explanation(model, tensor, idx, 'occlusion')
        
        return jsonify({
            'success': True,
            'method': 'occlusion',
            'crop_applied': did_crop,
            'predicted_class': class_names[idx],
            'explanation_image': explanation_base64,
            'original_image': image_to_base64(original_img),
            'processed_image': image_to_base64(processed_img) if did_crop else None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@inference_bp.route('/inference/generate_explain', methods=['POST'])
def run_explain_endpoint():
    selected_model_name = request.form.get('model_name')
    model, device = model_state.get_6class_model_by_name(selected_model_name)
    _, _, class_names = model_state.get_1vsall_resources()
    
    if model is None: return jsonify({'error': 'Base model not loaded'}), 500
    
    try:
        image_file = request.files.get('image')
        if not image_file: return jsonify({'error': 'No image provided'}), 400
            
        transform_pipeline = getTransforms(WIDTH, HEIGHT, True, MEAN, STD)
        tensor, original_img, processed_img, did_crop = get_processed_tensor(image_file, transform_pipeline, device)
        
        idx, conf, _, _ = predict_6class(model, tensor, device)
        if idx == -1: return jsonify({'error': 'Prediction failed'}), 400

        explanation_base64 = generate_explanation(model, tensor, idx, 'integrated_gradients')
        
        return jsonify({
            'success': True,
            'method': 'integrated_gradients',
            'crop_applied': did_crop,
            'predicted_class': class_names[idx],
            'explanation_image': explanation_base64,
            'original_image': image_to_base64(original_img),
            'processed_image': image_to_base64(processed_img) if did_crop else None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@inference_bp.route('/inference/models/available', methods=['GET'])
def get_available_models():
    models_root = os.path.join(os.getcwd(), 'models', 'detection_models')
    valid_ext = ('.pt', '.pth', '.onnx')
    response = {"6class": []}
    
    try:
        category_path = os.path.join(models_root, "6class")
        if os.path.exists(category_path):
            files = [f for f in os.listdir(category_path) if f.lower().endswith(valid_ext)]
            files.sort()
            for idx, filename in enumerate(files):
                response["6class"].append({
                    "id": idx,
                    "filename": filename,
                    "label": os.path.splitext(filename)[0].replace('_', ' ').capitalize()
                })
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500