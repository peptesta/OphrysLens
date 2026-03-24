# --- IMPORTS ---
import io
import os
from flask import request, jsonify, Blueprint, current_app
from PIL import Image
from dotenv import dotenv_values
from concurrent.futures import ThreadPoolExecutor


# --- LOCAL IMPORTS ---
from app import model_state                                 
from app.model_fun.preprocess_data import getTransforms     
from app.model_fun.inference_handler import predict_6class, predict_1vsall
from app.model_fun.explainability import (
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

def get_processed_tensor(image_file, transform_pipeline, device, use_crop=False):
    """Helper per gestire il caricamento e l'eventuale crop dell'immagine"""
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
    use_crop_param = request.form.get('use_crop', 'false').lower() == 'true'
    
    if model is None: return jsonify({'error': 'Selected 6class model not found or loaded'}), 500
    
    try:
        image_file = request.files.get('image')
        if not image_file: return jsonify({'error': 'No image provided'}), 400
        
        transform_pipeline = getTransforms(WIDTH, HEIGHT, True, MEAN, STD)
        tensor, _, cropped_img, did_crop = get_processed_tensor(image_file, transform_pipeline, device, use_crop=use_crop_param)
        
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


executor = ThreadPoolExecutor(max_workers=4)

@inference_bp.route('/inference/6class_batch', methods=['POST'])
def run_6class_batch_inference():
    # 1. Recuperiamo i dati comuni dal thread principale
    selected_model_name = request.form.get('model_name')
    use_crop_val = request.form.get('use_crop', 'false').lower() == 'true'
    image_files = request.files.getlist('images') # 'images' è la chiave che abbiamo visto dai log
    
    if not image_files:
        return jsonify({'error': 'No images provided'}), 400

    # Risorse del modello caricate una volta sola per il batch
    model, device = model_state.get_6class_model_by_name(selected_model_name)
    _, _, class_names = model_state.get_1vsall_resources()
    transform_pipeline = getTransforms(WIDTH, HEIGHT, True, MEAN, STD)

    if model is None:
        return jsonify({'error': 'Model not found'}), 500

    # 2. Questa è la tua logica originale adattata per girare in un thread
    # Riceve i bytes e il nome invece dell'oggetto 'request'
    def process_single_image(file_bytes, filename, use_crop):
        try:
            # Convertiamo i bytes in un oggetto leggibile dalle tue funzioni
            img_io = io.BytesIO(file_bytes)
            
            # --- TUA LOGICA ORIGINALE ---
            tensor, _, _, did_crop = get_processed_tensor(img_io, transform_pipeline, device, use_crop)
            idx, conf, probs, err = predict_6class(model, tensor, device)
            
            return {
                'success': True,
                'filename': filename,
                'predicted_class': class_names[idx] if idx != -1 else "Unknown",
                'confidence': float(conf),
                'error': err
            }
            # ----------------------------
        except Exception as e:
            return {'success': False, 'filename': filename, 'error': str(e)}

    # 3. Leggiamo i dati dei file e lanciamo i thread
    # Leggiamo f.read() qui perché i thread non possono accedere a request.files
    tasks = [(f.read(), f.filename) for f in image_files]
    
    # Esecuzione parallela
    futures = [executor.submit(process_single_image, fb, fn, use_crop_val) for fb, fn in tasks]
    
    # Aspettiamo che tutti finiscano e appendiamo i risultati
    results = [f.result() for f in futures]

    # Restituiamo il batch di risultati
    return jsonify({'results': results})
    
@inference_bp.route('/inference/generate_explainability', methods=['POST'])
def run_explainability_endpoint():
    selected_model_name = request.form.get('model_name')
    model, device = model_state.get_6class_model_by_name(selected_model_name)
    _, _, class_names = model_state.get_1vsall_resources()
    
    if model is None: 
        return jsonify({'error': 'Base model not loaded'}), 500
    
    try:
        image_file = request.files.get('image')
        if not image_file: 
            return jsonify({'error': 'No image provided'}), 400
            
        # Preprocessing
        transform_pipeline = getTransforms(WIDTH, HEIGHT, True, MEAN, STD)
        tensor, original_img, processed_img, did_crop = get_processed_tensor(image_file, transform_pipeline, device)
        
        # Inferenza per determinare il target dell'explainability
        idx, conf, _, _ = predict_6class(model, tensor, device)
        if idx == -1: 
            return jsonify({'error': 'Prediction failed'}), 400

        # Parametri opzionali per l'occlusione
        sw_size = int(request.form.get('window_size', 15))
        stride = int(request.form.get('stride', 8))

        # Generazione immagine combinata (Occlusion + IG)
        explanation_base64 = generate_explanation(
            model, 
            tensor, 
            idx, 
            class_names,
            sw_size,
            stride
        )
        
        return jsonify({
            'success': True,
            'methods': ['occlusion', 'integrated_gradients'],
            'predicted_class': class_names[idx],
            'confidence': float(conf),
            'explanation_image': explanation_base64, # Immagine affiancata
            'original_image': image_to_base64(original_img),
            'crop_applied': did_crop
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc()) # Log utile per il debug lato server
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