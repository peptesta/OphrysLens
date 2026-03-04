import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
from dotenv import dotenv_values
import numpy as np

config = dotenv_values(".env")
DETECTION_MODEL_PATH = config.get("DETECTION_MODEL_PATH", "models/detection_models/fasterrcnn_orchid3.pth")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_cropping_model():
    try:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
        num_classes = 2 # Background + Orchid
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        
        model.load_state_dict(torch.load(DETECTION_MODEL_PATH, map_location=device))
        model.to(device).eval()
        print(f"Faster R-CNN caricato su: {device}")
        return model
    except Exception as e:
        print(f"ERRORE CARICAMENTO MODELLO: {e}")
        return None

DETECTOR = load_cropping_model()

def crop(image: Image.Image):
    """
    Rileva tutti gli oggetti e ritaglia l'immagine basandosi sul migliore.
    Ritorna: (cropped_image, all_boxes_list, all_scores_list)
    """
    if DETECTOR is None:
        return image, [], []
    
    # Prepara l'immagine per il modello
    img_tensor = F.to_tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = DETECTOR(img_tensor)[0]

    # Sposta i risultati su CPU e converti in tipi standard Python
    scores = predictions['scores'].cpu().numpy().tolist()
    boxes = predictions['boxes'].cpu().numpy().astype(int).tolist()
    
    cropped_img = image # Immagine originale come fallback

    if len(boxes) > 0:
        # Usiamo comunque la prima box (la più affidabile) per il ritaglio fisico
        x_min, y_min, x_max, y_max = boxes[0]
        
        # Aggiungiamo un padding del 10% per non tagliare troppo vicino ai petali
        padding_w = int((x_max - x_min) * 0.10)
        padding_h = int((y_max - y_min) * 0.10)
        
        crop_coords = (
            max(0, x_min - padding_w),
            max(0, y_min - padding_h),
            min(image.width, x_max + padding_w),
            min(image.height, y_max + padding_h)
        )
        cropped_img = image.crop(crop_coords)
        
    # RESTITUIAMO TUTTE LE BOX (non solo la prima)
    return cropped_img, boxes, scores