from typing import Tuple, Optional, Any
import torch
import torch
from torchvision import models
import torch.nn as nn

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
    
def inference(model, image, device):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        values = model(image)
        _, predicted = torch.max(values, 1)
    return values, predicted
    


def testInference(test_dataset, model, device, classNames):
    class_counts = {label: [0] * len(classNames) for _, label, _ in test_dataset}
    
    print('Inference on test dataset')
    for i in range(len(test_dataset)):
        image, label, _ = test_dataset[i]
        values, predicted = inference(model, image.unsqueeze(0), device)
        class_counts[label][predicted.item()] += 1

    for class_label, predictionCount in class_counts.items():
        print(f'\nClass {test_dataset.classes[class_label]}:')
        for i in range(len(classNames)):
            print(f'  {classNames[i]}: {predictionCount[i]}')

    return class_counts


def loadDevice(forceCpu=False):
    if torch.cuda.is_available() and not forceCpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

def loadModel(modelPath, classSize, device):
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, classSize)
    model_dict = torch.load(modelPath, map_location=torch.device('cpu'), weights_only=False)
    model.load_state_dict(model_dict['model'])
    model.to(device)
    return model

def inference1vsAll(models, image, device, swapIndex):
    values = torch.zeros((len(models), 2))
    for i, model in enumerate(models):
        values[i], _ = inference(model, image, device)
        if i >= swapIndex:
            values[i][0], values[i][1] = values[i][1], values[i][0]
    
    # Se tutti i valori sono negativi, allora la predizione è -1 (fuori distribuzione)
    if torch.all(values[:, 0] < 0):
        predicted = torch.tensor(-1)
    else:
        predicted = torch.argmax(values[:, 0]) # Altrimenti si prende il valore più alto
    return values, predicted
    

# Testa il modello 1 vs All
# models: lista di modelli
# test_dataset: dataset di test
# device: dispositivo su cui eseguire l'inference
# classNames: nomi delle classi
# swapIndex: indice a partire dal quale invertire i valori delle attribuzioni, è necessario invertirli ad un certo punto perché ImageFolder carica le classi in ordine alfabetico
# di conseguenza la classe Others (fuori distribuzione) potrebbe non essere sempre la seconda, nel nostro caso attuale, la classe Others è la penultima
# quindi swapIndex = len(classNames) - 2
def testInference1vsAll(models, test_dataset, device, classNames):
    swapIndex = len(classNames) - 2
    class_counts = {label: [0] * (len(classNames) + 1) for _, label, _ in test_dataset} # +1 per contare le immagini fuori distribuzione

    print('Inference on test dataset')
    for i in range(len(test_dataset)):
        image, label, path = test_dataset[i]
        print(path)
        values, predicted = inference1vsAll(models, image.unsqueeze(0), device, swapIndex)
        class_counts[label][predicted.item()] += 1        


    for class_label, predictionCount in class_counts.items():
        print(f'\nClass {test_dataset.classes[class_label]}:')
        for i in range(len(classNames)):
            print(f'  {classNames[i]}: {predictionCount[i]}')
        print(f'  Other: {predictionCount[-1]}')


    return class_counts

def getValues6ClassModel(model, processed_image, device):
    values, predicted = inference(model, processed_image, device)   # Inference on standard image and returns logits (values) and the index of the predicted class (predicted)
    probs = torch.softmax(values, dim=1)                            # Softmax to get probabilities in a 0.00 - 1.00 range (normalizing the logits, they are now in a matrix form)
    all_classes_probs = probs[0].cpu().detach().numpy().tolist()    # Probabilities for all classes
    pred_class_idx = predicted.item()                               # Predicted class index
    conf = probs[0][pred_class_idx].item() * 100                          # Confidence of the predicted class
    all_classes_probs = [p * 100 for p in all_classes_probs]          # Convert probabilities to

    return pred_class_idx, conf, all_classes_probs

def getValues1vsAllModel(models, processed_image, device):
    values, predicted = inference1vsAll(models, processed_image, device, swapIndex=len(models))    # Inference on standard image and returns logits (values) and the index of the predicted class (predicted)
    probs = torch.softmax(values, dim=1)  
    all_classes_probs = probs[:, 0].cpu().detach().numpy().tolist()    
    pred_class_idx = predicted.item() 
    conf = probs[pred_class_idx][0].item() * 100 # Confidence of the predicted class
    all_classes_probs = [round(p * 100, 2) for p in all_classes_probs]          # Convert probabilities to

    print(f"All class probabilities:' {all_classes_probs}, '\nPredicted class index:' {pred_class_idx} '\nConfidence:', {conf}", flush=True)
    return pred_class_idx, conf, all_classes_probs