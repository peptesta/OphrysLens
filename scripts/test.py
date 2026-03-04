import os
import requests
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import f1_score, confusion_matrix

# --- CONFIGURAZIONE ---
BASE_URL = "http://localhost:5000/inference"
MODELS_LIST_URL = "http://localhost:5000/inference/models/available"
# Cartella locale che contiene le immagini reali da testare
# Assicurarsi che i nomi delle sottocartelle corrispondano alle classi
TEST_IMAGES_DIR = "../datasets/test/images_raw" 
CLASS_NAMES = ['O. exaltata', 'O. garganica', 'O. incubacea', 'O. majellensis', 'O. sphegodes', 'O. sphegodes_Palena']
RESULTS_DIR = f"api_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

os.makedirs(RESULTS_DIR, exist_ok=True)

def get_available_models():
    """Recupera la lista dei modelli 6class dal backend"""
    try:
        res = requests.get(MODELS_LIST_URL)
        res.raise_for_status()
        return [m['filename'] for m in res.json().get('6class', [])]
    except Exception as e:
        print(f"Errore nel recupero dei modelli: {e}")
        return []

def run_api_benchmark(model_name, image_paths, labels):
    """Esegue il benchmark di un singolo modello tramite chiamate API"""
    print(f"\n>>> Benchmarking Model: {model_name}")
    
    y_true = []
    y_pred = []
    confidences = []

    for img_path, true_label_idx in zip(image_paths, labels):
        try:
            with open(img_path, 'rb') as f:
                files = {'image': f}
                data = {
                    'model_name': model_name,
                    'use_crop': 'true' # Usiamo il crop per testare la precisione sull'orchidea
                }
                
                response = requests.post(f"{BASE_URL}/6class", files=files, data=data)
                response.raise_for_status()
                res_json = response.json()

                if res_json['success']:
                    pred_class = res_json['predicted_class']
                    # Convertiamo il nome della classe nell'indice corrispondente
                    pred_idx = CLASS_NAMES.index(pred_class) if pred_class in CLASS_NAMES else -1
                    
                    y_true.append(true_label_idx)
                    y_pred.append(pred_idx)
                    confidences.append(res_json['confidence'])
        except Exception as e:
            print(f"Errore processando {os.path.basename(img_path)}: {e}")

    # Calcolo metriche
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    acc = np.mean(y_true == y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    
    return {
        "name": model_name,
        "accuracy": acc,
        "f1_macro": f1,
        "cm": cm,
        "avg_conf": np.mean(confidences) if confidences else 0
    }

if __name__ == '__main__':
    # 1. Ottieni modelli
    models = get_available_models()
    if not models:
        print("Nessun modello trovato. Esco.")
        exit()

    # 2. Prepara dataset di test (Immagini locali)
    # Assumiamo una struttura: TEST_IMAGES_DIR/nome_classe/immagine.jpg
    image_paths = []
    labels = []
    for idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(TEST_IMAGES_DIR, class_name)
        if os.path.exists(class_dir):
            imgs = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            image_paths.extend(imgs)
            labels.extend([idx] * len(imgs))

    if not image_paths:
        print(f"Nessuna immagine trovata in {TEST_IMAGES_DIR}. Controlla la struttura delle cartelle.")
        exit()

    print(f"Inizio benchmark su {len(image_paths)} immagini con {len(models)} modelli.")

    results = []
    for model in models:
        res = run_api_benchmark(model, image_paths, labels)
        results.append(res)

    # 3. Report e Grafici
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    # Salvataggio Report Testuale
    with open(os.path.join(RESULTS_DIR, "api_comparison_report.txt"), "w") as f:
        f.write(f"API BENCHMARK REPORT - {datetime.now()}\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'MODEL':<25} | {'ACC':<10} | {'F1':<10} | {'CONF':<10}\n")
        f.write("-" * 60 + "\n")
        for r in results:
            f.write(f"{r['name']:<25} | {r['accuracy']:<10.4f} | {r['f1_macro']:<10.4f} | {r['avg_conf']:<10.2f}\n")

    # Matrici di Confusione e Grafico Finale
    names = [r['name'] for r in results]
    accs = [r['accuracy'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.barh(names, accs, color='teal')
    plt.xlabel('Accuracy')
    plt.title('Confronto Accuratezza Modelli (via API)')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_comparison.png"))

    print(f"\nBenchmark completato. Risultati salvati in: {RESULTS_DIR}")