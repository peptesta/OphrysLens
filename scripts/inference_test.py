import os
import requests
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import itertools
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score

# --- CONFIGURAZIONE ---
BASE_URL = "http://localhost:5000/inference/6class_batch"
MODELS_LIST_URL = "http://localhost:5000/inference/models/available"
TEST_IMAGES_DIR = "./datasets/cropped" 
CLASS_NAMES = ['O. exaltata', 'O. garganica', 'O. incubacea', 'O. majellensis', 'O. sphegodes', 'O. sphegodes_Palena']

def get_available_models():
    """Recupera la lista dei modelli 6class dal backend"""
    try:
        res = requests.get(MODELS_LIST_URL)
        res.raise_for_status()
        return [m['filename'] for m in res.json().get('6class', [])]
    except Exception as e:
        print(f"Errore nel recupero dei modelli: {e}")
        return []

def plot_confusion_matrix(cm, classes, model_name, folder):
    """Genera e salva un'immagine della matrice di confusione"""
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix: {model_name}')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Classe Reale')
    plt.xlabel('Classe Predetta')
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"cm_{model_name.replace('.pt', '')}.png"))
    plt.close()

def run_api_benchmark(model_name, image_paths, labels, batch_size=10):
    print(f"\n>>> Benchmarking Model: {model_name}")
    
    y_true = []
    y_pred = []
    confidences = []
    
    start_time = time.time()

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_labels = labels[i : i + batch_size]
        
        opened_files = []
        files = []
        
        try:
            for img_path in batch_paths:
                f = open(img_path, 'rb')
                opened_files.append(f)
                files.append(('images', (os.path.basename(img_path), f, 'image/jpeg')))

            data = {'model_name': model_name}
            response = requests.post(BASE_URL, files=files, data=data)
            response.raise_for_status()
            res_json = response.json()

            for j, res in enumerate(res_json.get('results', [])):
                if res.get('success'):
                    pred_class = res['predicted_class']
                    p_idx = CLASS_NAMES.index(pred_class) if pred_class in CLASS_NAMES else -1
                    
                    y_true.append(batch_labels[j])
                    y_pred.append(p_idx)
                    confidences.append(res['confidence'])

            print(f"Processed: {min(i + batch_size, len(image_paths))}/{len(image_paths)}", end='\r')

        except Exception as e:
            print(f"\n!!! Errore critico durante la batch {i//batch_size}: {e}")
        finally:
            for f in opened_files: f.close()

    end_time = time.time()
    total_time = end_time - start_time
    
    num_images = len(y_pred)
    avg_time_per_image = total_time / num_images if num_images > 0 else 0

    if num_images == 0:
        return {"name": model_name, "accuracy": 0, "total_time": total_time}

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    
    # --- CALCOLO METRICHE RICHIESTE ---
    acc = accuracy_score(y_true_np, y_pred_np)
    precision = precision_score(y_true_np, y_pred_np, average='macro', zero_division=0)
    recall = recall_score(y_true_np, y_pred_np, average='macro', zero_division=0)
    f1_macro = f1_score(y_true_np, y_pred_np, average='macro', zero_division=0)
    cm = confusion_matrix(y_true_np, y_pred_np, labels=list(range(len(CLASS_NAMES))))
    
    return {
        "name": model_name,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_macro": f1_macro,
        "cm": cm,
        "avg_conf": np.mean(confidences) if confidences else 0,
        "total_time": total_time,
        "ms_per_img": avg_time_per_image * 1000,
        "processed_count": num_images
    }

if __name__ == '__main__':
    folder_input = input("Inserisci il nome della cartella di salvataggio (senza spazi): ").strip()
    folder_clean = "".join(folder_input.split())
    if not folder_clean:
        folder_clean = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    RESULTS_DIR = folder_clean
    os.makedirs(RESULTS_DIR, exist_ok=True)

    models = get_available_models()
    if not models:
        print("Nessun modello trovato.")
        exit()

    image_paths = []
    labels = []
    for idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(TEST_IMAGES_DIR, class_name)
        if os.path.exists(class_dir):
            imgs = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            image_paths.extend(imgs)
            labels.extend([idx] * len(imgs))

    if not image_paths:
        print(f"Nessuna immagine trovata in {TEST_IMAGES_DIR}.")
        exit()

    print(f"Inizio benchmark su {len(image_paths)} immagini con {len(models)} modelli.")

    results = []
    for model in models:
        res = run_api_benchmark(model, image_paths, labels)
        results.append(res)
        plot_confusion_matrix(res['cm'], CLASS_NAMES, res['name'], RESULTS_DIR)

    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    # --- REPORT FINALE CON NUOVE METRICHE ---
    report_path = os.path.join(RESULTS_DIR, "final_benchmark_report.txt")
    with open(report_path, "w") as f:
        f.write(f"API BENCHMARK REPORT - {datetime.now()}\n")
        f.write("-" * 110 + "\n")
        header = f"{'MODEL':<30} | {'ACC':<8} | {'PREC':<8} | {'REC':<8} | {'F1':<8} | {'AVG CONF':<10} | {'MS/IMG':<8}\n"
        f.write(header)
        f.write("-" * 110 + "\n")
        for r in results:
            line = f"{r['name']:<30} | {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1_macro']:.4f} | {r['avg_conf']:.4f}   | {r['ms_per_img']:.2f}\n"
            f.write(line)
            print(line.strip())

    plt.figure(figsize=(12, 6))
    names = [r['name'] for r in results]
    accs = [r['accuracy'] for r in results]
    plt.barh(names, accs, color='skyblue')
    plt.xlabel('Accuracy')
    plt.title('Confronto Accuratezza Modelli')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_chart.png"))

    print(f"\nBenchmark completato.")
    print(f"I file sono stati salvati nella cartella: {os.path.abspath(RESULTS_DIR)}")
