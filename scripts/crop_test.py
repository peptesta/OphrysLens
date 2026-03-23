import os
import requests
import time
import statistics
from datetime import datetime

# --- CONFIGURAZIONE ---
URL = "http://localhost:5000/dbinference/boxes"
TEST_IMAGES_DIR = "./datasets/test"
BATCH_SIZE = 10  # Numero di immagini per ogni richiesta
LIMIT_IMAGES = 100  # Limite massimo di immagini da testare (es. 100 o 1000)

def run_benchmark():
    # 1. Raccolta immagini
    all_files = []
    for root, _, files in os.walk(TEST_IMAGES_DIR):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_files.append(os.path.join(root, f))
    
    all_files = all_files[:LIMIT_IMAGES]
    num_total = len(all_files)
    
    if num_total == 0:
        print(f"Errore: Nessuna immagine trovata in {TEST_IMAGES_DIR}")
        return

    print(f"🚀 Inizio Benchmark su {num_total} immagini (Batch Size: {BATCH_SIZE})")
    print(f"🔗 Endpoint: {URL}\n")

    batch_times = []
    total_payload_received = 0
    images_processed = 0

    # 2. Ciclo di invio Batch
    global_start = time.time()

    for i in range(0, num_total, BATCH_SIZE):
        batch_paths = all_files[i : i + BATCH_SIZE]
        
        # Prepariamo la lista dei file
        files = []
        handles = [] # Lista di supporto per chiudere i file correttamente
        
        for p in batch_paths:
            f = open(p, 'rb')
            handles.append(f)
            files.append(('images', (os.path.basename(p), f, 'image/jpeg')))
        
        try:
            batch_start = time.time()
            response = requests.post(URL, files=files)
            response.raise_for_status() # Verifica se il server risponde 200 OK
            batch_end = time.time()
            
            elapsed = batch_end - batch_start
            batch_times.append(elapsed)
            
            total_payload_received += len(response.content)
            images_processed += len(batch_paths)

            print(f"📦 Batch {i//BATCH_SIZE + 1}: {len(batch_paths)} img in {elapsed:.3f}s")

        except Exception as e:
            print(f"❌ Errore nel batch {i//BATCH_SIZE}: {e}")
        finally:
            # Chiusura corretta dei file handles
            for h in handles:
                h.close()
                
    global_end = time.time()
    total_duration = global_end - global_start

    # 3. Report Finale
    print("\n" + "="*50)
    print(f"📊 RISULTATI BENCHMARK ({datetime.now().strftime('%H:%M:%S')})")
    print("="*50)
    print(f"Modello testato:     {URL.split('/')[-1]}")
    print(f"Immagini totali:    {images_processed}")
    print(f"Tempo Totale:       {total_duration:.2f} s")
    print(f"Throughput Medio:   {images_processed / total_duration:.2f} img/s")
    print(f"Latenza Media Batch: {statistics.mean(batch_times):.3f} s")
    print(f"Latenza Min/Max:    {min(batch_times):.3f}s / {max(batch_times):.3f}s")
    print(f"Dati Ricevuti:      {total_payload_received / (1024*1024):.2f} MB")
    print(f"Efficienza Dati:    {(total_payload_received / images_processed) / 1024:.2f} KB/img")
    print("="*50)

if __name__ == "__main__":
    # Assicurati che il server Flask sia attivo prima di lanciare lo script
    run_benchmark()