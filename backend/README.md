# OphrysLens - Backend Service

Il backend di **OphrysLens** è un'API REST sviluppata in Python con il framework **Flask**.  
Gestisce il ciclo di vita dei modelli di Deep Learning, il processamento delle immagini e la generazione di spiegazioni visive tramite tecniche di **Explainable AI (XAI)**.

---

## Architettura Tecnica

Il servizio è modulare e scalabile, separando la logica del server dalla pipeline di inferenza neurale.

### Componenti Principali

- **`main.py`**: Entry point dell'applicazione e configurazione del server WSGI.  
- **`model_state.py`**: Gestore globale dello stato dei modelli; carica dinamicamente i pesi `.pt` all'avvio.  
- **`model_fun/`**: Contiene la logica core per l'inferenza (`inference_handler.py`) e gli algoritmi XAI (`explainability_fun.py`).  
- **`cropping_fun/`**: Modulo dedicato al pre-processing intelligente tramite **Faster R-CNN**.

---

## Deployment con Docker

Il backend è containerizzato per garantire la riproducibilità dell'ambiente di esecuzione (dipendenze PyTorch, Torchvision, ecc.).

### Avvio tramite Docker Compose (Raccomandato)

Dalla root del progetto, avvia il backend insieme al frontend con:

```bash
docker-compose up --build backend
```

O se si vuole avviare solo il backend (dalla directory del backend):

```bash
docker build -t ophryslens-backend .
docker run -p 5000:5000 ophryslens-backend
```

| Metodo | Endpoint                        | Descrizione                             | Parametri Principali              |
| ------ | ------------------------------- | --------------------------------------- | --------------------------------- |
| GET    | `/inference/models/available`   | Elenca i modelli 6-classi caricati      | Nessuno                           |
| POST   | `/inference/6class`             | Classificazione su un modello specifico | `image`, `model_name`, `use_crop` |
| POST   | `/inference/1vsall`             | Classificazione tramite Ensemble        | `image`, `use_crop`               |
| POST   | `/inference/generate_occlusion` | Heatmap tramite Occlusion Sensitivity   | `image`, `model_name`, `use_crop` |
| POST   | `/inference/generate_explain`   | Heatmap tramite Integrated Gradients    | `image`, `model_name`, `use_crop` |

### Explainable AI (XAI)

Il backend implementa tecniche per interpretare le decisioni del modello:

Integrated Gradients: Attribuisce l'importanza ai pixel calcolando l'integrale dei gradienti.

Occlusion Sensitivity: Misura la variazione di confidenza oscurando parti dell'immagine.

Configurazione

Le configurazioni globali (dimensioni immagini, medie e deviazioni standard) sono gestite tramite il file .env
