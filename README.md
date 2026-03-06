# OphrysLens - Orchid Classification System

**OphrysLens** è una piattaforma integrata per l'identificazione tassonomica e l'analisi visiva delle orchidee del genere *Ophrys*.  
Il sistema combina modelli di Deep Learning all'avanguardia con tecniche di Explainable AI (XAI) per fornire una classificazione accurata e interpretabile.

---

## Architettura del Sistema

Il progetto è basato su un'architettura a microservizi orchestrata tramite Docker, suddivisa in tre componenti principali:

### Backend (Python/Flask)
- Gestione dell'inferenza tramite modelli PyTorch  
- Rilevamento e isolamento dei fiori mediante Faster R-CNN  
- Generazione di heatmaps e metodi di explainability basate sulle seguenti tecniche:
  - Occlusion Sensitivity
  - Integrated Gradients  

### Frontend (Next.js/TypeScript)
- Interfaccia utente reattiva per il caricamento delle immagini  
- Configurazione dei parametri di analisi  
- Visualizzazione comparativa dei risultati  

### Utility Scripts
- Strumenti per il benchmarking delle performance  
- Manipolazione batch dei dataset di test  

---

## Screenshot del Frontend

### Home Principale

![Dashboard Screenshot](docs/images/home.png)

---

### Caricamento Immagine, Configurazione Parametri e Mappe di Calore

![Upload Screenshot](docs/images/compare.png)

---

### Visualizzazione Bounding Box e Intervento Manuale

![Results Screenshot](docs/images/bounding_box.png)

---

## Requisiti di Sistema

- Docker (versione 20.10+)  
- Docker Compose  
- Modelli PyTorch (.pt) salvati nella directory:


---

## Guida all'Avvio Rapido

Il sistema è completamente automatizzato tramite Docker Compose.  
Non è richiesta l'installazione locale di Python o Node.js.

### 1. Clonazione del Repository

```bash
git clone https://github.com/peptesta/OphrysLens.git
cd OphrysLens
```

### 2. Avvio con docker compose

```bash
docker-compose up --build
```

## Accesso ai Servizi

Al completamento dell'avvio dello stack Docker, il sistema sarà disponibile ai seguenti indirizzi:

- **Interfaccia Web (Frontend):**  
  http://localhost:3000

- **Servizi API (Backend):**  
  http://localhost:5000

---

## Documentazione di Dettaglio

Per informazioni tecniche approfondite su ogni modulo, consultare i README dedicati:

- **Backend Documentation**  
  Dettagli su API, architetture PyTorch e logica XAI.

- **Frontend Documentation**  
  Guida ai componenti React e gestione dello stato.

- **Scripts Documentation**  
  Istruzioni per benchmark e pre-processing dei dati.

