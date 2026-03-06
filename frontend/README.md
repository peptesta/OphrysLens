# OphrysLens - Frontend Application

Il frontend di **OphrysLens** è un'applicazione web sviluppata con **Next.js (React + TypeScript)**.

Fornisce un'interfaccia moderna e interattiva per:

- Upload e preview delle immagini
- Selezione del modello (6-class o 1-vs-All)
- Modalità di analisi (Original, Cropped, Compare)
- Visualizzazione delle heatmap di Explainable AI (Integrated Gradients e Occlusion)
- Confronto tra immagine originale e croppata

L'applicazione comunica con il backend Flask tramite API REST.

---

## Stack Tecnologico

- **Next.js**
- **React**
- **TypeScript**
- **Tailwind CSS**
- Docker

---

## Struttura del Progetto

```

frontend/
├── public/                 # Asset statici accessibili via URL (immagini, icone, logo)
│   ├── images/             # Risorse grafiche del progetto
│   └── logo.png            # Logo dell'applicazione
├── src/                    # Codice sorgente dell'applicazione
│   ├── app/                # App Router di Next.js (pagine, layout e routing)
│   ├── components/         # Componenti React riutilizzabili (Sidebar, ResultDisplay, ecc.)
│   ├── hooks/              # Custom Hooks di React per la logica di stato
│   ├── types/              # Definizioni TypeScript (interfacce per API e modelli)
│   └── utils/              # Funzioni helper e utility JavaScript/TypeScript
├── dockerfile              # Istruzioni per la creazione dell'immagine Docker

````

---

## Variabili d'Ambiente

Crea un file `.env.local`:

```env
NEXT_PUBLIC_BACKEND_URL=http://localhost:5000
````

Nel container Docker, questa variabile può essere impostata tramite `docker-compose`.

---

## Avvio (senza Docker)

Dalla cartella `frontend`:

```bash
npm install
npm run build
npm start
```

L'app sarà disponibile su:

```
http://localhost:3000
```

---

## 🐳 Deployment con Docker

Il frontend è containerizzato per garantire coerenza tra ambienti e per aumentare l'autonomia del frontend

### Build dell'immagine

```bash
docker build -t ophryslens-frontend .
```

### Avvio del container

```bash
docker run -p 3000:3000 \
  -e NEXT_PUBLIC_BACKEND_URL=http://localhost:5000 \
  ophryslens-frontend
```

---

## Avvio tramite Docker Compose (Raccomandato)

Dalla root del progetto:

```bash
docker-compose up --build frontend
```

Il frontend sarà disponibile su:

```
http://localhost:3000
```

---

## Flusso di Comunicazione

1. L’utente carica un’immagine
2. Il frontend invia una richiesta al backend:

   * `/inference/6class`
   * `/inference/1vsall`
3. Se Explainability attiva, invia anche:

   * `/inference/generate_explain`
   * `/inference/generate_occlusion`
4. Riceve:

   * Classe predetta
   * Probabilità
   * Heatmap in Base64
5. Renderizza:

   * Single View
   * Compare View

---

## Modalità di Analisi

* **Original (Integrated)** → Analisi sull'immagine completa
* **External (Cropped)** → Analisi sull'area rilevata dal detector
* **Compare** → Confronto tra Original vs Cropped

---
