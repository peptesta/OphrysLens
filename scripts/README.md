# OphrysLens - Utility Scripts

Questa directory contiene script Python di supporto per il preprocessing, il testing e la gestione batch delle immagini utilizzate in OphrysLens.

Gli script sono dipendenti dal backend Flask e possono essere eseguiti localmente tramite un ambiente virtuale Python.

---

## Struttura Directory

```

.
├── batch_processor.py
├── resize_processor.py
├── test.py
├── requirements.txt

````

---

## Requisiti

- Python 3.9+ (consigliato 3.10)
- pip

Verifica la versione:

```bash
python --version
````

---

## Creazione Ambiente Virtuale (venv)

### 1. Creare l’ambiente

Dalla directory degli script:

```bash
python -m venv venv
```

Questo creerà la cartella:

```
venv/
```

---

### 2. Attivare l’ambiente

#### macOS / Linux

```bash
source venv/bin/activate
```

#### Windows (PowerShell)

```bash
venv\Scripts\Activate.ps1
```

#### Windows (CMD)

```bash
venv\Scripts\activate
```

Quando attivo, vedrai:

```
(venv) $
```

---

### 3. Installare le dipendenze

Con l’ambiente attivo:

```bash
pip install -r requirements.txt
```

---

## Utilizzo Script

### batch_processor.py

Script per elaborare i crop sulle immagini in modalità batch.

Esempio:

```bash
python batch_processor.py
```

(Configurare eventuali parametri direttamente nel file)

---

### resize_processor.py

Script per ridimensionamento massivo delle immagini.

Esempio:

```bash
python resize_processor.py
```

---

### inference_test.py

Script di test per testare e comparare le performance dei modelli.

Esempio:

```bash
python inference_test.py
```

---

## Disattivare l’Ambiente

Quando hai finito:

```bash
deactivate
```

---
