# app/model_state.py

# Variabili globali per mantenere lo stato dei modelli nel ciclo di vita dell'app
models_6class_dict = {}  # Dizionario { "nome_file.pt": istanza_modello }
onevall_models = None    # Lista dei modelli per l'ensemble
device = None
CLASS_NAMES = ["O. exaltata", "O. garganica", "O. incubacea", "O. majellensis", "O. sphegodes", "O. sphegodes_Palena"]

def load_and_set_models(resources):
    """Assegna le risorse caricate dal loader al dizionario globale."""
    global models_6class_dict, onevall_models, device
    # Carichiamo il dizionario dei modelli 6class
    models_6class_dict = resources.get('models_6class', {})
    onevall_models = resources.get('onevall_models', [])
    device = resources.get('device')

def get_6class_model_by_name(model_name: str):
    """
    Ritorna il modello specifico richiesto dal frontend.
    Se il nome non esiste o è nullo, ritorna il primo modello disponibile (default).
    """
    global models_6class_dict, device
    
    if not models_6class_dict:
        return None, device

    # Se non viene specificato un nome, prendiamo il primo modello caricato
    if not model_name or model_name not in models_6class_dict:
        first_key = list(models_6class_dict.keys())[0]
        return models_6class_dict[first_key], device
    
    return models_6class_dict[model_name], device

def get_1vsall_resources():
    """Ritorna solo le risorse necessarie per la strategia 1-vs-All."""
    global onevall_models, device, CLASS_NAMES
    return onevall_models, device, CLASS_NAMES

def get_all_6class_names():
    """Ritorna la lista dei nomi dei modelli disponibili per il frontend."""
    global models_6class_dict
    return list(models_6class_dict.keys())