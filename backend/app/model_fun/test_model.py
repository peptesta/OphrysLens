import logging
import torch
from dotenv import dotenv_values
from torch import nn
from torchvision import models


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

config = dotenv_values(".env")

def modelLoader(SIXCLASS_MODEL_PATH, CLASS_SIZE, device):
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, CLASS_SIZE)
    model_dict = torch.load(SIXCLASS_MODEL_PATH, weights_only=False, map_location=device)
    model.load_state_dict(model_dict['model'])
    model = model.to(device)
    model.eval()
    return model

def deviceLoader():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prima di eseguire questo è codice e quindi visualizzare l'interfaccia interattiva è necessario configurare il file .env con i parametri corretti
# e lanciare il comando python app/test_model.py --generate-results
# o in alternativa modificare il seguente codice e inserire staticamente i parametri di ambiente
if __name__ == '__main__':
    CLASS_NAMES = eval(config['CLASS_NAMES'])
    CLASS_SIZE = len(CLASS_NAMES)

    PROCESSED_DATA_TEST_PATH = config['PROCESSED_DATA_TEST_PATH']
    SIXCLASS_MODEL_PATH = config['SIXCLASS_MODEL_PATH']
    BATCH_SIZE = int(config['BATCH_SIZE_TEST'])
    NUM_WORKERS = int(config['NUM_WORKERS_TEST'])
    DESTINATION_PATH = config['DESTINATION_PATH']
    SLIDING_WINDOW_SIZE = int(config['SLIDING_WINDOW_SIZE'])
    STRIDE = int(config['SLIDING_WINDOW_STRIDE'])