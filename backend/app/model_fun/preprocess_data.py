import logging
import torchvision.transforms as T
from dotenv import dotenv_values

import app.model_fun.normalization as normalization


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

config = dotenv_values(".env")

# Questa funzione permette di applicare ulteriori trasformazioni alle immagini
# In questo caso le immagini vengono convertite in tensori e normalizzate
def getTransforms(width, height, shouldNormalize, mean, std):
    transforms_list = [T.Resize((height, width)), T.ToTensor()]
    if shouldNormalize:
        mean = normalization.get_mean()
        std = normalization.get_std()
        transforms_list.append(normalization.NormalizeImageTransform(mean, std))
    return T.Compose(transforms_list)
