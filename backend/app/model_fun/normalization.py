import torch
import torchvision.transforms as T
import logging
from dotenv import dotenv_values, set_key

config = dotenv_values(".env")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# WIDTH = int(config['WIDTH'])
# HEIGHT = int(config['HEIGHT'])
# MEAN = [float(x) for x in config['MEAN'].split()]
# STD = [float(x) for x in config['STD'].split()]

# STANDARD_STATS = {
#     # '256x512': { # FULL DATASET
#     #     'mean': [0.5363, 0.5478, 0.3813],  #rgb(137, 140, 97) HEX: #898C61
#     #     'std': [0.2073, 0.2317, 0.2043]
#     # },
#     '256x512': { # BALANCED DATASET
#         'mean': [0.5364, 0.5518, 0.3866],  #rgb(136, 140, 98) HEX: #888C62
#         'std': [0.2045, 0.2296, 0.2025]
#     },
#     '128x256': {
#         'mean': [0.5363, 0.5478, 0.3814],
#         'std': [0.2060, 0.2303, 0.2028]
#     },
#     '64x128': {
#         'mean': [0.5363, 0.5479, 0.3814],
#         'std': [0.2037, 0.2281, 0.1999]
#     }
# }


def calculate_fresh_mean_std(loader):
    mean = torch.zeros(3)
    squared_sum = torch.zeros(3)
    total_pixels = 0
    total_batches = len(loader)

    for batch_idx, (images, labels, filenames) in enumerate(loader):
        batch_size, channels, height, width = images.shape
        num_pixels_in_batch = batch_size * height * width

        images = images.view(batch_size, channels, -1)
        
        mean += images.sum([0, 2])
        squared_sum += (images ** 2).sum([0, 2])
        total_pixels += num_pixels_in_batch

        # Calculate and log the progress
        progress = (batch_idx + 1) / total_batches * 100
        logging.info(f'Processing batch {batch_idx + 1}/{total_batches} ({progress:.2f}%)')

    mean /= total_pixels
    std = torch.sqrt(squared_sum / total_pixels - mean ** 2)

    logging.info(f'Mean: {mean}')
    converto_to_rgb = lambda x: (x * 255).int().tolist()
    logging.info(f'Mean RGB: {converto_to_rgb(mean)}')
    logging.info(f'Std: {std}')
    logging.info(f'Std RGB: {converto_to_rgb(std)}')
    logging.info(f'Total pixels: {total_pixels}')

    set_key('.env', 'MEAN', ' '.join([str(x.item()) for x in mean]))
    set_key('.env', 'STD', ' '.join([str(x.item()) for x in std]))
    
    mean = str(mean.tolist())
    std = str(std.tolist())

    return mean, std

def get_mean():
    return [float(x) for x in config['MEAN'].split()]

def get_std():
    return [float(x) for x in config['STD'].split()]

def converto_to_rgb(x):
    return (x * 255).int().tolist()

def denormalize_image(image, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1).to(image.device)
    std = torch.tensor(std).view(3, 1, 1).to(image.device)
    denormalized_image = image * std + mean
    return torch.clamp(denormalized_image, 0, 1)

class NormalizeImageTransform:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        return T.Normalize(self.mean, self.std)(image)