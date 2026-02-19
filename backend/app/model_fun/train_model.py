import torch
from torch.utils.data import Dataset
import logging
import glob
from dotenv import dotenv_values
import os

config = dotenv_values(".env")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ChunkDataset(Dataset):
    """
    Dataset che carica i chunk uno alla volta in memoria.
    Cerca chunk con pattern: {base_path}_chunk_*.pt
    """
    def __init__(self, base_path, chunk_size=3000):
        self.base_path = base_path
        self.chunk_files = self._find_chunks()
        
        if not self.chunk_files:
            raise FileNotFoundError(f"Nessun chunk trovato per: {base_path}. Pattern cercato: {base_path}_chunk_*.pt")
        
        self.chunk_size = chunk_size
        
        # Carica il primo chunk per ottenere info
        first_chunk = torch.load(self.chunk_files[0], weights_only=False)
        self.classes = first_chunk['classes']
        
        # Calcola totale immagini
        self.total_images = 0
        for chunk_file in self.chunk_files:
            chunk_data = torch.load(chunk_file, weights_only=False)
            self.total_images += len(chunk_data['images'])
            del chunk_data
        
        logging.info(f"Trovati {len(self.chunk_files)} chunk, totale {self.total_images} immagini")
        
        self.current_chunk_idx = -1
        self.current_chunk_data = None
        
    def _find_chunks(self):
        """Trova tutti i chunk: base_path_chunk_*.pt"""
        base_no_ext = self.base_path.replace('.pt', '')
        chunk_pattern = f"{base_no_ext}_chunk_*.pt"
        
        chunk_files = sorted(glob.glob(chunk_pattern))
        
        if chunk_files:
            return chunk_files
        
        # Se non ci sono chunk ma esiste il file base, usa quello
        if os.path.exists(self.base_path):
            return [self.base_path]
        
        raise FileNotFoundError(f"Nessun file trovato: {chunk_pattern} o {self.base_path}")
    
    def __len__(self):
        return self.total_images
    
    def _load_chunk(self, chunk_idx):
        """Carica un chunk specifico e libera il precedente"""
        if self.current_chunk_idx == chunk_idx and self.current_chunk_data is not None:
            return
        
        # Libera memoria del chunk precedente
        if self.current_chunk_data is not None:
            del self.current_chunk_data
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logging.info(f"Caricamento chunk {chunk_idx + 1}/{len(self.chunk_files)}: {self.chunk_files[chunk_idx]}")
        chunk_data = torch.load(self.chunk_files[chunk_idx], weights_only=False)
        
        self.current_chunk_data = {
            'images': chunk_data['images'],
            'labels': chunk_data['labels'],
            'filenames': chunk_data['filenames']
        }
        self.current_chunk_idx = chunk_idx
        
        del chunk_data
    
    def __getitem__(self, idx):
        # Trova quale chunk contiene questo indice
        chunk_idx = idx // self.chunk_size
        local_idx = idx % self.chunk_size
        
        # Assicurati che l'indice del chunk sia valido
        chunk_idx = min(chunk_idx, len(self.chunk_files) - 1)
        
        # Carica il chunk se necessario
        self._load_chunk(chunk_idx)
        
        # Calcola l'indice locale corretto per l'ultimo chunk
        if chunk_idx == len(self.chunk_files) - 1 and chunk_idx > 0:
            images_in_prev_chunks = chunk_idx * self.chunk_size
            local_idx = idx - images_in_prev_chunks
        
        # Assicurati che l'indice locale sia valido
        local_idx = min(local_idx, len(self.current_chunk_data['images']) - 1)
        
        return (
            self.current_chunk_data['images'][local_idx],
            self.current_chunk_data['labels'][local_idx],
            self.current_chunk_data['filenames'][local_idx]
        )


 