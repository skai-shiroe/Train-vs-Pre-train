"""
Module des datasets PyTorch pour le rsum automatique.

Ce module contient les classes de datasets PyTorch pour charger
et manipuler les donnes tokenisres.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import logging
import torch
from datasets import load_from_disk

logger = logging.getLogger(__name__)


class SummarizationDataset(torch.utils.data.Dataset):
    """
    Dataset PyTorch pour la tche de rsum automatique.
    
    Ce dataset charge les donnes tokenisres depuis HuggingFace Datasets
    et les convertit en tenseurs PyTorch.
    
    Args:
        data_path: Chemin vers le dossier parent (ex: notebooks/data)
        split: Nom du split (ex: transformer/train_10 ou t5/train_50)
    """
    
    def __init__(self, data_path: Path, split: str = "train"):
        """
        Initialiser le dataset de rsum automatique.
        
        Args:
            data_path: Chemin vers le dossier parent des donnes
            split: Nom du split (ex: transformer/train_10)
        """
        self.data_path = Path(data_path)
        self.split = split
        
        # Construire le chemin complet
        self.full_path = self.data_path / split
        
        logger.info(f"Chargement du dataset depuis {self.full_path}...")
        
        # Charger le dataset HuggingFace
        self.dataset = load_from_disk(str(self.full_path))
        
        logger.info(f"Dataset chargr: {len(self.dataset):,} exemples")
        
    def __len__(self) -> int:
        """
        Retourner la taille du dataset.
        
        Returns:
            Nombre d'exemples dans le dataset
        """
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Rcupérer un exemple du dataset.
        
        Args:
            idx: Index de l'exemple
            
        Returns:
            Dictionnaire contenant input_ids, attention_mask, labels
            en tenseurs PyTorch
        """
        # Rcupurer l'exemple
        example = self.dataset[idx]
        
        # Convertir en tenseurs PyTorch
        return {
            'input_ids': torch.tensor(example['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(example['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(example['labels'], dtype=torch.long)
        }


def load_dataset(data_path: Path, split: str = "train") -> SummarizationDataset:
    """
    Fonction utilitaire pour charger un dataset.
    
    Args:
        data_path: Chemin vers le dossier parent des donnes
        split: Nom du split (ex: transformer/train_10)
        
    Returns:
        Instance de SummarizationDataset
    """
    return SummarizationDataset(data_path, split)
