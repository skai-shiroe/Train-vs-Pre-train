"""
Module de chargement des donnes avec DataLoaders PyTorch.

Ce module contient les fonctions pour créer les DataLoaders
pour l'entraînement et l'évaluation.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import logging
from torch.utils.data import DataLoader

from .dataset import SummarizationDataset, load_dataset
from .collator import create_collator

logger = logging.getLogger(__name__)


def create_dataloader(
    data_path: Path,
    split: str = "train",
    model_type: str = "transformer",
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    pad_token_id: int = 0
) -> DataLoader:
    """
    Créer un DataLoader PyTorch pour le rsum automatique.
    
    Args:
        data_path: Chemin vers le dossier parent des donnes (ex: notebooks/data)
        split: Nom du split (ex: transformer/train_10 ou t5/train_10)
        model_type: Type de modrle ("transformer" ou "t5")
        batch_size: Taille du batch
        shuffle: Mlranger les donnes (True pour train, False pour eval)
        num_workers: Nombre de workers pour le chargement parallrle
        pad_token_id: ID du token de padding
        
    Returns:
        DataLoader PyTorch prrt à l'utilisation
    """
    # Charger le dataset
    dataset = load_dataset(data_path, split)
    
    # Créer le collator approprié
    collate_fn = create_collator(model_type, pad_token_id)
    
    # Créer le DataLoader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,  # Optimisation pour GPU
        drop_last=False   # Garder tous les exemples
    )
    
    logger.info(f"DataLoader créé: {split} | {len(dataset):,} exemples | batch_size={batch_size}")
    
    return dataloader


def create_dataloaders(
    data_path: Path,
    model_type: str = "transformer",
    batch_size: int = 8,
    num_workers: int = 0,
    pad_token_id: int = 0
) -> Dict[str, DataLoader]:
    """
    Créer tous les DataLoaders (train, validation, test).
    
    Args:
        data_path: Chemin vers le dossier parent des donnes
        model_type: Type de modrle ("transformer" ou "t5")
        batch_size: Taille du batch
        num_workers: Nombre de workers
        pad_token_id: ID du token de padding
        
    Returns:
        Dictionnaire de DataLoaders {split: dataloader}
    """
    dataloaders = {}
    
    # Train loader (avec shuffle)
    dataloaders['train'] = create_dataloader(
        data_path=data_path,
        split=f"{model_type}/train_10",  # Utiliser train_10 par défaut
        model_type=model_type,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pad_token_id=pad_token_id
    )
    
    # Validation loader (sans shuffle)
    dataloaders['validation'] = create_dataloader(
        data_path=data_path,
        split=f"{model_type}/validation",
        model_type=model_type,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pad_token_id=pad_token_id
    )
    
    # Test loader (sans shuffle)
    dataloaders['test'] = create_dataloader(
        data_path=data_path,
        split=f"{model_type}/test",
        model_type=model_type,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pad_token_id=pad_token_id
    )
    
    return dataloaders
