"""
Module des data collators pour le rsum automatique.

Ce module contient les fonctions de collation pour grouper
les exemples en batches avec padding dynamique.
"""

from typing import Dict, Any, List
import logging
import torch

logger = logging.getLogger(__name__)


def transformer_collator(batch: List[Dict[str, Any]], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """
    Collator personnalisr pour le Transformer from scratch (SentencePiece).
    
    Args:
        batch: Liste d'exemples (chaque exemple est un dict avec input_ids, attention_mask, labels)
        pad_token_id: ID du token de padding (0 pour SentencePiece)
        
    Returns:
        Batch collatr avec padding dynamique
    """
    # Sparer les champs et convertir en listes si nressaire
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    
    for example in batch:
        # Convertir les tenseurs en listes si nressaire
        input_ids = example['input_ids']
        attention_mask = example['attention_mask']
        labels = example['labels']
        
        # Si c'est un tenseur, convertir en liste
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        if isinstance(attention_mask, torch.Tensor):
            attention_mask = attention_mask.tolist()
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()
        
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)
    
    # Trouver la longueur maximale dans le batch
    max_input_len = max(len(ids) for ids in input_ids_list)
    max_label_len = max(len(labels) for labels in labels_list)
    
    # Padder les sries input_ids et attention_mask
    padded_input_ids = []
    padded_attention_mask = []
    
    for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
        # Calculer le padding nressaire
        pad_len = max_input_len - len(input_ids)
        
        # Ajouter le padding
        padded_input_ids.append(input_ids + [pad_token_id] * pad_len)
        padded_attention_mask.append(attention_mask + [0] * pad_len)
    
    # Padder les labels
    padded_labels = []
    
    for labels in labels_list:
        pad_len = max_label_len - len(labels)
        # Pad avec -100 (ignorr par la loss dans PyTorch)
        padded_labels.append(labels + [-100] * pad_len)
    
    # Convertir en tenseurs PyTorch
    return {
        'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(padded_attention_mask, dtype=torch.long),
        'labels': torch.tensor(padded_labels, dtype=torch.long)
    }


def t5_collator(batch: List[Dict[str, Any]], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """
    Collator pour T5.
    
    Args:
        batch: Liste d'exemples
        pad_token_id: ID du token de padding
        
    Returns:
        Batch collatr avec padding dynamique
    """
    # Meme logique que transformer_collator
    return transformer_collator(batch, pad_token_id=pad_token_id)


def create_collator(model_type: str = "transformer", pad_token_id: int = 0):
    """
    Fabrique de collator selon le type de modrle.
    
    Args:
        model_type: Type de modrle ("transformer" ou "t5")
        pad_token_id: ID du token de padding
        
    Returns:
        Fonction de collation appropriee
    """
    if model_type == "transformer":
        return lambda batch: transformer_collator(batch, pad_token_id)
    elif model_type == "t5":
        return lambda batch: t5_collator(batch, pad_token_id)
    else:
        raise ValueError(f"Type de modrle inconnu: {model_type}. Utiliser 'transformer' ou 't5'.")
