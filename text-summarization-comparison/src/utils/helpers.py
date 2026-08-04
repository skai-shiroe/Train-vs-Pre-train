"""
Fonctions utilitaires.

Ce module contient des fonctions utilitaires utilisées
à travers le projet (logging, configuration, etc.).
"""

from typing import Dict, Any
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_logging(config: Dict[str, Any]) -> None:
    """
    Configurer le logging du projet.
    
    Args:
        config: Configuration contenant les paramètres de logging
        
    TODO: Implémenter la configuration du logging
    """
    # logging.basicConfig(
    #     level=config.get("log_level", "INFO"),
    #     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    # )
    pass


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Charger un fichier de configuration YAML.
    
    Args:
        config_path: Chemin vers le fichier de configuration
        
    Returns:
        Dictionnaire contenant la configuration
        
    TODO: Implémenter le chargement de configuration
    """
    # with open(config_path, 'r') as f:
    #     config = yaml.safe_load(f)
    # return config
    pass


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fusionner deux configurations (override dans base).
    
    Args:
        base_config: Configuration de base
        override_config: Configuration à fusionner
        
    Returns:
        Configuration fusionnée
        
    TODO: Implémenter la fusion de configurations
    """
    # merged = base_config.copy()
    # for key, value in override_config.items():
    #     if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
    #         merged[key] = merge_configs(merged[key], value)
    #     else:
    #         merged[key] = value
    # return merged
    pass


def count_parameters(model: Any) -> int:
    """
    Compter le nombre de paramètres d'un modèle.
    
    Args:
        model: Modèle PyTorch
        
    Returns:
        Nombre total de paramètres
        
    TODO: Implémenter le comptage de paramètres
    """
    # return sum(p.numel() for p in model.parameters() if p.requires_grad)
    pass


def set_seed(seed: int) -> None:
    """
    Fixer les graines pour la reproductibilité.
    
    Args:
        seed: Graine à utiliser
        
    TODO: Implémenter la fixation des graines
    """
    # import random
    # import numpy as np
    # import torch
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    pass