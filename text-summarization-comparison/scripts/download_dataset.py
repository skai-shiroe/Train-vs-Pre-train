#!/usr/bin/env python3
"""
Script de téléchargement du dataset CNN/DailyMail.

Ce script télécharge le dataset CNN/DailyMail depuis Hugging Face
et le sauvegarde dans le dossier data/raw/.

Usage:
    python scripts/download_dataset.py --config configs/default.yaml
"""

import argparse
from pathlib import Path
from typing import Dict, Any


def main(config: Dict[str, Any]) -> None:
    """
    Fonction principale de téléchargement du dataset.
    
    Args:
        config: Configuration du projet
        
    TODO: Implémenter le téléchargement du dataset
    """
    print("Téléchargement du dataset CNN/DailyMail...")
    
    # TODO: Implémenter la logique de téléchargement
    # - Charger la configuration
    # - Créer le dossier de sortie si nécessaire
    # - Télécharger le dataset depuis Hugging Face
    # - Sauvegarder les données brutes
    # - Afficher les statistiques du dataset
    
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Télécharger le dataset CNN/DailyMail")
    parser.add_argument("--config", type=str, default="configs/default.yaml", 
                       help="Chemin vers le fichier de configuration")
    args = parser.parse_args()
    
    # TODO: Charger la configuration
    # config = load_config(args.config)
    
    main(config={})