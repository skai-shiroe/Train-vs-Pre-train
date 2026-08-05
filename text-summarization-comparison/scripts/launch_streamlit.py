#!/usr/bin/env python3
"""
Script de lancement de l'interface Streamlit.

Ce script démarre l'interface web Streamlit pour interagir
avec les modèles de résumé automatique de manière interactive.

Usage:
    python scripts/launch_streamlit.py --config configs/default.yaml
"""

import argparse
from typing import Dict, Any


def main(config: Dict[str, Any]) -> None:
    """
    Fonction principale de lancement de Streamlit.
    
    Args:
        config: Configuration du projet
        
    TODO: Implémenter le lancement de Streamlit
    """
    print("Lancement de l'interface Streamlit...")
    
    # TODO: Implémenter la logique Streamlit
    # - Charger les modèles entraînés (Transformer et T5)
    # - Créer l'interface utilisateur avec :
    #   * Zone de texte pour entrer un article
    #   * Sélection du modèle (Transformer ou T5)
    #   * Bouton pour générer le résumé
    #   * Affichage du résumé généré
    #   * Comparaison côte à côte des deux modèles
    # - Ajouter des visualisations (attention, scores)
    # - Lancer l'application Streamlit
    
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lancer l'interface Streamlit")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Chemin vers le fichier de configuration")
    parser.add_argument("--port", type=int, default=8501,
                       help="Port de l'application Streamlit")
    args = parser.parse_args()
    
    # TODO: Charger la configuration
    # config = load_config(args.config)
    
    main(config={})