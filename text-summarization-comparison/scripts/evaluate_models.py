#!/usr/bin/env python3
"""
Script d'évaluation des modèles entraînés.

Ce script évalue et compare les performances des modèles Transformer et T5
sur le test set en utilisant les métriques ROUGE et BLEU.

Usage:
    python scripts/evaluate_models.py --config configs/default.yaml
"""

import argparse
from typing import Dict, Any, List


def main(config: Dict[str, Any]) -> None:
    """
    Fonction principale d'évaluation des modèles.
    
    Args:
        config: Configuration du projet
        
    TODO: Implémenter l'évaluation des modèles
    """
    print("Évaluation des modèles...")
    
    # TODO: Implémenter la logique d'évaluation
    # - Charger les modèles entraînés (Transformer et T5)
    # - Charger le test set
    # - Générer les prédictions pour chaque modèle
    # - Calculer les métriques (ROUGE-1, ROUGE-2, ROUGE-L, BLEU)
    # - Comparer les performances
    # - Sauvegarder les résultats dans outputs/metrics/
    # - Générer des visualisations dans outputs/figures/
    # - Créer un rapport d'évaluation dans reports/
    
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluer les modèles entraînés")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Chemin vers le fichier de configuration")
    parser.add_argument("--models", nargs="+", default=["transformer", "t5"],
                       help="Modèles à évaluer")
    args = parser.parse_args()
    
    # TODO: Charger la configuration
    # config = load_config(args.config)
    
    main(config={})