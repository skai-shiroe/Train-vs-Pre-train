# Dossier Outputs

Ce dossier contient tous les résultats et sorties générés pendant le projet.

## Structure

```
outputs/
├── checkpoints/         # Modèles entraînés sauvegardés
│   ├── transformer/    # Checkpoints du Transformer from scratch
│   └── t5/             # Checkpoints du modèle T5 fine-tuné
├── predictions/        # Prédictions générées par les modèles
├── figures/            # Graphiques et visualisations
├── reports/            # Rapports d'évaluation et résultats
│   ├── figures/        # Figures pour les rapports
│   └── tables/         # Tableaux de résultats
└── metrics/            # Métriques d'évaluation sauvegardées
```

## Description des sous-dossiers

### checkpoints/
Contient les modèles entraînés sauvegardés à différentes étapes :

**transformer/**
- Modèle Transformer from scratch
- Checkpoints après chaque epoch
- Meilleur modèle basé sur la validation
- Fichiers .pt (PyTorch)

**t5/**
- Modèle T5 fine-tuné
- Checkpoints du fine-tuning
- Meilleur modèle
- Fichiers .pt (PyTorch)

### predictions/
Contient les prédictions générées :
- Résumés générés par le Transformer
- Résumés générés par T5
- Résumés de référence
- Comparaisons côte à côte

### figures/
Visualisations générées :
- Courbes d'apprentissage (loss, ROUGE)
- Comparaisons de modèles
- Distributions des métriques
- Exemples de résumés

### reports/
Rapports d'évaluation complets :
- Rapports HTML/PDF
- Tableaux de résultats
- Figures et graphiques
- Analyses statistiques

### metrics/
Métriques sauvegardées :
- Scores ROUGE par modèle
- Scores BLEU
- Temps d'inférence
- Statistiques de génération

## Notes

- Les fichiers de sortie ne sont pas versionnés (voir .gitignore)
- Organiser les résultats par date et par expérience
- Utiliser des noms de fichiers descriptifs