# Dossier Experiments

Ce dossier contient toutes les expériences et résultats de recherche du projet.

## Structure

```
experiments/
├── configs/      # Configurations spécifiques aux expériences
├── results/      # Résultats bruts des expériences
└── plots/        # Graphiques générés par les expériences
```

## Description des sous-dossiers

### configs/
Configurations YAML pour chaque expérience :
- Configurations d'entraînement spécifiques
- Hyperparamètres testés
- Configurations d'ablation
- Comparaisons de modèles

Format : `YYYYMMDD_experiment_name.yaml`

Exemples :
- `20240515_transformer_baseline.yaml`
- `20240515_t5_small_finetune.yaml`
- `20240520_ablation_num_layers.yaml`

### results/
Résultats bruts des expériences :
- Métriques d'évaluation (ROUGE, BLEU)
- Logs d'entraînement
- Prédictions sur le test set
- Statistiques computationnelles (temps, mémoire)

Format : JSON ou CSV pour faciliter l'analyse

### plots/
Graphiques générés pendant les expériences :
- Courbes de convergence
- Comparaisons de modèles
- Analyses d'hyperparamètres
- Visualisations d'attention (optionnel)

## Types d'expériences

### 1. Expériences de baseline
- Entraînement Transformer from scratch
- Fine-tuning T5
- Comparaison des performances de base

### 2. Expériences d'ablation
- Impact de la taille du modèle
- Impact de la taille du corpus
- Impact des hyperparamètres

### 3. Expériences comparatives
- Transformer vs T5
- Différentes tailles de T5 (small, base, large)
- Différentes tailles de corpus

## Workflow

1. Définir l'expérience dans `configs/`
2. Exécuter l'expérience via `scripts/run_ablation.py`
3. Sauvegarder les résultats dans `results/`
4. Générer les visualisations dans `plots/`
5. Documenter dans `docs/experiments.md`

## Notes

- Organiser les expériences par date et par type
- Documenter chaque expérience (protocole, résultats, conclusions)
- Sauvegarder toutes les configurations utilisées
- Comparer systématiquement les résultats