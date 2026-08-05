# Architecture du Projet

## Vue d'ensemble

Ce document décrit l'architecture technique du projet de recherche sur la comparaison de modèles de résumé automatique.

## Architecture Globale

```
text-summarization-comparison/
├── configs/                 # Configuration centralisée (YAML)
├── data/                   # Gestion des données
├── src/                    # Code source modulaire
│   ├── data/              # Chargement et prétraitement
│   ├── models/            # Modèles Transformer et T5
│   ├── evaluation/        # Métriques d'évaluation
│   ├── experiments/       # Gestion des expériences
│   ├── utils/             # Fonctions utilitaires
│   ├── api/               # API REST (optionnel)
│   └── app/               # Interface web (optionnel)
├── scripts/               # Scripts d'automatisation
├── notebooks/             # Exploration et analyse
├── tests/                 # Tests unitaires
└── outputs/               # Résultats et modèles
```

## Architecture des Modèles

### Transformer From Scratch

```
src/models/transformer/
├── embedding.py           # Embeddings de tokens
├── positional_encoding.py # Encodage positionnel sinusoidal
├── multi_head_attention.py # Attention multi-tête
├── feed_forward.py        # Couche feed-forward
├── encoder.py             # Encodeur (couches d'attention)
├── decoder.py             # Décodeur (avec cross-attention)
├── transformer.py         # Modèle complet encodeur-décodeur
├── trainer.py             # Logique d'entraînement
└── inference.py           # Génération de résumés
```

**Flux** :
1. Input texte → Tokenization
2. Embedding + Positional Encoding
3. Encodeur (N couches d'attention)
4. Décodeur (N couches avec cross-attention)
5. Projection sur vocabulaire
6. Génération du résumé

### T5 Pré-entraîné

```
src/models/t5/
├── zero_shot.py           # Inférence sans fine-tuning
├── fine_tuning.py         # Fine-tuning du modèle
├── trainer.py             # Logique d'entraînement
└── inference.py           # Génération de résumés
```

**Flux** :
1. Charger T5 depuis Hugging Face
2. Ajouter préfixe "summarize: "
3. Fine-tuning sur CNN/DailyMail
4. Génération avec beam search

## Flux de Données

```
Dataset CNN/DailyMail
    ↓
download_dataset.py (scripts/)
    ↓
data/raw/
    ↓
preprocess_dataset.py (scripts/)
    ↓
data/processed/ → data/tokenized/
    ↓
DataLoader (src/data/)
    ↓
Modèle (src/models/)
    ↓
Entraînement (src/experiments/)
    ↓
Évaluation (src/evaluation/)
    ↓
outputs/ (checkpoints, predictions, reports)
```

## Gestion de Configuration

- **Fichiers YAML** dans `configs/`
- Configuration par défaut : `default.yaml`
- Configurations spécifiques : `transformer.yaml`, `t5.yaml`, `training.yaml`
- Fusion des configurations avec `src/utils/helpers.py`

## Technologies Utilisées

- **Deep Learning** : PyTorch 2.0+
- **NLP** : Hugging Face Transformers, Datasets
- **Évaluation** : ROUGE, BLEU
- **Visualisation** : Matplotlib, Seaborn, TensorBoard
- **Configuration** : PyYAML
- **Notebooks** : Jupyter Lab

## Points d'Extension

- Ajout de nouveaux modèles dans `src/models/`
- Ajout de métriques dans `src/evaluation/`
- Ajout d'expériences dans `experiments/`
- Ajout de scripts d'automatisation dans `scripts/`

## Notes

- Architecture modulaire pour faciliter l'extension
- Séparation claire des responsabilités
- Pas de logique métier dans les placeholders
- Prêt pour le développement complet