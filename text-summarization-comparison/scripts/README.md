# Dossier Scripts

Ce dossier contient tous les scripts d'automatisation et d'exécution du projet.

## Structure

```
scripts/
├── download_dataset.py      # Téléchargement du dataset CNN/DailyMail
├── run_eda.py               # Analyse exploratoire des données
├── preprocess_dataset.py    # Prétraitement et tokenization
├── train_transformer.py     # Entraînement du Transformer from scratch
├── train_t5.py              # Fine-tuning du modèle T5
├── evaluate_models.py       # Évaluation des modèles entraînés
├── run_ablation.py          # Exécution des expériences d'ablation
├── launch_api.py            # Lancement de l'API REST
└── launch_streamlit.py      # Lancement de l'interface Streamlit
```

## Description des scripts

### Scripts de données
- **download_dataset.py** : Télécharge le dataset CNN/DailyMail depuis Hugging Face
- **run_eda.py** : Effectue l'analyse exploratoire (statistiques, visualisations)
- **preprocess_dataset.py** : Nettoie et tokenize les données

### Scripts d'entraînement
- **train_transformer.py** : Entraîne le Transformer from scratch
- **train_t5.py** : Fine-tune le modèle T5 pré-entraîné

### Scripts d'évaluation
- **evaluate_models.py** : Évalue et compare les modèles entraînés
- **run_ablation.py** : Lance les expériences d'ablation

### Scripts de déploiement
- **launch_api.py** : Démarre l'API REST pour les prédictions
- **launch_streamlit.py** : Démarre l'interface web Streamlit

## Utilisation

Tous les scripts suivent la même structure :

```bash
# Exécuter un script
python scripts/nom_du_script.py --config configs/default.yaml

# Avec arguments spécifiques
python scripts/train_transformer.py --model transformer --epochs 20
```

## Notes

- Tous les scripts acceptent des arguments en ligne de commande
- La configuration par défaut est dans `configs/default.yaml`
- Les logs sont sauvegardés dans `logs/`
- Les résultats sont sauvegardés dans `outputs/`