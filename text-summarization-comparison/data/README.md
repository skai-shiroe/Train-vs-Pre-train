# Dossier Data

Ce dossier contient toutes les données du projet de résumé automatique.

## Structure

```
data/
├── raw/          # Données brutes du dataset CNN/DailyMail
├── processed/    # Données prétraitées et nettoyées
├── tokenized/    # Données tokenizées prêtes pour l'entraînement
└── cache/        # Cache des données Hugging Face
```

## Description des sous-dossiers

### raw/
Contient les données brutes téléchargées depuis Hugging Face.
- Dataset CNN/DailyMail (version 3.0.0)
- Fichiers JSON ou format Hugging Face Dataset

### processed/
Contient les données après prétraitement :
- Textes nettoyés et normalisés
- Suppression des caractères spéciaux
- Normalisation des espaces et de la ponctuation

### tokenized/
Contient les données tokenizées :
- Input IDs pour les articles
- Attention masks
- Labels pour les résumés

### cache/
Cache pour accélérer le chargement des données :
- Cache Hugging Face datasets
- Cache des tokenizers

## Notes

- Les fichiers de données ne sont pas versionnés (voir .gitignore)
- Utiliser les scripts dans `scripts/` pour télécharger et prétraiter les données
- Le dataset CNN/DailyMail est automatiquement téléchargé via Hugging Face