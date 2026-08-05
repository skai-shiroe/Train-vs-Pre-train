# Documentation des Expériences

Ce document répertorie toutes les expériences réalisées dans le cadre du projet.

## Format de Documentation

Pour chaque expérience, documenter :
- **Objectif** : Ce que l'expérience teste
- **Configuration** : Fichier YAML utilisé
- **Résultats** : Métriques obtenues
- **Analyse** : Interprétation des résultats
- **Conclusion** : Leçons apprises

## Liste des Expériences

### Expérience 1 : Baseline Transformer
- **Date** : [À compléter]
- **Objectif** : Entraîner un Transformer from scratch sur le dataset complet
- **Configuration** : `experiments/configs/transformer_baseline.yaml`
- **Résultats** :
  - ROUGE-1 : [À compléter]
  - ROUGE-2 : [À compléter]
  - ROUGE-L : [À compléter]
  - BLEU : [À compléter]
- **Analyse** : [À compléter]
- **Conclusion** : [À compléter]

### Expérience 2 : Baseline T5
- **Date** : [À compléter]
- **Objectif** : Fine-tuning de T5-small sur le dataset complet
- **Configuration** : `experiments/configs/t5_baseline.yaml`
- **Résultats** :
  - ROUGE-1 : [À compléter]
  - ROUGE-2 : [À compléter]
  - ROUGE-L : [À compléter]
  - BLEU : [À compléter]
- **Analyse** : [À compléter]
- **Conclusion** : [À compléter]

### Expérience 3 : Comparaison Transformer vs T5
- **Date** : [À compléter]
- **Objectif** : Comparer les performances des deux approches
- **Configuration** : `experiments/configs/comparison.yaml`
- **Résultats** : [À compléter]
- **Analyse** : [À compléter]
- **Conclusion** : [À compléter]

### Expérience 4 : Impact de la Taille du Corpus (Transformer)
- **Date** : [À compléter]
- **Objectif** : Évaluer l'impact de la taille du corpus d'entraînement sur le Transformer
- **Configuration** : `experiments/configs/ablation_corpus_size_transformer.yaml`
- **Tailles testées** : 1%, 5%, 10%, 25%, 50%, 100%
- **Résultats** : [À compléter]
- **Analyse** : [À compléter]
- **Conclusion** : [À compléter]

### Expérience 5 : Impact de la Taille du Corpus (T5)
- **Date** : [À compléter]
- **Objectif** : Évaluer l'impact de la taille du corpus d'entraînement sur T5
- **Configuration** : `experiments/configs/ablation_corpus_size_t5.yaml`
- **Tailles testées** : 1%, 5%, 10%, 25%, 50%, 100%
- **Résultats** : [À compléter]
- **Analyse** : [À compléter]
- **Conclusion** : [À compléter]

### Expérience 6 : Ablation - Nombre de Couches (Transformer)
- **Date** : [À compléter]
- **Objectif** : Tester l'impact du nombre de couches encodeur/décodeur
- **Configuration** : `experiments/configs/ablation_num_layers.yaml`
- **Variantes** : 2, 4, 6, 8 couches
- **Résultats** : [À compléter]
- **Analyse** : [À compléter]
- **Conclusion** : [À compléter]

### Expérience 7 : Ablation - Dimension du Modèle (Transformer)
- **Date** : [À compléter]
- **Objectif** : Tester l'impact de la dimension d_model
- **Configuration** : `experiments/configs/ablation_d_model.yaml`
- **Variantes** : 256, 512, 768, 1024
- **Résultats** : [À compléter]
- **Analyse** : [À compléter]
- **Conclusion** : [À compléter]

### Expérience 8 : Ablation - Nombre de Têtes d'Attention
- **Date** : [À compléter]
- **Objectif** : Tester l'impact du nombre de têtes d'attention
- **Configuration** : `experiments/configs/ablation_num_heads.yaml`
- **Variantes** : 4, 8, 12, 16
- **Résultats** : [À compléter]
- **Analyse** : [À compléter]
- **Conclusion** : [À compléter]

### Expérience 9 : T5 - Comparaison des Tailles
- **Date** : [À compléter]
- **Objectif** : Comparer T5-small, T5-base, T5-large
- **Configuration** : `experiments/configs/t5_size_comparison.yaml`
- **Modèles** : t5-small, t5-base, t5-large
- **Résultats** : [À compléter]
- **Analyse** : [À compléter]
- **Conclusion** : [À compléter]

### Expérience 10 : Analyse Qualitative
- **Date** : [À compléter]
- **Objectif** : Analyse qualitative des résumés générés
- **Configuration** : N/A
- **Méthode** : Évaluation humaine de 100 exemples
- **Résultats** : [À compléter]
- **Analyse** : [À compléter]
- **Conclusion** : [À compléter]

## Protocole Expérimental Standard

### Pour chaque expérience :
1. **Préparation** :
   - Charger la configuration depuis `experiments/configs/`
   - Préparer le dataset selon la taille spécifiée
   - Initialiser les seeds pour la reproductibilité

2. **Entraînement** :
   - Entraîner le modèle avec la configuration donnée
   - Sauvegarder les checkpoints dans `outputs/checkpoints/`
   - Logger les métriques dans `logs/training/`

3. **Évaluation** :
   - Évaluer sur le test set
   - Calculer ROUGE-1, ROUGE-2, ROUGE-L, BLEU
   - Sauvegarder les résultats dans `experiments/results/`

4. **Analyse** :
   - Générer les visualisations dans `experiments/plots/`
   - Documenter les résultats dans ce fichier
   - Mettre à jour `reports/` si nécessaire

## Résultats Synthétiques

### Tableau Comparatif Global

| Expérience | Modèle | Taille Corpus | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU |
|------------|--------|---------------|---------|---------|---------|------|
| Exp 1 | Transformer | 100% | - | - | - | - |
| Exp 2 | T5-small | 100% | - | - | - | - |
| Exp 3 | Comparison | 100% | - | - | - | - |
| Exp 4 | Transformer | Variable | - | - | - | - |
| Exp 5 | T5 | Variable | - | - | - | - |

## Notes

- Mettre à jour ce document après chaque expérience
- Lier vers les résultats détaillés dans `experiments/results/`
- Conserver toutes les configurations utilisées
- Documenter les échecs et les leçons apprises