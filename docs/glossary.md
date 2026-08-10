# Glossaire

La terminologie de cette page est imposée par le linter de prose. Un concept porte un seul nom dans tout le dépôt.

## Modèles

**Modèle from scratch**
: Transformer encodeur-décodeur implémenté composant par composant en PyTorch dans `src/models/scratch/`. Ne jamais écrire « modèle maison », « modèle fait main » ou « modèle custom ».

**Modèle pré-entraîné**
: `t5-small`, chargé depuis Hugging Face. Sert de référence.

**Zero-shot**
: Évaluation du modèle pré-entraîné sans aucun entraînement sur le corpus de travail. Une seule mesure, indépendante de la taille du corpus.

**Fine-tuning**
: Poursuite de l'entraînement du modèle pré-entraîné sur le corpus de travail.

**Champion et challenger**
: Rôles dans le Model Registry. Le champion est la version servie par défaut, le challenger la version candidate.

## Données

**Corpus de travail**
: Sous-ensemble figé de XSum, tiré avec la graine 42 : 20 000 exemples d'entraînement, 1 000 de validation, 1 000 de test. Toutes les proportions d'ablation s'y rapportent.

**Proportion**
: Fraction du corpus de travail utilisée pour l'entraînement : 10 %, 50 % ou 100 %.

**Jeu de test commun**
: Les 1 000 exemples de test, identiques pour toutes les expériences. Aucun modèle ne les voit pendant l'entraînement.

**Version du dataset**
: Empreinte calculée sur le corpus de travail, tracée dans MLflow avec chaque expérience.

## Évaluation

**ROUGE-1, ROUGE-2, ROUGE-L**
: Métriques de recouvrement entre le résumé généré et le résumé de référence, respectivement sur les unigrammes, les bigrammes et la plus longue sous-séquence commune.

**Métrique reportée**
: ROUGE-L en F-mesure. C'est l'axe des ordonnées de la courbe performance vs taille du corpus.

**Analyse qualitative**
: Sélection d'exemples générés, commentés, stockés dans `reports/results/qualitative_examples.json`.

## Statuts d'expérience

**NOT_RUN**
: Expérience définie mais jamais exécutée.

**FAILED**
: Expérience exécutée qui a échoué.

**MOCK**
: Valeur factice réservée aux tests techniques. Ne devient jamais un résultat scientifique.

## Casse imposée des noms propres

```text
PyTorch
FastAPI
Pydantic
GitHub
MLflow
Hugging Face
Docker
PostgreSQL
MinIO
MkDocs
Trivy
Bandit
OpenAPI
```
