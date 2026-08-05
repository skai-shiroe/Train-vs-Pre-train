# Dossier Reports

Ce dossier contient les rapports d'évaluation et d'analyse du projet.

## Structure

```
reports/
├── figures/    # Figures et graphiques pour les rapports
└── tables/     # Tableaux de résultats et données tabulaires
```

## Description des sous-dossiers

### figures/
Contient toutes les figures et visualisations :
- Courbes d'apprentissage (loss, métriques)
- Comparaisons entre modèles (Transformer vs T5)
- Distributions des scores ROUGE/BLEU
- Exemples de résumés générés (avec visualisation)
- Graphiques d'analyse d'impact de la taille du corpus
- Histogrammes et boxplots

Formats : PNG, PDF, SVG (haute résolution pour publications)

### tables/
Contient les tableaux de résultats :
- Tableaux comparatifs des performances
- Résultats statistiques (moyenne, écart-type, intervalles de confiance)
- Tableaux d'hyperparamètres
- Résultats par taille de corpus
- Analyses de significativité statistique

Formats : CSV, LaTeX, Markdown

## Types de rapports

### Rapports d'évaluation
- Rapport de performance par modèle
- Comparaison Transformer vs T5
- Analyse par taille de corpus
- Cas d'échec et erreurs

### Rapports d'expériences
- Résultats d'ablation
- Comparaisons d'hyperparamètres
- Analyses statistiques

### Rapport final de Master
- Synthèse complète
- Résultats et analyses
- Conclusions
- Perspectives

## Conventions

- Nommage : `YYYYMMDD_type_rapport.ext`
- Utiliser des noms descriptifs
- Inclure la date et le type d'expérience
- Versionner les rapports importants