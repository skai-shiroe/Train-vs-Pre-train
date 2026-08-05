# Plan de Projet

## Informations Générales

- **Titre** : Automatic Text Summarization: Transformer From Scratch vs Pre-trained T5
- **Niveau** : Master en NLP
- **Durée** : 6 mois
- **Date de début** : [À compléter]
- **Date de fin** : [À compléter]

## Objectifs

### Objectif Principal
Comparer deux approches de résumé automatique :
1. Transformer encodeur-décodeur entraîné from scratch
2. Modèle T5 pré-entraîné fine-tuné

### Objectifs Spécifiques
- [ ] Implémenter un Transformer from scratch avec attention multi-tête
- [ ] Fine-tuner T5 sur le dataset CNN/DailyMail
- [ ] Évaluer les deux modèles avec ROUGE et BLEU
- [ ] Analyser l'impact de la taille du corpus d'entraînement
- [ ] Comparer les performances et la qualité des résumés
- [ ] Rédiger le mémoire de Master

## Timeline et Milestones

### Phase 1 : Préparation et Setup (Semaine 1-2)
- [x] Mise en place de l'architecture du projet
- [ ] Installation de l'environnement de développement
- [ ] Téléchargement et exploration du dataset CNN/DailyMail
- [ ] Revue de littérature (articles de référence)

**Livrable** : Architecture du projet fonctionnelle

### Phase 2 : Implémentation du Transformer (Semaine 3-6)
- [ ] Implémentation des composants de base (embeddings, attention)
- [ ] Construction de l'encodeur et du décodeur
- [ ] Assemblage du modèle complet
- [ ] Implémentation de la génération (greedy, beam search)
- [ ] Tests unitaires

**Livrable** : Transformer from scratch fonctionnel

### Phase 3 : Fine-tuning de T5 (Semaine 7-8)
- [ ] Intégration de T5 depuis Hugging Face
- [ ] Prétraitement adapté à T5
- [ ] Fine-tuning sur CNN/DailyMail
- [ ] Optimisation des hyperparamètres

**Livrable** : Modèle T5 fine-tuné

### Phase 4 : Évaluation et Comparaison (Semaine 9-10)
- [ ] Implémentation des métriques (ROUGE, BLEU)
- [ ] Évaluation sur le test set
- [ ] Comparaison quantitative des modèles
- [ ] Analyse qualitative des résumés

**Livrable** : Résultats d'évaluation complets

### Phase 5 : Expériences et Analyses (Semaine 11-12)
- [ ] Expériences d'ablation
- [ ] Analyse de l'impact de la taille du corpus
- [ ] Visualisations et graphiques
- [ ] Analyses statistiques

**Livrable** : Jeu complet d'expériences

### Phase 6 : Rédaction et Finalisation (Semaine 13-14)
- [ ] Rédaction du mémoire
- [ ] Création des visualisations finales
- [ ] Préparation de la soutenance
- [ ] Documentation du code

**Livrable** : Mémoire de Master et présentation

## Ressources Nécessaires

### Matériel
- GPU NVIDIA (minimum 8GB VRAM)
- CPU multi-core pour le prétraitement
- Stockage : 50GB minimum

### Logiciels
- Python 3.11+
- PyTorch 2.0+
- Hugging Face Transformers
- CUDA (si GPU NVIDIA)

### Données
- Dataset CNN/DailyMail (version 3.0.0)
- ~300k articles avec résumés

## Risques et Mitigation

| Risque | Impact | Probabilité | Mitigation |
|--------|--------|-------------|------------|
| Temps d'entraînement trop long | Élevé | Moyen | Utiliser des subsets du dataset, optimiser le code |
| Problèmes de mémoire GPU | Élevé | Faible | Réduire batch size, utiliser gradient checkpointing |
| Résultats insuffisants | Moyen | Faible | Ajuster hyperparamètres, augmenter le temps d'entraînement |
| Retard dans le développement | Moyen | Moyen | Planifier des buffers, prioriser les fonctionnalités core |

## Tâches par Semaine

### Semaine 1-2 : Setup
- [x] Créer l'architecture du projet
- [ ] Installer les dépendances
- [ ] Configurer l'environnement
- [ ] Télécharger le dataset
- [ ] Faire l'EDA

### Semaine 3-4 : Transformer - Base
- [ ] Implémenter embeddings
- [ ] Implémenter attention multi-tête
- [ ] Implémenter positional encoding
- [ ] Implémenter feed-forward

### Semaine 5-6 : Transformer - Complet
- [ ] Implémenter encodeur
- [ ] Implémenter décodeur
- [ ] Assembler le modèle complet
- [ ] Implémenter l'entraînement
- [ ] Tester sur un petit subset

### Semaine 7-8 : T5
- [ ] Intégrer T5
- [ ] Adapter le prétraitement
- [ ] Fine-tuning
- [ ] Optimisation

### Semaine 9-10 : Évaluation
- [ ] Implémenter métriques
- [ ] Évaluer les deux modèles
- [ ] Comparer les résultats
- [ ] Analyser les erreurs

### Semaine 11-12 : Expériences
- [ ] Concevoir les expériences
- [ ] Exécuter les ablations
- [ ] Analyser l'impact de la taille du corpus
- [ ] Générer les visualisations

### Semaine 13-14 : Finalisation
- [ ] Rédiger le mémoire
- [ ] Préparer la soutenance
- [ ] Finaliser le code
- [ ] Soumettre

## Points de Contrôle

- **Fin Semaine 2** : Setup complet, dataset prêt
- **Fin Semaine 6** : Transformer fonctionnel
- **Fin Semaine 8** : T5 fine-tuné
- **Fin Semaine 10** : Évaluations complètes
- **Fin Semaine 12** : Expériences terminées
- **Fin Semaine 14** : Projet finalisé

## Notes

- Ajuster la timeline selon les progrès
- Documenter régulièrement les avancées
- Faire des backups réguliers du code et des résultats
- Prévoir du temps pour les imprévus