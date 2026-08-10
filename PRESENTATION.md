# Syntra, présentation du travail

Le détail scientifique est dans [le rapport](RAPPORT.md), le détail technique dans le [README](README.md).

## 1. Le projet en une phrase

Syntra compare deux façons d'apprendre à résumer un article de presse : un Transformer écrit à la main, qui part de zéro, et un modèle déjà pré-entraîné qu'on adapte à la tâche. Les deux sont mesurés sur exactement les mêmes textes.

## 2. La question posée

> À partir de quelle quantité de données un modèle entraîné de zéro rattrape-t-il un modèle pré-entraîné ?

C'est une question de coût. Un modèle pré-entraîné est gratuit à télécharger mais opaque. Un modèle écrit soi-même est transparent mais réclame des données et du calcul. Le projet chiffre l'écart au lieu de le supposer.

## 3. Ce qui a été construit

La chaîne complète, du texte brut à l'API qui répond.

```text
XSum ──> préparation ──> entraînement ──> évaluation ──> MLflow ──> API
         corpus figé     2 familles       ROUGE          runs       /predict
         graine 42       de modèles       + bootstrap    tracés     /compare
```

Quatre briques :

| Brique | Contenu |
| --- | --- |
| Données | Téléchargement, validation, tirage figé, découpage en tokens |
| Modèles | Transformer encodeur-décodeur PyTorch écrit composant par composant, plus `t5-small` |
| Expériences | Entraînement, évaluation, ablations, traçabilité MLflow, registre de modèles |
| Service | API FastAPI, images Docker, pipeline GitHub Actions en six étapes, documentation publiée |

## 4. Le corpus

XSum, des articles de la BBC avec un résumé d'une phrase écrit par un journaliste.

Le corpus de travail est un tirage figé sous graine 42 : 20 000 exemples d'entraînement, 1 000 de validation, 1 000 de test. Le « 100 % » du projet désigne toujours ces 20 000 exemples, jamais les 204 045 de XSum complet. La convention est rappelée sur chaque tableau et chaque figure.

Deux propriétés comptent pour la suite :

- la compression est extrême, le résumé médian fait 7,2 % de son document, donc recopier ne suffit pas ;
- les documents sont coupés à 512 tokens, ce qui écarte une part du texte, mais la coupe est identique pour les deux modèles, donc la comparaison reste juste.

## 5. Les deux modèles

**Le Transformer from scratch**, 15,6 M de paramètres, écrit pièce par pièce en PyTorch : attention, masques, encodage de position, encodeur, décodeur. Rien n'est repris d'une bibliothèque de haut niveau. La [section 2 du rapport](RAPPORT.md#2-le-transformer-from-scratch) le détaille bloc par bloc, formules comprises.

**`t5-small`**, 60,5 M de paramètres, déjà pré-entraîné sur environ 34 milliards de tokens. Il est mesuré deux fois : tel quel (zero-shot), puis adapté à la tâche (fine-tuné).

Les deux partagent le même tokenizer T5 et le même jeu de test. Sans ce partage, un écart de score pourrait venir du découpage des mots plutôt que du modèle.

## 6. Comment les scores sont mesurés

La métrique est ROUGE, qui compte les mots et les suites de mots communs entre le résumé produit et le résumé de référence. Le score reporté est ROUGE-L.

Chaque score porte un intervalle de confiance à 95 % calculé par bootstrap. Un écart n'est annoncé que si les intervalles ne se recouvrent pas.

## 7. Les résultats

Neuf expériences exécutées le 8 août 2026, sur GPU portable RTX 5060, pour 125 minutes de calcul cumulé.

| Modèle | Corpus | ROUGE-L | IC 95 % |
| --- | --- | --- | --- |
| `t5-small` fine-tuné | 100 % | **0,2295** | [0,2227, 0,2365] |
| `t5-small` fine-tuné | 50 % | 0,2191 | [0,2129, 0,2256] |
| `t5-small` fine-tuné | 10 % | 0,1843 | [0,1782, 0,1900] |
| from scratch | 100 % | 0,1634 | [0,1581, 0,1686] |
| from scratch | 50 % | 0,1550 | [0,1499, 0,1600] |
| `t5-small` zero-shot | sans objet | 0,1366 | [0,1328, 0,1406] |
| from scratch | 10 % | 0,1250 | [0,1207, 0,1293] |

Ligne à retenir : le modèle pré-entraîné adapté sur 2 000 exemples fait mieux que le modèle from scratch entraîné sur 20 000.

## 8. La réponse à la question

**Contre le pré-entraîné adapté : jamais, sur la plage mesurée.** L'écart relatif va de 47 % à 40 % selon la taille du corpus, et l'écart absolu grandit au lieu de se réduire. En extrapolant la pente la plus favorable, il faudrait environ 694 000 exemples pour combler l'écart, soit plus de trois fois XSum complet.

**Contre le pré-entraîné non adapté : entre 2 000 et 10 000 exemples.** C'est le seul seuil réellement observé. Il donne un prix concret au pré-entraînement : quelques milliers d'exemples annotés, dès lors qu'on renonce à adapter le modèle.

L'analyse qualitative éclaire le chiffre. Le zero-shot ne résume pas, il recopie trois phrases du document. Le from scratch a appris la forme d'un résumé XSum mais invente le fond : la phrase est correcte, le fait est faux. Il n'apprend même la majuscule initiale qu'entre 10 000 et 20 000 exemples.

Une seconde ablation fait varier la profondeur du modèle. Elle ne montre aucun gain, et le résultat est publié tel quel, négatif.

## 9. Ce qui entoure la science

Le projet n'est pas qu'une campagne d'entraînement. Autour :

- **une API FastAPI** versionnée, `/api/v1` avec `health`, `ready`, `models`, `predict` et `compare`, contrat OpenAPI versionné dans le dépôt ;
- **MLflow**, les neuf runs tracés, deux modèles publiés au registre sous les alias `champion` et `challenger` ;
- **Docker**, trois images et une stack locale ;
- **un pipeline GitHub Actions** en six étapes, de la plus rapide à la plus coûteuse : qualité, sécurité, tests, documentation, conteneur, publication ;
- **1 081 tests** répartis dans 69 fichiers, avec un seuil de couverture de 80 % ;
- **un site de documentation** construit en mode strict, dont les contenus dérivés du code sont régénérés et vérifiés, pour qu'un tableau ne puisse pas mentir sur ce que le code produit.

Une commande, `make ci`, rejoue localement la séquence complète du pipeline.

## 10. Les règles d'intégrité

Trois règles tenues du début à la fin :

1. **Aucun résultat inventé.** Une expérience non lancée porte le statut `NOT_RUN` et garde sa ligne dans le tableau, sans score. Une valeur factice ne devient jamais un résultat.
2. **Le « 100 % » est honnête.** Il désigne le corpus de travail de 20 000 exemples, et la convention est répétée partout.
3. **Les chiffres sont reproductibles.** Les mesures `t5-small` ont été refaites sous une révision épinglée, et la chaîne se rejoue avec des commandes documentées.

## 11. Ce qui reste à faire

| Point | État |
| --- | --- |
| Corpus, modèles, entraînement, évaluation, ablations, figures | Fait, les neuf expériences mesurées |
| MLflow et registre de modèles | Fait |
| API FastAPI | Fait |
| Images Docker | Écrites, jamais construites |
| Pipeline GitHub Actions | Écrit et validé localement, jamais exécuté sur un runner |
| `make reproduce` | Fait, deux modes, `quick` joué de bout en bout en 51 secondes |
| Déploiement | À faire |
| Frontend | Hors périmètre de ce lot |

## 12. Démonstration en cinq commandes

```bash
make install                                  # dépendances et hooks
make ci                                       # la séquence complète de vérification
make api                                      # l'API en local, documentation sur /docs
make mlflow-ui                                # les runs sur localhost:5000
```

Pour rejouer la science, et non seulement la vérifier :

```bash
make reproduce MODE=quick                     # la chaîne entière, plafonnée, 51 secondes
make reproduce MODE=full                      # la campagne réelle, plusieurs heures de GPU
```

Les cinq étapes restent appelables une par une, `make data`, `python -m src.experiments.run --all`, `make ablation`, `make figures`, `make report-sync`, ce que fait la chaîne complète dans cet ordre.
