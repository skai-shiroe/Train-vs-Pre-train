# Tests

## Niveaux

| Niveau | Emplacement | Marqueur | Ce qui est vérifié |
| --- | --- | --- | --- |
| Unitaire | `tests/unit/`, `backend/tests/` | `unit` | Un composant isolé, sans entrée sortie |
| Intégration | `tests/integration/` | `integration` | Deux composants qui se parlent |
| Bout en bout | `tests/e2e/` | `e2e` | L'API complète, en mémoire |
| Smoke | `tests/smoke/` | `smoke` | Un environnement réellement déployé |

Les marqueurs sont déclarés dans `pyproject.toml` et vérifiés par `--strict-markers`. Un marqueur inconnu fait échouer la collecte.

## Commandes

```bash
make test              # tout sauf les smoke tests
make test-unit
make test-integration
make test-e2e
make coverage          # avec le seuil de 80 pour cent
make test-smoke BASE_URL=http://localhost:8000
```

## Couverture

Le seuil est de 80 %, appliqué mécaniquement :

```bash
pytest --cov=src --cov=backend/app \
       --cov-report=xml:reports/coverage.xml \
       --cov-report=term-missing \
       --cov-fail-under=80
```

Le seuil est un plancher, pas un objectif. Les tests portent en priorité sur l'attention, les masques, la génération, les métriques et les endpoints, pas sur du remplissage.

Aucun module n'est exclu de la mesure pour atteindre le chiffre. Toute exclusion dans `pyproject.toml` porte un commentaire de justification.

## Ce qui doit être testé

Exigences du cahier des charges, à couvrir au fur et à mesure de l'implémentation :

```text
attention                    scaled dot product, formes, valeurs
multi-head attention         découpage et concaténation des têtes
masques                      padding et causal
positional encoding          périodicité et bornes
encoder, decoder             formes de sortie, propagation des masques
génération autorégressive    arrêt, longueur maximale, déterminisme
boucle d'entraînement        comptage des pas, écrêtage, reprise
batching par longueur        couverture, reproductibilité, gain mesuré
checkpoints                  aller-retour complet, rotation, générateurs
baseline pré-entraînée       préfixe, troncature, décodage, adaptateur de perte
métriques ROUGE              ordre des arguments, cas limites, chaînes vides
évaluation                   alignement, run partiel, artefacts écrits
sélection qualitative        extrêmes, tirage, déterminisme
tokenisation                 aller-retour, troncature
configuration d'expérience   règles refusées au chargement
lanceur d'expériences        poids relus, run partiel, échec enregistré
tableaux d'ablation          statuts, cellules vides, refus de comparer
schemas FastAPI              validation, valeurs refusées
chargeur de modèles          architecture reconstruite, poids incompatibles, révision épinglée
magasin de modèles           cache par version, échec isolé, registre vide
endpoints                    enveloppe d'erreur, budget plafonné, décodage rapporté
conteneurs                   étapes, utilisateur, sonde, profil, extra suffisant
```

## Bout en bout

Les tests de `tests/e2e/` montent l'application complète en mémoire, contre un registre temporaire dans lequel de vrais poids ont été publiés. Rien n'est simulé sous la couche HTTP : les versions sont écrites par `LocalModelRegistry`, relues par le chargeur de la section 27, et les résumés sont générés par un vrai Transformer. Seul le tokenizer est remplacé, parce que télécharger un vocabulaire est la seule étape qui demande un réseau.

Les modèles y sont minuscules et non entraînés. Ce qu'un test de bout en bout vérifie est qu'une requête atteint les bons poids et revient dans la forme promise par le contrat ; ce que ces poids valent se mesure sur le jeu de test figé, et nulle part ailleurs.

## Smoke tests

Les smoke tests visent un environnement déployé, jamais un mock. Ils sont pilotés par `BASE_URL`, ignorés quand il n'est pas défini, et doivent tourner en moins d'une minute.

```text
GET  /api/v1/health    répond 200
GET  /api/v1/ready     répond 200 et les modèles sont chargés
GET  /api/v1/models    liste au moins un modèle
POST /api/v1/predict   retourne une sortie non vide et une latence mesurée
```

Un échec de smoke test après déploiement fait échouer le pipeline.

## Tests GPU

Les tests marqués `gpu` sont ignorés sur une machine sans CUDA. Ils ne remplacent pas les tests CPU : tout composant doit rester testable sans GPU, sinon la CI ne peut pas le valider.

## Tests lents

Le marqueur `slow` couvre les tests qui téléchargent de vrais poids, aujourd'hui ceux de `t5-small`. Ils sont exclus du hook pre-push et rejoués en intégration continue. Un composant qui ne serait testable qu'avec un téléchargement resterait non couvert sur le poste du développeur, donc chaque module a d'abord ses tests hors ligne.

## État actuel

Les tests présents couvrent le socle, le pipeline de données, le Transformer from scratch, la chaîne d'entraînement, la baseline pré-entraînée, l'évaluation, les ablations, la chaîne de reproduction, la traçabilité MLflow, le registre de modèles, l'API et les fichiers de conteneurisation. Reste à couvrir avec son implémentation : le déploiement lui-même.

`tests/integration/test_reproduce_chain.py` joue les cinq étapes de `make reproduce` en mode `quick`, sur le corpus synthétique des fixtures. Une propriété y est plus importante que les autres : la chaîne ne doit rien écrire hors des destinations qu'on lui donne. Un chemin codé en dur dans l'une des cinq étapes écraserait une campagne, et c'est le seul endroit du dépôt qui le verrait.

`tests/unit/test_container_config.py` lit les `Dockerfile` et `compose.yaml` sans rien construire. Ce sont des tests de fichiers, et ils sont assumés comme tels : construire une image demande un démon Docker que ni le hook pre-push ni le poste de développement n'ont. Le détail de ce qu'ils tiennent en place est dans [Déploiement](deployment.md).

Cinq tests portent plus de valeur que les autres, parce qu'ils échouent sur des défauts qu'une perte qui baisse ne révèle pas :

```text
causalité du décodeur      modifier la position k laisse les logits < k inchangés
invariance au padding      le padding source ne change pas les prédictions
surapprentissage d'un lot  la perte tombe sous 20 % de sa valeur initiale
asymétrie de ROUGE         une prédiction courte donne une précision pleine
poids évalués              le chemin relu est celui du meilleur checkpoint
```

Le dernier mérite un mot lui aussi. Les poids en mémoire à la fin d'un entraînement sont ceux de la dernière époque, et l'early stopping existe parce que ce n'est pas la meilleure. Tous les autres tests du lanceur passeraient si l'évaluation mesurait la dernière époque : le score serait plausible, simplement plus bas, et rien ne le signalerait.

Le dernier mérite un mot. `RougeScorer.score` attend la référence avant la prédiction, et inverser les deux ne change pas la F-mesure : la précision et le rappel échangent leur valeur sans qu'aucun test écrit sur F ne s'en aperçoive. Le test compare `the cat sat` à `the cat sat on the mat` et attend une précision de 1 et un rappel de 0,5, valeurs calculées à la main. C'est le seul endroit du dépôt où l'erreur devient visible.

Chiffres du dernier passage de `make coverage` : 1267 tests passés, 1 ignoré, 4 déselectionnés parce qu'ils visent un environnement déployé, couverture 99,17 % pour un plancher de 80 %.
