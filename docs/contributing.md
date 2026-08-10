# Contribuer

## Installation

```bash
py -3.12 -m venv .venv
source .venv/Scripts/activate   # Windows, depuis Git Bash
make install
```

`make install` installe PyTorch depuis l'index CUDA 12.8, puis les dépendances du projet, puis pose les hooks. Un dépôt cloné est conforme dès le premier commit.

!!! warning "GPU Blackwell"
    Les cartes RTX 50 (architecture Blackwell, `sm_120`) exigent les roues PyTorch compilées avec CUDA 12.8. L'index par défaut de PyPI ne convient pas. La variable `TORCH_INDEX` du `Makefile` porte cette contrainte.

### GNU Make sous Windows

`make` n'est pas fourni avec Windows. L'installer avant toute chose :

```powershell
winget install ezwinports.make
```

Sans `make`, les commandes restent exécutables une par une, mais `make ci` ne peut pas reproduire le verdict du pipeline en une seule fois.

### Activation de l'environnement

Les hooks locaux (`no-long-dashes`, `conventional-commit`, `docs-sync-check`) appellent `python`. L'environnement virtuel doit donc être activé avant de committer, sinon c'est le Python système qui est utilisé et les dépendances du projet manquent.

## Format des messages de commit

Le projet suit Conventional Commits. Le hook `commit-msg` rejette tout message hors format.

```text
type(scope): sujet
```

Types acceptés :

```text
feat      nouvelle fonctionnalité
fix       correction de défaut
docs      documentation seule
test      ajout ou correction de tests
refactor  restructuration sans changement de comportement
perf      amélioration de performance
build     système de build ou dépendances
ci        configuration d'intégration continue
chore     tâche d'entretien
revert    annulation d'un commit précédent
```

Règles :

```text
le sujet fait au plus 72 caractères
le sujet ne se termine pas par un point
le scope est optionnel, en minuscules
le point d'exclamation avant le deux-points signale une rupture de compatibilité
```

Exemples valides :

```text
feat(api): add the compare endpoint
fix(scratch): correct the causal mask shape
docs(ml): document the ablation protocol
refactor(training)!: change the checkpoint format
```

## Hooks

```bash
make hooks
```

installe les trois types de hooks.

| Stage | Contenu | Durée visée |
| --- | --- | --- |
| `pre-commit` | Formatage, style, typage, sécurité, orthographe, YAML, JSON, TOML, secrets, tirets longs | Quelques secondes |
| `commit-msg` | Format Conventional Commits | Immédiat |
| `pre-push` | Tests rapides, synchronisation de la documentation | Moins d'une minute |

L'option `--no-verify` n'est pas une pratique acceptée. La CI rejoue les mêmes hooks sur tout le dépôt : contourner localement ne fait que déplacer l'échec.

## Commandes qualité

```bash
make format      # black et corrections automatiques de ruff
make lint        # black --check, flake8, ruff, yamllint, tirets longs
make spell       # codespell
make prose       # linter de prose Vale
make typecheck   # mypy en mode strict
make security    # bandit et pip-audit
make coverage    # pytest avec seuil de 80 pour cent
make ci          # séquence complète du pipeline, en local
```

Lancer `make ci` avant de pousser. Il produit le même verdict que le pipeline GitHub Actions.

## Le pipeline

`.github/workflows/ci.yml` déclare six étapes et appelle un workflow réutilisable par étape depuis `.github/workflows/`. Aucun job n'est défini dans le fichier racine. Les jobs appellent les cibles du Makefile plutôt que de réécrire les commandes : deux listes maintenues en parallèle auraient fini par diverger, et c'est la phrase ci-dessus qui serait devenue fausse.

| Étape | Job | Ce qu'il vérifie |
| --- | --- | --- |
| `qualite` | format et style, orthographe, typage statique | `make lint`, `make spell`, `make typecheck`, en parallèle |
| `qualite` | hooks sur tout le dépôt | `pre-commit run --all-files` |
| `securite` | sécurité du code et des dépendances | `make security` |
| `tests` | tests et couverture | `make coverage`, seuil et rapports JUnit et Cobertura |
| `documentation` | documentation et contrat | `make docs-lint` puis `make docs` |
| `conteneur` | image backend, scan de l'image | Construction, puis Trivy sur l'archive produite |
| `publication` | pages | Publie le site déjà construit, sur la branche par défaut |

Les étapes vont du moins cher au plus cher : le style échoue en une minute, l'image prend plusieurs minutes. L'erreur la plus fréquente est donc celle dont la boucle est la plus courte.

L'installation commune vit dans l'action composite `.github/actions/environnement-python`, appelée par chaque job Python : environnement virtuel, roues CPU de torch, projet installé, `.venv/bin` ajouté au `PATH`. Ce dernier point n'est pas cosmétique : le hook `mypy` est déclaré `language: system` et lancerait le Python du runner sans lui.

Le pipeline installe les roues CPU de torch, pas l'index CUDA du Makefile. Un runner partagé n'a pas de GPU, et c'est la conséquence directe de la règle de `docs/testing.md` : tout composant doit rester testable sans GPU.

L'étape `conteneur` ne déclare aucun service Docker in Docker : le runner `ubuntu-latest` porte déjà un démon Docker. L'étape `publication` suppose que GitHub Pages est activé sur le dépôt avec GitHub Actions comme source ; sans ce réglage, le déploiement échoue sur la configuration du dépôt et non sur le contenu du site.

GitHub ne rend nativement ni JUnit ni Cobertura. Les deux rapports sont donc téléversés en artefacts, et le taux de couverture est recopié dans le résumé du job depuis `coverage.xml`, le fichier sur lequel le seuil s'est déjà prononcé.

Une poussée sur une branche qui porte une pull request déclenche deux événements. Le groupe de concurrence les réunit sous la même clé pour que la vérification ne soit payée qu'une fois, sauf sur la branche par défaut, où annuler une exécution en cours annulerait la publication du site.

## Règle de rédaction

Les tirets cadratin et demi-cadratin sont interdits dans tout le dépôt : documentation, docstrings, commentaires, messages de commit, messages d'erreur et descriptions de jobs.

Remplacer par un deux-points, une virgule, une parenthèse, ou couper en deux phrases. Le trait d'union ordinaire reste autorisé.

La règle est vérifiée par `scripts/check_dashes.py`, exécuté en hook et en CI. Elle est bloquante.

## Orthographe et style

`codespell` couvre l'orthographe du code et de la documentation. Les faux positifs, notamment le vocabulaire technique et les mots français courts, vont dans `.codespell-ignore`, un mot par ligne. Ne jamais désactiver globalement le contrôle.

Le linter de prose Vale couvre la grammaire, le style et la terminologie. Il s'installe séparément :

```bash
# Windows
winget install errata-ai.Vale
# Linux et macOS
brew install vale
```

Les règles projet vivent dans `docs/styles/Syntra/`. La terminologie imposée est listée dans le [glossaire](glossary.md).

## Docstrings

Toute fonction publique, classe et module de `src` et `backend/app` porte une docstring de style Google, rédigée en anglais. La couverture est mesurée par `interrogate`, seuil à 90 %.

Les docstrings des composants du Transformer expliquent la formule implémentée.
