# Sécurité

La sécurité est validée au plus tôt, sur le poste du développeur, puis rejouée à l'identique en intégration continue.

## Analyse statique du code

Bandit analyse `src` et `backend/app`.

```bash
make security
```

soit :

```bash
bandit -c pyproject.toml -r src backend/app -f json -o reports/bandit.json
bandit -c pyproject.toml -r src backend/app -ll
```

Règles :

```text
le job échoue sur les findings de sévérité HIGH
les findings MEDIUM sont rapportés et doivent être traités ou justifiés
le rapport JSON est publié en artefact
```

### Exceptions

Une exception s'écrit avec `# nosec` suivi du code de la règle et de la raison. Une exception sans justification est un défaut de revue.

```python
subprocess.run(  # nosec B603 - liste d'arguments fixe, aucun shell
    [sys.executable, str(script), "--output", str(target)],
    check=False,
)
```

Exceptions actives à ce jour :

| Fichier | Règle | Raison |
| --- | --- | --- |
| `scripts/check_docs_sync.py` | B404, B603 | Invoque uniquement un script du dépôt, avec une liste d'arguments fixe et sans shell |
| `src/tracking/provenance.py` | B404, B603 | Invoque `git`, liste d'arguments fixe, sans shell |
| `src/training/checkpoint.py` | B614 | Écriture d'un checkpoint, et lecture déjà en `weights_only=True` |
| `src/experiments/publish.py` | B614 | Écriture des poids réduits dans le registre |
| `backend/app/inference/loader.py` | B614 | Lecture des poids publiés, en `weights_only=True` |
| `src/training/sampler.py` | B311 | Le tirage ordonne des lots d'entraînement, il ne protège rien |

La règle B614 signale tout appel à `torch.load` et `torch.save`. Les quatre exceptions ci-dessus portent sur le faux positif, jamais sur la protection : chaque lecture de poids passe par `weights_only=True`, y compris celle que l'API fait au démarrage.

## Dépendances

`pip-audit` couvre les vulnérabilités connues des dépendances.

```bash
pip-audit --strict -s osv
```

Les versions de l'outillage sont épinglées à l'exact dans `pyproject.toml`. Les dépendances applicatives portent une borne haute majeure, pour qu'une version majeure ne soit jamais tirée sans revue.

### Le service d'avis est OSV, pas PyPI

Le service par défaut de `pip-audit` interroge PyPI, qui ne sait pas résoudre une version locale. Or `make install` installe torch depuis un index CUDA, donc la version portée est de la forme `2.13.0+cu130`. Le service PyPI la déclare introuvable et, sous `--strict`, fait échouer le job.

Le contournement évident, retirer `--strict`, est le mauvais : torch resterait non audité, silencieusement. Le service OSV accepte la version locale et audite réellement la dépendance la plus lourde du projet. Le choix se paie d'un audit un peu plus large, ce qui est le sens recherché.

### Vulnérabilités acceptées

Une alerte se traite par une montée de version. Quand aucune montée n'est possible, elle s'inscrit ici et dans `AUDIT_IGNORES` du `Makefile`. Une entrée sans justification est une vulnérabilité masquée, pas une vulnérabilité traitée.

| Identifiant | Paquet | Pourquoi elle n'est pas corrigée |
| --- | --- | --- |
| `PYSEC-2025-217` | `transformers` | Aucune version corrective publiée |
| `PYSEC-2026-2290` | `transformers` | Aucune version corrective publiée |
| `PYSEC-2026-2288` | `transformers` | Corrigée en 5.0.0, au-delà de la borne `<5.0` |
| `PYSEC-2026-2289` | `transformers` | Corrigée en 5.3.0, au-delà de la borne `<5.0` |
| `PYSEC-2026-3552` | `cryptography` | Corrigée en 50.0.0, que `mlflow` refuse par sa borne `cryptography<50` |

Les quatre entrées `transformers` tombent avec le passage en 5.x, qui n'est pas une montée de version mais un changement de baseline : la couche pré-entraînée et les résultats T5 publiés devraient être revalidés. Tant que ce travail n'est pas fait, la borne `<5.0` tient et les alertes restent ouvertes.

L'entrée `cryptography` ne dépend pas du projet : elle tombera quand `mlflow` relèvera son plafond. La liste se relit à chaque montée de `transformers` et de `mlflow`.

## Images Docker

Trivy scanne l'image construite, après le build et avant le déploiement.

```bash
trivy image --exit-code 1 --severity CRITICAL --ignore-unfixed \
  --format table "$IMAGE_TAG"
```

Règles :

```text
scan systématique de l'image construite
le job échoue si une vulnérabilité CRITICAL est détectée
les niveaux HIGH sont rapportés mais ne bloquent pas par défaut
le rapport est publié en artefact
```

Toute exception vit dans `.trivyignore`, avec la référence CVE et la raison. Le scan n'est jamais neutralisé globalement pour faire passer un pipeline. Le fichier ne contient aujourd'hui aucune exception active.

Aucune image n'a encore été construite : le scan n'a donc jamais tourné, et aucun résultat n'est publié. Voir [Déploiement](deployment.md).

## Secrets

```text
detect-secrets tourne en hook, avec une baseline versionnée
detect-private-key refuse toute clé privée commitée
les secrets applicatifs utilisent SecretStr et ne sont jamais journalisés
.env est ignoré par git, seul .env.example est versionné
la CI lit ses secrets depuis les secrets chiffrés du dépôt, jamais de secret en clair
aucun secret n'entre dans une image Docker ni dans ses layers
```

Les valeurs présentes dans `.env.example` sont des marqueurs. Elles doivent être remplacées hors du poste de développement.

## Durcissement de l'image

```text
build multi-stage, aucune toolchain de compilation dans l'image finale
image de base légère
utilisateur non-root dédié
HEALTHCHECK déclaré
dépendances épinglées
aucun checkpoint lourd embarqué : les modèles viennent du Model Registry
```

Ces règles ne sont pas relues, elles sont vérifiées : `tests/unit/test_container_config.py` lit les fichiers de construction et échoue si l'une d'elles disparaît. Le détail est dans [Déploiement](deployment.md).

## Surface exposée par l'API

```text
CORS piloté par la configuration, l'origine étoile est refusée par validation
taille du texte source bornée par le schéma Pydantic
délai maximal d'inférence configurable, traduit en erreur TIMEOUT
aucune trace d'exception renvoyée au client
aucun texte utilisateur journalisé intégralement, seulement sa longueur
la documentation interactive est désactivable par configuration
```
