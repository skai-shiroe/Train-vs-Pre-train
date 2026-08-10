"""Unit tests for the container definitions of section 28 and section 29.

Nothing here builds an image. What these tests hold in place are the rules that
a Dockerfile silently stops respecting: a stage merged back into one, a ``USER``
line dropped during a rewrite, a ``COPY .`` that pulls the corpus into a layer,
a service that starts training on ``docker compose up``.

The last test is the one that would otherwise only be found by a failing
deployment: the backend image installs the ``api`` extra alone, so every package
the application imports has to be declared in it. A module added under
``backend/app`` that reaches into ``mlflow`` or ``pandas`` builds, passes the
whole suite in the development environment, and crashes on the first start of
the container.
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

BACKEND_DOCKERFILE = REPO_ROOT / "backend" / "Dockerfile"
TRAINING_DOCKERFILE = REPO_ROOT / "docker" / "training.Dockerfile"
MLFLOW_DOCKERFILE = REPO_ROOT / "docker" / "mlflow.Dockerfile"
COMPOSE_FILE = REPO_ROOT / "compose.yaml"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"
PYPROJECT = REPO_ROOT / "pyproject.toml"

DOCKERFILES = [BACKEND_DOCKERFILE, TRAINING_DOCKERFILE, MLFLOW_DOCKERFILE]

#: Names that suggest a credential. Used to check that no Dockerfile and no
#: compose service carries a literal value for one of them.
SECRET_NAME = re.compile(r"(PASSWORD|SECRET|TOKEN|_KEY\b|APIKEY)", re.IGNORECASE)

#: Distributions whose import name differs from their name on PyPI.
IMPORT_NAMES = {
    "pydantic-settings": "pydantic_settings",
    "pyyaml": "yaml",
    "rouge-score": "rouge_score",
    "absl-py": "absl",
}

REQUIREMENT_SEPARATORS = re.compile(r"[<>=!~;\[\s]")


def instructions(dockerfile: Path) -> list[tuple[str, str]]:
    """Return the instructions of a Dockerfile, continuations joined.

    Args:
        dockerfile: Path of the file to read.

    Returns:
        One ``(keyword, argument)`` pair per instruction, keyword upper cased.
    """
    joined: list[str] = []
    buffer = ""
    for raw in dockerfile.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not buffer and (not line or line.startswith("#")):
            continue
        if line.startswith("#"):
            continue
        if line.endswith("\\"):
            buffer += line[:-1] + " "
            continue
        joined.append(buffer + line)
        buffer = ""
    if buffer:
        joined.append(buffer)

    parsed: list[tuple[str, str]] = []
    for entry in joined:
        keyword, _, argument = entry.partition(" ")
        parsed.append((keyword.upper(), argument.strip()))
    return parsed


def keyword_arguments(dockerfile: Path, keyword: str) -> list[str]:
    """Return the arguments of every instruction carrying one keyword.

    Args:
        dockerfile: Path of the file to read.
        keyword: Instruction to look for, for example ``USER``.

    Returns:
        The arguments, in file order.
    """
    return [argument for found, argument in instructions(dockerfile) if found == keyword]


def compose() -> dict[str, Any]:
    """Return the parsed compose file.

    Returns:
        The whole document.
    """
    loaded: dict[str, Any] = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    return loaded


def compose_services() -> dict[str, dict[str, Any]]:
    """Return the services declared by the compose file.

    Returns:
        The service definitions, keyed by service name.
    """
    services: dict[str, dict[str, Any]] = compose()["services"]
    return services


# ---------------------------------------------------------------------------
# Dockerfiles, section 28
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_every_declared_dockerfile_exists() -> None:
    for dockerfile in DOCKERFILES:
        assert dockerfile.is_file(), f"{dockerfile} est declare et manquant."


@pytest.mark.unit
@pytest.mark.parametrize(
    "dockerfile", [BACKEND_DOCKERFILE, TRAINING_DOCKERFILE], ids=["backend", "training"]
)
def test_the_build_is_multi_stage(dockerfile: Path) -> None:
    stages = keyword_arguments(dockerfile, "FROM")

    assert len(stages) >= 2, "Section 28.1 impose une etape builder separee de l'etape finale."
    assert all(" AS " in stage.upper() for stage in stages), "Chaque etape doit etre nommee."


@pytest.mark.unit
@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda path: path.name)
def test_the_base_image_is_pinned(dockerfile: Path) -> None:
    for stage in keyword_arguments(dockerfile, "FROM"):
        image = stage.split()[0]
        assert ":latest" not in image, f"{image} suit latest, donc rien n'est reproductible."
        assert ":" in image, f"{image} n'est pas tague."


@pytest.mark.unit
@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda path: path.name)
def test_the_image_drops_root(dockerfile: Path) -> None:
    users = keyword_arguments(dockerfile, "USER")

    assert users, "Section 28 impose un utilisateur non-root dedie."
    assert "root" not in users[-1], f"La derniere instruction USER est {users[-1]}."


@pytest.mark.unit
@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda path: path.name)
def test_nothing_is_copied_wholesale(dockerfile: Path) -> None:
    # COPY . et ADD . embarqueraient le corpus, les rapports et tout fichier
    # local oublie par .dockerignore. Les copies sont nommees une par une.
    for keyword in ("COPY", "ADD"):
        for argument in keyword_arguments(dockerfile, keyword):
            sources = [part for part in argument.split() if not part.startswith("--")]
            assert sources[0] != ".", f"{keyword} {argument} copie tout le contexte."


@pytest.mark.unit
@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda path: path.name)
def test_no_credential_is_baked_into_a_layer(dockerfile: Path) -> None:
    for keyword in ("ENV", "ARG"):
        for argument in keyword_arguments(dockerfile, keyword):
            for assignment in argument.split():
                name, separator, value = assignment.partition("=")
                if separator and SECRET_NAME.search(name):
                    assert not value.strip('"'), f"{name} porte une valeur en dur."


@pytest.mark.unit
@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda path: path.name)
def test_apt_cleans_its_lists_in_the_same_instruction(dockerfile: Path) -> None:
    for command in keyword_arguments(dockerfile, "RUN"):
        if "apt-get install" not in command:
            continue
        assert "rm -rf /var/lib/apt/lists" in command, "Le cache apt reste dans le layer."
        assert "--no-install-recommends" in command, "Les paquets recommandes gonflent l'image."


@pytest.mark.unit
@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda path: path.name)
def test_pip_leaves_no_cache_behind(dockerfile: Path) -> None:
    declared = " ".join(keyword_arguments(dockerfile, "ENV"))
    commands = " ".join(keyword_arguments(dockerfile, "RUN"))

    if "pip install" not in commands:
        return
    assert "PIP_NO_CACHE_DIR=1" in declared or "--no-cache-dir" in commands


@pytest.mark.unit
def test_the_backend_declares_a_healthcheck_on_health() -> None:
    probes = keyword_arguments(BACKEND_DOCKERFILE, "HEALTHCHECK")

    assert len(probes) == 1, "Section 28 impose un HEALTHCHECK, et un seul."
    probe = probes[0]
    assert "/health" in probe, "La sonde doit interroger /health."
    assert "/ready" not in probe, (
        "La sonde de vivacite ne doit pas dependre de /ready : un modele qui n'a "
        "pas charge se rapporte, il ne fait pas tuer le conteneur."
    )
    assert "--start-period" in probe, "Le chargement des poids doit etre couvert."


@pytest.mark.unit
def test_the_backend_serves_without_the_uvicorn_access_log() -> None:
    # Le log d'acces d'uvicorn ecrit l'URL complete, query string comprise. Un
    # document passe en parametre finirait sur disque, ce que la politique de
    # confidentialite du projet interdit. L'application journalise elle-meme
    # chaque requete.
    commands = keyword_arguments(BACKEND_DOCKERFILE, "CMD")

    assert len(commands) == 1
    assert "--no-access-log" in commands[0]
    assert "backend.app.main:app" in commands[0]


@pytest.mark.unit
def test_the_training_image_does_not_train_by_default() -> None:
    # Section 29 : aucun entrainement lourd ne demarre tout seul. La regle vaut
    # aussi pour l'image, sinon elle depend du fichier compose qui la lance.
    commands = keyword_arguments(TRAINING_DOCKERFILE, "CMD")

    assert len(commands) == 1
    assert "--help" in commands[0]


@pytest.mark.unit
def test_the_two_images_share_the_same_service_account_id() -> None:
    # Le cache du tokenizer est un volume partage par les deux conteneurs. Deux
    # uid differents rendraient le volume illisible pour l'un des deux.
    def uid(dockerfile: Path) -> str:
        for command in keyword_arguments(dockerfile, "RUN"):
            match = re.search(r"useradd .*--uid (\d+)", command)
            if match:
                return match.group(1)
        raise AssertionError(f"{dockerfile.name} ne cree aucun utilisateur.")

    assert uid(BACKEND_DOCKERFILE) == uid(TRAINING_DOCKERFILE) == "10001"


# ---------------------------------------------------------------------------
# .dockerignore, section 28
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "pattern",
    [".git/", "data/", "artifacts/", "checkpoints/", "*.pt", ".env", "tests/", "backend/tests/"],
)
def test_the_build_context_excludes_what_must_never_enter_a_layer(pattern: str) -> None:
    entries = {
        line.strip()
        for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }

    assert pattern in entries


# ---------------------------------------------------------------------------
# compose.yaml, section 29
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("service", ["backend", "mlflow", "postgres", "minio"])
def test_the_stack_declares_the_four_required_services(service: str) -> None:
    assert service in compose_services()


@pytest.mark.unit
def test_training_is_the_only_service_behind_a_profile() -> None:
    behind_a_profile = {
        name: definition.get("profiles")
        for name, definition in compose_services().items()
        if definition.get("profiles")
    }

    assert behind_a_profile == {"training": ["training"]}


@pytest.mark.unit
def test_a_plain_up_starts_no_training() -> None:
    # Section 29 : docker compose up ne doit pas lancer d'entrainement lourd.
    # Le profil suffit, encore faut-il que rien d'autre ne demarre le service.
    for name, definition in compose_services().items():
        if name == "training":
            continue
        for dependency in definition.get("depends_on", {}):
            assert dependency != "training", f"{name} tirerait l'entrainement au demarrage."


@pytest.mark.unit
def test_no_service_carries_a_literal_credential() -> None:
    for name, definition in compose_services().items():
        environment: dict[str, str] = definition.get("environment", {})
        for key, value in environment.items():
            if SECRET_NAME.search(key):
                assert str(value).startswith("${"), (
                    f"{name}.{key} porte une valeur en dur. Les identifiants entrent "
                    "par interpolation depuis .env."
                )


@pytest.mark.unit
def test_the_metadata_store_is_not_published_on_the_host() -> None:
    assert "ports" not in compose_services()["postgres"]


@pytest.mark.unit
@pytest.mark.parametrize("service", ["postgres", "minio"])
def test_a_third_party_service_gets_its_probe_from_compose(service: str) -> None:
    # Ces deux images ne sont pas les notres : la sonde ne peut se declarer
    # qu'ici, et sans elle depends_on ne saurait pas attendre.
    assert "healthcheck" in compose_services()[service]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("service", "dockerfile"),
    [("backend", BACKEND_DOCKERFILE), ("mlflow", MLFLOW_DOCKERFILE)],
)
def test_a_service_we_build_carries_its_probe_in_its_image(service: str, dockerfile: Path) -> None:
    # La sonde appartient a l'image : c'est la seule que connaisse un docker run
    # lance sans compose. La redefinir ici la ferait diverger en silence.
    assert keyword_arguments(dockerfile, "HEALTHCHECK")
    assert "healthcheck" not in compose_services()[service]


@pytest.mark.unit
def test_the_registry_is_mounted_read_only() -> None:
    mounts = compose_services()["backend"]["volumes"]

    registry = [mount for mount in mounts if "/srv/registry" in mount]
    assert registry, "Le registre n'est pas monte dans le conteneur de service."
    assert registry[0].endswith(":ro"), "L'API sert des versions, elle n'en publie pas."


@pytest.mark.unit
def test_every_built_service_points_at_an_existing_dockerfile() -> None:
    for name, definition in compose_services().items():
        build = definition.get("build")
        if not build:
            continue
        dockerfile = REPO_ROOT / build["dockerfile"]
        assert dockerfile.is_file(), f"{name} construit depuis un fichier absent."


# ---------------------------------------------------------------------------
# Ce que l'image de service doit pouvoir importer
# ---------------------------------------------------------------------------


def declared_import_names(extra: str) -> set[str]:
    """Return the import names installed by one extra of the project.

    Args:
        extra: Name of the optional dependency group, for example ``api``.

    Returns:
        The import names, base dependencies included.
    """
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    requirements = list(project["dependencies"])
    requirements += project["optional-dependencies"][extra]

    names: set[str] = set()
    for requirement in requirements:
        distribution = REQUIREMENT_SEPARATORS.split(requirement.strip())[0].lower()
        names.add(IMPORT_NAMES.get(distribution, distribution.replace("-", "_")))
    return names


def reachable_third_party_imports() -> set[str]:
    """Return the third party packages imported by the API and what it reaches.

    The walk follows ``backend`` and ``src`` imports across files, so a module
    of the research code pulled in by the API is covered too.

    Returns:
        The top level names of every package imported outside the project.
    """
    queue = list((REPO_ROOT / "backend" / "app").rglob("*.py"))
    seen: set[Path] = set()
    third_party: set[str] = set()

    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)

        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                imported = [node.module or ""]
            else:
                continue

            for name in imported:
                top = name.split(".")[0]
                if top not in {"backend", "src"}:
                    third_party.add(top)
                    continue
                module = REPO_ROOT / Path(name.replace(".", "/"))
                queue += [
                    candidate
                    for candidate in (module.with_suffix(".py"), module / "__init__.py")
                    if candidate.is_file()
                ]

    return third_party


@pytest.mark.unit
def test_the_api_extra_covers_everything_the_backend_imports() -> None:
    installed = declared_import_names("api") | set(sys.stdlib_module_names)

    missing = sorted(name for name in reachable_third_party_imports() if name not in installed)

    assert not missing, (
        "L'image backend installe le seul extra api. Ces paquets sont importes "
        f"et ne s'y trouvent pas : {missing}"
    )
