"""Compare the code of the campaign with the code of today, logic only.

The results of the report were measured in a directory without a ``.git``. The
commit their records name, ``5a9c95df844f``, exists in no repository anyone can
reach, so no hash ties the numbers to a state of the code. What replaces it is
this comparison, and the difference matters: a hash is believed, a comparison is
re-run.

The campaign tree is frozen on the ``archive/campagne-2026-08`` branch. This
script reads ``src`` from there and from the working tree, and compares the two
after removing comments and docstrings:

```bash
python -m scripts.compare_archive
python -m scripts.compare_archive --ref archive/campagne-2026-08 --json
```

**Comments are not a difference, docstrings are not either.** Both are compared
away on purpose. What is being asked is whether the code still computes the same
thing, and prose does not compute. Reformatting is compared away too, since the
comparison is made on syntax trees rather than on text.

**Two exit codes, and the second is not a failure.** Zero when nothing that
computes has moved, one when something has. One is the answer that requires
reading, not a broken script: the four modules that generate the tables of the
report have changed since the campaign, and that is documented rather than
wrong. Section "Tracabilite" of RAPPORT.md carries the current answer.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess  # nosec B404 - read only git plumbing, fixed argument list
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Branch the campaign tree was frozen on.
DEFAULT_REF = "archive/campagne-2026-08"

#: Subtree the comparison covers. The tests and the configurations are archived
#: too, but only what runs during a measurement can change a number.
DEFAULT_SUBTREE = "src"


def list_files(ref: str, subtree: str) -> list[str]:
    """List the Python files a reference carries under a subtree.

    Args:
        ref: Git reference to read.
        subtree: Path the listing is restricted to.

    Returns:
        Repository relative paths, sorted.

    Raises:
        SystemExit: When the reference does not exist, which is the one failure
            worth stopping on: comparing against nothing would report every file
            as removed.
    """
    try:
        completed = subprocess.run(  # nosec B603 - fixed argument list, no shell
            ["git", "ls-tree", "-r", "--name-only", ref, "--", subtree],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"cannot read {ref!r}: {error}") from error

    return sorted(line for line in completed.stdout.splitlines() if line.endswith(".py"))


def read_at(ref: str, path: str) -> str | None:
    """Read one file as a reference carries it.

    Args:
        ref: Git reference to read.
        path: Repository relative path.

    Returns:
        The content, or ``None`` when the reference does not carry that file.
    """
    completed = subprocess.run(  # nosec B603 - fixed argument list, no shell
        ["git", "show", f"{ref}:{path}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout if completed.returncode == 0 else None


def logic_of(source: str) -> str:
    """Render what a module computes, without what it says about itself.

    Args:
        source: Python source.

    Returns:
        A dump of the syntax tree with every docstring removed. Two modules with
        the same dump compute the same thing, whatever their comments, their
        formatting and their prose.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree)


def compare(ref: str, subtree: str) -> dict[str, list[str]]:
    """Compare a reference with the working tree, logic only.

    Args:
        ref: Git reference holding the campaign tree.
        subtree: Path the comparison covers.

    Returns:
        Four sorted lists under the keys ``identical``, ``changed``,
        ``only_in_archive`` and ``only_in_worktree``. A file that no longer
        parses on either side is reported as changed, since a comparison that
        cannot be made is not a comparison that succeeded.
    """
    result: dict[str, list[str]] = {
        "identical": [],
        "changed": [],
        "only_in_archive": [],
        "only_in_worktree": [],
    }

    archived = list_files(ref, subtree)
    for path in archived:
        source = read_at(ref, path)
        current_file = REPO_ROOT / path
        if source is None or not current_file.is_file():
            result["only_in_archive"].append(path)
            continue

        try:
            same = logic_of(source) == logic_of(current_file.read_text(encoding="utf-8"))
        except SyntaxError:
            same = False
        result["identical" if same else "changed"].append(path)

    known = set(archived)
    for current_file in sorted((REPO_ROOT / subtree).rglob("*.py")):
        path = current_file.relative_to(REPO_ROOT).as_posix()
        if path not in known:
            result["only_in_worktree"].append(path)

    return result


def format_report(ref: str, result: dict[str, list[str]]) -> str:
    """Render the comparison for a reader.

    Args:
        ref: Reference that was compared against.
        result: What :func:`compare` returned.

    Returns:
        The report, without a trailing newline.
    """
    lines = [
        f"Comparaison de src avec {ref}, commentaires et docstrings retires.",
        "",
        f"  identiques en logique   {len(result['identical'])}",
        f"  logique modifiee        {len(result['changed'])}",
        f"  seulement dans archive  {len(result['only_in_archive'])}",
        f"  seulement ici           {len(result['only_in_worktree'])}",
    ]
    for key, title in (
        ("changed", "Logique modifiee depuis la campagne"),
        ("only_in_archive", "Presents dans l'archive seulement"),
        ("only_in_worktree", "Ajoutes depuis la campagne"),
    ):
        if result[key]:
            lines.extend(["", f"{title} :"])
            lines.extend(f"  {path}" for path in result[key])
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    """Describe the command line of ``python -m scripts.compare_archive``.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m scripts.compare_archive",
        description=(
            "Compare le code du depot avec l'arbre de campagne archive, "
            "commentaires et docstrings retires."
        ),
    )
    parser.add_argument("--ref", default=DEFAULT_REF, help="Reference git de l'arbre de campagne.")
    parser.add_argument("--subtree", default=DEFAULT_SUBTREE, help="Sous-arbre compare.")
    parser.add_argument("--json", action="store_true", help="Emet le resultat en JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Compare from the command line.

    Args:
        argv: Arguments to parse. Defaults to the process arguments.

    Returns:
        Zero when nothing that computes has moved, one otherwise.
    """
    args = build_argument_parser().parse_args(argv)
    result = compare(args.ref, args.subtree)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(format_report(args.ref, result))

    moved = result["changed"] + result["only_in_archive"] + result["only_in_worktree"]
    return 1 if moved else 0


if __name__ == "__main__":
    sys.exit(main())
