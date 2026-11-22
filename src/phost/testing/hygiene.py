"""Is this git repository safe to publish? Looks at the tree, and at every commit, for what must not leave the machine.

* generated files that were committed (bytecode, caches, build output, virtual environments);
* the current user's home directory path, which compiled files and tracebacks carry into a commit;
* terms from a deny list, for names that must stay private. The deny list is never part of a repository (a public repo
  would then spell the very names it hides): give a path, set ``PHOST_HYGIENE_DENY``, or put the file at
  ``~/.config/phost/deny.txt``. One regular expression per line; blank lines and ``#`` comments are ignored;
* the author and committer identities in the history, reported as notices: whether they are fine to publish is for
  the owner to decide.

A file removed in a later commit is still in the history, so the history is checked, not only the tree.
Run it as ``python -m phost.testing.hygiene [repository] [--deny FILE] [--no-history]``.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

GENERATED = ("*.pyc", "*.pyo", "__pycache__", "*.egg-info", ".venv", "venv", "build", "dist", "*.so", ".pytest_cache",
             ".mypy_cache", ".ruff_cache", ".DS_Store", "*.log")
DENY_ENV = "PHOST_HYGIENE_DENY"


@dataclass(frozen=True)
class Finding:
    kind: str          # "generated-file" | "private-path" | "denied-term" | "identity"
    where: str         # a path, or "<commit>:<path>" for history
    detail: str

    @property
    def is_problem(self) -> bool:
        return self.kind != "identity"

    def __str__(self) -> str:
        return f"[{self.kind}] {self.where}: {self.detail}"


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout


def load_deny(path: str | os.PathLike | None = None) -> list[re.Pattern]:
    """Compile the deny list from ``path``, ``$PHOST_HYGIENE_DENY`` or ``~/.config/phost/deny.txt`` (empty if none)."""
    candidate = path or os.environ.get(DENY_ENV) or os.path.join(os.path.expanduser("~"), ".config", "phost", "deny.txt")
    if not os.path.isfile(candidate):
        return []
    lines = (line.strip() for line in Path(candidate).read_text().splitlines())
    return [re.compile(line, re.IGNORECASE) for line in lines if line and not line.startswith("#")]


def _is_generated(path: str) -> str | None:
    parts = path.split("/")
    for pattern in GENERATED:
        if any(fnmatch.fnmatch(part, pattern) for part in parts):
            return pattern
    return None


def _scan(text: str, where: str, patterns: Sequence[tuple[str, re.Pattern]]) -> Iterable[Finding]:
    for kind, pattern in patterns:
        found = pattern.search(text)
        if found:
            yield Finding(kind, where, f"matches {pattern.pattern!r}")


def _private_patterns(home: str | None, deny: Sequence[re.Pattern]) -> list[tuple[str, re.Pattern]]:
    home = os.path.expanduser("~") if home is None else home
    patterns = [("denied-term", pattern) for pattern in deny]
    if home and home != "/":
        patterns.append(("private-path", re.compile(re.escape(home.rstrip("/")) + r"(?=/|\b|$)")))
    return patterns


def check_tree(root: str | os.PathLike, *, deny: Sequence[re.Pattern] = (), home: str | None = None) -> list[Finding]:
    """Findings in the files git tracks right now."""
    root = Path(root)
    patterns = _private_patterns(home, deny)
    findings: list[Finding] = []
    for name in _git(root, "ls-files", "-z").split("\0"):
        if not name:
            continue
        generated = _is_generated(name)
        if generated:
            findings.append(Finding("generated-file", name, f"is tracked but matches {generated!r}"))
            continue
        try:
            text = (root / name).read_bytes().decode("utf-8")
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue                                    # binary files are caught by name above, not by content
        findings.extend(_scan(text, name, patterns))
    return findings


def check_history(root: str | os.PathLike, *, deny: Sequence[re.Pattern] = (), home: str | None = None) -> list[Finding]:
    """Findings in any commit on any ref: every file ever committed (by name, so binary files count), and every added line."""
    root = Path(root)
    patterns = _private_patterns(home, deny)
    findings: list[Finding] = []
    for name in sorted(set(_git(root, "log", "--all", "--no-renames", "--name-only", "--format=").splitlines())):
        generated = _is_generated(name) if name else None
        if generated:
            findings.append(Finding("generated-file", name, f"was committed and matches {generated!r}"))
    seen: set[tuple[str, str]] = set()
    commit = path = ""
    for line in _git(root, "log", "--all", "--no-renames", "--format=COMMIT %h", "-p", "-U0", "--no-color").splitlines():
        if line.startswith("COMMIT "):                    # patch lines all start with +, - , space or @@, never this
            commit = line[7:]
        elif line.startswith("+++ b/"):
            path = line[6:]
        elif line.startswith("+") and not line.startswith("+++") and not _is_generated(path):
            for finding in _scan(line[1:], f"{commit}:{path}", patterns):
                if (path, finding.detail) not in seen:
                    seen.add((path, finding.detail))
                    findings.append(finding)
    return findings


def identities(root: str | os.PathLike) -> list[Finding]:
    """The distinct author and committer identities in the history, as notices."""
    out = _git(Path(root), "log", "--all", "--format=%an <%ae>%n%cn <%ce>")
    return [Finding("identity", "history", who) for who in sorted(set(filter(None, out.splitlines())))]


def check(root: str | os.PathLike, *, history: bool = True, deny: Sequence[re.Pattern] | None = None,
          home: str | None = None) -> list[Finding]:
    """Every finding for the repository at ``root``. ``deny`` defaults to ``load_deny()``."""
    deny = load_deny() if deny is None else deny
    findings = check_tree(root, deny=deny, home=home)
    if history:
        findings += check_history(root, deny=deny, home=home)
        findings += identities(root)
    return findings


def problems(findings: Iterable[Finding]) -> list[Finding]:
    return [f for f in findings if f.is_problem]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("repository", nargs="?", default=".")
    parser.add_argument("--deny", help="file of regular expressions for names that must stay private")
    parser.add_argument("--no-history", action="store_true", help="check only the files tracked now")
    args = parser.parse_args(argv)
    findings = check(args.repository, history=not args.no_history, deny=load_deny(args.deny))
    for finding in findings:
        print(finding)
    bad = problems(findings)
    print(f"{len(bad)} problem(s), {len(findings) - len(bad)} notice(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
