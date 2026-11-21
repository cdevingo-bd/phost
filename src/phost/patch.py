"""One guarded, scoped install rule for every patch point.

Replaces a function the host owns with a dispatch that routes the owner's calls to a
replacement and every other call to the original. The rule, for every stage:

1. The target must not already be patched by phost (no nested installs).
2. Every required guard must match a reviewed source hash, else GuardError naming the hash found.
3. Install and restore are serialized by one process-wide re-entrant lock.
4. The original is restored on every exit path.

Graph capture: a graph captured while installed keeps the replacement after the scope ends.
The scope brackets warmup and capture, not replay.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Mapping

from .errors import GuardError
from .guard import Guard

_LOCK = threading.RLock()
_MARK = "__phost_installed__"


@dataclass(frozen=True)
class Target:
    """Where a host keeps the function to replace: read it, write it."""

    label: str
    get: Callable[[], Callable]
    set: Callable[[Callable], None]


@dataclass(frozen=True)
class PatchPoint:
    """A replaceable host function plus the guards that protect assumptions about it.

    ``guards["target"]`` hashes the function being replaced and is always checked.
    Other guards cover further assumptions (for example the code that assembles an input) and are
    checked only when an implementation declares that it relies on them.
    """

    stage: str
    target: Target
    guards: Mapping[str, Guard]

    def __post_init__(self):
        if "target" not in self.guards:
            raise ValueError(f"patch point {self.stage!r} needs a 'target' guard")


def is_installed(function: Callable) -> bool:
    return bool(getattr(function, _MARK, False))


def original_of(function: Callable) -> Callable:
    """The host's own function behind a phost dispatch (or ``function`` itself when nothing is installed)."""
    return getattr(function, "__phost_original__", function)


@contextmanager
def install(point: PatchPoint, make_dispatch: Callable[[Callable], Callable],
            require: Iterable[str] = ()) -> Iterator[Callable]:
    """Install ``make_dispatch(original)`` at ``point`` for the duration of the scope."""
    names = ("target", *[n for n in require if n != "target"])
    with _LOCK:
        original = point.target.get()
        if is_installed(original):
            raise GuardError(f"{point.target.label}: already patched by phost (nested install)")
        for name in names:
            if name not in point.guards:
                raise GuardError(f"{point.stage}: no guard named {name!r}; known: {sorted(point.guards)}")
            point.guards[name].check()
        dispatch = make_dispatch(original)
        try:
            setattr(dispatch, _MARK, True)
            setattr(dispatch, "__phost_original__", original)
        except (AttributeError, TypeError) as error:
            raise GuardError(f"{point.target.label}: dispatch cannot be marked as installed: {error}") from error
        point.target.set(dispatch)
        try:
            yield dispatch
        finally:
            point.target.set(original)
