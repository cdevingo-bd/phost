"""Source guards: refuse to attach to a host build whose source nobody reviewed."""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from typing import Callable, Mapping

from .errors import GuardError


def source_hash(*objects) -> str:
    """SHA-256 of the concatenated source text of functions or classes.

    An object that keeps its Python function in ``.func`` (a Warp kernel does) is hashed by that function.
    """
    try:
        text = "".join(inspect.getsource(getattr(obj, "func", obj)) for obj in objects)
    except (OSError, TypeError) as error:
        raise GuardError(f"cannot read source to verify it: {error}") from error
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class Guard:
    """An assumption about the host, checked by hashing the source it depends on.

    ``read`` returns the current hash. ``reviewed`` maps accepted hashes to a label naming the
    build they were reviewed on. An unreviewed hash is refused; the error carries the hash found,
    which is what a reviewer adds after re-reviewing the new source.
    """

    name: str
    read: Callable[[], str]
    reviewed: Mapping[str, str]
    doc: str = ""

    def check(self) -> str:
        found = self.read()
        label = self.reviewed.get(found)
        if label is None:
            known = ", ".join(sorted(set(self.reviewed.values()))) or "none"
            raise GuardError(f"guard {self.name!r}: unreviewed source sha256 {found}; reviewed builds: {known}")
        return label
