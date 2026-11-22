"""The nonzero structure of a system matrix, as something a host provides and an implementation reads.

A stage whose ``h`` port is a dense matrix hides the fact that the matrix is mostly zeros. A direct solve that exploits the
structure (factor only the coupled rows, or reorder to a narrow band and factor that) needs the pattern, and the pattern is a
fact about the host's model, not about any one solve. ``Sparsity`` is that fact: the symmetric pattern in compressed rows,
with the questions a solver asks of it (which rows have an off-diagonal entry, how wide is the band, what order narrows it).

It is pure Python so the core keeps no required dependency; arrays are tuples and the sizes involved (hundreds to a few
thousand rows) do not need more. A stage offers it as the extension ``sparsity``: an implementation that declares the
extension is handed ``system.sparsity``; one that does not is handed ``None``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class Sparsity:
    """The symmetric nonzero pattern of an ``n`` x ``n`` matrix, diagonal included, as compressed rows.

    ``colind[rowptr[i]:rowptr[i + 1]]`` are the columns of row ``i``, sorted and without repeats. Build one with
    ``from_pairs``, ``from_dense``, ``banded`` or ``dense``; the constructors symmetrize and add the diagonal.
    """

    n: int
    rowptr: tuple[int, ...]
    colind: tuple[int, ...]

    # ---- construction ----

    @staticmethod
    def from_pairs(n: int, pairs: Iterable[tuple[int, int]]) -> "Sparsity":
        """The pattern with an entry at every ``(row, col)`` of ``pairs``, its transpose, and the diagonal."""
        rows: list[set[int]] = [{i} for i in range(n)]
        for i, j in pairs:
            if not (0 <= i < n and 0 <= j < n):
                raise ValueError(f"entry ({i}, {j}) is outside a {n} x {n} matrix")
            rows[i].add(j)
            rows[j].add(i)
        rowptr, colind = [0], []
        for row in rows:
            colind.extend(sorted(row))
            rowptr.append(len(colind))
        return Sparsity(n, tuple(rowptr), tuple(colind))

    @staticmethod
    def from_dense(matrix: Any, tol: float = 0.0) -> "Sparsity":
        """The pattern of the entries of ``matrix`` (any 2-D array) whose magnitude exceeds ``tol``."""
        import numpy as np
        a = np.abs(np.asarray(matrix))
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError("a matrix must be square")
        rows, cols = np.nonzero(a > tol)
        return Sparsity.from_pairs(int(a.shape[0]), zip(rows.tolist(), cols.tolist()))

    @staticmethod
    def banded(n: int, half_bandwidth: int) -> "Sparsity":
        return Sparsity.from_pairs(n, ((i, j) for i in range(n) for j in range(max(0, i - half_bandwidth), i)))

    @staticmethod
    def dense(n: int) -> "Sparsity":
        return Sparsity.banded(n, n)

    # ---- questions ----

    @property
    def nnz(self) -> int:
        return len(self.colind)

    def row(self, i: int) -> Sequence[int]:
        return self.colind[self.rowptr[i]:self.rowptr[i + 1]]

    def degrees(self) -> list[int]:
        """Off-diagonal entries per row."""
        return [self.rowptr[i + 1] - self.rowptr[i] - 1 for i in range(self.n)]

    def coupled_rows(self) -> list[int]:
        """1 for every row that has an off-diagonal entry, else 0 (the flags a Newton ``structure_hint`` sets)."""
        return [1 if d > 0 else 0 for d in self.degrees()]

    def bandwidth(self, order: Sequence[int] | None = None) -> int:
        """Largest ``|i - j|`` over entries, in the given order (``order[k]`` is the original index placed at position k)."""
        position = list(range(self.n)) if order is None else _inverse(order, self.n)
        return max((abs(position[i] - position[j]) for i in range(self.n) for j in self.row(i)), default=0)

    def rcm_order(self) -> list[int]:
        """Reverse Cuthill-McKee: a reordering that tends to narrow the band. ``order[k]`` is the original index at position k."""
        degree = self.degrees()
        seen = [False] * self.n
        order: list[int] = []
        for start in sorted(range(self.n), key=lambda i: (degree[i], i)):
            if seen[start]:
                continue
            seen[start] = True
            queue = deque([start])
            while queue:
                i = queue.popleft()
                order.append(i)
                for j in sorted((j for j in self.row(i) if not seen[j]), key=lambda j: (degree[j], j)):
                    seen[j] = True
                    queue.append(j)
        return order[::-1]

    def describe(self) -> dict:
        """Size, density and what a structure-exploiting factorization could save, in floating-point operations.

        ``dense_flops`` is n^3 / 3 (dense Cholesky). ``coupled_flops`` factors only the coupled rows densely (the rest are
        diagonal): c^3 / 3 for c coupled rows. ``banded_flops`` is n b^2 for the band b left by reverse Cuthill-McKee.
        These are operation counts, not times: a kernel's time depends on how it moves data.
        """
        n, c = self.n, sum(self.coupled_rows())
        b = self.bandwidth(self.rcm_order())
        dense = n ** 3 / 3
        return dict(n=n, nnz=self.nnz, density=self.nnz / (n * n) if n else 0.0, coupled=c,
                    bandwidth=self.bandwidth(), bandwidth_rcm=b, dense_flops=dense,
                    coupled_flops=c ** 3 / 3, banded_flops=float(n * max(b, 1) ** 2))


def _inverse(order: Sequence[int], n: int) -> list[int]:
    if sorted(order) != list(range(n)):
        raise ValueError("order must be a permutation of 0..n-1")
    position = [0] * n
    for k, i in enumerate(order):
        position[i] = k
    return position
