"""Ports on hosts whose arrays cannot be written in place.

A stage contract says an implementation writes its outputs into the arrays it is handed. NumPy and Warp arrays are mutable, so
that is an assignment. A JAX array is not: the only way to change one is to make another. A host over such arrays hands out a
``Ref`` instead: a cell that holds the current array and takes numpy's in-place syntax,

    system.force[...] = law            # replace the whole array
    system.damping[:, :3] = part       # replace a slice (functionally: ``value.at[...].set``)
    system.passive_force += extra

and the host reads ``ref.value`` back when the stage returns. An implementation written for numpy's syntax runs unchanged; one
written for JAX (``unwrap(system.q)`` to read, assign to write) is the same function whether or not it is jitted, because the
cell is rebound at trace time and the host carries what it holds as state through the compiled step.

The class names no array library: it calls ``.at[...].set`` when the value has one and assigns in place when it does not, so
it works over NumPy arrays too (a host can use one code path for both).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def unwrap(value):
    """The array inside a ``Ref``, or ``value`` itself."""
    return value.value if isinstance(value, Ref) else value


class Ref:
    """A mutable cell around an immutable array. See the module docstring."""

    __slots__ = ("value",)

    def __init__(self, value: Any):
        self.value = value

    # -- what the array is
    shape = property(lambda self: tuple(self.value.shape))
    dtype = property(lambda self: self.value.dtype)
    ndim = property(lambda self: len(self.value.shape))
    size = property(lambda self: int(np.prod(self.value.shape)))

    def __len__(self) -> int:
        return int(self.value.shape[0])

    def __repr__(self) -> str:
        return f"Ref({self.value!r})"

    def __array__(self, dtype=None, copy=None):
        array = np.array(self.value, copy=True)
        return array if dtype is None else array.astype(dtype)

    def __jax_array__(self):
        return self.value

    def astype(self, dtype):
        return self.value.astype(dtype)

    def __getitem__(self, index):
        return self.value[index]

    # -- writing
    def set(self, new) -> None:
        """Replace the whole array (cast to this array's dtype, broadcast to its shape)."""
        new = unwrap(new)
        if hasattr(self.value, "at"):
            import jax.numpy as jnp
            self.value = jnp.broadcast_to(jnp.asarray(new).astype(self.value.dtype), self.value.shape)
        else:
            self.value[...] = new

    def __setitem__(self, index, new) -> None:
        if index is Ellipsis or (isinstance(index, slice) and index == slice(None)) :
            self.set(new)              # a whole-array write never reads the old value, so a stale one cannot leak into a trace
        elif hasattr(self.value, "at"):
            self.value = self.value.at[index].set(unwrap(new))
        else:
            self.value[index] = unwrap(new)

    def __iadd__(self, other):
        self.set(self.value + unwrap(other))
        return self

    def __isub__(self, other):
        self.set(self.value - unwrap(other))
        return self

    def __imul__(self, other):
        self.set(self.value * unwrap(other))
        return self

    # -- reading as an array
    def __neg__(self):
        return -self.value

    def __add__(self, other):
        return self.value + unwrap(other)

    def __radd__(self, other):
        return unwrap(other) + self.value

    def __sub__(self, other):
        return self.value - unwrap(other)

    def __rsub__(self, other):
        return unwrap(other) - self.value

    def __mul__(self, other):
        return self.value * unwrap(other)

    def __rmul__(self, other):
        return unwrap(other) * self.value

    def __truediv__(self, other):
        return self.value / unwrap(other)

    def __rtruediv__(self, other):
        return unwrap(other) / self.value
