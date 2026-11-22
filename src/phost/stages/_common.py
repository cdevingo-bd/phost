"""Helpers the pointwise stage modules share (numpy is imported lazily; the core stays stdlib-only)."""

from __future__ import annotations

from typing import Any, Mapping

from ..impl import Impl, ImplKind


def native_impl(stage: str, version: int, title: str, doc: str) -> Impl:
    return Impl(id=f"{stage}/native", stage=stage, stage_version=version, version="host", title=title,
                kind=ImplKind.NATIVE, doc=doc)


def relative_error(got, want, mask=None) -> float:
    """max |got - want| / max |want| over the masked entries, in float64. inf if got is not finite.

    ``mask`` selects entries along the last axis. A reference that is identically zero is judged on
    the absolute error.
    """
    import numpy as np
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    if mask is not None:
        keep = np.asarray(mask).astype(bool)
        got, want = got[..., keep], want[..., keep]
    if got.size == 0:
        return 0.0
    if not np.all(np.isfinite(got)):
        return float("inf")
    scale = float(np.max(np.abs(want)))
    err = float(np.max(np.abs(got - want)))
    return err / scale if scale > 0 else err


def padded_entries(port, array, run):
    """The entries of a padded system port the contract defines.

    A dimension written ``nv+`` is padded: only the leading ``nv`` entries mean anything. A port padded in both of its
    last two dimensions is a symmetric matrix of which only the upper triangle is valid. Everything else is returned whole.
    """
    import numpy as np
    array, nv = np.asarray(array), run["nv"]
    padded = [isinstance(dim, str) and dim.endswith("+") for dim in port.shape]
    if len(padded) >= 3 and padded[-1] and padded[-2]:
        rows, cols = np.triu_indices(nv)
        return array[:, rows, cols]
    if padded and padded[-1]:
        return array[:, :nv]
    return array
