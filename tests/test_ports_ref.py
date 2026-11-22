"""``Ref``: numpy's in-place syntax over an array that cannot be written in place."""

import numpy as np
import pytest

from phost.ports import Ref, unwrap

jnp = pytest.importorskip("jax.numpy", reason="needs jax")


def test_a_whole_array_write_replaces_the_array_and_casts_to_its_dtype():
    ref = Ref(jnp.zeros((2, 3), jnp.float32))
    ref[...] = np.ones((2, 3), np.float64)
    assert ref.dtype == jnp.float32 and np.asarray(ref).tolist() == [[1.0] * 3] * 2
    ref[:] = 5.0                                              # broadcast
    assert np.asarray(ref).tolist() == [[5.0] * 3] * 2


def test_a_slice_write_is_functional_and_leaves_the_rest():
    original = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    ref = Ref(original)
    ref[:, 1:] = -1.0
    assert np.asarray(ref).tolist() == [[0.0, -1.0, -1.0], [3.0, -1.0, -1.0]]
    assert np.asarray(original).tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]       # the array it replaced is untouched


def test_in_place_arithmetic_rebinds_the_cell():
    ref = Ref(jnp.ones(3, jnp.float32))
    same = ref
    ref += 2.0
    ref *= Ref(jnp.full(3, 2.0, jnp.float32))
    assert ref is same and np.asarray(ref).tolist() == [6.0, 6.0, 6.0]


def test_a_cell_reads_as_an_array_and_converts_to_numpy():
    ref = Ref(jnp.arange(4, dtype=jnp.float32))
    assert ref.shape == (4,) and len(ref) == 4 and ref.ndim == 1 and ref.size == 4
    assert np.asarray(ref).dtype == np.float32 and np.asarray(ref, np.float64).dtype == np.float64
    assert float((ref * 2.0 + 1.0)[3]) == 7.0 and float((2.0 - ref)[1]) == 1.0 and float((1.0 / (ref + 1.0))[1]) == 0.5
    assert unwrap(ref) is ref.value and unwrap(3) == 3


def test_a_cell_over_a_numpy_array_writes_in_place_so_one_code_path_serves_both():
    base = np.zeros(3, np.float32)
    ref = Ref(base)
    ref[1] = 4.0
    ref[...] = ref.value + 1.0
    assert base.tolist() == [1.0, 5.0, 1.0] and ref.value is base


def test_a_whole_array_write_never_reads_the_old_value():
    """A stale tracer left in a cell by a finished trace must not be touched by the next whole-array write."""
    class Stale:
        shape, dtype = (2,), jnp.float32

        def __getattr__(self, name):
            raise AssertionError(f"the old value was read ({name})")

    ref = Ref.__new__(Ref)
    ref.value = Stale()
    with pytest.raises(AssertionError):
        ref[1] = 1.0                                            # a partial write needs the base, so it does read it
    fresh = Ref(jnp.zeros(2, jnp.float32))
    fresh[...] = jnp.ones(2)
    assert np.asarray(fresh).tolist() == [1.0, 1.0]
