import pytest

from phost import ContractError, Port, Role, StageContract, SwapClass, check_ports
from phost.stages import newton_linear as nl


def test_contract_roundtrips_through_json():
    again = StageContract.from_json(nl.CONTRACT.to_json())
    assert again == nl.CONTRACT


def test_contract_rejects_undeclared_dimension():
    with pytest.raises(ContractError, match="undeclared"):
        StageContract("s", 1, "", SwapClass.EXACT, dims=("a",), ports=(Port("x", Role.IN, "float32", ("b",)),))


def test_contract_rejects_tolerance_on_unknown_metric():
    from phost import Tolerance
    with pytest.raises(ContractError, match="unknown metric"):
        StageContract("s", 1, "", SwapClass.EXACT, dims=("a",), ports=(), tolerances=(Tolerance("nope", 1.0),))


def test_contract_rejects_duplicate_ports_and_bad_version():
    p = Port("x", Role.IN, "float32", ("a",))
    with pytest.raises(ContractError, match="duplicate"):
        StageContract("s", 1, "", SwapClass.EXACT, dims=("a",), ports=(p, p))
    with pytest.raises(ContractError, match="version"):
        StageContract("s", 0, "", SwapClass.EXACT, dims=("a",), ports=())


def test_from_dict_reports_malformed_input():
    with pytest.raises(ContractError, match="malformed"):
        StageContract.from_dict({"name": "x"})


def test_port_rejects_bad_dimension():
    with pytest.raises(ContractError, match="bad dimension"):
        Port("x", Role.IN, "float32", ("Bad Dim",))


class A:
    def __init__(self, shape, dtype):
        self.shape, self.dtype = shape, dtype


def _good(world=3, nv=5, pad=8):
    return dict(h=A((world, pad, pad), "float32"), grad=A((world, pad), "float32"), active=A((world,), "int32"),
                refactor=A((world,), "int32"), search=A((world, nv), "float32"), search_dot=A((world,), "float32"),
                newton_decrement=A((world,), "float32"))


def test_check_ports_accepts_padded_matrices():
    assert check_ports(nl.CONTRACT, _good(), {"world": 3, "nv": 5}) == []


def test_check_ports_reports_each_problem():
    bad = _good()
    bad["h"] = A((3, 4, 4), "float32")          # smaller than nv
    bad["search"] = A((2, 5), "float64")        # wrong world count and dtype
    bad.pop("active")
    bad["extra"] = A((1,), "float32")
    text = "\n".join(check_ports(nl.CONTRACT, bad, {"world": 3, "nv": 5}))
    for needle in ("'h'", "'search'", "float64", "missing port 'active'", "unknown ports"):
        assert needle in text


def test_check_ports_binds_symbols_from_the_ports_when_unbound():
    bad = _good()
    bad["search"] = A((4, 5), "float32")        # world 3 elsewhere
    assert any("world=3" in p for p in check_ports(nl.CONTRACT, bad))
