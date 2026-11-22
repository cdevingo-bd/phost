"""Registries of stage contracts and implementations. Tools enumerate these as JSON."""

from __future__ import annotations

from typing import Iterable

from .errors import ContractError
from .impl import Impl
from .judge import Judge, StageSpec
from .stage import StageContract, SwapClass


class Registry:
    def __init__(self):
        self._contracts: dict[str, StageContract] = {}
        self._impls: dict[str, Impl] = {}
        self._judges: dict[str, Judge] = {}
        self._equations: dict[str, object] = {}

    def add_stage(self, spec: StageSpec) -> StageSpec:
        """Register a stage: its contract and how it is judged. Adding the same stage again changes nothing."""
        self.add_contract(spec.contract)
        old = self._judges.get(spec.contract.key)
        if old is not None and old != spec.judge:
            raise ContractError(f"stage {spec.contract.key} already registered with a different judge")
        old_form = self._equations.get(spec.contract.key)
        if old_form is not None and old_form != spec.equation:
            raise ContractError(f"stage {spec.contract.key} already registered with a different equation form")
        self._judges[spec.contract.key] = spec.judge
        if spec.equation is not None:
            self._equations[spec.contract.key] = spec.equation
        return spec

    def equation(self, name: str, version: int | None = None):
        """The form a typed law for this stage takes (``phost.equation``), or None when the stage takes none."""
        return self._equations.get(self.contract(name, version).key)

    def equation_stages(self) -> list[str]:
        """Names of the stages that take a typed law, in registration order (the latest version of each)."""
        return [c.name for c in self._contracts.values() if c.key in self._equations
                and c.version == max(x.version for x in self._contracts.values() if x.name == c.name)]

    def judge(self, name: str, version: int | None = None) -> Judge:
        contract = self.contract(name, version)
        try:
            return self._judges[contract.key]
        except KeyError:
            raise ContractError(f"stage {contract.key} has no judge registered: add it with Registry.add_stage(StageSpec(...)) "
                                "so check_impl knows how to measure it") from None

    def add_contract(self, contract: StageContract) -> StageContract:
        old = self._contracts.get(contract.key)
        if old is not None and old != contract:
            raise ContractError(f"contract {contract.key} already registered with different content")
        self._contracts[contract.key] = contract
        return contract

    def contract(self, name: str, version: int | None = None) -> StageContract:
        if version is not None:
            found = self._contracts.get(f"{name}@{version}")
        else:
            versions = [c for c in self._contracts.values() if c.name == name]
            found = max(versions, key=lambda c: c.version) if versions else None
        if found is None:
            raise ContractError(f"no contract {name!r}" + (f" version {version}" if version else ""))
        return found

    def add_impl(self, impl: Impl) -> Impl:
        if impl.id in self._impls:
            raise ContractError(f"implementation {impl.id!r} already registered")
        contract = self._contracts.get(f"{impl.stage}@{impl.stage_version}")
        if contract is None:
            raise ContractError(f"impl {impl.id!r} targets unknown contract {impl.stage}@{impl.stage_version}")
        known = {m.name for m in contract.metrics}
        clash = known & {m.name for m in impl.metrics}
        if clash:
            raise ContractError(f"impl {impl.id!r} redefines contract metrics {sorted(clash)}")
        if impl.swap_class is not None and contract.swap_class not in (SwapClass.EXACT, SwapClass.APPROXIMATE):
            raise ContractError(f"impl {impl.id!r}: stage {contract.name!r} is {contract.swap_class.value}, which an "
                                "implementation cannot relax")
        if contract.reference == "impl" and impl.reference is None and impl.kind.value != "native":
            raise ContractError(f"impl {impl.id!r}: stage {contract.name!r} requires every implementation to supply a "
                                "float64 reference")
        self._impls[impl.id] = impl
        return impl

    def impl(self, impl_id: str) -> Impl:
        try:
            return self._impls[impl_id]
        except KeyError:
            raise ContractError(f"no implementation {impl_id!r}; known {sorted(self._impls)}") from None

    def impls_for(self, stage: str) -> list[Impl]:
        return [i for i in self._impls.values() if i.stage == stage]

    def contracts(self) -> Iterable[StageContract]:
        return list(self._contracts.values())

    def describe(self) -> dict:
        return dict(contracts=[dict(c.to_dict(), equation=self._equations[c.key].to_dict() if c.key in self._equations else None)
                               for c in self._contracts.values()],
                    impls=[i.describe() for i in self._impls.values()])


default = Registry()
