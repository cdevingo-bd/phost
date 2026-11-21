"""Exceptions raised by phost. Every refusal names what was found and what was expected."""


class PhostError(Exception):
    """Base class."""


class ContractError(PhostError):
    """A stage contract is malformed, or values violate it."""


class GuardError(PhostError):
    """A source guard refused an install: the host build was not reviewed, or is already patched."""


class CapabilityError(PhostError):
    """A host lacks a capability, or a stage binding, that was asked for."""


class ApplicabilityError(PhostError):
    """An implementation declares that it does not apply to this model."""
