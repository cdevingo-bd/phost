"""phost: attach to a physics host, and swap, guard and measure its stages.

The core imports only the standard library. Hosts and arrays come from adapters and from the
packages that implement stages.
"""

from .errors import ApplicabilityError, CapabilityError, ContractError, GuardError, PhostError
from .guard import Guard, source_hash
from .host import Capability, Host, HostInfo, ModelFacts, StageBinding
from .impl import Impl, ImplKind, ParamSpec, StageRuntime
from .patch import PatchPoint, Target, install, original_of
from .registry import Registry, default
from .session import Selection, Session
from .stage import (Extension, MetricSpec, Port, Role, StageContract, SwapClass, Tolerance, check_ports)

__version__ = "0.1.0.dev0"

__all__ = [
    "ApplicabilityError", "CapabilityError", "ContractError", "GuardError", "PhostError", "Guard", "source_hash",
    "Capability", "Host", "HostInfo", "ModelFacts", "StageBinding", "Impl", "ImplKind", "ParamSpec", "StageRuntime",
    "PatchPoint", "Target", "install", "original_of", "Registry", "default", "Selection", "Session", "Extension", "MetricSpec",
    "Port", "Role", "StageContract", "SwapClass", "Tolerance", "check_ports",
]
