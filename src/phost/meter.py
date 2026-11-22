"""Counting how often a stage executes.

An implementation cannot count its own executions honestly. Its Python code runs once per eager call and once
per warmup or graph capture, never once per replay of a captured graph, and an implementation could report
anything. So the host owns the count, and the session ticks it at every execution of the stage.

A meter is whatever the host can count with. A host that runs eagerly counts in Python (``PythonMeter``). A host
that replays captured device work enqueues the increment as device work too, so the replay repeats it
(``phost.hosts.mjwarp`` has the Warp one).
"""

from __future__ import annotations

from typing import Protocol


class Meter(Protocol):
    def tick(self) -> None:
        """Record one execution. May enqueue device work; must be cheap."""

    def count(self) -> int:
        """Executions recorded so far. May synchronize the device."""

    def reset(self) -> None: ...


class PythonMeter:
    """Counts calls made from Python. Right for hosts that execute every call eagerly."""

    def __init__(self) -> None:
        self._count = 0

    def tick(self) -> None:
        self._count += 1

    def count(self) -> int:
        return self._count

    def reset(self) -> None:
        self._count = 0


class Metered:
    """The runtime as the host's dispatch sees it: its entry method ticks the meter, everything else passes through.

    Looks the entry method up on the runtime at every call, so a wrapper installed on the runtime later (the
    conformance recorder does this) is still called.
    """

    def __init__(self, runtime, entry: str, meter: Meter):
        self.__dict__.update(_runtime=runtime, _entry=entry, _meter=meter)

    def __getattr__(self, name):
        member = getattr(self.__dict__["_runtime"], name)
        if name != self.__dict__["_entry"]:
            return member
        meter = self.__dict__["_meter"]

        def ticking(*args, **kwargs):
            meter.tick()
            return member(*args, **kwargs)

        return ticking
