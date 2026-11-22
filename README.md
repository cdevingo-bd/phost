# phost

Attach to a physics host, and swap, guard and measure its stages.

A stage is a named step of a host's computation with a machine-readable contract. An implementation
is written against the contract; a host adapter maps the host's own function onto it. phost provides
the contract types, one guarded and scoped install rule, a session that applies a selection of
implementations to a host, a registry that describes itself as JSON, and a contract check that
judges an implementation against a float64 reference.

Today: five stages (two exact solver swaps, a passive force law, a controller and a probe; see
[docs/CATALOG.md](docs/CATALOG.md)), MuJoCo Warp and MJLab adapters, and a simulator-free fake host.
The core has no required dependencies. See [docs/SPEC.md](docs/SPEC.md).

## How it fits together

![How phost works: an implementation is written in phost-author, phost-coupledchol or your own code, registered against a stage contract, guarded and installed by a session on a host adapter that hooks the MuJoCo Warp step; phost-lab runs graphs in isolated workers, phost-score judges and ranks them, phost-dash shows it in a browser.](docs/architecture.svg)

Read it left to right.

1. **Write an implementation.** [phost-author](https://github.com/cdevingo-bd/phost-author) turns a typed equation or a sandboxed
   kernel into one, [phost-coupledchol](https://github.com/cdevingo-bd/phost-coupledchol) ships a ready-made one, and you can write your own.
   An implementation is written against a stage contract, never against a particular host.
2. **The core does the careful part.** The contract says what a stage's ports, tolerances and swap class are. A session takes a
   selection (stage, implementation, parameters), checks that the host's source is one that has been reviewed, installs the
   implementation for the length of a `with` block, and puts everything back on every way out of it.
3. **The host adapter reaches into the simulator.** It maps the host's own functions onto the contracts. The strip along the
   bottom is the MuJoCo Warp step: orange boxes are the places an implementation can sit, and the label says whether it changes
   the physics, replaces a solve with the same answer, or only reads.
4. **Run it, judge it, look at it.** [phost-lab](https://github.com/cdevingo-bd/phost-lab) runs a graph in an isolated worker against native and
   a native control, [phost-score](https://github.com/cdevingo-bd/phost-score) applies the gates for the swap class and keeps the ledger and the Pareto board, and
   [phost-dash](https://github.com/cdevingo-bd/phost-dash) is the browser view over all of it, running on your own GPU.

```python
import phost
from phost.hosts.mjwarp import MJWarpHost
from phost.stages import newton_linear
import phost_coupledchol.phost_impl as coupled

registry = phost.default
newton_linear.register(registry); coupled.register(registry)

host = MJWarpHost(model, data)
with phost.Session(host, [phost.Selection("newton.linear_solve", coupled.SCAN)]) as session:
    ...  # warm up and capture graphs here
    print(session.metrics())
```

## The set

phost is one of six packages. None is on PyPI yet, so install from a clone, `phost` first:

| package | what it is |
| --- | --- |
| [phost](https://github.com/cdevingo-bd/phost) | this: the host layer and the stage contracts |
| [phost-coupledchol](https://github.com/cdevingo-bd/phost-coupledchol) | the coupled-block Cholesky and its phost implementation |
| [phost-lab](https://github.com/cdevingo-bd/phost-lab) | graphs, scenes, an isolated runner and a live view feed |
| [phost-author](https://github.com/cdevingo-bd/phost-author) | typed equations and sandboxed kernels |
| [phost-score](https://github.com/cdevingo-bd/phost-score) | judging rules, a Pareto board and a tamper-evident ledger |
| [phost-dash](https://github.com/cdevingo-bd/phost-dash) | a local dashboard that runs it all on your own GPU |

```
git clone https://github.com/cdevingo-bd/phost && pip install -e phost     # repeat for the others, in the order above
```

The GPU parts are tested against the MuJoCo Warp 3.17.0 source build (with `mujoco==3.17.0`, `warp-lang==1.16.0`); the
adapter refuses a build it has not been reviewed against.

MIT licensed.
