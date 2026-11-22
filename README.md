# phost

Attach to a physics host, and swap, guard and measure its stages.

A stage is a named step of a host's computation with a machine-readable contract. An implementation
is written against the contract; a host adapter maps the host's own function onto it. phost provides
the contract types, one guarded and scoped install rule, a session that applies a selection of
implementations to a host, a registry that describes itself as JSON, and a contract check that
judges an implementation against a float64 reference.

Today: four stages (an exact solver swap, a passive force law, a controller and a probe; see
[docs/CATALOG.md](docs/CATALOG.md)), MuJoCo Warp and MJLab adapters, and a simulator-free fake host.
The core imports only the standard library. See [docs/SPEC.md](docs/SPEC.md).

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
