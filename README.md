# phost

Hook the stages of a GPU physics step, put your own kernels behind a typed contract, and check them against a float64 reference.

phost attaches to MuJoCo Warp's step at named points. Each point is a **stage**, such as the Newton linear solve, the passive
force or the controller, and each stage has a **contract**: the arrays it reads and writes with their shapes and dtypes, a swap
class (exact, approximate, model changing or observer), metrics and tolerances. An implementation is written against the
contract and never against the host. A session installs it for the length of a `with` block, behind a check that the hooked
function's source is one that has been reviewed, and removes it on every way out. `check_impl` runs an implementation on a real
host and judges it against a float64 reference.

Six stages today (see [docs/CATALOG.md](docs/CATALOG.md)), MuJoCo Warp and MJLab adapters, a simulator-free fake host, and a NumPy spring-mass chain that is a host in its own right
(`phost.hosts.chain`: no MuJoCo, no GPU, its own vocabulary, and a banded effective system). The
core has no required dependencies. [docs/SPEC.md](docs/SPEC.md) is the specification;
[docs/ADDING_A_STAGE.md](docs/ADDING_A_STAGE.md) is how to add a stage.

## How it fits together

![MuJoCo Warp's step as six functions with six numbered splice points and a table of what each reads and writes; one splice in detail (derive the contract's ports from device arrays, call the implementation, write back, with a source-hash guard and a scoped install); and the check against a float64 reference, then the lab, score and dashboard that time, judge and show a graph.](docs/architecture.svg)

1. **The host step.** `mjw.step` is a fixed sequence of functions, captured as one CUDA graph. phost can splice at six points
   (the numbered squares). The table gives, for each, the hooked function, whether the implementation runs before, after or
   instead of it, the contract's ports, and the swap class. Only the mujoco_warp 3.17.0 source build has been reviewed; the
   adapter refuses any other.
2. **One splice.** The adapter derives the contract's ports from the host's device arrays (an alias, or one kernel launch),
   calls the implementation's entry method, writes the result back, and lets the host function run or skips it. An
   implementation comes from [phost-author](https://github.com/cdevingo-bd/phost-author) (a typed equation compiled to a Warp
   kernel, or a sandboxed kernel), [phost-coupledchol](https://github.com/cdevingo-bd/phost-coupledchol), or your own code.
   A stage is data: a contract, a judge that says how a run of it is measured, and a binding that says where it attaches
   ([docs/ADDING_A_STAGE.md](docs/ADDING_A_STAGE.md)).
3. **Check, then measure.** `check_impl` drives the host, records what the implementation was handed and wrote, and compares
   it with a float64 reference within the contract's tolerances. [phost-lab](https://github.com/cdevingo-bd/phost-lab)
   runs a graph in an isolated worker against native and a native control, and reports where a change could pay
   (`profile`, `opportunities`) and where it applies (`applicability`);
   [phost-score](https://github.com/cdevingo-bd/phost-score) applies the gate for the swap class and keeps a hash-chained
   ledger and a Pareto board; [phost-dash](https://github.com/cdevingo-bd/phost-dash) is the browser view over all of it.

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

phost is one of seven packages. None is on PyPI yet, so install from a clone, `phost` first:

| package | what it is |
| --- | --- |
| [phost](https://github.com/cdevingo-bd/phost) | this: stage contracts, the guarded install, host adapters, the contract check |
| [phost-coupledchol](https://github.com/cdevingo-bd/phost-coupledchol) | coupled-block Cholesky, as an implementation of two stages |
| [phost-lab](https://github.com/cdevingo-bd/phost-lab) | runs a graph of implementations in an isolated worker; timing, profile, opportunity and applicability reports |
| [phost-author](https://github.com/cdevingo-bd/phost-author) | typed equations compiled to Warp kernels, and a sandbox for submitted kernels |
| [phost-score](https://github.com/cdevingo-bd/phost-score) | gates by swap class, a Pareto board, a hash-chained ledger |
| [phost-dash](https://github.com/cdevingo-bd/phost-dash) | a local browser view over the above; the GPU work stays in the lab's workers |
| [phost-agent](https://github.com/cdevingo-bd/phost-agent) | what an agent is told, a funnel of cheap checks before a timed run, a memo, a device lease and budgets; it calls no model |

```
git clone https://github.com/cdevingo-bd/phost && pip install -e phost     # repeat for the others, in the order above
```

The GPU parts are tested against the MuJoCo Warp 3.17.0 source build (with `mujoco==3.17.0`, `warp-lang==1.16.0`); the
adapter refuses a build it has not been reviewed against.

MIT licensed.
