# phost

Put your own implementations behind typed contracts at the named stages of a simulation step, and check them against a float64
reference.

A **stage** is a named place in a simulator's step, such as a linear solve, a passive force or a controller. Each stage has a
**contract**: the arrays it reads and writes with their shapes and dtypes, a swap class (exact, approximate, model changing or
observer), metrics and tolerances, and a float64 reference. An implementation is written against the contract and never against
a simulator. A **host adapter** maps one simulator's own function onto the contract, so swapping the simulator means swapping an
adapter, and swapping the math means swapping an implementation. A session installs an implementation for the length of a `with`
block, behind a check that the hooked function's source is one that has been reviewed, and removes it on every way out.
`check_impl` runs an implementation on a host and judges it against the float64 reference.

![Implementations on the left are written against a stage contract in the middle (ports, swap class, metrics and tolerances, a float64 reference, a judge); host adapters on the right map a simulator's own function onto it: MuJoCo Warp, MJLab, a NumPy chain, a JAX chain, a fake host, or yours. Below: a session installs an implementation behind a source guard for a with-block, and check_impl judges it against the reference.](docs/architecture.svg)

The core imports only the standard library. Six stages exist today ([docs/CATALOG.md](docs/CATALOG.md)), and hosts of different
shapes ship with it:

| host | what it is | needs |
| --- | --- | --- |
| `phost.hosts.mjwarp` | [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) on a CUDA device: Warp arrays, the step captured as a CUDA graph. The worked example below. Only the 3.17.0 source build has been reviewed; the adapter refuses any other | warp, mujoco_warp |
| `phost.hosts.mjwarp.MJLabHost` | MJLab, through the MuJoCo Warp data it holds. Tested against a stand-in, not a real MJLab | as above |
| `phost.hosts.chain` | a damped spring-mass chain stepped in NumPy: its own names (`x`, `v`, `a`), a banded effective system, no MuJoCo and no GPU. It offers the stages a chain has | numpy |
| `phost.hosts.jaxchain` | the same chain over JAX arrays, which cannot be written in place: the step compiles | jax |
| `phost.testing.fake` | no simulator: a host with faulty runtimes, so the check itself can be tested | numpy |

The chain hosts exist to answer one question, whether the contracts and everything built on them belong to phost or only ever fit
one simulator. A host offers the stages it has (a chain has no bodies, so no `body.wrench`), and the same typed equations,
implementations and checks run on all of them.

## How it fits together

1. **A contract in the middle.** [docs/SPEC.md](docs/SPEC.md) is the specification. A stage is data: a contract, a judge that
   says how a run of it is measured, and a binding that says where it attaches ([docs/ADDING_A_STAGE.md](docs/ADDING_A_STAGE.md)).
   The swap class says what kind of replacement the stage accepts, because judging depends on it: the same math in a different
   algorithm (exact), the same math to a bounded error (approximate), different physics (model changing), or a read-only
   measurement (observer).
2. **Implementations on one side.** A typed equation or solver step from
   [phost-author](https://github.com/cdevingo-bd/phost-author) (compiled to NumPy, Warp or JAX), a sandboxed kernel,
   [phost-coupledchol](https://github.com/cdevingo-bd/phost-coupledchol), the built-in laws, or your own class with the stage's
   entry method.
3. **Hosts on the other.** The adapter derives the contract's ports from the host's own arrays (an alias, or one kernel launch),
   calls the implementation, writes the result back, and lets the host function run, run first, or be skipped. Stage dispatch is
   data (`phost.hook.Hook`), shared by every host.
4. **Install, then check.** The session refuses a hooked function whose source is not in the reviewed table for that build, holds
   the patch for the length of a `with` block, and restores on every exit. `check_impl` drives the host, records what the
   implementation was handed and wrote, and compares it with a float64 reference within the contract's tolerances. The same check
   gates a submission, a graph run and a leaderboard entry.
5. **Then measure.** [phost-lab](https://github.com/cdevingo-bd/phost-lab) runs a graph in an isolated worker on any engine it has
   against native and a native control, and reports where a change could pay and where it applies;
   [phost-score](https://github.com/cdevingo-bd/phost-score) applies the gate for the swap class and keeps a hash-chained ledger and
   a Pareto board; [phost-dash](https://github.com/cdevingo-bd/phost-dash) is the browser view over all of it, and
   [phost-agent](https://github.com/cdevingo-bd/phost-agent) lets language models search.

## An example with no GPU

```python
import phost
from phost.hosts.chain import ChainHost, make_model
from phost.impls import register_builtin
from phost.stages import register_all
from phost.testing.contract import check_impl

registry = phost.Registry()
register_all(registry); register_builtin(registry)

host = ChainHost(make_model(24, 3), worlds=3)        # a spring-mass chain in NumPy: no MuJoCo, no GPU
selection = phost.Selection("dof.passive_force", "dof.passive_force/spring-damper", dict(k=3.0, c=0.4))
verdict = check_impl(host, selection, registry)
print(verdict.ok, verdict.metrics["max_law_error"])   # True, about 4e-8: the law matches its float64 reference
```

## The worked example: MuJoCo Warp

![MuJoCo Warp's step as six functions with six numbered splice points and a table of what each reads and writes, and one splice in detail: derive the contract's ports from device arrays, call the implementation, write back, with a source-hash guard and a scoped install.](docs/example-mjwarp.svg)

`mjw.step` is a fixed sequence of functions, captured as one CUDA graph, and phost can splice at six points (the numbered squares).
The table gives, for each, the hooked function, whether the implementation runs before, after or instead of it, the contract's
ports and the swap class. Only the mujoco_warp 3.17.0 source build has been reviewed; the adapter refuses any other and names the
hash it found.

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

## Adding a host

A host is a subclass of `phost.host.Host`, and the core needs nothing else from a simulator:

- `info` and `capabilities`: who it is, and whether stages can be replaced;
- `facts()`: portable facts an implementation checks to decide whether it applies (`nv`, world count, solver, integrator, device kind);
- `stages()`: a `StageBinding` for each stage it offers: the contract, the function to patch with a source-hash guard for each
  assumption, how a call becomes the contract's ports (a `Hook` table, or a function written by hand), why a stage cannot be
  installed on this model, how to time the host's own version, and, if it can say, the matrix's sparsity;
- `meter()`: how calls are counted, so a stage's `calls` come from the host and not from the implementation.

`phost/hosts/chain.py` is a complete host of about 375 lines, and `phost.testing.contract.check_impl` runs against it unchanged. To
run scenes of a new simulator through phost-lab, add an `Engine` there (see its README).

## The set

phost is one of seven packages. None is on PyPI yet, so install from a clone, `phost` first:

| package | what it is |
| --- | --- |
| [phost](https://github.com/cdevingo-bd/phost) | this: stage contracts, the guarded install, host adapters, the contract check |
| [phost-coupledchol](https://github.com/cdevingo-bd/phost-coupledchol) | a coupled-block Cholesky solver in Warp, as an implementation of two stages |
| [phost-lab](https://github.com/cdevingo-bd/phost-lab) | runs a graph of implementations on any engine in an isolated worker; timing, profile, opportunity and applicability reports |
| [phost-author](https://github.com/cdevingo-bd/phost-author) | typed equations compiled to NumPy, Warp or JAX, and a sandbox for submitted kernels |
| [phost-score](https://github.com/cdevingo-bd/phost-score) | gates by swap class, a Pareto board, a hash-chained ledger |
| [phost-dash](https://github.com/cdevingo-bd/phost-dash) | a local browser view over the above; the work stays in the lab's workers |
| [phost-agent](https://github.com/cdevingo-bd/phost-agent) | models compete on a leaderboard behind a funnel of cheap checks, with a memo, a device lease and budgets; any provider |

```
git clone https://github.com/cdevingo-bd/phost && pip install -e phost     # repeat for the others, in the order above
```

The chain hosts need NumPy (and JAX for `jaxchain`). The GPU host is tested against the MuJoCo Warp 3.17.0 source build (with
`mujoco==3.17.0`, `warp-lang==1.16.0`); the adapter refuses a build it has not been reviewed against.

MIT licensed.
