# Adding a stage

A stage is added by writing data. Nothing in `check_impl`, `Session`, the lab or the leaderboard names a stage, so a new one
shows up in all of them, including the applicability matrix and the opportunity report, without editing any of them.

The recipe is three files and two one-line registrations. The worked example is `body.wrench` (an applied force and torque on
each body), which was added after the abstraction was written, as a check that it is not fitted to the stage it was taken from.

| file | what it holds | `body.wrench` |
| --- | --- | --- |
| `stages/<stage>.py` | the contract, the system the runtime is handed, and the judge | 88 lines |
| `hosts/mjwarp_<stage>.py` | the binding: where it attaches, where each port comes from, when it applies, how to time and check it | 48 lines |
| `tests/test_<stage>.py` | that the check passes a correct implementation and catches each fault | 170 lines (both hosts) |
| registrations | one name in `stages.MODULES`, one import at the bottom of `hosts/mjwarp.py` | 2 lines |
| `testing/fake_<stage>.py` (optional) | the same stage on the fake host, so it runs with no simulator and no GPU | 32 lines |

`effective.solve`, written before this existed, took 14 files and 625 lines: its own dispatch function, a branch in the
checker, a branch in the GPU driver, a blocker flag threaded through the host facts and a hand-picked scene in two other repos.

## 1. The stage: contract and judge

```python
CONTRACT = StageContract(name="body.wrench", version=1, swap_class=SwapClass.MODEL_CHANGING, dims=("world", "body"),
                         ports=(Port("xpos", Role.IN, "float32", ("world", "body", 3)), ...,
                                Port("wrench", Role.OUT, "float32", ("world", "body", 6))),
                         metrics=(MetricSpec("max_law_error", ...), MetricSpec("max_effect_error", ...), CALLS),
                         tolerances=(Tolerance("max_law_error", 1e-5, "..."), ...),
                         reference="impl", entry="apply")

SPEC = StageSpec(CONTRACT, Judge(evaluate=evaluate, repeatable=True))

def register(registry):
    registry.add_stage(SPEC)          # the contract and its judge, together
    ...                               # then the stage's own implementation, if it ships one
```

The contract is what the stage *is*: ports with shapes and dtypes, a swap class, metrics, tolerances, and the name of the
runtime's entry method. The judge is how a run of it is *measured*. `Judge.evaluate(call, reference, effect, ctx)` returns
the contract's metrics for one recorded call of the runtime: what it was handed and wrote (`call`), the float64 answer for
those inputs (`reference`), and what the host did with it (`effect`). The check keeps the worst of each metric over the calls
and compares it with the tolerances. Optional judge fields:

| field | when you need it |
| --- | --- |
| `reference` | the check should compute the float64 answer from a call's inputs (leave unset for `reference="impl"` or a residual) |
| `defined` | only some entries of a port mean anything (the Newton matrix is padded to a larger size) |
| `repeatable` | the same call twice must give identical outputs |
| `calls` | the drives to make, as keyword dicts for `host.drive`; a function of the host when they depend on its size |
| `driven` | the driver, not the check, observes the host's hook (only the Newton solve, whose inputs are built per host call) |

`check_impl` then does the rest from the contract: the runtime was called, inputs unchanged, outputs finite, repeatable,
and every metric inside its tolerance.

## 2. The binding: where it attaches

A `StageBinding` ties the contract to one host. It is a table, not a function (`phost/hook.py`; a host supplies the few operations a table needs through `hook_support()`):

```python
@adapter.register_binding
def body_wrench(host) -> StageBinding:
    forward = adapter._forward_module()
    hook = Hook("before", bw.WrenchSystem, ports=dict(xpos=Data("xpos"), cvel=Data("cvel"), time=Data("time"),
                                                      wrench=Data("xfrc_applied")))
    return StageBinding(bw.CONTRACT, adapter._point(bw.NAME, forward, "xfrc_accumulate", REVIEWED_XFRC),
                        hook.make(bw.CONTRACT.entry), native_timing=..., oracle=_oracle)
```

* `Hook(when, system, ports, prepare, finish, handles)`: `when` is `"before"`, `"after"` or `"instead"` of the host
  function. `ports` says where each of the contract's ports comes from: `Data("qvel")` (an array of the host's data),
  `Model(...)`, `Buf(...)` (a buffer the host owns), `Arg(i)` (an argument of the hooked call), `Dim(...)`, `Once(make)`,
  `When(extension, source)`. `prepare` and `finish` are steps that run around the runtime: `Launch(kernel, dim, inputs,
  outputs)` of a kernel in `_mjwarp_kernels`, and `Zero(target)`. `handles(args, kwargs)` says which calls the stage covers.
  A dispatch that does something a table cannot express is still a hand-written function: `make_dispatch` takes any callable.
* The patch point (`adapter._point`) carries the source hashes the hook was reviewed against. A build whose hash is not in the
  table is refused. The reading of the host that justifies the hook belongs in `MJWARP_STEP.md`.
* `applicable(host)`: the reason the stage cannot be installed on this host's model, or None. A session refuses with it, the
  applicability matrix prints it, and a check scene is chosen by it.
* `native_timing`: how to run the host's own version of the stage once (`NativeTiming(run, enclosing="")`), so the opportunity
  report can say what the stage is worth. If the host's version cannot run alone, `run` times the enclosing phase and
  `enclosing` names it; the report then labels the figure an upper bound and gives no speedup ceiling.
* `oracle(host)`: what the check reads off the host after a driven run, for the judge's host-side metric (the `effect`).

## 3. The test

Drive a correct implementation through `check_impl` on a real host and assert it passes; then one test per fault the contract
rules out (a wrong law, an input written, nondeterminism, a NaN, an adapter that drops the effect). Break the thing the
metric is meant to catch and assert the metric, and not some other one, reports it. The held-out stage's first draft of one
test assumed the wrong metric would fire; the test was rewritten to match what the contract actually checks.

## The fake host

`phost.testing.fake.FakeHost` takes the same tables: a factory `register_binding`-ed in `testing/fake_stages.py` (or its own
module, as `fake_body_wrench.py` is) gives the stage to the fake host with numpy arrays behind the sources, and
`FakeHost.drive` runs it through `check_impl` with no simulator. The `body.wrench` tests run on both hosts from one file.
A binding's `oracle` is what makes this work for any stage: the fake host's driver has no stage names in it either. What
`Launch` names runs as a Python function on the fake host (`fake.KERNELS`) and as a Warp kernel on the GPU host.

## A host that is not MuJoCo

`phost.hosts.chain.ChainHost` is a real simulator, a damped spring-mass chain in NumPy, and it takes the same tables: its
arrays are `x`, `v`, `a`, `t` and `command`, its model has `particle_scalar` where MuJoCo has `dof_*`, and each `Hook` in
`hosts/chain.py` says where a port comes from in those names. Its effective system is tridiagonal, so the `sparsity` extension
it hands out is `Sparsity.banded(n, 1)`, and its `implicit_damping` is a wrapper around its own assembly rather than around
an integrator. It offers four of the six stages and not `body.wrench` or `newton.linear_solve`: a host offers what it has.
The one thing a table assumes of a host is that the function it hooks is called as `f(model, data, *args)`.

A stage you add runs on the chain host if its binding is one more factory in `_BINDINGS`; whether it should depends on
whether the stage means anything for a chain. `tests/test_chain_host.py` runs the shipped implementations and the contract
checker on it, and `phost-author`'s `tests/test_chain_forms.py` runs typed equations on it.

## Structure of the system matrix

A stage whose port is a matrix can offer the `sparsity` extension (`effective.solve` does). An implementation that declares
`extensions_used=("sparsity",)` is handed `system.sparsity`, a `phost.sparsity.Sparsity`: the nonzero pattern of the
matrix, a fact about the host's model that holds for every call, with the questions a solver asks of it (which rows are
coupled, how wide the band is, which order narrows it). The binding's `sparsity(host)` gives the same fact to a tool without
installing anything; `phost-lab sparsity --scene S` prints it with the work a structure-exploiting factorization could save.
The pattern is static. The Newton solve's `structure_hint` is the per-call analogue (its pattern changes with the contacts),
and its flags are `Sparsity.coupled_rows()` of that call's matrix.

## Letting a person type the law

A stage says how a law for it can be typed by attaching an `equation` form to its `StageSpec` (`phost.equation`). The form is
data, checked against the contract when the spec is built, and it names no host, no array library and no code generator.

```python
EQUATION = Pointwise(element="body",
    inputs=(Name("vx", "cvel", 3), ..., Name("time", "time")),     # a name is one entry of a port: (port, column)
    outputs=(Name("fx", "wrench", 0), ..., Name("tz", "wrench", 5)))
SPEC = StageSpec(CONTRACT, Judge(...), equation=EQUATION)
```

| Form | One law is | Shipped for |
|---|---|---|
| `Pointwise` | an expression per output, evaluated for every element (DOF, actuator, body) of every world; an optional `mask` port and an optional `Slope` (the law's derivative, reported on a port) | `dof.passive_force`, `control.law`, `body.wrench` |
| `Channels` | one expression per column of a `(world, width)` port, over arrays, with `sum norm dot maxabs ...` and `name[3]` | `probe.state` |
| `Iteration` | a program (`init`, then `step` repeated) that solves `matrix x = rhs` per world, with `A @ name`, `dot`, `rhs`, `diag` | `effective.solve`, `newton.linear_solve` |

`phost-author` registers a `<stage>/expression` implementation for every stage in the registry that declares a form, so a new
stage with a form is typeable (and appears in the dashboard's equation tab) with no change there. How the text becomes code is
a backend's job (`phost_author.backends`: NumPy and Warp ship; a JAX or CUDA-C backend is one class and a `register_backend`
call, chosen by the host's `backend` name). The float64 reference is the NumPy backend run on the same checked trees, so the
contract check measures the generated code against exactly what was typed.

A typed solver is *approximate*: an `Impl` may declare `swap_class=SwapClass.APPROXIMATE` on an exact stage, and then
`Impl.tolerances(params)` replaces the stage's gates for the metrics it names, with the accuracy the program claims. The check
enforces the claim, the lab ranks the run as approximate (speed and accuracy, not speed alone), and any implementation that
supplies a float64 reference gets `max_program_error`: the generated code against what was typed, apart from whether what was
typed is any good.

## What stays outside the recipe

* A stage that needs a derived port no table source expresses (a reordering applied on the device, say) needs a new source or
  step in `hook.py`. Add it there once and every stage on every host can use it.
* Judging is host-neutral; a binding is per host. A second host (MJLab has its own adapter) needs its own binding module.

## Measuring where a change could pay, and where it applies

* `phost-lab applicability`: stages by scenes, from each binding's `applicable` and each implementation's own `applicable`.
* `phost-lab opportunities --scene S`: each stage's native share of the step, its speedup ceiling `1 / (1 - share)` for
  speed swaps, and whether it applies here and why not. See the phost-lab README.
* `phost-lab sparsity --scene S`: the structure of each stage's matrix. The same reports are a card in phost-dash's Build screen.
