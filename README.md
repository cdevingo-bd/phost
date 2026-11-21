# phost

Attach to a physics host, and swap, guard and measure its stages.

A stage is a named step of a host's computation with a machine-readable contract. An implementation
is written against the contract; a host adapter maps the host's own function onto it. phost provides
the contract types, one guarded and scoped install rule, a session that applies a selection of
implementations to a host, a registry that describes itself as JSON, and a contract check that
judges an implementation against a float64 reference.

Today: the `newton.linear_solve` stage, MuJoCo Warp and MJLab adapters, and a simulator-free fake host.
The core imports only the standard library. See [docs/SPEC.md](docs/SPEC.md).

```python
import phost
from phost.hosts.mjwarp import MJWarpHost
from phost.stages import newton_linear
import coupledchol.phost_impl as coupled

registry = phost.default
newton_linear.register(registry); coupled.register(registry)

host = MJWarpHost(model, data)
with phost.Session(host, [phost.Selection("newton.linear_solve", coupled.SCAN)]) as session:
    ...  # warm up and capture graphs here
    print(session.metrics())
```

MIT licensed.
