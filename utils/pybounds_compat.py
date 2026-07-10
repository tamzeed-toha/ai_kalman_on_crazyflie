"""Runtime compatibility shim for pybounds under newer numpy releases.

Some numpy releases no longer allow implicit conversion of a size-1, non-0-d array to a Python
scalar via int()/np.round(). do_mpc passes its internal time (self._t0, shape (1,)) into
pybounds.Simulator.simulator_tvp_function / mpc_tvp_function, which do exactly that conversion,
so on such numpy versions Simulator.simulate(mpc=True) raises:
    TypeError: only 0-dimensional arrays can be converted to Python scalars

patch_pybounds_simulator_time_conversion() patches around it at runtime (no installs, no version
changes) by coercing the time argument to a plain float before handing it to the original
methods. It must be called before any pybounds.Simulator (or subclass) instance is constructed,
since Simulator.__init__ captures these methods as callbacks at construction time.
"""

import numpy as np
from pybounds import Simulator as _PyBoundsSimulator

_PATCHED = False


def _scalar_time(t):
    """Coerce a possibly-array-shaped time value to a plain Python float."""
    return float(np.asarray(t).reshape(-1)[0])


def patch_pybounds_simulator_time_conversion():
    """Idempotently patch pybounds.Simulator's tvp callbacks to tolerate newer numpy."""
    global _PATCHED
    if _PATCHED:
        return

    orig_simulator_tvp_function = _PyBoundsSimulator.simulator_tvp_function
    orig_mpc_tvp_function = _PyBoundsSimulator.mpc_tvp_function

    def patched_simulator_tvp_function(self, t):
        return orig_simulator_tvp_function(self, _scalar_time(t))

    def patched_mpc_tvp_function(self, t):
        return orig_mpc_tvp_function(self, _scalar_time(t))

    _PyBoundsSimulator.simulator_tvp_function = patched_simulator_tvp_function
    _PyBoundsSimulator.mpc_tvp_function = patched_mpc_tvp_function
    _PATCHED = True
