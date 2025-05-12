"""Compatibility shim.

This makes the project importable as `import maaya` even though the actual
implementation currently lives in the `src` top-level module.  It simply
imports the real package (called ``src``) and re-exports its public symbols.

Once the codebase is reorganised (e.g. by renaming the ``src`` directory to
``maaya``), this shim can be removed.
"""

import importlib as _importlib
import sys as _sys

# Ensure the root directory (two levels up from this file) is on sys.path so we
# can import the sibling ``src`` package regardless of the current working
# directory.
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parent.parent
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))

# Import the real implementation, which lives in ``src``.
_src = _importlib.import_module('src')

# Re-export everything declared public there.
for _name in getattr(_src, '__all__', dir(_src)):
    globals()[_name] = getattr(_src, _name)

__all__ = list(getattr(_src, '__all__', []))

# Also expose sub-modules so that ``import maaya.core`` works.
for _modname in list(_sys.modules.keys()):
    if _modname.startswith('src.'):
        _sys.modules['maaya' + _modname[3:]] = _sys.modules[_modname]

# Finally register this shim as the canonical module so future ``import maaya``
# statements get this object.
_sys.modules['maaya'] = _sys.modules[__name__] 