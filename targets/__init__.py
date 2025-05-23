"""
Target registry and selection.
"""

_targets = {}

def register_target(name):
    """
    Decorator to register a target configuration under a name.
    """
    def decorator(fn):
        _targets[name] = fn
        return fn
    return decorator


def get_target(name):
    """
    Retrieve target configuration by name.
    """
    try:
        return _targets[name]()
    except KeyError:
        raise KeyError(f"Target '{name}' not found") 