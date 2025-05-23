import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH when running as script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from targets.sim import Simulator

if __name__ == '__main__':
    import os
    RENDER = int(os.getenv('RENDER', '1'))
    render_mode = 'human' if RENDER else None
    sim = Simulator(render_mode=render_mode)
    sim.run() 