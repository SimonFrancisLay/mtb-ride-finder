# region-enabled shim
from dataclasses import dataclass
from typing import Tuple
# Import existing symbols from current hotfix; keep names intact.
try:
    from mtb_agent import *  # noqa: F401,F403
except Exception:
    pass

# Redefine Location to include region if not present; keep compatibility
try:
    # If original Location lacks region, define a new one for typing only.
    from mtb_agent import Location as _OldLocation  # type: ignore
    if "region" not in _OldLocation.__annotations__:
        @dataclass
        class Location(_OldLocation):  # type: ignore
            region: str = ""
except Exception:
    pass
