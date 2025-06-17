REGISTRY = {}
from .gate import Gate
from .gate_v2 import Gate_v2
from .gate_v3 import Gate_v3
from .gate_v4 import Gate_v4

REGISTRY["gate"] = Gate
REGISTRY["gate_v2"] = Gate_v2
REGISTRY["gate_v3"] = Gate_v3
REGISTRY["gate_v4"] = Gate_v4