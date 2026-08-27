"""Analysis modules. Every one is a pure function over the extracted layer:
no network, no model, no I/O. All of them emit the same `Finding` type.
"""

from api.findings.adversarial import detect_adversarial
from api.findings.asymmetry import measure_asymmetry
from api.findings.backtoback import find_gaps
from api.findings.silence import detect_silence
from api.findings.termination import termination_cost

__all__ = [
    "detect_adversarial",
    "measure_asymmetry",
    "find_gaps",
    "detect_silence",
    "termination_cost",
]
