"""
Constraint-system sub-package for comgen.

Each module defines a family of Z3 constraints that can be combined to
describe a target composition space:

- :class:`TargetComposition` / :class:`UnitCell` -- core composition variables
  and charge-balance, element-count, and species-quantity constraints.
- :class:`EMD` -- Earth Mover's Distance between normalised element vectors.
- :class:`Synthesis` -- linear-combination constraints that ensure a
  composition can be made from given ingredient compositions.
- :class:`ONNX` -- encodes a feed-forward neural network (loaded from an
  ONNX file) as Z3 constraints for property prediction.
"""

from .composition import TargetComposition, UnitCell
from .distance import EMD
from .synthesis import Synthesis
from .nn import ONNX
