"""
Query sub-package for comgen.

Provides the base :class:`Query` class that wraps the Z3 solver, along with
utility functions for the modified Pettifor scale and Shannon ionic radii.
"""

from .base import Query
from .common import PETTIFOR_KEYS, element_to_pettifor, get_radii