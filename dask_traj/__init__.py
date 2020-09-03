try:
    from . import version
except ImportError:  # pragma: no cover
    from . import _version as version

__version__ = version.version

from .core import *
from .geometry import *
from .utils import ensure_type
