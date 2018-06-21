try:
    from . import version
except ImportError:  # pragma: no cover
    from . import _version as version

__version__ = version.version

from .dask_traj import DaskTraj
