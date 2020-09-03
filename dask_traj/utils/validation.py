# Copy of various mdtraj utils that need to loosen, to work with dask
import collections
import warnings

import dask.array as da
import numpy as np
from mdtraj.utils.six.moves import zip_longest


class TypeCastPerformanceWarning(RuntimeWarning):
    pass


def ensure_type(
    val,
    dtype,
    ndim,
    name,
    length=None,
    can_be_none=False,
    shape=None,
    warn_on_cast=True,
    add_newaxis_on_deficient_ndim=False,
    cast_da_to_np=False,
):
    """Typecheck the size, shape and dtype of a numpy array, with optional
    casting.
    Parameters
    ----------
    val : {np.ndaraay, None}
        The array to check
    dtype : {nd.dtype, str}
        The dtype you'd like the array to have
    ndim : int
        The number of dimensions you'd like the array to have
    name : str
        name of the array. This is used when throwing exceptions, so that
        we can describe to the user which array is messed up.
    length : int, optional
        How long should the array be?
    can_be_none : bool
        Is ``val == None`` acceptable?
    shape : tuple, optional
        What should be shape of the array be? If the provided tuple has
        Nones in it, those will be semantically interpreted as matching
        any length in that dimension. So, for example, using the shape
        spec ``(None, None, 3)`` will ensure that the last dimension is of
        length three without constraining the first two dimensions
    warn_on_cast : bool, default=True
        Raise a warning when the dtypes don't match and a cast is done.
    add_newaxis_on_deficient_ndim : bool, default=False
        Add a new axis to the beginining of the array if the number of
        dimensions is deficient by one compared to your specification. For
        instance, if you're trying to get out an array of ``ndim == 3``,
        but the user provides an array of ``shape == (10, 10)``, a new axis
        will be created with length 1 in front, so that the return value is of
        shape ``(1, 10, 10)``.
    cast_da_to_np : bool, default=False
        Cast dask.arrays to np.arrays, this will trigger a computation on the
        dask array, which is needed to make them C-contigious in memory for
        several low-level mdtraj computations
    Notes
    -----
    The returned value will always be C-contiguous if it is an np.array or if
    cast_da_to_np is True.

    Returns
    -------
    typechecked_val : ndarray, None
        If `val=None` and `can_be_none=True`, then this will return None.
        Otherwise, it will return val (or a copy of val). If the dtype wasn't
        right, it'll be casted to the right shape. If the array was not
        C-contiguous, it'll be copied as well.
    """
    if can_be_none and val is None:
        return None

    if not isinstance(val, (np.ndarray, da.core.Array)):
        if isinstance(val, collections.Iterable):
            # If they give us an iterator, let's try...
            if isinstance(val, collections.Sequence):
                # sequences are easy. these are like lists and stuff
                val = np.array(val, dtype=dtype)
            else:
                # this is a generator...
                val = np.array(list(val), dtype=dtype)
        elif np.isscalar(val) and add_newaxis_on_deficient_ndim and ndim == 1:
            # special case: if the user is looking for a 1d array, and
            # they request newaxis upconversion, and provided a scalar
            # then we should reshape the scalar to be a 1d length-1 array
            val = np.array([val])
        else:
            raise TypeError(
                ("%s must be numpy array. " " You supplied type %s" % (name, type(val)))
            )

    if warn_on_cast and val.dtype != dtype:
        warnings.warn(
            "Casting %s dtype=%s to %s " % (name, val.dtype, dtype),
            TypeCastPerformanceWarning,
        )

    if not val.ndim == ndim:
        if add_newaxis_on_deficient_ndim and val.ndim + 1 == ndim:
            val = val[np.newaxis, ...]
        else:
            raise ValueError(
                ("%s must be ndim %s. " "You supplied %s" % (name, ndim, val.ndim))
            )

    if isinstance(val, np.ndarray) or (
        cast_da_to_np and isinstance(val, da.core.Array)
    ):
        val = np.ascontiguousarray(val, dtype=dtype)

    if length is not None and len(val) != length:
        raise ValueError(
            ("%s must be length %s. " "You supplied %s" % (name, length, len(val)))
        )

    if shape is not None:
        # the shape specified given by the user can look like (None, None 3)
        # which indicates that ANY length is accepted in dimension 0 or
        # dimension 1
        sentenel = object()
        error = ValueError(
            (
                "%s must be shape %s. You supplied  "
                "%s" % (name, str(shape).replace("None", "Any"), val.shape)
            )
        )
        for a, b in zip_longest(val.shape, shape, fillvalue=sentenel):
            if a is sentenel or b is sentenel:
                # if the sentenel was reached, it means that the ndim didn't
                # match or something. this really shouldn't happen
                raise error
            if b is None:
                # the user's shape spec has a None in it, it matches anything
                continue
            if a != b:
                # check for equality
                raise error
    return val


def lengths_and_angles_to_box_vectors(a_length, b_length, c_length, alpha, beta, gamma):
    """Convert from the lengths/angles of the unit cell to the box
    vectors (Bravais vectors). The angles should be in degrees.

    Mimics mdtraj.core.unitcell.lengths_and_angles_to_box_vectors()

    Parameters
    ----------
    a_length : scalar or ndarray
        length of Bravais unit vector **a**
    b_length : scalar or ndarray
        length of Bravais unit vector **b**
    c_length : scalar or ndarray
        length of Bravais unit vector **c**
    alpha : scalar or ndarray
        angle between vectors **b** and **c**, in degrees.
    beta : scalar or ndarray
        angle between vectors **c** and **a**, in degrees.
    gamma : scalar or ndarray
        angle between vectors **a** and **b**, in degrees.

    Returns
    -------
    a : dask.array
        If the inputs are scalar, the vectors will one dimensional (length 3).
        If the inputs are one dimension, shape=(n_frames, ), then the output
        will be (n_frames, 3)
    b : dask.array
        If the inputs are scalar, the vectors will one dimensional (length 3).
        If the inputs are one dimension, shape=(n_frames, ), then the output
        will be (n_frames, 3)
    c : dask.array
        If the inputs are scalar, the vectors will one dimensional (length 3).
        If the inputs are one dimension, shape=(n_frames, ), then the output
        will be (n_frames, 3)

    This code is adapted from gyroid, which is licensed under the BSD
    http://pythonhosted.org/gyroid/_modules/gyroid/unitcell.html
    """
    # Fix for da that requires angles and lengths to be arrays
    lengths = [a_length, b_length, c_length]
    for i, e in enumerate(lengths):
        # Use python logic shortcutting to not compute dask Arrays
        if not isinstance(e, da.core.Array) and np.isscalar(e):
            lengths[i] = np.array([e])
    a_length, b_length, c_length = tuple(lengths)

    angles = [alpha, beta, gamma]
    for i, e in enumerate(angles):
        if not isinstance(e, da.core.Array) and np.isscalar(e):
            angles[i] = np.array([e])
    alpha, beta, gamma = tuple(angles)

    if da.all(alpha < 2 * np.pi) and (
        da.all(beta < 2 * np.pi) and da.all(gamma < 2 * np.pi)
    ):
        warnings.warn(
            "All your angles were less than 2*pi."
            " Did you accidentally give me radians?"
        )

    alpha = alpha * np.pi / 180
    beta = beta * np.pi / 180
    gamma = gamma * np.pi / 180

    a = da.stack([a_length, da.zeros_like(a_length), da.zeros_like(a_length)])
    b = da.stack(
        [b_length * da.cos(gamma), b_length * da.sin(gamma), da.zeros_like(b_length)]
    )
    cx = c_length * da.cos(beta)
    cy = c_length * (da.cos(alpha) - da.cos(beta) * da.cos(gamma)) / da.sin(gamma)
    cz = da.sqrt(c_length * c_length - cx * cx - cy * cy)
    c = da.stack([cx, cy, cz])
    if not a.shape == b.shape == c.shape:
        raise TypeError("Shape is messed up.")

    # Make sure that all vector components that are _almost_ 0 are set exactly
    # to 0
    tol = 1e-6
    a[da.logical_and(a > -tol, a < tol)] = 0.0
    b[da.logical_and(b > -tol, b < tol)] = 0.0
    c[da.logical_and(c > -tol, c < tol)] = 0.0

    return a.T, b.T, c.T


def box_vectors_to_lengths_and_angles(a, b, c):
    """Convert box vectors into the lengths and angles defining the box.

    Addapted from mdtraj.utils.unitcell.box_vectors_to_lengths_and_angles()
    Parameters
    ----------
    a : np.ndarray
        the vector defining the first edge of the periodic box (length 3), or
        an array of this vector in multiple frames, where a[i,:] gives the
        length 3 array of vector a in each frame of a simulation
    b : np.ndarray
        the vector defining the second edge of the periodic box (length 3), or
        an array of this vector in multiple frames, where b[i,:] gives the
        length 3 array of vector a in each frame of a simulation
    c : np.ndarray
        the vector defining the third edge of the periodic box (length 3), or
        an array of this vector in multiple frames, where c[i,:] gives the
        length 3 array of vector a in each frame of a simulation

    Returns
    -------
    a_length : dask.array
        length of Bravais unit vector **a**
    b_length : dask.array
        length of Bravais unit vector **b**
    c_length : dask.array
        length of Bravais unit vector **c**
    alpha : dask.array
        angle between vectors **b** and **c**, in degrees.
    beta : dask.array
        angle between vectors **c** and **a**, in degrees.
    gamma : dask.array
        angle between vectors **a** and **b**, in degrees.
    """
    if not a.shape == b.shape == c.shape:
        raise TypeError("Shape is messed up.")
    if not a.shape[-1] == 3:
        raise TypeError("The last dimension must be length 3")
    if not (a.ndim in [1, 2]):
        raise ValueError(
            "vectors must be 1d or 2d (for a vectorized "
            "operation on multiple frames)"
        )
    last_dim = a.ndim - 1

    a_length = (da.sum(a * a, axis=last_dim)) ** (1 / 2)
    b_length = (da.sum(b * b, axis=last_dim)) ** (1 / 2)
    c_length = (da.sum(c * c, axis=last_dim)) ** (1 / 2)

    # we allow 2d input, where the first dimension is the frame index
    # so we want to do the dot product only over the last dimension
    alpha = da.arccos(da.einsum("...i, ...i", b, c) / (b_length * c_length))
    beta = da.arccos(da.einsum("...i, ...i", c, a) / (c_length * a_length))
    gamma = da.arccos(da.einsum("...i, ...i", a, b) / (a_length * b_length))

    # convert to degrees
    alpha = alpha * 180.0 / np.pi
    beta = beta * 180.0 / np.pi
    gamma = gamma * 180.0 / np.pi

    return a_length, b_length, c_length, alpha, beta, gamma


def wrap_da(f, chunk_size, **kwargs):
    """Convenience function to wrap dask.array.from_delayed()"""
    return da.from_delayed(f(**kwargs), dtype=np.float32, shape=chunk_size)
