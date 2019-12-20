import dask
import dask.array as da
from dask_traj.utils import ensure_type, wrap_da
from mdtraj.geometry import _geometry
from mdtraj.geometry.distance import (_distance, _distance_mic,
                                      _displacement, _displacement_mic)
import numpy as np


@dask.delayed
def _compute_distances_chunk(xyz, pairs, box=None, periodic=True, opt=True,
                             orthogonal=False):
    """Compute distances for a single chunk

    Parameters
    ----------
    xyz : ndarray of shape (any, any, 3)
        The xyz coordinates of the chunk
    pairs : array of shape (any, 2)
        The indices for which to compute an angle
    box : ndarray of shape (any, 3, 3)
        The box vectors of the chunk
    periodic : bool
        Wether to use the periodc boundary during the calculation.
    opt : bool, default=True
        Use an optimized native library to calculate distances. MDTraj's
        optimized SSE angle calculation implementation is 10-20x faster than
        the (itself optimized) numpy implementation.
    orthogonal : bool or da.bool
        Wether all angles are close to 90 degrees
    """

    xyz = ensure_type(xyz, dtype=np.float32, ndim=3, name='xyz',
                      shape=(None, None, 3), warn_on_cast=False,
                      cast_da_to_np=True)
    # Cast orthogonal to bool (incase we get a delayed bool)
    orthogonal = bool(orthogonal)
    if periodic and box is not None:
        if opt:
            out = np.empty((xyz.shape[0], pairs.shape[0]), dtype=np.float32)
            _geometry._dist_mic(xyz, pairs, box.transpose(0, 2, 1).copy(), out,
                                orthogonal)
            return out
        else:
            return _distance_mic(xyz, pairs, box.transpose(0, 2, 1),
                                 orthogonal)

    # Either there are no unitcell vectors or they dont want to use them
    if opt:
        out = np.empty((xyz.shape[0], pairs.shape[0]), dtype=np.float32)
        _geometry._dist(xyz, pairs, out)
        return out
    else:
        return _distance(xyz, pairs)


def compute_distances(traj, atom_pairs, periodic=True, **kwargs):
    """Daskified version of mdtraj.compute_distances().

    This mimics py:method:`mdtraj.compute_distances()` but returns the answer
    as a py:class:`dask.array` object

    Parameters
    ----------
    traj : :py:class:`dask_traj.Trajectory`
        The trajectory to compute the angles for.
    atom_pairs : array of shape(any, 2)
        The indices for which to compute an distances.
    periodic : bool
        Wether to use the periodc boundary during the calculation.
    opt : bool, default=True
        Use an optimized native library to calculate distances. MDTraj's
        optimized SSE angle calculation implementation is 10-20x faster than
        the (itself optimized) numpy implementation.

    Returns
    -------
    distances : dask.array, shape(n_frames, atom_pairs)
        Dask array with the delayed calculated distance for each item in
        atom_pairs for each frame.
    """
    xyz = traj.xyz
    length = len(xyz)
    atoms = len(atom_pairs)
    pairs = ensure_type(atom_pairs, dtype=np.int32, ndim=2, name='atom_pairs',
                        shape=(None, 2), warn_on_cast=False)
    if not np.all(np.logical_and(pairs < traj.n_atoms, pairs >= 0)):
        raise ValueError('atom_pairs must be between 0 and %d' % traj.n_atoms)
    if len(pairs) == 0:  # If pairs is an empty slice of an array
        return da.zeros((length, 0), dtype=np.float32)
    if periodic and traj._have_unitcell:
        box = ensure_type(traj.unitcell_vectors, dtype=np.float32, ndim=3,
                          name='unitcell_vectors', shape=(len(xyz), 3, 3),
                          warn_on_cast=False)
    else:
        box = None
        orthogonal = False
    lazy_results = []
    current_frame = 0
    for frames in xyz.chunks[0]:
        chunk_size = (frames, atoms)
        next_frame = current_frame+frames
        if box is not None:
            current_box = box[current_frame:next_frame]
            orthogonal = da.allclose(
                traj.unitcell_angles[current_frame:next_frame], 90)
        else:
            current_box = None
        lazy_results.append(wrap_da(_compute_distances_chunk, chunk_size,
                                    xyz=xyz[current_frame:next_frame],
                                    pairs=pairs,
                                    box=current_box,
                                    orthogonal=orthogonal,
                                    **kwargs))
        current_frame = next_frame
    max_result = da.concatenate(lazy_results)
    result = max_result[:length]
    return result


@dask.delayed
def _compute_displacements_chunk(xyz, pairs, box=None, periodic=True, opt=True,
                                 orthogonal=False):
    """Compute displacements for a single chunk

    Parameters
    ----------
    xyz : ndarray of shape (any, any, 3)
        The xyz coordinates of the chunk
    pairs : array of shape (any, 2)
        The indices for which to compute an angle
    box : ndarray of shape (any, 3, 3)
        The box vectors of the chunk
    periodic : bool
        Wether to use the periodc boundary during the calculation.
    opt : bool, default=True
        Use an optimized native library to calculate distances. MDTraj's
        optimized SSE angle calculation implementation is 10-20x faster than
        the (itself optimized) numpy implementation.
    orthogonal : bool or da.bool
        Wether all angles are close to 90 degrees
    """

    # Cast orthogonal to a bool, just incase we got a delayed object
    orthogonal = bool(orthogonal)
    xyz = ensure_type(xyz, dtype=np.float32, ndim=3, name='xyz',
                      shape=(None, None, 3), warn_on_cast=False,
                      cast_da_to_np=True)
    if periodic and box is not None:
        if opt:
            out = np.empty((xyz.shape[0], pairs.shape[0], 3), dtype=np.float32)
            _geometry._dist_mic_displacement(xyz, pairs,
                                             box.transpose(0, 2, 1).copy(),
                                             out,
                                             orthogonal)
            return out
        else:
            return _displacement_mic(xyz, pairs, box.transpose(0, 2, 1),
                                     orthogonal)

    # Either there are no unitcell vectors or they dont want to use them
    if opt:
        out = np.empty((xyz.shape[0], pairs.shape[0], 3), dtype=np.float32)
        _geometry._dist_displacement(xyz, pairs, out)
        return out
    else:
        return _displacement(xyz, pairs)


def compute_displacements(traj, atom_pairs, periodic=True, **kwargs):
    """Daskified version of mdtraj.geometry.compute_displacements

    This mimics py:method:`mdtraj.compute_displacements()` but returns the
    answer as a py:class:`dask.array` object

    Parameters
    ----------
    traj : :py:class:`dask_traj.Trajectory`
        The trajectory to compute the angles for.
    atom_pairs : array of shape(any, 2)
        The indices for which to compute an distances.
    periodic : bool
        Wether to use the periodc boundary during the calculation.
    opt : bool, default=True
        Use an optimized native library to calculate distances. MDTraj's
        optimized SSE angle calculation implementation is 10-20x faster than
        the (itself optimized) numpy implementation.

    Returns
    -------
    displacements : dask.array, shape(n_frames, atom_pairs, 3)
        Dask array with the delayed calculated displacements for each item in
        atom_pairs for each frame.
    """

    xyz = traj.xyz
    length = len(xyz)
    atoms = len(atom_pairs)
    pairs = ensure_type(atom_pairs, dtype=np.int32, ndim=2, name='atom_pairs',
                        shape=(None, 2), warn_on_cast=False)
    if not np.all(np.logical_and(pairs < traj.n_atoms, pairs >= 0)):
        raise ValueError('atom_pairs must be between 0 and %d' % traj.n_atoms)
    if len(pairs) == 0:  # If pairs is an empty slice of an array
        return da.zeros((length, 0, 3), dtype=np.float32)
    if periodic and traj._have_unitcell:
        box = ensure_type(traj.unitcell_vectors, dtype=np.float32, ndim=3,
                          name='unitcell_vectors', shape=(len(xyz), 3, 3),
                          warn_on_cast=False)
    else:
        box = None
        orthogonal = False
    lazy_results = []
    current_frame = 0
    for frames in xyz.chunks[0]:
        chunk_size = (frames, atoms, 3)
        next_frame = current_frame+frames
        if box is not None:
            current_box = box[current_frame:next_frame]
            orthogonal = da.allclose(
                traj.unitcell_angles[current_frame:next_frame], 90)
        else:
            current_box = None
        lazy_results.append(wrap_da(_compute_displacements_chunk, chunk_size,
                                    xyz=xyz[current_frame:next_frame],
                                    pairs=pairs,
                                    box=current_box,
                                    orthogonal=orthogonal,
                                    **kwargs))
        current_frame = next_frame
    max_result = da.concatenate(lazy_results)
    result = max_result[:length]
    return result


@dask.delayed
def _compute_center_of_mass_chunk(xyz, masses):
    """Compute center of mass coordinates for a single chunk

    Parameters
    ----------
    xyz : ndarray of shape (any, any, 3)
        The xyz coordinates of the chunk
    masses : ndarray of shape (n_atoms, )
        The masses of each atom
    """
    com = np.zeros((xyz.shape[0], 3))
    for i, x in enumerate(xyz):
        com[i, :] = x.astype('float64').T.dot(masses)
    return com


def compute_center_of_mass(traj):
    """Daskified version of mdtraj.geometry.compute_center_of_mass

    This mimics py:method:`mdtraj.compute_center_of_mass()` but returns the
    answer as a py:class:`dask.array` object

    Parameters
    ----------
    traj : :py:class:`dask_traj.Trajectory`
        The trajectory to compute the angles for.

    Returns
    -------
    com : dask.array, shape(n_frames, 3)
        Dask array with the delayed calculated Coordinates of center of mass
        for each frame.
    """

    xyz = traj.xyz
    length = len(xyz)
    masses = np.array([a.element.mass for a in traj.top.atoms])
    masses /= masses.sum()
    lazy_results = []
    current_frame = 0
    for frames in xyz.chunks[0]:
        chunk_size = (frames, 3)
        next_frame = current_frame+frames
        lazy_results.append(wrap_da(f=_compute_center_of_mass_chunk,
                                    chunk_size=chunk_size,
                                    xyz=xyz[current_frame:next_frame],
                                    masses=masses
                                    ))
        current_frame = next_frame
    max_result = da.concatenate(lazy_results)
    results = max_result[:length]
    return results


@dask.delayed
def _compute_center_of_geometry_chunk(xyz):
    """Compute center of geometry coordinates for a single chunk

    Parameters
    ----------
    xyz : ndarray of shape (any, any, 3)
        The xyz coordinates of the chunk
    """
    centers = np.zeros((xyz.shape[0], 3))
    for i, x in enumerate(xyz):
        centers[i, :] = x.astype('float64').T.mean(axis=1)
    return centers


def compute_center_of_geometry(traj):
    """Daskified version of mdtraj.geometry.compute_center_of_geometry

    This mimics py:method:`mdtraj.compute_center_of_geometry()` but returns the
    answer as a py:class:`dask.array` object

    Parameters
    ----------
    traj : :py:class:`dask_traj.Trajectory`
        The trajectory to compute the angles for.

    Returns
    -------
    com : dask.array, shape(n_frames, 3)
        Dask array with the delayed calculated Coordinates of center of
        geometry for each frame.
    """

    xyz = traj.xyz
    length = len(xyz)
    lazy_results = []
    current_frame = 0
    for frames in xyz.chunks[0]:
        chunk_size = (frames, 3)
        next_frame = current_frame+frames
        lazy_results.append(wrap_da(f=_compute_center_of_geometry_chunk,
                                    chunk_size=chunk_size,
                                    xyz=xyz[current_frame:next_frame],
                                    ))
        current_frame = next_frame
    max_result = da.concatenate(lazy_results)
    results = max_result[:length]
    return results


@dask.delayed
def _find_closest_contact(traj, group1, group2, frame=0, periodic=True):

    xyz = ensure_type(traj.xyz[frame], dtype=np.float32, ndim=2,
                      name='xyz', shape=(None, 3), warn_on_cast=False,
                      cast_da_to_np=True)
    atoms1 = ensure_type(group1, dtype=np.int32, ndim=1,
                         name='group1', warn_on_cast=False)
    atoms2 = ensure_type(group2, dtype=np.int32, ndim=1,
                         name='group2', warn_on_cast=False)
    if periodic and traj._have_unitcell:
        box = ensure_type(traj.unitcell_vectors, dtype=np.float32,
                          ndim=3, name='unitcell_vectors',
                          shape=(len(traj.xyz), 3, 3),
                          warn_on_cast=False)[frame]
    else:
        box = None
    ans = _geometry._find_closest_contact(xyz, atoms1, atoms2, box)
    return np.asarray(ans)


def find_closest_contact(traj, group1, group2, frame=0, periodic=True):
    """Daskified version fo mdtraj.find_closest_contact()

    Mimics mdtraj.find_closest_contact, but instead returns a delayed tuple

    Parameters
    ----------
    traj : Trajectory
        An dask-traj trajectory.
    group1 : ndarray, shape=(num_atoms), dtype=int
        The indices of atoms in the first group.
    group2 : ndarray, shape=(num_atoms), dtype=int
        The indices of atoms in the second group.
    frame : int, default=0
        The frame of the Trajectory to take positions from
    periodic : bool, default=True
        If `periodic` is True and the trajectory contains unitcell
        information, we will compute distances under the minimum image
        convention.
    Returns
    -------
    result : delayed tuple (int, int, float)
         A delayed tuple with the indices of the two atoms forming the closest
         contact, and the distance between them.
    """

    # This shouldn't be used, as it is the worst possible example of the
    # scaling (1 frame only).
    # Just here for completenes
    ans = wrap_da(f=_find_closest_contact,
                  chunk_size=(3,),
                  traj=traj,
                  group1=group1,
                  group2=group2,
                  frame=frame,
                  periodic=periodic)
    return dask.delayed(tuple)([int(ans[0]), int(ans[1]), float(ans[2])])
