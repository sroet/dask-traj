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
    """Compute distances for a single chunk"""
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

    # either there are no unitcell vectors or they dont want to use them
    if opt:
        out = np.empty((xyz.shape[0], pairs.shape[0]), dtype=np.float32)
        _geometry._dist(xyz, pairs, out)
        return out
    else:
        return _distance(xyz, pairs)


def compute_distances(traj, atom_pairs, periodic=True, **kwargs):
    xyz = traj.xyz
    length = len(xyz)
    atoms = len(atom_pairs)
    pairs = ensure_type(atom_pairs, dtype=np.int32, ndim=2, name='atom_pairs',
                        shape=(None, 2), warn_on_cast=False)
    if not np.all(np.logical_and(pairs < traj.n_atoms, pairs >= 0)):
        raise ValueError('atom_pairs must be between 0 and %d' % traj.n_atoms)
    if len(pairs) == 0:
        return np.zeros((length, 0), dtype=np.float32)
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
            orthogonal = np.allclose(
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

    # either there are no unitcell vectors or they dont want to use them
    if opt:
        out = np.empty((xyz.shape[0], pairs.shape[0], 3), dtype=np.float32)
        _geometry._dist_displacement(xyz, pairs, out)
        return out
    else:
        return _displacement(xyz, pairs)


def compute_displacements(traj, atom_pairs, periodic=True, **kwargs):
    """ Daskified version of mdtraj.geometry.distance.compute_displacements"""
    xyz = traj.xyz
    length = len(xyz)
    atoms = len(atom_pairs)
    pairs = ensure_type(atom_pairs, dtype=np.int32, ndim=2, name='atom_pairs',
                        shape=(None, 2), warn_on_cast=False)
    if not np.all(np.logical_and(pairs < traj.n_atoms, pairs >= 0)):
        raise ValueError('atom_pairs must be between 0 and %d' % traj.n_atoms)
    if len(pairs) == 0:
        return np.zeros((length, 0), dtype=np.float32)
    if periodic and traj._have_unitcell:
        box = ensure_type(traj.unitcell_vectors, dtype=np.float32, ndim=3,
                          name='unitcell_vectors', shape=(len(xyz), 3, 3),
                          warn_on_cast=False)
        orthogonal = np.allclose(traj.unitcell_angles, 90)
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
    com = np.zeros((xyz.shape[0], 3))
    for i, x in enumerate(xyz):
        com[i, :] = x.astype('float64').T.dot(masses)
    return com


def compute_center_of_mass(traj):
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
    centers = np.zeros((xyz.shape[0], 3))
    for i, x in enumerate(xyz):
        centers[i, :] = x.astype('float64').T.mean(axis=1)
    return centers


def compute_center_of_geometry(traj):
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
    # This shouldn't be used, as it is the worst possible example of the
    # scaling (1 frame only).
    # Just here
    ans = wrap_da(f=_find_closest_contact,
                  chunk_size=(3,),
                  traj=traj,
                  group1=group1,
                  group2=group2,
                  frame=frame,
                  periodic=periodic)
    return dask.delayed(tuple)([int(ans[0]), int(ans[1]), float(ans[2])])
