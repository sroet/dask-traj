import dask
import dask.array as da
from .utils import ensure_type
from mdtraj.utils.six.moves import range
from mdtraj.geometry import _geometry
from mdtraj.geometry.distance import _distance, _distance_mic
import numpy as np

def wrap_da(f, chunk_size, **kwargs):
    return da.from_delayed(f(**kwargs), dtype=np.float32, shape=chunk_size)

@dask.delayed
def _compute_distances_chunk(xyz, pairs, box=None, periodic=True, opt=True,
                             orthogonal=False):
    xyz = ensure_type(xyz, dtype=np.float32, ndim=3, name='xyz',
                      shape=(None, None, 3), warn_on_cast=False,
                      cast_da_to_np = True)

    if periodic and box is not None:
        if opt:
            out = np.empty((xyz.shape[0], pairs.shape[0]), dtype=np.float32)
            _geometry._dist_mic(xyz, pairs, box.transpose(0, 2, 1).copy(), out,
                                orthogonal)
            return out
        else:
            return _distance_mic(xyz, pairs, box.transpose(0, 2, 1), orthogonal)

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
        box = ensure_type(traj.unitcell_vectors, dtype=np.float32, ndim=3, name='unitcell_vectors', shape=(len(xyz), 3, 3),
                          warn_on_cast=False)
        orthogonal = np.allclose(traj.unitcell_angles, 90)
    else:
        box = None
    lazy_results = []
    current_frame = 0
    for frames in xyz.chunks[0]:
        chunk_size = (frames, atoms)
        next_frame = current_frame+frames
        lazy_results.append(wrap_da(_compute_distances_chunk, chunk_size,
                                   xyz=xyz[current_frame:next_frame],
                                   pairs = pairs,
                                   box=box[current_frame:next_frame],
                                   orthogonal = orthogonal,
                                   **kwargs))
        current_frame = next_frame
    max_result = da.concatenate(lazy_results)
    result = max_result[:length]
    return result

