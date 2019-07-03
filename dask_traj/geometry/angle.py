import dask
import dask.array as da
from dask_traj.utils import ensure_type, wrap_da
from dask_traj.geometry import distance
from mdtraj.geometry import _geometry
import numpy as np


@dask.delayed
def _compute_angles_chunk(xyz, triplets, box=None, periodic=True,
                          opt=True, orthogonal=False):
    xyz = ensure_type(xyz, dtype=np.float32, ndim=3, name='xyz',
                      shape=(None, None, 3), warn_on_cast=False,
                      cast_da_to_np=True)

    out = np.empty((xyz.shape[0], triplets.shape[0]), dtype=np.float32)
    if opt:
        if periodic and box is not None:
            _geometry._angle_mic(xyz, triplets, box.transpose(0, 2, 1).copy(),
                                 out, orthogonal)
        else:
            _geometry._angle(xyz, triplets, out)
    else:
        out = _angle(xyz, triplets, periodic, out).compute()
    return out


def compute_angles(traj, angle_indices, periodic=True, **kwargs):
    xyz = traj.xyz
    length = len(xyz)
    atoms = len(angle_indices)
    triplets = ensure_type(angle_indices, dtype=np.int32, ndim=2,
                           name='angle_indices', shape=(None, 3),
                           warn_on_cast=False)
    if not np.all(np.logical_and(triplets < traj.n_atoms, triplets >= 0)):
        raise ValueError('angle_indices must be between 0 and %d' %
                         traj.n_atoms)

    if len(triplets) == 0:
        return np.zeros((len(xyz), 0), dtype=np.float32)

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
        next_frame = current_frame+frames
        if box is not None:
            current_box = box[current_frame:next_frame]
        else:
            current_box = None
        chunk_size = (frames, atoms)
        lazy_results.append(wrap_da(_compute_angles_chunk, chunk_size,
                                    xyz=xyz[current_frame:next_frame],
                                    triplets=triplets,
                                    box=current_box,
                                    orthogonal=orthogonal,
                                    **kwargs))
        current_frame = next_frame
    max_result = da.concatenate(lazy_results)
    result = max_result[:length]
    return result


def _angle(xyz, triplets, periodic, out):
    ix01 = triplets[:, [1, 0]]
    ix21 = triplets[:, [1, 2]]

    u_prime = distance._compute_displacements_chunk(xyz, ix01,
                                                    periodic=periodic,
                                                    opt=False)
    v_prime = distance._compute_displacements_chunk(xyz, ix21,
                                                    periodic=periodic,
                                                    opt=False)
    u_norm = ((u_prime**2).sum(-1))**(1/2)
    v_norm = ((v_prime**2).sum(-1))**(1/2)

    # adding a new axis makes sure that broasting rules kick in on the third
    # dimension
    u = u_prime / (u_norm[..., np.newaxis])
    v = v_prime / (v_norm[..., np.newaxis])

    return dask.delayed(np.arccos)((u * v).sum(-1), out=out)
