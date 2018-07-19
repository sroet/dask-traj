import dask
import dask.array as da
import numpy as np
import mdtraj
from mdtraj.utils import in_units_of, box_vectors_to_lengths_and_angles
from mdtraj.core.trajectory import (open, _get_extension, _parse_topology,
                                    _TOPOLOGY_EXTS)
from .utils import ensure_type
# dictionary to tell what is actually returned by read per extension type
# can consist of [xyz, time, unitcell_lengths, unitcell_angles,
# unitcell_vectors]
file_returns = {'.arc': ['xyz', 'unitcell_lengths', 'unitcell_angles'],
                '.dcd': ['xyz', 'unitcell_lengths', 'unitcell_angles'],
                '.binpos': ['xyz'],
                '.xtc': ['xyz', 'time', 'step', 'unitcell_vectors'],
                '.trr': ['xyz', 'time', 'step', 'unitcell_vectors', '_'],
                '.hdf5': ['data'], #Need special case
                '.h5': ['data'], # Same as hdf5
                '.ncdf': ['xyz', 'time', 'unitcell_lengths',
                          'unitcell_angles'],
                '.netcdf': ['xyz', 'time', 'unitcell_lengths',
                          'unitcell_angles'],
                '.nc': ['xyz', 'time', 'unitcell_lengths',
                          'unitcell_angles'],
                '.pdb.gz': [], # Not implemented for now
                '.pdb': [], # Not implemented for now
                '.lh5': ['xyz'],
                '.crd': ['xyz', 'unitcell_lengths'
                         ], # Needs to assume angles to be 90
                '.mdcrd': ['xyz', 'unitcell_lengths'
                           ],  # Needs to assume angles to be 90
                '.inpcrd': ['xyz', 'time', 'unitcell_lengths',
                            'unitcell_angles'],
                '.restrt': ['xyz', 'time', 'unitcell_lengths',
                            'unitcell_angles'],
                '.rst7': ['xyz', 'time', 'unitcell_lengths',
                          'unitcell_angles'],
                '.ncrst': ['xyz', 'time', 'unitcell_lengths',
                          'unitcell_angles'],
                '.lammpstrj': ['xyz', 'unitcell_lengths', 'unitcell_angles'],
                '.dtr': ['xyz', 'time', 'unitcell_lengths', 'unitcell_angles'],
                '.stk': ['xyz', 'time', 'unitcell_lengths', 'unitcell_angles'],
                '.gro': ['xyz', 'time', 'unitcell_vectors'],
                '.xyz.gz': ['xyz'],
                '.xyz': ['xyz'],
                '.tng': ['xyz','time', 'unitcell_vectors'],
                '.xml': [], #not implemented for now
                '.mol2': [], #not implemented for now
                '.hoomdxml': [] #not implemented for now
                }

# TODO make class of trajectory chunk

def load(filename, chunks=10, **kwargs):
    """ A loader that will mimic mdtraj.Trajectory.load, but construct a
    dasktraj.Trajectory with a dask.array as xyz
    """

    top = kwargs.pop('top', None)
    extension = _get_extension(filename)
    if extension not in _TOPOLOGY_EXTS:
        topology = _parse_topology(top)


    length = len(open(filename))
    n_chunks = int(length/chunks)
    frames_left = length % chunks
    if frames_left != 0:
        n_chunks += 1
    # TODO this needs to be closed at some point
    data = load_chunks(filename, extension, chunks, range(n_chunks),
                           **kwargs)

    #TODO: use this to construct unitcells
    # Pop out irelevant info
    uv = data.pop('unitcell_vectors')
    traj = Trajectory(topology=topology, delayed_objects=data, **data)
    if uv is not None:
        traj.unitcell_vectors = uv
    return traj

def load_chunks(filename, extension, chunk_size, chunks, **kwargs):
    read_returns = file_returns[extension]
    # TODO: Add other kwargs like stride, atom_indices, time
    xyzs = []
    with open(filename) as f:
        length = len(f)
        distance_unit = f.distance_unit
    frames_left = length
    results = []
    for chunk in chunks:
        frames = min(frames_left, chunk_size)
        results.append(dask.delayed(read_chunk, pure=True)(filename,
                                                           extension,
                                                           frames,
                                                           chunk))
        frames_left -= frames

    result_dict = build_result_dict(results, extension, length, chunk_size,
                                    distance_unit)
    return result_dict

def build_result_dict(results, extension, length, chunk_size, distance_unit):
    read_returns = file_returns[extension]
    sample = results[0].compute()
    result_dict = {key: [result[i] for result in results]
                   for i,key in enumerate(read_returns)}
    xyz = get_xyz(result_dict, length, distance_unit )
    time = get_time(result_dict, length, chunk_size)
    unit_cell = get_unitcell(result_dict, length)

    #Only keep xyz lazy
    not_lazy_dict = {'time': time,
                     'unitcell_lengths': unit_cell[0],
                     'unitcell_angles': unit_cell[1],
                     'unitcell_vectors': unit_cell[2]}
    return_dict = dask.compute(not_lazy_dict)[0]
    return_dict['xyz'] = xyz
    return return_dict

def make_da(delayed_list, length):
    sample = delayed_list[0].compute()
    arrays = [da.from_delayed(item, dtype=sample.dtype, shape=sample.shape)
              for item in delayed_list]
    result = da.concatenate(arrays, axis=0)[:length]
    return result

def get_xyz(result_dict, length, distance_unit):
    xyz_list = result_dict.pop('xyz', None)
    if xyz_list is None:
        return None
    else:
        for xyz in xyz_list:
            in_units_of(xyz, distance_unit, Trajectory._distance_unit,
                        inplace=True)
    result = make_da(xyz_list, length)
    return result

def get_time(result_dict, length, chunk_size):
    time_list = result_dict.pop('time', None)
    if time_list is None:
        # TODO incorporate stride
        result = da.arange(length, chunks=(chunk_size,))
    else:
        result = make_da(time_list, length)
    return result

def get_unitcell(result_dict, length):
    # TODO add ensure type on these lengths
    unitcell_lengths = result_dict.pop('unitcell_lengths', None)
    unitcell_angles = result_dict.pop('unitcell_angles', None)
    unitcell_vectors = result_dict.pop('unitcell_vectors', None)
    if (unitcell_lengths is None
        and unitcell_angles is None
        and unitcell_vectors is None):
        ul = None
        ua = None
        uv = None
    elif unitcell_vectors is not None:
        ul = None
        ua = None
        uv = make_da(unitcell_vectors, length)
        return None, None, uv
    elif unitcell_lengths is not None and unitcell_angles is None:
        ul = make_da(unitcell_vectors, length)
        ua = da.ones_like(ul)
        uv = None
    else:
        ul = make_da(unitcell_vectors, length)
        ua = make_da(unitcell_angles, length)
        uv = None
    return ul, ua, uv


def read_chunk(filename, extension, chunk_size, chunk):
    with open(filename) as f:
        # Get current possition
        pos = f.tell()
        # position we want
        seek_pos = chunk_size*chunk
        rel_pos = seek_pos-pos

        f.seek(rel_pos, 1)
        result = f.read(chunk_size)

    if isinstance(result, tuple):
        return result
    else:
        return result,

class Trajectory(mdtraj.Trajectory):
    # TODO add other kwargs from MDtraj.trajectory
    def __init__(self, xyz,  topology, delayed_objects, time=None, **kwargs):
        dask.persist(**kwargs)
        super(Trajectory, self).__init__(xyz=xyz, topology=topology, **kwargs)



    @property
    def xyz(self):
        """Cartesian coordinates of each atom in each simulation frame
        Returns
        -------
        xyz : np.ndarray, shape=(n_frames, n_atoms, 3)
            A three dimensional numpy array, with the cartesian coordinates
            of each atoms in each frame.
        """
        return self._xyz


    @xyz.setter
    def xyz(self, value):
        "Set the cartesian coordinates of each atom in each simulation frame"
        if self.top is not None:
            # if we have a topology and its not None
            shape = (None, self.topology._numAtoms, 3)
        else:
            shape = (None, None, 3)

        # TODO: make ensure_type work on dask arrays
        value = ensure_type(value, np.float32, 3, 'xyz', shape=shape,
                            warn_on_cast=True,
                            add_newaxis_on_deficient_ndim=True)
        self._xyz = value
        self._rmsd_traces = None
