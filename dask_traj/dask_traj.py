import dask
import dask.array as da
import mdtraj
from mdtraj.utils import in_units_of
from mdtraj.core.trajectory import (open, _get_extension, _parse_topology,
                                    _TOPOLOGY_EXTS)

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

    chunk_xyz = dask.delayed(load_chunk)
    lazy_xyzs = [chunk_xyz(filename, chunks, chunk, **kwargs)
           for chunk in range(n_chunks)]
    sample = lazy_xyzs[0].compute()
    # TODO: add correct last chunk shape
    xyz_arrays = [da.from_delayed(lazy_xyz,
                                 dtype = sample.dtype,
                                 shape = sample.shape)
                  for lazy_xyz in lazy_xyzs]

    xyz = da.concatenate(xyz_arrays, axis=0)[:length]
    #TODO: add time
    return Trajectory(xyz=xyz, topology=topology)

def load_chunk(filename, chunks, chunk, **kwargs):
    frames_per_chunk = chunks
    read_returns = file_returns[_get_extension(filename)]
    with open(filename) as f:
        # Seek the right frame
        f.seek(frames_per_chunk*chunk)
        # TODO: Add other kwargs like stride, atom_indices, time
        result = f.read(frames_per_chunk)
        if len(read_returns) == 1:
            #only xyz is returned
            result_dict = {read_returns[0]: result}
        else:
            result_dict = {read_returns[i]: e for i, e in enumerate(result)}

        # TODO actualy return the full dict and to building somewhere else
        xyz = result_dict.pop('xyz')

        in_units_of(xyz, f.distance_unit, Trajectory._distance_unit,
                    inplace=True)
    return xyz

class Trajectory(mdtraj.Trajectory):
    # TODO add other kwargs from MDtraj.trajectory
    def __init__(self, xyz,  topology, time=None, **kwargs):
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
        #value = ensure_type(value, np.float32, 3, 'xyz', shape=shape,
        #                    warn_on_cast=False, add_newaxis_on_deficient_ndim=True)
        self._xyz = value
        self._rmsd_traces = None
