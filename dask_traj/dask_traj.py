import dask
import dask.array as da
import numpy as np
import mdtraj
import os
from mdtraj.utils import in_units_of
from mdtraj.core.trajectory import (open, _get_extension, _parse_topology,
                                    _TOPOLOGY_EXTS, _hash_numpy_array)
from .utils import (ensure_type, box_vectors_to_lengths_and_angles,
                    lengths_and_angles_to_box_vectors)

from copy import deepcopy
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

    filename = os.path.abspath(filename)
    length = len(open(filename))
    n_chunks = int(length/chunks)
    frames_left = length % chunks
    if frames_left != 0:
        n_chunks += 1
    # TODO this needs to be closed at some point
    data = load_chunks(filename, extension, chunks, range(n_chunks),
                           **kwargs)

    #TODO: use this to construct unitcells
    # Pop out irrelevant info
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
        start = length-frames_left
        results.append(dask.delayed(read_chunk, pure=True)(filename,
                                                           extension,
                                                           frames,
                                                           start))
        frames_left -= frames

    result_dict = build_result_dict(results, extension, length, chunk_size,
                                    distance_unit)
    return result_dict

def build_result_dict(results, extension, length, chunk_size, distance_unit):
    read_returns = file_returns[extension]
    # Persis the sample for quick building
    sample = results[0].persist()
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
    return_dict = not_lazy_dict # dask.compute(not_lazy_dict)[0]
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


def read_chunk(filename, extension, chunk_size, start):
    with open(filename) as f:
        # Get current possition
        pos = f.tell()
        # position we want
        seek_pos = start
        rel_pos = seek_pos-pos

        f.seek(rel_pos, 1)
        result = f.read(chunk_size)

    if isinstance(result, tuple):
        return result
    else:
        return result,

class Trajectory(mdtraj.Trajectory):
    # TODO add other kwargs from MDtraj.trajectory
    def __init__(self, xyz,  topology, time=None, delayed_objects=None,
                 **kwargs):
        dask.persist(**kwargs)
        self._unitcell_vectors = None
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

    @property
    def _have_unitcell(self):
        return ((self._unitcell_lengths is not None and
                 self._unitcell_angles is not None) or
                self._unitcell_vectors is not None)

    @property
    def unitcell_angles(self):
        """Angles that define the shape of the unit cell in each frame.
        Returns
        -------
        lengths : np.ndarray, shape=(n_frames, 3)
            The angles between the three unitcell vectors in each frame,
            ``alpha``, ``beta``, and ``gamma``. ``alpha' gives the angle
            between vectors ``b`` and ``c``, ``beta`` gives the angle between
            vectors ``c`` and ``a``, and ``gamma`` gives the angle between
            vectors ``a`` and ``b``. The angles are in degrees.
        """
        if self._unitcell_angles is None:
            self._calc_length_and_angles(self._unitcell_vectors)
        return self._unitcell_angles

    @unitcell_angles.setter
    def unitcell_angles(self, value):
        """Set the lengths that define the shape of the unit cell in each frame
        Parameters
        ----------
        value : np.ndarray, shape=(n_frames, 3)
            The angles ``alpha``, ``beta`` and ``gamma`` that define the
            shape of the unit cell in each frame. The angles should be in
            degrees.
        """
        self._unitcell_angles = ensure_type(value, np.float32, 2,
            'unitcell_angles', can_be_none=True, shape=(len(self), 3),
                warn_on_cast=False, add_newaxis_on_deficient_ndim=True)

    @property
    def unitcell_lengths(self):
        """Lengths that define the shape of the unit cell in each frame.
        Returns
        -------
        lengths : {np.ndarray, shape=(n_frames, 3), None}
            Lengths of the unit cell in each frame, in nanometers, or None
            if the Trajectory contains no unitcell information.
        """
        if self._unitcell_lengths is None:
            self._calc_length_and_angles(self._unitcell_vectors)
        return self._unitcell_lengths

    @unitcell_lengths.setter
    def unitcell_lengths(self, value):
        """Set the lengths that define the shape of the unit cell in each frame
        Parameters
        ----------
        value : np.ndarray, shape=(n_frames, 3)
            The distances ``a``, ``b``, and ``c`` that define the shape of the
            unit cell in each frame, or None
        """
        self._unitcell_lengths = ensure_type(value, np.float32, 2,
            'unitcell_lengths', can_be_none=True, shape=(len(self), 3),
            warn_on_cast=False, add_newaxis_on_deficient_ndim=True)

    #TODO:Add unitcell_vectors
    @property
    def unitcell_vectors(self):
        if self._unitcell_vectors is None:
            return self._calc_unitcell_vectors()
        else:
            return self._unitcell_vectors

    def _calc_unitcell_vectors(self):
        """The vectors that define the shape of the unit cell in each frame
        Returns
        -------
        vectors : np.ndarray, shape(n_frames, 3, 3)
            Vectors defining the shape of the unit cell in each frame.
            The semantics of this array are that the shape of the unit cell
            in frame ``i`` are given by the three vectors, ``value[i, 0, :]``,
            ``value[i, 1, :]``, and ``value[i, 2, :]``.
        """
        if self.unitcell_lengths is None or self.unitcell_angles is None:
            return None


        v1, v2, v3 = lengths_and_angles_to_box_vectors(
            self._unitcell_lengths[:, 0],  # a
            self._unitcell_lengths[:, 1],  # b
            self._unitcell_lengths[:, 2],  # c
            self._unitcell_angles[:, 0],   # alpha
            self._unitcell_angles[:, 1],   # beta
            self._unitcell_angles[:, 2],   # gamma
        )
        return da.swapaxes(da.dstack((v1, v2, v3)), 1, 2)

    @unitcell_vectors.setter
    def unitcell_vectors(self, vectors):
        self._unitcell_vectors = vectors

    def _calc_length_and_angles(self, vectors):
        """Set the three vectors that define the shape of the unit cell
        Parameters
        ----------
        vectors : tuple of three arrays, each of shape=(n_frames, 3)
            The semantics of this array are that the shape of the unit cell
            in frame ``i`` are given by the three vectors, ``value[i, 0, :]``,
            ``value[i, 1, :]``, and ``value[i, 2, :]``.
        """
        if vectors is None:# or da.all(abs(vectors) < 1e-15):
            self._unitcell_lengths = None
            self._unitcell_angles = None
            return

        if not len(vectors) == len(self):
            raise TypeError('unitcell_vectors must be the same length as '
                            'the trajectory. you provided %s' % str(vectors))

        v1 = vectors[:, 0, :]
        v2 = vectors[:, 1, :]
        v3 = vectors[:, 2, :]
        a, b, c, alpha, beta, gamma = box_vectors_to_lengths_and_angles(v1, v2, v3)

        self._unitcell_lengths = da.vstack((a, b, c)).T
        self._unitcell_angles = da.vstack((alpha, beta, gamma)).T



    def join(self, other, check_topology=True,
             discard_overlapping_frames=False):
        """ This is a daskified version of md.Trajectory.join """

        if isinstance(other, Trajectory):
            other = [other]
        if isinstance(other, list):
            if not all(isinstance(o, Trajectory) for o in other):
                raise TypeError('You can only join Trajectory instances')
            if not all(self.n_atoms == o.n_atoms for o in other):
                raise ValueError('Number of atoms in self (%d) is not equal '
                                 'to number of atoms in other' %
                                 (self.n_atoms))
            if check_topology and not all(self.topology == o.topology
                                          for o in other):
                raise ValueError('The topologies of the Trajectories are not '
                                 'the same')
            if not all(self._have_unitcell == o._have_unitcell for o in other):
                raise ValueError('Mixing trajectories with and without '
                                 'unitcell')
        else:
            raise TypeError(
                '`other` must be a list of Trajectory. You supplied %d' %
                type(other))

        trajectories = [self] + other
        if discard_overlapping_frames:
            for i in range(len(trajectories)-1):
                # last frame of trajectory i
                x0 = trajectories[i].xyz[-1]
                # first frame of trajectory i+1
                x1 = trajectories[i + 1].xyz[0]

                # check that all atoms are within 2e-3 nm
                # (this is kind of arbitrary)
                if np.all(np.abs(x1 - x0) < 2e-3):
                    trajectories[i] = trajectories[i][:-1]

        # Only difference between original code and current code
        xyz = da.concatenate([t.xyz for t in trajectories])

        time = np.concatenate([t.time for t in trajectories])
        angles = lengths = None
        if self._have_unitcell:
            angles = np.concatenate([t.unitcell_angles for t in trajectories])
            lengths = np.concatenate([t.unitcell_lengths for
                                      t in trajectories])

        # use this syntax so that if you subclass Trajectory,
        # the subclass's join() will return an instance of the subclass
        return self.__class__(xyz, deepcopy(self._topology), time=time,
                              unitcell_lengths=lengths,
                              unitcell_angles=angles)

    def __hash__(self):
        ''' updated hash to use the name of the dask array'''
        hash_value = hash(self.top)
        # combine with hashes of arrays
        hash_value ^= self._xyz.name
        hash_value ^= _hash_numpy_array(self.time)
        hash_value ^= _hash_numpy_array(self._unitcell_lengths)
        hash_value ^= _hash_numpy_array(self._unitcell_angles)
        return hash_value
