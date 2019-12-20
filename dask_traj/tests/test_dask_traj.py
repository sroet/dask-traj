import pytest
import os
from dask_traj.core.dask_traj import (build_result_dict,
                                      get_xyz,
                                      get_time,
                                      get_unitcell,
                                      read_chunk,
                                      load)
import dask
import numpy as np
import mdtraj as md

rel_dir = os.path.dirname(__file__)
xtc = os.path.join(rel_dir, 'test.xtc')
pdb = os.path.join(rel_dir, 'test.pdb')
txt = os.path.join(rel_dir, 'test.xml')
xyz = os.path.join(rel_dir, 'test.xyz')


def f(x):
    return np.asarray(x)

class TestDaskTraj(object):
    def setup(self):
        pass

    def teardown(self):
        pass

    def test_not_implemented_filetype(self):
        with pytest.raises(NotImplementedError):
            build_result_dict(txt, extension='.xml',
                              length=1, chunk_size=1,
                              distance_unit='nm')

    def test_no_xyz_in_result_dict(self):
        assert get_xyz(result_dict={}, length=1, distance_unit='nm') is None

    def test_no_time_in_result_dict(self):
        time = get_time(result_dict={},
                        length=10,
                        chunk_size=2)
        assert list(time) == [0,1,2,3,4,5,6,7,8,9]

    def test_no_unitcell_in_result_dict(self):
        assert get_unitcell(result_dict={}, length=1) == (None, None, None)

    def test_no_uv_and_ua_in_result_dict(self):
        rd = {'unitcell_lengths':[dask.delayed(f)([1,2,3])]}
        ul, ua, uv = get_unitcell(result_dict=rd, length=3)
        assert uv is None
        assert (ul.compute() == np.asarray([1,2,3])).all()
        assert (ua.compute() == np.asarray([1,1,1])).all()

    def test_no_ul_in_result_dict(self):
        rd = {'unitcell_angles':[dask.delayed(f)([1,2,3])],
              'unitcell_vectors':[dask.delayed(f)([3,4,5])]}
        ul, ua, uv = get_unitcell(result_dict=rd, length=3)
        assert (uv.compute() == np.asarray([3,4,5])).all()
        assert (ul.compute() == np.asarray([3,4,5])).all()
        assert (ua.compute() == np.asarray([1,2,3])).all()

    def test_chunk_with_one_return(self):
        assert type(read_chunk(xyz, 1, 0)) is tuple

    def test_trajectory_to_mdtraj(self):
        dask_traj = load(xtc, top=pdb)
        md_traj = md.load(xtc, top=pdb)
        dask_traj.to_mdtraj()
        assert dask_traj == md_traj

    def test_trajectory_to_mdtraj_with_unitcell_lengths(self):
        dask_traj = load(xtc, top=pdb)
        #Trigger the calculation of unitcell lengths
        _ = dask_traj.unitcell_lengths
        md_traj = md.load(xtc, top=pdb)
        dask_traj.to_mdtraj()
        assert dask_traj == md_traj

    def test_set_xyz_without_top(self):
        dask_traj = load(xtc, top=pdb)
        dask_traj.top = None
        assert dask_traj.top is None
        dask_traj.xyz = np.array([[[1.,2.,3.]]], dtype=np.float32)
        assert (dask_traj._xyz == np.array([[[1.,2.,3.]]])).all()

    def test_calc_uv_without_ul_or_ua(self):
        dask_traj = load(xtc, top=pdb)
        dask_traj.unitcell_vectors = None
        assert dask_traj._calc_unitcell_vectors() is None

    def test_wrong_uv_length(self):
        dask_traj = load(xtc, top=pdb)
        with pytest.raises(TypeError) as err:
            dask_traj._calc_length_and_angles([1,2,3])

    def test_join(self):
         dt = load(xtc, top=pdb)
         dt2 = load(xtc, top=pdb)
         dt3 = dt.join(dt2)
         assert len(dt3) == len(dt)+len(dt2)

    def test_join_failures(self):
         dt = load(xtc, top=pdb)
         dt2 = 'not a trajectory'

         # Not a trajectory
         with pytest.raises(TypeError) as err:
             _ = dt.join(dt2)
         assert 'must be a list of Trajectory' in str(err.value)

         with pytest.raises(TypeError) as err:
             _ = dt.join([dt2])
         assert 'only join Trajectory instances' in str(err.value)

         dt2 = load(xtc, top=pdb)
         dt2.top = None # Remove topology
         with pytest.raises(ValueError) as err:
             _ = dt.join([dt2])
         assert 'The topologies of the Trajectories are not' in str(err.value)

         dt2 = load(xtc, top=pdb)
         _ = dt2.unitcell_lengths
         dt2.atom_slice([1,2,3], inplace=True)
         with pytest.raises(ValueError) as err:
             _ = dt.join([dt2])
         assert 'Number of atoms' in str(err.value)


         dt2 = load(xtc, top=pdb)
         dt2.unitcell_vectors = None # Remove unitcell
         with pytest.raises(ValueError) as err:
             _ = dt.join([dt2])
         assert 'unitcell' in str(err.value)

    def test_join_discard_overlap(self):
         dt = load(xtc, top=pdb)
         dt2 = load(xtc, top=pdb)
         dt3 = dt.join(dt2[::-1], discard_overlapping_frames=True)
         assert len(dt3) == len(dt)+len(dt2)-1

    def test_hash(self):
         dt = load(xtc, top=pdb)
         dt2 = load(xtc, top=pdb)
         assert dt == dt2 # This will compare hashes

