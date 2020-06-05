import os
from .helpers import function_test
import mdtraj as md
import dask_traj as dmd
import itertools as itt
import pytest
import numpy as np

rel_dir = os.path.dirname(__file__)
xtc = os.path.join(rel_dir, 'test.xtc')
pdb = os.path.join(rel_dir, 'test.pdb')


class TestGeometry(object):
    def setup(self):
        self.md_traj = md.load(xtc, top=pdb)
        self.dask_traj = dmd.load(xtc, top=pdb)
        self.indices = range(self.md_traj.n_atoms)

    def teardown(self):
        pass

    def slice_trajs(self, opt):
        if opt:
            return self.dask_traj, self.md_traj
        else:
            return self.dask_traj[:5], self.md_traj[:5]

    @pytest.mark.parametrize('periodic', [True, False])
    @pytest.mark.parametrize('opt', [True, False])
    def test_compute_distances(self, opt, periodic):
        dask_trj, md_trj = self.slice_trajs(opt)

        c = list(itt.combinations(self.indices[:10], 2))
        function_test(dask_trj=dask_trj,
                      md_trj=md_trj,
                      funcname='geometry.compute_distances',
                      atom_pairs=c,
                      opt=opt,
                      periodic=periodic)

    def test_compute_distances_wrong_pairs(self):
        dask_trj = self.dask_traj
        c = [[12345, -1]]
        with pytest.raises(ValueError) as err:
            dmd.compute_distances(traj=dask_trj,
                                  atom_pairs=c)
        assert 'atom_pairs must be between 0' in str(err.value)

    def test_compute_distances_empty_pairs(self):
        c = np.array([[12345, -1]])
        c = c[0:0]  # Make an empty slice
        function_test(dask_trj=self.dask_traj,
                      md_trj=self.md_traj,
                      funcname='compute_distances',
                      atom_pairs=c)

    @pytest.mark.parametrize('periodic', [True, False])
    @pytest.mark.parametrize('opt', [True, False])
    def test_compute_displacements(self, opt, periodic):
        c = list(itt.combinations(self.indices[:10], 2))
        function_test(dask_trj=self.dask_traj,
                      md_trj=self.md_traj,
                      funcname='compute_displacements',
                      atom_pairs=c,
                      opt=opt,
                      periodic=periodic)

    def test_compute_displacements_wrong_pairs(self):
        dask_trj = self.dask_traj
        c = [[12345, -1]]
        with pytest.raises(ValueError) as err:
            dmd.compute_displacements(traj=dask_trj,
                                      atom_pairs=c)
        assert 'atom_pairs must be between 0' in str(err.value)

    def test_compute_displacements_empty_pairs(self):
        c = np.array([[1234, -9999]])
        c = c[0:0]
        function_test(dask_trj=self.dask_traj,
                      md_trj=self.md_traj,
                      funcname='compute_displacements',
                      atom_pairs=c)

    def test_compute_center_of_mass(self):
        function_test(dask_trj=self.dask_traj,
                      md_trj=self.md_traj,
                      funcname='compute_center_of_mass')

    def test_compute_center_of_geometry(self):
        function_test(dask_trj=self.dask_traj,
                      md_trj=self.md_traj,
                      funcname='compute_center_of_geometry')

    @pytest.mark.parametrize('periodic', [True, False])
    def test_find_closest_contact(self, periodic):
        a = self.indices[:10]
        b = self.indices[10:20]
        function_test(dask_trj=self.dask_traj,
                      md_trj=self.md_traj,
                      funcname='find_closest_contact',
                      group1=a,
                      group2=b,
                      periodic=periodic)

    def test_find_closest_contacts_empty_slice(self):
        a = self.indices[10:10]
        b = self.indices[10:20]
        function_test(dask_trj=self.dask_traj,
                      md_trj=self.md_traj,
                      funcname='find_closest_contact',
                      group1=a,
                      group2=b)

    @pytest.mark.parametrize('periodic', [True, False])
    @pytest.mark.parametrize('opt', [True, False])
    def test_compute_angles(self, opt, periodic):
        dask_trj, md_trj = self.slice_trajs(opt)

        c = list(itt.combinations(self.indices[:10], 3))
        function_test(dask_trj=dask_trj,
                      md_trj=md_trj,
                      funcname='compute_angles',
                      angle_indices=c,
                      opt=opt,
                      periodic=periodic)

    def test_compute_angles_wrong_indices(self):
        c = [[-1, 3000, 8]]
        with pytest.raises(ValueError) as err:
            dmd.compute_angles(traj=self.dask_traj,
                               angle_indices=c)
        assert 'angle_indices' in str(err.value)

    def test_compute_angles_empty_pairs(self):
        c = np.array([[1234, -9999, 8]])
        c = c[0:0]
        function_test(dask_trj=self.dask_traj,
                      md_trj=self.md_traj,
                      funcname='compute_angles',
                      angle_indices=c)
