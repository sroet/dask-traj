import os
from .helpers import function_test
import mdtraj as md
import dask_traj as dmd
import itertools as itt
import pytest

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

    @pytest.mark.parametrize('opt', [True, False])
    def test_compute_distances(self, opt):
        dask_trj, md_trj = self.slice_trajs(opt)

        c = list(itt.combinations(self.indices[:10], 2))
        function_test(dask_trj=dask_trj,
                      md_trj=md_trj,
                      funcname='compute_distances',
                      atom_pairs=c,
                      opt=opt)

    @pytest.mark.parametrize('opt', [True, False])
    def test_compute_displacements(self, opt):
        dask_trj, md_trj = self.slice_trajs(opt)

        c = list(itt.combinations(self.indices[:10], 2))
        function_test(dask_trj=self.dask_traj,
                      md_trj=self.md_traj,
                      funcname='compute_displacements',
                      atom_pairs=c,
                      opt=opt)

    def test_compute_center_of_mass(self):
        function_test(dask_trj=self.dask_traj,
                      md_trj=self.md_traj,
                      funcname='compute_center_of_mass')

    def test_compute_center_of_geometry(self):
        function_test(dask_trj=self.dask_traj,
                      md_trj=self.md_traj,
                      funcname='compute_center_of_geometry')

    def test_find_closest_contact(self):
        a = self.indices[:10]
        b = self.indices[10:20]
        function_test(dask_trj=self.dask_traj,
                      md_trj=self.md_traj,
                      funcname='find_closest_contact',
                      group1=a,
                      group2=b)

    @pytest.mark.parametrize('opt', [True, False])
    def test_compute_angles(self, opt):
        dask_trj, md_trj = self.slice_trajs(opt)

        c = list(itt.combinations(self.indices[:10], 3))
        function_test(dask_trj=dask_trj,
                      md_trj=md_trj,
                      funcname='compute_angles',
                      angle_indices=c,
                      opt=opt)
