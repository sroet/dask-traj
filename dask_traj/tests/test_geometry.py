import os
from .helpers import function_test
import mdtraj as md
import dask_traj as dmd
import itertools as itt

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

    def test_compute_distances(self):
        c = list(itt.combinations(self.indices[:10], 2))
        function_test(dask_trj=self.dask_traj,
                      md_trj=self.md_traj,
                      funcname='compute_distances',
                      atom_pairs=c)
