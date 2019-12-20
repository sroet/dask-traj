import pytest
from dask_traj.utils import (ensure_type,
                             lengths_and_angles_to_box_vectors,
                             box_vectors_to_lengths_and_angles)
import numpy as np
import itertools as itt
import mdtraj


class TestEnsureType(object):
    def setup(self):
        self.kwargs = {'dtype': np.int32,
                       'ndim': 1,
                       'name': 'test',
                       'add_newaxis_on_deficient_ndim': True}

    def teardown(self):
        pass

    def test_generators(self):
        test = itt.repeat(1, 3)  # This is a generator in python3
        b = ensure_type(test, **self.kwargs)
        assert all(b == np.array([1, 1, 1], dtype=np.int32))

    def test_scalar_conversion(self):
        test = 1
        self.kwargs['warn_on_cast'] = False
        b = ensure_type(test, **self.kwargs)
        assert all(b == np.array([1], dtype=np.int32))

    def test_type_error(self):
        test = 1.0
        self.kwargs.pop('add_newaxis_on_deficient_ndim')
        with pytest.raises(TypeError) as err:
            _ = ensure_type(test, **self.kwargs)
        assert 'must be numpy array' in str(err.value)

    def test_adding_ndim_to_arrays(self):
        test = np.array([1.0], dtype=np.int32)
        self.kwargs['ndim'] = 2
        b = ensure_type(test, **self.kwargs)
        assert all(b == np.array([[1.0]], dtype=np.int32))

    def test_raising_on_wrong_ndim_arrays(self):
        test = np.array([1.0], dtype=np.int32)
        self.kwargs['ndim'] = 2
        self.kwargs.pop('add_newaxis_on_deficient_ndim')
        with pytest.raises(ValueError) as err:
            _ = ensure_type(test, **self.kwargs)
        assert 'must be ndim' in str(err.value)

    def test_raising_on_wrong_length(self):
        test = np.array([1.0], dtype=np.int32)
        with pytest.raises(ValueError) as err:
            _ = ensure_type(test, length=3, **self.kwargs)
        assert 'must be length' in str(err.value)

    def test_wrong_shape_length(self):
        test = np.array([1.0], dtype=np.int32)
        with pytest.raises(ValueError) as err:
            _ = ensure_type(test, shape=(1, 3), **self.kwargs)
        assert 'must be shape' in str(err.value)

    def test_wrong_shape(self):
        test = np.array([1.0], dtype=np.int32)
        with pytest.raises(ValueError) as err:
            _ = ensure_type(test, shape=(3,), **self.kwargs)
        assert 'must be shape' in str(err.value)


class TestLengthAndAnglesToBoxVectors(object):
    def setup(self):
        self.kwargs = {'a_length': 1,
                       'b_length': 1,
                       'c_length': 1,
                       'alpha': 90,
                       'beta': 90,
                       'gamma': 90}

    def teardown(self):
        pass

    def test_warn_on_radian(self):
        self.kwargs['alpha'] = 1
        self.kwargs['beta'] = 2
        self.kwargs['gamma'] = 3

        with pytest.warns(Warning) as record:
            _ = lengths_and_angles_to_box_vectors(**self.kwargs)
        assert 'radian' in record[0].message.args[0]

    def test_messed_up_shape(self):
        self.kwargs['a_length'] = np.asarray([[1]])
        with pytest.raises(TypeError) as err:
            _ = lengths_and_angles_to_box_vectors(**self.kwargs)
        assert 'Shape is messed up' in str(err)

    def test_same_as_mdtraj(self):
        a = mdtraj.utils.lengths_and_angles_to_box_vectors(**self.kwargs)
        b = lengths_and_angles_to_box_vectors(**self.kwargs)
        assert (a[0] == b[0].compute()).all()
        assert (a[1] == b[1].compute()).all()
        assert (a[2] == b[2].compute()).all()


class TestBoxVectorsToLengthAndAngles(object):
    def setup(self):
        self.kwargs = {'a': np.array([1, 0, 0]),
                       'b': np.array([0, 1, 0]),
                       'c': np.array([0, 0, 1])}

    def teardown(self):
        pass

    def test_non_uniform_shape(self):
        self.kwargs['a'] = np.array([1, 0, 0, 0])
        with pytest.raises(TypeError) as err:
            _ = box_vectors_to_lengths_and_angles(**self.kwargs)
        assert "Shape is messed up" in str(err)

    def test_wrong_shape(self):
        self.kwargs['a'] = np.array([1, 0])
        self.kwargs['b'] = np.array([1, 0])
        self.kwargs['c'] = np.array([1, 0])

        with pytest.raises(TypeError) as err:
            _ = box_vectors_to_lengths_and_angles(**self.kwargs)
        assert "length 3" in str(err)

    def test_wrong_ndim(self):
        self.kwargs['a'] = np.array([[[1, 0, 0]]])
        self.kwargs['b'] = np.array([[[1, 0, 0]]])
        self.kwargs['c'] = np.array([[[1, 0, 0]]])

        with pytest.raises(ValueError) as err:
            _ = box_vectors_to_lengths_and_angles(**self.kwargs)
        assert "1d or 2d" in str(err)
