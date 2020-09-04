[![Build Status](https://travis-ci.org/sroet/dask-traj.svg?branch=master)](https://travis-ci.org/sroet/dask-traj)
[![Documentation Status](https://readthedocs.org/projects/dask-traj/badge/?version=latest)](https://dask-traj.readthedocs.io/en/latest/?badge=latest)
# dask-traj
This is a parallel implementation of parts of [MDTraj](https://mdtraj.org/), using [dask](https://dask.org/).  

It tries to alleviate some restrictions of MDTraj, by allowing for out-of-memory
computation. Combined with
[dask-distributed](https://distributed.dask.org/en/latest/) this allows for
out-of-machine parallelization, essential for HPCs and results in a (surprising)
speed-up [even on a single
machine](https://github.com/sroet/dask-traj/blob/master/examples/dask-traj_distributed%20example.ipynb).

Bare code documentation can be read [here](dask-traj.readthedocs.io).

If a function of MDTraj that you want to use is not yet
supported here, please raise an issue. That way we know where to focus our efforts on.

# Installation

This code can be installed with `pip`

```bash
pip install dask-traj
``` 

# Examples

In order to run the example in the 'examples' directory, please also install the
following dependencies, all available through `pip` and `conda` (via
`conda-forge`):
 * jupyter
 * distributed
 * python-graphviz 

Please have a look at [this
example](https://github.com/sroet/dask-traj/blob/master/examples/dask-traj_distributed%20example.ipynb)
for an indication of the possible speedups when used with [dask
distributed](https://distributed.dask.org/en/latest/).
