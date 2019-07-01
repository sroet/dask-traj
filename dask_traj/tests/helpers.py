from numpy.testing import assert_equal


def function_test(dask_trj, md_trj, funcname, **kwargs):
    if '.' in funcname:
        funcmod, funcn = funcname.rsplit('.', 1)
        dmdfuncmod = 'dask_traj.{}'.format(funcmod)
        mdfuncmod = 'mdtraj.{}'.format(funcmod)
    else:
        funcn = funcname
        dmdfuncmod = 'dask_traj'
        mdfuncmod = 'mdtraj'

    dmdfunc = getattr(__import__(dmdfuncmod), funcn)
    mdfunc = getattr(__import__(mdfuncmod), funcn)
    dmdr = dmdfunc(dask_trj, **kwargs).compute()
    mdr = mdfunc(md_trj, **kwargs)
    assert_equal(dmdr, mdr)
