import numpy as np
from mpi4py import MPI


from . import constants


def main_check_warm_beta(comm, workers_segments):
    """Check that beta computed in overlapping workers is identical.

    This check is performed only for workers overlapping with the first one.
    """

    pt_global = workers_segments.get_seg_support(0, inner=True)
    sum_beta = np.empty(1, 'd')
    value = []
    for i_worker in range(workers_segments.effective_n_seg):

        pt = workers_segments.get_local_coordinate(i_worker, pt_global)
        if workers_segments.is_contained_coordinate(i_worker, pt):
            comm.Recv([sum_beta, MPI.DOUBLE], source=i_worker)
            value.append(sum_beta[0])
    if len(value) > 1:
        assert np.allclose(value[1:], value[0]), value


def worker_check_warm_beta(rank, workers_segments, beta, D_shape):
    """Helper function for main_check_warm_beta, to be run in the workers."""
    pt_global = workers_segments.get_seg_support(0, inner=True)
    pt = workers_segments.get_local_coordinate(rank, pt_global)
    if workers_segments.is_contained_coordinate(rank, pt):
        _, _, *atom_support = D_shape
        beta_slice = (Ellipsis,) + tuple([
            slice(v - size_ax + 1, v + size_ax - 1)
            for v, size_ax in zip(pt, atom_support)
        ])
        sum_beta = np.array(beta[beta_slice].sum(), dtype='d')

        comm = MPI.Comm.Get_parent()
        comm.Send([sum_beta, MPI.DOUBLE], dest=0,
                  tag=constants.TAG_ROOT + rank)
