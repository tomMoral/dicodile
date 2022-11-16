import itertools
import numpy as np
from mpi4py import MPI


from . import constants


def get_global_test_points(workers_segments):
    test_points = []
    for i_seg in range(workers_segments.effective_n_seg):
        seg_bounds = workers_segments.get_seg_bounds(i_seg)
        pt_coord = [(s, s+1, s+2, (s+e) // 2, e-1, e-2, e-3)
                    for s, e in seg_bounds]
        test_points.extend(itertools.product(*pt_coord))
    return test_points


def main_check_beta(comm, workers_segments):
    """Check that beta computed in overlapping workers is identical.

    This check is performed only for workers overlapping with the first one.
    """
    global_test_points = get_global_test_points(workers_segments)
    for i_probe, pt_global in enumerate(global_test_points):
        sum_beta = np.empty(1, 'd')
        value = []
        for i_worker in range(workers_segments.effective_n_seg):

            pt = workers_segments.get_local_coordinate(i_worker, pt_global)
            if workers_segments.is_contained_coordinate(i_worker, pt):
                comm.Recv([sum_beta, MPI.DOUBLE], source=i_worker,
                          tag=constants.TAG_ROOT + i_probe)
                value.append(sum_beta[0])
        if len(value) > 1:
            # print("hello", pt_global)
            assert np.allclose(value[1:], value[0]), value


def worker_check_beta(rank, workers_segments, beta, D_shape):
    """Helper function for main_check_warm_beta, to be run in the workers."""

    assert beta.shape[0] == D_shape[0]

    global_test_points = get_global_test_points(workers_segments)
    for i_probe, pt_global in enumerate(global_test_points):
        pt = workers_segments.get_local_coordinate(rank, pt_global)
        if workers_segments.is_contained_coordinate(rank, pt):
            beta_slice = (Ellipsis,) + pt
            sum_beta = np.array(beta[beta_slice].sum(), dtype='d')

            comm = MPI.Comm.Get_parent()
            comm.Send([sum_beta, MPI.DOUBLE], dest=0,
                      tag=constants.TAG_ROOT + i_probe)
