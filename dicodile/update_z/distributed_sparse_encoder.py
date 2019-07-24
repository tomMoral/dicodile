import weakref
import numpy as np
from mpi4py import MPI

from ..utils import constants
from ..utils.mpi import broadcast_array
from ..utils.csc import compute_objective
from ..workers.reusable_workers import get_reusable_workers
from ..workers.reusable_workers import send_command_to_reusable_workers

from .dicod import recv_z_hat, recv_z_nnz
from .dicod import recv_cost, recv_sufficient_statistics
from .dicod import _send_task, _send_signal
from .dicod import _gather_run_statistics


class DistributedSparseEncoder:
    def __init__(self, n_jobs, w_world='auto', hostfile=None, verbose=0):
        # check the parameters
        if w_world != 'auto':
            assert n_jobs % w_world == 0, (
                "`w_world={}` should divide the number of jobs `n_jobs={}` "
                "used.".format(w_world, n_jobs))

        # Store the parameters
        self.n_jobs = n_jobs
        self.w_world = w_world
        self.hostfile = hostfile
        self.verbose = verbose

    def init_workers(self, X, D_hat, reg, params, z0=None):

        # Create the workers with MPI
        self.comm = get_reusable_workers(self.n_jobs, hostfile=self.hostfile)
        send_command_to_reusable_workers(constants.TAG_WORKER_RUN_DICODILE,
                                         verbose=self.verbose)

        # compute the partition fo the signals
        assert D_hat.ndim - 1 == X.ndim
        n_channels, *sig_support = X.shape
        n_atoms, n_channels, *atom_support = self.D_shape = D_hat.shape

        self.params = params.copy()
        self.t_init, self.workers_segments = _send_task(
            self.comm, X, D_hat, z0, self.w_world, self.params)

    def set_worker_D(self, D):
        msg = "The shape of the dictionary cannot be changed on an encoder."
        assert D.shape == self.D_shape, msg
        send_command_to_reusable_workers(constants.TAG_DICODILE_SET_D,
                                         verbose=self.verbose)
        broadcast_array(self.comm, D)

    def set_worker_params(self, params=None, **kwargs):
        if params is None:
            assert kwargs is not {}
            params = kwargs
        self.params.update(params)

        send_command_to_reusable_workers(constants.TAG_DICODILE_SET_PARAMS,
                                         verbose=self.verbose)
        self.comm.bcast(self.params, root=MPI.ROOT)

    def set_worker_signal(self, X, z0=None):

        if self.is_same_signal(X):
            return

        send_command_to_reusable_workers(constants.TAG_DICODILE_SET_SIGNAL,
                                         verbose=self.verbose)
        self.workers_segments = _send_signal(
            self.comm, self.w_world, self.atom_support, X, z0)
        self._ref_X = weakref.ref(X)

    def compute_z_hat(self):
        send_command_to_reusable_workers(constants.TAG_DICODILE_COMPUTE_Z_HAT,
                                         verbose=self.verbose)
        # Then wait for the end of the computation
        self.comm.Barrier()
        return _gather_run_statistics(self.comm, self.n_jobs,
                                      verbose=self.verbose)

    def get_cost(self):
        send_command_to_reusable_workers(constants.TAG_DICODILE_GET_COST,
                                         verbose=self.verbose)
        return recv_cost(self.comm)

    def get_z_hat(self):
        send_command_to_reusable_workers(constants.TAG_DICODILE_GET_Z_HAT,
                                         verbose=self.verbose)
        return recv_z_hat(self.comm, self.D_shape[0], self.workers_segments)

    def get_z_nnz(self):
        send_command_to_reusable_workers(constants.TAG_DICODILE_GET_Z_NNZ,
                                         verbose=self.verbose)
        return recv_z_nnz(self.comm, self.D_shape[0])

    def get_sufficient_statistics(self):
        send_command_to_reusable_workers(
            constants.TAG_DICODILE_GET_SUFFICIENT_STAT,
            verbose=self.verbose)
        return recv_sufficient_statistics(self.comm, self.D_shape)

    def release_workers(self):
        send_command_to_reusable_workers(constants.TAG_DICODILE_STOP)

    def check_cost(self, X, D_hat, reg):
        cost = self.get_cost()
        z_hat = self.get_z_hat()
        cost_2 = compute_objective(X, z_hat, D_hat, reg)
        assert np.isclose(cost, cost_2), (cost, cost_2)
        print("check cost ok", cost, cost_2)

    def is_same_signal(self, X):
        if not hasattr(self, '_ref_X') or self._ref_X() is not X:
            return False
        return True
